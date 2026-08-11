"""Orchestrates one Analyze-time external-knowledge lookup: builds
search terms from what the Recommendation Engine already computed
(never new NLP), queries TFS and Wiki concurrently with independent
timeouts, ranks and caps results, and never lets either source's
failure propagate into a broken recommendation.

Concurrency: both connectors are submitted to a small thread pool and
awaited independently -- TFS being slow/down must not delay Wiki
results or vice versa, and neither may block the rest of
``RecommendationEngine.generate()`` beyond the configured timeout.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.domain.enums import EntityType
from app.domain.external_knowledge import ExternalKnowledgeResult, ExternalMatch, ExternalSource, TfsCase, WikiPage
from app.engines.external_knowledge.cache import TtlCache
from app.engines.external_knowledge.ranking import confidence_for_score, score_candidate
from app.engines.external_knowledge.tfs_rest_client import TfsConnectorError
from app.engines.external_knowledge.wiki_rest_client import WikiConnectorError

if TYPE_CHECKING:
    from app.config import Settings
    from app.domain.entities import ExtractedEntity
    from app.domain.investigation import InvestigationSession
    from app.domain.recommendation import MatchedComponent
    from app.engines.external_knowledge.tfs_connector import TfsConnector
    from app.engines.external_knowledge.wiki_connector import WikiConnector

logger = logging.getLogger(__name__)

_MIN_WORD_LEN = 4
_MAX_TITLE_WORDS = 5
_MAX_TOTAL_TERMS = 8
_TFS_WORK_ITEM_TYPES = ["Bug", "Issue"]
_ENTITY_TYPES_WORTH_SEARCHING = {EntityType.EXCEPTION_TYPE}
"""Deliberately narrow: most extracted entity types (thread/session/
correlation ids, serial numbers, ...) are per-instance values specific
to *this* occurrence and will never appear verbatim in a different
historical incident, even one caused by the same underlying bug.
Exception type names are the one category that genuinely recurs across
unrelated real-world occurrences of the same class of problem."""


@dataclass(frozen=True)
class SearchTerms:
    """Split, not a flat bag of keywords -- real live-observed problem
    this fixes: OR-ing every title word together (the original design)
    let generic words flood a recency-ordered candidate window with
    noise, crowding out genuinely relevant matches (confirmed against
    the real TFS server: adding "Meter"/"state"/"after" alongside
    "CommandProcessorHost"/"Discovered" buried the two real matches
    found with just the specific terms under twenty unrelated
    "[Meter.comm][DLMS SW Tool]" UI tickets that merely happened to
    contain the word "Meter").

    ``anchor`` (component/technology -- the investigation's strongest,
    most reusable signals) is what actually scopes the live query when
    present: a connector requires at least one anchor term to match,
    never ORs it in as just one more equally-weighted keyword.
    ``supporting`` (title keywords, exception-type entities) is used
    for local ranking always, and as the query filter only when there
    is no anchor to scope by at all."""

    anchor: list[str] = field(default_factory=list)
    supporting: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        seen: set[str] = set()
        combined: list[str] = []
        for term in [*self.anchor, *self.supporting]:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                combined.append(term)
        return combined


def build_search_terms(
    investigation: "InvestigationSession",
    entities: list["ExtractedEntity"],
    matched_component: "MatchedComponent | None",
    technology: str | None,
) -> SearchTerms:
    """Deterministic, reuses signals the Recommendation Engine already
    computed -- no new NLP."""
    anchor: list[str] = []
    supporting: list[str] = []
    seen: set[str] = set()

    def _add(bucket: list[str], term: str | None, *, limit: int) -> None:
        if not term:
            return
        cleaned = term.strip()
        key = cleaned.lower()
        if cleaned and key not in seen and len(bucket) < limit:
            seen.add(key)
            bucket.append(cleaned)

    _add(anchor, technology, limit=_MAX_TOTAL_TERMS)
    if matched_component is not None:
        _add(anchor, matched_component.component_name, limit=_MAX_TOTAL_TERMS)

    title_words = [w for w in re.findall(r"[A-Za-z]+", investigation.title) if len(w) >= _MIN_WORD_LEN]
    for word in title_words[:_MAX_TITLE_WORDS]:
        _add(supporting, word, limit=_MAX_TOTAL_TERMS)

    for entity in entities:
        if entity.entity_type in _ENTITY_TYPES_WORTH_SEARCHING:
            _add(supporting, entity.value, limit=_MAX_TOTAL_TERMS)

    return SearchTerms(anchor=anchor, supporting=supporting)


def _cache_key(source: ExternalSource, terms: SearchTerms) -> str:
    return f"{source.value}:{','.join(sorted(t.lower() for t in terms.all))}"


def _entity_values(entities: list["ExtractedEntity"]) -> list[str]:
    return [e.value for e in entities if e.entity_type in _ENTITY_TYPES_WORTH_SEARCHING]


def _tfs_recommended_action(case: TfsCase) -> str | None:
    """Quotes the extracted resolution directly -- never a paraphrase.
    None when TFS genuinely has no resolution content to quote."""
    if not case.resolution_text:
        return None
    return f"Based on TFS-{case.tfs_id} ({case.state}): {case.resolution_text}"


def _score_tfs(case: TfsCase, *, investigation_title: str, matched_component_name, technology, entity_values):
    body = f"{case.description_text}\n{case.root_cause or ''}\n{case.resolution_text or ''}"
    score, reasons = score_candidate(
        investigation_title=investigation_title,
        candidate_title=case.title,
        candidate_body=body,
        state=case.state,
        matched_component_name=matched_component_name,
        technology=technology,
        entity_values=entity_values,
    )
    return ExternalMatch(
        source=ExternalSource.TFS,
        tfs_case=case,
        score=score,
        confidence=confidence_for_score(score),
        match_reasons=reasons,
        recommended_action=_tfs_recommended_action(case),
    )


def _score_wiki(page: WikiPage, *, investigation_title: str, matched_component_name, technology, entity_values):
    score, reasons = score_candidate(
        investigation_title=investigation_title,
        candidate_title=page.title,
        candidate_body=page.excerpt,
        state=None,
        matched_component_name=matched_component_name,
        technology=technology,
        entity_values=entity_values,
    )
    return ExternalMatch(source=ExternalSource.WIKI, wiki_page=page, score=score, confidence=confidence_for_score(score), match_reasons=reasons)


class ExternalKnowledgeService:
    def __init__(
        self,
        *,
        tfs_connector: "TfsConnector",
        wiki_connector: "WikiConnector",
        settings: "Settings",
    ) -> None:
        self._tfs = tfs_connector
        self._wiki = wiki_connector
        self._settings = settings
        self._cache: TtlCache[ExternalKnowledgeResult] = TtlCache(
            max_entries=50, ttl_seconds=settings.external_knowledge_cache_ttl_seconds
        )

    def gather(
        self,
        investigation: "InvestigationSession",
        entities: list["ExtractedEntity"],
        matched_component: "MatchedComponent | None",
        technology: str | None,
    ) -> tuple[ExternalKnowledgeResult, ExternalKnowledgeResult]:
        if not self._settings.external_knowledge_enabled:
            disabled = "External knowledge lookups are disabled (RESOLVEIQ_EXTERNAL_KNOWLEDGE_ENABLED=false)."
            return (
                ExternalKnowledgeResult(source=ExternalSource.TFS, available=False, error=disabled),
                ExternalKnowledgeResult(source=ExternalSource.WIKI, available=False, error=disabled),
            )

        terms = build_search_terms(investigation, entities, matched_component, technology)
        component_name = matched_component.component_name if matched_component else None
        entity_values = _entity_values(entities)
        title = investigation.title
        max_results = self._settings.external_knowledge_max_results
        timeout = self._settings.external_knowledge_timeout_seconds

        with ThreadPoolExecutor(max_workers=2) as executor:
            tfs_future = executor.submit(
                self._run_tfs, terms, title, component_name, technology, entity_values, max_results
            )
            wiki_future = executor.submit(
                self._run_wiki, terms, title, component_name, technology, entity_values, max_results
            )
            # Each future's own connector call already carries the
            # configured request timeout internally (see the REST
            # clients) -- this wrapper timeout is a safety margin, not
            # the primary bound, in case something hangs outside the
            # HTTP call itself (e.g. auth negotiation).
            tfs_result = self._safe_result(tfs_future, ExternalSource.TFS, timeout)
            wiki_result = self._safe_result(wiki_future, ExternalSource.WIKI, timeout)

        return tfs_result, wiki_result

    @staticmethod
    def _safe_result(future, source: ExternalSource, timeout: float) -> ExternalKnowledgeResult:
        try:
            return future.result(timeout=timeout + 2)
        except Exception as exc:  # noqa: BLE001 -- external knowledge must never break Analyze
            logger.warning("External knowledge (%s) failed: %s", source.value, exc.__class__.__name__)
            return ExternalKnowledgeResult(source=source, available=False, error=f"{source.value.upper()} lookup failed unexpectedly.")

    def _run_tfs(self, terms: SearchTerms, title, component_name, technology, entity_values, max_results) -> ExternalKnowledgeResult:
        cache_key = _cache_key(ExternalSource.TFS, terms)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"from_cache": True})

        query_summary = ", ".join(terms.all)
        if not self._tfs.is_configured():
            result = ExternalKnowledgeResult(
                source=ExternalSource.TFS, available=False, error="TFS is not configured.", query_summary=query_summary
            )
            return result

        try:
            cases = self._tfs.search_work_items(
                anchor_terms=terms.anchor,
                supporting_terms=terms.supporting,
                work_item_types=_TFS_WORK_ITEM_TYPES,
                max_results=max_results,
            )
        except TfsConnectorError as exc:
            logger.warning("TFS search failed: %s", exc.__class__.__name__)
            return ExternalKnowledgeResult(
                source=ExternalSource.TFS,
                available=False,
                error="TFS could not be reached. Local ResolveIQ knowledge was used instead.",
                query_summary=query_summary,
            )

        matches = [
            _score_tfs(c, investigation_title=title, matched_component_name=component_name, technology=technology, entity_values=entity_values)
            for c in cases
        ]
        matches = sorted((m for m in matches if m.score >= 0.25), key=lambda m: m.score, reverse=True)[:max_results]
        result = ExternalKnowledgeResult(source=ExternalSource.TFS, matches=matches, available=True, query_summary=query_summary)
        self._cache.set(cache_key, result)
        return result

    def _run_wiki(self, terms: SearchTerms, title, component_name, technology, entity_values, max_results) -> ExternalKnowledgeResult:
        cache_key = _cache_key(ExternalSource.WIKI, terms)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached.model_copy(update={"from_cache": True})

        query_summary = ", ".join(terms.all)
        if not self._wiki.is_configured():
            return ExternalKnowledgeResult(
                source=ExternalSource.WIKI, available=False, error="Wiki is not configured.", query_summary=query_summary
            )

        try:
            pages = self._wiki.search(anchor_terms=terms.anchor, supporting_terms=terms.supporting, max_results=max_results)
        except WikiConnectorError as exc:
            logger.warning("Wiki search failed: %s", exc.__class__.__name__)
            return ExternalKnowledgeResult(
                source=ExternalSource.WIKI,
                available=False,
                error="Wiki could not be reached. Local ResolveIQ knowledge was used instead.",
                query_summary=query_summary,
            )

        matches = [
            _score_wiki(p, investigation_title=title, matched_component_name=component_name, technology=technology, entity_values=entity_values)
            for p in pages
        ]
        matches = sorted((m for m in matches if m.score >= 0.25), key=lambda m: m.score, reverse=True)[:max_results]
        result = ExternalKnowledgeResult(source=ExternalSource.WIKI, matches=matches, available=True, query_summary=query_summary)
        self._cache.set(cache_key, result)
        return result
