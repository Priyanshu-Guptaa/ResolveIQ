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
_MAX_DESCRIPTION_WORDS = 5
_MAX_TOTAL_TERMS = 8
_TFS_WORK_ITEM_TYPES = ["Bug", "Issue"]
_ENTITY_TYPES_WORTH_SEARCHING = {
    EntityType.EXCEPTION_TYPE,
    EntityType.METER_NUMBER,
    EntityType.SERIAL_NUMBER,
    EntityType.CORRELATION_ID,
    EntityType.HOST_NAME,
}
"""Deliberately still a narrow allowlist, not every entity type: most
(thread/session/process ids, request ids, ...) are per-instance values
specific to *this* occurrence and will never appear verbatim in a
different historical incident, even one caused by the same underlying
bug. The five kept here each have a real reason to recur: exception
type names recur across unrelated occurrences of the same class of
problem; a meter/serial number recurs when the *same physical device*
shows up in more than one ticket (a genuinely strong signal, not a
coincidence); a host name can be a stable, reused server identity
rather than a per-request value. Correlation ID is the shakiest of the
five (usually unique per transaction, so it will rarely actually
recur) -- kept anyway because on the rare occasion it *does* match,
that's about as strong and precise a signal as exists; it costs
nothing when it doesn't match, since supporting terms only affect the
live query when there's no anchor at all, and always affect ranking
without ever being trusted alone."""

_TICKET_NUMBER_RE = re.compile(r"\b(?:CSTASK|CS|TASK|INC)\d{5,}\b", re.IGNORECASE)
"""Real prefixes observed in this project's actual ServiceNow/TFS data
this session (e.g. CSTASK0066554, CS0122697, TASK0469743, INC0045821 --
see [[external-knowledge-feature]]/Task Import) -- not a guessed
pattern. TFS Bugs carry the matching ticket number verbatim in
``LandisGyr.CRMID`` (e.g. "CS0122697/CSTASK0087353"), so a ticket
number mentioned in the investigation's own text is one of the
strongest possible anchors: it can point straight at the TFS work item
that actually fixed this exact customer's exact case."""


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

    ``anchor`` (ticket number, product/version, technology, matched
    component -- the investigation's strongest, most reusable signals)
    is what actually scopes the live query when present: a connector
    requires at least one anchor term to match, never ORs it in as
    just one more equally-weighted keyword. ``supporting`` (title
    keywords, description keywords, a narrow allowlist of extracted
    entities) is used for local ranking always, and as the query
    filter only when there is no anchor to scope by at all.

    Customer name is deliberately in neither list: it's a real,
    valuable ranking signal (a same-customer prior case is worth
    surfacing higher) but was found live to be a bad *query* term --
    anchors are OR'd, so including it let a ticket qualify as a
    candidate on customer-name overlap alone, regardless of technology
    relevance, diluting results the same way flat-OR-everything did
    before the anchor/supporting split existed. It's applied directly
    in ``ranking.py``'s ``score_candidate()`` instead, from
    ``investigation.customer``, never from ``SearchTerms``."""

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
    computed -- no new NLP. Anchors, strongest first: a ticket number
    (points at one specific TFS work item, if this exact case was
    already logged there), product/version (structured Summary Card
    fields -- real, curated, not free-text guesses), technology,
    matched component -- deliberately NOT customer, see SearchTerms'
    own docstring. Supporting: title keywords, description keywords,
    a narrow allowlist of extracted entities."""
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

    description_text = _first_evidence_text(investigation)

    for ticket in _TICKET_NUMBER_RE.findall(f"{investigation.title}\n{description_text}"):
        _add(anchor, ticket.upper(), limit=_MAX_TOTAL_TERMS)

    _add(anchor, investigation.product, limit=_MAX_TOTAL_TERMS)
    _add(anchor, investigation.version, limit=_MAX_TOTAL_TERMS)
    _add(anchor, technology, limit=_MAX_TOTAL_TERMS)
    if matched_component is not None:
        _add(anchor, matched_component.component_name, limit=_MAX_TOTAL_TERMS)

    title_words = [w for w in re.findall(r"[A-Za-z]+", investigation.title) if len(w) >= _MIN_WORD_LEN]
    for word in title_words[:_MAX_TITLE_WORDS]:
        _add(supporting, word, limit=_MAX_TOTAL_TERMS)

    # Description keywords, same treatment as title keywords -- only
    # ever broadens the live query when there's no anchor at all
    # (SearchTerms' own contract), otherwise just improves local
    # ranking. Capped separately from title words so a long
    # description can't crowd out every title word before dedup even
    # gets a chance to run.
    description_words = [w for w in re.findall(r"[A-Za-z]+", description_text) if len(w) >= _MIN_WORD_LEN]
    for word in description_words[:_MAX_DESCRIPTION_WORDS]:
        _add(supporting, word, limit=_MAX_TOTAL_TERMS)

    for entity in entities:
        if entity.entity_type in _ENTITY_TYPES_WORTH_SEARCHING:
            _add(supporting, entity.value, limit=_MAX_TOTAL_TERMS)

    return SearchTerms(anchor=anchor, supporting=supporting)


def _first_evidence_text(investigation: "InvestigationSession", *, max_chars: int = 4000) -> str:
    """The task description (first piece of evidence) is where a
    ticket number is actually likely to appear (e.g. a pasted
    ServiceNow template) -- capped, not the full context_text, since
    this only feeds a regex scan, not a search query itself."""
    for item in investigation.evidence:
        if item.raw_content:
            return item.raw_content[:max_chars]
    return ""


def _cache_key(source: ExternalSource, terms: SearchTerms) -> str:
    return f"{source.value}:{','.join(sorted(t.lower() for t in terms.all))}"


def _entity_values(entities: list["ExtractedEntity"]) -> list[str]:
    return [e.value for e in entities if e.entity_type in _ENTITY_TYPES_WORTH_SEARCHING]


def _tfs_recommended_action(case: TfsCase, *, confidence: str) -> str | None:
    """Quotes the extracted resolution directly -- never a paraphrase --
    but only when ``confidence`` clears "Low". A Low-confidence match
    (technology-only overlap, weak title overlap) hasn't earned the
    "ResolveIQ recommendation:" framing -- found live: a real Low-
    confidence TFS match about a completely different defect (Gap Recon
    command delivery vs. a Get LP command DCW error) was still being
    boxed as a confident recommendation just because it had *any*
    resolution text, the same class of bug already fixed twice in the
    local-KB synthesis layer. The raw "TFS reported resolution" text
    (rendered separately, unconditionally, straight from the source)
    is untouched -- this only gates ResolveIQ's own endorsement of it."""
    if not case.resolution_text or confidence == "Low":
        return None
    return f"Based on TFS-{case.tfs_id} ({case.state}): {case.resolution_text}"


def _score_tfs(case: TfsCase, *, investigation_title: str, matched_component_name, technology, entity_values, customer):
    body = f"{case.description_text}\n{case.root_cause or ''}\n{case.resolution_text or ''}"
    score, reasons = score_candidate(
        investigation_title=investigation_title,
        candidate_title=case.title,
        candidate_body=body,
        state=case.state,
        matched_component_name=matched_component_name,
        technology=technology,
        entity_values=entity_values,
        customer=customer,
    )
    confidence = confidence_for_score(score)
    return ExternalMatch(
        source=ExternalSource.TFS,
        tfs_case=case,
        score=score,
        confidence=confidence,
        match_reasons=reasons,
        recommended_action=_tfs_recommended_action(case, confidence=confidence),
    )


def _score_wiki(page: WikiPage, *, investigation_title: str, matched_component_name, technology, entity_values, customer):
    score, reasons = score_candidate(
        investigation_title=investigation_title,
        candidate_title=page.title,
        candidate_body=page.excerpt,
        state=None,
        matched_component_name=matched_component_name,
        technology=technology,
        entity_values=entity_values,
        customer=customer,
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
        customer = investigation.customer
        max_results = self._settings.external_knowledge_max_results
        timeout = self._settings.external_knowledge_timeout_seconds

        with ThreadPoolExecutor(max_workers=2) as executor:
            tfs_future = executor.submit(
                self._run_tfs, terms, title, component_name, technology, entity_values, customer, max_results
            )
            wiki_future = executor.submit(
                self._run_wiki, terms, title, component_name, technology, entity_values, customer, max_results
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

    def _run_tfs(self, terms: SearchTerms, title, component_name, technology, entity_values, customer, max_results) -> ExternalKnowledgeResult:
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
            _score_tfs(
                c,
                investigation_title=title,
                matched_component_name=component_name,
                technology=technology,
                entity_values=entity_values,
                customer=customer,
            )
            for c in cases
        ]
        matches = sorted((m for m in matches if m.score >= 0.25), key=lambda m: m.score, reverse=True)[:max_results]
        result = ExternalKnowledgeResult(source=ExternalSource.TFS, matches=matches, available=True, query_summary=query_summary)
        self._cache.set(cache_key, result)
        return result

    def _run_wiki(self, terms: SearchTerms, title, component_name, technology, entity_values, customer, max_results) -> ExternalKnowledgeResult:
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
            _score_wiki(
                p,
                investigation_title=title,
                matched_component_name=component_name,
                technology=technology,
                entity_values=entity_values,
                customer=customer,
            )
            for p in pages
        ]
        matches = sorted((m for m in matches if m.score >= 0.25), key=lambda m: m.score, reverse=True)[:max_results]
        result = ExternalKnowledgeResult(source=ExternalSource.WIKI, matches=matches, available=True, query_summary=query_summary)
        self._cache.set(cache_key, result)
        return result
