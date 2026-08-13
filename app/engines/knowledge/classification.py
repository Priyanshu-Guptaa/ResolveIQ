"""Metadata Classification -- confidence-tiered, review-gated extraction
of Product/Technology/Component/Customer/Region tags for already-
imported Documentation (Context Dimensions phase, 2026-08-12 design
assessment Part C.1/F, approved product decisions #2/#3/#4).

Extractive only, against already-governed values: every suggestion
matches a real row already sitting in the ``products``/``technologies``/
``component_profiles``/``customers``/``regions`` tables (by name or, for
Customer/Region, a real alias). This engine never proposes a *new*
dimension value from free text -- which is the structural guarantee
that a customer name can never become a suggested Technology/Component/
Product (see app/domain/classification.py's module docstring): those
candidate lists are curated from real product-architecture/TFS/Wiki
evidence, never from arbitrary document titles, so "TEPCO" (which only
ever exists in the ``customers`` table) has no path to ever being
suggested as anything else.

Confidence tiers (see ``app.domain.classification.ConfidenceTier`` for
the full contract):

- HIGH: the governed name/alias appears verbatim (word-boundary) in the
  document's own TITLE. Applied immediately -- but a scalar field
  (Technology/Product on ``DocumentationRecord``) is only ever set when
  currently ``None``; a document that already carries real metadata is
  never touched (approved product decision #4, "without destroying or
  overwriting existing knowledge"). Customer/Region apply as a real
  ``KnowledgeRelationship`` (APPLIES_TO) instead of a scalar field --
  additive by nature. Component applies as an addition to
  ``related_components`` -- also additive (a list, not overwritten).
- MEDIUM: no title match, but the name/alias is repeated (>=2
  occurrences) in the body. Recorded as a PENDING suggestion for human
  review -- never applied automatically.
- LOW: a single, unrepeated body mention. Recorded (``RECORDED``
  status) for traceability only -- never applied, never queued for
  review. Approved product decision: "low-confidence metadata should
  remain unclassified rather than being guessed."

Real bug fixed here (2026-08-13, found by the Phase 1 acceptance test):
matching used to pick the *first* candidate (in ``list_technologies()``'s
alphabetical order) whose name matched, rather than the *most specific*
one -- so a document titled "RF Mesh IP doc" was tagged "RF Mesh"
(alphabetically first, and a substring of "RF Mesh IP") instead of "RF
Mesh IP". ``_best_match`` now uses ``app.engines.shared.hierarchy.
most_specific`` to prefer a matched candidate over any matched
*ancestor* of it, via the real, already-governed
``Technology.parent_technology_id`` field -- generic (no technology
names hardcoded), works for a hierarchy of any depth, and is a no-op
for every other dimension (Customer/Region/Product/Component), since
none of those carry a parent id at all.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.domain.classification import (
    ClassificationDimension,
    ClassificationRunSummary,
    ConfidenceTier,
    MetadataClassificationSuggestion,
    SuggestionStatus,
)
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.engines.shared.hierarchy import most_specific
from app.engines.shared.text_matching import FULL_MATCH_SCORE, keyword_match_score

if TYPE_CHECKING:
    from app.domain.evidence import DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
    from app.engines.knowledge_object_framework.service import KnowledgeObjectService
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.infrastructure.db.classification_repository import ClassificationRepository
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.knowledge_repository import KnowledgeRepository
    from app.infrastructure.db.lookup_repository import LookupRepository

logger = logging.getLogger(__name__)

_MEDIUM_MIN_BODY_MENTIONS = 2
_SNIPPET_WINDOW = 60
_MIN_CANDIDATE_NAME_LEN = 3
"""Below this, a governed name is too likely to false-positive as an
incidental substring/short-word match to trust for classification at
all -- same floor discipline as
``RecommendationEngine._MIN_COMPONENT_NAME_LEN_FOR_COLLECTED_MATCH``,
lowered slightly (3 not 4) so real short real names in this corpus
("NMS", "EIC") aren't excluded."""
_CONTENT_SCAN_CHARS = 20_000
"""Caps how much of a document's body is scanned for repeated mentions
-- perf safety net, same discipline as
``InvestigationSession.context_text``'s cap; a classification signal
this document-level, if it exists at all, shows up well within the
first 20K characters."""


@dataclass(frozen=True)
class _Candidate:
    id: str
    canonical_name: str
    match_names: tuple[str, ...]
    """canonical_name plus any aliases -- what's actually checked
    against title/body text."""
    parent_id: str | None = None
    """The candidate's parent in a real governed hierarchy (today:
    ``Technology.parent_technology_id``), or None -- either because
    this candidate has no parent, or because this dimension has no
    hierarchy concept at all (Customer/Region/Product/Component).
    Consumed by ``_best_match`` via ``most_specific`` to prefer a more
    specific matched candidate over a matched ancestor of it."""


def _title_match(candidate: "_Candidate", title: str) -> str | None:
    """Returns the matched name (longest first, so a more specific
    alias/name wins over a shorter coincidental one) or None."""
    if not title:
        return None
    for name in sorted(candidate.match_names, key=len, reverse=True):
        if len(name.strip()) < _MIN_CANDIDATE_NAME_LEN:
            continue
        if keyword_match_score(name, title) >= FULL_MATCH_SCORE:
            return name
    return None


def _body_mention_count(candidate: "_Candidate", body: str) -> tuple[int, str | None, str]:
    """Returns (mention_count, matched_name, first_snippet) using the
    longest matching name found."""
    best_name: str | None = None
    best_count = 0
    best_snippet = ""
    for name in sorted(candidate.match_names, key=len, reverse=True):
        cleaned = name.strip()
        if len(cleaned) < _MIN_CANDIDATE_NAME_LEN:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)", re.IGNORECASE)
        matches = list(pattern.finditer(body))
        if len(matches) > best_count:
            best_count = len(matches)
            best_name = name
            first = matches[0]
            lo = max(0, first.start() - _SNIPPET_WINDOW)
            hi = min(len(body), first.end() + _SNIPPET_WINDOW)
            prefix = "..." if lo > 0 else ""
            suffix = "..." if hi < len(body) else ""
            best_snippet = f"{prefix}{body[lo:hi].strip()}{suffix}"
    return best_count, best_name, best_snippet


def _most_specific_candidates(matched: list["_Candidate"], universe: list["_Candidate"]) -> list["_Candidate"]:
    """Filters ``matched`` down to the most specific entries via
    ``app.engines.shared.hierarchy.most_specific``, using every
    candidate in ``universe`` (not just the matched ones) to resolve
    parent ids -- an ancestor several levels up that didn't itself
    match still needs to be walked through for a hierarchy deeper than
    one level. Preserves ``matched``'s original relative order (already
    alphabetical, from ``list_technologies()``/etc.) so the final
    selection stays fully deterministic without a second sort."""
    if len(matched) <= 1:
        return matched
    parent_of = {c.id: c.parent_id for c in universe}
    survivor_ids = most_specific({c.id for c in matched}, parent_of)
    return [c for c in matched if c.id in survivor_ids]


def _best_match(
    candidates: list["_Candidate"], title: str, body: str
) -> tuple["_Candidate", ConfidenceTier, str, str] | None:
    """One dimension's best candidate for one document, or None if
    nothing matched at all. Title match always wins (HIGH) over any
    body-only signal, regardless of body mention count.

    When more than one candidate matches the same text (e.g. both "RF
    Mesh" and "RF Mesh IP" full-match a title that says "RF Mesh IP"),
    the most specific one wins -- see ``_most_specific_candidates``.
    Ties among equally-specific candidates (unrelated technologies, or
    siblings in the same hierarchy) keep ``candidates``'s own order,
    which is always the real, already-alphabetical order the lookup
    repositories return -- the same deterministic tie-break this
    function always used, now applied only among genuine ties rather
    than among every match."""
    title_matches = [c for c in candidates if _title_match(c, title)]
    if title_matches:
        winner = _most_specific_candidates(title_matches, candidates)[0]
        return winner, ConfidenceTier.HIGH, title, "title_exact_match"

    body_hits: list[tuple["_Candidate", int, str]] = []
    for candidate in candidates:
        count, matched_name, snippet = _body_mention_count(candidate, body)
        if count > 0 and matched_name is not None:
            body_hits.append((candidate, count, snippet))
    if not body_hits:
        return None

    specific = _most_specific_candidates([c for c, _count, _snippet in body_hits], candidates)
    specific_ids = {c.id for c in specific}
    filtered = [row for row in body_hits if row[0].id in specific_ids]
    # Among the most-specific survivors, prefer the strongest raw
    # signal (highest mention count); candidates.index gives a stable,
    # fully deterministic final tie-break using the same real,
    # alphabetical order as everywhere else in this function.
    filtered.sort(key=lambda row: (-row[1], candidates.index(row[0])))
    candidate, count, snippet = filtered[0]
    tier = ConfidenceTier.MEDIUM if count >= _MEDIUM_MIN_BODY_MENTIONS else ConfidenceTier.LOW
    rule = "body_repeated_mention" if tier == ConfidenceTier.MEDIUM else "body_single_mention"
    return candidate, tier, snippet, rule


class DocumentClassificationEngine:
    """See module docstring for the full confidence-tiering contract."""

    def __init__(
        self,
        knowledge_repo: "KnowledgeRepository",
        lookup_repo: "LookupRepository",
        component_repo: "ComponentProfileRepository",
        classification_repo: "ClassificationRepository",
        knowledge_object_service: "KnowledgeObjectService",
        relationship_engine: "KnowledgeRelationshipEngine",
    ) -> None:
        self._knowledge = knowledge_repo
        self._lookup = lookup_repo
        self._components = component_repo
        self._suggestions = classification_repo
        self._objects = knowledge_object_service
        self._relationships = relationship_engine

    def run(self, *, actor: str | None = None, limit: int | None = None) -> ClassificationRunSummary:
        """Scans every Documentation, Historical Investigation, and
        Known Bug record (or the first ``limit`` across all three
        combined, for a bounded dry run) and produces/applies
        classification suggestions. Idempotent: re-running skips any
        (object, dimension, value) pair that already has a non-rejected
        suggestion on file (``ClassificationRepository.exists_for_object``),
        so it's safe to run repeatedly (e.g. after a new document is
        published) without creating duplicate review-queue entries.

        Historical Investigations/Known Bugs (2026-08-13, Phase 0 --
        Chat/Structured Resolution Knowledge architecture, closing the
        gap flagged in
        RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md's Section 9
        correction): only Customer/Region (``_classify_relationship``)
        and Component (``_classify_related_components``) run for these
        two types, never ``_classify_scalar_field`` -- unlike
        ``DocumentationRecord``, neither ``HistoricalInvestigationRecord``
        nor ``KnownBugRecord`` has a scalar ``technology``/``product``
        field to set (``getattr`` would simply fail), so Technology/
        Product classification stays Document-only until/unless those
        record types grow real scalar fields of their own."""
        summary = ClassificationRunSummary()

        technologies = [_Candidate(t.id, t.name, (t.name,), parent_id=t.parent_technology_id) for t in self._lookup.list_technologies()]
        products = [_Candidate(p.id, p.name, (p.name,)) for p in self._lookup.list_products()]
        customers = [_Candidate(c.id, c.name, (c.name, *c.aliases)) for c in self._lookup.list_customers()]
        regions = [_Candidate(r.id, r.name, (r.name, *r.aliases)) for r in self._lookup.list_regions()]
        components = [_Candidate(c.id, c.name, (c.name,)) for c in self._components.list_all()]

        documents = [self._knowledge.get_documentation(item.id) for item in self._knowledge.list_documentation_summaries(page_size=100_000).items]
        for document in documents:
            if document is None:
                continue
            if limit is not None and summary.documents_scanned >= limit:
                break
            summary.documents_scanned += 1
            title = document.title or ""
            body = (document.content or "")[:_CONTENT_SCAN_CHARS]

            self._classify_scalar_field(document, technologies, title, body, ClassificationDimension.TECHNOLOGY, "technology", actor, summary)
            self._classify_scalar_field(document, products, title, body, ClassificationDimension.PRODUCT, "product", actor, summary)
            self._classify_relationship(
                document, customers, title, body, ClassificationDimension.CUSTOMER, KnowledgeObjectType.CUSTOMER,
                actor, summary, KnowledgeObjectType.DOCUMENT,
            )
            self._classify_relationship(
                document, regions, title, body, ClassificationDimension.REGION, KnowledgeObjectType.REGION,
                actor, summary, KnowledgeObjectType.DOCUMENT,
            )
            self._classify_related_components(document, components, title, body, actor, summary, KnowledgeObjectType.DOCUMENT)

        for investigation in self._knowledge.list_historical_investigations():
            if limit is not None and summary.documents_scanned >= limit:
                break
            summary.documents_scanned += 1
            title = investigation.title or ""
            body = (investigation.description or "")[:_CONTENT_SCAN_CHARS]
            object_type = KnowledgeObjectType.HISTORICAL_INVESTIGATION

            self._classify_relationship(investigation, customers, title, body, ClassificationDimension.CUSTOMER, KnowledgeObjectType.CUSTOMER, actor, summary, object_type)
            self._classify_relationship(investigation, regions, title, body, ClassificationDimension.REGION, KnowledgeObjectType.REGION, actor, summary, object_type)
            self._classify_related_components(investigation, components, title, body, actor, summary, object_type)

        for bug in self._knowledge.list_known_bugs():
            if limit is not None and summary.documents_scanned >= limit:
                break
            summary.documents_scanned += 1
            title = bug.title or ""
            body = (bug.description or "")[:_CONTENT_SCAN_CHARS]
            object_type = KnowledgeObjectType.KNOWN_BUG

            self._classify_relationship(bug, customers, title, body, ClassificationDimension.CUSTOMER, KnowledgeObjectType.CUSTOMER, actor, summary, object_type)
            self._classify_relationship(bug, regions, title, body, ClassificationDimension.REGION, KnowledgeObjectType.REGION, actor, summary, object_type)
            self._classify_related_components(bug, components, title, body, actor, summary, object_type)

        logger.info(
            "Classification run complete: %d objects scanned, %d suggestions (%d auto-accepted, %d pending, %d low-confidence), "
            "%d fields set, %d relationships created",
            summary.documents_scanned, summary.suggestions_created, summary.auto_accepted, summary.pending_review,
            summary.recorded_low_confidence, summary.fields_set, summary.relationships_created,
        )
        return summary

    # --- Scalar fields (Technology/Product on DocumentationRecord) ---------

    def _classify_scalar_field(
        self,
        document: "DocumentationRecord",
        candidates: list["_Candidate"],
        title: str,
        body: str,
        dimension: ClassificationDimension,
        field_name: str,
        actor: str | None,
        summary: ClassificationRunSummary,
    ) -> None:
        if getattr(document, field_name):
            # Already has real metadata -- never overwritten (approved
            # product decision #4). No suggestion needed; existing data
            # wins outright.
            return
        match = _best_match(candidates, title, body)
        if match is None:
            return
        candidate, tier, evidence, rule = match
        if self._suggestions.exists_for_object(
            KnowledgeObjectType.DOCUMENT.value, document.id, dimension, candidate.id
        ):
            return

        suggestion = MetadataClassificationSuggestion(
            object_type=KnowledgeObjectType.DOCUMENT.value,
            object_id=document.id,
            dimension=dimension,
            suggested_value_id=candidate.id,
            suggested_value_text=candidate.canonical_name,
            confidence_tier=tier,
            evidence_snippet=evidence,
            evidence_rule=rule,
        )
        summary.suggestions_created += 1

        if tier == ConfidenceTier.HIGH:
            try:
                self._objects.edit_metadata(
                    KnowledgeObjectType.DOCUMENT, document.id, updated_by=actor, **{field_name: candidate.canonical_name}
                )
                suggestion.status = SuggestionStatus.AUTO_ACCEPTED
                suggestion.reviewed_by = actor
                summary.auto_accepted += 1
                summary.fields_set += 1
                setattr(document, field_name, candidate.canonical_name)  # keep in-memory record consistent for later dimensions this same run
            except Exception as exc:  # noqa: BLE001 -- one bad row must not abort the whole run
                summary.errors.append(f"{document.id} {field_name}: {exc}")
                suggestion.status = SuggestionStatus.PENDING
                summary.pending_review += 1
        elif tier == ConfidenceTier.MEDIUM:
            summary.pending_review += 1
        else:
            suggestion.status = SuggestionStatus.RECORDED
            summary.recorded_low_confidence += 1

        self._suggestions.save(suggestion)

    # --- Relationships (Customer/Region -- APPLIES_TO) ----------------------

    def _classify_relationship(
        self,
        record: "DocumentationRecord | HistoricalInvestigationRecord | KnownBugRecord",
        candidates: list["_Candidate"],
        title: str,
        body: str,
        dimension: ClassificationDimension,
        target_type: KnowledgeObjectType,
        actor: str | None,
        summary: ClassificationRunSummary,
        object_type: KnowledgeObjectType = KnowledgeObjectType.DOCUMENT,
    ) -> None:
        """``object_type`` (2026-08-13, Phase 0) -- which governed type
        ``record`` actually is; defaults to DOCUMENT for source
        compatibility, but Historical Investigations/Known Bugs pass
        their own real type so the resulting relationship's ``from_type``
        (and the suggestion's own ``object_type``/idempotency check) is
        correct rather than silently mislabeling every non-Document
        suggestion as a Document one."""
        match = _best_match(candidates, title, body)
        if match is None:
            return
        candidate, tier, evidence, rule = match
        if self._suggestions.exists_for_object(
            object_type.value, record.id, dimension, candidate.id
        ):
            return

        suggestion = MetadataClassificationSuggestion(
            object_type=object_type.value,
            object_id=record.id,
            dimension=dimension,
            suggested_value_id=candidate.id,
            suggested_value_text=candidate.canonical_name,
            confidence_tier=tier,
            evidence_snippet=evidence,
            evidence_rule=rule,
        )
        summary.suggestions_created += 1

        if tier == ConfidenceTier.HIGH:
            from app.engines.knowledge_relationships.engine import DuplicateRelationshipError, KnowledgeObjectNotFoundError

            try:
                self._relationships.add_relationship(
                    object_type, record.id, target_type, candidate.id, RelationshipType.APPLIES_TO, created_by=actor
                )
                summary.relationships_created += 1
            except DuplicateRelationshipError:
                pass  # already tagged -- not an error, nothing new to count
            except KnowledgeObjectNotFoundError as exc:
                summary.errors.append(f"{record.id} {dimension.value}: {exc}")
            suggestion.status = SuggestionStatus.AUTO_ACCEPTED
            suggestion.reviewed_by = actor
            summary.auto_accepted += 1
        elif tier == ConfidenceTier.MEDIUM:
            summary.pending_review += 1
        else:
            suggestion.status = SuggestionStatus.RECORDED
            summary.recorded_low_confidence += 1

        self._suggestions.save(suggestion)

    # --- related_components (additive list) ---------------------------------

    def _classify_related_components(
        self,
        record: "DocumentationRecord | HistoricalInvestigationRecord | KnownBugRecord",
        candidates: list["_Candidate"],
        title: str,
        body: str,
        actor: str | None,
        summary: ClassificationRunSummary,
        object_type: KnowledgeObjectType = KnowledgeObjectType.DOCUMENT,
    ) -> None:
        """``object_type`` (2026-08-13, Phase 0) -- see
        ``_classify_relationship``'s docstring for why this parameter
        exists. ``related_components`` is a real field on all three
        record types this now runs against (``DocumentationRecord``,
        ``HistoricalInvestigationRecord``, ``KnownBugRecord``) -- same
        additive-list update via ``KnowledgeObjectService.edit_metadata``
        either way, just addressed at the right governed type."""
        match = _best_match(candidates, title, body)
        if match is None:
            return
        candidate, tier, evidence, rule = match
        if candidate.canonical_name in record.related_components:
            return  # already linked -- nothing to suggest
        if self._suggestions.exists_for_object(
            object_type.value, record.id, ClassificationDimension.COMPONENT, candidate.id
        ):
            return

        suggestion = MetadataClassificationSuggestion(
            object_type=object_type.value,
            object_id=record.id,
            dimension=ClassificationDimension.COMPONENT,
            suggested_value_id=candidate.id,
            suggested_value_text=candidate.canonical_name,
            confidence_tier=tier,
            evidence_snippet=evidence,
            evidence_rule=rule,
        )
        summary.suggestions_created += 1

        if tier == ConfidenceTier.HIGH:
            try:
                updated_components = [*record.related_components, candidate.canonical_name]
                self._objects.edit_metadata(
                    object_type, record.id, updated_by=actor, related_components=updated_components
                )
                record.related_components = updated_components
                suggestion.status = SuggestionStatus.AUTO_ACCEPTED
                suggestion.reviewed_by = actor
                summary.auto_accepted += 1
                summary.fields_set += 1
            except Exception as exc:  # noqa: BLE001
                summary.errors.append(f"{record.id} component: {exc}")
                summary.pending_review += 1
        elif tier == ConfidenceTier.MEDIUM:
            summary.pending_review += 1
        else:
            suggestion.status = SuggestionStatus.RECORDED
            summary.recorded_low_confidence += 1

        self._suggestions.save(suggestion)

    # --- Review queue (Medium-confidence suggestions) -----------------------

    def list_pending(self) -> list[MetadataClassificationSuggestion]:
        return self._suggestions.list_by_status(SuggestionStatus.PENDING)

    def accept(self, suggestion_id: str, *, actor: str | None = None) -> MetadataClassificationSuggestion:
        """Applies a Medium-confidence suggestion exactly as the
        High-confidence auto-accept path would have -- same target
        (scalar field / relationship / related_components), just gated
        by a human decision instead of the title-match rule."""
        suggestion = self._suggestions.get(suggestion_id)
        if suggestion is None:
            raise ValueError(f"No such suggestion: {suggestion_id}")
        object_type = KnowledgeObjectType(suggestion.object_type)

        if suggestion.dimension in (ClassificationDimension.TECHNOLOGY, ClassificationDimension.PRODUCT):
            field_name = suggestion.dimension.value
            current = self._objects.get(object_type, suggestion.object_id)
            if current is not None and not getattr(current, field_name, None):
                self._objects.edit_metadata(object_type, suggestion.object_id, updated_by=actor, **{field_name: suggestion.suggested_value_text})
        elif suggestion.dimension == ClassificationDimension.COMPONENT:
            current = self._objects.get(object_type, suggestion.object_id)
            if current is not None and suggestion.suggested_value_text not in current.related_components:
                self._objects.edit_metadata(
                    object_type,
                    suggestion.object_id,
                    updated_by=actor,
                    related_components=[*current.related_components, suggestion.suggested_value_text],
                )
        else:
            from app.engines.knowledge_relationships.engine import DuplicateRelationshipError

            target_type = KnowledgeObjectType.CUSTOMER if suggestion.dimension == ClassificationDimension.CUSTOMER else KnowledgeObjectType.REGION
            try:
                self._relationships.add_relationship(
                    object_type, suggestion.object_id, target_type, suggestion.suggested_value_id, RelationshipType.APPLIES_TO, created_by=actor
                )
            except DuplicateRelationshipError:
                pass

        suggestion.status = SuggestionStatus.ACCEPTED
        suggestion.reviewed_by = actor
        from datetime import datetime, timezone

        suggestion.reviewed_at = datetime.now(timezone.utc)
        self._suggestions.save(suggestion)
        return suggestion

    def reject(self, suggestion_id: str, *, actor: str | None = None) -> MetadataClassificationSuggestion:
        suggestion = self._suggestions.get(suggestion_id)
        if suggestion is None:
            raise ValueError(f"No such suggestion: {suggestion_id}")
        suggestion.status = SuggestionStatus.REJECTED
        suggestion.reviewed_by = actor
        from datetime import datetime, timezone

        suggestion.reviewed_at = datetime.now(timezone.utc)
        self._suggestions.save(suggestion)
        return suggestion
