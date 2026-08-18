"""Structured Resolution Knowledge -- the standalone-record assembly
path and the two conflict-ordering functions (2026-08-14, Phase 1 of
RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md).

Deliberately separate from ``RecommendationEngine``: this module never
touches investigation-scoped state (``_SynthesizedSources``,
``_local_match_for_cause``, etc. stay private to that engine, reused
not re-implemented) -- it only ever reads real, already-governed
records directly via the existing ``KnowledgeRepository``/
``KnowledgeRelationshipEngine``, the same two dependencies every other
read-only browsing surface in this codebase already uses. The
investigation-scoped assembly path (``from_strategy``-equivalent) lives
as a small, additional private method on ``RecommendationEngine``
itself (``_build_structured_resolution``), following the exact same
precedent ``_build_provenance_record`` already established -- see that
method's own module for why (it already has every private helper this
needs in scope; duplicating that state here would be a second
implementation of the same decision logic, which this whole phase
exists to avoid).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.domain.knowledge_relationships import KnowledgeObjectRef, KnowledgeObjectType, RelationshipType
from app.domain.provenance import EvidenceKind, EvidenceReference, ResolutionProvenance, ValidationStep
from app.domain.structured_resolution import ApplicabilitySummary, ResolutionCandidate, StructuredResolution
from app.engines.knowledge_relationships.engine import KnowledgeObjectNotFoundError

if TYPE_CHECKING:
    from app.domain.evidence import HistoricalInvestigationRecord, KnownBugRecord
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.infrastructure.db.knowledge_repository import KnowledgeRepository

_PLACEHOLDER_VALUES = {"", "-", "—"}
"""Empty / em-dash / hyphen placeholders the Knowledge Objects UI's
generic "New" form fills required fields with when an admin leaves them
blank (see ``ui/views/11_Knowledge_Objects.py``'s create form:
``fields.update(root_cause="-", resolution="-")``) -- these are real,
observed placeholder values in this codebase's own UI, not a guessed
list. Excluded from the standalone-record confidence rule so a
manually-created stub record doesn't masquerade as real recorded
content."""

_TIER_RANK = {
    ResolutionProvenance.CONFIRMED: 3,
    ResolutionProvenance.LIKELY: 2,
    ResolutionProvenance.POSSIBLE: 1,
    ResolutionProvenance.UNKNOWN: 0,
}

_SOURCE_PRIORITY = {
    EvidenceKind.HISTORICAL_INVESTIGATION: 3,
    EvidenceKind.TFS_CASE: 2,
    EvidenceKind.WIKI_PAGE: 1,
    EvidenceKind.KNOWN_BUG: 0,
}
"""Tie-break only (used only when two candidates have the exact same
real relevance score) -- mirrors the fixed order
``_synthesize_recommendation`` already evaluates sources in
(local -> TFS -> Wiki -> Known Bug), not a new, independent priority
judgment."""


def _has_real_content(value: str | None) -> bool:
    return bool(value) and value.strip() not in _PLACEHOLDER_VALUES


def order_resolution_candidates(candidates: list[ResolutionCandidate]) -> list[ResolutionCandidate]:
    """Deterministic, explainable ordering for the resolution texts
    contributed by *one investigation's* multiple sources (approved
    design §26, applied within a single ``StructuredResolution``):
    the source already chosen as primary by the existing, unchanged
    ``_synthesize_recommendation`` always sorts first (this function
    never overrides that decision, only orders the alternates around
    it); among the rest, real relevance score wins; ties broken by a
    fixed source priority. No opaque ranking -- every key here is a
    real, already-computed value or a documented fixed order."""
    return sorted(
        candidates,
        key=lambda c: (c.is_primary, c.evidence.score or 0.0, _SOURCE_PRIORITY.get(c.evidence.kind, -1)),
        reverse=True,
    )


def order_resolutions(candidates: list[StructuredResolution]) -> "StructuredResolution | None":
    """Deterministic, explainable ordering across *multiple governed
    records* that all seem relevant to the same question (approved
    design §26) -- e.g. a future Chat Assistant surfacing two Historical
    Investigations that disagree. Returns the single primary result
    with every other real candidate preserved in ``.also_seen`` --
    never silently dropped. None only when ``candidates`` is empty.

    Rule, in order, first one that discriminates wins:
    1. A real ``SUPERSEDES`` edge -- a candidate another candidate in
       this same set explicitly supersedes can never be primary,
       regardless of tier/recency (a human already made this call --
       see ``RelationshipType.SUPERSEDES``'s own docstring).
    2. Higher ``ResolutionProvenance`` tier (CONFIRMED > LIKELY >
       POSSIBLE > UNKNOWN) -- the existing, unchanged four-tier rule.
    3. More recent ``updated_at`` -- the real, existing
       ``GovernanceFields`` timestamp, last resort only.
    """
    if not candidates:
        return None

    # A candidate with superseded_by set IS the superseded (older) one --
    # it names who replaces it, not who it replaces -- so it's excluded
    # by its own source_id, never by the id it points at.
    superseded_source_ids = {c.source_id for c in candidates if c.superseded_by is not None}
    eligible = [c for c in candidates if c.source_id not in superseded_source_ids] or list(candidates)
    """Falls back to the full set if every candidate is somehow marked
    superseded (a real data inconsistency, e.g. a cycle) rather than
    returning nothing -- same "never silently produce an empty answer
    when real candidates exist" discipline as everywhere else in this
    codebase."""

    ordered = sorted(
        eligible,
        key=lambda c: (_TIER_RANK[c.confidence], c.updated_at or datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True,
    )
    primary = ordered[0]
    alternates = [c.model_copy(update={"also_seen": []}) for c in candidates if c.source_id != primary.source_id]
    return primary.model_copy(update={"also_seen": alternates})


def applicability_for_object(
    relationship_engine: "KnowledgeRelationshipEngine | None", object_type: KnowledgeObjectType, object_id: str
) -> ApplicabilitySummary:
    """Real, governed Customer/Region tags only, via the existing
    relationship graph -- the exact mechanism ``ApplicabilityRanker``
    already uses for retrieval-time boosting, reused here for display.
    Never inferred from title/body text (that guarantee lives
    structurally in the classification engine, unchanged by this
    phase -- see app/engines/knowledge/classification.py). A module-
    level function, not a method, specifically so
    ``RecommendationEngine`` can call it too without duplicating this
    logic for the investigation-scoped assembly path."""
    if relationship_engine is None:
        return ApplicabilitySummary()
    customer_names: list[str] = []
    region_names: list[str] = []
    for rel in relationship_engine.list_relationships(object_type, object_id):
        if rel.relationship.relationship_type != RelationshipType.APPLIES_TO:
            continue
        is_from = rel.relationship.from_type == object_type
        other = rel.to_object if is_from else rel.from_object
        if other.type == KnowledgeObjectType.CUSTOMER:
            customer_names.append(other.title)
        elif other.type == KnowledgeObjectType.REGION:
            region_names.append(other.title)
    return ApplicabilitySummary(customer_names=customer_names, region_names=region_names)


class StructuredResolutionEngine:
    """Assembles a ``StructuredResolution`` from a standalone, already-
    governed Historical Investigation or Known Bug record -- no open
    investigation, no live TFS/Wiki query (this is meant to be a cheap
    detail view, not a second Analyze). Both dependencies are the exact
    same ones every other read-only browsing surface in this codebase
    already uses."""

    def __init__(
        self,
        knowledge_repo: "KnowledgeRepository",
        relationship_engine: "KnowledgeRelationshipEngine | None" = None,
    ) -> None:
        self._knowledge = knowledge_repo
        self._relationships = relationship_engine

    def from_historical_investigation(self, record_id: str) -> "StructuredResolution | None":
        record = self._knowledge.get_historical_investigation(record_id)
        if record is None:
            return None
        return self._assemble(
            kind="historical_investigation",
            object_type=KnowledgeObjectType.HISTORICAL_INVESTIGATION,
            evidence_kind=EvidenceKind.HISTORICAL_INVESTIGATION,
            record=record,
            root_cause=record.root_cause,
            resolution_text=record.resolution,
            next_step=record.next_step,
            strong_content_value=record.root_cause,
        )

    def from_known_bug(self, bug_id: str) -> "StructuredResolution | None":
        record = self._knowledge.get_known_bug(bug_id)
        if record is None:
            return None
        # KnownBugRecord has no root_cause field at all -- never
        # invented; see app/domain/evidence.py's KnownBugRecord. It also
        # has no next_step field, so Known-Bug-sourced validation_steps
        # are honestly empty -- a real, documented Phase 1 limitation.
        return self._assemble(
            kind="known_bug",
            object_type=KnowledgeObjectType.KNOWN_BUG,
            evidence_kind=EvidenceKind.KNOWN_BUG,
            record=record,
            root_cause=None,
            resolution_text=record.workaround,
            next_step=None,
            strong_content_value=record.workaround,
        )

    def _assemble(
        self,
        *,
        kind: str,
        object_type: KnowledgeObjectType,
        evidence_kind: EvidenceKind,
        record: "HistoricalInvestigationRecord | KnownBugRecord",
        root_cause: str | None,
        resolution_text: str | None,
        next_step: str | None,
        strong_content_value: str | None,
    ) -> StructuredResolution:
        applicability = self._applicability_for(object_type, record.id)
        applicability.component_names = list(record.related_components)

        confidence, rationale = self._standalone_tier(record, strong_content_value)

        root_cause_evidence: list[EvidenceReference] = []
        if _has_real_content(root_cause):
            root_cause_evidence.append(
                EvidenceReference(
                    kind=evidence_kind,
                    source_id=record.id,
                    title=record.title,
                    reason="The record's own recorded root cause.",
                    contributes_to=["root_cause"],
                )
            )

        resolution_candidates: list[ResolutionCandidate] = []
        if _has_real_content(resolution_text):
            resolution_candidates.append(
                ResolutionCandidate(
                    text=resolution_text,
                    evidence=EvidenceReference(
                        kind=evidence_kind,
                        source_id=record.id,
                        title=record.title,
                        reason="The record's own recorded resolution/workaround.",
                        contributes_to=["resolution"],
                    ),
                    is_primary=True,
                )
            )

        validation_steps: list[ValidationStep] = []
        if _has_real_content(next_step):
            validation_steps.append(
                ValidationStep(
                    instruction=next_step,
                    source=evidence_kind,
                    source_id=record.id,
                    source_title=record.title,
                )
            )

        return StructuredResolution(
            source_kind=kind,
            source_id=record.id,
            problem=record.title,
            symptoms=record.description,
            applicability=applicability,
            root_cause=root_cause if _has_real_content(root_cause) else None,
            root_cause_evidence=root_cause_evidence,
            resolution_candidates=resolution_candidates,
            validation_steps=validation_steps,
            related=self._related_for(object_type, record.id),
            confidence=confidence,
            confidence_rationale=rationale,
            updated_at=record.updated_at,
        )

    def _applicability_for(self, object_type: KnowledgeObjectType, object_id: str) -> ApplicabilitySummary:
        return applicability_for_object(self._relationships, object_type, object_id)

    def _related_for(self, object_type: KnowledgeObjectType, object_id: str) -> list[KnowledgeObjectRef]:
        """Reuses the existing Relationship Explorer wholesale -- no
        new graph traversal."""
        if self._relationships is None:
            return []
        try:
            view = self._relationships.get_explorer_view(object_type, object_id)
        except KnowledgeObjectNotFoundError:
            return []
        return [ref for group in view.groups for ref in group.objects]

    @staticmethod
    def _standalone_tier(
        record: "HistoricalInvestigationRecord | KnownBugRecord", strong_content_value: str | None
    ) -> tuple[ResolutionProvenance, str]:
        """The standalone-record analogue of
        ``RecommendationEngine._resolve_provenance_tier`` -- same four-
        tier vocabulary, same "never CONFIRMED without explicit human
        verification" rule, adapted for "trust this record on its own"
        (nothing is being matched against a query here, so there is no
        similarity score to *not* trust alone in the first place).

        CONFIRMED: ``resolution_verified`` -- the exact same field, the
        exact same explicit-human-action requirement, as the
        investigation-scoped rule.
        LIKELY: the record's own most relevant content field
        (``root_cause`` for a Historical Investigation, ``workaround``
        for a Known Bug -- mirroring exactly which field each source
        type's investigation-scoped LIKELY rule already checks) has
        real, non-placeholder content.
        POSSIBLE: the record exists with a real description but its
        strongest content field doesn't.
        UNKNOWN: never reached for either governed type today, since
        ``description`` is a required field on both -- documented as a
        known Phase 1 limitation, not silently glossed over."""
        if getattr(record, "resolution_verified", False):
            by = record.resolution_verified_by or "an administrator"
            when = record.resolution_verified_at.isoformat() if record.resolution_verified_at else "an unrecorded date"
            note = f" -- {record.resolution_verification_note}" if record.resolution_verification_note else ""
            return ResolutionProvenance.CONFIRMED, f"Explicitly verified by {by} on {when}{note}."

        if _has_real_content(strong_content_value):
            return (
                ResolutionProvenance.LIKELY,
                "This record's own recorded content is real and non-placeholder -- no independent "
                "corroborating source, so this stays Likely rather than Confirmed.",
            )
        if _has_real_content(record.description):
            return (
                ResolutionProvenance.POSSIBLE,
                "A real description exists on this record, but its strongest resolution-relevant field "
                "is missing or a placeholder -- treat as a hypothesis to verify.",
            )
        return ResolutionProvenance.UNKNOWN, "No real content recorded on this record yet."
