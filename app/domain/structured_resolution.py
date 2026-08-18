"""Structured Resolution Knowledge (2026-08-14, Phase 1 of
RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md).

A READ MODEL, not a new source of truth. Every field here is either a
direct projection of data ``RecommendationEngine`` already computes
(``InvestigationStrategy``, ``ProvenanceRecord``, ``RecommendedSolution``,
``_SynthesizedSources``) or, for the standalone-record path, a real,
already-governed field/relationship fetched fresh from the existing
repositories (``KnowledgeRepository``, ``KnowledgeRelationshipEngine``).
Nothing in this module introduces a new retrieval mechanism, a new
matching algorithm, or a new confidence system -- see
``app.domain.provenance`` for the one, unchanged, four-tier
``ResolutionProvenance`` this module reuses verbatim.

Two assembly paths (see ``app.engines.structured_resolution.engine``):
- ``from_strategy()`` -- from an already-computed ``InvestigationStrategy``
  (the common case: an engineer just ran Analyze). Pure projection.
- ``from_record()`` -- from a standalone ``HistoricalInvestigationRecord``/
  ``KnownBugRecord`` id, with no open investigation (browsing Historical
  Investigations/Known Bugs directly). Small, new, self-contained.

Both paths produce this same shape, so a future Chat Assistant (not
built yet) never needs to know which path produced the evidence it's
citing.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.knowledge_relationships import KnowledgeObjectRef
from app.domain.provenance import EvidenceReference, ResolutionProvenance, ValidationStep


class ApplicabilitySummary(BaseModel):
    """Which real, governed Customer/Region/Component/Technology values
    this resolution is known to apply to -- never inferred, never
    guessed. Empty lists are an honest "not yet tagged" (see the
    classification review-queue backlog), not "applies everywhere."
    Product/Version are omitted here (not scalar/relationship fields on
    ``HistoricalInvestigationRecord``/``KnownBugRecord`` today -- adding
    them would be inventing a field with no real source, which Part 1
    explicitly rules out)."""

    customer_names: list[str] = Field(default_factory=list)
    region_names: list[str] = Field(default_factory=list)
    component_names: list[str] = Field(default_factory=list)
    technology_name: str | None = None
    """Singular (unlike the others, which are real multi-valued
    relationships) -- mirrors ``RetrievalContext.technology_name``,
    already resolved elsewhere by the existing, unchanged hierarchy-
    aware ``_match_single_technology``. None, not guessed, when not
    available to the caller (see ``from_strategy``'s docstring)."""


class ResolutionCandidate(BaseModel):
    """One source's own literal resolution/workaround text -- never
    blended with another source's text, never silently dropped just
    because a different source's text ended up as ``RecommendedSolution.
    recommended_resolution``. Added 2026-08-14 (Phase 1) to close a real
    gap found in the existing ``_synthesize_recommendation``: today, if
    two sources both have real resolution content, only one survives
    into that single string (and, for Local vs. TFS specifically, the
    later block unconditionally overwrites the earlier one -- see this
    phase's report for the exact finding). This model does not change
    that existing selection -- ``is_primary`` faithfully reports which
    source's text is *actually* the one already in
    ``recommended_resolution`` today; it does not invent a new,
    independent ranking that could drift from it."""

    text: str
    evidence: EvidenceReference
    is_primary: bool
    """True for exactly the one candidate whose text equals
    ``RecommendedSolution.recommended_resolution`` -- reports the
    existing engine's real decision, never a second opinion."""


class StructuredResolution(BaseModel):
    """The full "Problem -> Symptoms -> Applicability -> Root Cause ->
    Evidence -> Resolution -> Validation -> Related -> Confidence"
    shape, computed fresh every time (like ``ProvenanceRecord`` and
    ``RecommendedSolution`` already are) -- never persisted, never a
    second source of truth to keep in sync with the real records it
    projects.

    Answers "why do you believe this resolution applies?" by carrying
    every ``EvidenceReference`` needed to trace back to the real
    underlying source -- the same shape ``ProvenanceRecord`` already
    uses, reused here rather than inventing a second evidence format.
    """

    source_kind: str
    """"historical_investigation" | "known_bug" | "investigation" --
    what this was assembled from. Uses plain strings, not
    ``EvidenceKind``, for the same reason ``KnowledgeRelationship.
    from_type`` does: "investigation" (a live, in-progress
    InvestigationSession) isn't a governed knowledge object type and
    has no ``EvidenceKind`` member of its own."""
    source_id: str
    problem: str
    symptoms: str
    applicability: ApplicabilitySummary = Field(default_factory=ApplicabilitySummary)
    root_cause: str | None = None
    root_cause_evidence: list[EvidenceReference] = Field(default_factory=list)
    """Reused verbatim from ``ProvenanceRecord.root_cause_evidence`` --
    already a list (Recommendation Engine V2 already surfaces >=1
    competing root-cause hypothesis when more than one exists), so
    root-cause conflicts were already represented before this phase;
    nothing new needed here beyond projecting it."""
    resolution_candidates: list[ResolutionCandidate] = Field(default_factory=list)
    """Every source with real resolution content, each attributed and
    ordered (see ``order_resolution_candidates`` -- deterministic:
    real relevance score first, ties broken by the same fixed source
    precedence ``_synthesize_recommendation`` already uses). Never
    empty when ``resolution_evidence`` on the underlying
    ``ProvenanceRecord`` is non-empty; may contain more than one entry
    even when ``RecommendedSolution.recommended_resolution`` only ever
    shows one -- this is the explicit "represent the conflict, don't
    hide it" requirement."""
    validation_steps: list[ValidationStep] = Field(default_factory=list)
    """Reused verbatim from ``RecommendedSolution.validation_steps`` --
    knowledge describing how an engineer can verify a resolution
    actually worked, never a claim that ResolveIQ (or anyone) already
    performed that verification. See ``ValidationStep``'s own docstring
    for the exact "instruct vs. already-done" distinction this
    maintains."""
    log_evidence: list[EvidenceReference] = Field(default_factory=list)
    """Reused verbatim from ``ProvenanceRecord.log_recommendation_evidence``."""
    sql_evidence: list[EvidenceReference] = Field(default_factory=list)
    """Reused verbatim from ``ProvenanceRecord.sql_recommendation_evidence``."""
    related: list[KnowledgeObjectRef] = Field(default_factory=list)
    """From the existing Relationship Explorer (``KnowledgeRelationshipEngine.
    get_explorer_view``) when a real object id is available -- reused
    as-is, not a new graph traversal."""
    confidence: ResolutionProvenance = ResolutionProvenance.UNKNOWN
    """The exact same four-tier enum ``ProvenanceRecord`` already uses --
    never a second confidence system. For ``from_strategy``, copied
    directly from ``ProvenanceRecord.resolution_provenance``. For
    ``from_record`` (a standalone record with nothing to correlate
    against), computed by a small, separate rule documented on
    ``StructuredResolutionEngine.from_record`` -- same vocabulary, same
    "never Confirmed without explicit verification" guarantee, adapted
    for "trust this record on its own" rather than "trust this match
    against a query."""
    confidence_rationale: str = ""
    superseded_by: KnowledgeObjectRef | None = None
    """Set when a real ``SUPERSEDES`` relationship names a newer record
    -- this one is still returned in full (never silently dropped),
    just labeled. None when no such edge exists (the common case today
    -- zero ``SUPERSEDES`` edges exist in the live corpus yet)."""
    also_seen: list["StructuredResolution"] = Field(default_factory=list)
    """Other real candidates that were considered but ranked below this
    one by ``order_resolutions`` -- never hidden, always attributed.
    Empty in the common single-candidate case."""
    updated_at: datetime | None = None
    """The underlying record's own ``GovernanceFields.updated_at`` --
    the real, existing source for ``order_resolutions``'s recency
    tiebreak (approved design §26, rule 3). None only when this
    ``StructuredResolution`` wasn't assembled from a single governed
    record (e.g. an investigation-scoped ``from_strategy`` result with
    no one clearly-backing record)."""
