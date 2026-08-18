"""Resolution Provenance (approved design, 2026-08-13) -- makes every
recommendation ResolveIQ produces traceable back to the real evidence
that produced it, and separates "how strong was the textual match"
from "how much should an engineer trust this as a resolution."

Deliberately the same shape sketched for the (separately gated, not
approved, not started) LLM Gateway design's ``EvidenceItem`` -- this is
the correct shape either way, and building it now means that future
phase reuses it instead of inventing a second one. No LLM, no new
retrieval, no external dependency: every field here is populated from
data ``RecommendationEngine`` already computes today (``KnowledgeMatch``,
``ExternalMatch``, ``RecommendedLogCollectionItem``, ``SuggestedSqlItem``,
``RootCauseHypothesis``, ``RecommendedSolution``) -- this module only
gives that existing reasoning a uniform, structured, traceable shape.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResolutionProvenance(str, Enum):
    """How much an engineer should trust a proposed resolution --
    deliberately separate from any similarity/confidence *score*. A
    score measures how well text matched; this measures how much
    corroboration the underlying claim actually has. See
    ``RecommendationEngine._resolve_provenance_tier`` for the exact,
    deterministic rule -- summarized here so the four values are never
    read in isolation from what they mean:

    - CONFIRMED: a real cross-source correlation (two independent
      systems -- e.g. TFS and a local historical investigation --
      agreeing on the same real-world case via a shared ticket/CRM
      reference) or an explicit human verification
      (``HistoricalInvestigationRecord.resolution_verified`` /
      ``KnownBugRecord.resolution_verified``). Never from a similarity
      score alone, however high.
    - LIKELY: a single strong source with real resolution content (a
      local historical investigation at/above the root-cause
      similarity threshold *with a recorded root cause on file*, or a
      High-confidence TFS/Wiki match with real resolution text) --
      but no independent corroboration.
    - POSSIBLE: real evidence exists (entity heuristics, weaker
      external matches, a locally similar case with no recorded root
      cause) but doesn't reach the LIKELY bar -- a hypothesis to
      verify, not a resolution to act on.
    - UNKNOWN: no source clears any bar -- ResolveIQ says so rather
      than guessing (mirrors ``RecommendedSolution.insufficient_evidence``,
      which this tier is built from, not a second, independent signal).
    """

    CONFIRMED = "confirmed"
    LIKELY = "likely"
    POSSIBLE = "possible"
    UNKNOWN = "unknown"


class EvidenceKind(str, Enum):
    """Which real ResolveIQ knowledge source a piece of evidence came
    from -- always resolvable back to a real record via ``source_id``."""

    HISTORICAL_INVESTIGATION = "historical_investigation"
    KNOWN_BUG = "known_bug"
    DOCUMENTATION = "documentation"
    TFS_CASE = "tfs_case"
    WIKI_PAGE = "wiki_page"
    SQL_TEMPLATE = "sql_template"
    LOG_COLLECTION_STEP = "log_collection_step"
    ENTITY_HEURISTIC = "entity_heuristic"
    COMPONENT_MATCH = "component_match"


class EvidenceReference(BaseModel):
    """One piece of evidence, uniformly shaped regardless of source --
    answers "why was this recommended" for any of the object types
    ResolveIQ surfaces. Every field is copied from data the engine
    already computed; this model never derives a new signal."""

    kind: EvidenceKind
    source_id: str
    """The real id on the underlying record -- ``KnowledgeMatch.record_id``,
    ``TfsCase.tfs_id``, ``WikiPage.page_id``, ``QueryTemplate.id``,
    ``LogCollectionScenario.id``, or (for an entity-heuristic hypothesis
    with no governed record behind it) the hypothesis's own description
    text, so it's still a stable, referenceable string."""
    title: str
    url: str | None = None
    """Deep link back to the source system -- already present on
    TFS/Wiki matches; None for local-only sources (Historical
    Investigations/Known Bugs/SQL/Log steps have no external URL,
    only a local record id an internal UI can resolve)."""
    score: float | None = None
    """The real, already-computed relevance/confidence score, when one
    exists -- None for evidence with no numeric score (an entity
    heuristic, a log collection step)."""
    reason: str
    """Deterministic explanation, reused wherever one already exists
    (``ExternalMatch.match_reasons``, ``RecommendedLogCollectionItem.
    match_reason``, ``RootCauseHypothesis.rationale``) -- newly
    synthesized only for the two real gaps this phase closes
    (``KnowledgeMatch.reason``, ``SuggestedSqlItem.match_reason``)."""
    contributes_to: list[str] = Field(default_factory=list)
    """Which part(s) of the answer this evidence supports --
    "root_cause" | "resolution" | "log_recommendation" |
    "sql_recommendation". An item can contribute to more than one
    (e.g. the same historical investigation can back both the root
    cause and the resolution)."""


class ValidationStep(BaseModel):
    """Knowledge describing how an engineer can verify whether a
    proposed resolution actually applies -- added 2026-08-14, Phase 1
    (Structured Resolution Knowledge). Deliberately just an
    ``instruction`` (imperative/descriptive text) plus real source
    attribution -- there is no ``completed``/``executed``/``verified``
    field anywhere on this model, and that absence is the design: this
    object can only ever say "here is how to check," never "this was
    checked." The two are fundamentally different claims (per the
    approved design), and the *only* place ResolveIQ ever asserts the
    second one is the existing, separate, human-driven
    ``resolution_verified`` flag on ``HistoricalInvestigationRecord``/
    ``KnownBugRecord`` (which drives the CONFIRMED provenance tier) --
    never this model. Not an execution engine: nothing here runs a
    check, calls a script, or queries a live system; it is text a human
    reads and then, separately, acts on.

    Lives here (not in ``app.domain.recommendation``, where
    ``RecommendedSolution`` -- its only other consumer -- is defined)
    specifically so both ``recommendation.py`` and
    ``structured_resolution.py`` can depend on it without a circular
    import between those two modules -- this is the same "base layer
    both higher modules import from" role ``EvidenceKind``/
    ``EvidenceReference`` already play."""

    instruction: str
    """What to check/do to confirm the resolution worked -- quoted
    verbatim from a real source field when one exists (today, only
    ``HistoricalInvestigationRecord.next_step`` -- see
    ``RecommendationEngine._synthesize_recommendation``), never
    fabricated. Left as an empty list on ``RecommendedSolution`` (not a
    placeholder instruction) when no source provides one -- true for
    Known-Bug-, TFS-, and Wiki-sourced resolutions today, since none of
    those carry an equivalent field yet."""
    expected_result: str | None = None
    """What a result confirming the hypothesis looks like, when the
    source states one explicitly -- None (never guessed) otherwise."""
    disproving_result: str | None = None
    """What a result *disproving* the hypothesis looks like -- the
    explicit counterpart ``expected_result`` needed (added 2026-08-14,
    same phase, per the approved design's requirement that a
    ValidationStep answer both "what would confirm this" and "what
    would disprove it"). Same discipline as every other field here:
    None, not guessed, since the one real source this phase draws from
    (``HistoricalInvestigationRecord.next_step``) is a single free-text
    instruction that doesn't itself separate a confirming outcome from
    a disproving one -- inventing that split from unstructured text
    would be exactly the kind of fabrication this whole model exists to
    avoid. The field exists so a future, richer source (e.g. a
    ``ReleaseNoteEntry`` or a more structured Known Bug workaround) can
    populate it without a schema change; it is honestly unpopulated
    today."""
    source: EvidenceKind
    source_id: str
    source_title: str


class ProvenanceRecord(BaseModel):
    """The full, structured "why" behind one InvestigationStrategy --
    answers every one of the 8 provenance questions in one traceable
    object. Attached as ``InvestigationStrategy.provenance`` --
    additive, computed fresh on every ``generate()`` call exactly like
    ``RecommendedSolution`` already is, never persisted, never a second
    source of truth to keep in sync with the objects it references.

    Documentation matches are deliberately never assigned a
    ``resolution_provenance`` tier of their own -- a wiki/doc match is
    troubleshooting guidance, never itself a resolution claim. Only
    the single, synthesized resolution (``RecommendedSolution``) gets a
    tier; individual pieces of evidence only ever get a ``reason``.
    """

    root_cause_evidence: list[EvidenceReference] = Field(default_factory=list)
    resolution_evidence: list[EvidenceReference] = Field(default_factory=list)
    log_recommendation_evidence: list[EvidenceReference] = Field(default_factory=list)
    sql_recommendation_evidence: list[EvidenceReference] = Field(default_factory=list)
    resolution_provenance: ResolutionProvenance = ResolutionProvenance.UNKNOWN
    provenance_rationale: str = ""
    """Deterministic, human-readable explanation of why this specific
    tier was assigned -- e.g. "Confirmed: TFS-2467475 and local
    investigation ... share ticket CS0122697" or "Likely: single local
    match at 82% similarity with a recorded root cause, no cross-source
    corroboration." Never omitted -- even UNKNOWN states why."""
