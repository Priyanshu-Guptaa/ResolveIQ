"""Output models produced by the Recommendation Engine."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.domain.enums import KnowledgeCollection


class KnowledgeMatch(BaseModel):
    """A single semantic-search hit against a knowledge collection."""

    collection: KnowledgeCollection
    record_id: str
    title: str
    snippet: str
    score: float = Field(ge=0.0, le=1.0)
    """Similarity score normalized to [0, 1], 1.0 = exact match."""
    metadata: dict = Field(default_factory=dict)


class RootCauseHypothesis(BaseModel):
    """A candidate root cause with supporting rationale."""

    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RecommendedLogCollectionItem(BaseModel):
    """One entry in the structured "Recommended Log Collection" guidance
    (Log Intelligence) -- what replaces the generic "upload logs" hint
    with the exact log, in order, and why. Sourced entirely from a
    ``LogCollectionScenario`` matched against the investigation's own
    text (see ``RecommendationEngine._recommend_logs``); every field
    below except ``priority_label`` is copied straight from the
    wiki-derived record, never fabricated.
    """

    priority_label: str
    """"Critical" | "Recommended" | "Optional" -- Critical requires the
    scenario's *technology* and *issue type* (``scenario_type``) to both
    match the investigation text; Recommended is a technology-only full
    match; Optional is a partial/single-keyword match. See
    ``RecommendationEngine._recommend_logs`` -- keyword match strength,
    never an LLM judgment."""
    match_reason: str
    """A one-line, deterministically-built explanation of *why this
    scenario* was selected for this investigation (which matched
    keywords drove the label) -- distinct from ``explanation``, which
    explains this step's position in the message flow, not why the
    scenario itself was recommended."""
    order: int
    """The wiki-preserved collection order within its scenario, 1 =
    first -- copied from ``LogCollectionStep.priority``."""
    component_name: str
    scenario_technology: str
    scenario_type: str
    scenario_region: str | None = None
    explanation: str
    """The wiki-derived explanation, verbatim from ``LogCollectionStep``."""
    repository_platform: str
    repository_root_path: str
    repository_subdirectory: str | None = None
    filename_patterns: list[str] = Field(default_factory=list)
    log_source_id: str
    scenario_id: str
    already_collected: bool = False
    """True when a piece of already-uploaded evidence appears to be
    this exact log (filename or component name match against the
    investigation's own evidence titles) -- lets the guided workflow
    skip straight to what's actually still missing instead of asking
    for logs already in hand."""
    linked_component_id: str | None = None
    linked_component_name: str | None = None
    """Set when this log source has a real, deterministic
    IMPLEMENTS_LOGGING_FOR relationship to a Product Intelligence
    Component Registry entry (Sprint 3, Phase 3.3/3.4 relationship
    graph) -- lets the UI jump straight to that component's Architecture
    Explorer profile. None when no such match exists; never guessed."""


class InvestigationStage(str, Enum):
    """Where this investigation currently stands -- derived purely from
    what's actually present (evidence, recommended-vs-collected logs,
    root-cause match confidence), never from an engineer-set status
    field (that's ``InvestigationSession.status``, a different,
    manually-controlled concept). Four stages only, deliberately: each
    boundary is a real, checkable signal already computed elsewhere in
    this engine -- adding a "Confirmation"/"Resolution" split would
    require inventing a signal (e.g. "was this SQL actually run and did
    it confirm the hypothesis?") that nothing in the system tracks yet.
    """

    TRIAGE = "triage"
    """No evidence at all yet."""
    EVIDENCE_COLLECTION = "evidence_collection"
    """Evidence exists, but something identified as required (a
    recommended log, mainly) hasn't been collected yet."""
    ANALYSIS = "analysis"
    """Required evidence is in hand, but no historical/known-bug match
    has reached the confidence threshold that would count as a root
    cause."""
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    """A historical investigation or known bug matched above the
    confidence threshold -- next step is confirming it, not searching
    for it."""


class RequiredEvidenceItem(BaseModel):
    """One thing this investigation needs, deduplicated at the
    component/source level (not per-file) from the ordered log
    collection plus any non-log entity-heuristic hints -- a checklist
    view, not a duplicate of ``RecommendedLogCollectionItem``'s detail
    (repository path, filenames, etc. stay in ``ordered_log_collection``
    only; this just points back to it via ``related_log_source_id``)."""

    description: str
    satisfied: bool
    source: str
    """"log_intelligence" | "entity_heuristic" -- which existing signal
    this item came from, never a new one invented for this view."""
    related_log_source_id: str | None = None
    """Set for source == "log_intelligence": the LogSourceApplication id
    to cross-reference against ``ordered_log_collection`` for full detail."""


class SuggestedSqlItem(BaseModel):
    """One SQL suggestion -- either a real, governed SQL Library
    ``QueryTemplate`` (when the matched component links to one via its
    real, already-populated ``related_components`` field) or, absent a
    component match, the existing entity-heuristic snippet. Never a
    fabricated query."""

    title: str
    sql_text: str
    explanation: str = ""
    source: str
    """"sql_library" | "entity_heuristic"."""
    template_id: str | None = None
    """Set for source == "sql_library" -- the real QueryTemplate id, so
    the UI can link to SQL Studio instead of duplicating its content."""


class MatchedComponent(BaseModel):
    """The Product Intelligence Component Registry entry this
    investigation matched against, if any -- deterministic name/keyword
    matching against the investigation's text and extracted entities
    (see ``RecommendationEngine._match_component``), never an LLM
    judgment. A reference only (id/name/why) -- the UI fetches the full
    ``ComponentProfile`` from the existing Product Intelligence engine/
    panel rather than this duplicating its content."""

    component_id: str
    component_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    match_reason: str


class InvestigationStrategy(BaseModel):
    """The single, explainable, orchestrated investigation plan --
    Recommendation Engine V2's primary output (approved design:
    "the new Investigation Strategy should become the PRIMARY
    recommendation experience"). Ties together every knowledge module
    already built without duplicating any of their content: the
    historical-investigation/known-bug/documentation/log-collection
    sections below are the *exact same* objects already computed for
    ``Recommendation``'s legacy flat fields, just organized into one
    guided hierarchy instead of parallel, independent panels. Those
    legacy fields remain on ``Recommendation`` unchanged for API
    backward compatibility; this is the new UI's primary surface.

    Every field here traces back to a real, already-computed signal --
    nothing in this class introduces a new matching mechanism beyond
    what ``_match_component`` (technology/component keyword matching,
    the same deterministic pattern Log Intelligence already uses) adds.
    No LLM reasoning anywhere, consistent with every prior phase.
    """

    current_stage: InvestigationStage
    stage_rationale: str
    progress: float = Field(ge=0.0, le=1.0)
    progress_summary: str
    recommended_next_action: str
    """Reuses the exact same computation as the legacy
    ``Recommendation.next_best_step`` -- not re-derived, just also
    surfaced here as the strategy's headline action."""
    next_action_rationale: str
    required_evidence: list[RequiredEvidenceItem] = Field(default_factory=list)
    missing_evidence: list[RequiredEvidenceItem] = Field(default_factory=list)
    """The subset of ``required_evidence`` where ``satisfied`` is False --
    computed, not independently maintained."""
    ordered_log_collection: list[RecommendedLogCollectionItem] = Field(default_factory=list)
    """The exact same list as ``Recommendation.recommended_logs`` --
    reused by reference, not recomputed."""
    suggested_sql: list[SuggestedSqlItem] = Field(default_factory=list)
    matched_component: MatchedComponent | None = None
    historical_investigations: list[KnowledgeMatch] = Field(default_factory=list)
    """The exact same list as ``Recommendation.similar_investigations``."""
    known_bugs: list[KnowledgeMatch] = Field(default_factory=list)
    """The exact same list as ``Recommendation.known_bugs``."""
    documentation: list[KnowledgeMatch] = Field(default_factory=list)
    """The exact same list as ``Recommendation.relevant_documentation``."""
    decision_checkpoint: str | None = None
    """The specific thing to verify/decide next -- built from how many
    root-cause candidates exist (distinguish between them if >1, confirm
    the one if exactly 1, None if there's nothing yet to decide between)."""


class Recommendation(BaseModel):
    """The full output of the Recommendation Engine for one investigation.

    This is deliberately a flat, display-ready shape -- the Streamlit UI
    (and any future UI) should be able to render it directly without extra
    business logic.
    """

    investigation_id: str
    root_causes: list[RootCauseHypothesis] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    similar_investigations: list[KnowledgeMatch] = Field(default_factory=list)
    relevant_documentation: list[KnowledgeMatch] = Field(default_factory=list)
    known_bugs: list[KnowledgeMatch] = Field(default_factory=list)
    suggested_logs: list[str] = Field(default_factory=list)
    recommended_logs: list[RecommendedLogCollectionItem] = Field(default_factory=list)
    """Structured Log Intelligence guidance (Sprint 3 follow-up) --
    populated when a ``LogCollectionScenario`` matches the investigation
    text; empty when Log Intelligence has no matching wiki-derived
    scenario yet, in which case ``suggested_logs`` remains the fallback
    guidance."""
    suggested_sql: list[str] = Field(default_factory=list)
    next_best_step: str = ""
    strategy: InvestigationStrategy
    """Recommendation Engine V2's primary output -- the orchestrated,
    single investigation plan the UI now guides engineers through.
    Always populated, even for a brand-new investigation with no
    evidence yet (stage TRIAGE, empty sections) -- the UI renders one
    consistent hierarchy regardless of investigation state rather than
    branching on a null case. Every other field on this model stays
    populated exactly as before for API backward compatibility."""
