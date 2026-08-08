"""Output models produced by the Recommendation Engine."""

from __future__ import annotations

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
    """"Critical" | "Recommended" | "Optional" -- derived from how
    confidently the *scenario* matched the investigation (keyword match
    strength, not an LLM judgment); the wiki itself carries no such
    label."""
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
