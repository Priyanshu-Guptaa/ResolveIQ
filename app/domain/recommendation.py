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
    suggested_sql: list[str] = Field(default_factory=list)
    next_best_step: str = ""
