"""History (Sprint 3, Phase 3.4 -- Knowledge Object Framework).

One shared version-snapshot shape for every governed object type,
exactly RFC-003's originally-planned ``EntityVersion`` (§ "Version
Control"), delivered now rather than deferred again -- the same "one
generic table instead of nine per-type ones" move as
``KnowledgeRelationship`` in Phase 3.3.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.domain.knowledge_relationships import KnowledgeObjectType


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EntityVersion(BaseModel):
    id: str = ""
    object_type: KnowledgeObjectType
    object_id: str
    version_number: int
    snapshot: dict = Field(default_factory=dict)
    """The object's full ``model_dump(mode="json")`` at save time --
    generic across every domain model, so History needed zero per-type
    schema knowledge to add."""
    changed_by: str | None = None
    changed_at: datetime = Field(default_factory=_utcnow)
    change_summary: str = ""
