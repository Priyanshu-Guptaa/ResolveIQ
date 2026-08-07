"""API request/response schemas.

Response payloads mostly re-export domain models directly (they're already
Pydantic and already display-ready) -- introducing a parallel DTO layer for
every response would be premature for Sprint 1. Request bodies get their
own thin schemas since "what the API accepts" and "what the domain models"
are different concerns from day one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.domain.investigation import ActivityItem, DashboardStats
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.recommendation import KnowledgeMatch
from app.domain.sql_studio import QueryTemplate


class CreateInvestigationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    description: str = ""


class AddNoteRequest(BaseModel):
    text: str = Field(min_length=1)


class UpdateInvestigationDetailsRequest(BaseModel):
    """Persistent Summary Card fields (RFC rev 3, Phase 2A) -- all
    optional, engineer-entered. Fields left as None are left unchanged,
    not cleared (see InvestigationEngine.update_details)."""

    customer: str | None = None
    product: str | None = None
    version: str | None = None
    technology: str | None = None
    assigned_engineer: str | None = None


class UpdateDocumentMetadataRequest(BaseModel):
    """Knowledge Management's Metadata step (Sprint 3, Phase 3.2) --
    all optional, same "None means unchanged" convention as
    UpdateInvestigationDetailsRequest. ``tags``/``related_components``
    are lists, not strings, so an empty list *does* mean "clear it" --
    only ``None`` (the field omitted from the request) leaves it alone."""

    title: str | None = None
    tags: list[str] | None = None
    product: str | None = None
    version: str | None = None
    technology: str | None = None
    related_components: list[str] | None = None


class CreateRelationshipRequest(BaseModel):
    """Knowledge Relationship Manager (Sprint 3, Phase 3.3) -- both ends
    are always a real (type, id) pair from a searchable picker, never
    free text; the engine independently re-verifies both actually exist
    before writing the edge."""

    from_type: KnowledgeObjectType
    from_id: str
    to_type: KnowledgeObjectType
    to_id: str
    relationship_type: RelationshipType = RelationshipType.RELATED_TO
    created_by: str | None = None


class CreatePlaybookRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    product: str | None = None
    description: str = ""
    steps: list[str] = Field(default_factory=list)
    created_by: str | None = None


class CreateLookupEntityRequest(BaseModel):
    """Shared by Product and Technology -- both are just a name."""

    name: str = Field(min_length=1, max_length=200)
    created_by: str | None = None


class CreateVersionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    product_id: str | None = None
    created_by: str | None = None


class KnowledgeObjectWriteRequest(BaseModel):
    """Knowledge Object Framework (Sprint 3, Phase 3.4) -- ``fields``
    is intentionally an open bag of keys rather than a per-type schema:
    ``KnowledgeObjectService.create``/``edit_metadata`` hand it straight
    to the target type's own Pydantic model, which is what actually
    validates it. Keeping request validation generic here is what lets
    one router endpoint serve all nine object types without knowing
    any of their individual shapes."""

    fields: dict[str, Any] = Field(default_factory=dict)
    actor: str | None = None


class InvestigationSummary(BaseModel):
    """Lightweight shape for the investigation list view."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    last_viewed_at: str | None = None
    evidence_count: int


class DashboardResponse(BaseModel):
    """Everything the Dashboard (RFC rev 3, §08) renders, in one call --
    every field backed by a real query, none of it illustrative."""

    stats: DashboardStats
    active_investigations: list[InvestigationSummary]
    recently_viewed: list[InvestigationSummary]
    recent_activity: list[ActivityItem]
    recent_knowledge: list[KnowledgeMatch]
    recent_known_bugs: list[KnowledgeMatch]
    query_library_preview: list[QueryTemplate]
