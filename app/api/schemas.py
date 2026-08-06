"""API request/response schemas.

Response payloads mostly re-export domain models directly (they're already
Pydantic and already display-ready) -- introducing a parallel DTO layer for
every response would be premature for Sprint 1. Request bodies get their
own thin schemas since "what the API accepts" and "what the domain models"
are different concerns from day one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CreateInvestigationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    description: str = ""


class AddNoteRequest(BaseModel):
    text: str = Field(min_length=1)


class InvestigationSummary(BaseModel):
    """Lightweight shape for the investigation list view."""

    id: str
    title: str
    status: str
    created_at: str
    updated_at: str
    evidence_count: int
