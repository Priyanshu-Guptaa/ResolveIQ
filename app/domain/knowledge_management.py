"""Domain models for the Knowledge Management admin module (Sprint 3,
Phase 3.2) -- the Dashboard's counts and the Library's paginated,
filtered, sorted result shape. Kept separate from ``evidence.py``'s
``DocumentationRecord``/``DocumentationListItem`` since these describe
*views over* documents, not a document itself.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from app.domain.evidence import DocumentationListItem, DocumentationRecord
from app.domain.ingestion import ParseWarning


class KnowledgeDashboardStats(BaseModel):
    """Real counts only -- no illustrative numbers, same discipline as
    ``DashboardStats`` (Phase 1)."""

    total_count: int = 0
    published_count: int = 0
    draft_count: int = 0
    under_review_count: int = 0
    archived_count: int = 0
    recently_added: list[DocumentationListItem] = Field(default_factory=list)


class DocumentPage(BaseModel):
    """One page of the Knowledge Library's search/filter/sort results."""

    items: list[DocumentationListItem] = Field(default_factory=list)
    total_count: int = 0
    page: int = 1
    page_size: int = 20

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_pages(self) -> int:
        """A plain ``@property`` is invisible to Pydantic's JSON
        serialization -- found live: the API returned a response with
        no ``total_pages`` key at all, and the UI's f-string KeyError'd
        reading it. ``@computed_field`` is what actually includes a
        computed value in ``model_dump()``/the FastAPI response body."""
        if self.page_size <= 0:
            return 1
        return max(1, -(-self.total_count // self.page_size))  # ceil division


class UploadedDocumentResult(BaseModel):
    """One document created by an upload -- a .zip fans out into many of
    these, any other file into exactly one. Carries the Evidence
    Ingestion Pipeline's parse warnings alongside the created (Draft)
    record so the Preview step can show "extracted, but OCR wasn't
    available for page 3" instead of a silent gap."""

    document: DocumentationRecord
    warnings: list[ParseWarning] = Field(default_factory=list)
