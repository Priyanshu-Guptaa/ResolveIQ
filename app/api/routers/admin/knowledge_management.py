"""Knowledge Management endpoints (Sprint 3, Phase 3.2) -- the first
Administration module. See ``app/api/routers/admin/__init__.py`` for why
there's no role check yet.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from app.api.dependencies import get_knowledge_management_engine
from app.api.schemas import UpdateDocumentMetadataRequest
from app.domain.evidence import DocumentationRecord
from app.domain.knowledge_management import DocumentPage, KnowledgeDashboardStats, UploadedDocumentResult
from app.engines.knowledge_management.engine import DocumentNotFoundError, KnowledgeManagementEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/knowledge", tags=["admin-knowledge-management"])


def _not_found(exc: DocumentNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


@router.get("/dashboard", response_model=KnowledgeDashboardStats)
def get_knowledge_dashboard(
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> KnowledgeDashboardStats:
    return engine.get_dashboard_stats()


@router.get("/documents", response_model=DocumentPage)
def list_documents(
    status: str | None = Query(default=None),
    product: str | None = Query(default=None),
    technology: str | None = Query(default=None),
    component: str | None = Query(default=None),
    search: str | None = Query(default=None),
    sort_by: str = Query(default="updated_at"),
    sort_desc: bool = Query(default=True),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> DocumentPage:
    """Backs the Knowledge Library. Never returns document content --
    see DocumentationListItem's docstring."""
    return engine.list_documents(
        status=status,
        product=product,
        technology=technology,
        component=component,
        search=search,
        sort_by=sort_by,
        sort_desc=sort_desc,
        page=page,
        page_size=page_size,
    )


@router.get("/documents/{document_id}", response_model=DocumentationRecord)
def get_document(
    document_id: str,
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> DocumentationRecord:
    """Full record, content included -- only called for one document at
    a time (Document Details / Preview), never for the Library list."""
    document = engine.get_document(document_id)
    if document is None:
        raise _not_found(DocumentNotFoundError(document_id))
    return document


@router.post("/documents/upload", response_model=list[UploadedDocumentResult], status_code=201)
async def upload_documents(
    files: list[UploadFile],
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> list[UploadedDocumentResult]:
    """Upload + Extract steps. Accepts PDF/DOCX/PPTX/XLSX/TXT/LOG or a
    ZIP of any of those -- the same Evidence Ingestion Pipeline that
    parses investigation evidence, not a second parser. A ZIP fans out
    into multiple created documents, which is why the response is a
    flat list rather than one-to-one with the upload. Every created
    document starts as Draft -- nothing here is searchable until
    Publish."""
    results: list[UploadedDocumentResult] = []
    for file in files:
        raw_bytes = await file.read()
        results.extend(engine.upload_document(file.filename or "upload", raw_bytes))
    return results


@router.patch("/documents/{document_id}", response_model=DocumentationRecord)
def update_document_metadata(
    document_id: str,
    request: UpdateDocumentMetadataRequest,
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> DocumentationRecord:
    """The Metadata step -- also usable to edit an already-published
    document (re-indexes it automatically if so)."""
    try:
        return engine.update_metadata(document_id, **request.model_dump())
    except DocumentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/documents/{document_id}/validate", response_model=list[str])
def validate_document(
    document_id: str,
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> list[str]:
    """The Validate step -- returns human-readable warnings; publishing
    past them is allowed (this is not a hard gate, see the engine's
    docstring)."""
    try:
        return engine.validate_document(document_id)
    except DocumentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/documents/{document_id}/publish", response_model=DocumentationRecord)
def publish_document(
    document_id: str,
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> DocumentationRecord:
    try:
        return engine.publish_document(document_id)
    except DocumentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/documents/{document_id}/archive", response_model=DocumentationRecord)
def archive_document(
    document_id: str,
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> DocumentationRecord:
    try:
        return engine.archive_document(document_id)
    except DocumentNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/documents/{document_id}/restore", response_model=DocumentationRecord)
def restore_document(
    document_id: str,
    engine: KnowledgeManagementEngine = Depends(get_knowledge_management_engine),
) -> DocumentationRecord:
    """Archived (or Published) -> Draft."""
    try:
        return engine.restore_to_draft(document_id)
    except DocumentNotFoundError as exc:
        raise _not_found(exc) from exc
