"""Metadata Classification admin endpoints (Context Dimensions phase,
2026-08-12).

Runs the confidence-tiered classification pass (see
``app.engines.knowledge.classification``) over existing Documentation
and exposes the Medium-confidence review queue. High-confidence results
apply automatically as part of ``run()``; Low-confidence results are
recorded but never surfaced here for action (see that module's
docstring for why).

See ``app/api/routers/admin/__init__.py`` for why there's no role check
yet.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_document_classification_engine
from app.domain.classification import ClassificationRunSummary, MetadataClassificationSuggestion, PendingSuggestionView
from app.engines.knowledge.classification import DocumentClassificationEngine

router = APIRouter(prefix="/admin/classification", tags=["admin-classification"])


class RunRequest(BaseModel):
    actor: str = "admin"
    limit: int | None = None


class ReviewRequest(BaseModel):
    actor: str = "admin"


@router.post("/run", response_model=ClassificationRunSummary)
def run_classification(
    request: RunRequest,
    engine: DocumentClassificationEngine = Depends(get_document_classification_engine),
) -> ClassificationRunSummary:
    return engine.run(actor=request.actor, limit=request.limit)


@router.get("/pending", response_model=list[PendingSuggestionView])
def list_pending_suggestions(
    engine: DocumentClassificationEngine = Depends(get_document_classification_engine),
) -> list[PendingSuggestionView]:
    """Medium-confidence suggestions awaiting human review, enriched
    with each one's real object title and a re-derived real mention
    count (2026-08-14, Phase 5 UI improvement -- see
    ``DocumentClassificationEngine.list_pending_with_context``'s own
    docstring: read-only, presentation-only, never a new matching rule)."""
    return engine.list_pending_with_context()


@router.post("/{suggestion_id}/accept", response_model=MetadataClassificationSuggestion)
def accept_suggestion(
    suggestion_id: str,
    request: ReviewRequest,
    engine: DocumentClassificationEngine = Depends(get_document_classification_engine),
) -> MetadataClassificationSuggestion:
    try:
        return engine.accept(suggestion_id, actor=request.actor)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{suggestion_id}/reject", response_model=MetadataClassificationSuggestion)
def reject_suggestion(
    suggestion_id: str,
    request: ReviewRequest,
    engine: DocumentClassificationEngine = Depends(get_document_classification_engine),
) -> MetadataClassificationSuggestion:
    try:
        return engine.reject(suggestion_id, actor=request.actor)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
