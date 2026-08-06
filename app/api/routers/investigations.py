"""Investigation lifecycle endpoints: start an investigation, add evidence
(notes/logs), fetch state, and get recommendations.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.dependencies import get_investigation_engine, get_recommendation_engine
from app.api.schemas import AddNoteRequest, CreateInvestigationRequest, InvestigationSummary
from app.domain.evidence import Evidence
from app.domain.investigation import InvestigationSession
from app.domain.recommendation import Recommendation
from app.engines.investigation.engine import InvestigationEngine, InvestigationNotFoundError
from app.engines.recommendation.engine import RecommendationEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigations", tags=["investigations"])


def _not_found(exc: InvestigationNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


@router.post("", response_model=InvestigationSession, status_code=201)
def create_investigation(
    request: CreateInvestigationRequest,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> InvestigationSession:
    return engine.start_investigation(request.title, request.description)


@router.get("", response_model=list[InvestigationSummary])
def list_investigations(
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> list[InvestigationSummary]:
    investigations = engine.list_investigations()
    return [_to_summary(inv) for inv in investigations]


@router.get("/{investigation_id}", response_model=InvestigationSession)
def get_investigation(
    investigation_id: str,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> InvestigationSession:
    """Fetching an investigation counts as "viewing" it -- backs the
    Dashboard's Recently Viewed panel. Called once when the Workspace page
    loads, not on every poll, so it reflects real engineer attention."""
    try:
        investigation = engine.get_investigation(investigation_id)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc
    engine.mark_viewed(investigation_id)
    return investigation


def _to_summary(inv: InvestigationSession) -> InvestigationSummary:
    return InvestigationSummary(
        id=inv.id,
        title=inv.title,
        status=inv.status.value,
        created_at=inv.created_at.isoformat(),
        updated_at=inv.updated_at.isoformat(),
        last_viewed_at=inv.last_viewed_at.isoformat() if inv.last_viewed_at else None,
        evidence_count=len(inv.evidence),
    )


@router.post("/{investigation_id}/evidence/notes", response_model=Evidence, status_code=201)
def add_note(
    investigation_id: str,
    request: AddNoteRequest,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> Evidence:
    try:
        return engine.add_text_evidence(investigation_id, request.text, title="Manual note")
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/{investigation_id}/evidence/logs", response_model=list[Evidence], status_code=201)
async def upload_logs(
    investigation_id: str,
    files: list[UploadFile],
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> list[Evidence]:
    results: list[Evidence] = []
    try:
        for file in files:
            raw_bytes = await file.read()
            content = raw_bytes.decode("utf-8", errors="replace")
            evidence = engine.add_log_evidence(investigation_id, file.filename or "upload.log", content)
            results.append(evidence)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc
    return results


@router.get("/{investigation_id}/recommendations", response_model=Recommendation)
def get_recommendations(
    investigation_id: str,
    investigation_engine: InvestigationEngine = Depends(get_investigation_engine),
    recommendation_engine: RecommendationEngine = Depends(get_recommendation_engine),
) -> Recommendation:
    try:
        investigation = investigation_engine.get_investigation(investigation_id)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc
    return recommendation_engine.generate(investigation)
