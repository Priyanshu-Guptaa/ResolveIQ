"""Investigation lifecycle endpoints: start an investigation, add evidence
(notes/logs), fetch state, and get recommendations.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile

from app.api.dependencies import get_investigation_engine, get_recommendation_engine
from app.api.schemas import (
    AddNoteRequest,
    CreateInvestigationRequest,
    InvestigationSummary,
    UpdateInvestigationDetailsRequest,
)
from app.domain.evidence import Evidence, EvidencePreview
from app.domain.investigation import (
    ActivityItem,
    EvidenceSummary,
    InvestigationDetailSummary,
    InvestigationListItem,
    InvestigationSession,
)
from app.domain.recommendation import Recommendation
from app.engines.investigation.engine import (
    EvidenceNotFoundError,
    InvestigationEngine,
    InvestigationNotFoundError,
)
from app.engines.recommendation.engine import RecommendationEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/investigations", tags=["investigations"])


def _not_found(exc: InvestigationNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


def _evidence_not_found(exc: EvidenceNotFoundError) -> HTTPException:
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
    """Lightweight list -- deliberately does not hydrate evidence content
    (see InvestigationListItem). Every current caller of this endpoint
    (Dashboard, investigation pickers) only needs id/title/status/count."""
    return [_to_summary(item) for item in engine.list_investigation_summaries()]


@router.get("/{investigation_id}", response_model=InvestigationDetailSummary)
def get_investigation(
    investigation_id: str,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> InvestigationDetailSummary:
    """Fetching an investigation counts as "viewing" it -- backs the
    Dashboard's Recently Viewed panel. Called once when the Workspace page
    loads, not on every poll, so it reflects real engineer attention.

    Lightweight by design (Investigation loading redesign): no evidence
    content is embedded here -- a real investigation with 97 evidence
    items (one legitimately 15.7MB) made the old full-hydration response
    151MB of JSON. See GET .../evidence, .../evidence/{evidence_id}, and
    .../evidence/{evidence_id}/preview for evidence content, loaded only
    when actually requested."""
    try:
        summary = engine.get_investigation_summary(investigation_id)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc
    engine.mark_viewed(investigation_id)
    return summary


@router.get("/{investigation_id}/evidence", response_model=list[EvidenceSummary])
def list_evidence(
    investigation_id: str,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> list[EvidenceSummary]:
    """Metadata-only evidence list -- backs the Explorer. No raw_content,
    extracted_entities, or log_events; sizes/counts only."""
    try:
        return engine.list_evidence(investigation_id)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{investigation_id}/evidence/{evidence_id}", response_model=Evidence)
def get_evidence(
    investigation_id: str,
    evidence_id: str,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> Evidence:
    """Full content for exactly one piece of evidence -- fetched only
    when actually requested, never automatically for every item on an
    investigation."""
    try:
        return engine.get_evidence(investigation_id, evidence_id)
    except EvidenceNotFoundError as exc:
        raise _evidence_not_found(exc) from exc


@router.get("/{investigation_id}/evidence/{evidence_id}/preview", response_model=EvidencePreview)
def get_evidence_preview(
    investigation_id: str,
    evidence_id: str,
    max_chars: int = Query(default=2000, ge=100, le=20000),
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> EvidencePreview:
    """Capped preview -- what the Overview/Notes tabs use to show "what
    does this look like" without transferring a potentially multi-
    megabyte file in full."""
    try:
        return engine.get_evidence_preview(investigation_id, evidence_id, max_chars)
    except EvidenceNotFoundError as exc:
        raise _evidence_not_found(exc) from exc


def _to_summary(inv: InvestigationListItem) -> InvestigationSummary:
    return InvestigationSummary(
        id=inv.id,
        title=inv.title,
        status=inv.status.value,
        created_at=inv.created_at.isoformat(),
        updated_at=inv.updated_at.isoformat(),
        last_viewed_at=inv.last_viewed_at.isoformat() if inv.last_viewed_at else None,
        evidence_count=inv.evidence_count,
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
    """Accepts any file the Evidence Ingestion Pipeline recognizes (log,
    txt, csv, json, xml, docx, xlsx, pdf, jpg, png, evtx) or a zip of any
    of those -- the pipeline detects the type and runs the matching
    parser before this evidence is ever looked at for entities. A zip can
    expand into multiple Evidence items per uploaded file, which is why
    the response is a flat list rather than one-to-one with the upload."""
    results: list[Evidence] = []
    try:
        for file in files:
            raw_bytes = await file.read()
            evidence_items = engine.add_file_evidence(investigation_id, file.filename or "upload", raw_bytes)
            results.extend(evidence_items)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc
    return results


@router.patch("/{investigation_id}/details", response_model=InvestigationDetailSummary)
def update_investigation_details(
    investigation_id: str,
    request: UpdateInvestigationDetailsRequest,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> InvestigationDetailSummary:
    """Saves the persistent Summary Card's engineer-entered fields
    (customer, product, version, technology, assigned engineer)."""
    try:
        return engine.update_details(investigation_id, **request.model_dump())
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc


@router.get("/{investigation_id}/timeline", response_model=list[ActivityItem])
def get_investigation_timeline(
    investigation_id: str,
    engine: InvestigationEngine = Depends(get_investigation_engine),
) -> list[ActivityItem]:
    """Real chronological events for one investigation -- backs the
    Workspace's Timeline nav item (Phase 2A)."""
    try:
        return engine.get_timeline(investigation_id)
    except InvestigationNotFoundError as exc:
        raise _not_found(exc) from exc


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
