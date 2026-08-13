"""Human Verification for Resolution Provenance (2026-08-13, Phase 0 --
Chat/Structured Resolution Knowledge architecture, closing the gap
flagged in RESOLVEIQ_CHAT_AND_RESOLUTION_ARCHITECTURE.md Section 20).

Deliberately its own small, dedicated endpoint pair per governed type --
NOT the generic ``PATCH /admin/objects/{type}/{id}`` -- because the four
``resolution_verified*`` fields must always be set together, atomically,
with a real actor and timestamp (the audit-friendly structure the
approved Resolution Provenance design required): a free-form generic
edit could set ``resolution_verified=True`` with no ``resolution_verified_by``
at all, or edit the note without the flag, which is exactly the
unrestricted, ungoverned control the approved design explicitly said not
to expose yet. ``app/api/routers/admin/knowledge_objects.py``'s generic
PATCH route rejects these four field names outright and points here.

Reuses ``KnowledgeObjectService.edit_metadata`` unchanged underneath --
same repository, same indexing/versioning side effects as any other
edit; this router only adds the "these four fields move together, with
a real actor" discipline on top.

See ``app/api/routers/admin/__init__.py`` for why there's no role check
yet -- flagged as a real, unresolved open question in the design
document (any authenticated admin caller can verify a resolution today).
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.dependencies import get_knowledge_object_service
from app.domain.evidence import HistoricalInvestigationRecord, KnownBugRecord
from app.domain.knowledge_relationships import KnowledgeObjectType
from app.engines.knowledge_object_framework.service import KnowledgeObjectNotFoundError, KnowledgeObjectService

router = APIRouter(prefix="/admin", tags=["admin-resolution-verification"])


class VerifyResolutionRequest(BaseModel):
    actor: str = "admin"
    note: str | None = None
    """Free-text reason/evidence for the verification, e.g. "Confirmed
    with the customer after applying the fix; issue did not recur." --
    stored verbatim as ``resolution_verification_note``, never required
    but strongly encouraged (the UI prompts for it)."""


class UnverifyRequest(BaseModel):
    actor: str = "admin"


def _not_found(exc: KnowledgeObjectNotFoundError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(exc))


@router.post("/historical-investigations/{investigation_id}/verify", response_model=HistoricalInvestigationRecord)
def verify_historical_investigation(
    investigation_id: str,
    request: VerifyResolutionRequest,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> HistoricalInvestigationRecord:
    try:
        return service.edit_metadata(
            KnowledgeObjectType.HISTORICAL_INVESTIGATION,
            investigation_id,
            updated_by=request.actor,
            resolution_verified=True,
            resolution_verified_by=request.actor,
            resolution_verified_at=datetime.now(timezone.utc),
            resolution_verification_note=request.note,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/historical-investigations/{investigation_id}/unverify", response_model=HistoricalInvestigationRecord)
def unverify_historical_investigation(
    investigation_id: str,
    request: UnverifyRequest,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> HistoricalInvestigationRecord:
    """Reverses a verification made in error -- same "never a dead end"
    discipline as classification's accept/reject pair. Clears all four
    fields together, never leaves a stale ``resolution_verified_by``
    behind on an un-verified record."""
    try:
        return service.edit_metadata(
            KnowledgeObjectType.HISTORICAL_INVESTIGATION,
            investigation_id,
            updated_by=request.actor,
            resolution_verified=False,
            resolution_verified_by=None,
            resolution_verified_at=None,
            resolution_verification_note=None,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/known-bugs/{bug_id}/verify", response_model=KnownBugRecord)
def verify_known_bug(
    bug_id: str,
    request: VerifyResolutionRequest,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> KnownBugRecord:
    try:
        return service.edit_metadata(
            KnowledgeObjectType.KNOWN_BUG,
            bug_id,
            updated_by=request.actor,
            resolution_verified=True,
            resolution_verified_by=request.actor,
            resolution_verified_at=datetime.now(timezone.utc),
            resolution_verification_note=request.note,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc


@router.post("/known-bugs/{bug_id}/unverify", response_model=KnownBugRecord)
def unverify_known_bug(
    bug_id: str,
    request: UnverifyRequest,
    service: KnowledgeObjectService = Depends(get_knowledge_object_service),
) -> KnownBugRecord:
    try:
        return service.edit_metadata(
            KnowledgeObjectType.KNOWN_BUG,
            bug_id,
            updated_by=request.actor,
            resolution_verified=False,
            resolution_verified_by=None,
            resolution_verified_at=None,
            resolution_verification_note=None,
        )
    except KnowledgeObjectNotFoundError as exc:
        raise _not_found(exc) from exc
