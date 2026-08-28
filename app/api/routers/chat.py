"""Chat endpoints (2026-08-14, Phase 4) -- thin router over
``ChatOrchestrator``, same discipline as every other router in this
app: no business logic here, only request/response wiring and
converting engine-layer exceptions into the right HTTP status.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import (
    get_chat_enhancement_service,
    get_chat_log_upload_service,
    get_chat_orchestrator,
    get_investigation_engine,
)
from app.api.schemas import ChatEnhancementResponse, ChatLogUploadResponse, ChatMessageRequest, CreateChatSessionRequest
from app.domain.chat import ChatMessage, ChatResponse, ChatSession
from app.engines.chat.enhancement import ChatEnhancementService
from app.engines.chat.log_upload import ChatLogUploadError, ChatLogUploadService, validate_log_upload
from app.engines.chat.orchestrator import ChatOrchestrator, ChatSessionNotFoundError, EmptyMessageError
from app.engines.investigation.engine import InvestigationEngine, InvestigationNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/sessions", response_model=ChatSession, status_code=201)
def create_chat_session(
    request: CreateChatSessionRequest,
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> ChatSession:
    try:
        return orchestrator.create_session(request.investigation_id)
    except InvestigationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No such investigation: {exc}") from exc


@router.get("/sessions/{session_id}", response_model=ChatSession)
def get_chat_session(
    session_id: str,
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> ChatSession:
    session = orchestrator.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No such chat session: {session_id}")
    return session


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessage])
def list_chat_messages(
    session_id: str,
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> list[ChatMessage]:
    session = orchestrator.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No such chat session: {session_id}")
    return orchestrator.list_messages(session_id)


@router.post("/sessions/{session_id}/messages", response_model=ChatResponse)
def post_chat_message(
    session_id: str,
    request: ChatMessageRequest,
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
) -> ChatResponse:
    try:
        return orchestrator.handle_message(session_id, request.message)
    except ChatSessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No such chat session: {exc}") from exc
    except EmptyMessageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- never leak an internal stack trace to the client
        logger.exception("Chat message handling failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Internal error while processing the chat message.") from exc


@router.post("/sessions/{session_id}/logs", response_model=ChatLogUploadResponse, status_code=201)
async def upload_chat_log(
    session_id: str,
    file: UploadFile = File(...),
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator),
    investigation_engine: InvestigationEngine = Depends(get_investigation_engine),
    log_upload_service: ChatLogUploadService = Depends(get_chat_log_upload_service),
) -> ChatLogUploadResponse:
    """Chat Assistant Phase 39 -- USER UPLOADS LOG THROUGH CHAT. The
    uploaded content is DATA, never an instruction: it is handed
    unmodified to the existing Evidence Ingestion Pipeline and the
    existing ``LogIntelligenceEngine`` -- never executed, never
    interpolated into a prompt or filesystem path, never persisted to
    disk. Two paths, both reusing existing, unmodified engines:

    * Investigation-scoped session -> ``InvestigationEngine.
      add_file_evidence`` (the exact same call the Investigation
      Workspace's own upload button already makes), so the log becomes
      real, persisted evidence on that investigation.
    * Standalone session -> ``ChatLogUploadService`` (Phase 39, new --
      see ``app/engines/chat/log_upload.py``), an in-memory, per-session
      registry, since a standalone chat session has no real, persisted
      investigation to attach evidence to.

    Either way, ``ChatOrchestrator._resolve_investigation_for_
    retrieval`` picks the resulting evidence up automatically on the
    next chat turn -- no other orchestration change was needed.
    """
    session = orchestrator.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"No such chat session: {session_id}")

    content = await file.read()
    filename = file.filename or "upload"
    try:
        validate_log_upload(filename, content)
    except ChatLogUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        if session.investigation_id is not None:
            evidence = investigation_engine.add_file_evidence(session.investigation_id, filename, content)[0]
        else:
            evidence = log_upload_service.upload(session_id, filename, content)
    except InvestigationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"No such investigation: {exc}") from exc
    except ChatLogUploadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- never leak an internal stack trace to the client
        logger.exception("Chat log upload failed for session %s", session_id)
        raise HTTPException(status_code=500, detail="Internal error while processing the uploaded log.") from exc

    warnings = [w.get("message", "") for w in evidence.metadata.get("warnings", []) if isinstance(w, dict)]
    return ChatLogUploadResponse(
        title=evidence.title,
        event_count=len(evidence.log_events),
        entity_count=len(evidence.extracted_entities),
        warnings=[w for w in warnings if w],
    )


@router.get("/enhancements/{job_id}", response_model=ChatEnhancementResponse)
def get_chat_enhancement(
    job_id: str,
    service: ChatEnhancementService = Depends(get_chat_enhancement_service),
) -> ChatEnhancementResponse:
    """Chat Assistant Phase 37 -- polls one asynchronous LLM enhancement
    job (see ``ChatResponse.enhancement``). 404 for an unknown OR
    already-expired job id (see ``ChatEnhancementService``'s TTL-based
    purge) -- indistinguishable from the caller's point of view, since
    neither case has anything left to report; the deterministic answer
    already delivered in the original ``ChatResponse`` remains valid
    either way."""
    job = service.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such enhancement job: {job_id}")
    return ChatEnhancementResponse(job_id=job.id, status=job.status, answer_text=job.answer_text, error=job.error)
