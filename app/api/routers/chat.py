"""Chat endpoints (2026-08-14, Phase 4) -- thin router over
``ChatOrchestrator``, same discipline as every other router in this
app: no business logic here, only request/response wiring and
converting engine-layer exceptions into the right HTTP status.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_chat_orchestrator
from app.api.schemas import ChatMessageRequest, CreateChatSessionRequest
from app.domain.chat import ChatMessage, ChatResponse, ChatSession
from app.engines.chat.orchestrator import ChatOrchestrator, ChatSessionNotFoundError, EmptyMessageError
from app.engines.investigation.engine import InvestigationNotFoundError

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
