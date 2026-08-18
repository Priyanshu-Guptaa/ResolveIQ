"""Repository for Conversation State (2026-08-14, Phase 3) --
``chat_sessions``/``chat_messages``, same repository pattern as
``SqlAlchemyInvestigationRepository`` (``app/infrastructure/db/repository.py``):
Protocol boundary so ``ConversationStateEngine`` depends on it, not on
SQLAlchemy directly, and a real-temp-SQLite integration-test fixture is
the natural way to exercise it (same as every other repository in this
codebase).
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy import func
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.chat import ChatMessage, ChatSession, ConversationSlots, MessageRole, ReferenceResolution
from app.domain.query_understanding import ParsedQuery
from app.infrastructure.db.models import ChatMessageModel, ChatSessionModel

logger = logging.getLogger(__name__)


class ChatRepository(Protocol):
    def save_session(self, session: ChatSession) -> None:
        ...

    def get_session(self, session_id: str) -> ChatSession | None:
        ...

    def save_message(self, message: ChatMessage) -> None:
        ...

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        ...

    def next_sequence(self, session_id: str) -> int:
        ...


class SqlAlchemyChatRepository:
    """SQLite-backed (via SQLAlchemy) implementation."""

    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def save_session(self, session: ChatSession) -> None:
        """Upsert by id -- calling this twice with the same
        ``ChatSession`` (e.g. after a merge that produced identical
        state) is a safe no-op-equivalent, never a duplicate row. Same
        idempotency discipline as every other ``save`` in this
        codebase (``SqlAlchemyInvestigationRepository.save``,
        ``SqlAlchemyLookupRepository.save_customer``, ...)."""
        with self._session_factory() as orm_session:
            model = orm_session.get(ChatSessionModel, session.id)
            if model is None:
                model = ChatSessionModel(id=session.id)
                orm_session.add(model)
            model.investigation_id = session.investigation_id
            model.slots = session.slots.model_dump(mode="json")
            model.created_at = session.created_at
            model.updated_at = session.updated_at
            orm_session.commit()
            logger.debug("Saved chat session %s", session.id)

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._session_factory() as orm_session:
            model = orm_session.get(ChatSessionModel, session_id)
            if model is None:
                return None
            return _session_to_domain(model)

    def save_message(self, message: ChatMessage) -> None:
        """Upsert by id -- see ``save_session``'s docstring for the
        same idempotency discipline. The (session_id, sequence) unique
        constraint on ``ChatMessageModel`` still guards against two
        genuinely different messages colliding on the same ordinal."""
        with self._session_factory() as orm_session:
            model = orm_session.get(ChatMessageModel, message.id)
            if model is None:
                model = ChatMessageModel(id=message.id)
                orm_session.add(model)
            model.session_id = message.session_id
            model.sequence = message.sequence
            model.role = message.role.value
            model.content = message.content
            model.parsed_query = message.parsed_query.model_dump(mode="json") if message.parsed_query else None
            model.reference_resolution = (
                message.reference_resolution.model_dump(mode="json") if message.reference_resolution else None
            )
            model.created_at = message.created_at
            orm_session.commit()
            logger.debug("Saved chat message %s (session %s, seq %d)", message.id, message.session_id, message.sequence)

    def list_messages(self, session_id: str) -> list[ChatMessage]:
        with self._session_factory() as orm_session:
            models = (
                orm_session.query(ChatMessageModel)
                .filter(ChatMessageModel.session_id == session_id)
                .order_by(ChatMessageModel.sequence)
                .all()
            )
            return [_message_to_domain(m) for m in models]

    def next_sequence(self, session_id: str) -> int:
        with self._session_factory() as orm_session:
            current_max = (
                orm_session.query(func.max(ChatMessageModel.sequence))
                .filter(ChatMessageModel.session_id == session_id)
                .scalar()
            )
            return (current_max or 0) + 1


def _session_to_domain(model: ChatSessionModel) -> ChatSession:
    return ChatSession(
        id=model.id,
        investigation_id=model.investigation_id,
        slots=ConversationSlots(**(model.slots or {})),
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def _message_to_domain(model: ChatMessageModel) -> ChatMessage:
    return ChatMessage(
        id=model.id,
        session_id=model.session_id,
        sequence=model.sequence,
        role=MessageRole(model.role),
        content=model.content,
        parsed_query=ParsedQuery(**model.parsed_query) if model.parsed_query else None,
        reference_resolution=ReferenceResolution(**model.reference_resolution) if model.reference_resolution else None,
        created_at=model.created_at,
    )
