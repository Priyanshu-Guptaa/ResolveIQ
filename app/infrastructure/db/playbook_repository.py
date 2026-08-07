"""Repository for the governed ``playbooks`` table (Sprint 3, Phase
3.3). Same Protocol-plus-SQLAlchemy shape as every other repository in
this codebase.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.playbook import Playbook
from app.infrastructure.db.models import PlaybookModel

logger = logging.getLogger(__name__)


class PlaybookRepository(Protocol):
    def save(self, playbook: Playbook) -> None:
        ...

    def get(self, playbook_id: str) -> Playbook | None:
        ...

    def list_all(self, *, active_only: bool = True) -> list[Playbook]:
        ...

    def count(self) -> int:
        ...


class SqlAlchemyPlaybookRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def save(self, playbook: Playbook) -> None:
        with self._session_factory() as session:
            model = session.get(PlaybookModel, playbook.id)
            if model is None:
                model = PlaybookModel(id=playbook.id)
                session.add(model)
            model.title = playbook.title
            model.product = playbook.product
            model.description = playbook.description
            model.steps = playbook.steps
            model.created_at = playbook.created_at
            model.updated_at = playbook.updated_at
            model.created_by = playbook.created_by
            model.updated_by = playbook.updated_by
            model.is_active = playbook.is_active
            session.commit()
            logger.debug("Saved playbook %s", playbook.id)

    def get(self, playbook_id: str) -> Playbook | None:
        with self._session_factory() as session:
            model = session.get(PlaybookModel, playbook_id)
            return _to_domain(model) if model is not None else None

    def list_all(self, *, active_only: bool = True) -> list[Playbook]:
        with self._session_factory() as session:
            query = session.query(PlaybookModel)
            if active_only:
                query = query.filter(PlaybookModel.is_active.is_(True))
            return [_to_domain(m) for m in query.order_by(PlaybookModel.title).all()]

    def count(self) -> int:
        with self._session_factory() as session:
            return session.query(PlaybookModel).count()


def _to_domain(model: PlaybookModel) -> Playbook:
    return Playbook(
        id=model.id,
        title=model.title,
        product=model.product,
        description=model.description,
        steps=model.steps,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
    )
