"""Repository for the shared History table (Sprint 3, Phase 3.4)."""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from sqlalchemy import func
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.entity_version import EntityVersion
from app.domain.knowledge_relationships import KnowledgeObjectType
from app.infrastructure.db.models import EntityVersionModel

logger = logging.getLogger(__name__)


class VersionRepository(Protocol):
    def record_version(
        self,
        object_type: KnowledgeObjectType,
        object_id: str,
        snapshot: dict,
        *,
        changed_by: str | None = None,
        change_summary: str = "",
    ) -> EntityVersion:
        ...

    def list_versions(self, object_type: KnowledgeObjectType, object_id: str) -> list[EntityVersion]:
        ...


class SqlAlchemyVersionRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def record_version(
        self,
        object_type: KnowledgeObjectType,
        object_id: str,
        snapshot: dict,
        *,
        changed_by: str | None = None,
        change_summary: str = "",
    ) -> EntityVersion:
        with self._session_factory() as session:
            current_max = (
                session.query(func.max(EntityVersionModel.version_number))
                .filter(
                    EntityVersionModel.object_type == object_type.value,
                    EntityVersionModel.object_id == object_id,
                )
                .scalar()
            )
            version_number = (current_max or 0) + 1
            model = EntityVersionModel(
                id=str(uuid.uuid4()),
                object_type=object_type.value,
                object_id=object_id,
                version_number=version_number,
                snapshot=snapshot,
                changed_by=changed_by,
                change_summary=change_summary,
            )
            session.add(model)
            session.commit()
            logger.debug("Recorded version %d for %s:%s (%s)", version_number, object_type.value, object_id, change_summary)
            return _to_domain(model)

    def list_versions(self, object_type: KnowledgeObjectType, object_id: str) -> list[EntityVersion]:
        with self._session_factory() as session:
            models = (
                session.query(EntityVersionModel)
                .filter(
                    EntityVersionModel.object_type == object_type.value,
                    EntityVersionModel.object_id == object_id,
                )
                .order_by(EntityVersionModel.version_number.desc())
                .all()
            )
            return [_to_domain(m) for m in models]


def _to_domain(model: EntityVersionModel) -> EntityVersion:
    return EntityVersion(
        id=model.id,
        object_type=KnowledgeObjectType(model.object_type),
        object_id=model.object_id,
        version_number=model.version_number,
        snapshot=model.snapshot,
        changed_by=model.changed_by,
        changed_at=model.changed_at,
        change_summary=model.change_summary,
    )
