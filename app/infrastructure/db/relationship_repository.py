"""Repository for the generic knowledge-graph edge table (Sprint 3,
Phase 3.3). See ``app/domain/knowledge_relationships.py``'s module
docstring for why this is one generic table rather than a hardcoded
table per object-type pair.
"""

from __future__ import annotations

import logging
import uuid
from typing import Protocol

from sqlalchemy import or_
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.knowledge_relationships import KnowledgeObjectType, KnowledgeRelationship, RelationshipType
from app.infrastructure.db.models import KnowledgeRelationshipModel

logger = logging.getLogger(__name__)


class RelationshipRepository(Protocol):
    def add(
        self,
        from_type: KnowledgeObjectType,
        from_id: str,
        to_type: KnowledgeObjectType,
        to_id: str,
        relationship_type: RelationshipType,
        created_by: str | None,
    ) -> KnowledgeRelationship:
        ...

    def remove(self, relationship_id: str) -> None:
        ...

    def get(self, relationship_id: str) -> KnowledgeRelationship | None:
        ...

    def list_for_object(self, object_type: KnowledgeObjectType, object_id: str) -> list[KnowledgeRelationship]:
        ...

    def list_all(
        self,
        *,
        from_type: KnowledgeObjectType | None = None,
        to_type: KnowledgeObjectType | None = None,
        relationship_type: RelationshipType | None = None,
    ) -> list[KnowledgeRelationship]:
        ...

    def find_pair(
        self,
        type_a: KnowledgeObjectType,
        id_a: str,
        type_b: KnowledgeObjectType,
        id_b: str,
        relationship_type: RelationshipType,
    ) -> KnowledgeRelationship | None:
        """Matches either direction -- (A,B) and (B,A) are the same
        undirected edge for duplicate-detection purposes."""
        ...

    def count(self) -> int:
        ...


class SqlAlchemyRelationshipRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def add(
        self,
        from_type: KnowledgeObjectType,
        from_id: str,
        to_type: KnowledgeObjectType,
        to_id: str,
        relationship_type: RelationshipType,
        created_by: str | None,
    ) -> KnowledgeRelationship:
        relationship = KnowledgeRelationship(
            id=str(uuid.uuid4()),
            from_type=from_type,
            from_id=from_id,
            to_type=to_type,
            to_id=to_id,
            relationship_type=relationship_type,
            created_by=created_by,
        )
        with self._session_factory() as session:
            model = KnowledgeRelationshipModel(
                id=relationship.id,
                from_type=from_type.value,
                from_id=from_id,
                to_type=to_type.value,
                to_id=to_id,
                relationship_type=relationship_type.value,
                created_at=relationship.created_at,
                created_by=created_by,
            )
            session.add(model)
            session.commit()
        logger.info(
            "Added relationship %s: %s:%s -[%s]-> %s:%s",
            relationship.id,
            from_type.value,
            from_id,
            relationship_type.value,
            to_type.value,
            to_id,
        )
        return relationship

    def remove(self, relationship_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(KnowledgeRelationshipModel, relationship_id)
            if model is not None:
                session.delete(model)
                session.commit()
                logger.info("Removed relationship %s", relationship_id)

    def get(self, relationship_id: str) -> KnowledgeRelationship | None:
        with self._session_factory() as session:
            model = session.get(KnowledgeRelationshipModel, relationship_id)
            return _to_domain(model) if model is not None else None

    def list_for_object(self, object_type: KnowledgeObjectType, object_id: str) -> list[KnowledgeRelationship]:
        with self._session_factory() as session:
            models = (
                session.query(KnowledgeRelationshipModel)
                .filter(
                    or_(
                        (KnowledgeRelationshipModel.from_type == object_type.value)
                        & (KnowledgeRelationshipModel.from_id == object_id),
                        (KnowledgeRelationshipModel.to_type == object_type.value)
                        & (KnowledgeRelationshipModel.to_id == object_id),
                    )
                )
                .order_by(KnowledgeRelationshipModel.created_at.desc())
                .all()
            )
            return [_to_domain(m) for m in models]

    def list_all(
        self,
        *,
        from_type: KnowledgeObjectType | None = None,
        to_type: KnowledgeObjectType | None = None,
        relationship_type: RelationshipType | None = None,
    ) -> list[KnowledgeRelationship]:
        with self._session_factory() as session:
            query = session.query(KnowledgeRelationshipModel)
            if from_type:
                query = query.filter(KnowledgeRelationshipModel.from_type == from_type.value)
            if to_type:
                query = query.filter(KnowledgeRelationshipModel.to_type == to_type.value)
            if relationship_type:
                query = query.filter(KnowledgeRelationshipModel.relationship_type == relationship_type.value)
            models = query.order_by(KnowledgeRelationshipModel.created_at.desc()).all()
            return [_to_domain(m) for m in models]

    def find_pair(
        self,
        type_a: KnowledgeObjectType,
        id_a: str,
        type_b: KnowledgeObjectType,
        id_b: str,
        relationship_type: RelationshipType,
    ) -> KnowledgeRelationship | None:
        with self._session_factory() as session:
            model = (
                session.query(KnowledgeRelationshipModel)
                .filter(
                    KnowledgeRelationshipModel.relationship_type == relationship_type.value,
                    or_(
                        (KnowledgeRelationshipModel.from_type == type_a.value)
                        & (KnowledgeRelationshipModel.from_id == id_a)
                        & (KnowledgeRelationshipModel.to_type == type_b.value)
                        & (KnowledgeRelationshipModel.to_id == id_b),
                        (KnowledgeRelationshipModel.from_type == type_b.value)
                        & (KnowledgeRelationshipModel.from_id == id_b)
                        & (KnowledgeRelationshipModel.to_type == type_a.value)
                        & (KnowledgeRelationshipModel.to_id == id_a),
                    ),
                )
                .first()
            )
            return _to_domain(model) if model is not None else None

    def count(self) -> int:
        with self._session_factory() as session:
            return session.query(KnowledgeRelationshipModel).count()


def _to_domain(model: KnowledgeRelationshipModel) -> KnowledgeRelationship:
    return KnowledgeRelationship(
        id=model.id,
        from_type=KnowledgeObjectType(model.from_type),
        from_id=model.from_id,
        to_type=KnowledgeObjectType(model.to_type),
        to_id=model.to_id,
        relationship_type=RelationshipType(model.relationship_type),
        created_at=model.created_at,
        created_by=model.created_by,
    )
