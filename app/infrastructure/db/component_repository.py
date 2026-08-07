"""Repository for the governed ``component_profiles`` table (Sprint 3,
Phase 3.1).

Same Protocol-plus-SQLAlchemy-implementation shape as
:class:`~app.infrastructure.db.repository.InvestigationRepository` -- the
Product Intelligence Engine depends on this Protocol, not on SQLAlchemy,
via :meth:`~app.engines.product_intelligence.component_registry.ComponentRegistry.load_from_repository`.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.product_intelligence import ComponentProfile
from app.infrastructure.db.models import ComponentProfileModel

logger = logging.getLogger(__name__)

_LIST_FIELDS = (
    "responsibilities",
    "related_components",
    "dependencies",
    "consumes",
    "produces",
    "related_services",
    "related_queues",
    "database_tables",
    "configuration",
    "events",
    "message_flows",
    "data_flows",
    "typical_failures",
    "required_logs",
    "common_sql",
    "known_bugs",
    "documentation_links",
    "playbooks",
    "historical_investigations",
)


class ComponentProfileRepository(Protocol):
    """Persistence boundary for the Component Registry / Domain
    Intelligence knowledge graph."""

    def save(self, profile: ComponentProfile) -> None:
        ...

    def get(self, component_id: str) -> ComponentProfile | None:
        ...

    def get_by_name(self, name: str) -> ComponentProfile | None:
        ...

    def list_all(self, *, active_only: bool = True) -> list[ComponentProfile]:
        ...

    def count(self) -> int:
        ...


class SqlAlchemyComponentProfileRepository:
    """SQLite-backed implementation."""

    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def save(self, profile: ComponentProfile) -> None:
        with self._session_factory() as session:
            model = session.get(ComponentProfileModel, profile.id)
            if model is None:
                model = ComponentProfileModel(id=profile.id)
                session.add(model)
            model.name = profile.name
            model.product = profile.product
            for field in _LIST_FIELDS:
                setattr(model, field, getattr(profile, field))
            model.version_differences = [vd.model_dump(mode="json") for vd in profile.version_differences]
            model.created_at = profile.created_at
            model.updated_at = profile.updated_at
            model.created_by = profile.created_by
            model.updated_by = profile.updated_by
            model.is_active = profile.is_active
            session.commit()
            logger.debug("Saved component profile %s (%s)", profile.id, profile.name)

    def get(self, component_id: str) -> ComponentProfile | None:
        with self._session_factory() as session:
            model = session.get(ComponentProfileModel, component_id)
            return _to_domain(model) if model is not None else None

    def get_by_name(self, name: str) -> ComponentProfile | None:
        with self._session_factory() as session:
            model = session.query(ComponentProfileModel).filter(ComponentProfileModel.name == name).first()
            return _to_domain(model) if model is not None else None

    def list_all(self, *, active_only: bool = True) -> list[ComponentProfile]:
        with self._session_factory() as session:
            query = session.query(ComponentProfileModel)
            if active_only:
                query = query.filter(ComponentProfileModel.is_active.is_(True))
            return [_to_domain(model) for model in query.order_by(ComponentProfileModel.name).all()]

    def count(self) -> int:
        with self._session_factory() as session:
            return session.query(ComponentProfileModel).count()


def _to_domain(model: ComponentProfileModel) -> ComponentProfile:
    kwargs = {field: getattr(model, field) for field in _LIST_FIELDS}
    return ComponentProfile(
        id=model.id,
        name=model.name,
        product=model.product,
        version_differences=model.version_differences,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
        **kwargs,
    )
