"""Repository for the governed ``sql_templates`` table (Sprint 3, Phase
3.1) -- the SQL Library, previously the ``QUERY_LIBRARY`` Python constant
in ``app/domain/sql_studio.py``.

Same association-table-join discipline as ``knowledge_repository.py``:
listing N templates is always exactly two queries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.sql_studio import QueryTemplate
from app.infrastructure.db.models import ComponentProfileModel, SqlTemplateModel, sql_template_components

logger = logging.getLogger(__name__)


class SqlTemplateRepository(Protocol):
    def save(self, template: QueryTemplate) -> None:
        ...

    def list_all(self, *, active_only: bool = True) -> list[QueryTemplate]:
        ...

    def get(self, template_id: str) -> QueryTemplate | None:
        ...

    def count(self) -> int:
        ...

    def delete(self, template_id: str) -> None:
        ...


class SqlAlchemySqlTemplateRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def save(self, template: QueryTemplate) -> None:
        with self._session_factory() as session:
            model = session.get(SqlTemplateModel, template.id)
            if model is None:
                model = SqlTemplateModel(id=template.id)
                session.add(model)
            model.title = template.title
            model.category = template.category
            model.sql_text = template.sql_text
            model.explanation = template.explanation
            model.tags = template.tags
            model.created_at = template.created_at
            model.updated_at = template.updated_at
            model.created_by = template.created_by
            model.updated_by = template.updated_by
            model.is_active = template.is_active
            model.status = template.status.value
            session.commit()

            session.execute(
                sql_template_components.delete().where(sql_template_components.c.sql_template_id == template.id)
            )
            if template.related_components:
                name_to_id = dict(
                    session.query(ComponentProfileModel.name, ComponentProfileModel.id)
                    .filter(ComponentProfileModel.name.in_(template.related_components))
                    .all()
                )
                rows = [
                    {"sql_template_id": template.id, "component_id": name_to_id[name]}
                    for name in template.related_components
                    if name in name_to_id
                ]
                if rows:
                    session.execute(sql_template_components.insert(), rows)
            session.commit()
            logger.debug("Saved SQL template %s", template.id)

    def list_all(self, *, active_only: bool = True) -> list[QueryTemplate]:
        with self._session_factory() as session:
            query = session.query(SqlTemplateModel)
            if active_only:
                query = query.filter(SqlTemplateModel.is_active.is_(True))
            models = query.order_by(SqlTemplateModel.title).all()
            links = self._links(session, [m.id for m in models])
            return [_to_domain(m, links.get(m.id, [])) for m in models]

    def get(self, template_id: str) -> QueryTemplate | None:
        with self._session_factory() as session:
            model = session.get(SqlTemplateModel, template_id)
            if model is None:
                return None
            links = self._links(session, [template_id])
            return _to_domain(model, links.get(template_id, []))

    def count(self) -> int:
        with self._session_factory() as session:
            return session.query(SqlTemplateModel).count()

    def delete(self, template_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(SqlTemplateModel, template_id)
            if model is not None:
                session.execute(sql_template_components.delete().where(sql_template_components.c.sql_template_id == template_id))
                session.delete(model)
                session.commit()

    def _links(self, session: OrmSession, template_ids: list[str]) -> dict[str, list[str]]:
        if not template_ids:
            return {}
        rows = (
            session.query(sql_template_components.c.sql_template_id, ComponentProfileModel.name)
            .join(ComponentProfileModel, ComponentProfileModel.id == sql_template_components.c.component_id)
            .filter(sql_template_components.c.sql_template_id.in_(template_ids))
            .all()
        )
        grouped: dict[str, list[str]] = defaultdict(list)
        for template_id, component_name in rows:
            grouped[template_id].append(component_name)
        return dict(grouped)


def _to_domain(model: SqlTemplateModel, related_components: list[str]) -> QueryTemplate:
    return QueryTemplate(
        id=model.id,
        title=model.title,
        category=model.category,
        sql_text=model.sql_text,
        explanation=model.explanation,
        tags=model.tags,
        related_components=related_components,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
        status=model.status,
    )
