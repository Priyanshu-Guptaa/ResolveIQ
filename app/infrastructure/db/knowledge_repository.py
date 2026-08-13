"""Repository for the three knowledge sources the Knowledge Engine owns:
known bugs, historical investigations, and documentation metadata
(Sprint 3, Phase 3.1).

Grouped into one repository (rather than three near-identical files)
because they were already grouped conceptually -- one engine
(:class:`~app.engines.knowledge.engine.KnowledgeEngine`), one sample
seed directory, one JSON-import shape. Each entity's relationship to
Component Profiles is resolved via a dedicated association table, joined
explicitly here (never an ORM lazy ``relationship()`` collection) so
listing N records is always exactly two queries, not N+1 -- the same
discipline already applied to ``list_recent_activity`` in
``repository.py``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Protocol

from sqlalchemy import func
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.evidence import DocumentationListItem, DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
from app.domain.knowledge_management import DocumentPage
from app.infrastructure.db.models import (
    ComponentProfileModel,
    DocumentationModel,
    HistoricalInvestigationModel,
    KnownBugModel,
    documentation_components,
    historical_investigation_components,
    known_bug_components,
)

logger = logging.getLogger(__name__)

_DOCUMENTATION_SORT_COLUMNS = {
    "title": DocumentationModel.title,
    "status": DocumentationModel.status,
    "created_at": DocumentationModel.created_at,
    "updated_at": DocumentationModel.updated_at,
}


class KnowledgeRepository(Protocol):
    """Persistence boundary for known bugs, historical investigations,
    and documentation metadata."""

    def save_known_bug(self, record: KnownBugRecord) -> None:
        ...

    def list_known_bugs(self, *, active_only: bool = True) -> list[KnownBugRecord]:
        ...

    def get_known_bug(self, bug_id: str) -> KnownBugRecord | None:
        ...

    def count_known_bugs(self) -> int:
        ...

    def delete_known_bug(self, bug_id: str) -> None:
        ...

    def save_historical_investigation(self, record: HistoricalInvestigationRecord) -> None:
        ...

    def list_historical_investigations(self, *, active_only: bool = True) -> list[HistoricalInvestigationRecord]:
        ...

    def get_historical_investigation(self, record_id: str) -> HistoricalInvestigationRecord | None:
        ...

    def count_historical_investigations(self) -> int:
        ...

    def delete_historical_investigation(self, record_id: str) -> None:
        ...

    def save_documentation(self, record: DocumentationRecord) -> None:
        ...

    def get_documentation(self, document_id: str) -> DocumentationRecord | None:
        ...

    def list_all_documentation(
        self, *, status: str | None = None, active_only: bool = True
    ) -> list[DocumentationRecord]:
        ...

    def list_documentation_summaries(
        self,
        *,
        status: str | None = None,
        product: str | None = None,
        technology: str | None = None,
        component: str | None = None,
        search: str | None = None,
        sort_by: str = "updated_at",
        sort_desc: bool = True,
        page: int = 1,
        page_size: int = 20,
        active_only: bool = True,
    ) -> DocumentPage:
        ...

    def count_documentation_by_status(self) -> dict[str, int]:
        ...

    def list_recently_added_documentation(self, limit: int = 5) -> list[DocumentationListItem]:
        ...

    def count_documentation(self) -> int:
        ...

    def delete_documentation(self, document_id: str) -> None:
        ...


class SqlAlchemyKnowledgeRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    # --- Known bugs --------------------------------------------------------

    def save_known_bug(self, record: KnownBugRecord) -> None:
        with self._session_factory() as session:
            model = session.get(KnownBugModel, record.id)
            if model is None:
                model = KnownBugModel(id=record.id)
                session.add(model)
            model.title = record.title
            model.description = record.description
            model.bug_status = record.bug_status
            model.affected_components = record.affected_components
            model.workaround = record.workaround
            model.resolution_verified = record.resolution_verified
            model.resolution_verified_by = record.resolution_verified_by
            model.resolution_verified_at = record.resolution_verified_at
            model.resolution_verification_note = record.resolution_verification_note
            model.created_at = record.created_at
            model.updated_at = record.updated_at
            model.created_by = record.created_by
            model.updated_by = record.updated_by
            model.is_active = record.is_active
            model.lifecycle_status = record.status.value
            session.commit()

            self._set_component_links(session, known_bug_components, "known_bug_id", record.id, record.related_components)
            logger.debug("Saved known bug %s", record.id)

    def list_known_bugs(self, *, active_only: bool = True) -> list[KnownBugRecord]:
        with self._session_factory() as session:
            query = session.query(KnownBugModel)
            if active_only:
                query = query.filter(KnownBugModel.is_active.is_(True))
            models = query.order_by(KnownBugModel.title).all()
            links = self._component_links(session, known_bug_components, "known_bug_id", [m.id for m in models])
            return [_known_bug_to_domain(m, links.get(m.id, [])) for m in models]

    def get_known_bug(self, bug_id: str) -> KnownBugRecord | None:
        with self._session_factory() as session:
            model = session.get(KnownBugModel, bug_id)
            if model is None:
                return None
            links = self._component_links(session, known_bug_components, "known_bug_id", [bug_id])
            return _known_bug_to_domain(model, links.get(bug_id, []))

    def count_known_bugs(self) -> int:
        with self._session_factory() as session:
            return session.query(KnownBugModel).count()

    def delete_known_bug(self, bug_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(KnownBugModel, bug_id)
            if model is not None:
                session.execute(known_bug_components.delete().where(known_bug_components.c.known_bug_id == bug_id))
                session.delete(model)
                session.commit()

    # --- Historical investigations ------------------------------------------

    def save_historical_investigation(self, record: HistoricalInvestigationRecord) -> None:
        with self._session_factory() as session:
            model = session.get(HistoricalInvestigationModel, record.id)
            if model is None:
                model = HistoricalInvestigationModel(id=record.id)
                session.add(model)
            model.title = record.title
            model.description = record.description
            model.root_cause = record.root_cause
            model.resolution = record.resolution
            model.next_step = record.next_step
            model.tags = record.tags
            model.domain = record.domain
            model.resolution_verified = record.resolution_verified
            model.resolution_verified_by = record.resolution_verified_by
            model.resolution_verified_at = record.resolution_verified_at
            model.resolution_verification_note = record.resolution_verification_note
            model.created_at = record.created_at
            model.updated_at = record.updated_at
            model.created_by = record.created_by
            model.updated_by = record.updated_by
            model.is_active = record.is_active
            model.status = record.status.value
            session.commit()

            self._set_component_links(
                session,
                historical_investigation_components,
                "historical_investigation_id",
                record.id,
                record.related_components,
            )
            logger.debug("Saved historical investigation %s", record.id)

    def list_historical_investigations(self, *, active_only: bool = True) -> list[HistoricalInvestigationRecord]:
        with self._session_factory() as session:
            query = session.query(HistoricalInvestigationModel)
            if active_only:
                query = query.filter(HistoricalInvestigationModel.is_active.is_(True))
            models = query.order_by(HistoricalInvestigationModel.title).all()
            links = self._component_links(
                session, historical_investigation_components, "historical_investigation_id", [m.id for m in models]
            )
            return [_historical_investigation_to_domain(m, links.get(m.id, [])) for m in models]

    def get_historical_investigation(self, record_id: str) -> HistoricalInvestigationRecord | None:
        with self._session_factory() as session:
            model = session.get(HistoricalInvestigationModel, record_id)
            if model is None:
                return None
            links = self._component_links(
                session, historical_investigation_components, "historical_investigation_id", [record_id]
            )
            return _historical_investigation_to_domain(model, links.get(record_id, []))

    def count_historical_investigations(self) -> int:
        with self._session_factory() as session:
            return session.query(HistoricalInvestigationModel).count()

    def delete_historical_investigation(self, record_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(HistoricalInvestigationModel, record_id)
            if model is not None:
                session.execute(
                    historical_investigation_components.delete().where(
                        historical_investigation_components.c.historical_investigation_id == record_id
                    )
                )
                session.delete(model)
                session.commit()

    # --- Documentation (Sprint 3, Phase 3.2 -- Knowledge Management) -------

    def save_documentation(self, record: DocumentationRecord) -> None:
        with self._session_factory() as session:
            model = session.get(DocumentationModel, record.id)
            if model is None:
                model = DocumentationModel(id=record.id)
                session.add(model)
            model.title = record.title
            model.content = record.content
            model.tags = record.tags
            model.source = record.source
            model.product = record.product
            model.version = record.version
            model.technology = record.technology
            model.status = record.status.value
            model.original_filename = record.original_filename
            model.file_type = record.file_type
            model.file_path = record.file_path
            model.created_at = record.created_at
            model.updated_at = record.updated_at
            model.created_by = record.created_by
            model.updated_by = record.updated_by
            model.is_active = record.is_active
            session.commit()

            self._set_component_links(
                session, documentation_components, "documentation_id", record.id, record.related_components
            )
            logger.debug("Saved documentation %s (status=%s)", record.id, model.status)

    def get_documentation(self, document_id: str) -> DocumentationRecord | None:
        with self._session_factory() as session:
            model = session.get(DocumentationModel, document_id)
            if model is None:
                return None
            links = self._component_links(session, documentation_components, "documentation_id", [document_id])
            return _documentation_to_domain(model, links.get(document_id, []))

    def list_all_documentation(
        self, *, status: str | None = None, active_only: bool = True
    ) -> list[DocumentationRecord]:
        """Full records, content included -- used by KnowledgeEngine to
        seed ChromaDB (only ever called with ``status="published"``) and
        by the migration's idempotent-per-row backfill. Never used to
        serve the Library's list view -- see ``list_documentation_summaries``."""
        with self._session_factory() as session:
            query = session.query(DocumentationModel)
            if active_only:
                query = query.filter(DocumentationModel.is_active.is_(True))
            if status:
                query = query.filter(DocumentationModel.status == status)
            models = query.order_by(DocumentationModel.title).all()
            links = self._component_links(
                session, documentation_components, "documentation_id", [m.id for m in models]
            )
            return [_documentation_to_domain(m, links.get(m.id, [])) for m in models]

    def list_documentation_summaries(
        self,
        *,
        status: str | None = None,
        product: str | None = None,
        technology: str | None = None,
        component: str | None = None,
        search: str | None = None,
        sort_by: str = "updated_at",
        sort_desc: bool = True,
        page: int = 1,
        page_size: int = 20,
        active_only: bool = True,
    ) -> DocumentPage:
        """Backs the Knowledge Library: search/filter/sort/pagination,
        with ``DocumentationModel.content`` deliberately never selected
        -- Phase 3.2's explicit "avoid loading document content unless
        requested." ``search`` matches document titles (a WHERE LIKE,
        not semantic search -- semantic relevance search stays exactly
        where it already lives, ``KnowledgeEngine.search_documentation``,
        and only ever covers *published* documents; the Library has to
        find drafts too, which were never indexed)."""
        with self._session_factory() as session:
            query = session.query(DocumentationModel)
            if active_only:
                query = query.filter(DocumentationModel.is_active.is_(True))
            if status:
                query = query.filter(DocumentationModel.status == status)
            if product:
                query = query.filter(DocumentationModel.product == product)
            if technology:
                query = query.filter(DocumentationModel.technology == technology)
            if search:
                query = query.filter(DocumentationModel.title.ilike(f"%{search}%"))
            if component:
                query = query.join(
                    documentation_components, documentation_components.c.documentation_id == DocumentationModel.id
                ).join(
                    ComponentProfileModel, ComponentProfileModel.id == documentation_components.c.component_id
                ).filter(ComponentProfileModel.name == component)

            total_count = query.count()

            sort_column = _DOCUMENTATION_SORT_COLUMNS.get(sort_by, DocumentationModel.updated_at)
            query = query.order_by(sort_column.desc() if sort_desc else sort_column.asc())
            query = query.offset(max(0, page - 1) * page_size).limit(page_size)

            models = query.all()
            links = self._component_links(
                session, documentation_components, "documentation_id", [m.id for m in models]
            )
            items = [
                DocumentationListItem(
                    id=m.id,
                    title=m.title,
                    tags=m.tags,
                    source=m.source,
                    product=m.product,
                    version=m.version,
                    technology=m.technology,
                    related_components=links.get(m.id, []),
                    status=m.status,
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    created_by=m.created_by,
                    updated_by=m.updated_by,
                    is_active=m.is_active,
                )
                for m in models
            ]
            return DocumentPage(items=items, total_count=total_count, page=page, page_size=page_size)

    def count_documentation_by_status(self) -> dict[str, int]:
        with self._session_factory() as session:
            rows = (
                session.query(DocumentationModel.status, func.count(DocumentationModel.id))
                .filter(DocumentationModel.is_active.is_(True))
                .group_by(DocumentationModel.status)
                .all()
            )
            return {status: count for status, count in rows}

    def list_recently_added_documentation(self, limit: int = 5) -> list[DocumentationListItem]:
        return self.list_documentation_summaries(
            sort_by="created_at", sort_desc=True, page=1, page_size=limit
        ).items

    def count_documentation(self) -> int:
        with self._session_factory() as session:
            return session.query(DocumentationModel).count()

    def delete_documentation(self, document_id: str) -> None:
        """Removes the database row only -- the Knowledge Object
        Framework service unindexes it from ChromaDB first (this
        repository has no knowledge of the search index) and leaves the
        uploaded original file on disk (harmless orphan, not cleaned up
        this phase -- see the Phase 3.4 report's technical debt)."""
        with self._session_factory() as session:
            model = session.get(DocumentationModel, document_id)
            if model is not None:
                session.execute(documentation_components.delete().where(documentation_components.c.documentation_id == document_id))
                session.delete(model)
                session.commit()

    # --- Shared association-table helpers -----------------------------------

    def _set_component_links(self, session: OrmSession, table, owner_column: str, owner_id: str, component_names: list[str]) -> None:
        """Replaces every link row for one owning record with the given
        component names, resolved to component ids by name. Unknown
        names (no matching component) are silently skipped -- this is a
        link table, not a validator; ``related_components`` is expected
        to already be the resolved/matched subset by the time it reaches
        here (see ``seed_migration.py``)."""
        session.execute(table.delete().where(getattr(table.c, owner_column) == owner_id))
        if component_names:
            name_to_id = dict(
                session.query(ComponentProfileModel.name, ComponentProfileModel.id)
                .filter(ComponentProfileModel.name.in_(component_names))
                .all()
            )
            rows = [
                {owner_column: owner_id, "component_id": name_to_id[name]}
                for name in component_names
                if name in name_to_id
            ]
            if rows:
                session.execute(table.insert(), rows)
        session.commit()

    def _component_links(
        self, session: OrmSession, table, owner_column: str, owner_ids: list[str]
    ) -> dict[str, list[str]]:
        """One bounded, joined query resolving every owner id's linked
        component *names* at once -- never N+1, regardless of how many
        owner_ids are passed."""
        if not owner_ids:
            return {}
        rows = (
            session.query(getattr(table.c, owner_column), ComponentProfileModel.name)
            .join(ComponentProfileModel, ComponentProfileModel.id == table.c.component_id)
            .filter(getattr(table.c, owner_column).in_(owner_ids))
            .all()
        )
        grouped: dict[str, list[str]] = defaultdict(list)
        for owner_id, component_name in rows:
            grouped[owner_id].append(component_name)
        return dict(grouped)


def _known_bug_to_domain(model: KnownBugModel, related_components: list[str]) -> KnownBugRecord:
    return KnownBugRecord(
        id=model.id,
        title=model.title,
        description=model.description,
        bug_status=model.bug_status,
        affected_components=model.affected_components,
        workaround=model.workaround,
        related_components=related_components,
        resolution_verified=model.resolution_verified,
        resolution_verified_by=model.resolution_verified_by,
        resolution_verified_at=model.resolution_verified_at,
        resolution_verification_note=model.resolution_verification_note,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
        status=model.lifecycle_status,
    )


def _historical_investigation_to_domain(
    model: HistoricalInvestigationModel, related_components: list[str]
) -> HistoricalInvestigationRecord:
    return HistoricalInvestigationRecord(
        id=model.id,
        title=model.title,
        description=model.description,
        root_cause=model.root_cause,
        resolution=model.resolution,
        next_step=model.next_step,
        tags=model.tags,
        domain=model.domain,
        related_components=related_components,
        resolution_verified=model.resolution_verified,
        resolution_verified_by=model.resolution_verified_by,
        resolution_verified_at=model.resolution_verified_at,
        resolution_verification_note=model.resolution_verification_note,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
        status=model.status,
    )


def _documentation_to_domain(model: DocumentationModel, related_components: list[str]) -> DocumentationRecord:
    return DocumentationRecord(
        id=model.id,
        title=model.title,
        content=model.content,
        tags=model.tags,
        source=model.source,
        product=model.product,
        version=model.version,
        technology=model.technology,
        related_components=related_components,
        status=model.status,
        original_filename=model.original_filename,
        file_type=model.file_type,
        file_path=model.file_path,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
    )
