"""Repository pattern for InvestigationSession persistence.

The Investigation Engine depends on the :class:`InvestigationRepository`
Protocol, not on SQLAlchemy -- keeping persistence swappable and the engine
unit-testable with an in-memory fake.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy import func
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.entities import ExtractedEntity, LogEvent
from app.domain.evidence import Evidence
from app.domain.investigation import ActivityItem, InvestigationListItem, InvestigationSession
from app.infrastructure.db.models import EvidenceModel, InvestigationModel

logger = logging.getLogger(__name__)


class InvestigationRepository(Protocol):
    """Persistence boundary for investigations and their evidence."""

    def save(self, investigation: InvestigationSession) -> None:
        ...

    def get(self, investigation_id: str) -> InvestigationSession | None:
        ...

    def list_all(self) -> list[InvestigationSession]:
        ...

    def list_summaries(self, limit: int | None = None) -> list[InvestigationListItem]:
        ...

    def add_evidence(self, investigation_id: str, evidence: Evidence) -> None:
        ...

    def touch_viewed(self, investigation_id: str) -> None:
        ...

    def list_recent_activity(self, limit: int = 10) -> list[ActivityItem]:
        ...

    def list_recently_viewed(self, limit: int = 5) -> list[InvestigationListItem]:
        ...

    def update_details(self, investigation_id: str, **fields: str | None) -> None:
        ...

    def list_activity_for_investigation(self, investigation_id: str, limit: int = 50) -> list[ActivityItem]:
        ...


class SqlAlchemyInvestigationRepository:
    """SQLite-backed (via SQLAlchemy) implementation."""

    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def save(self, investigation: InvestigationSession) -> None:
        with self._session_factory() as session:
            model = session.get(InvestigationModel, investigation.id)
            if model is None:
                model = InvestigationModel(id=investigation.id)
                session.add(model)
            model.title = investigation.title
            model.status = investigation.status.value
            model.created_at = investigation.created_at
            model.updated_at = investigation.updated_at
            model.last_viewed_at = investigation.last_viewed_at
            model.customer = investigation.customer
            model.product = investigation.product
            model.version = investigation.version
            model.technology = investigation.technology
            model.assigned_engineer = investigation.assigned_engineer
            session.commit()
            logger.debug("Saved investigation %s", investigation.id)

    def get(self, investigation_id: str) -> InvestigationSession | None:
        with self._session_factory() as session:
            model = session.get(InvestigationModel, investigation_id)
            if model is None:
                return None
            return _to_domain(model)

    def list_all(self) -> list[InvestigationSession]:
        with self._session_factory() as session:
            models = session.query(InvestigationModel).order_by(
                InvestigationModel.updated_at.desc()
            ).all()
            return [_to_domain(model) for model in models]

    def list_summaries(self, limit: int | None = None) -> list[InvestigationListItem]:
        """Lightweight list view: one grouped query, no evidence-content
        hydration, no N+1. Phase 1.5 fix -- see InvestigationListItem's
        docstring for why this exists alongside list_all()."""
        with self._session_factory() as session:
            query = (
                session.query(
                    InvestigationModel.id,
                    InvestigationModel.title,
                    InvestigationModel.status,
                    InvestigationModel.created_at,
                    InvestigationModel.updated_at,
                    InvestigationModel.last_viewed_at,
                    func.count(EvidenceModel.id).label("evidence_count"),
                )
                .outerjoin(EvidenceModel, EvidenceModel.investigation_id == InvestigationModel.id)
                .group_by(InvestigationModel.id)
                .order_by(InvestigationModel.updated_at.desc())
            )
            if limit is not None:
                query = query.limit(limit)
            return [
                InvestigationListItem(
                    id=row.id,
                    title=row.title,
                    status=row.status,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    last_viewed_at=row.last_viewed_at,
                    evidence_count=row.evidence_count,
                )
                for row in query.all()
            ]

    def add_evidence(self, investigation_id: str, evidence: Evidence) -> None:
        with self._session_factory() as session:
            model = EvidenceModel(
                id=evidence.id,
                investigation_id=investigation_id,
                evidence_type=evidence.evidence_type.value,
                source=evidence.source,
                title=evidence.title,
                raw_content=evidence.raw_content,
                created_at=evidence.created_at,
                extracted_entities=[e.model_dump(mode="json") for e in evidence.extracted_entities],
                log_events=[e.model_dump(mode="json") for e in evidence.log_events],
                evidence_metadata=evidence.metadata,
            )
            session.add(model)

            investigation = session.get(InvestigationModel, investigation_id)
            if investigation is not None:
                from datetime import datetime, timezone

                investigation.updated_at = datetime.now(timezone.utc)

            session.commit()
            logger.debug("Added evidence %s to investigation %s", evidence.id, investigation_id)

    def touch_viewed(self, investigation_id: str) -> None:
        from datetime import datetime, timezone

        with self._session_factory() as session:
            model = session.get(InvestigationModel, investigation_id)
            if model is not None:
                model.last_viewed_at = datetime.now(timezone.utc)
                session.commit()

    def list_recent_activity(self, limit: int = 10) -> list[ActivityItem]:
        """Cross-investigation feed: recent investigation-created and
        evidence-added events, most recent first. Backs the Dashboard's
        Recent Activity feed and Recent Documents panel -- real rows only,
        nothing fabricated.

        Phase 1.5 fix: this previously loaded every investigation ever
        created (unbounded ``.all()``) and full ``EvidenceModel`` ORM
        objects -- including their ``raw_content``/``extracted_entities``/
        ``log_events`` JSON blobs -- plus one lazy-loaded query per row to
        read ``evidence.investigation.title`` (N+1). None of that is
        needed for an activity label. Both queries below select only the
        columns used, join instead of lazy-loading, and bound to ``limit``
        rather than "everything, sorted after the fact."
        """
        with self._session_factory() as session:
            items: list[ActivityItem] = []

            recent_investigations = (
                session.query(
                    InvestigationModel.id,
                    InvestigationModel.title,
                    InvestigationModel.created_at,
                )
                .order_by(InvestigationModel.created_at.desc())
                .limit(limit)
                .all()
            )
            for inv_id, inv_title, created_at in recent_investigations:
                items.append(
                    ActivityItem(
                        investigation_id=inv_id,
                        investigation_title=inv_title,
                        kind="investigation_created",
                        label=f"Investigation created: {inv_title}",
                        occurred_at=created_at,
                    )
                )

            recent_evidence = (
                session.query(
                    EvidenceModel.investigation_id,
                    EvidenceModel.evidence_type,
                    EvidenceModel.title,
                    EvidenceModel.created_at,
                    InvestigationModel.title.label("investigation_title"),
                )
                .join(InvestigationModel, EvidenceModel.investigation_id == InvestigationModel.id)
                .order_by(EvidenceModel.created_at.desc())
                .limit(limit)
                .all()
            )
            for inv_id, ev_type, ev_title, created_at, inv_title in recent_evidence:
                items.append(
                    ActivityItem(
                        investigation_id=inv_id,
                        investigation_title=inv_title,
                        kind="evidence_added",
                        label=f"{ev_title or ev_type} added to {inv_title}",
                        evidence_type=ev_type,
                        occurred_at=created_at,
                    )
                )

            items.sort(key=lambda item: item.occurred_at, reverse=True)
            return items[:limit]

    def list_recently_viewed(self, limit: int = 5) -> list[InvestigationListItem]:
        with self._session_factory() as session:
            rows = (
                session.query(
                    InvestigationModel.id,
                    InvestigationModel.title,
                    InvestigationModel.status,
                    InvestigationModel.created_at,
                    InvestigationModel.updated_at,
                    InvestigationModel.last_viewed_at,
                    func.count(EvidenceModel.id).label("evidence_count"),
                )
                .outerjoin(EvidenceModel, EvidenceModel.investigation_id == InvestigationModel.id)
                .filter(InvestigationModel.last_viewed_at.isnot(None))
                .group_by(InvestigationModel.id)
                .order_by(InvestigationModel.last_viewed_at.desc())
                .limit(limit)
                .all()
            )
            return [
                InvestigationListItem(
                    id=row.id,
                    title=row.title,
                    status=row.status,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    last_viewed_at=row.last_viewed_at,
                    evidence_count=row.evidence_count,
                )
                for row in rows
            ]

    def update_details(self, investigation_id: str, **fields: str | None) -> None:
        """Saves engineer-entered case metadata (customer, product,
        version, technology, assigned_engineer) for the persistent Summary
        Card. Only known InvestigationModel columns are ever set --
        unrecognized kwargs are ignored rather than raising, so callers
        can pass a partial dict without filtering it first."""
        with self._session_factory() as session:
            model = session.get(InvestigationModel, investigation_id)
            if model is None:
                return
            for key, value in fields.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            session.commit()

    def list_activity_for_investigation(self, investigation_id: str, limit: int = 50) -> list[ActivityItem]:
        """Same shape as list_recent_activity(), scoped to one
        investigation -- backs the Workspace's Timeline nav item until the
        dedicated Timeline Engine (a later phase) replaces the underlying
        query without changing this method's contract."""
        with self._session_factory() as session:
            items: list[ActivityItem] = []

            investigation_row = (
                session.query(InvestigationModel.id, InvestigationModel.title, InvestigationModel.created_at)
                .filter(InvestigationModel.id == investigation_id)
                .first()
            )
            if investigation_row is None:
                return []
            inv_id, inv_title, created_at = investigation_row
            items.append(
                ActivityItem(
                    investigation_id=inv_id,
                    investigation_title=inv_title,
                    kind="investigation_created",
                    label=f"Investigation created: {inv_title}",
                    occurred_at=created_at,
                )
            )

            evidence_rows = (
                session.query(
                    EvidenceModel.evidence_type,
                    EvidenceModel.title,
                    EvidenceModel.created_at,
                )
                .filter(EvidenceModel.investigation_id == investigation_id)
                .order_by(EvidenceModel.created_at.desc())
                .limit(limit)
                .all()
            )
            for ev_type, ev_title, ev_created_at in evidence_rows:
                items.append(
                    ActivityItem(
                        investigation_id=investigation_id,
                        investigation_title=inv_title,
                        kind="evidence_added",
                        label=f"{ev_title or ev_type} added",
                        evidence_type=ev_type,
                        occurred_at=ev_created_at,
                    )
                )

            items.sort(key=lambda item: item.occurred_at, reverse=True)
            return items[:limit]


def _to_domain(model: InvestigationModel) -> InvestigationSession:
    evidence_list = [
        Evidence(
            id=e.id,
            investigation_id=e.investigation_id,
            evidence_type=e.evidence_type,
            source=e.source,
            title=e.title,
            raw_content=e.raw_content,
            created_at=e.created_at,
            extracted_entities=[ExtractedEntity(**ent) for ent in e.extracted_entities],
            log_events=[LogEvent(**ev) for ev in e.log_events],
            metadata=e.evidence_metadata,
        )
        for e in model.evidence
    ]
    return InvestigationSession(
        id=model.id,
        title=model.title,
        status=model.status,
        created_at=model.created_at,
        updated_at=model.updated_at,
        last_viewed_at=model.last_viewed_at,
        customer=model.customer,
        product=model.product,
        version=model.version,
        technology=model.technology,
        assigned_engineer=model.assigned_engineer,
        evidence=evidence_list,
    )
