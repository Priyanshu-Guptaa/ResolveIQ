"""Repository pattern for InvestigationSession persistence.

The Investigation Engine depends on the :class:`InvestigationRepository`
Protocol, not on SQLAlchemy -- keeping persistence swappable and the engine
unit-testable with an in-memory fake.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Protocol

from sqlalchemy import func, text
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.entities import ExtractedEntity, LogEvent
from app.domain.evidence import Evidence
from app.domain.investigation import (
    ActivityItem,
    EntityTypeSummary,
    EvidenceSummary,
    InvestigationDetailSummary,
    InvestigationListItem,
    InvestigationSession,
)
from app.infrastructure.db.models import EvidenceModel, InvestigationModel

logger = logging.getLogger(__name__)


class InvestigationRepository(Protocol):
    """Persistence boundary for investigations and their evidence."""

    def save(self, investigation: InvestigationSession) -> None:
        ...

    def get(self, investigation_id: str) -> InvestigationSession | None:
        ...

    def exists(self, investigation_id: str) -> bool:
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

    def get_summary(self, investigation_id: str) -> InvestigationDetailSummary | None:
        ...

    def list_evidence_summaries(self, investigation_id: str) -> list[EvidenceSummary]:
        ...

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        ...

    def find_duplicate_evidence(self, investigation_id: str, content_hash: str) -> str | None:
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

    def exists(self, investigation_id: str) -> bool:
        """Cheap existence check -- selects only the primary key, never
        touches the ``evidence`` relationship. Investigation loading
        redesign: ``InvestigationEngine._ensure_exists`` used to call
        ``get()`` (full hydration, including every evidence row's
        raw_content) just to check a row exists -- invisible while the
        main GET /investigations/{id} was already slow for the same
        reason, but once that was fixed this became the dominant cost
        (~6.4s of a ~6.4s request, measured on a real investigation)."""
        with self._session_factory() as session:
            return session.query(InvestigationModel.id).filter(InvestigationModel.id == investigation_id).first() is not None

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
                content_hash=evidence.content_hash,
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

    def get_summary(self, investigation_id: str) -> InvestigationDetailSummary | None:
        """The lightweight detail-view query (Investigation loading
        redesign) -- never selects ``raw_content`` or ``log_events``.

        First pass at this method selected the full ``extracted_entities``
        JSON column and aggregated it in Python -- wrong assumption:
        "proportional to distinct entities found, not raw log size" broke
        down on a real noisy log file where the regex entity extractor
        matched ~19,500 times, an 1.56MB JSON blob for one evidence row
        alone. Deserializing and Python-aggregating that (and five more
        substantial ones) took ~7 seconds per request -- far better than
        151MB/17s, but still not "lightweight." Fixed by pushing the
        aggregation into SQLite itself via the JSON1 extension's
        ``json_each`` table-valued function: only the already-grouped
        (type, value, count) rows -- 6,167 for this real investigation,
        not 19,500+ raw entries -- ever cross into Python. Measured
        0.05s for the same investigation, vs ~7s before this fix.
        """
        with self._session_factory() as session:
            investigation = session.get(InvestigationModel, investigation_id)
            if investigation is None:
                return None

            type_rows = (
                session.query(
                    EvidenceModel.evidence_type,
                    EvidenceModel.created_at,
                )
                .filter(EvidenceModel.investigation_id == investigation_id)
                .all()
            )
            evidence_count_by_type: dict[str, int] = {}
            last_activity_at = None
            for evidence_type, created_at in type_rows:
                evidence_count_by_type[evidence_type] = evidence_count_by_type.get(evidence_type, 0) + 1
                if last_activity_at is None or created_at > last_activity_at:
                    last_activity_at = created_at

            entity_rows = session.execute(
                text(
                    """
                    SELECT
                        json_extract(je.value, '$.entity_type') AS entity_type,
                        json_extract(je.value, '$.value') AS entity_value,
                        COUNT(*) AS cnt
                    FROM evidence, json_each(evidence.extracted_entities) AS je
                    WHERE evidence.investigation_id = :investigation_id
                    GROUP BY entity_type, entity_value
                    """
                ),
                {"investigation_id": investigation_id},
            ).all()

            entity_agg: dict[str, Counter] = {}
            for entity_type, entity_value, count in entity_rows:
                entity_agg.setdefault(entity_type, Counter())[entity_value] = count

            entity_summary = [
                EntityTypeSummary(
                    entity_type=entity_type,
                    count=sum(counter.values()),
                    sample_values=[value for value, _ in counter.most_common(10)],
                )
                for entity_type, counter in entity_agg.items()
            ]
            entity_summary.sort(key=lambda item: item.count, reverse=True)

            return InvestigationDetailSummary(
                id=investigation.id,
                title=investigation.title,
                status=investigation.status,
                created_at=investigation.created_at,
                updated_at=investigation.updated_at,
                last_viewed_at=investigation.last_viewed_at,
                customer=investigation.customer,
                product=investigation.product,
                version=investigation.version,
                technology=investigation.technology,
                assigned_engineer=investigation.assigned_engineer,
                evidence_count=len(type_rows),
                evidence_count_by_type=evidence_count_by_type,
                entity_summary=entity_summary,
                last_activity_at=last_activity_at,
            )

    def list_evidence_summaries(self, investigation_id: str) -> list[EvidenceSummary]:
        """Backs ``GET /investigations/{id}/evidence`` -- the Explorer's
        evidence list. Deliberately never selects ``raw_content``/
        ``extracted_entities``/``log_events`` themselves: sizes/counts are
        computed at the SQL level (``LENGTH()``, ``json_array_length()``)
        so a 15MB log file's content never leaves the database just to
        report "here's how big it is\""."""
        with self._session_factory() as session:
            rows = (
                session.query(
                    EvidenceModel.id,
                    EvidenceModel.evidence_type,
                    EvidenceModel.source,
                    EvidenceModel.title,
                    EvidenceModel.created_at,
                    EvidenceModel.evidence_metadata,
                    func.length(EvidenceModel.raw_content).label("content_length"),
                    func.json_array_length(EvidenceModel.extracted_entities).label("entity_count"),
                    func.json_array_length(EvidenceModel.log_events).label("log_event_count"),
                )
                .filter(EvidenceModel.investigation_id == investigation_id)
                .order_by(EvidenceModel.created_at)
                .all()
            )
            return [
                EvidenceSummary(
                    id=row.id,
                    evidence_type=row.evidence_type,
                    source=row.source,
                    title=row.title,
                    created_at=row.created_at,
                    file_kind=(row.evidence_metadata or {}).get("file_kind"),
                    content_length=row.content_length or 0,
                    entity_count=row.entity_count or 0,
                    log_event_count=row.log_event_count or 0,
                )
                for row in rows
            ]

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        """Single-row full fetch -- the on-demand path
        (``GET .../evidence/{evidence_id}`` and the preview endpoint built
        on top of it). Fetching one row's full content is the intended
        cost of "load content only when requested"; the problem this
        redesign fixes was fetching *every* row's full content on *every*
        page load, not fetching one row's content when asked for."""
        with self._session_factory() as session:
            model = session.get(EvidenceModel, evidence_id)
            if model is None:
                return None
            return _evidence_to_domain(model)

    def find_duplicate_evidence(self, investigation_id: str, content_hash: str) -> str | None:
        with self._session_factory() as session:
            row = (
                session.query(EvidenceModel.id)
                .filter(
                    EvidenceModel.investigation_id == investigation_id,
                    EvidenceModel.content_hash == content_hash,
                )
                .first()
            )
            return row[0] if row else None


def _evidence_to_domain(model: EvidenceModel) -> Evidence:
    return Evidence(
        id=model.id,
        investigation_id=model.investigation_id,
        evidence_type=model.evidence_type,
        source=model.source,
        title=model.title,
        raw_content=model.raw_content,
        created_at=model.created_at,
        extracted_entities=[ExtractedEntity(**e) for e in model.extracted_entities],
        log_events=[LogEvent(**e) for e in model.log_events],
        metadata=model.evidence_metadata,
        content_hash=model.content_hash,
    )


def _to_domain(model: InvestigationModel) -> InvestigationSession:
    evidence_list = [_evidence_to_domain(e) for e in model.evidence]
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
