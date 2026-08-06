"""Repository pattern for InvestigationSession persistence.

The Investigation Engine depends on the :class:`InvestigationRepository`
Protocol, not on SQLAlchemy -- keeping persistence swappable and the engine
unit-testable with an in-memory fake.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.entities import ExtractedEntity, LogEvent
from app.domain.evidence import Evidence
from app.domain.investigation import ActivityItem, InvestigationSession
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

    def add_evidence(self, investigation_id: str, evidence: Evidence) -> None:
        ...

    def touch_viewed(self, investigation_id: str) -> None:
        ...

    def list_recent_activity(self, limit: int = 10) -> list[ActivityItem]:
        ...

    def list_recently_viewed(self, limit: int = 5) -> list[InvestigationSession]:
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
        """Cross-investigation feed: every investigation-created and
        evidence-added event, most recent first. Backs the Dashboard's
        Recent Activity feed and Recent Documents panel -- real Evidence
        rows only, nothing fabricated.
        """
        with self._session_factory() as session:
            items: list[ActivityItem] = []

            for inv in session.query(InvestigationModel).all():
                items.append(
                    ActivityItem(
                        investigation_id=inv.id,
                        investigation_title=inv.title,
                        kind="investigation_created",
                        label=f"Investigation created: {inv.title}",
                        occurred_at=inv.created_at,
                    )
                )

            evidence_rows = (
                session.query(EvidenceModel)
                .order_by(EvidenceModel.created_at.desc())
                .limit(limit * 3)  # over-fetch; final sort/limit happens below
                .all()
            )
            for ev in evidence_rows:
                inv_title = ev.investigation.title if ev.investigation else "(deleted investigation)"
                items.append(
                    ActivityItem(
                        investigation_id=ev.investigation_id,
                        investigation_title=inv_title,
                        kind="evidence_added",
                        label=f"{ev.title or ev.evidence_type} added to {inv_title}",
                        evidence_type=ev.evidence_type,
                        occurred_at=ev.created_at,
                    )
                )

            items.sort(key=lambda item: item.occurred_at, reverse=True)
            return items[:limit]

    def list_recently_viewed(self, limit: int = 5) -> list[InvestigationSession]:
        with self._session_factory() as session:
            models = (
                session.query(InvestigationModel)
                .filter(InvestigationModel.last_viewed_at.isnot(None))
                .order_by(InvestigationModel.last_viewed_at.desc())
                .limit(limit)
                .all()
            )
            return [_to_domain(model) for model in models]


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
        evidence=evidence_list,
    )
