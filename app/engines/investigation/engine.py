"""Investigation Engine: owns the investigation lifecycle and shared
context.

Per the product philosophy, this is the module that ensures "the user
should never have to repeatedly enter the same information" -- every piece
of evidence added here is immediately run through the Log Intelligence
Engine and folded into the investigation's shared context, which the
Recommendation Engine then reads without needing anything re-supplied.
"""

from __future__ import annotations

import logging

from app.domain.enums import EvidenceType, InvestigationStatus
from app.domain.evidence import Evidence
from app.domain.investigation import (
    ActivityItem,
    DashboardStats,
    InvestigationListItem,
    InvestigationSession,
)
from app.engines.ingestion.engine import IngestionEngine
from app.engines.log_intelligence.engine import LogIntelligenceEngine
from app.infrastructure.db.repository import InvestigationRepository

logger = logging.getLogger(__name__)


class InvestigationNotFoundError(Exception):
    def __init__(self, investigation_id: str) -> None:
        super().__init__(f"Investigation {investigation_id!r} not found")
        self.investigation_id = investigation_id


class InvestigationEngine:
    def __init__(
        self,
        repository: InvestigationRepository,
        log_intelligence: LogIntelligenceEngine,
        ingestion: IngestionEngine,
    ) -> None:
        self._repository = repository
        self._log_intelligence = log_intelligence
        self._ingestion = ingestion

    def start_investigation(self, title: str, description: str = "") -> InvestigationSession:
        """Create a new investigation, optionally seeded with the initial
        task description as its first piece of evidence -- the common case
        when an engineer pastes a ServiceNow task."""
        investigation = InvestigationSession(title=title)
        self._repository.save(investigation)
        logger.info("Started investigation %s: %s", investigation.id, title)

        if description.strip():
            self.add_text_evidence(
                investigation.id,
                description,
                evidence_type=EvidenceType.TASK_DESCRIPTION,
                source="manual",
                title="Task description",
            )
            investigation = self.get_investigation(investigation.id)

        return investigation

    def add_text_evidence(
        self,
        investigation_id: str,
        text: str,
        *,
        evidence_type: EvidenceType = EvidenceType.MANUAL_NOTE,
        source: str = "manual",
        title: str = "",
    ) -> Evidence:
        self._ensure_exists(investigation_id)
        evidence = Evidence(
            investigation_id=investigation_id,
            evidence_type=evidence_type,
            source=source,
            title=title or evidence_type.value.replace("_", " ").title(),
            raw_content=text,
        )
        self._log_intelligence.analyze_evidence(evidence)
        self._repository.add_evidence(investigation_id, evidence)
        return evidence

    def add_file_evidence(self, investigation_id: str, filename: str, content: bytes) -> list[Evidence]:
        """Upload path for any file (log, docx, xlsx, pdf, image, evtx, or
        a zip of any of those). Runs the Evidence Ingestion Pipeline
        first -- file-type detection, the matching parser, normalized text
        -- so entity extraction only ever sees real extracted text, never
        raw bytes (RFC rev 1 §04 / Problem 1).

        Returns a list because a .zip fans out into one Evidence item per
        archive entry; every other upload returns a single-item list.
        """
        self._ensure_exists(investigation_id)
        parsed_files = self._ingestion.parse(filename, content)

        results: list[Evidence] = []
        for parsed in parsed_files:
            evidence = Evidence(
                investigation_id=investigation_id,
                evidence_type=EvidenceType.LOG_FILE,
                source="upload",
                title=parsed.filename,
                raw_content=parsed.text,
                metadata={
                    "filename": parsed.filename,
                    "file_kind": parsed.kind.value,
                    "warnings": [w.model_dump() for w in parsed.warnings],
                    **parsed.metadata,
                },
            )
            self._log_intelligence.analyze_evidence(evidence)
            self._repository.add_evidence(investigation_id, evidence)
            logger.info(
                "Added %s evidence '%s' to investigation %s (%d events, %d entities, %d warnings)",
                parsed.kind.value,
                parsed.filename,
                investigation_id,
                len(evidence.log_events),
                len(evidence.extracted_entities),
                len(parsed.warnings),
            )
            results.append(evidence)
        return results

    def get_investigation(self, investigation_id: str) -> InvestigationSession:
        investigation = self._repository.get(investigation_id)
        if investigation is None:
            raise InvestigationNotFoundError(investigation_id)
        return investigation

    def list_investigations(self) -> list[InvestigationSession]:
        """Full hydration, evidence included. Only use this where evidence
        content is actually needed -- for list/summary views, use
        :meth:`list_investigation_summaries` instead (see
        InvestigationListItem's docstring)."""
        return self._repository.list_all()

    def list_investigation_summaries(self, limit: int | None = None) -> list[InvestigationListItem]:
        return self._repository.list_summaries(limit)

    def mark_viewed(self, investigation_id: str) -> None:
        """Record that an engineer opened this investigation -- backs the
        Dashboard's Recently Viewed panel. Call from the Workspace page on
        load, not from every API poll."""
        self._ensure_exists(investigation_id)
        self._repository.touch_viewed(investigation_id)

    def list_recently_viewed(self, limit: int = 5) -> list[InvestigationListItem]:
        return self._repository.list_recently_viewed(limit)

    def list_recent_activity(self, limit: int = 10) -> list[ActivityItem]:
        return self._repository.list_recent_activity(limit)

    def get_timeline(self, investigation_id: str, limit: int = 50) -> list[ActivityItem]:
        """Real events for one investigation, chronological -- backs the
        Workspace's Timeline nav item (Phase 2A). Built from the same
        Evidence rows Dashboard's activity feed uses; a dedicated Timeline
        Engine (later phase) can enrich this with recommendation-generated
        and status-change events without changing this method's shape."""
        self._ensure_exists(investigation_id)
        return self._repository.list_activity_for_investigation(investigation_id, limit)

    def update_details(
        self,
        investigation_id: str,
        *,
        customer: str | None = None,
        product: str | None = None,
        version: str | None = None,
        technology: str | None = None,
        assigned_engineer: str | None = None,
    ) -> InvestigationSession:
        """Saves engineer-entered case metadata for the persistent Summary
        Card (Phase 2A). Only fields explicitly passed are updated --
        pass only what changed, existing values for other fields are
        preserved (see the router for how partial updates are built)."""
        self._ensure_exists(investigation_id)
        fields = {
            "customer": customer,
            "product": product,
            "version": version,
            "technology": technology,
            "assigned_engineer": assigned_engineer,
        }
        self._repository.update_details(investigation_id, **{k: v for k, v in fields.items() if v is not None})
        return self.get_investigation(investigation_id)

    def get_dashboard_stats(self) -> DashboardStats:
        # Lightweight summaries -- stats only need status/created_at/
        # updated_at, never evidence content (Phase 1.5 fix).
        investigations = self._repository.list_summaries()
        active = [i for i in investigations if i.status in (InvestigationStatus.OPEN, InvestigationStatus.IN_PROGRESS)]
        resolved = [i for i in investigations if i.status == InvestigationStatus.RESOLVED]

        avg_hours: float | None = None
        if resolved:
            total_seconds = sum((i.updated_at - i.created_at).total_seconds() for i in resolved)
            avg_hours = round((total_seconds / len(resolved)) / 3600, 1)

        return DashboardStats(
            total_count=len(investigations),
            active_count=len(active),
            resolved_count=len(resolved),
            avg_resolution_hours=avg_hours,
        )

    def _ensure_exists(self, investigation_id: str) -> None:
        if self._repository.get(investigation_id) is None:
            raise InvestigationNotFoundError(investigation_id)
