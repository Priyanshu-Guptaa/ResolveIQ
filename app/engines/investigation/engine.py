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

from app.domain.enums import EvidenceType
from app.domain.evidence import Evidence
from app.domain.investigation import InvestigationSession
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
    ) -> None:
        self._repository = repository
        self._log_intelligence = log_intelligence

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

    def add_log_evidence(self, investigation_id: str, filename: str, content: str) -> Evidence:
        self._ensure_exists(investigation_id)
        evidence = Evidence(
            investigation_id=investigation_id,
            evidence_type=EvidenceType.LOG_FILE,
            source="upload",
            title=filename,
            raw_content=content,
            metadata={"filename": filename},
        )
        self._log_intelligence.analyze_evidence(evidence)
        self._repository.add_evidence(investigation_id, evidence)
        logger.info(
            "Added log evidence '%s' to investigation %s (%d events, %d entities)",
            filename,
            investigation_id,
            len(evidence.log_events),
            len(evidence.extracted_entities),
        )
        return evidence

    def get_investigation(self, investigation_id: str) -> InvestigationSession:
        investigation = self._repository.get(investigation_id)
        if investigation is None:
            raise InvestigationNotFoundError(investigation_id)
        return investigation

    def list_investigations(self) -> list[InvestigationSession]:
        return self._repository.list_all()

    def _ensure_exists(self, investigation_id: str) -> None:
        if self._repository.get(investigation_id) is None:
            raise InvestigationNotFoundError(investigation_id)
