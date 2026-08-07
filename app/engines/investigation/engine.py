"""Investigation Engine: owns the investigation lifecycle and shared
context.

Per the product philosophy, this is the module that ensures "the user
should never have to repeatedly enter the same information" -- every piece
of evidence added here is immediately run through the Log Intelligence
Engine and folded into the investigation's shared context, which the
Recommendation Engine then reads without needing anything re-supplied.
"""

from __future__ import annotations

import hashlib
import logging

from app.domain.enums import EvidenceType, InvestigationStatus
from app.domain.evidence import Evidence, EvidencePreview
from app.domain.investigation import (
    ActivityItem,
    DashboardStats,
    EvidenceSummary,
    InvestigationDetailSummary,
    InvestigationListItem,
    InvestigationSession,
)
from app.engines.ingestion.engine import IngestionEngine
from app.engines.log_intelligence.engine import LogIntelligenceEngine
from app.infrastructure.db.repository import InvestigationRepository

logger = logging.getLogger(__name__)

_DEFAULT_PREVIEW_CHARS = 2000


def _hash_content(text: str) -> str | None:
    """None for empty/whitespace-only content -- an empty hash can't
    distinguish two genuinely different textless files (e.g. two
    different unsupported binary uploads), so those are never
    deduplicated; every distinct upload of empty-text content is kept."""
    if not text or not text.strip():
        return None
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class InvestigationNotFoundError(Exception):
    def __init__(self, investigation_id: str) -> None:
        super().__init__(f"Investigation {investigation_id!r} not found")
        self.investigation_id = investigation_id


class EvidenceNotFoundError(Exception):
    def __init__(self, investigation_id: str, evidence_id: str) -> None:
        super().__init__(f"Evidence {evidence_id!r} not found on investigation {investigation_id!r}")
        self.investigation_id = investigation_id
        self.evidence_id = evidence_id


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
        content_hash = _hash_content(text)
        if content_hash is not None:
            existing_id = self._repository.find_duplicate_evidence(investigation_id, content_hash)
            if existing_id is not None:
                logger.info(
                    "Duplicate text evidence detected for investigation %s -- reusing existing %s instead of "
                    "creating a new row",
                    investigation_id,
                    existing_id,
                )
                return self._repository.get_evidence(existing_id)

        evidence = Evidence(
            investigation_id=investigation_id,
            evidence_type=evidence_type,
            source=source,
            title=title or evidence_type.value.replace("_", " ").title(),
            raw_content=text,
            content_hash=content_hash,
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

        Duplicate detection (Investigation loading redesign): found via a
        real 1.5MB zip that got processed twice after its first upload's
        client-side timeout led to a retry -- the server had actually
        succeeded, so the retry created 67 byte-identical duplicate rows.
        Each parsed file's content is hashed and checked against this
        investigation's existing evidence before inserting; a match
        reuses the existing row instead of creating a new one, so a
        retried upload (or literally re-uploading the same file) is a
        safe no-op, not a duplicate.
        """
        self._ensure_exists(investigation_id)
        parsed_files = self._ingestion.parse(filename, content)

        results: list[Evidence] = []
        for parsed in parsed_files:
            content_hash = _hash_content(parsed.text)
            if content_hash is not None:
                existing_id = self._repository.find_duplicate_evidence(investigation_id, content_hash)
                if existing_id is not None:
                    logger.info(
                        "Duplicate evidence detected ('%s') for investigation %s -- reusing existing %s instead "
                        "of re-adding",
                        parsed.filename,
                        investigation_id,
                        existing_id,
                    )
                    results.append(self._repository.get_evidence(existing_id))
                    continue

            evidence = Evidence(
                investigation_id=investigation_id,
                evidence_type=EvidenceType.LOG_FILE,
                source="upload",
                title=parsed.filename,
                raw_content=parsed.text,
                content_hash=content_hash,
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
        """Full hydration -- every piece of evidence's raw_content,
        extracted_entities, and log_events, all loaded at once. Reserved
        for internal callers that genuinely need it (the Recommendation
        Engine's merged_entities/context_text, most notably). Never call
        this to serve a client response directly -- see
        get_investigation_summary() for that (Investigation loading
        redesign: a real investigation with 97 evidence items made this
        method's result serialize to 151MB of JSON)."""
        investigation = self._repository.get(investigation_id)
        if investigation is None:
            raise InvestigationNotFoundError(investigation_id)
        return investigation

    def get_investigation_summary(self, investigation_id: str) -> InvestigationDetailSummary:
        """The lightweight external contract -- everything the Workspace
        needs (Summary Card, counts, aggregated entity findings) with zero
        embedded evidence content."""
        summary = self._repository.get_summary(investigation_id)
        if summary is None:
            raise InvestigationNotFoundError(investigation_id)
        return summary

    def list_evidence(self, investigation_id: str) -> list[EvidenceSummary]:
        """Backs the Explorer's evidence list -- metadata only, no
        content. Opening an investigation must not load every uploaded
        file's content; this is what makes that possible."""
        self._ensure_exists(investigation_id)
        return self._repository.list_evidence_summaries(investigation_id)

    def get_evidence(self, investigation_id: str, evidence_id: str) -> Evidence:
        """The on-demand full-content fetch for exactly one piece of
        evidence -- the intended cost of "load content when requested,"
        as opposed to get_investigation's "load everything, always."""
        evidence = self._repository.get_evidence(evidence_id)
        if evidence is None or evidence.investigation_id != investigation_id:
            raise EvidenceNotFoundError(investigation_id, evidence_id)
        return evidence

    def get_evidence_preview(
        self, investigation_id: str, evidence_id: str, max_chars: int = _DEFAULT_PREVIEW_CHARS
    ) -> EvidencePreview:
        evidence = self.get_evidence(investigation_id, evidence_id)
        text = evidence.raw_content or ""
        return EvidencePreview(
            id=evidence.id,
            evidence_type=evidence.evidence_type,
            title=evidence.title,
            source=evidence.source,
            created_at=evidence.created_at,
            preview_text=text[:max_chars],
            truncated=len(text) > max_chars,
            full_length=len(text),
        )

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
    ) -> InvestigationDetailSummary:
        """Saves engineer-entered case metadata for the persistent Summary
        Card (Phase 2A). Only fields explicitly passed are updated --
        pass only what changed, existing values for other fields are
        preserved (see the router for how partial updates are built).
        Returns the lightweight summary (Investigation loading redesign)
        -- the Summary Card never needed full evidence hydration here
        either."""
        self._ensure_exists(investigation_id)
        fields = {
            "customer": customer,
            "product": product,
            "version": version,
            "technology": technology,
            "assigned_engineer": assigned_engineer,
        }
        self._repository.update_details(investigation_id, **{k: v for k, v in fields.items() if v is not None})
        return self.get_investigation_summary(investigation_id)

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
        if not self._repository.exists(investigation_id):
            raise InvestigationNotFoundError(investigation_id)
