"""InvestigationSession -- the shared-context container.

Per the platform spec: "An investigation should maintain a shared context.
All modules should automatically use this context. The user should never
have to repeatedly enter the same information."

This model is that shared context. Every engine that touches an
investigation reads its evidence and merged entities from here rather than
asking the caller to re-supply them.
"""

from __future__ import annotations

import uuid
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from app.domain.entities import ExtractedEntity
from app.domain.enums import EvidenceType, InvestigationStatus
from app.domain.evidence import Evidence


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_MAX_CONTEXT_CHARS = 50_000
"""Cap for InvestigationSession.context_text -- see that property's
docstring."""


class InvestigationSession(BaseModel):
    """A single investigation an engineer is working, and everything known
    about it so far.

    ``evidence`` accumulates as the engineer pastes the task description,
    uploads logs, and adds notes. ``merged_entities`` is derived (not
    independently editable) -- it is the de-duplicated union of every
    entity extracted from every piece of evidence, which is what the
    Recommendation Engine actually reasons over.
    """

    id: str = Field(default_factory=_new_id)
    title: str
    status: InvestigationStatus = InvestigationStatus.OPEN
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    last_viewed_at: datetime | None = None
    """Set whenever the Dashboard/Workspace opens this investigation --
    backs the "Recently Viewed" panel (RFC rev 3, Dashboard)."""
    evidence: list[Evidence] = Field(default_factory=list)

    # --- Manually-entered case metadata (Phase 2A, persistent Summary Card) ---
    # These are engineer-entered, not derived -- Product Intelligence (Phase
    # 2B) and richer entity extraction may later suggest/prefill them, but
    # nothing today can compute them, so they start blank rather than
    # fabricated. All optional; the Summary Card shows "--" when unset.
    customer: str | None = None
    product: str | None = None
    version: str | None = None
    technology: str | None = None
    assigned_engineer: str | None = None

    @property
    def merged_entities(self) -> list[ExtractedEntity]:
        """De-duplicated entities across all evidence, most-frequent first.

        Frequency matters for recommendations: an entity (e.g. a
        correlation ID) that shows up in both the task description and an
        uploaded log is a much stronger investigation anchor than one that
        appears once.
        """
        counts: Counter[ExtractedEntity] = Counter()
        for item in self.evidence:
            counts.update(item.extracted_entities)
        return [entity for entity, _ in counts.most_common()]

    @property
    def context_text(self) -> str:
        """Flattened text representation of the whole investigation so
        far, used as the query for semantic search against historical
        knowledge.

        Capped at ``_MAX_CONTEXT_CHARS`` -- found necessary on a real
        investigation whose evidence (97 items, several multi-MB log
        files) produced a 40MB context_text, which made embedding it
        (the sentence-transformer model must tokenize the whole string
        before its own ~256-token window truncates it anyway) slow
        enough to reliably time out the Recommendation Engine's client.
        The model's effective window is a few hundred tokens (roughly
        1-2KB of text) -- 50,000 characters is already generous
        headroom, not a tight squeeze. Evidence contributes in creation
        order (title, then oldest evidence first) and the cap is
        enforced while accumulating, not by slicing an already-built
        40MB string -- so a still-relevant early piece of context (the
        original task description, say) is never crowded out by
        whichever evidence happened to be uploaded last, and the huge
        string is never actually built in memory in the first place.
        """
        parts: list[str] = [self.title]
        remaining = _MAX_CONTEXT_CHARS - len(self.title)
        for item in self.evidence:
            if remaining <= 0:
                break
            if not item.raw_content:
                continue
            chunk = item.raw_content[:remaining]
            parts.append(chunk)
            remaining -= len(chunk)
        return "\n".join(parts)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)
        self.updated_at = _utcnow()


class EvidenceSummary(BaseModel):
    """Lightweight evidence shape for the Explorer list and
    ``GET /investigations/{id}/evidence`` -- id/type/source/title/
    timestamp/counts only, never ``raw_content``/``extracted_entities``/
    ``log_events``. Found necessary after a real investigation's evidence
    (97 items, one legitimately 15.7MB) made the old embedded-evidence
    ``GET /investigations/{id}`` return 151MB of JSON. Content is fetched
    separately and only on demand -- see ``EvidencePreview`` (capped) and
    the full ``Evidence`` model (uncapped, single-item) for that."""

    id: str
    evidence_type: EvidenceType
    source: str
    title: str
    created_at: datetime
    file_kind: str | None = None
    """From upload metadata (log/docx/xlsx/pdf/image/...) -- None for
    evidence that was never a file upload (a manual note, the task
    description)."""
    content_length: int = 0
    """Character count of raw_content, computed at the database level
    (SQL LENGTH()) -- never by loading the content itself."""
    entity_count: int = 0
    log_event_count: int = 0


class EntityTypeSummary(BaseModel):
    """One row of the aggregated Findings view -- how many entities of
    this type were found across all evidence, and a capped sample of
    distinct values. Computed server-side once (in the lightweight
    summary query) instead of the client re-aggregating full per-evidence
    entity lists on every render."""

    entity_type: str
    count: int
    sample_values: list[str] = Field(default_factory=list)


class InvestigationDetailSummary(BaseModel):
    """The ``GET /investigations/{id}`` response -- everything the
    Workspace's Summary Card, evidence counts, and Findings tab need,
    with zero embedded evidence content. This is to a single
    investigation's detail view what ``InvestigationListItem`` already
    was to the list view (same Phase 1.5 "list views never need evidence
    content" reasoning, extended to the detail view once a real
    investigation's evidence made that endpoint return 151MB of JSON).

    Internal callers that genuinely need full evidence content (the
    Recommendation Engine, most notably) keep using
    ``InvestigationEngine.get_investigation`` (full hydration) directly
    -- this model is specifically the external, lightweight contract."""

    id: str
    title: str
    status: InvestigationStatus
    created_at: datetime
    updated_at: datetime
    last_viewed_at: datetime | None = None
    customer: str | None = None
    product: str | None = None
    version: str | None = None
    technology: str | None = None
    assigned_engineer: str | None = None
    evidence_count: int = 0
    evidence_count_by_type: dict[str, int] = Field(default_factory=dict)
    entity_summary: list[EntityTypeSummary] = Field(default_factory=list)
    last_activity_at: datetime | None = None
    """Most recent evidence's created_at -- a cheap "is this investigation
    still being actively worked" signal without fetching the full
    Timeline (``GET /investigations/{id}/timeline``, unchanged, still the
    source of truth for the actual chronological event list)."""


class InvestigationListItem(BaseModel):
    """Lightweight investigation shape for list views -- Dashboard rows,
    dropdowns, pickers.

    Phase 1.5 fix: list views were going through ``InvestigationSession``
    (full evidence hydration -- raw_content, extracted_entities, log_events
    for every piece of evidence, on every investigation) just to render a
    title and a count. List views never need evidence content, only a
    count of it, so this model -- and the repository query that produces
    it -- deliberately never touch the evidence table's JSON columns.
    """

    id: str
    title: str
    status: InvestigationStatus
    created_at: datetime
    updated_at: datetime
    last_viewed_at: datetime | None = None
    evidence_count: int = 0


class ActivityItem(BaseModel):
    """One entry in the Dashboard's Recent Activity feed / Recent Documents
    panel -- always derived from real Evidence rows, never fabricated.

    ``kind`` distinguishes "investigation created" from "evidence added" so
    the feed reads as a timeline even before the dedicated Investigation
    Timeline (RFC rev 3, Timeline) exists.
    """

    investigation_id: str
    investigation_title: str
    kind: str
    """"investigation_created" | "evidence_added" """
    label: str
    evidence_type: str | None = None
    occurred_at: datetime


class DashboardStats(BaseModel):
    """Investigation statistics for the Dashboard's KPI row -- computed
    from real InvestigationSession rows, no illustrative/fake numbers.
    """

    total_count: int = 0
    active_count: int = 0
    resolved_count: int = 0
    avg_resolution_hours: float | None = None
    """None (rendered as "--") until at least one investigation has
    status=resolved -- there is no fabricated default."""
