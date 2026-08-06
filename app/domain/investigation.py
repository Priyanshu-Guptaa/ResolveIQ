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
from app.domain.enums import InvestigationStatus
from app.domain.evidence import Evidence


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
        """Flattened text representation of the whole investigation so far,
        used as the query for semantic search against historical knowledge.
        """
        parts: list[str] = [self.title]
        for item in self.evidence:
            if item.raw_content:
                parts.append(item.raw_content)
        return "\n".join(parts)

    def add_evidence(self, evidence: Evidence) -> None:
        self.evidence.append(evidence)
        self.updated_at = _utcnow()


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
