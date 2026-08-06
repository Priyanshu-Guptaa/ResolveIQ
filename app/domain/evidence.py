"""The Evidence model -- the unifying concept of ResolveIQ.

Per the product philosophy: logs, tasks, wiki pages, bug reports, SQL
results, and manual notes all become Evidence. Every engine reasons over
Evidence, never over raw files. Source-specific detail (e.g. parsed
``LogEvent`` lines) lives in ``extra`` so the base shape stays uniform.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.domain.entities import ExtractedEntity, LogEvent
from app.domain.enums import EvidenceType


def _new_id() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Evidence(BaseModel):
    """A single piece of evidence attached to an investigation.

    ``raw_content`` holds the original text (task description, note, log
    file content, ...). ``extracted_entities`` is populated by the Log
    Intelligence Engine (or, for non-log evidence, by lightweight text
    scanning). ``log_events`` is populated only for ``LOG_FILE`` evidence.
    """

    id: str = Field(default_factory=_new_id)
    investigation_id: str
    evidence_type: EvidenceType
    source: str = "manual"
    """Where this evidence came from, e.g. 'servicenow', 'upload', 'manual'.
    In later sprints this becomes the connector name."""
    title: str = ""
    raw_content: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    extracted_entities: list[ExtractedEntity] = Field(default_factory=list)
    log_events: list[LogEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HistoricalInvestigationRecord(BaseModel):
    """A closed, resolved investigation loaded from sample knowledge (and,
    in future sprints, from ServiceNow/ADO exports). This is what the
    Knowledge Engine indexes and the Recommendation Engine matches against.
    """

    id: str
    title: str
    description: str
    root_cause: str
    resolution: str
    next_step: str = ""
    tags: list[str] = Field(default_factory=list)
    domain: str = "general"
    """Free-text domain tag, e.g. 'sql-server', 'kafka', 'meter-comm',
    'kubernetes' -- used for display, not a hard filter."""


class DocumentationRecord(BaseModel):
    """A knowledge-base / playbook snippet the Knowledge Engine indexes."""

    id: str
    title: str
    content: str
    tags: list[str] = Field(default_factory=list)
    source: str = "sample"


class KnownBugRecord(BaseModel):
    """A known-bug record the Knowledge Engine indexes."""

    id: str
    title: str
    description: str
    status: str = "open"
    affected_components: list[str] = Field(default_factory=list)
    workaround: Optional[str] = None
