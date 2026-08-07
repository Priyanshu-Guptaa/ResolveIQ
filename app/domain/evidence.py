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
from app.domain.governance import GovernanceFields


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


class HistoricalInvestigationRecord(GovernanceFields):
    """A closed, resolved investigation loaded from sample knowledge (and,
    in future sprints, from ServiceNow/ADO exports). This is what the
    Knowledge Engine indexes and the Recommendation Engine matches against.

    Governed table since Sprint 3 Phase 3.1 (``historical_investigations``)
    -- previously a static seed JSON file, see ``seed_migration.py``.
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
    related_components: list[str] = Field(default_factory=list)
    """Component Registry names this investigation is linked to via a real
    foreign-key association table (``historical_investigation_components``)
    -- not free text. Populated at migration time by an exact,
    normalized-name match against ``component_profiles``; may legitimately
    be empty when no component was mentioned. Distinct from ``tags``,
    which stays free text."""


class DocumentationRecord(GovernanceFields):
    """A knowledge-base / playbook snippet the Knowledge Engine indexes.

    Sprint 3 Phase 3.1 note: only *metadata* (title/tags/source +
    governance fields) is governed in a database table
    (``documentation``) this phase -- ``content`` keeps coming from the
    sample JSON file / whatever produced this record, per the explicit
    Phase 3.1 scope ("the actual document storage mechanism can remain
    unchanged for now"). ``content`` is empty when this record was
    constructed from the metadata-only table rather than the JSON file.
    """

    id: str
    title: str
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str = "sample"


class KnownBugRecord(GovernanceFields):
    """A known-bug record the Knowledge Engine indexes.

    Governed table since Sprint 3 Phase 3.1 (``known_bugs``) -- previously
    a static seed JSON file, see ``seed_migration.py``.
    """

    id: str
    title: str
    description: str
    status: str = "open"
    affected_components: list[str] = Field(default_factory=list)
    """Original free-text component/service names, preserved exactly as
    authored -- unchanged by the Sprint 3 migration."""
    workaround: Optional[str] = None
    related_components: list[str] = Field(default_factory=list)
    """The subset of ``affected_components`` (if any) that exactly
    matches a real Component Registry entry, resolved via a foreign-key
    association table (``known_bug_components``). See
    ``HistoricalInvestigationRecord.related_components`` for the same
    pattern."""
