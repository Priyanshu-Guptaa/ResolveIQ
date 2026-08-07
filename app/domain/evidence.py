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
from app.domain.enums import DocumentStatus, EvidenceType
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
    content_hash: str | None = None
    """SHA-256 of ``raw_content``, set at write time -- the duplicate-
    upload guard's lookup key (found necessary after a real retried
    zip upload created 67 duplicate rows; see InvestigationEngine's
    add_text_evidence/add_file_evidence). None for evidence with empty
    content (an unsupported binary file, say) -- those are deliberately
    never deduplicated, since an empty hash can't distinguish two
    genuinely different textless files. None also for any row that
    predates this field (an existing DB's evidence is never backfilled
    with a fabricated hash -- it simply never matches as a duplicate,
    which is the safe default)."""


class EvidencePreview(BaseModel):
    """On-demand content preview for one piece of evidence -- the shape
    ``GET /investigations/{id}/evidence/{evidence_id}/preview`` returns.
    Capped so a client can always ask "what does this look like" without
    risking a multi-megabyte transfer; ``truncated``/``full_length`` tell
    the caller whether there's more (fetch the full ``Evidence`` via the
    sibling non-preview endpoint if genuinely needed)."""

    id: str
    evidence_type: EvidenceType
    title: str
    source: str
    created_at: datetime
    preview_text: str
    truncated: bool
    full_length: int


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
    """A knowledge-base document -- the Knowledge Management module's
    unit of content (Sprint 3, Phase 3.2). Fully governed: real extracted
    text, real lifecycle, real Component Registry links.

    History: Phase 3.1 stored *metadata only* here, deferring content
    storage ("the actual document storage mechanism can remain unchanged
    for now"). Phase 3.2 is that "later" -- ``content`` is now the real
    extracted text (from the Evidence Ingestion Pipeline for uploads, or
    backfilled from the original sample JSON for the seven pre-existing
    sample documents, see ``seed_migration.py``), and this is the only
    source the Knowledge Engine indexes from -- the JSON file is no
    longer read at all past initial backfill.
    """

    id: str
    title: str
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    source: str = "sample"
    """Where this came from: 'sample' (seed data), 'upload' (an admin's
    file upload), or later a connector name."""

    # --- Classification (Phase 3.2 "Document Details") ---------------------
    product: str | None = None
    version: str | None = None
    technology: str | None = None
    related_components: list[str] = Field(default_factory=list)
    """Component Registry names this document is linked to via a real
    foreign-key association table (``documentation_components``) --
    same pattern as KnownBugRecord/HistoricalInvestigationRecord.
    Editable by an administrator during the Metadata step; not
    auto-matched against uploaded content (that would be a form of
    inference this phase deliberately doesn't add -- an admin picks the
    component explicitly, same "no free text where a real relationship
    exists" discipline, just human-driven instead of string-matched)."""

    # --- Lifecycle -----------------------------------------------------
    status: DocumentStatus = DocumentStatus.DRAFT
    """Only PUBLISHED documents are indexed/searchable -- see
    KnowledgeManagementEngine.publish_document()."""

    # --- Upload provenance (empty for pre-existing sample documents) -------
    original_filename: str | None = None
    file_type: str | None = None
    file_path: str | None = None
    """Relative path under the configured upload directory where the
    original uploaded bytes are kept (for provenance/audit -- not read
    back by search, which only ever uses ``content``)."""


class DocumentationListItem(GovernanceFields):
    """Lightweight shape for the Knowledge Library's list/search view --
    everything a list row needs, deliberately *without* ``content``.

    Same reasoning as ``InvestigationListItem`` (Phase 1.5): a library
    of documents never needs every row's full extracted text just to
    render a table, and fetching it would be the exact unbounded-
    hydration mistake that phase fixed for investigations. See Phase
    3.2's explicit "avoid loading document content unless requested."
    """

    id: str
    title: str
    tags: list[str] = Field(default_factory=list)
    source: str = "sample"
    product: str | None = None
    version: str | None = None
    technology: str | None = None
    related_components: list[str] = Field(default_factory=list)
    status: DocumentStatus = DocumentStatus.DRAFT


class KnownBugRecord(GovernanceFields):
    """A known-bug record the Knowledge Engine indexes.

    Governed table since Sprint 3 Phase 3.1 (``known_bugs``) -- previously
    a static seed JSON file, see ``seed_migration.py``.
    """

    id: str
    title: str
    description: str
    bug_status: str = "open"
    """The bug's own state -- 'open', 'fixed-in-2.3.2', etc. Renamed
    from ``status`` in Phase 3.4: ``GovernanceFields`` now defines
    ``status`` for the *governance* lifecycle (Draft/Published/Archived/
    ...) shared by every knowledge object, and this field meant
    something entirely different (was it "open" was silently shadowing
    the mixin's field before this rename -- Phase 3.4 found this
    collision while generalizing lifecycle status onto every entity).
    ``known_bugs.json``'s "status" key still maps here; only the Python
    attribute name changed, see seed_migration.py."""
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
