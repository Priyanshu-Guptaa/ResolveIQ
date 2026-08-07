"""Structured Task Import (post-Sprint-3-Phase-3.4 follow-up).

One canonical intermediate shape (``TaskRecord``) that every source
format (ServiceNow JSON export, ServiceNow XLSX report export, future
formats) maps into, so the actual import logic
(``app.engines.task_import.importer.TaskImporter``) is written once and
shared -- format-specific code only has to extract rows into this
shape, never touch persistence, duplicate detection, or indexing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskRecord(BaseModel):
    """One resolved support ticket, normalized from whatever source
    format it came from. ``ticket_number`` is the stable identifier
    duplicate detection keys on -- every source format is expected to
    have some real, stable ticket/task number; there is no synthetic
    fallback id, because a fabricated one couldn't be matched against a
    re-import."""

    ticket_number: str
    title: str
    description: str
    resolution: str
    priority: str = ""
    state: str = ""
    extra_tags: list[str] = Field(default_factory=list)
    """Additional source-specific traceability tags (assignment group,
    location, assignee, CI reference, ...) -- appended to the standard
    ticket/priority/state tags, never used for dedup matching."""


class TaskImportSummary(BaseModel):
    """What the admin sees after running an import -- every number here
    must add up: imported + updated + duplicates + failed == total_rows
    (rows skipped for having no resolution content are not counted in
    total_rows at all; they never became TaskRecords in the first
    place, see each extractor's docstring)."""

    source_label: str
    total_rows: int = 0
    imported: int = 0
    updated: int = 0
    duplicates: int = 0
    failed: int = 0
    relationships_created: int = 0
    errors: list[str] = Field(default_factory=list)
    imported_ids: list[str] = Field(default_factory=list)
    updated_ids: list[str] = Field(default_factory=list)
