"""TaskImporter -- the one shared import pipeline every source format's
extractor feeds into (see ``extractors.py``).

Deliberately reuses existing machinery rather than introducing anything
parallel to it:
 - Persistence, versioning, and search indexing all go through
   ``KnowledgeObjectService.create``/``edit_metadata`` (Sprint 3, Phase
   3.4) -- exactly what any other Historical Investigation write does,
   through the Knowledge Engine's existing Chroma indexing. No second
   indexing implementation exists here.
 - Component relationships go through ``KnowledgeRelationshipEngine.
   add_relationship`` (Phase 3.3) -- "no orphan relationships" and
   duplicate-relationship rejection are enforced there, not
   reimplemented.

Duplicate detection: every ``TaskRecord`` carries a ``ticket_number``.
Before importing, existing Historical Investigation records are
scanned for a ``ticket:<number>`` tag (the same tag format the import
itself writes) to build a ticket -> record lookup. A ticket not seen
before is a fresh Import; a ticket seen before whose title/description/
resolution are unchanged is a Duplicate (no-op); a ticket seen before
whose content differs is an Update. This makes every import idempotent
-- re-running the same file, or importing a second export that
overlaps an earlier one, never creates duplicate records.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.domain.evidence import HistoricalInvestigationRecord
from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.task_import import TaskImportSummary, TaskRecord

if TYPE_CHECKING:
    from app.engines.knowledge_object_framework.service import KnowledgeObjectService
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine

logger = logging.getLogger(__name__)

_NO_ROOT_CAUSE_CAPTURED = ""
"""Deliberately empty, not a placeholder sentence. An earlier version of
this importer used a boilerplate "not separately captured" sentence
here -- ``RecommendationEngine._build_root_causes`` (see
app/engines/recommendation/engine.py) takes the top similarity matches
and surfaces their ``root_cause`` field VERBATIM as a "Likely Root
Cause" finding; a repeated boilerplate sentence showed up in the
Investigation Workspace as if it were two distinct diagnosed causes,
which is actively misleading, not just uninformative. Leaving this
empty makes that engine's own ``if not root_cause: continue`` guard
correctly skip these tickets as a root-cause source -- they still fully
participate in similarity ranking and show up in Historical Matches
(based on title/description/resolution/tags), just never contribute a
fabricated "Likely Root Cause" bullet, since the source data genuinely
never captured one."""


def _ticket_from_tags(tags: list[str]) -> str | None:
    for tag in tags:
        if tag.startswith("ticket:"):
            return tag[len("ticket:") :]
    return None


class TaskImporter:
    def __init__(self, service: "KnowledgeObjectService", relationship_engine: "KnowledgeRelationshipEngine") -> None:
        self._service = service
        self._relationships = relationship_engine

    def import_records(self, records: list[TaskRecord], *, source_label: str, actor: str | None = None) -> TaskImportSummary:
        summary = TaskImportSummary(source_label=source_label, total_rows=len(records))
        existing_by_ticket = self._existing_by_ticket()
        components = self._service.list_all(KnowledgeObjectType.COMPONENT)

        for record in records:
            try:
                existing = existing_by_ticket.get(record.ticket_number)
                tags = self._tags_for(record)

                if existing is None:
                    created = self._service.create(
                        KnowledgeObjectType.HISTORICAL_INVESTIGATION,
                        created_by=actor,
                        title=record.title,
                        description=record.description,
                        resolution=record.resolution,
                        root_cause=_NO_ROOT_CAUSE_CAPTURED,
                        domain="general",
                        tags=tags,
                    )
                    summary.imported += 1
                    summary.imported_ids.append(created.id)
                    summary.relationships_created += self._link_components(created.id, record, components, actor)
                elif self._content_changed(existing, record):
                    updated = self._service.edit_metadata(
                        KnowledgeObjectType.HISTORICAL_INVESTIGATION,
                        existing.id,
                        updated_by=actor,
                        title=record.title,
                        description=record.description,
                        resolution=record.resolution,
                        tags=tags,
                    )
                    summary.updated += 1
                    summary.updated_ids.append(updated.id)
                    summary.relationships_created += self._link_components(updated.id, record, components, actor)
                else:
                    summary.duplicates += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not abort the whole import
                summary.failed += 1
                summary.errors.append(f"{record.ticket_number}: {exc}")
                logger.warning("Task import failed for ticket %s: %s", record.ticket_number, exc)

        logger.info(
            "Task import (%s): %d total, %d imported, %d updated, %d duplicates, %d failed, %d relationships",
            source_label, summary.total_rows, summary.imported, summary.updated,
            summary.duplicates, summary.failed, summary.relationships_created,
        )
        return summary

    def _existing_by_ticket(self) -> dict[str, HistoricalInvestigationRecord]:
        lookup: dict[str, HistoricalInvestigationRecord] = {}
        for record in self._service.list_all(KnowledgeObjectType.HISTORICAL_INVESTIGATION):
            ticket = _ticket_from_tags(record.tags)
            if ticket:
                lookup[ticket] = record
        return lookup

    @staticmethod
    def _content_changed(existing: HistoricalInvestigationRecord, record: TaskRecord) -> bool:
        return (
            existing.title != record.title
            or existing.description != record.description
            or existing.resolution != record.resolution
        )

    @staticmethod
    def _tags_for(record: TaskRecord) -> list[str]:
        tags = [f"ticket:{record.ticket_number}"]
        if record.priority:
            tags.append(f"priority:{record.priority}")
        if record.state:
            tags.append(f"state:{record.state}")
        tags.extend(record.extra_tags)
        return tags

    def _link_components(self, investigation_id: str, record: TaskRecord, components: list, actor: str | None) -> int:
        """Word-boundary (not plain substring) match against real
        Component Registry names in the ticket's title+description --
        cheap and conservative; a short name like "NMS" won't
        false-positive inside an unrelated word. Duplicate-relationship
        rejection (already linked -- e.g. a re-run) is caught and simply
        not recounted, not treated as a failure."""
        from app.engines.knowledge_relationships.engine import DuplicateRelationshipError, KnowledgeObjectNotFoundError

        haystack = f"{record.title} {record.description}"
        created = 0
        for component in components:
            if not re.search(rf"\b{re.escape(component.name)}\b", haystack, re.IGNORECASE):
                continue
            try:
                self._relationships.add_relationship(
                    KnowledgeObjectType.HISTORICAL_INVESTIGATION,
                    investigation_id,
                    KnowledgeObjectType.COMPONENT,
                    component.id,
                    RelationshipType.RELATED_TO,
                    created_by=actor,
                )
                created += 1
            except DuplicateRelationshipError:
                pass
            except KnowledgeObjectNotFoundError:
                logger.warning("Component %s vanished mid-import -- skipping relationship", component.id)
        return created
