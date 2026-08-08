"""LogWikiImporter -- turns an extracted wiki page (``extractor.extract``)
into governed ``LogSourceApplication``/``LogCollectionScenario`` records.

Deliberately reuses existing machinery, the same discipline
``app.engines.task_import.importer.TaskImporter`` established:
 - Persistence and versioning go through ``KnowledgeObjectService.create``/
   ``edit_metadata`` (Sprint 3, Phase 3.4) -- the same path any other
   governed knowledge write uses. Log Intelligence records are not
   semantically indexed (they aren't in ``KnowledgeObjectService.
   _INDEXED_TYPES``): they're matched deterministically by technology/
   name, not by embedding similarity, per the explicit "no LLM reasoning
   in this phase" instruction.
 - Component relationships go through ``KnowledgeRelationshipEngine.
   add_relationship`` -- "no orphan relationships" and duplicate-
   relationship rejection are enforced there, not reimplemented.

Idempotent, name-keyed upsert (not create-only): re-importing the same
page, or importing a second page that redescribes an already-known
application, enriches the existing record instead of duplicating it --
required so future wiki imports (Component architecture, Application
responsibilities, Known bugs, ...) can layer onto the same objects
without redesigning anything (explicit design requirement).

One id-remapping wrinkle worth calling out: ``extractor.build_records``
assigns each ``LogSourceApplication`` a *deterministic* id
(``log-src-<slug>``) purely so a scenario's steps can reference their
source before either is persisted. ``KnowledgeObjectService.create``
always mints its own fresh id (so every governed object's id is a real,
service-issued identity, not importer-chosen) -- so ``_upsert_sources``
returns a name -> real-persisted-id map, and every step's
``log_source_id`` is rewritten through that map before its scenario is
saved. This is exactly the map an idempotent upsert-by-name needs to
build anyway, so it costs nothing extra.

Component Registry matching (refinement #1, explicit): queries
``component_repo.list_all()`` fresh on every import -- never a
hardcoded snapshot of today's entries -- so the importer scales
automatically as the registry grows.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from app.domain.knowledge_relationships import KnowledgeObjectType, RelationshipType
from app.domain.log_intelligence_kb import LogCollectionScenario, LogSourceApplication, LogWikiImportSummary

if TYPE_CHECKING:
    from app.engines.knowledge_object_framework.service import KnowledgeObjectService
    from app.engines.knowledge_relationships.engine import KnowledgeRelationshipEngine
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.domain.product_intelligence import ComponentProfile

logger = logging.getLogger(__name__)

_MIN_MATCH_LEN = 4
"""Below this, a prefix match is too easy to hit by accident (e.g. a
3-letter log source name prefixing an unrelated component) -- exact
(case-insensitive) matches are unaffected by this floor."""


def _match_component(name: str, components: list["ComponentProfile"]) -> "ComponentProfile | None":
    """Deterministic name matching against the *complete* live Component
    Registry -- exact (case-insensitive) match first, then a prefix
    match in either direction (e.g. wiki's "CommandProcessor" ->
    registry's "CommandProcessorHost", or the reverse for an
    abbreviated wiki name). First match wins; the registry is small and
    curated enough that this hasn't produced an ambiguous double-match
    in practice, and exact matches are always tried across the whole
    registry before any prefix match is attempted."""
    name_lower = name.lower()
    for component in components:
        if component.name.lower() == name_lower:
            return component
    if len(name_lower) >= _MIN_MATCH_LEN:
        for component in components:
            if component.name.lower().startswith(name_lower):
                return component
    for component in components:
        if len(component.name) >= _MIN_MATCH_LEN and name_lower.startswith(component.name.lower()):
            return component
    return None


class LogWikiImporter:
    def __init__(
        self,
        service: "KnowledgeObjectService",
        relationship_engine: "KnowledgeRelationshipEngine",
        component_repo: "ComponentProfileRepository",
    ) -> None:
        self._service = service
        self._relationships = relationship_engine
        self._components = component_repo

    def import_page(
        self,
        sources: list[LogSourceApplication],
        scenarios: list[LogCollectionScenario],
        *,
        source_wiki_page: str,
        actor: str | None = None,
    ) -> LogWikiImportSummary:
        summary = LogWikiImportSummary(source_wiki_page=source_wiki_page)

        source_id_by_name = self._upsert_sources(sources, actor, summary)
        summary.component_relationships_created = self._link_components(source_id_by_name, actor)
        self._upsert_scenarios(scenarios, source_id_by_name, actor, summary)

        logger.info(
            "Log wiki import (%s): %d sources created, %d enriched, %d scenarios created, "
            "%d updated, %d component relationships",
            source_wiki_page, summary.sources_created, summary.sources_enriched,
            summary.scenarios_created, summary.scenarios_updated, summary.component_relationships_created,
        )
        return summary

    # --- LogSourceApplication ------------------------------------------

    def _upsert_sources(
        self, sources: list[LogSourceApplication], actor: str | None, summary: LogWikiImportSummary
    ) -> dict[str, str]:
        """Returns name -> real persisted id, for remapping scenario
        steps (see module docstring)."""
        source_id_by_name: dict[str, str] = {}
        for source in sources:
            try:
                existing = self._find_source(source.name)
                if existing is None:
                    created = self._service.create(
                        KnowledgeObjectType.LOG_SOURCE_APPLICATION,
                        created_by=actor,
                        **source.model_dump(exclude={"id", "created_at", "updated_at", "created_by", "updated_by", "status", "is_active"}),
                    )
                    source_id_by_name[source.name] = created.id
                    summary.sources_created += 1
                else:
                    merged_fields = self._merge_source_fields(existing, source)
                    updated = self._service.edit_metadata(
                        KnowledgeObjectType.LOG_SOURCE_APPLICATION, existing.id, updated_by=actor, **merged_fields
                    )
                    source_id_by_name[source.name] = updated.id
                    summary.sources_enriched += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not abort the whole import
                summary.errors.append(f"log source '{source.name}': {exc}")
                logger.warning("Log wiki import failed for source %s: %s", source.name, exc)
        return source_id_by_name

    def _find_source(self, name: str) -> LogSourceApplication | None:
        for existing in self._service.list_all(KnowledgeObjectType.LOG_SOURCE_APPLICATION):
            if existing.name == name:
                return existing
        return None

    @staticmethod
    def _merge_source_fields(existing: LogSourceApplication, new: LogSourceApplication) -> dict:
        """"Enrich, don't clobber": list fields union; scalar text
        fields already set on the existing record (e.g. a hand-edited
        ``purpose``) are preserved rather than overwritten by an empty
        or redundant re-import value."""
        notes = existing.notes
        if new.notes and (notes is None or new.notes not in notes):
            notes = f"{notes}; {new.notes}" if notes else new.notes

        location = existing.location.model_copy(
            update={
                "filename_patterns": _union(existing.location.filename_patterns, new.location.filename_patterns),
                "raw_paths": _union(existing.location.raw_paths, new.location.raw_paths),
            }
        )
        if existing.location.platform in (None, "unknown") and new.location.platform not in (None, "unknown"):
            location.platform = new.location.platform
        if not existing.location.subdirectory and new.location.subdirectory:
            location.subdirectory = new.location.subdirectory

        return {
            "location": location,
            "technology": _union(existing.technology, new.technology),
            "product": existing.product or new.product,
            "purpose": existing.purpose or new.purpose,
            "log_level_support": _union(existing.log_level_support, new.log_level_support),
            "typical_issues": _union(existing.typical_issues, new.typical_issues),
            "common_errors": _union(existing.common_errors, new.common_errors),
            "related_sql": _union(existing.related_sql, new.related_sql),
            "related_documentation": _union(existing.related_documentation, new.related_documentation),
            "related_known_bugs": _union(existing.related_known_bugs, new.related_known_bugs),
            "related_playbooks": _union(existing.related_playbooks, new.related_playbooks),
            "notes": notes,
            "source_wiki_pages": _union(existing.source_wiki_pages, new.source_wiki_pages),
        }

    # --- Component Registry linking -------------------------------------

    def _link_components(self, source_id_by_name: dict[str, str], actor: str | None) -> int:
        """Deterministic name matching against the *complete* live
        registry (refinement #1) -- ``component_repo.list_all()`` is
        called fresh here, never a hardcoded list, so newly-registered
        components automatically become matchable on the next import
        without any importer change."""
        from app.engines.knowledge_relationships.engine import DuplicateRelationshipError, KnowledgeObjectNotFoundError

        components = self._components.list_all()
        created = 0
        for name, source_id in source_id_by_name.items():
            match = _match_component(name, components)
            if match is None:
                continue
            try:
                self._relationships.add_relationship(
                    KnowledgeObjectType.LOG_SOURCE_APPLICATION,
                    source_id,
                    KnowledgeObjectType.COMPONENT,
                    match.id,
                    RelationshipType.IMPLEMENTS_LOGGING_FOR,
                    created_by=actor,
                )
                created += 1
            except DuplicateRelationshipError:
                pass
            except KnowledgeObjectNotFoundError:
                logger.warning("Component %s vanished mid-import -- skipping relationship", match.id)
        return created

    # --- LogCollectionScenario -------------------------------------------

    def _upsert_scenarios(
        self,
        scenarios: list[LogCollectionScenario],
        source_id_by_name: dict[str, str],
        actor: str | None,
        summary: LogWikiImportSummary,
    ) -> None:
        for scenario in scenarios:
            try:
                remapped_steps = [
                    step.model_copy(update={"log_source_id": source_id_by_name.get(step.component_name, step.log_source_id)})
                    for step in scenario.steps
                ]
                existing = self._service.list_all(KnowledgeObjectType.LOG_COLLECTION_SCENARIO)
                match = next(
                    (
                        s
                        for s in existing
                        if s.product == scenario.product
                        and s.technology == scenario.technology
                        and s.scenario_type == scenario.scenario_type
                        and s.region == scenario.region
                    ),
                    None,
                )
                if match is None:
                    created = self._service.create(
                        KnowledgeObjectType.LOG_COLLECTION_SCENARIO,
                        created_by=actor,
                        product=scenario.product,
                        technology=scenario.technology,
                        version=scenario.version,
                        scenario_type=scenario.scenario_type,
                        region=scenario.region,
                        steps=remapped_steps,
                        notes=scenario.notes,
                        source_wiki_page=scenario.source_wiki_page,
                    )
                    summary.scenarios_created += 1
                else:
                    self._service.edit_metadata(
                        KnowledgeObjectType.LOG_COLLECTION_SCENARIO,
                        match.id,
                        updated_by=actor,
                        steps=remapped_steps,
                        notes=scenario.notes or match.notes,
                        source_wiki_page=scenario.source_wiki_page,
                    )
                    summary.scenarios_updated += 1
            except Exception as exc:  # noqa: BLE001 -- one bad row must not abort the whole import
                summary.errors.append(f"scenario '{scenario.technology} / {scenario.scenario_type}': {exc}")
                logger.warning(
                    "Log wiki import failed for scenario %s / %s: %s", scenario.technology, scenario.scenario_type, exc
                )


def _union(existing: list[str], new: list[str]) -> list[str]:
    result = list(existing)
    for item in new:
        if item not in result:
            result.append(item)
    return result
