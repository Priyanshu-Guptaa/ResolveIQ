"""One-time (idempotent) migration of static seed data into governed
database tables -- Sprint 3, Phase 3.1 (Knowledge Foundation & Data
Model Migration), extended in Phase 3.2 (Knowledge Management).

Five sources move from JSON files / a Python constant into SQLite:

    component_profiles.json        -> component_profiles
    known_bugs.json                -> known_bugs
    historical_investigations.json -> historical_investigations
    documentation.json             -> documentation
    QUERY_LIBRARY (code constant)  -> sql_templates

Four of the five migrations only populate an *empty* table -- if a table
already has rows, the function logs that it skipped and returns 0. That
makes the whole thing safe to call on every app startup (same idiom
already used by ``KnowledgeEngine.seed_from_directory``'s ``force``
guard against re-seeding a non-empty ChromaDB collection): re-running
never duplicates rows, and there is no separate "have we migrated"
flag to keep in sync with reality.

Documentation is the one exception, and deliberately so: Phase 3.1 only
migrated *metadata* (content stayed on the JSON file), so the seven rows
it created already exist with an empty ``content`` column in any
database that ran that migration. Phase 3.2 needs that content -- so
``migrate_documentation`` is idempotent *per row*, not per table: an
existing row with empty content gets backfilled (content, product/
version/technology left None, status set to PUBLISHED so these already-
"live" sample documents don't silently disappear from search), a row
that already has content (an administrator's real edit) is never
touched, and a row that doesn't exist yet is created fresh. Re-running
is still always safe.

Relationship linking (KnownBug/SqlTemplate/HistoricalInvestigation ->
Component) is deterministic, not fuzzy: every match is either an exact
normalized-string match (hyphens/underscores/spaces/case stripped, so
"command-processor-host" matches "CommandProcessorHost") or an exact
substring match of a real component name in free text. No component
match is invented -- today's sample data was authored before the
Component Registry existed, so several of these association tables
legitimately import empty. That's a correct result, not a bug; see the
Phase 3.1 report for exactly which links exist after migration.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.domain.enums import DocumentStatus
from app.domain.evidence import DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
from app.domain.product_intelligence import ComponentProfile
from app.domain.sql_studio import QUERY_LIBRARY, QueryTemplate
from app.infrastructure.db.component_repository import ComponentProfileRepository
from app.infrastructure.db.knowledge_repository import KnowledgeRepository
from app.infrastructure.db.sql_template_repository import SqlTemplateRepository

logger = logging.getLogger(__name__)

_MIGRATION_ACTOR = "system:migration"


def _normalize_component_key(value: str) -> str:
    """Deterministic, no ML: lowercase, strip separators. Lets
    "command-processor-host" (a SQL Studio tag) match the Component
    Registry's "CommandProcessorHost"."""
    return value.lower().replace("-", "").replace("_", "").replace(" ", "")


def _match_by_normalized_key(candidates: list[str], component_names: list[str]) -> list[str]:
    by_key = {_normalize_component_key(name): name for name in component_names}
    matched = []
    for candidate in candidates:
        real_name = by_key.get(_normalize_component_key(candidate))
        if real_name and real_name not in matched:
            matched.append(real_name)
    return matched


def _match_by_substring(text: str, component_names: list[str]) -> list[str]:
    """Exact, case-sensitive substring match of a component name inside
    free text -- the same technique already used by the Investigation
    Workspace's Architecture Explorer and the Investigation Intelligence
    Engine design to find a component mentioned in prose."""
    return [name for name in component_names if name in text]


def migrate_component_profiles(component_repo: ComponentProfileRepository, sample_knowledge_dir: Path) -> int:
    if component_repo.count() > 0:
        logger.info("component_profiles already populated -- skipping migration")
        return 0
    path = sample_knowledge_dir / "component_profiles.json"
    if not path.exists():
        logger.warning("Migration source not found: %s", path)
        return 0

    raw: list[dict[str, Any]] = json.loads(path.read_text())
    imported = 0
    for item in raw:
        profile = ComponentProfile(**item, created_by=_MIGRATION_ACTOR, updated_by=_MIGRATION_ACTOR)
        component_repo.save(profile)
        imported += 1
    logger.info("Migrated %d component profile(s) from %s", imported, path.name)
    return imported


def migrate_known_bugs(
    knowledge_repo: KnowledgeRepository, component_repo: ComponentProfileRepository, sample_knowledge_dir: Path
) -> int:
    if knowledge_repo.count_known_bugs() > 0:
        logger.info("known_bugs already populated -- skipping migration")
        return 0
    path = sample_knowledge_dir / "known_bugs.json"
    if not path.exists():
        logger.warning("Migration source not found: %s", path)
        return 0

    component_names = [c.name for c in component_repo.list_all(active_only=False)]
    raw: list[dict[str, Any]] = json.loads(path.read_text())
    imported = 0
    for item in raw:
        related = _match_by_normalized_key(item.get("affected_components", []), component_names)
        record = KnownBugRecord(
            **item, related_components=related, created_by=_MIGRATION_ACTOR, updated_by=_MIGRATION_ACTOR
        )
        knowledge_repo.save_known_bug(record)
        imported += 1
    logger.info("Migrated %d known bug(s) from %s", imported, path.name)
    return imported


def migrate_historical_investigations(
    knowledge_repo: KnowledgeRepository, component_repo: ComponentProfileRepository, sample_knowledge_dir: Path
) -> int:
    if knowledge_repo.count_historical_investigations() > 0:
        logger.info("historical_investigations already populated -- skipping migration")
        return 0
    path = sample_knowledge_dir / "historical_investigations.json"
    if not path.exists():
        logger.warning("Migration source not found: %s", path)
        return 0

    component_names = [c.name for c in component_repo.list_all(active_only=False)]
    raw: list[dict[str, Any]] = json.loads(path.read_text())
    imported = 0
    for item in raw:
        haystack = f"{item.get('title', '')}\n{item.get('description', '')}"
        related = _match_by_substring(haystack, component_names)
        record = HistoricalInvestigationRecord(
            **item, related_components=related, created_by=_MIGRATION_ACTOR, updated_by=_MIGRATION_ACTOR
        )
        knowledge_repo.save_historical_investigation(record)
        imported += 1
    logger.info("Migrated %d historical investigation(s) from %s", imported, path.name)
    return imported


def migrate_documentation(
    knowledge_repo: KnowledgeRepository, component_repo: ComponentProfileRepository, sample_knowledge_dir: Path
) -> int:
    """Idempotent per row, not per table -- see this module's docstring
    for why documentation is the one exception. Returns the number of
    rows created or backfilled this call (0 once every sample document
    already has content, i.e. every subsequent startup)."""
    path = sample_knowledge_dir / "documentation.json"
    if not path.exists():
        logger.warning("Migration source not found: %s", path)
        return 0

    component_names = [c.name for c in component_repo.list_all(active_only=False)]
    raw: list[dict[str, Any]] = json.loads(path.read_text())
    changed = 0
    for item in raw:
        existing = knowledge_repo.get_documentation(item["id"])
        if existing is not None and existing.content:
            continue  # real content already present -- never overwrite an admin's edit

        content = item.get("content", "")
        haystack = f"{item['title']}\n{content}"
        related = _match_by_substring(haystack, component_names)

        if existing is None:
            record = DocumentationRecord(
                id=item["id"],
                title=item["title"],
                content=content,
                tags=item.get("tags", []),
                source=item.get("source", "sample"),
                related_components=related,
                status=DocumentStatus.PUBLISHED,
                created_by=_MIGRATION_ACTOR,
                updated_by=_MIGRATION_ACTOR,
            )
        else:
            # Row exists from Phase 3.1's metadata-only migration --
            # backfill content/relationships, leave everything else
            # (including any admin edits to title/tags) untouched.
            record = existing.model_copy(
                update={"content": content, "related_components": related, "updated_by": _MIGRATION_ACTOR}
            )
        knowledge_repo.save_documentation(record)
        changed += 1
    logger.info("Migrated/backfilled %d documentation row(s) from %s", changed, path.name)
    return changed


def migrate_sql_templates(sql_repo: SqlTemplateRepository, component_repo: ComponentProfileRepository) -> int:
    """Source is the ``QUERY_LIBRARY`` Python constant, not a JSON file
    -- there never was one for SQL Studio."""
    if sql_repo.count() > 0:
        logger.info("sql_templates already populated -- skipping migration")
        return 0

    component_names = [c.name for c in component_repo.list_all(active_only=False)]
    imported = 0
    for template in QUERY_LIBRARY:
        related = _match_by_normalized_key(template.tags, component_names)
        record = QueryTemplate(
            id=template.id,
            title=template.title,
            category=template.category,
            sql_text=template.sql_text,
            explanation=template.explanation,
            tags=template.tags,
            related_components=related,
            created_by=_MIGRATION_ACTOR,
            updated_by=_MIGRATION_ACTOR,
        )
        sql_repo.save(record)
        imported += 1
    logger.info("Migrated %d SQL template(s) from QUERY_LIBRARY", imported)
    return imported


def migrate_all(
    *,
    component_repo: ComponentProfileRepository,
    knowledge_repo: KnowledgeRepository,
    sql_repo: SqlTemplateRepository,
    sample_knowledge_dir: Path,
) -> dict[str, int]:
    """Runs every migration in dependency order (components first --
    the other four link against them) and returns a per-table count of
    rows actually imported/changed this call -- 0 for any table that was
    already fully populated, except ``documentation`` which reports a
    per-row content backfill count (see ``migrate_documentation``).
    Safe to call on every startup."""
    report = {
        "component_profiles": migrate_component_profiles(component_repo, sample_knowledge_dir),
    }
    report["known_bugs"] = migrate_known_bugs(knowledge_repo, component_repo, sample_knowledge_dir)
    report["historical_investigations"] = migrate_historical_investigations(
        knowledge_repo, component_repo, sample_knowledge_dir
    )
    report["documentation"] = migrate_documentation(knowledge_repo, component_repo, sample_knowledge_dir)
    report["sql_templates"] = migrate_sql_templates(sql_repo, component_repo)
    logger.info("Knowledge foundation migration complete: %s", report)
    return report
