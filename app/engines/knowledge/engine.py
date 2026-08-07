"""Knowledge Engine: import + semantic search over historical
investigations, documentation, and known bugs.

Sprint 1 imported sample JSON files directly. Since Sprint 3 Phase 3.1,
historical investigations and known bugs are governed database tables
(``historical_investigations``, ``known_bugs`` -- see
``seed_migration.py``); this engine seeds ChromaDB *from those tables*
rather than re-parsing JSON, so an administrator's future edit (Phase
3.3+) is what search actually reflects. Documentation is unchanged this
phase -- its content still comes straight from the sample JSON file, per
Phase 3.1's explicit "storage mechanism can remain unchanged for now"
scope; only its metadata is separately mirrored into a governed table.

``repository`` is optional so existing callers/tests that construct
``KnowledgeEngine(store)`` directly (no database, no repository) keep
working unchanged -- they exercise search only, which never touches the
repository. Production wiring (``app/api/dependencies.py``) always
supplies one.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.enums import KnowledgeCollection
from app.domain.evidence import DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.knowledge_store import KnowledgeStore

if TYPE_CHECKING:
    from app.infrastructure.db.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)


class KnowledgeEngine:
    """Facade over the :class:`KnowledgeStore` for import and search."""

    def __init__(self, store: KnowledgeStore, repository: "KnowledgeRepository | None" = None) -> None:
        self._store = store
        self._repository = repository

    # --- Import ------------------------------------------------------------

    def seed_from_directory(self, sample_dir: Path, *, force: bool = False) -> dict[str, int]:
        """Seeds ChromaDB from each collection's source of truth.
        Idempotent by default: a Chroma collection already holding data
        is left alone unless ``force=True``. Returns a count of records
        seeded per collection, for logging/diagnostics.
        """
        loaded = {
            "historical_investigations": self._seed_historical_investigations(force=force),
            "documentation": self._seed_documentation(sample_dir / "documentation.json", force=force),
            "known_bugs": self._seed_known_bugs(force=force),
        }
        logger.info("Knowledge Engine seed complete: %s", loaded)
        return loaded

    def _seed_historical_investigations(self, *, force: bool) -> int:
        collection = KnowledgeCollection.HISTORICAL_INVESTIGATIONS
        if not force and self._store.count(collection) > 0:
            return 0
        if self._repository is None:
            logger.warning("No KnowledgeRepository wired -- skipping historical investigation seeding")
            return 0

        records = self._repository.list_historical_investigations()
        base_time = datetime.now(timezone.utc)
        for index, record in enumerate(records):
            searchable_text = (
                f"{record.title}\n{record.description}\nRoot cause: {record.root_cause}\n"
                f"Resolution: {record.resolution}\nTags: {', '.join(record.tags)}"
            )
            self._store.upsert(
                collection,
                record.id,
                searchable_text,
                record.title,
                metadata={
                    "root_cause": record.root_cause,
                    "resolution": record.resolution,
                    "next_step": record.next_step,
                    "tags": ", ".join(record.tags),
                    "domain": record.domain,
                    "imported_at": _seed_timestamp(base_time, index),
                },
            )
        return len(records)

    # --- Plain listing (Phase 3.1 -- ready for Phase 3.3+'s admin UI) ------

    def list_known_bugs(self) -> list[KnownBugRecord]:
        return self._repository.list_known_bugs() if self._repository else []

    def get_known_bug(self, bug_id: str) -> KnownBugRecord | None:
        return self._repository.get_known_bug(bug_id) if self._repository else None

    def list_historical_investigations(self) -> list[HistoricalInvestigationRecord]:
        return self._repository.list_historical_investigations() if self._repository else []

    def get_historical_investigation(self, record_id: str) -> HistoricalInvestigationRecord | None:
        return self._repository.get_historical_investigation(record_id) if self._repository else None

    def list_documentation_metadata(self) -> list[DocumentationRecord]:
        return self._repository.list_documentation_metadata() if self._repository else []

    def _seed_documentation(self, path: Path, *, force: bool) -> int:
        collection = KnowledgeCollection.DOCUMENTATION
        if not force and self._store.count(collection) > 0:
            return 0
        if not path.exists():
            logger.warning("Sample file not found: %s", path)
            return 0

        records = [DocumentationRecord(**raw) for raw in json.loads(path.read_text())]
        base_time = datetime.now(timezone.utc)
        for index, record in enumerate(records):
            searchable_text = f"{record.title}\n{record.content}\nTags: {', '.join(record.tags)}"
            self._store.upsert(
                collection,
                record.id,
                searchable_text,
                record.title,
                metadata={
                    "source": record.source,
                    "tags": ", ".join(record.tags),
                    "imported_at": _seed_timestamp(base_time, index),
                },
            )
        return len(records)

    def _seed_known_bugs(self, *, force: bool) -> int:
        collection = KnowledgeCollection.KNOWN_BUGS
        if not force and self._store.count(collection) > 0:
            return 0
        if self._repository is None:
            logger.warning("No KnowledgeRepository wired -- skipping known bug seeding")
            return 0

        records = self._repository.list_known_bugs()
        base_time = datetime.now(timezone.utc)
        for index, record in enumerate(records):
            searchable_text = (
                f"{record.title}\n{record.description}\n"
                f"Affected: {', '.join(record.affected_components)}\n"
                f"Workaround: {record.workaround or 'none'}"
            )
            self._store.upsert(
                collection,
                record.id,
                searchable_text,
                record.title,
                metadata={
                    "status": record.status,
                    "affected_components": ", ".join(record.affected_components),
                    "workaround": record.workaround or "",
                    "imported_at": _seed_timestamp(base_time, index),
                },
            )
        return len(records)

    # --- Search --------------------------------------------------------

    def search_historical_investigations(self, query_text: str, top_k: int = 5) -> list[KnowledgeMatch]:
        return self._store.query(KnowledgeCollection.HISTORICAL_INVESTIGATIONS, query_text, top_k)

    def search_documentation(self, query_text: str, top_k: int = 5) -> list[KnowledgeMatch]:
        return self._store.query(KnowledgeCollection.DOCUMENTATION, query_text, top_k)

    def search_known_bugs(self, query_text: str, top_k: int = 5) -> list[KnowledgeMatch]:
        return self._store.query(KnowledgeCollection.KNOWN_BUGS, query_text, top_k)

    # --- Recent (Dashboard, Knowledge Center default sort) -----------------

    def list_recent_documentation(self, limit: int = 5) -> list[KnowledgeMatch]:
        return self._store.list_recent(KnowledgeCollection.DOCUMENTATION, limit)

    def list_recent_known_bugs(self, limit: int = 5) -> list[KnowledgeMatch]:
        return self._store.list_recent(KnowledgeCollection.KNOWN_BUGS, limit)


def _seed_timestamp(base_time: datetime, index: int) -> str:
    """Deterministic, monotonically increasing ISO timestamp for seed
    records: later entries in the JSON file are treated as more recently
    imported. This is the literal truth of a JSON-seeded Sprint 1/2 system
    -- not a stand-in for real per-record import history, which arrives
    with real connectors (RFC rev 2, Future Connectors)."""
    return (base_time + timedelta(seconds=index)).isoformat()
