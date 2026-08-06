"""Knowledge Engine: import + semantic search over historical
investigations, documentation, and known bugs.

Sprint 1 imports sample JSON files (standing in for ServiceNow/Wiki/ADO
exports, which arrive as connectors in a later sprint). The import format
is intentionally the same shape the real connectors will eventually
produce, so swapping the source later doesn't change this engine.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.domain.enums import KnowledgeCollection
from app.domain.evidence import DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class KnowledgeEngine:
    """Facade over the :class:`KnowledgeStore` for import and search."""

    def __init__(self, store: KnowledgeStore) -> None:
        self._store = store

    # --- Import ------------------------------------------------------------

    def seed_from_directory(self, sample_dir: Path, *, force: bool = False) -> dict[str, int]:
        """Load sample knowledge JSON into their respective collections.

        Idempotent by default: a collection already holding data is left
        alone unless ``force=True``. Returns a count of records loaded per
        collection, for logging/diagnostics.
        """
        loaded = {
            "historical_investigations": self._seed_historical_investigations(
                sample_dir / "historical_investigations.json", force=force
            ),
            "documentation": self._seed_documentation(sample_dir / "documentation.json", force=force),
            "known_bugs": self._seed_known_bugs(sample_dir / "known_bugs.json", force=force),
        }
        logger.info("Knowledge Engine seed complete: %s", loaded)
        return loaded

    def _seed_historical_investigations(self, path: Path, *, force: bool) -> int:
        collection = KnowledgeCollection.HISTORICAL_INVESTIGATIONS
        if not force and self._store.count(collection) > 0:
            return 0
        if not path.exists():
            logger.warning("Sample file not found: %s", path)
            return 0

        records = [HistoricalInvestigationRecord(**raw) for raw in json.loads(path.read_text())]
        for record in records:
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
                },
            )
        return len(records)

    def _seed_documentation(self, path: Path, *, force: bool) -> int:
        collection = KnowledgeCollection.DOCUMENTATION
        if not force and self._store.count(collection) > 0:
            return 0
        if not path.exists():
            logger.warning("Sample file not found: %s", path)
            return 0

        records = [DocumentationRecord(**raw) for raw in json.loads(path.read_text())]
        for record in records:
            searchable_text = f"{record.title}\n{record.content}\nTags: {', '.join(record.tags)}"
            self._store.upsert(
                collection,
                record.id,
                searchable_text,
                record.title,
                metadata={"source": record.source, "tags": ", ".join(record.tags)},
            )
        return len(records)

    def _seed_known_bugs(self, path: Path, *, force: bool) -> int:
        collection = KnowledgeCollection.KNOWN_BUGS
        if not force and self._store.count(collection) > 0:
            return 0
        if not path.exists():
            logger.warning("Sample file not found: %s", path)
            return 0

        records = [KnownBugRecord(**raw) for raw in json.loads(path.read_text())]
        for record in records:
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
