"""Component Registry: an in-memory lookup over ComponentProfile records,
loaded from a JSON seed file.

Deliberately the simplest thing that works: no database table, no vector
index -- there's no search-relevance problem to solve yet (six profiles,
exact/substring name lookup). If this registry later needs to be
user-editable or grow into the hundreds, promoting it to a real table is
a contained change behind this same interface, not a redesign -- the same
reasoning already applied to Evidence's JSON columns and the SQL Studio
Query Library.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from app.domain.product_intelligence import ComponentProfile

if TYPE_CHECKING:
    from app.infrastructure.db.component_repository import ComponentProfileRepository

logger = logging.getLogger(__name__)


class ComponentRegistry:
    def __init__(self, profiles: list[ComponentProfile] | None = None) -> None:
        self._profiles: list[ComponentProfile] = profiles or []
        self._by_name: dict[str, ComponentProfile] = {p.name: p for p in self._profiles}

    @classmethod
    def load_from_file(cls, path: Path) -> "ComponentRegistry":
        """Reads the JSON seed file directly. Kept for tests and for the
        one-time migration in ``seed_migration.py`` -- the running
        application no longer loads the registry this way (see
        ``load_from_repository``, used by dependency wiring since
        Sprint 3 Phase 3.1)."""
        if not path.exists():
            logger.warning("Component profile file not found: %s", path)
            return cls([])
        raw = json.loads(path.read_text())
        profiles = [ComponentProfile(**item) for item in raw]
        logger.info("Loaded %d component profile(s) from %s", len(profiles), path)
        return cls(profiles)

    @classmethod
    def load_from_repository(cls, repository: "ComponentProfileRepository") -> "ComponentRegistry":
        """Builds the in-memory registry from the governed
        ``component_profiles`` table instead of the JSON seed file --
        the production path since Sprint 3 Phase 3.1. Still a plain
        in-memory index after loading (six-ish rows, no per-request DB
        hit needed for a name lookup); only the *source* changed."""
        profiles = repository.list_all()
        logger.info("Loaded %d component profile(s) from the database", len(profiles))
        return cls(profiles)

    def get(self, name: str) -> ComponentProfile | None:
        return self._by_name.get(name)

    def list_all(self) -> list[ComponentProfile]:
        return list(self._profiles)

    def search(self, query: str) -> list[ComponentProfile]:
        """Substring match on name only -- no ranking, no semantics.
        Exists for a search box with a handful of profiles; revisit if
        the registry grows large enough to need real relevance."""
        q = query.strip().lower()
        if not q:
            return self.list_all()
        return [p for p in self._profiles if q in p.name.lower()]
