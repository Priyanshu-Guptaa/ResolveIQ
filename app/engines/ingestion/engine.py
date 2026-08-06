"""Ingestion Engine: thin facade over the FileTypeRegistry.

Exists mainly for consistency with the other five engines (all DI'd as
"Engine" classes, not raw registries/stores) and so it's swappable in
tests without touching the registry's internals.
"""

from __future__ import annotations

from app.domain.ingestion import ParsedFile
from app.engines.ingestion.file_type_registry import FileTypeRegistry


class IngestionEngine:
    def __init__(self, registry: FileTypeRegistry) -> None:
        self._registry = registry

    def parse(self, filename: str, content: bytes) -> list[ParsedFile]:
        return self._registry.parse(filename, content)
