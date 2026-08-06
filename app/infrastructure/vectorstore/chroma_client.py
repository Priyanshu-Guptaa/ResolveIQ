"""Thin infrastructure wrapper around the ChromaDB client.

Isolating client construction here (persist directory, singleton lifetime)
keeps every other module talking to Chroma through the higher-level
:class:`~app.engines.knowledge.knowledge_store.KnowledgeStore` abstraction
instead of the raw SDK.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_client_cache: dict[str, object] = {}


def get_chroma_client(persist_dir: Path):
    """Returns a process-wide persistent Chroma client for ``persist_dir``.

    Cached per directory (not just a bare singleton) so tests can point at
    a temp directory without interfering with the app's own data.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    key = str(persist_dir)
    if key not in _client_cache:
        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Opening ChromaDB persistent store at %s", persist_dir)
        # Anonymized telemetry is irrelevant for an on-prem tool and, on
        # some chromadb/posthog version combinations, fails noisily at
        # ERROR level on every call -- disable it outright.
        _client_cache[key] = chromadb.PersistentClient(
            path=key, settings=ChromaSettings(anonymized_telemetry=False)
        )
    return _client_cache[key]
