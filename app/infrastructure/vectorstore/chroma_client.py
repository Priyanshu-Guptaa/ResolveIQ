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


def get_chroma_http_client(host: str, port: int = 8000, *, ssl: bool = False, auth_token: str | None = None):
    """Returns a process-wide client for a standalone Chroma server.

    Used when the API runs as more than one replica (an embedded
    ``PersistentClient`` is single-process: two processes opening the same
    directory corrupt each other) or when the vector index must live on its
    own service. The collection layout and every query are identical to the
    embedded mode -- only where the index lives changes.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    key = f"http://{host}:{port}|ssl={ssl}"
    if key not in _client_cache:
        logger.info("Connecting to ChromaDB server at %s:%s (ssl=%s)", host, port, ssl)
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        _client_cache[key] = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=ssl,
            headers=headers,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client_cache[key]
