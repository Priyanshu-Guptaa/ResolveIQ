"""Semantic storage/search over knowledge records (historical
investigations, documentation, known bugs).

:class:`KnowledgeStore` is the abstraction every engine depends on;
:class:`ChromaKnowledgeStore` is the Sprint 1 implementation. Swapping the
vector database later (e.g. to a managed/enterprise service) means writing
one new class against this same Protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from app.domain.enums import KnowledgeCollection
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.embedding_provider import EmbeddingProvider
from app.infrastructure.vectorstore.chroma_client import get_chroma_client

logger = logging.getLogger(__name__)


class KnowledgeStore(Protocol):
    """Anything that can index and semantically search knowledge records."""

    def upsert(
        self,
        collection: KnowledgeCollection,
        record_id: str,
        text: str,
        title: str,
        metadata: dict,
    ) -> None:
        ...

    def query(
        self, collection: KnowledgeCollection, text: str, top_k: int = 5
    ) -> list[KnowledgeMatch]:
        ...

    def count(self, collection: KnowledgeCollection) -> int:
        ...

    def list_recent(self, collection: KnowledgeCollection, limit: int = 5) -> list[KnowledgeMatch]:
        ...

    def delete(self, collection: KnowledgeCollection, record_id: str) -> None:
        ...


class ChromaKnowledgeStore:
    """ChromaDB-backed :class:`KnowledgeStore`.

    Embeddings are computed explicitly via the injected
    :class:`EmbeddingProvider` (rather than letting Chroma manage its own
    embedding function) so the whole app uses one consistent, swappable
    embedding source.
    """

    def __init__(self, persist_dir: Path, embedding_provider: EmbeddingProvider) -> None:
        self._client = get_chroma_client(persist_dir)
        self._embedder = embedding_provider
        self._collections: dict[KnowledgeCollection, object] = {}

    def _collection(self, collection: KnowledgeCollection):
        if collection not in self._collections:
            self._collections[collection] = self._client.get_or_create_collection(
                name=collection.value,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[collection]

    def upsert(
        self,
        collection: KnowledgeCollection,
        record_id: str,
        text: str,
        title: str,
        metadata: dict,
    ) -> None:
        embedding = self._embedder.embed([text])[0]
        self._collection(collection).upsert(
            ids=[record_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{**metadata, "title": title}],
        )

    def query(
        self, collection: KnowledgeCollection, text: str, top_k: int = 5
    ) -> list[KnowledgeMatch]:
        if self.count(collection) == 0:
            return []

        embedding = self._embedder.embed([text])[0]
        results = self._collection(collection).query(
            query_embeddings=[embedding],
            n_results=min(top_k, self.count(collection)),
        )

        matches: list[KnowledgeMatch] = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            # Cosine distance is in [0, 2] (0 = identical); convert to a
            # [0, 1] similarity score for display, clamped for safety.
            score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
            metadata = dict(metadata or {})
            title = metadata.pop("title", record_id)
            matches.append(
                KnowledgeMatch(
                    collection=collection,
                    record_id=record_id,
                    title=title,
                    snippet=(document or "")[:400],
                    score=score,
                    metadata=metadata,
                )
            )
        return matches

    def count(self, collection: KnowledgeCollection) -> int:
        return self._collection(collection).count()

    def delete(self, collection: KnowledgeCollection, record_id: str) -> None:
        """Removes one record from the index -- used when a document is
        archived (Sprint 3, Phase 3.2), so it stops appearing in search
        without needing to rebuild the whole collection. A no-op if the
        id was never indexed (e.g. archiving a document straight from
        Draft, which was never published)."""
        self._collection(collection).delete(ids=[record_id])

    def list_recent(self, collection: KnowledgeCollection, limit: int = 5) -> list[KnowledgeMatch]:
        """Most recently imported records, newest first -- not a search,
        so ``score`` is not a similarity and is always 1.0. Backs the
        Dashboard's Recent Knowledge panel and Knowledge Center's default
        sort order.

        Chroma has no server-side "order by metadata field" for ``get()``,
        so this fetches everything in the collection and sorts in Python --
        fine at the seed-data/early-adoption scale this targets (dozens to
        low hundreds of records). Revisit if a collection grows large.
        """
        if self.count(collection) == 0:
            return []

        results = self._collection(collection).get(include=["documents", "metadatas"])
        ids = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        rows = list(zip(ids, documents, metadatas))
        rows.sort(key=lambda row: row[2].get("imported_at", ""), reverse=True)

        matches: list[KnowledgeMatch] = []
        for record_id, document, metadata in rows[:limit]:
            metadata = dict(metadata or {})
            title = metadata.pop("title", record_id)
            matches.append(
                KnowledgeMatch(
                    collection=collection,
                    record_id=record_id,
                    title=title,
                    snippet=(document or "")[:400],
                    score=1.0,
                    metadata=metadata,
                )
            )
        return matches
