"""Text embedding for semantic search.

Kept behind a Protocol so the Knowledge Engine never depends on
``sentence_transformers`` directly -- swapping in Azure OpenAI embeddings
later (as discussed for a future sprint) means adding one new class here,
no changes anywhere else.
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Anything that turns text into fixed-size vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    def dimensions(self) -> int:
        ...


class SentenceTransformerEmbeddingProvider:
    """Local, on-prem-friendly embeddings via ``sentence-transformers``.

    No network calls, no per-query cost, no evidence content leaving the
    network -- appropriate default for logs/tickets that may contain
    customer or meter data. The model is loaded once and reused (loading it
    is the expensive part, ~1-2s).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        # Imported lazily so importing this module doesn't force-load the
        # (fairly heavy) sentence-transformers/torch stack for code paths
        # that don't need embeddings (e.g. running the entity-extractor
        # unit tests).
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model '%s' (first load may take a moment)...", model_name)
        self._model = SentenceTransformer(model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding model ready (dimensions=%d)", self._dimensions)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()

    @property
    def dimensions(self) -> int:
        return self._dimensions


class OnnxEmbeddingProvider:
    """``all-MiniLM-L6-v2`` through onnxruntime (via ChromaDB's bundled
    runner) -- the same weights and tokenizer as
    :class:`SentenceTransformerEmbeddingProvider`, without torch/
    transformers (roughly 1.5-2 GB smaller in a container image).

    Measured against sentence-transformers on sample texts: cosine
    similarity 1.000000, so an index built with either backend is valid
    for the other. The model file (~80 MB) is downloaded once to
    ``~/.cache/chroma`` on first use -- bake it into the image at build
    time rather than letting every container fetch it on start.
    """

    DIMENSIONS = 384

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        logger.info("Loading ONNX embedding model (all-MiniLM-L6-v2)...")
        self._embed = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return [[float(x) for x in vector] for vector in self._embed(texts)]

    @property
    def dimensions(self) -> int:
        return self.DIMENSIONS
