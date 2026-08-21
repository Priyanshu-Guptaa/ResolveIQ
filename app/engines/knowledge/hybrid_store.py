"""``HybridKnowledgeStore`` -- composes the existing, unchanged
``ChromaKnowledgeStore`` with the new ``LexicalKnowledgeStore``, fused
via ``reciprocal_rank_fusion()`` (Chat Assistant Phase 2 -- Hybrid
Retrieval + RRF foundation).

Implements the exact same ``KnowledgeStore`` Protocol both backing
stores already do -- nothing above this class (``KnowledgeEngine``,
``ApplicabilityRanker``, ``RecommendationEngine``) needs to change or
even know this composition exists; wiring it in is a single line in
``app/api/dependencies.py``'s ``_knowledge_store()``, gated behind
``Settings.rrf_enabled`` (default ``False``).

=== CRITICAL SCORE CONTRACT (this phase's approved design, Section 3) ===
``KnowledgeMatch.score`` means real vector/cosine similarity everywhere
else in this codebase -- it is displayed to engineers as an "X%
similarity" percentage, compared against ``min_similarity_for_root_cause``,
used directly as ``RootCauseHypothesis.confidence``, and summed
additively with ``ApplicabilityRanker``'s fixed, calibrated weights.
This class NEVER substitutes a fused RRF score or a BM25 score for
``.score``:

- A record the vector store found (alone, or also found by the lexical
  store) keeps ITS OWN real vector score, untouched.
- A record found ONLY by the lexical store has no computed cosine
  similarity at all -- ``.score`` is set to ``0.0`` (an honest "no
  semantic-similarity measurement exists," not a confidence claim),
  with ``metadata["semantic_score"] = None`` and
  ``metadata["match_source"] = "lexical_only"`` making this fully
  explainable, never a silent reshuffle -- the same discipline
  ``ApplicabilityRanker``'s own ``metadata["semantic_score"]``/
  ``metadata["applicability_reasons"]`` already establish.

``RecommendationEngine._annotate_match_reasons()`` is deliberately NOT
modified in this phase to narrate any of this fusion metadata into
``KnowledgeMatch.reason`` -- it stays present and inspectable in
``metadata`` only, a natural, small future enhancement.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.engines.knowledge.rank_fusion import reciprocal_rank_fusion

if TYPE_CHECKING:
    from app.domain.enums import KnowledgeCollection
    from app.domain.recommendation import KnowledgeMatch
    from app.engines.knowledge.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_DEFAULT_RRF_K = 60


class HybridKnowledgeStore:
    """See module docstring. Structurally satisfies ``KnowledgeStore``
    the same way ``ChromaKnowledgeStore``/``LexicalKnowledgeStore``
    already do."""

    def __init__(self, vector_store: "KnowledgeStore", lexical_store: "KnowledgeStore", *, rrf_k: int = _DEFAULT_RRF_K) -> None:
        self._vector = vector_store
        self._lexical = lexical_store
        self._rrf_k = rrf_k

    # --- KnowledgeStore Protocol -------------------------------------------

    def upsert(
        self,
        collection: "KnowledgeCollection",
        record_id: str,
        text: str,
        title: str,
        metadata: dict,
    ) -> None:
        """Fans out to both backing stores with the exact same
        arguments. The vector store is authoritative -- a vector-store
        failure propagates unchanged (this phase must not weaken the
        vector-store consistency guarantee that already existed before
        it). A lexical-store failure is logged as a WARNING and never
        raised -- a corpus-sync gap is made visible, never silent, but
        never blocks a real knowledge-management action either."""
        self._vector.upsert(collection, record_id, text, title, metadata)
        try:
            self._lexical.upsert(collection, record_id, text, title, metadata)
        except Exception as exc:  # noqa: BLE001 -- see docstring: logged, never raised, never hidden
            logger.warning(
                "Lexical store upsert failed for %s/%s -- vector store remains authoritative and up to date; "
                "lexical retrieval for this record may be stale until the next successful write: %s",
                getattr(collection, "value", collection),
                record_id,
                exc.__class__.__name__,
            )

    def delete(self, collection: "KnowledgeCollection", record_id: str) -> None:
        """Same authoritative-vector / logged-lexical-failure contract
        as upsert()."""
        self._vector.delete(collection, record_id)
        try:
            self._lexical.delete(collection, record_id)
        except Exception as exc:  # noqa: BLE001 -- see docstring
            logger.warning(
                "Lexical store delete failed for %s/%s -- vector store remains authoritative: %s",
                getattr(collection, "value", collection),
                record_id,
                exc.__class__.__name__,
            )

    def count(self, collection: "KnowledgeCollection") -> int:
        """Delegates to the vector store only -- the same source of
        truth seed_from_directory()'s own idempotency check
        (``if not force and self._store.count(collection) > 0``)
        already relies on, unchanged by this phase."""
        return self._vector.count(collection)

    def list_recent(self, collection: "KnowledgeCollection", limit: int = 5) -> list["KnowledgeMatch"]:
        """Delegates to the vector store only -- a pure lexical index
        has no recency semantics of its own (see
        LexicalKnowledgeStore.list_recent()'s own docstring), and this
        method's existing "display convenience, not a search" contract
        is unchanged by this phase."""
        return self._vector.list_recent(collection, limit)

    def query(self, collection: "KnowledgeCollection", text: str, top_k: int = 5) -> list["KnowledgeMatch"]:
        """1. Query both backing stores at the SAME top_k this method
        itself received -- RecommendationEngine already overfetches
        (top_k * 4) before calling KnowledgeStore.query() at all, so no
        second overfetch multiplier is introduced here; both retrievers
        get exactly the depth their caller already asked for.
        2. Fuse via reciprocal_rank_fusion(), top_k applied after
        fusion (never before).
        3. Reconcile each fused record's real score/metadata per the
        module's own CRITICAL SCORE CONTRACT above.

        Either backing store failing is caught narrowly (around that
        one store's own query() call only, never around this whole
        method) and degrades to the other's results, logged as a
        WARNING -- never raised, never crashes the caller. Both empty
        or both failing returns []."""
        vector_matches = self._safe_query(self._vector, "vector", collection, text, top_k)
        lexical_matches = self._safe_query(self._lexical, "lexical", collection, text, top_k)

        vector_ids = {match.record_id for match in vector_matches}
        lexical_scores = {match.record_id: match.score for match in lexical_matches}

        fused = reciprocal_rank_fusion(
            {"vector": vector_matches, "lexical": lexical_matches}, k=self._rrf_k, top_k=top_k
        )

        results: list["KnowledgeMatch"] = []
        for rank, match in enumerate(fused, start=1):
            in_vector = match.record_id in vector_ids
            in_lexical = match.record_id in lexical_scores
            match_source = "both" if (in_vector and in_lexical) else ("vector" if in_vector else "lexical_only")

            metadata = dict(match.metadata)
            metadata["match_source"] = match_source
            metadata["rrf_rank"] = rank
            if match.record_id in lexical_scores:
                metadata["lexical_score"] = lexical_scores[match.record_id]

            if match_source == "lexical_only":
                # No vector store ever computed a similarity for this record --
                # 0.0 is an honest "not measured," never a confidence claim.
                # See module docstring's CRITICAL SCORE CONTRACT.
                metadata["semantic_score"] = None
                results.append(match.model_copy(update={"score": 0.0, "metadata": metadata}))
            else:
                # "vector" or "both": the fused representative object is
                # already the vector store's own object (reciprocal_rank_fusion
                # iterates the "vector" key first, by construction below) --
                # its real .score is preserved untouched, only metadata gains
                # the fusion-provenance fields.
                metadata["semantic_score"] = match.score
                results.append(match.model_copy(update={"metadata": metadata}))

        return results

    def _safe_query(
        self, store: "KnowledgeStore", store_name: str, collection: "KnowledgeCollection", text: str, top_k: int
    ) -> list["KnowledgeMatch"]:
        try:
            return store.query(collection, text, top_k)
        except Exception as exc:  # noqa: BLE001 -- narrow to this one backing-store call only, never the whole method
            logger.warning(
                "%s retrieval failed for collection %s -- degrading to the other retriever's results: %s",
                store_name,
                getattr(collection, "value", collection),
                exc.__class__.__name__,
            )
            return []
