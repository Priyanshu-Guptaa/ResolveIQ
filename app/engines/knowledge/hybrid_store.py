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

=== FUSION MODE (Chat Assistant Phase 2C calibration) ===
Phase 2C's read-only investigation found the RRF path above
structurally vulnerable to a rank-based tie-break artifact at the real
production overfetch depth -- see PHASE 2C — RRF CALIBRATION REPORT,
Sections 2-3. Its recommended calibration, weighted score fusion
(``app.engines.knowledge.rank_fusion.weighted_score_fusion``), is
available here as a second, explicitly opt-in ``fusion_mode="score"``
alongside the original, completely unmodified ``fusion_mode="rrf"``
path -- selected per instance at construction time, defaulting to
``"rrf"`` so every existing caller (production DI, every existing
test) is byte-for-byte unaffected unless it explicitly asks for
``"score"``.

The CRITICAL SCORE CONTRACT above describes the "rrf" path only. The
"score" path has its own, deliberately different contract: ``.score``
on the returned ``KnowledgeMatch`` IS the fused value itself (the
whole point of score fusion is that it *is* the blended relevance
signal, not a stand-in for it) -- never forced to ``0.0`` for a
lexical-only record the way the "rrf" path's ``.score`` is. See
``rank_fusion.weighted_score_fusion``'s own docstring for why this
makes it structurally immune to the tie-break artifact. Observability
metadata (``match_source``, ``semantic_score``, ``lexical_score``) is
still populated for both paths, in the same shape, so existing
diagnostics/tests work unchanged regardless of which mode produced a
given result.

=== EXACT-IDENTIFIER PROTECTION (Chat Assistant Phase 2D) ===
Phase 2D's read-only investigation found that even Phase 2C's score
fusion can mis-rank a candidate containing the EXACT ticket ID,
version string, or CamelCase technical term the user searched for --
see PHASE 2D — EXACT IDENTIFIER PROTECTION DESIGN. When
``identifier_protection_enabled=True`` (default ``False``), ``query()``
calls ``app.engines.knowledge.identifier_protection.protect_exact_identifiers()``
on the already-fused result, AFTER fusion (either mode) and BEFORE the
results leave this class (i.e. before ``ApplicabilityRanker`` -- called
by this class's own caller -- ever sees them). That module is
self-contained and untouched by anything else in this class; see its
own docstring for the full detection/protection design.

KNOWN LIMITATION (Phase 2E — RRF / Identifier Protection Compatibility
Review): the CRITICAL SCORE CONTRACT above is intentional and remains
unchanged by Phase 2D/2E -- ``.score`` must NOT be interpreted as a
normalized RRF fusion strength; a lexical-only ``fusion_mode="rrf"``
candidate's ``.score == 0.0`` is a real, deliberate part of that
contract, not a placeholder. Consequently, identifier protection
should NOT be considered a reliable mechanism for recovering a
lexical-only exact identifier under ``fusion_mode="rrf"`` -- real-corpus
evidence showed the fixed boost frequently cannot overcome a genuinely
unrelated candidate's real vector score. A ``both``-sourced candidate
is unaffected by this and still benefits from protection under
``"rrf"``. ``fusion_mode="score"`` remains the recommended
configuration whenever exact identifier protection is required;
``app.config.Settings`` logs a warning when the two are combined.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.engines.knowledge.identifier_protection import protect_exact_identifiers
from app.engines.knowledge.rank_fusion import reciprocal_rank_fusion, weighted_score_fusion

if TYPE_CHECKING:
    from app.domain.enums import KnowledgeCollection
    from app.domain.recommendation import KnowledgeMatch
    from app.engines.knowledge.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_DEFAULT_RRF_K = 60
_DEFAULT_FUSION_MODE = "rrf"
"""Preserves exact Phase 2 behavior for every caller that doesn't
explicitly opt into Phase 2C's "score" mode -- see module docstring's
FUSION MODE section."""
_DEFAULT_VECTOR_WEIGHT = 0.25
_DEFAULT_LEXICAL_WEIGHT = 0.75
"""Phase 2C's recommended weights (PHASE 2C — RRF CALIBRATION REPORT,
Section 14) -- only consulted at all when ``fusion_mode="score"``."""
_VALID_FUSION_MODES = ("rrf", "score")
_DEFAULT_IDENTIFIER_PROTECTION_ENABLED = False
"""Preserves exact pre-Phase-2D behavior for every caller that doesn't
explicitly opt in -- see module docstring's EXACT-IDENTIFIER PROTECTION
section."""


class HybridKnowledgeStore:
    """See module docstring. Structurally satisfies ``KnowledgeStore``
    the same way ``ChromaKnowledgeStore``/``LexicalKnowledgeStore``
    already do."""

    def __init__(
        self,
        vector_store: "KnowledgeStore",
        lexical_store: "KnowledgeStore",
        *,
        rrf_k: int = _DEFAULT_RRF_K,
        fusion_mode: str = _DEFAULT_FUSION_MODE,
        vector_weight: float = _DEFAULT_VECTOR_WEIGHT,
        lexical_weight: float = _DEFAULT_LEXICAL_WEIGHT,
        identifier_protection_enabled: bool = _DEFAULT_IDENTIFIER_PROTECTION_ENABLED,
    ) -> None:
        if fusion_mode not in _VALID_FUSION_MODES:
            raise ValueError(f"Unknown fusion_mode {fusion_mode!r} -- must be one of {_VALID_FUSION_MODES}")
        self._vector = vector_store
        self._lexical = lexical_store
        self._rrf_k = rrf_k
        self._fusion_mode = fusion_mode
        self._vector_weight = vector_weight
        self._lexical_weight = lexical_weight
        self._identifier_protection_enabled = identifier_protection_enabled

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
        2. Fuse via this instance's configured ``fusion_mode`` -- either
        the original ``reciprocal_rank_fusion()`` (default, unchanged
        since Phase 2) or Phase 2C's ``weighted_score_fusion()``
        (opt-in) -- top_k applied after fusion in both cases (never
        before).
        3. Reconcile each fused record's real score/metadata per the
        mode-appropriate contract (see module docstring's CRITICAL
        SCORE CONTRACT / FUSION MODE sections).
        4. When ``identifier_protection_enabled=True`` (default
        ``False``), apply Phase 2D's exact-identifier boost -- see
        module docstring's EXACT-IDENTIFIER PROTECTION section.

        Either backing store failing is caught narrowly (around that
        one store's own query() call only, never around this whole
        method) and degrades to the other's results, logged as a
        WARNING -- never raised, never crashes the caller. Both empty
        or both failing returns []."""
        vector_matches = self._safe_query(self._vector, "vector", collection, text, top_k)
        lexical_matches = self._safe_query(self._lexical, "lexical", collection, text, top_k)

        vector_ids = {match.record_id for match in vector_matches}
        vector_scores = {match.record_id: match.score for match in vector_matches}
        lexical_scores = {match.record_id: match.score for match in lexical_matches}

        if self._fusion_mode == "rrf":
            fused = reciprocal_rank_fusion(
                {"vector": vector_matches, "lexical": lexical_matches}, k=self._rrf_k, top_k=top_k
            )
            results = self._reconcile_rrf(fused, vector_ids, lexical_scores)
        elif self._fusion_mode == "score":
            fused = weighted_score_fusion(
                vector_matches,
                lexical_matches,
                vector_weight=self._vector_weight,
                lexical_weight=self._lexical_weight,
                top_k=top_k,
            )
            results = self._reconcile_score_fusion(fused, vector_ids, vector_scores, lexical_scores)
        else:
            # Unreachable given __init__'s own validation -- defensive,
            # never silently falls back to "rrf" for an unrecognized
            # mode (see Phase 2C implementation Section 10).
            raise ValueError(f"Unknown fusion_mode {self._fusion_mode!r} -- must be one of {_VALID_FUSION_MODES}")

        if self._identifier_protection_enabled:
            # AFTER fusion, BEFORE this class's own caller ever hands
            # these to ApplicabilityRanker -- see module docstring's
            # EXACT-IDENTIFIER PROTECTION section and the Phase 2D
            # design's architecture diagram (Section 23).
            results = protect_exact_identifiers(text, results)

        return results

    def _reconcile_rrf(
        self, fused: list["KnowledgeMatch"], vector_ids: set[str], lexical_scores: dict[str, float]
    ) -> list["KnowledgeMatch"]:
        """Unchanged since Phase 2 -- see module docstring's CRITICAL
        SCORE CONTRACT. Extracted verbatim from ``query()`` only to make
        room for ``_reconcile_score_fusion``'s sibling method; no
        behavior change."""
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

    def _reconcile_score_fusion(
        self,
        fused: list["KnowledgeMatch"],
        vector_ids: set[str],
        vector_scores: dict[str, float],
        lexical_scores: dict[str, float],
    ) -> list["KnowledgeMatch"]:
        """Phase 2C's "score" path. ``fused`` already carries the real
        fused value in ``.score`` (set by ``weighted_score_fusion``
        itself) -- this method only ever patches ``metadata``, never
        ``.score``, unlike ``_reconcile_rrf`` (which forces ``.score``
        to ``0.0`` for a lexical-only record). See module docstring's
        FUSION MODE section for why that's the deliberate, different
        contract for this path."""
        results: list["KnowledgeMatch"] = []
        for match in fused:
            in_vector = match.record_id in vector_ids
            in_lexical = match.record_id in lexical_scores
            match_source = "both" if (in_vector and in_lexical) else ("vector" if in_vector else "lexical_only")

            metadata = dict(match.metadata)
            metadata["match_source"] = match_source
            metadata["fusion_mode"] = "score"
            metadata["semantic_score"] = vector_scores[match.record_id] if in_vector else None
            if in_lexical:
                metadata["lexical_score"] = lexical_scores[match.record_id]

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
