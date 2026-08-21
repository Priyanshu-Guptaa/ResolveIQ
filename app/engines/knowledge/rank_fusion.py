"""Reciprocal Rank Fusion (RRF) -- a pure, domain-agnostic function that
combines two or more independently-ranked candidate lists into one
fused ranking (Chat Assistant Phase 2 -- Hybrid Retrieval + RRF
foundation).

Deliberately a standalone module with no dependency on ``KnowledgeStore``,
``ChromaKnowledgeStore``, or ``LexicalKnowledgeStore`` -- it only knows
about ``KnowledgeMatch`` objects and their ``record_id``/rank position,
the same "shared primitive, no framework" idiom already established by
``app.engines.shared.text_matching``/``app.engines.shared.hierarchy``.

Standard RRF formula (Cormack et al.): for a document appearing at rank
``r`` (1-based) in a ranked list, it contributes ``1 / (k + r)`` to its
fused score; a document appearing in more than one list sums every
list's contribution. ``k=60`` is the literature-standard default.

IMPORTANT (see this phase's approved design, Section 3 -- "CRITICAL
SCORE CONTRACT"): the fused score this function computes is an
*internal ranking signal only*. It is never exposed as
``KnowledgeMatch.score`` -- ``HybridKnowledgeStore`` (the only caller)
is responsible for keeping ``.score`` equal to the real vector/cosine
similarity for any vector-found candidate, per that design. This
module has no opinion about that at all; it only fuses order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.recommendation import KnowledgeMatch

_DEFAULT_K = 60
"""The literature-standard RRF constant (Cormack, Clarke, Buettcher
2009) -- robust across very different score distributions between
retrievers, which is exactly the case here (BM25's unbounded scores
vs. cosine similarity's [0,1] range). Kept as a keyword default, not
hardcoded inline, so callers (HybridKnowledgeStore, fed from
Settings.rrf_k) can override it without touching this module."""


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list["KnowledgeMatch"]],
    *,
    k: int = _DEFAULT_K,
    top_k: int | None = None,
) -> list["KnowledgeMatch"]:
    """Fuses any number of independently-ranked ``KnowledgeMatch``
    lists (keyed by an arbitrary, caller-chosen retriever name, e.g.
    ``"vector"``/``"lexical"``) into one deterministic ranked list.

    Identity is by ``record_id`` alone -- each call is already scoped
    to one collection by its caller (``HybridKnowledgeStore.query()``
    never mixes collections in a single fusion call), so no
    collection-qualified key is needed here.

    Rank starts at 1 (not 0) -- the standard RRF convention.

    A record appearing in only one list receives only that list's
    ``1/(k+rank)`` contribution -- never penalized further for the
    signal it didn't get from the other list (absence of evidence is
    not evidence of a mismatch, the same principle
    ``ApplicabilityRanker`` already applies). A record appearing in
    more than one list sums every list's contribution, genuinely
    rewarding cross-signal agreement.

    ``top_k`` is applied strictly AFTER fusion -- a candidate ranked
    outside one retriever's own window still gets to contribute if it
    ranked well in the other; truncating before fusion would silently
    discard exactly the candidates hybrid retrieval exists to surface.

    Deterministic regardless of ``ranked_lists``' own key order or any
    input list's internal tie ordering: the final sort key is
    ``(-fused_score, record_id)`` via Python's stable ``sorted()``, so
    ties are always broken by ``record_id`` ascending -- never by
    dict/set iteration order, which this function never relies on for
    anything observable in its output.

    Returns ``[]`` for empty/all-empty input, never raises. A single
    populated list works correctly (degrades to a plain, RRF-scored
    re-expression of that one list's own order) -- this is also what
    lets ``HybridKnowledgeStore`` degrade gracefully when one backing
    store fails or returns nothing (see that module's own docstring).
    """
    fused_scores: dict[str, float] = {}
    match_by_id: dict[str, "KnowledgeMatch"] = {}

    for matches in ranked_lists.values():
        for rank, match in enumerate(matches, start=1):
            fused_scores[match.record_id] = fused_scores.get(match.record_id, 0.0) + 1.0 / (k + rank)
            # Representative-object selection (which KnowledgeMatch object's
            # score/metadata/title/snippet survives into the output for a
            # record present in more than one list): first list wins, by
            # ranked_lists' own iteration order -- a real, guaranteed-since-
            # Python-3.7 dict property, not an accident. This is a
            # deliberate, load-bearing contract, not an implementation
            # detail: HybridKnowledgeStore relies on it explicitly by always
            # constructing ranked_lists as {"vector": ..., "lexical": ...}
            # in that literal order, which is what guarantees the real
            # cosine-similarity score survives fusion for any record the
            # vector store found -- see this phase's approved "CRITICAL
            # SCORE CONTRACT." Never overwritten once set.
            match_by_id.setdefault(match.record_id, match)

    ordered_ids = sorted(fused_scores, key=lambda record_id: (-fused_scores[record_id], record_id))
    if top_k is not None:
        ordered_ids = ordered_ids[:top_k]

    return [match_by_id[record_id] for record_id in ordered_ids]
