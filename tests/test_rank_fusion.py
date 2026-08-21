"""Tests for reciprocal_rank_fusion() (Chat Assistant Phase 2 -- Hybrid
Retrieval + RRF foundation).

Pure unit tests -- no database, no HTTP, no ChromaDB, no BM25. Exercise
the fusion function directly against hand-built KnowledgeMatch fixtures.
"""

from __future__ import annotations

from app.domain.enums import KnowledgeCollection
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.rank_fusion import reciprocal_rank_fusion


def _match(record_id: str, score: float = 0.5) -> KnowledgeMatch:
    return KnowledgeMatch(
        collection=KnowledgeCollection.HISTORICAL_INVESTIGATIONS,
        record_id=record_id,
        title=f"Title {record_id}",
        snippet="snippet",
        score=score,
    )


def test_empty_input_returns_empty_list():
    assert reciprocal_rank_fusion({}) == []


def test_all_empty_lists_returns_empty_list():
    assert reciprocal_rank_fusion({"vector": [], "lexical": []}) == []


def test_single_retriever_preserves_its_own_order():
    matches = [_match("a"), _match("b"), _match("c")]
    result = reciprocal_rank_fusion({"vector": matches})
    assert [m.record_id for m in result] == ["a", "b", "c"]


def test_single_populated_retriever_among_an_empty_one():
    matches = [_match("a"), _match("b")]
    result = reciprocal_rank_fusion({"vector": matches, "lexical": []})
    assert [m.record_id for m in result] == ["a", "b"]


def test_two_retrievers_with_full_overlap_agree_and_reinforce_top_rank():
    # "a" is rank 1 in both lists -- must end up first.
    vector = [_match("a"), _match("b")]
    lexical = [_match("a"), _match("b")]
    result = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    assert [m.record_id for m in result] == ["a", "b"]


def test_disjoint_retrievers_each_contribute_their_own_evidence():
    vector = [_match("a"), _match("b")]
    lexical = [_match("c"), _match("d")]
    result = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    assert {m.record_id for m in result} == {"a", "b", "c", "d"}
    # Rank-1 entries from each list (a, c) must outrank rank-2 entries (b, d).
    ranks = {m.record_id: i for i, m in enumerate(result)}
    assert ranks["a"] < ranks["b"]
    assert ranks["c"] < ranks["d"]


def test_overlapping_record_outranks_a_single_list_rank_one_entry():
    # "shared" is rank 2 in both lists but appears in both (real
    # cross-signal agreement); "vector-only" is rank 1 in vector alone.
    # RRF must let the reinforced, cross-signal record win: cross-list
    # agreement is exactly the property RRF exists to reward.
    vector = [_match("vector-only"), _match("shared")]
    lexical = [_match("lexical-only"), _match("shared")]
    result = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    ranks = {m.record_id: i for i, m in enumerate(result)}
    assert ranks["shared"] < ranks["vector-only"]
    assert ranks["shared"] < ranks["lexical-only"]


def test_duplicate_record_id_across_lists_produces_one_final_record():
    vector = [_match("a")]
    lexical = [_match("a")]
    result = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    assert len(result) == 1
    assert result[0].record_id == "a"


def test_rank_starts_at_one_not_zero():
    # With k=0, the formula is 1/(k+rank) = 1/rank. If rank incorrectly
    # started at 0 for the first (best) candidate, this would be a
    # division by zero -- a real, mechanical proof, not just an
    # inference: this must not raise, and must preserve list order.
    result = reciprocal_rank_fusion({"only": [_match("a"), _match("b")]}, k=0)
    assert [m.record_id for m in result] == ["a", "b"]


def test_deterministic_tie_broken_by_record_id_ascending():
    # "b" and "z" both appear only in their own single-item lists, at
    # rank 1 -- identical fused scores. Ties must resolve to record_id
    # ascending, not whichever dict key was inserted first.
    result = reciprocal_rank_fusion({"list_z": [_match("z")], "list_b": [_match("b")]})
    assert [m.record_id for m in result] == ["b", "z"]


def test_top_k_applied_after_fusion_not_before():
    # "shared" ranks 3rd in both individual lists (would be excluded by
    # a naive per-list top_k=2 cutoff applied before fusion) but its
    # cross-list agreement must let it still surface in the final top_k.
    vector = [_match("v1"), _match("v2"), _match("shared")]
    lexical = [_match("l1"), _match("l2"), _match("shared")]
    result = reciprocal_rank_fusion({"vector": vector, "lexical": lexical}, top_k=3)
    assert len(result) == 3
    assert "shared" in {m.record_id for m in result}


def test_top_k_none_returns_every_fused_candidate():
    vector = [_match("a"), _match("b"), _match("c")]
    result = reciprocal_rank_fusion({"vector": vector}, top_k=None)
    assert len(result) == 3


def test_configurable_k_materially_changes_the_fused_order():
    # "x" is rank 1 in vector alone: score(k) = 1/(k+1).
    # "y" is rank 4 in BOTH lists: score(k) = 2/(k+4).
    # At k=0: score_x=1.0 > score_y=0.5 -- x wins (a single strong rank
    # beats a doubly-corroborated weak one when k is small).
    # At k=1000: score_x=1/1001~=0.000999 < score_y=2/1004~=0.001992 --
    # y wins (with k large, corroboration from two lists outweighs one
    # list's single top rank). This flip is only possible if k is
    # actually threaded into the formula, not silently ignored.
    vector = [_match("x"), _match("filler-a"), _match("filler-b"), _match("y")]
    lexical = [_match("filler-c"), _match("filler-d"), _match("filler-e"), _match("y")]

    ranks_k0 = {m.record_id: i for i, m in enumerate(reciprocal_rank_fusion({"vector": vector, "lexical": lexical}, k=0))}
    ranks_k1000 = {
        m.record_id: i for i, m in enumerate(reciprocal_rank_fusion({"vector": vector, "lexical": lexical}, k=1000))
    }

    assert ranks_k0["x"] < ranks_k0["y"]
    assert ranks_k1000["y"] < ranks_k1000["x"]


def test_deterministic_output_across_repeated_calls():
    vector = [_match("a"), _match("c"), _match("b")]
    lexical = [_match("b"), _match("a")]
    first = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    second = reciprocal_rank_fusion({"vector": vector, "lexical": lexical})
    assert [m.record_id for m in first] == [m.record_id for m in second]


def test_representative_object_is_taken_from_first_iterated_list():
    """The record's returned KnowledgeMatch object (score/metadata/
    title/snippet) is the one from whichever list is iterated first in
    ranked_lists -- HybridKnowledgeStore relies on this exact,
    documented contract to guarantee the real vector score always
    survives fusion (see rank_fusion.py's own docstring)."""
    vector_match = _match("a", score=0.91)
    lexical_match = _match("a", score=0.10)
    result = reciprocal_rank_fusion({"vector": [vector_match], "lexical": [lexical_match]})
    assert result[0].score == 0.91
