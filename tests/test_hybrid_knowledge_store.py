"""Tests for HybridKnowledgeStore (Chat Assistant Phase 2 -- Hybrid
Retrieval + RRF foundation).

Uses hand-written fake KnowledgeStore implementations (never
unittest.mock), matching this codebase's own established test-double
convention (FakeKnowledgeStore in test_recommendation_engine.py,
ConfigurableFakeKnowledgeStore in test_chat_orchestrator.py). No
database, no HTTP, no real ChromaDB, no real rank_bm25 required for
most tests -- the two real backing stores are only used where their
real behavior is the point (upsert/delete fan-out).
"""

from __future__ import annotations

from app.domain.enums import KnowledgeCollection
from app.domain.recommendation import KnowledgeMatch
from app.engines.knowledge.hybrid_store import HybridKnowledgeStore
from app.engines.knowledge.lexical_store import LexicalKnowledgeStore

_HI = KnowledgeCollection.HISTORICAL_INVESTIGATIONS


def _match(record_id: str, score: float = 0.5, metadata: dict | None = None) -> KnowledgeMatch:
    return KnowledgeMatch(
        collection=_HI, record_id=record_id, title=f"Title {record_id}", snippet="snippet",
        score=score, metadata=metadata or {},
    )


class FakeStore:
    """Structurally satisfies KnowledgeStore -- canned query() results,
    real in-memory upsert()/delete()/count() bookkeeping so fan-out
    tests can verify real state, not just call counts."""

    def __init__(self, canned: list[KnowledgeMatch] | None = None, *, raise_on_query: bool = False, raise_on_upsert: bool = False):
        self._canned = canned or []
        self._raise_on_query = raise_on_query
        self._raise_on_upsert = raise_on_upsert
        self.upserted: list[tuple] = []
        self.deleted: list[tuple] = []

    def upsert(self, collection, record_id, text, title, metadata) -> None:
        if self._raise_on_upsert:
            raise RuntimeError("simulated upsert failure")
        self.upserted.append((collection, record_id, text, title, metadata))

    def delete(self, collection, record_id) -> None:
        self.deleted.append((collection, record_id))

    def query(self, collection, text, top_k: int = 5) -> list[KnowledgeMatch]:
        if self._raise_on_query:
            raise RuntimeError("simulated query failure")
        return list(self._canned)[:top_k]

    def count(self, collection) -> int:
        return len(self._canned)

    def list_recent(self, collection, limit: int = 5) -> list[KnowledgeMatch]:
        return list(self._canned)[:limit]


# --- Retrieval behavior -----------------------------------------------------


def test_vector_only_result_surfaces():
    vector = FakeStore([_match("a", score=0.8)])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert [m.record_id for m in results] == ["a"]
    assert results[0].metadata["match_source"] == "vector"


def test_lexical_only_result_surfaces():
    vector = FakeStore([])
    lexical = FakeStore([_match("a", score=0.9)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert [m.record_id for m in results] == ["a"]
    assert results[0].metadata["match_source"] == "lexical_only"


def test_overlapping_result_is_fused_once():
    vector = FakeStore([_match("a", score=0.8)])
    lexical = FakeStore([_match("a", score=0.9)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert len(results) == 1
    assert results[0].record_id == "a"
    assert results[0].metadata["match_source"] == "both"


def test_rrf_ranking_promotes_cross_signal_agreement():
    # "shared" is rank 2 in both lists; "vector-top" is rank 1 in
    # vector alone. Cross-list agreement must outrank a single list's
    # own top rank -- the entire point of RRF.
    vector = FakeStore([_match("vector-top", score=0.9), _match("shared", score=0.7)])
    lexical = FakeStore([_match("lexical-top", score=0.9), _match("shared", score=0.6)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    ranks = {m.record_id: i for i, m in enumerate(results)}
    assert ranks["shared"] < ranks["vector-top"]
    assert ranks["shared"] < ranks["lexical-top"]


def test_top_k_respected():
    vector = FakeStore([_match(f"v{i}", score=0.9 - i * 0.01) for i in range(10)])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=3)
    assert len(results) == 3


def test_deterministic_output_across_repeated_calls():
    vector = FakeStore([_match("a", score=0.8), _match("b", score=0.7)])
    lexical = FakeStore([_match("b", score=0.6), _match("c", score=0.5)])
    store = HybridKnowledgeStore(vector, lexical)
    first = [m.record_id for m in store.query(_HI, "question", top_k=5)]
    second = [m.record_id for m in store.query(_HI, "question", top_k=5)]
    assert first == second


def test_both_retrievers_empty_returns_empty_list():
    store = HybridKnowledgeStore(FakeStore([]), FakeStore([]))
    assert store.query(_HI, "question", top_k=5) == []


def test_one_retriever_empty_the_other_still_works():
    vector = FakeStore([_match("a", score=0.8)])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert [m.record_id for m in results] == ["a"]


# --- CRITICAL SCORE CONTRACT (this phase's hard correctness requirement) ---


def test_vector_score_is_never_overwritten_by_fusion_for_vector_only_match():
    vector = FakeStore([_match("a", score=0.8234)])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].score == 0.8234


def test_vector_score_is_never_overwritten_by_fusion_for_a_record_in_both():
    # Vector's real cosine score (0.7) must survive fusion untouched --
    # never replaced by the lexical store's own score (0.95) or by any
    # fused/normalized RRF value.
    vector = FakeStore([_match("a", score=0.7)])
    lexical = FakeStore([_match("a", score=0.95)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].score == 0.7


def test_lexical_only_match_score_is_exactly_zero():
    vector = FakeStore([])
    lexical = FakeStore([_match("a", score=0.95)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].score == 0.0


def test_semantic_score_metadata_is_none_for_lexical_only_match():
    vector = FakeStore([])
    lexical = FakeStore([_match("a", score=0.95)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].metadata["semantic_score"] is None


def test_semantic_score_metadata_preserves_real_score_for_vector_match():
    vector = FakeStore([_match("a", score=0.8234)])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].metadata["semantic_score"] == 0.8234


def test_match_source_metadata_values():
    vector = FakeStore([_match("v", score=0.8), _match("both", score=0.7)])
    lexical = FakeStore([_match("l", score=0.9), _match("both", score=0.6)])
    store = HybridKnowledgeStore(vector, lexical)
    by_id = {m.record_id: m for m in store.query(_HI, "question", top_k=5)}
    assert by_id["v"].metadata["match_source"] == "vector"
    assert by_id["l"].metadata["match_source"] == "lexical_only"
    assert by_id["both"].metadata["match_source"] == "both"


def test_rrf_rank_metadata_present_and_ordered():
    vector = FakeStore([_match("a", score=0.9), _match("b", score=0.8)])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert [m.metadata["rrf_rank"] for m in results] == [1, 2]


def test_lexical_score_metadata_present_when_lexical_found_the_record():
    vector = FakeStore([_match("a", score=0.8)])
    lexical = FakeStore([_match("a", score=0.55)])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].metadata["lexical_score"] == 0.55


def test_other_metadata_keys_from_the_vector_match_are_preserved():
    vector = FakeStore([_match("a", score=0.8, metadata={"root_cause": "lost route"})])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "question", top_k=5)
    assert results[0].metadata["root_cause"] == "lost route"


# --- Failure / degradation --------------------------------------------------


def test_vector_retriever_failure_degrades_to_lexical(caplog):
    import logging

    vector = FakeStore([], raise_on_query=True)
    lexical = FakeStore([_match("a", score=0.9)])
    store = HybridKnowledgeStore(vector, lexical)
    caplog.set_level(logging.WARNING, logger="app.engines.knowledge.hybrid_store")
    results = store.query(_HI, "question", top_k=5)
    assert [m.record_id for m in results] == ["a"]
    assert "vector retrieval failed" in caplog.text.lower()


def test_lexical_retriever_failure_degrades_to_vector(caplog):
    import logging

    vector = FakeStore([_match("a", score=0.9)])
    lexical = FakeStore([], raise_on_query=True)
    store = HybridKnowledgeStore(vector, lexical)
    caplog.set_level(logging.WARNING, logger="app.engines.knowledge.hybrid_store")
    results = store.query(_HI, "question", top_k=5)
    assert [m.record_id for m in results] == ["a"]
    assert "lexical retrieval failed" in caplog.text.lower()


def test_both_retrievers_failing_returns_empty_list_not_an_exception():
    vector = FakeStore([], raise_on_query=True)
    lexical = FakeStore([], raise_on_query=True)
    store = HybridKnowledgeStore(vector, lexical)
    assert store.query(_HI, "question", top_k=5) == []


# --- Index lifecycle ---------------------------------------------------------


def test_upsert_fans_out_identical_arguments_to_both_stores():
    vector = FakeStore([])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    store.upsert(_HI, "a", "full text", "Title", {"root_cause": "x"})
    assert vector.upserted == [(_HI, "a", "full text", "Title", {"root_cause": "x"})]
    assert lexical.upserted == [(_HI, "a", "full text", "Title", {"root_cause": "x"})]


def test_delete_fans_out_to_both_stores():
    vector = FakeStore([])
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    store.delete(_HI, "a")
    assert vector.deleted == [(_HI, "a")]
    assert lexical.deleted == [(_HI, "a")]


def test_vector_upsert_failure_propagates():
    import pytest

    vector = FakeStore([], raise_on_upsert=True)
    lexical = FakeStore([])
    store = HybridKnowledgeStore(vector, lexical)
    with pytest.raises(RuntimeError):
        store.upsert(_HI, "a", "text", "Title", {})


def test_lexical_upsert_failure_is_logged_not_raised(caplog):
    import logging

    vector = FakeStore([])
    lexical = FakeStore([], raise_on_upsert=True)
    store = HybridKnowledgeStore(vector, lexical)
    caplog.set_level(logging.WARNING, logger="app.engines.knowledge.hybrid_store")
    store.upsert(_HI, "a", "text", "Title", {})  # must not raise
    assert vector.upserted == [(_HI, "a", "text", "Title", {})]
    assert "lexical store upsert failed" in caplog.text.lower()


def test_count_delegates_to_vector_store():
    vector = FakeStore([_match("a"), _match("b")])
    lexical = FakeStore([_match("a")])
    store = HybridKnowledgeStore(vector, lexical)
    assert store.count(_HI) == 2  # vector's count, not lexical's


def test_list_recent_delegates_to_vector_store():
    vector = FakeStore([_match("a"), _match("b")])
    lexical = FakeStore([_match("z")])
    store = HybridKnowledgeStore(vector, lexical)
    results = store.list_recent(_HI, limit=5)
    assert [m.record_id for m in results] == ["a", "b"]


# --- End-to-end with the real LexicalKnowledgeStore (not a fake) -----------


def test_real_lexical_store_finds_a_document_vector_search_missed():
    """The mechanical proof hybrid retrieval does something vector-only
    search structurally cannot: a real LexicalKnowledgeStore surfaces a
    keyword-exact document the (fake, empty-result) vector store never
    returned at all."""
    vector = FakeStore([])
    lexical = LexicalKnowledgeStore()
    lexical.upsert(_HI, "a", "collector lost network route to the mesh gateway", "Real title", {"tags": "mesh"})
    store = HybridKnowledgeStore(vector, lexical)
    results = store.query(_HI, "mesh gateway", top_k=5)
    assert [m.record_id for m in results] == ["a"]
    assert results[0].score == 0.0
    assert results[0].metadata["match_source"] == "lexical_only"
    assert results[0].title == "Real title"
