"""Tests for LexicalKnowledgeStore (Chat Assistant Phase 2 -- Hybrid
Retrieval + RRF foundation).

Pure unit tests -- no database, no HTTP, no ChromaDB, no embedding
model. Exercises the real rank_bm25 library directly (in-memory, no
network, no external process).
"""

from __future__ import annotations

from app.domain.enums import KnowledgeCollection
from app.engines.knowledge.lexical_store import LexicalKnowledgeStore

_HI = KnowledgeCollection.HISTORICAL_INVESTIGATIONS
_DOC = KnowledgeCollection.DOCUMENTATION


def test_exact_keyword_match_ranks_highest():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "collector lost network route to the mesh gateway", "A", {})
    store.upsert(_HI, "b", "unrelated database timeout issue", "B", {})
    results = store.query(_HI, "mesh gateway", top_k=5)
    assert results[0].record_id == "a"


def test_partial_token_match_still_surfaces():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "RF Mesh IP command timeout during firmware download", "A", {})
    results = store.query(_HI, "firmware download", top_k=5)
    assert [m.record_id for m in results] == ["a"]


def test_case_insensitivity():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "Collector Lost Network Route", "A", {})
    results_lower = store.query(_HI, "collector lost network route", top_k=5)
    results_upper = store.query(_HI, "COLLECTOR LOST NETWORK ROUTE", top_k=5)
    results_mixed = store.query(_HI, "Collector Lost Network Route", top_k=5)
    assert [m.record_id for m in results_lower] == ["a"]
    assert [m.record_id for m in results_upper] == ["a"]
    assert [m.record_id for m in results_mixed] == ["a"]


def test_punctuation_does_not_prevent_matching():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "Command Request (Outbound) failed to respond", "A", {})
    results = store.query(_HI, "Command Request Outbound", top_k=5)
    assert [m.record_id for m in results] == ["a"]


def test_empty_query_returns_empty_list():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "some real content", "A", {})
    assert store.query(_HI, "", top_k=5) == []
    assert store.query(_HI, "   ", top_k=5) == []


def test_empty_corpus_returns_empty_list():
    store = LexicalKnowledgeStore()
    assert store.query(_HI, "anything", top_k=5) == []


def test_multiple_collections_stay_independent():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "mesh gateway timeout", "A", {})
    store.upsert(_DOC, "b", "installation guide for mesh gateway", "B", {})
    hi_results = store.query(_HI, "mesh gateway", top_k=5)
    doc_results = store.query(_DOC, "mesh gateway", top_k=5)
    assert [m.record_id for m in hi_results] == ["a"]
    assert [m.record_id for m in doc_results] == ["b"]
    assert store.count(_HI) == 1
    assert store.count(_DOC) == 1


def test_upsert_makes_a_record_immediately_searchable():
    store = LexicalKnowledgeStore()
    assert store.query(_HI, "mesh gateway", top_k=5) == []
    store.upsert(_HI, "a", "mesh gateway timeout", "A", {})
    results = store.query(_HI, "mesh gateway", top_k=5)
    assert [m.record_id for m in results] == ["a"]


def test_update_existing_record_replaces_its_searchable_text():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "mesh gateway timeout", "A", {})
    store.upsert(_HI, "a", "completely different SQL blocking issue", "A", {})
    # A single-document corpus with zero term overlap still returns that
    # document (score 0.0, never excluded) -- matches ChromaKnowledgeStore's
    # own "always return up to top_k, let the caller threshold" contract.
    # The real proof the update took effect is that "SQL blocking" now
    # matches, and the record's own snippet reflects the new text.
    no_overlap = store.query(_HI, "mesh gateway", top_k=5)
    assert [m.record_id for m in no_overlap] == ["a"]
    assert no_overlap[0].score == 0.0
    assert "SQL blocking" in no_overlap[0].snippet

    results = store.query(_HI, "SQL blocking", top_k=5)
    assert [m.record_id for m in results] == ["a"]
    assert results[0].score > 0.0


def test_delete_removes_the_record():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "mesh gateway timeout", "A", {})
    store.delete(_HI, "a")
    assert store.query(_HI, "mesh gateway", top_k=5) == []
    assert store.count(_HI) == 0


def test_delete_of_never_indexed_id_is_a_noop():
    store = LexicalKnowledgeStore()
    store.delete(_HI, "does-not-exist")  # must not raise
    assert store.count(_HI) == 0


def test_count_reflects_real_corpus_size():
    store = LexicalKnowledgeStore()
    assert store.count(_HI) == 0
    store.upsert(_HI, "a", "text a", "A", {})
    store.upsert(_HI, "b", "text b", "B", {})
    assert store.count(_HI) == 2
    store.delete(_HI, "a")
    assert store.count(_HI) == 1


def test_list_recent_is_honestly_empty():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "text a", "A", {})
    assert store.list_recent(_HI, limit=5) == []


def test_deterministic_ordering_across_repeated_queries():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "mesh gateway timeout collector", "A", {})
    store.upsert(_HI, "b", "mesh gateway", "B", {})
    store.upsert(_HI, "c", "unrelated content entirely", "C", {})
    first = store.query(_HI, "mesh gateway timeout", top_k=5)
    second = store.query(_HI, "mesh gateway timeout", top_k=5)
    assert [m.record_id for m in first] == [m.record_id for m in second]


def test_score_stays_within_zero_to_one():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "mesh gateway timeout collector route", "A", {})
    store.upsert(_HI, "b", "mesh gateway", "B", {})
    store.upsert(_HI, "c", "totally unrelated content", "C", {})
    results = store.query(_HI, "mesh gateway timeout", top_k=5)
    for match in results:
        assert 0.0 <= match.score <= 1.0


def test_no_query_term_overlap_normalizes_to_zero_not_a_crash():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "completely unrelated content here", "A", {})
    results = store.query(_HI, "mesh gateway timeout", top_k=5)
    # rank_bm25 still returns a (zero) score for every document even
    # with no term overlap -- must not raise, and must stay in [0,1].
    assert len(results) == 1
    assert results[0].score == 0.0


def test_metadata_and_title_are_preserved_verbatim():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "mesh gateway timeout", "Real Title", {"root_cause": "lost route", "tags": "mesh"})
    results = store.query(_HI, "mesh gateway", top_k=5)
    assert results[0].title == "Real Title"
    assert results[0].metadata == {"root_cause": "lost route", "tags": "mesh"}


def test_full_untruncated_text_is_used_for_indexing_not_a_400_char_snippet():
    long_text = "irrelevant filler word " * 50 + "distinctivemeshgatewaytoken"
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", long_text, "A", {})
    # "distinctivemeshgatewaytoken" appears only after the first ~400
    # characters -- if indexing only used a truncated snippet, this
    # query would find nothing.
    assert len(long_text) > 400
    results = store.query(_HI, "distinctivemeshgatewaytoken", top_k=5)
    assert [m.record_id for m in results] == ["a"]


def test_empty_document_text_retains_identity_for_count_and_delete():
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "", "A", {})
    assert store.count(_HI) == 1
    store.delete(_HI, "a")
    assert store.count(_HI) == 0


def test_single_document_corpus_with_real_match_normalizes_to_max_not_zero():
    """Real, verified BM25 property: rank_bm25.BM25Okapi's classic IDF
    term is negative when a query term appears in more than half the
    corpus -- for a single-document corpus, ANY matching term hits this
    (100% > 50%), producing a negative raw score even for a genuine
    match. Score sign alone must not be trusted to distinguish "matched"
    from "didn't match" -- a real match in a degenerate corpus must
    still normalize to 1.0 (found via a direct token-overlap check),
    never collapse to the same 0.0 a genuine non-match gets."""
    store = LexicalKnowledgeStore()
    store.upsert(_HI, "a", "completely different SQL blocking issue", "A", {})
    matched = store.query(_HI, "SQL blocking", top_k=5)
    unmatched = store.query(_HI, "mesh gateway timeout", top_k=5)
    assert matched[0].score == 1.0
    assert unmatched[0].score == 0.0


def test_top_k_limits_result_count():
    store = LexicalKnowledgeStore()
    for i in range(10):
        store.upsert(_HI, f"id-{i}", "mesh gateway timeout", f"Title {i}", {})
    results = store.query(_HI, "mesh gateway timeout", top_k=3)
    assert len(results) == 3
