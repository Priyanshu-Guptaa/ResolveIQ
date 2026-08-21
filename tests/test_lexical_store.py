"""Tests for LexicalKnowledgeStore (Chat Assistant Phase 2 -- Hybrid
Retrieval + RRF foundation).

Pure unit tests -- no database, no HTTP, no ChromaDB, no embedding
model. Exercises the real rank_bm25 library directly (in-memory, no
network, no external process).
"""

from __future__ import annotations

from app.domain.enums import KnowledgeCollection
from app.engines.knowledge.lexical_store import BM25_STOPWORDS, LexicalKnowledgeStore, _tokenize

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


# =============================================================================
# Phase 2A -- BM25 tokenizer calibration (real-corpus-evidence-grounded).
# Tests _tokenize() directly (pure function, no I/O) as well as through
# LexicalKnowledgeStore.query() where the retrieval-level effect matters.
# =============================================================================

# --- Stopwords ---------------------------------------------------------------


def test_generic_function_word_is_removed():
    assert "the" not in _tokenize("the mesh gateway")
    assert "with" not in _tokenize("connect with the gateway")


def test_not_retained():
    assert "not" in _tokenize("meter did not respond")


def test_no_retained():
    assert "no" in _tokenize("no response from collector")


def test_without_retained():
    assert "without" in _tokenize("meter without response")


def test_failed_retained():
    assert "failed" in _tokenize("command failed to execute")


def test_failure_retained():
    assert "failure" in _tokenize("connection failure detected")


def test_missing_retained():
    assert "missing" in _tokenize("meter number missing in CC")


def test_unavailable_retained():
    assert "unavailable" in _tokenize("service unavailable")


def test_stuck_retained():
    assert "stuck" in _tokenize("meters stuck in discovered status")


def test_timeout_retained():
    assert "timeout" in _tokenize("collector command timeout")


def test_error_retained():
    assert "error" in _tokenize("SQL error occurred")


def test_bm25_stopwords_excludes_every_troubleshooting_critical_word():
    protected = {
        "not", "no", "without", "failed", "failure", "missing", "unavailable",
        "stuck", "timeout", "never", "unable", "down", "error", "fail", "fails",
        "failing", "issue", "problem", "lost", "disconnected", "refused",
        "denied", "invalid", "broken", "absent",
    }
    assert BM25_STOPWORDS.isdisjoint(protected)


def test_bm25_stopwords_is_independent_of_recommendation_engine_list():
    # BM25_STOPWORDS must be independently authored, not an import/alias of
    # RecommendationEngine._ENGLISH_STOPWORDS -- concretely proven by the one
    # real, deliberate divergence: "without" is in that list, not in this one.
    assert "without" not in BM25_STOPWORDS


# --- CamelCase ----------------------------------------------------------------


def test_camelcase_command_timeout_splits_and_retains_original():
    tokens = _tokenize("CommandTimeout")
    assert "commandtimeout" in tokens
    assert "command" in tokens
    assert "timeout" in tokens


def test_camelcase_null_pointer_exception_splits_and_retains_original():
    tokens = _tokenize("NullPointerException")
    assert "nullpointerexception" in tokens
    assert {"null", "pointer", "exception"} <= set(tokens)


def test_camelcase_inbound_message_processor_splits_and_retains_original():
    tokens = _tokenize("InboundMessageProcessor")
    assert "inboundmessageprocessor" in tokens
    assert {"inbound", "message", "processor"} <= set(tokens)


def test_camelcase_original_unsplit_token_always_present():
    # Explicit, dedicated assertion on the preservation guarantee itself --
    # not merely incidental to the split tests above.
    for word in ("CommandTimeout", "NullPointerException", "APIResponse", "CSTASK0078039"):
        assert word.lower() in _tokenize(word)


def test_acronym_led_camelcase_is_not_split():
    # Deliberate scope decision (Phase 2A approved design): no real corpus
    # evidence for this exact pattern; the simple boundary rule naturally
    # leaves it whole with no special-casing needed.
    assert _tokenize("APIResponse") == ["apiresponse"]
    assert _tokenize("HTTPClient") == ["httpclient"]
    assert _tokenize("SQLServer") == ["sqlserver"]


# --- Versions ------------------------------------------------------------------


def test_version_v3_4_0_preserved_as_whole_token():
    tokens = _tokenize("v3.4.0")
    assert "v3.4.0" in tokens


def test_version_ss8_6_1_463_preserved_as_whole_token():
    tokens = _tokenize("SS8.6.1.463")
    assert "ss8.6.1.463" in tokens


def test_version_1_8_2_preserved_as_whole_token():
    tokens = _tokenize("1.8.2")
    assert "1.8.2" in tokens


def test_version_fragments_still_present_alongside_whole_token():
    # Additive guarantee: the whole-span token never replaces the
    # already-existing fragment-level tokens.
    tokens = _tokenize("SS8.6.1.463")
    assert tokens == ["ss8", "6", "1", "463", "ss8.6.1.463"]


def test_ordinary_decimal_also_matches_version_pattern_documented_tradeoff():
    # Honest, documented ambiguity (Phase 2A approved design): a plain
    # decimal metric is structurally identical to a short real version
    # string (e.g. "1.9") -- not fixable without losing real required
    # examples or adding semantic knowledge this tokenizer doesn't have.
    tokens = _tokenize("0.026")
    assert "0.026" in tokens


def test_long_garbage_digit_sequence_does_not_preserve_the_full_span():
    garbage = "0.0.0.0.0.1.301.0.0.0.0.0.0.0.0.0.108.0"
    tokens = _tokenize(garbage)
    assert garbage not in tokens
    assert all(len(t) <= 20 for t in tokens)


# --- Existing behavior (regression guard) --------------------------------------


def test_ticket_identifier_remains_single_token_unaffected_by_camelcase_or_version_rules():
    assert _tokenize("CSTASK0078039") == ["cstask0078039"]


def test_hyphenated_terms_unaffected():
    assert _tokenize("Wi-Sun") == ["wi", "sun"]
    assert _tokenize("order-service") == ["order", "service"]
    assert _tokenize("checkout-api") == ["checkout", "api"]


def test_case_insensitivity_still_holds():
    assert _tokenize("MESH gateway") == _tokenize("mesh GATEWAY")


def test_punctuation_still_handled():
    assert _tokenize("Command Request (Outbound)") == ["command", "request", "outbound"]


# --- Determinism -----------------------------------------------------------------


def test_tokenize_is_deterministic_across_repeated_calls():
    text = "TEPCO RF Mesh IP CommandTimeout on collector SS8.6.1.463, ticket CSTASK0078039."
    assert _tokenize(text) == _tokenize(text)


# --- Frequency (the mandatory regression test) ------------------------------------


def test_camelcase_split_does_not_double_count_a_separately_occurring_standalone_word():
    text = "CommandTimeout error. The operation experienced a timeout."
    tokens = _tokenize(text)
    assert tokens.count("timeout") == 2  # once from CommandTimeout, once standalone -- never three


def test_naturally_repeated_word_frequency_is_preserved():
    tokens = _tokenize("timeout timeout")
    assert tokens.count("timeout") == 2
