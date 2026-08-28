"""Tests for app.engines.chat.scope_question (Chat Assistant Phase 31).

Pure unit tests -- deterministic pattern matching only, no LLM, no
database. See the module's own docstring for why this is a narrow,
closed-list mechanism rather than a general natural-language parser.
"""

from __future__ import annotations

from app.engines.chat.scope_question import SCOPE_PHRASES, contains_scope_question, split_out_scope_clause


def test_detects_every_minimum_required_phrase():
    required = [
        "does this affect other customers?",
        "are other customers affected?",
        "is this limited to this customer?",
        "does this affect anyone else?",
        "which other customers are affected?",
        "what other customers are affected?",
        "who else is affected?",
    ]
    for question in required:
        assert contains_scope_question(question), question


def test_detects_positive_control_phrasing():
    assert contains_scope_question("Which customers are explicitly shown to be affected?")


def test_does_not_flag_ordinary_questions():
    ordinary = [
        "Has this happened before?",
        "What should I check first?",
        "What is the root cause?",
        "What is the resolution?",
        "Is this confirmed?",
        "Has this been seen with this customer before?",
        "RF Mesh IP command timeout.",
    ]
    for question in ordinary:
        assert not contains_scope_question(question), question


def test_split_removes_trailing_scope_clause():
    remaining, had_scope = split_out_scope_clause(
        "Has this happened before, and does this affect other customers?"
    )
    assert had_scope is True
    assert remaining == "Has this happened before?"
    assert "other customers" not in remaining.lower()


def test_split_removes_leading_scope_clause():
    remaining, had_scope = split_out_scope_clause(
        "Does this affect other customers, and has this happened before?"
    )
    assert had_scope is True
    assert "other customers" not in remaining.lower()
    assert "happened before" in remaining.lower()


def test_split_removes_mid_sentence_scope_clause_preserving_the_rest():
    remaining, had_scope = split_out_scope_clause(
        "Has this happened before, does this affect other customers, and what should I check first?"
    )
    assert had_scope is True
    assert "other customers" not in remaining.lower()
    assert "happened before" in remaining.lower()
    assert "check first" in remaining.lower()


def test_split_on_standalone_scope_question_returns_empty_remainder():
    remaining, had_scope = split_out_scope_clause("Does this affect other customers?")
    assert had_scope is True
    assert remaining == ""


def test_split_on_non_scope_question_is_a_no_op():
    original = "Has this happened before, and what should I check first?"
    remaining, had_scope = split_out_scope_clause(original)
    assert had_scope is False
    assert remaining == original


def test_split_preserves_three_part_question_around_scope_clause():
    remaining, had_scope = split_out_scope_clause(
        "What happened, has this happened before, and does this affect other customers?"
    )
    assert had_scope is True
    assert "other customers" not in remaining.lower()
    assert "what happened" in remaining.lower()
    assert "happened before" in remaining.lower()


def test_scope_phrases_use_whole_phrase_matching_not_lone_keywords():
    """A generic question mentioning "customer" or "affected" in an
    unrelated way must never be misclassified -- SCOPE_PHRASES are
    multi-word phrases, never a single generic word."""
    assert all(len(phrase.split()) >= 3 for phrase in SCOPE_PHRASES)
    assert not contains_scope_question("Which customer reported this issue?")
    assert not contains_scope_question("How is the customer affected by the outage duration?")
