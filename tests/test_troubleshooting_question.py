"""Tests for app.engines.chat.troubleshooting_question (Chat Assistant
Phase 32).

Pure unit tests -- deterministic pattern matching only, no LLM, no
database. See the module's own docstring for why this is a narrow,
closed-list mechanism rather than a general natural-language parser,
mirroring app.engines.chat.scope_question exactly.
"""

from __future__ import annotations

from app.engines.chat.troubleshooting_question import (
    TROUBLESHOOTING_PHRASES,
    contains_troubleshooting_question,
    split_out_troubleshooting_clause,
)


def test_detects_every_minimum_required_phrase():
    required = [
        "What should I check first?",
        "What should I check?",
        "What do I check first?",
        "How do I troubleshoot this?",
        "What steps should I take first?",
        "What should I do first?",
        "What is the first thing I should check?",
    ]
    for question in required:
        assert contains_troubleshooting_question(question), question


def test_does_not_flag_ordinary_questions():
    ordinary = [
        "Has this happened before?",
        "What is the root cause?",
        "What is the resolution?",
        "Is this confirmed?",
        "Does this affect other customers?",
        "RF Mesh IP command timeout.",
        "Which checkbox did the customer select?",
    ]
    for question in ordinary:
        assert not contains_troubleshooting_question(question), question


def test_split_removes_trailing_troubleshooting_clause():
    remaining, had_troubleshooting = split_out_troubleshooting_clause(
        "Has this happened before, and what should I check first?"
    )
    assert had_troubleshooting is True
    assert remaining == "Has this happened before?"
    assert "check" not in remaining.lower()


def test_split_removes_leading_troubleshooting_clause():
    remaining, had_troubleshooting = split_out_troubleshooting_clause(
        "What should I check first, and has this happened before?"
    )
    assert had_troubleshooting is True
    assert "check" not in remaining.lower()
    assert "happened before" in remaining.lower()


def test_split_removes_mid_sentence_troubleshooting_clause_preserving_the_rest():
    remaining, had_troubleshooting = split_out_troubleshooting_clause(
        "Has this happened before, what should I check first, and what is the root cause?"
    )
    assert had_troubleshooting is True
    assert "check" not in remaining.lower()
    assert "happened before" in remaining.lower()
    assert "root cause" in remaining.lower()


def test_split_on_standalone_troubleshooting_question_returns_empty_remainder():
    remaining, had_troubleshooting = split_out_troubleshooting_clause("What should I check first?")
    assert had_troubleshooting is True
    assert remaining == ""


def test_split_on_non_troubleshooting_question_is_a_no_op():
    original = "Has this happened before, and what is the root cause?"
    remaining, had_troubleshooting = split_out_troubleshooting_clause(original)
    assert had_troubleshooting is False
    assert remaining == original


def test_split_preserves_three_part_question_around_troubleshooting_clause():
    remaining, had_troubleshooting = split_out_troubleshooting_clause(
        "What happened, what should I check first, and what is the resolution?"
    )
    assert had_troubleshooting is True
    assert "check" not in remaining.lower()
    assert "what happened" in remaining.lower()
    assert "resolution" in remaining.lower()


def test_troubleshooting_phrases_use_whole_phrase_matching_not_lone_keywords():
    """A generic question mentioning "check" or "first" in an unrelated
    way must never be misclassified -- TROUBLESHOOTING_PHRASES are
    multi-word phrases, never a single generic word."""
    assert all(len(phrase.split()) >= 3 for phrase in TROUBLESHOOTING_PHRASES)
    assert not contains_troubleshooting_question("Which checkbox did the customer select first?")
    assert not contains_troubleshooting_question("Can you check the ticket number?")


def test_combined_with_scope_clause_removal_preserves_remaining_parts():
    """Chat Assistant Phase 32 Step 9's real finding: after scope-clause
    removal (Phase 31) runs first, troubleshooting-clause removal must
    still correctly isolate a genuinely all-covered multi-part question
    down to an empty remainder, and must never re-introduce the scope
    clause it already dropped."""
    from app.engines.chat.scope_question import split_out_scope_clause

    after_scope, had_scope = split_out_scope_clause(
        "What should I check first, and does this affect other customers?"
    )
    remaining, had_troubleshooting = split_out_troubleshooting_clause(after_scope)
    assert had_scope is True
    assert had_troubleshooting is True
    assert remaining == ""
