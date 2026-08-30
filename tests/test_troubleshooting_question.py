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


# --- Chat Assistant Phase 47 -- "how (do/should) i troubleshoot this <noun>" -
# --- input-parsing fix. Reproduced real bug: the closed phrase table only  --
# --- had "how should i troubleshoot this" (bare "this"), so a natural      --
# --- phrasing like "...this issue?" left the meaningless remainder         --
# --- "issue?" rather than being recognized as a complete troubleshooting-  --
# --- only question. Already fully mitigated for SAFETY by Phase 46's       --
# --- output-side guard (which runs regardless of what question text        --
# --- reaches the LLM) -- this fix is a quality/completeness improvement    --
# --- to the input-side detector, not a new safety mechanism.


def test_troubleshoot_this_issue_phrasing_is_recognized_completely():
    """Test A/B/C (Phase 47) -- the exact reported phrasing is now
    detected, leaves NO meaningless remainder, and follows the same
    empty-remainder path as every other fully-covered troubleshooting-
    only question (see test_troubleshooting_only_question_skips_the_
    llm_entirely in test_chat_orchestrator.py for the end-to-end,
    no-LLM-call confirmation)."""
    remaining, had = split_out_troubleshooting_clause("How should I troubleshoot this issue?")
    assert had is True
    assert remaining == ""
    assert contains_troubleshooting_question("How should I troubleshoot this issue?")


def test_troubleshoot_this_problem_and_situation_variants_also_fully_consumed():
    for noun in ("issue", "problem", "situation"):
        for verb_phrase in ("how do i troubleshoot", "how should i troubleshoot"):
            question = f"{verb_phrase.capitalize()} this {noun}?"
            remaining, had = split_out_troubleshooting_clause(question)
            assert had is True, question
            assert remaining == "", (question, remaining)


def test_bare_troubleshoot_this_without_a_trailing_noun_still_works_unchanged():
    """Test D (Phase 47) -- the existing, pre-fix phrasing (no trailing
    noun at all) must continue to behave identically -- this fix must
    be purely additive, never a behavior change for what already
    worked."""
    for question in ("How do I troubleshoot this?", "How should I troubleshoot this?"):
        remaining, had = split_out_troubleshooting_clause(question)
        assert had is True, question
        assert remaining == "", question


def test_multipart_question_with_the_issue_variant_preserves_the_other_part():
    """The fix must correctly isolate just the troubleshooting clause
    even when it uses the new "this issue" phrasing inside a real
    multi-part question -- mirroring test_split_preserves_three_part_
    question_around_troubleshooting_clause's existing discipline."""
    remaining, had = split_out_troubleshooting_clause(
        "Has this happened before, and how should I troubleshoot this issue?"
    )
    assert had is True
    assert remaining == "Has this happened before?"


def test_ordinary_questions_mentioning_issue_or_problem_are_not_misclassified():
    """Test E (Phase 47) -- a question that merely contains the words
    "issue"/"problem"/"troubleshoot"/"check"/"steps" without matching one
    of the exact closed-list phrases must never be misclassified."""
    ordinary = [
        "This issue affects other customers.",
        "What is the root cause of this problem?",
        "Is this issue confirmed?",
        "What troubleshooting has already been done for this issue?",
        "Can you check the ticket for this issue?",
        "This has been a recurring problem.",
    ]
    for question in ordinary:
        assert not contains_troubleshooting_question(question), question


def test_new_phrases_are_still_whole_phrase_multi_word_entries():
    """The new entries must uphold the same whole-phrase discipline
    test_troubleshooting_phrases_use_whole_phrase_matching_not_lone_
    keywords already asserts for the full table."""
    new_entries = [p for p in TROUBLESHOOTING_PHRASES if p.endswith(("issue", "problem", "situation"))]
    assert len(new_entries) == 6
    assert all(len(phrase.split()) >= 5 for phrase in new_entries)
