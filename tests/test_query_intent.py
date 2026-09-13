"""Tests for app.engines.chat.query_intent (Knowledge Answering &
Evidence Synthesis phase). Unit-level, no database -- pure-function
tests against ``classify_intent``/``build_query_context``.
"""

from __future__ import annotations

from app.engines.chat.query_intent import AnswerIntent, build_query_context, classify_intent


def test_entity_definition_singular_what_is():
    assert classify_intent("What is AxeI meter?") == AnswerIntent.ENTITY_DEFINITION


def test_product_explanation_how_does_work():
    assert classify_intent("How does AxeI work?") == AnswerIntent.PRODUCT_EXPLANATION


def test_concept_explanation_plural_what_are():
    assert classify_intent("What are process settings in CC?") == AnswerIntent.CONCEPT_EXPLANATION


def test_configuration_how_do_i_configure():
    assert classify_intent("How do I configure process settings?") == AnswerIntent.CONFIGURATION


def test_troubleshooting_stuck_in_state():
    intent = classify_intent("AxeI meters are stuck in discovered how do I make it normal?")
    assert intent == AnswerIntent.TROUBLESHOOTING


def test_root_cause_why_did_x_fail_with_inserted_noun():
    assert classify_intent("Why did this meter fail?") == AnswerIntent.ROOT_CAUSE


def test_historical_lookup():
    assert classify_intent("Has this happened before?") == AnswerIntent.HISTORICAL_LOOKUP


def test_log_analysis_requires_log_evidence_flag():
    assert classify_intent("Analyze this log", has_log_evidence=True) == AnswerIntent.LOG_ANALYSIS
    # Without real log evidence present, the same text has nothing to analyze.
    assert classify_intent("Analyze this log", has_log_evidence=False) != AnswerIntent.LOG_ANALYSIS


def test_l2_task_notes_requires_log_evidence():
    assert classify_intent("Give me L2 task notes", has_log_evidence=True) == AnswerIntent.L2_TASK_NOTES


def test_l3_escalation_requires_log_evidence():
    assert classify_intent("Prepare an L3 escalation", has_log_evidence=True) == AnswerIntent.L3_ESCALATION


def test_root_cause_question_never_misclassified_as_a_knowledge_question():
    """Rule 3/4's tier-preservation guard: "What is the root cause?"
    must never be treated as a definition/knowledge question -- it
    must reach the existing, unmodified tier-based composer."""
    assert classify_intent("What is the root cause?") == AnswerIntent.UNKNOWN
    assert classify_intent("What is the resolution?") == AnswerIntent.UNKNOWN
    assert classify_intent("What is the confidence?") == AnswerIntent.UNKNOWN


def test_troubleshooting_what_should_i_check():
    assert classify_intent("What should I check first?") == AnswerIntent.TROUBLESHOOTING


def test_comparison():
    assert classify_intent("What is the difference between AxeI and Focus AX meters?") == AnswerIntent.COMPARISON


def test_build_query_context_extracts_state_and_subject():
    ctx = build_query_context("AxeI meters are stuck in discovered how do I make it normal?")
    assert ctx.intent == AnswerIntent.TROUBLESHOOTING
    assert ctx.state == "Discovered"
    assert ctx.subject is not None and "axei" in ctx.subject


def test_build_query_context_never_invents_a_state_when_none_is_named():
    ctx = build_query_context("What is AxeI meter?")
    assert ctx.state is None


def test_build_query_context_is_definitional_flag():
    assert build_query_context("What is AxeI meter?").is_definitional is True
    assert build_query_context("How does AxeI work?").is_definitional is True
    assert build_query_context("What are process settings in CC?").is_definitional is False
    assert build_query_context("Why did this meter fail?").is_definitional is False


def test_as_debug_dict_is_plain_serializable_data():
    ctx = build_query_context("What is AxeI meter?")
    d = ctx.as_debug_dict()
    assert d["intent"] == "entity_definition"
    assert isinstance(d, dict)
