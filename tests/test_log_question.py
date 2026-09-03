"""Tests for app.engines.chat.log_question (Chat + Log Intelligence
integration).

Pure unit tests -- deterministic pattern matching only, no parsing, no
LLM. See the module's own docstring for the real usage gap this closes
(an uploaded log's real, already-parsed detail was never surfaced in
Chat's own answer, only aggregate counts).
"""

from __future__ import annotations

from app.engines.chat.log_question import (
    contains_l2_task_note_question,
    contains_l3_escalation_question,
    contains_log_analysis_question,
    contains_log_comparison_question,
)


def test_log_analysis_phrases_are_detected():
    log_analysis = [
        "Analyze this log.",
        "Analyze this meter log.",
        "Analyse this log.",
        "What happened in this log?",
        "What happened here?",
        "Give me a summary of this meter communication.",
        "Summarize this log.",
        "What errors do you see?",
        "What errors are present?",
        "Where did the communication fail?",
        "Which command failed?",
        "What happened before the failure?",
        "What happened after the failure?",
        "Is there a timeout?",
        "Are there retries?",
        "Did the meter respond?",
        "What identifiers are present?",
        "Show me the timeline.",
        "Give me the timeline.",
        "What should I check next?",
        "Is this a known issue?",
        "Is this related to a known issue?",
        "Have we seen this before?",
        "Have we seen it before?",
    ]
    for question in log_analysis:
        assert contains_log_analysis_question(question), question


def test_ordinary_questions_are_not_misclassified():
    ordinary = [
        "What is the root cause?",
        "What is NMM?",
        "Tell me about dashboard in CC.",
        "Does this affect other customers?",
        "This has been a recurring problem.",
    ]
    for question in ordinary:
        assert not contains_log_analysis_question(question), question


def test_new_log_analysis_phrases_are_detected():
    """L2/L3 Investigation Copilot phase -- additional phrasing (§12 of
    that phase's own example question list)."""
    log_analysis = [
        "What caused the timeout?",
        "Why did the command fail?",
        "Did the retry succeed?",
        "Was the retry successful?",
        "Which request caused the failure?",
        "Show me the request/response flow.",
        "What is the likely failure point?",
        "Which meter is affected?",
        "Which meters are affected?",
        "Are multiple meters showing the same problem?",
        "Find all correlation IDs.",
        "Find all command IDs.",
        "Show me all errors related to this meter.",
        "Does the log indicate a communication problem?",
        "What happened to the meter?",
        "What happened to meter 12345?",
    ]
    for question in log_analysis:
        assert contains_log_analysis_question(question), question


def test_log_comparison_phrases_are_detected():
    comparison = [
        "Compare these logs.",
        "Compare these two logs.",
        "Which one failed?",
        "Which log failed?",
        "What is common between them?",
        "Show differences.",
        "Show me the differences.",
    ]
    for question in comparison:
        assert contains_log_comparison_question(question), question


def test_l2_and_l3_phrases_are_detected():
    l2 = ["Give me L2 task notes.", "Give me an L2 task-note summary.", "L2 summary."]
    for question in l2:
        assert contains_l2_task_note_question(question), question

    l3 = ["Prepare an L3 escalation.", "L3 escalation summary.", "Give me an L3 escalation."]
    for question in l3:
        assert contains_l3_escalation_question(question), question


def test_comparison_and_l2_l3_never_misclassify_ordinary_log_analysis_questions():
    ordinary_log_analysis = ["Analyze this log.", "Show me the timeline.", "What errors do you see?"]
    for question in ordinary_log_analysis:
        assert not contains_log_comparison_question(question), question
        assert not contains_l2_task_note_question(question), question
        assert not contains_l3_escalation_question(question), question
