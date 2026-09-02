"""Tests for app.engines.chat.log_question (Chat + Log Intelligence
integration).

Pure unit tests -- deterministic pattern matching only, no parsing, no
LLM. See the module's own docstring for the real usage gap this closes
(an uploaded log's real, already-parsed detail was never surfaced in
Chat's own answer, only aggregate counts).
"""

from __future__ import annotations

from app.engines.chat.log_question import contains_log_analysis_question


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
