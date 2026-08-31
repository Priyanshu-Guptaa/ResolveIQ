"""Tests for app.engines.chat.knowledge_question (Chat Knowledge-
Synthesis feature).

Pure unit tests -- deterministic pattern matching only, no LLM, no
database. See the module's own docstring for the real usage finding
this closes (an informational question like "tell me about dashboard
in CC" was answered with generic "evidence is insufficient" boilerplate
even though real, relevant documentation existed) and for why bare
"what is X" needed a reserved-lead-in exclusion list rather than a
blanket match.
"""

from __future__ import annotations

from app.engines.chat.knowledge_question import contains_knowledge_question


def test_informational_lead_ins_are_detected():
    informational = [
        "Tell me about dashboard in CC",
        "Tell me more about NMM",
        "Explain the different dashboard views.",
        "Describe the dashboard feature",
        "Give me an overview of process settings",
        "Give me a summary of the CC installation guide",
        "What documentation do we have for dashboard?",
        "What docs do we have on NMM",
        "What documentation is available for process settings",
        "What documentation exists about the dashboard",
        "Do we have any information about process setting in CC?",
        "Do we have information about NMM",
        "Do we have any documentation about dashboard",
        "Do we have documentation about process settings",
        "Do we have documentation on dashboard views",
        "What do we know about NMM",
        "What is known about dashboard configuration",
    ]
    for question in informational:
        assert contains_knowledge_question(question), question


def test_variable_middle_constructs_are_detected():
    variable_middle = [
        "What does process setting mean in CC?",
        "What does NMM mean?",
        "How does CC dashboard work?",
        "How does the collector command queue work?",
        "Have we seen dashboard issues in CC before?",
        "Have we seen this kind of failure before?",
    ]
    for question in variable_middle:
        assert contains_knowledge_question(question), question


def test_historical_phrases_are_detected():
    historical = [
        "Has this happened before?",
        "Have we seen this before?",
        "Any previous cases related to this?",
        "Any prior cases?",
        "Any past cases on record?",
        "Seen this before?",
        "Seen before?",
    ]
    for question in historical:
        assert contains_knowledge_question(question), question


def test_bare_what_is_is_informational_when_not_a_reserved_investigation_noun():
    informational = [
        "What is NMM?",
        "What's NMM?",
        "What is a collector?",
        "What is AMI Dashboard?",
    ]
    for question in informational:
        assert contains_knowledge_question(question), question


def test_bare_what_is_is_never_redirected_for_reserved_investigation_nouns():
    """The real safety-critical exclusion: these must reach the
    existing, unmodified tier-based composer -- never the knowledge
    synthesizer -- so Rule 3/4's tier-preservation and Rule 9's
    troubleshooting gate are never bypassed."""
    investigation_questions = [
        "What is the root cause?",
        "What is the likely root cause?",
        "What is the resolution?",
        "What is the likely resolution?",
        "What is the confidence?",
        "What is the evidence?",
        "What is the applicability?",
        "What is the customer?",
        "What is the scope?",
        "What is the tier?",
        "What's confirmed?",
        "What's verified?",
    ]
    for question in investigation_questions:
        assert not contains_knowledge_question(question), question


def test_troubleshooting_and_scope_questions_are_never_misclassified():
    ordinary = [
        "What should I check?",
        "What should I check first?",
        "How do I troubleshoot this?",
        "How should I troubleshoot this issue?",
        "What steps should I take?",
        "Does this affect other customers?",
        "Is this confirmed?",
        "What evidence supports this?",
        "Why are these meters stuck in discovered?",
        "What is the likely root cause, and what should I check first?",
    ]
    for question in ordinary:
        assert not contains_knowledge_question(question), question


def test_ordinary_unrelated_questions_are_not_misclassified():
    ordinary = [
        "Commands stuck in Pending status.",
        "RF Mesh IP command timeout.",
        "TEPCO reported an issue.",
    ]
    for question in ordinary:
        assert not contains_knowledge_question(question), question
