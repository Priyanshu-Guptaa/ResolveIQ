""""Why did this fail?"-style question detection (Final Hardening
Pass, Objective 2).

Deliberately a DIFFERENT closed phrase list from ``app.engines.chat.
troubleshooting_question.TROUBLESHOOTING_PHRASES`` (confirmed by
reading that module in full before writing this one, per this phase's
own Step 1 audit instruction): that module recognizes "what should I
check"/"how do I troubleshoot" -- a request for a NEXT ACTION, used to
strip the clause from the LLM's input when zero evidence-backed checks
exist. This module recognizes "why did this fail"/"what caused this"
-- a request for an EXPLANATION, used (see ``ChatOrchestrator.
_compose_troubleshooting_synthesis``) to trigger a richer, ranked-
hypothesis deterministic answer instead of the older single-paragraph
tier-based text. The two never fire on the same real question text (no
entry in either list is a substring of an entry in the other), so
there is no ordering dependency between them.

Same "closed list, whole-phrase match, never an LLM, conservative
default" idiom every other question classifier in this codebase
already uses (``knowledge_question.py``, ``log_question.py``,
``troubleshooting_question.py``)."""

from __future__ import annotations

from app.engines.shared.text_matching import phrase_present

TROUBLESHOOTING_SYNTHESIS_PHRASES: list[str] = [
    "why did this fail",
    "why did it fail",
    "why did that fail",
    "why did this request fail",
    "why did the request fail",
    "why is this failing",
    "why is it failing",
    "why does this fail",
    "why does it fail",
    "what caused this failure",
    "what caused the failure",
    "what is causing this failure",
    "what is causing the failure",
    "why did this happen",
    "why did that happen",
    "what went wrong",
    "why is this not working",
    "why isn't this working",
    "why is it not working",
    "why isn't it working",
    "what is causing this issue",
    "what is causing the issue",
    "why is this occurring",
    "why did this error occur",
]
"""Closed phrase table -- see module docstring. Every entry is an
EXPLANATION request ("why"/"what caused"/"what went wrong"), never an
ACTION request ("what should I check") -- the latter stays exclusively
``troubleshooting_question.py``'s concern."""


def contains_troubleshooting_synthesis_question(question: str) -> bool:
    """True if ``question`` contains any whole-phrase match from
    ``TROUBLESHOOTING_SYNTHESIS_PHRASES``. Conservative by
    construction: only ever True on an exact, closed-list phrase
    match, never a heuristic guess."""
    return any(phrase_present(phrase, question) for phrase in TROUBLESHOOTING_SYNTHESIS_PHRASES)
