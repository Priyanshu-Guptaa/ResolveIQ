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

import re

from app.engines.shared.text_matching import phrase_present

_STUCK_STATE_RE = re.compile(r"\b(?:stuck|stucked)\s+in\b", re.IGNORECASE)
"""Knowledge Answering & Evidence Synthesis phase -- the real reported
gap: "AxeI meters are stuck in discovered, how do I make it normal?"
matched none of this module's original phrase list (all "why"/"what
caused"/"what went wrong" explanation requests) -- a STATE complaint
("stuck in <state>") is just as clearly a troubleshooting request, and
just as clearly needs the ranked-hypothesis answer, not the bare
tier-based boilerplate. Variable-middle (any state name), so a fixed
phrase list can't express it -- a closed regex anchored on the fixed
"stuck in" idiom itself, never a general heuristic."""

_HOW_TO_RECOVER_PHRASES: list[str] = [
    "how do i make it normal",
    "how do i make this normal",
    "how do i fix this",
    "how do i fix it",
    "how do i resolve this",
    "how do i resolve it",
    "how can i fix this",
    "how can i fix it",
    "how can i resolve this",
    "how can i resolve it",
    "how do i clear this",
    "how do i clear it",
    "how do i recover this",
    "how do i recover it",
]
"""The companion recovery-request half of the same real gap -- "how do
I make it normal" is a request for a resolution, not an explanation,
but it is exactly the kind of request this module's ranked-hypothesis
answer (observed state + likely causes + next checks) is built to
serve, same as the "why did this fail" phrases below."""

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
    ``TROUBLESHOOTING_SYNTHESIS_PHRASES``/``_HOW_TO_RECOVER_PHRASES``,
    or the "stuck in <state>" regex. Conservative by construction:
    only ever True on an exact, closed-list phrase match or that one
    narrowly-scoped regex, never a heuristic guess."""
    if any(phrase_present(phrase, question) for phrase in TROUBLESHOOTING_SYNTHESIS_PHRASES):
        return True
    if any(phrase_present(phrase, question) for phrase in _HOW_TO_RECOVER_PHRASES):
        return True
    return _STUCK_STATE_RE.search(question) is not None
