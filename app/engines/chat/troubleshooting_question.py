"""Troubleshooting-request question detection and removal (Chat
Assistant Phase 32).

Phase 25 added a prompt-only guard (``PromptBuilder`` rule 9 + the
``AVAILABLE EVIDENCE-BACKED CHECKS`` field) meant to stop qwen2.5:3b
from inventing generic troubleshooting advice ("check the power
supply", "check connections") when no evidence-backed check exists.
Phase 32 found this guard fails at almost exactly the rate customer-
scope fabrication did before Phase 31's fix: 39/40 fresh real calls
(97.5%) on a fixture with a concrete root cause but zero resolution
candidates/validation steps fabricated a generic check anyway, most of
them stating the fabrication AND the correct "no evidence-backed check
exists" disclaimer in the same answer. Five stronger prompt-only
variants (an emphatic empty-check declaration, a structured
``CHECK_STATUS``/``ALLOWED_ACTIONS`` schema, withholding the raw root-
cause text, and combinations of these) were benchmarked at 240 more
real calls and never dropped below 60% unsafe; withholding the root-
cause text made it WORSE (100% unsafe), proving the fabrication is not
triggered by any specific root-cause wording -- it is a general
compulsion to answer "what should I check" with *something* regardless
of evidence. A root-cause-association attack (50 more real calls, five
unrelated root-cause wordings) confirmed this: 50/50 fabricated a
topically-matched generic check every time.

Only removing the troubleshooting sub-question from the LLM's input
entirely -- the same fix Phase 31 already validated for customer-scope
fabrication -- eliminated it in every real-call condition Phase 32
tested, while a genuine-checks positive control (20 real calls)
confirmed real evidence-backed checks are still phrased correctly,
unmodified, when they exist (so this is a gate on the "zero checks"
case specifically, never a blanket suppression of troubleshooting
answers).

This module is that "narrowest safe mechanism" for troubleshooting
requests, following the exact same idiom
``app.engines.chat.scope_question`` already established (itself
following ``QueryUnderstandingEngine``'s own closed-phrase-table
intent classifier): deterministic pattern matching, a fixed phrase
table, never an LLM, conservative default (trades recall for zero
false positives). See that module's docstring for the full rationale;
it is not repeated here beyond what differs.

Unlike scope removal (always applied), the troubleshooting clause is
only ever removed when ``ChatOrchestrator`` has already determined,
deterministically, that zero evidence-backed checks exist for this
answer (``bool(app.engines.llm.prompt_builder.available_checks(...))``
is ``False``) -- when real checks exist, Phase 32 Step 8's positive
control showed the existing, unmodified Rule 9 already handles the
question safely, and removing the clause in that case would only
suppress a legitimate answer for no safety benefit.
"""

from __future__ import annotations

import re

from app.engines.shared.text_matching import phrase_present

TROUBLESHOOTING_PHRASES: list[str] = [
    "what should i check first",
    "what should i check next",
    "what should i check",
    "what do i check first",
    "what do i check",
    "what can i check first",
    "what can i check",
    "what to check first",
    "how do i troubleshoot this",
    "how should i troubleshoot this",
    "how do i troubleshoot",
    "where should i start troubleshooting",
    "how do i start troubleshooting",
    "what troubleshooting steps should i take",
    "what steps should i take first",
    "what steps should i take",
    "what should i do first",
    "what should i do to fix this",
    "what is the first thing i should check",
    "what is the first step to take",
]
"""Closed phrase table -- see module docstring. Each entry is checked as
a whole phrase (``phrase_present``), never a lone significant word, so
an unrelated question mentioning "check" or "first" alone is never
misclassified."""

_PHRASES_PATTERN = "|".join(re.escape(p) for p in TROUBLESHOOTING_PHRASES)
_TROUBLESHOOTING_CLAUSE_RE = re.compile(
    rf"(?:,?\s*\band\b\s*|^\s*|,\s*)({_PHRASES_PATTERN})\s*\??",
    re.IGNORECASE,
)


def contains_troubleshooting_question(question: str) -> bool:
    """True if ``question`` contains any whole-phrase match from
    ``TROUBLESHOOTING_PHRASES``. Conservative by construction: only
    ever True on an exact, closed-list phrase match, never a heuristic
    guess."""
    return any(phrase_present(phrase, question) for phrase in TROUBLESHOOTING_PHRASES)


def split_out_troubleshooting_clause(question: str) -> tuple[str, bool]:
    """Removes the FIRST troubleshooting-request clause found in
    ``question`` (there is normally at most one) and returns
    ``(remaining_question, had_troubleshooting_clause)``.

    Handles this project's established compound-question shapes
    (leading, trailing, or mid-sentence clause joined by "and" or a
    comma) exactly like ``split_out_scope_clause``, by removing the
    matched phrase together with its connecting conjunction/comma, then
    normalizing the leftover punctuation. When the ENTIRE question is
    the troubleshooting clause, the remaining text is an empty string --
    callers must handle that case (see
    ``ChatOrchestrator._generate_answer``: no LLM call is made at all
    when nothing else remains to ask).

    Callers must only invoke this when zero evidence-backed checks
    exist (see module docstring) -- it has no opinion of its own about
    check availability and will remove the clause unconditionally
    whenever asked."""
    match = _TROUBLESHOOTING_CLAUSE_RE.search(question)
    if not match:
        return question, False
    remaining = question[: match.start()] + question[match.end() :]
    remaining = re.sub(r"^\s*,\s*and\s+", "", remaining, flags=re.IGNORECASE)
    remaining = re.sub(r"\s*,\s*and\s*$", "", remaining, flags=re.IGNORECASE)
    remaining = re.sub(r",\s*,", ",", remaining)
    remaining = re.sub(r"\s+,", ",", remaining)
    remaining = remaining.strip().strip(",").strip()
    if remaining and not remaining.endswith("?"):
        remaining += "?"
    return remaining, True
