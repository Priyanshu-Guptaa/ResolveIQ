"""Customer-impact-scope question detection and removal (Chat Assistant
Phase 31).

Four straight phases (27-30) tried to make qwen2.5:3b safely PHRASE a
customer-impact-scope conclusion, or safely SKIP answering one, purely
through prompt instructions -- a prose header, a compact status token,
a literal precomputed answer with an explicit "do not touch it"
instruction, and finally a rule-8 exception telling the model outright
not to answer the scope part of a question it could still plainly read.
Every one of those failed to reliably prevent fabrication; the last
(Phase 30) measurably made it WORSE (70% and 55% unsafe on the two hard
cases, up from Phase 29's ~10%/~30%) because the model kept trying to
answer the literal question text regardless of the instruction not to,
and without even the (imperfect) dedicated field Phase 29 gave it, it
had nothing to draw from but the confidence tier.

Phase 31's decisive real-call experiment (20 calls, CASE_B_MULTIPART)
tested the one mechanism not yet tried: never showing the model the
scope-related clause at all. Result: 0/20 unsafe -- every single
response either correctly answered the remaining (non-scope) question
or said "UNKNOWN", with zero mentions of "other customers" or any
scope-adjacent claim, because the model was never asked and had nothing
to react to.

This module is the "narrowest safe mechanism" Step 3 requires -- NOT a
general natural-language parser. It follows the exact "deterministic
pattern matching, fixed phrase table, never an LLM, conservative
default" idiom ``QueryUnderstandingEngine``'s own intent classifier
(``_INTENT_PATTERNS``) already established in this codebase, reusing
its underlying whole-phrase, word-boundary-safe primitive
(``app.engines.shared.text_matching.phrase_present``) rather than a
second implementation of phrase matching. The phrase table below is a
closed list (Phase 31 Step 3's specified minimum set plus the
Step 7 positive-control phrasing), not an attempt at exhaustive
natural-language coverage -- a question phrased outside this list
that asks about customer scope will not be detected, and will be
answered (or not) exactly as Phase 30's code already does. That is a
known, explicit limitation, not a silent gap: this mechanism trades
recall for zero false positives, matching every other deterministic
classifier already in this codebase.
"""

from __future__ import annotations

import re

from app.engines.shared.text_matching import phrase_present

SCOPE_PHRASES: list[str] = [
    "does this affect other customers",
    "does this affect anyone else",
    "are other customers affected",
    "is this limited to this customer",
    "which other customers are affected",
    "what other customers are affected",
    "who else is affected",
    "who else is impacted",
    "could other customers be affected",
    "are there any other affected customers",
    "is this a customer-specific issue",
    "does the evidence show that other customers are affected",
    "does the evidence show other customers are affected",
    "is there evidence that other customers are affected",
    "which customers are explicitly shown to be affected",
    "which customers are shown to be affected",
    "which customers are affected",
    "who is affected",
]
"""Closed phrase table -- see module docstring. Each entry is checked as
a whole phrase (``phrase_present``), never a lone significant word, so
an unrelated question mentioning "customer" or "affected" alone is
never misclassified."""

_PHRASES_PATTERN = "|".join(re.escape(p) for p in SCOPE_PHRASES)
_SCOPE_CLAUSE_RE = re.compile(
    rf"(?:,?\s*\band\b\s*|^\s*|,\s*)({_PHRASES_PATTERN})\s*\??",
    re.IGNORECASE,
)


def contains_scope_question(question: str) -> bool:
    """True if ``question`` contains any whole-phrase match from
    ``SCOPE_PHRASES``. Conservative by construction: only ever True on
    an exact, closed-list phrase match, never a heuristic guess."""
    return any(phrase_present(phrase, question) for phrase in SCOPE_PHRASES)


def split_out_scope_clause(question: str) -> tuple[str, bool]:
    """Removes the FIRST scope-question clause found in ``question``
    (there is normally at most one) and returns
    ``(remaining_question, had_scope_clause)``.

    Handles this project's established compound-question shapes
    (leading, trailing, or mid-sentence scope clause joined by "and" or
    a comma) by removing the matched phrase together with its
    connecting conjunction/comma, then normalizing the leftover
    punctuation. When the ENTIRE question is the scope clause, the
    remaining text is an empty string -- callers must handle that case
    (see ``ChatOrchestrator._generate_answer``: no LLM call is made at
    all when nothing non-scope remains to ask).

    This is intentionally narrow (Step 3): it targets the specific,
    closed set of phrasings in ``SCOPE_PHRASES`` and this project's own
    established conjunction patterns, not arbitrary natural language."""
    match = _SCOPE_CLAUSE_RE.search(question)
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
