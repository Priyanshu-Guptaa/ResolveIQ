"""Informational/historical-question detection (Chat Knowledge-Synthesis
feature).

Real usage finding: a question like "tell me about dashboard in CC" is
answered correctly by retrieval -- real, relevant Documentation,
Historical Investigation, Known Bug, and TFS/Wiki matches are all found
-- but the DETERMINISTIC answer composer (``ChatOrchestrator.
_compose_answer``) only ever renders investigation-confidence-tier
boilerplate ("A possible explanation may exist, but the evidence found
is limited...") because that composer's only inputs are
``structured.root_cause``/``resolution_candidates``/``confidence`` --
documentation matches were never wired into ``answer_text`` at all,
only ever shown separately in the UI's retrieval panel. An
informational question has no root cause to establish and no
resolution to recommend; "insufficient evidence to determine a
resolution" is a true statement about a DIFFERENT question than the
one the user actually asked.

This module is the narrow, closed-list classifier that decides whether
a question is asking for that kind of general/historical knowledge
synthesis (see ``ChatOrchestrator._compose_knowledge_synthesis``) at
all -- never a decision about whether real evidence supports one
(that remains entirely in the synthesizer, which returns ``None`` and
lets the existing tier-based text stand whenever nothing relevant was
actually retrieved). Same idiom as every other question classifier in
this codebase (``scope_question``, ``troubleshooting_question``):
closed phrase list, whole-phrase matching via ``phrase_present``,
never an LLM, conservative by construction (only ever True on an
exact match, never a guess) -- see those modules' own docstrings for
the shared rationale.

Deliberately excludes bare "what is X"/"what's X" from the fixed
phrase list -- unlike "tell me about"/"explain"/"how does X work",
"what is X" collides with this project's own core investigation
vocabulary ("What is the root cause?", "What is the resolution?",
"What is the confidence?" must never be redirected into a knowledge
answer -- Rule 3/4's tier-preservation and Rule 9's troubleshooting
gate both depend on those reaching the existing, unmodified tier-based
composer). ``contains_knowledge_question`` instead recognizes "what
is/what's X" as informational only when X does not start with one of
this project's own reserved investigation nouns (root cause,
resolution, confidence, evidence, applicability, ...) -- see
``_RESERVED_INVESTIGATION_LEAD_INS``.
"""

from __future__ import annotations

import re

from app.engines.shared.text_matching import phrase_present

INFORMATIONAL_PHRASES: list[str] = [
    "tell me about",
    "tell me more about",
    "explain",
    "describe",
    "give me an overview of",
    "give me a summary of",
    "what documentation do we have",
    "what docs do we have",
    "what documentation is available",
    "what documentation exists",
    "do we have any information about",
    "do we have information about",
    "do we have any documentation about",
    "do we have documentation about",
    "do we have documentation on",
    "what do we know about",
    "what is known about",
]
"""Closed list of fixed lead-ins that are unambiguously a request for
general knowledge, never a question about THIS investigation's own
root cause/resolution/confidence -- see module docstring."""

HISTORICAL_PHRASES: list[str] = [
    "has this happened before",
    "have we seen this before",
    "any previous cases",
    "any prior cases",
    "any past cases",
    "seen this before",
    "seen before",
]
"""Closed list for the historical/experience mode (§2.B) -- distinct
from ``INFORMATIONAL_PHRASES`` only for documentation purposes; both
route to the same synthesizer (``ChatOrchestrator.
_compose_knowledge_synthesis``), which already adapts its wording to
whichever real evidence (documentation vs. historical matches vs.
both) actually exists, so a single combined check is sufficient and
avoids two near-duplicate code paths."""

_WHAT_DOES_MEAN_RE = re.compile(r"\bwhat does\b.+\bmean\b", re.IGNORECASE)
_HOW_DOES_WORK_RE = re.compile(r"\bhow does\b.+\bwork\b", re.IGNORECASE)
_HAVE_WE_SEEN_BEFORE_RE = re.compile(r"\bhave we seen\b.+\bbefore\b", re.IGNORECASE)
"""Variable-middle constructs ("what does X mean", "how does X work",
"have we seen X before") that a fixed phrase list cannot express --
still closed-form regexes anchored on the same fixed lead-in/trailing
words as the phrase-list entries, never a free-form heuristic."""

_WHAT_IS_RE = re.compile(r"\bwhat(?:'s| is)\s+(.+?)[?.!]?\s*$", re.IGNORECASE)
_RESERVED_INVESTIGATION_LEAD_INS: tuple[str, ...] = (
    "the root cause",
    "the likely root cause",
    "the resolution",
    "the likely resolution",
    "the confidence",
    "the evidence",
    "the applicability",
    "the customer",
    "the scope",
    "confirmed",
    "verified",
    "the tier",
)
"""See module docstring -- "what is X" is informational UNLESS X
starts with one of this project's own reserved investigation nouns,
in which case it must reach the existing, unmodified tier-based
composer instead."""


def contains_knowledge_question(question: str) -> bool:
    """True if ``question`` is asking for general/historical ResolveIQ
    knowledge (§2.A/§2.B) rather than this specific investigation's own
    root cause/resolution/confidence/troubleshooting. Conservative by
    construction: only ever True on an exact closed-list match or one
    of the two narrow variable-middle regexes, never a heuristic
    guess -- see module docstring."""
    if any(phrase_present(phrase, question) for phrase in INFORMATIONAL_PHRASES):
        return True
    if any(phrase_present(phrase, question) for phrase in HISTORICAL_PHRASES):
        return True
    if _WHAT_DOES_MEAN_RE.search(question) or _HOW_DOES_WORK_RE.search(question) or _HAVE_WE_SEEN_BEFORE_RE.search(question):
        return True
    match = _WHAT_IS_RE.search(question)
    if match:
        remainder = match.group(1).strip().lower()
        if not any(remainder.startswith(lead_in) for lead_in in _RESERVED_INVESTIGATION_LEAD_INS):
            return True
    return False


_CONCEPT_STOPWORDS: frozenset[str] = frozenset(
    {
        "what", "whats", "is", "are", "was", "were", "tell", "me", "more", "about", "in", "on", "of", "the",
        "a", "an", "to", "does", "do", "did", "mean", "means", "how", "we", "have", "has", "had", "any",
        "information", "documentation", "docs", "know", "known", "explain", "describe", "give", "overview",
        "summary", "for", "this", "that", "these", "those", "seen", "before", "happened", "related", "case",
        "cases", "previous", "prior", "past", "and", "or", "with", "can", "you", "us", "please", "there",
    }
)
"""Closed stopword list for ``extract_concept_words`` -- the fixed
lead-in/trailing words every ``INFORMATIONAL_PHRASES``/
``HISTORICAL_PHRASES`` entry is built from, plus ordinary English
function words. Never a general-purpose NLP stopword list (no attempt
at completeness beyond what this project's own closed question
patterns actually use) -- conservative by construction, same as every
other list in this module."""

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def extract_concept_words(question: str) -> list[str]:
    """Chat Knowledge-Synthesis feature -- the deterministic "what is
    this question actually about" signal ``ChatOrchestrator.
    _compose_knowledge_synthesis`` uses to distinguish a retrieval
    candidate that merely scored well from one that actually contains
    the subject the user asked about (real usage finding: "what is
    process setting in emerge" retrieved a personal task-list export at
    69% similarity -- "task" shares no real subject with "process
    setting"/"emerge" at all, and was still presented as the answer
    because it was Chroma's single highest-scoring Documentation
    candidate). Not a POS tagger or an entity extractor -- just
    ``question``'s own content words, with this module's own fixed
    question-scaffolding vocabulary (see ``_CONCEPT_STOPWORDS``)
    removed, lowercased, deduplicated, in first-occurrence order.
    Short, real product/technology acronyms (e.g. "cc", "nmm") are kept
    -- ``_lexical_overlap`` (the only caller) applies word-boundary
    matching for anything 3 characters or shorter specifically so a
    short acronym never matches as a false-positive substring of an
    unrelated longer word."""
    seen: set[str] = set()
    words: list[str] = []
    for match in _WORD_RE.finditer(question):
        word = match.group(0).lower()
        if word in _CONCEPT_STOPWORDS or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return words


def lexical_overlap(concept_words: list[str], text: str) -> int:
    """How many of ``concept_words`` actually appear in ``text``
    (case-insensitive) -- substring matching for anything longer than 3
    characters (so "setting"/"settings", "dashboard"/"dashboards" etc.
    match regardless of this project's own real, observed pluralization
    without needing a stemmer), whole-word matching for anything 3
    characters or shorter (so a short acronym like "cc" never falsely
    matches inside an unrelated longer word like "access" or "occurred").
    Used by ``ChatOrchestrator._compose_knowledge_synthesis`` to re-rank
    already-retrieved candidates by real subject overlap, never to
    perform a new retrieval of its own."""
    lowered = text.lower()
    count = 0
    for word in concept_words:
        if len(word) <= 3:
            if phrase_present(word, lowered):
                count += 1
        elif word in lowered:
            count += 1
    return count
