"""Deterministic keyword/phrase matching against free text -- extracted
from ``app/engines/recommendation/engine.py`` (Log Intelligence
technology/scenario-type matching, Product Intelligence component
matching) so ``app/engines/external_knowledge/ranking.py`` reuses the
exact same, already-fixed logic instead of a second copy that could
silently drift or reintroduce the same bug.

Real bug this already had once (fixed before extraction, preserved
here): plain ``\\b...\\b`` word-boundary anchors silently never match a
phrase ending in punctuation followed by whitespace (e.g. "Command
Request (Outbound)"), since neither side of that boundary position is
a word character. Fixed with ``(?<!\\w)...(?!\\w)`` lookarounds, which
only require the phrase not be glued directly onto another word
character -- what "whole phrase" actually means regardless of what
punctuation borders it.
"""

from __future__ import annotations

import re

FULL_MATCH_SCORE = 1.0
PARTIAL_MATCH_SCORE = 0.5
MIN_KEYWORD_WORD_LEN = 4
"""Below this, a single word (e.g. "IP") is too generic to treat as a
meaningful partial-match signal on its own."""


def keyword_match_score(phrase: str, context: str, *, min_word_len: int = MIN_KEYWORD_WORD_LEN) -> float:
    """1.0 for the full phrase appearing verbatim (word-boundary-safe)
    in ``context``; 0.5 if only a significant individual word (>=
    ``min_word_len`` chars) from a multi-word phrase appears; 0.0
    otherwise."""
    cleaned = phrase.strip()
    if not cleaned:
        return 0.0
    if re.search(rf"(?<!\w){re.escape(cleaned)}(?!\w)", context, re.IGNORECASE):
        return FULL_MATCH_SCORE
    words = [w for w in re.findall(r"[A-Za-z]+", cleaned) if len(w) >= min_word_len]
    if any(re.search(rf"\b{re.escape(w)}\b", context, re.IGNORECASE) for w in words):
        return PARTIAL_MATCH_SCORE
    return 0.0


def phrase_present(phrase: str, text: str) -> bool:
    """Whole-phrase, word-boundary-safe, case-insensitive containment
    check -- the same ``(?<!\\w)...(?!\\w)`` lookaround discipline as
    ``keyword_match_score`` (a phrase ending in punctuation must still
    match), exposed standalone for callers that want a plain yes/no
    "does this literal phrase appear" check with no partial-word
    fallback -- e.g. a fixed cue-phrase table (intent classification,
    conversation-state correction/reference cues), where a lone
    significant word from a multi-word cue phrase must never count as
    the cue itself. Extracted 2026-08-14 (Phase 3, Conversation State)
    from ``app.engines.query_understanding.engine``'s former private
    ``_phrase_present`` so ``app.engines.chat.conversation_state``
    reuses the exact same primitive instead of a second copy."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, re.IGNORECASE) is not None


def evidence_coverage_match_score(name: str, context: str, *, min_word_len: int = MIN_KEYWORD_WORD_LEN) -> float:
    """Extracted from ``RecommendationEngine._technology_evidence_score``
    (2026-08-14, Phase 2 Query Understanding) so Query Understanding's
    Technology extraction reuses the exact same evidence-coverage rule
    instead of a second copy -- logic unchanged.

    Real bug fix (2026-08-13) this rule exists to preserve:
    ``keyword_match_score``'s partial-match fallback accepts *any
    single* significant (>= ``min_word_len`` char) word from a
    multi-word name -- so "Command processing appears delayed"
    (nothing but the ordinary English word "processing" in common with
    a technology's name) was enough to confidently match "Tool Data
    (BCS / HHU) processing", even though the far more distinctive words
    "Tool" and "Data" from that same name never appeared anywhere in
    the text.

    Generic evidence-coverage rule (no names or generic words
    hardcoded anywhere -- a property of how much of any given
    candidate's *own* name is actually present, computed identically
    for every candidate):

    1. An exact/full phrase match is always valid evidence (unchanged
       -- delegates to ``keyword_match_score``).
    2. Short of that, a partial match is only valid when *every* one of
       the candidate's own significant words is present in the text --
       full coverage of that candidate's identifying vocabulary, not a
       fragment of it. A candidate whose name has only one significant
       word to begin with (e.g. "RF Mesh", whose only word >= 4 chars
       is "Mesh") is unaffected: matching its one word is by definition
       100% coverage of its own name, so it still counts -- this is
       what keeps hierarchy-based specificity/ambiguity resolution
       (``app.engines.shared.hierarchy.most_specific``) working exactly
       as expected. A candidate with several significant words (e.g.
       "Tool Data (BCS / HHU) processing" -> {Tool, Data, processing})
       now needs all of them, not just one.
    3. Anything less -- a single word out of several, or no significant
       words present at all -- scores 0.0: no evidence, meant to be
       excluded outright by the caller's positive-score filter, never
       "the best we found by default"."""
    full = keyword_match_score(name, context, min_word_len=min_word_len)
    if full >= FULL_MATCH_SCORE:
        return FULL_MATCH_SCORE
    significant_words = [w for w in re.findall(r"[A-Za-z]+", name) if len(w) >= min_word_len]
    if not significant_words:
        return 0.0
    if all(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", context, re.IGNORECASE) for w in significant_words):
        return PARTIAL_MATCH_SCORE
    return 0.0
