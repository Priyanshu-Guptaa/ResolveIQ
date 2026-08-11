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
