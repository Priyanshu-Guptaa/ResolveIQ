"""Governed-value text matching -- "does a real governed value's
name/alias appear in this free text, and if several do, which is most
specific" -- extracted from ``app.engines.knowledge.classification``
(2026-08-14, Phase 2 Query Understanding) so any engine that needs this
same mechanic (Document/Historical-Investigation/Known-Bug
classification, and now Query Understanding's entity extraction) uses
the exact same matching primitives instead of a second copy that could
silently drift or reintroduce an already-fixed bug.

Behavior-preserving extraction: every function here is a verbatim lift
of classification.py's former private ``_Candidate``/``_title_match``/
``_body_mention_count``/``_most_specific_candidates`` (renamed only,
logic unchanged), captured against classification.py's own passing
test suite both before and after the extraction.

Deliberately NOT here: classification.py's dimension-specific
confidence tiering (title match = HIGH, body repeated mention =
MEDIUM, single body mention = LOW). That rule is specific to
classification's own confidence model, not a generic text-matching
primitive, and stays local to ``classification.py``'s own
``_best_match``. Query Understanding's own confidence model (EXACT/
PARTIAL/AMBIGUOUS -- see ``app.domain.query_understanding``) is a
different set of rules again, built on top of these same primitives
rather than reusing classification's tiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.engines.shared.hierarchy import most_specific
from app.engines.shared.text_matching import FULL_MATCH_SCORE, keyword_match_score

MIN_CANDIDATE_NAME_LEN = 3
"""Below this, a governed name is too likely to false-positive as an
incidental substring/short-word match to trust for matching at all --
same floor discipline as
``RecommendationEngine._MIN_COMPONENT_NAME_LEN_FOR_COLLECTED_MATCH``,
lowered slightly (3 not 4) so real short names in the real corpus
("NMS", "EIC") aren't excluded. This is classification.py's original
threshold, carried over unchanged."""
SNIPPET_WINDOW = 60
"""Characters of context kept on each side of a matched body mention,
for a human-readable evidence snippet. classification.py's original
value, carried over unchanged."""


@dataclass(frozen=True)
class GovernedCandidate:
    """One real, already-governed row (Technology/Product/Customer/
    Region/Component/...) eligible to be matched against free text.
    Renamed from classification.py's former private ``_Candidate`` --
    identical shape."""

    id: str
    canonical_name: str
    match_names: tuple[str, ...]
    """canonical_name plus any aliases -- what's actually checked
    against the text."""
    parent_id: str | None = None
    """The candidate's parent in a real governed hierarchy (today:
    ``Technology.parent_technology_id``), or None -- either because
    this candidate has no parent, or because this dimension has no
    hierarchy concept at all (Customer/Region/Product/Component).
    Consumed by ``most_specific_candidates`` to prefer a more specific
    matched candidate over a matched ancestor of it."""


def full_phrase_match(
    candidate: GovernedCandidate, text: str, *, min_name_len: int = MIN_CANDIDATE_NAME_LEN
) -> str | None:
    """Returns the matched name (longest first, so a more specific
    alias/name wins over a shorter coincidental one) or None. Renamed
    from classification.py's former private ``_title_match`` --
    identical logic, generalized name since callers other than
    classification's title check now use it too."""
    if not text:
        return None
    for name in sorted(candidate.match_names, key=len, reverse=True):
        if len(name.strip()) < min_name_len:
            continue
        if keyword_match_score(name, text) >= FULL_MATCH_SCORE:
            return name
    return None


def mention_count(
    candidate: GovernedCandidate,
    body: str,
    *,
    min_name_len: int = MIN_CANDIDATE_NAME_LEN,
    snippet_window: int = SNIPPET_WINDOW,
) -> tuple[int, str | None, str]:
    """Returns (mention_count, matched_name, first_snippet) using the
    longest matching name found. Renamed from classification.py's
    former private ``_body_mention_count`` -- identical logic."""
    best_name: str | None = None
    best_count = 0
    best_snippet = ""
    for name in sorted(candidate.match_names, key=len, reverse=True):
        cleaned = name.strip()
        if len(cleaned) < min_name_len:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(cleaned)}(?!\w)", re.IGNORECASE)
        matches = list(pattern.finditer(body))
        if len(matches) > best_count:
            best_count = len(matches)
            best_name = name
            first = matches[0]
            lo = max(0, first.start() - snippet_window)
            hi = min(len(body), first.end() + snippet_window)
            prefix = "..." if lo > 0 else ""
            suffix = "..." if hi < len(body) else ""
            best_snippet = f"{prefix}{body[lo:hi].strip()}{suffix}"
    return best_count, best_name, best_snippet


def most_specific_candidates(
    matched: list[GovernedCandidate], universe: list[GovernedCandidate]
) -> list[GovernedCandidate]:
    """Filters ``matched`` down to the most specific entries via
    ``app.engines.shared.hierarchy.most_specific``, using every
    candidate in ``universe`` (not just the matched ones) to resolve
    parent ids -- an ancestor several levels up that didn't itself
    match still needs to be walked through for a hierarchy deeper than
    one level. Preserves ``matched``'s original relative order so the
    final selection stays fully deterministic without a second sort.
    Renamed from classification.py's former private
    ``_most_specific_candidates`` -- identical logic."""
    if len(matched) <= 1:
        return matched
    parent_of = {c.id: c.parent_id for c in universe}
    survivor_ids = most_specific({c.id for c in matched}, parent_of)
    return [c for c in matched if c.id in survivor_ids]
