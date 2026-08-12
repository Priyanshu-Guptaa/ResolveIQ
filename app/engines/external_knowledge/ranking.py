"""Deterministic scoring for TFS/Wiki live-search candidates -- same
discipline as ``app/engines/recommendation/engine.py``'s
``_match_reason()``/``_match_component()``: fixed-weight signals,
summed, clamped to [0, 1], every contributing signal named in
``match_reasons`` so an engineer can see exactly why a result was
ranked where it was. No ML, no LLM -- this is why TFS/Wiki search
(coarse ``CONTAINS WORDS``/CQL matching, confirmed against the real
TFS instance: a two-keyword query returned 993 candidates) needs a
local re-rank pass at all.
"""

from __future__ import annotations

import re

from app.engines.shared.text_matching import keyword_match_score

_WEIGHT_COMPONENT = 0.30
_WEIGHT_TECHNOLOGY = 0.20
_WEIGHT_TITLE_OVERLAP_MAX = 0.25
_WEIGHT_ENTITY_MATCH = 0.15
_WEIGHT_ENTITY_MATCH_MAX = 0.30
"""Caps the entity-match contribution at two matching entities -- a
third+ matching entity is still real signal but shouldn't let entity
overlap alone dominate the score over component/technology matches."""
_WEIGHT_RESOLVED_STATE = 0.10
_WEIGHT_CUSTOMER = 0.15
"""Customer is deliberately a *ranking* signal only, never a live
*query* term (see SearchTerms' own docstring, app/engines/external_
knowledge/service.py) -- a same-customer prior case is real, useful
evidence once a candidate is already in hand, but letting it qualify a
candidate on its own (as an OR'd anchor) diluted results with
unrelated tickets that merely mentioned the same customer name."""

_RESOLVED_STATES = {"resolved", "closed", "done"}

_CONFIDENCE_HIGH = 0.65
_CONFIDENCE_MEDIUM = 0.40
_MIN_SCORE_TO_SURFACE = 0.25
"""Below this, a candidate is dropped entirely rather than shown as
low-value noise -- TFS/Wiki search is coarse (see module docstring),
so most of what comes back genuinely isn't relevant."""

_MIN_TITLE_WORD_LEN = 4


def _title_overlap_ratio(investigation_title: str, candidate_title: str) -> float:
    """Fraction of the investigation's own significant title words
    (>=4 chars) that also appear in the candidate's title -- a plain,
    explainable word-overlap ratio, not a similarity model."""
    inv_words = {w.lower() for w in re.findall(r"[A-Za-z]+", investigation_title) if len(w) >= _MIN_TITLE_WORD_LEN}
    if not inv_words:
        return 0.0
    cand_words = {w.lower() for w in re.findall(r"[A-Za-z]+", candidate_title) if len(w) >= _MIN_TITLE_WORD_LEN}
    if not cand_words:
        return 0.0
    overlap = inv_words & cand_words
    return len(overlap) / len(inv_words)


def confidence_for_score(score: float) -> str:
    if score >= _CONFIDENCE_HIGH:
        return "High"
    if score >= _CONFIDENCE_MEDIUM:
        return "Medium"
    return "Low"


def score_candidate(
    *,
    investigation_title: str,
    candidate_title: str,
    candidate_body: str,
    state: str | None,
    matched_component_name: str | None,
    technology: str | None,
    entity_values: list[str],
    customer: str | None = None,
) -> tuple[float, list[str]]:
    """Returns (score, match_reasons). ``candidate_body`` is whatever
    plain-text content the candidate carries (TFS description, Wiki
    excerpt) -- used for component/technology/entity matching
    alongside the title, since a real match is just as likely to be in
    the body as the title."""
    combined_context = f"{candidate_title}\n{candidate_body}"
    score = 0.0
    reasons: list[str] = []

    if matched_component_name and keyword_match_score(matched_component_name, combined_context) > 0:
        score += _WEIGHT_COMPONENT
        reasons.append(f"Same component: {matched_component_name}")

    if technology and keyword_match_score(technology, combined_context) > 0:
        score += _WEIGHT_TECHNOLOGY
        reasons.append(f"Same technology: {technology}")

    overlap = _title_overlap_ratio(investigation_title, candidate_title)
    if overlap > 0:
        score += overlap * _WEIGHT_TITLE_OVERLAP_MAX
        reasons.append(f"Title overlap: {overlap:.0%} of the investigation's key words also appear here")

    entity_hits = [v for v in entity_values if v and keyword_match_score(v, combined_context) > 0]
    if entity_hits:
        entity_score = min(len(entity_hits) * _WEIGHT_ENTITY_MATCH, _WEIGHT_ENTITY_MATCH_MAX)
        score += entity_score
        shown = ", ".join(entity_hits[:3])
        reasons.append(f"Matches extracted entity/entities: {shown}")

    if state and state.strip().lower() in _RESOLVED_STATES:
        score += _WEIGHT_RESOLVED_STATE
        reasons.append(f"Already resolved (state: {state})")

    if customer and keyword_match_score(customer, combined_context) > 0:
        score += _WEIGHT_CUSTOMER
        reasons.append(f"Same customer: {customer}")

    score = max(0.0, min(1.0, score))
    return score, reasons
