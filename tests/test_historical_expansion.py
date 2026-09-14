"""Unit tests for app.engines.chat.historical_expansion (Evidence-
Centered Knowledge Retrieval & Synthesis phase, §16)."""

from __future__ import annotations

from app.domain.provenance import ResolutionProvenance
from app.engines.chat.historical_expansion import contains_unsupported_resolution_claim


def test_settled_fix_language_is_unsupported_at_unknown_tier():
    assert contains_unsupported_resolution_claim("The solution is to reset the collector queue.", ResolutionProvenance.UNKNOWN)


def test_settled_fix_language_is_unsupported_at_possible_tier():
    assert contains_unsupported_resolution_claim("This resolves the issue by restarting the service.", ResolutionProvenance.POSSIBLE)


def test_settled_fix_language_is_supported_at_confirmed_tier():
    assert not contains_unsupported_resolution_claim("The solution is to reset the collector queue.", ResolutionProvenance.CONFIRMED)


def test_settled_fix_language_is_supported_at_likely_tier():
    assert not contains_unsupported_resolution_claim("This resolves the issue by restarting the service.", ResolutionProvenance.LIKELY)


def test_hedged_historical_language_is_never_flagged():
    text = (
        "A past case recorded taking an action -- this is a historical recommendation, not a confirmed "
        "current resolution. It has not been independently verified for this situation."
    )
    assert not contains_unsupported_resolution_claim(text, ResolutionProvenance.UNKNOWN)
    assert not contains_unsupported_resolution_claim(text, ResolutionProvenance.POSSIBLE)


def test_ordinary_answer_with_no_resolution_language_is_never_flagged():
    assert not contains_unsupported_resolution_claim("I don't have enough evidence to determine the cause.", ResolutionProvenance.UNKNOWN)
