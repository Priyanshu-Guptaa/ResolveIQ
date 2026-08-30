"""Tests for app.engines.chat.confidence_expansion (Chat Assistant
Phase 49).

Pure unit tests -- deterministic pattern matching only, no LLM, no
database. See the module's own docstring for why this exists (Phase
48's real 140-call qwen2.5:3b benchmark found a live "Has this happened
before? Confirmed." overclaim against a LIKELY-tier fixture, with no
deterministic protection) and for its integration-order requirement.
"""

from __future__ import annotations

from app.domain.provenance import ResolutionProvenance
from app.engines.chat.confidence_expansion import (
    SAFE_CONFIDENCE_HEDGE_PHRASES,
    UNSUPPORTED_CONFIDENCE_PHRASES,
    contains_unsupported_confidence_claim,
)

UNKNOWN = ResolutionProvenance.UNKNOWN
LIKELY = ResolutionProvenance.LIKELY
POSSIBLE = ResolutionProvenance.POSSIBLE
CONFIRMED = ResolutionProvenance.CONFIRMED


# --- The exact real Phase 48 captured failure -------------------------------


def test_detects_the_exact_phase_48_failure_text():
    text = "Has this happened before? Confirmed."
    assert contains_unsupported_confidence_claim(text, LIKELY)


# --- A-D: unsupported confidence upgrades below the Confirmed tier ----------


def test_a_unknown_confidence_plus_confirmed_is_rejected():
    assert contains_unsupported_confidence_claim("The root cause is confirmed.", UNKNOWN)


def test_b_unknown_confidence_plus_verified_is_rejected():
    assert contains_unsupported_confidence_claim("This has been verified.", UNKNOWN)


def test_c_likely_confidence_plus_confirmed_is_rejected():
    assert contains_unsupported_confidence_claim("This is confirmed based on the logs.", LIKELY)


def test_d_likely_confidence_plus_verified_is_rejected():
    assert contains_unsupported_confidence_claim("Verified from the available evidence.", LIKELY)


def test_possible_confidence_plus_confirmed_is_rejected():
    assert contains_unsupported_confidence_claim("The resolution is confirmed.", POSSIBLE)


# --- E-F: legitimate CONFIRMED tier allows the language ---------------------


def test_e_confirmed_confidence_plus_confirmed_is_allowed():
    assert not contains_unsupported_confidence_claim("The root cause is confirmed.", CONFIRMED)


def test_f_confirmed_confidence_plus_verified_is_allowed():
    assert not contains_unsupported_confidence_claim("This has been verified.", CONFIRMED)


def test_confirmed_confidence_bare_standalone_confirmed_is_allowed():
    assert not contains_unsupported_confidence_claim("Has this happened before? Confirmed.", CONFIRMED)


# --- G-H: case sensitivity ---------------------------------------------------


def test_g_lowercase_confirmed_is_detected():
    assert contains_unsupported_confidence_claim("confirmed.", LIKELY)


def test_h_uppercase_and_mixed_case_confirmed_is_detected():
    assert contains_unsupported_confidence_claim("CONFIRMED.", LIKELY)
    assert contains_unsupported_confidence_claim("Confirmed.", LIKELY)
    assert contains_unsupported_confidence_claim("ConFiRmEd.", LIKELY)


# --- I: ordinary safe answers with no confidence-upgrade language ----------


def test_i_ordinary_safe_answers_are_allowed():
    ordinary = [
        "This is likely.",
        "The evidence suggests a network route issue.",
        "Unknown.",
        "A possible explanation is a collector configuration issue.",
        "You should restart the collector service.",
        "The root cause is a collector lost network route to the mesh gateway.",
    ]
    for text in ordinary:
        assert not contains_unsupported_confidence_claim(text, UNKNOWN), text
        assert not contains_unsupported_confidence_claim(text, LIKELY), text


# --- Natural-language negation/hedge variants must remain allowed ----------


def test_natural_hedge_variants_remain_allowed():
    safe = [
        "This cannot be confirmed from the available evidence.",
        "I don't have enough evidence to confirm this.",
        "The evidence suggests...",
        "This has not been independently verified.",
        "A likely resolution, not yet independently verified: restart the collector service.",
        "A possible resolution, not independently verified: apply configuration change Y.",
        "This has not been verified.",
        "There is not enough evidence to verify this claim.",
    ]
    for text in safe:
        assert not contains_unsupported_confidence_claim(text, UNKNOWN), text
        assert not contains_unsupported_confidence_claim(text, LIKELY), text


def test_natural_positive_overclaim_variants_are_rejected():
    unsafe = [
        "The root cause is confirmed.",
        "This has been verified.",
        "Confirmed based on the logs.",
        "Verified from the available evidence.",
        "Has this happened before? Confirmed.",
        "The resolution is confirmed.",
    ]
    for text in unsafe:
        assert contains_unsupported_confidence_claim(text, UNKNOWN), text
        assert contains_unsupported_confidence_claim(text, LIKELY), text


# --- Structural / discipline guarantees -------------------------------------


def test_no_unsupported_phrase_is_a_substring_of_any_safe_hedge_phrase():
    """The design relies on this: stripping safe hedges first must never
    accidentally remove part of a genuine positive-assertion phrase."""
    for unsupported in UNSUPPORTED_CONFIDENCE_PHRASES:
        for hedge in SAFE_CONFIDENCE_HEDGE_PHRASES:
            assert unsupported not in hedge, (unsupported, hedge)


def test_confirmed_tier_never_flagged_regardless_of_phrasing():
    always_allowed_texts = [
        "This is confirmed.",
        "Confirmed.",
        "The root cause is confirmed and the resolution is verified.",
        "This has been verified based on two independent sources.",
    ]
    for text in always_allowed_texts:
        assert not contains_unsupported_confidence_claim(text, CONFIRMED), text
