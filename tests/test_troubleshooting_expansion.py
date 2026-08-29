"""Tests for app.engines.chat.troubleshooting_expansion (Chat Assistant
Phase 46).

Pure unit tests -- deterministic pattern matching only, no LLM, no
database. See the module's own docstring for why this exists (the Rule
9 analogue of app.engines.chat.scope_expansion, closing the exact real
gap Phase 45's replay of Phase 44's own captured qwen2.5:3b output
found) and for its integration-order requirement.
"""

from __future__ import annotations

from app.engines.chat.troubleshooting_expansion import (
    UNSUPPORTED_TROUBLESHOOTING_PHRASES,
    contains_unsupported_troubleshooting_action,
)

# --- The exact real Phase 44 captured unsafe output -------------------------


def test_detects_the_exact_phase_44_thin_unsafe_output():
    text = (
        "You should check the RF Mesh IP command and ensure it is properly configured. "
        "There are no evidence-backed checks provided, so no specific troubleshooting "
        "actions can be determined from the supplied information."
    )
    assert contains_unsupported_troubleshooting_action(text)


# --- Every named Rule 9 category (power, connections, cables, signal, ------
# --- configuration, restarting a device) plus the "plugged in" idiom -------


def test_detects_every_minimum_required_phrase():
    required = [
        "You should check the power.",
        "Please verify the power supply.",
        "Check the connections to the device.",
        "Verify the connection is stable.",
        "Check connectivity to the network.",
        "Check the cables for damage.",
        "Verify the wiring is correct.",
        "Check the signal strength.",
        "Ensure it is properly configured.",
        "Check the configuration settings.",
        "Ensure it is plugged in.",
        "Verify whether it is plugged in.",
        "Restart the device.",
        "Reboot the device to resolve this.",
        "Power cycle the device.",
    ]
    for text in required:
        assert contains_unsupported_troubleshooting_action(text), text


def test_case_insensitive_and_punctuation_variants():
    assert contains_unsupported_troubleshooting_action("CHECK THE POWER immediately.")
    assert contains_unsupported_troubleshooting_action("check the power!")
    assert contains_unsupported_troubleshooting_action("Check The Power.")
    assert contains_unsupported_troubleshooting_action("...check the power...")


# --- Must NOT flag legitimate, evidence-backed, specifically-named checks ---


def test_does_not_flag_a_real_evidence_backed_service_restart():
    """This codebase's own real fixtures always phrase a legitimate
    restart as a NAMED service, never 'the device' -- the phrase table
    is deliberately qualified so this distinction holds."""
    ordinary = [
        "Restart the collector service.",
        "You should restart the collector service to confirm the route table is restored.",
        "Confirm the collector's route table is restored.",
        "Apply configuration change Y instead of restarting.",
    ]
    for text in ordinary:
        assert not contains_unsupported_troubleshooting_action(text), text


def test_does_not_flag_unrelated_grounded_text():
    ordinary = [
        "The root cause is a collector lost network route to the mesh gateway.",
        "I don't have enough evidence to determine the cause.",
        "No evidence-backed troubleshooting check can be determined from the supplied evidence.",
        "The confidence level is Likely.",
    ]
    for text in ordinary:
        assert not contains_unsupported_troubleshooting_action(text), text


def test_phrases_are_never_lone_generic_words():
    """Whole-phrase discipline -- a bare topic word must never match on
    its own (mirrors test_scope_expansion.py's own equivalent guard)."""
    assert not any(
        phrase.strip() in ("power", "connections", "connection", "cables", "cable", "signal", "configuration", "device")
        for phrase in UNSUPPORTED_TROUBLESHOOTING_PHRASES
    )


def test_a_sentence_merely_mentioning_the_topic_is_not_flagged():
    """The phrase table matches ACTION recommendations (verb + object),
    never a bare topical mention -- e.g. a sentence explaining that
    configuration was not supplied must not itself be flagged."""
    assert not contains_unsupported_troubleshooting_action(
        "The device's configuration was not specified in the supplied evidence."
    )
