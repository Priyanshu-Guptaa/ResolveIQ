"""Whole-word overlap + question-frame handling for the troubleshooting
synthesis composer.

Real-corpus regression: "What should I check next?" ranked an RFC
change-review FAQ first (substring "check" in "Checklist", plus ordinary
prose "should"/"next"), and "Why did this request fail?" ranked unrelated
"Glitch Request" tickets first on the single shared word "request".
"""

import pytest

from app.engines.chat.knowledge_question import (
    TROUBLESHOOTING_ACTION_WORDS,
    TROUBLESHOOTING_FRAME_WORDS,
    extract_troubleshooting_subject_words,
    lexical_overlap,
    word_overlap,
)
from app.engines.chat.orchestrator import ChatOrchestrator


# --- word_overlap: the substring false positive ---------------------------


def test_lexical_overlap_still_substring_matches_check_in_checklist():
    """Documents the defect: the shared helper counts "Checklist" as a
    hit for "check". Left unchanged on purpose (many other callers)."""
    assert lexical_overlap(["check"], "RFC Basic Content Checklist items") == 1


def test_word_overlap_does_not_match_check_inside_checklist():
    assert word_overlap(["check"], "RFC Basic Content Checklist items") == 0


def test_word_overlap_matches_check_as_a_real_word():
    assert word_overlap(["check"], "Post-work check: validate register reads") == 1


@pytest.mark.parametrize(
    "text",
    [
        "Check that register reads are coming in",  # case + leading
        "CHECK THE COLLECTOR",  # upper case
        "please check.",  # trailing punctuation
        "(check)",  # parentheses
        "run the check, then wait",  # comma
        "query that checks for capability based reads",  # plural verb ending
        "we are checking the queue",  # -ing
        "already checked",  # -ed
        "post-work/check/validate",  # slash + hyphen separators
    ],
)
def test_word_overlap_matches_word_boundaries_and_inflections(text):
    assert word_overlap(["check"], text) == 1


@pytest.mark.parametrize(
    "text",
    [
        "Checklist",
        "checkpoint",
        "recheck",
        "uncheck",
        "double-checkbox",
        "checkout",
    ],
)
def test_word_overlap_rejects_longer_tokens_containing_the_word(text):
    assert word_overlap(["check"], text) == 0


def test_word_overlap_counts_each_concept_word_once():
    assert word_overlap(["meter", "stuck"], "meter meter meter") == 1
    assert word_overlap(["meter", "stuck"], "Meter stuck in Discovered") == 2


# --- technical terms ---------------------------------------------------------


def test_word_overlap_technical_terms():
    # underscore / hyphen / dot are boundaries, so real identifiers still match
    assert word_overlap(["collector"], "collector_reboot events") == 1
    assert word_overlap(["mesh"], "mesh-router firmware v3.4.0") == 1
    assert word_overlap(["axei"], "Focus AXeI. Stuck") == 1
    # short acronyms: whole word only, and no inflection (no "ntp"->"ntpd")
    assert word_overlap(["cc"], "CC 8.4 MR1 upgrade") == 1
    assert word_overlap(["cc"], "access denied") == 0
    assert word_overlap(["ntp"], "NTP sync failed") == 1
    assert word_overlap(["ntp"], "ntpd restarted") == 0
    # a concept word embedded in a longer identifier is not a match
    assert word_overlap(["process"], "readingsprocessorhost queue backing up") == 0
    assert word_overlap(["kafka"], "librdkafka 1.9.x rebalance storm") == 0
    assert word_overlap(["timeout"], "command timeouts on the collector") == 1
    assert word_overlap(["meter"], "Meters stopped responding") == 1


def test_word_overlap_empty_inputs():
    assert word_overlap([], "anything") == 0
    assert word_overlap(["check"], "") == 0


# --- frame words ---------------------------------------------------------------


def test_subject_words_of_the_three_reported_questions():
    assert extract_troubleshooting_subject_words("Why did this request fail?") == []
    assert extract_troubleshooting_subject_words("What should I check next?") == ["check"]
    assert extract_troubleshooting_subject_words("Why did it fail?") == []


def test_subject_words_keep_the_real_subject():
    assert extract_troubleshooting_subject_words("Why did the RF Mesh collector reboot?") == [
        "rf", "mesh", "collector", "reboot",
    ]
    assert extract_troubleshooting_subject_words("AxeI meters are stuck in discovered how do I make it normal?") == [
        "axei", "meters", "stuck", "discovered", "make", "normal",
    ]


def test_check_is_an_action_word_not_a_frame_word():
    assert "check" not in TROUBLESHOOTING_FRAME_WORDS
    assert "check" in TROUBLESHOOTING_ACTION_WORDS


# --- ChatOrchestrator._troubleshooting_subject_words ------------------------


_AXEI_CONTEXT = "AxeI meters are stuck in discovered.\nWhat should I check next?"


def test_fresh_check_next_has_no_subject_words():
    """Fix Remaining Off-Topic Answers & Subjectless Follow-Ups phase,
    Part 2 -- a completely fresh "What should I check next?" must have
    NO subject words (not even the bare action word "check"), so the
    caller asks for the missing detail instead of matching on "check"
    against whatever record happens to contain it."""
    assert ChatOrchestrator._troubleshooting_subject_words("What should I check next?") == []


def test_fresh_request_failure_has_no_subject_words():
    assert ChatOrchestrator._troubleshooting_subject_words("Why did this request fail?") == []


def test_followup_inherits_the_investigation_subject_and_drops_action_words():
    words = ChatOrchestrator._troubleshooting_subject_words("What should I check next?", _AXEI_CONTEXT)
    assert words == ["axei", "meters", "stuck", "discovered"]


def test_own_subject_wins_over_context():
    words = ChatOrchestrator._troubleshooting_subject_words("Why did the Kafka consumer stall?", _AXEI_CONTEXT)
    assert "axei" not in words
    assert words == ["kafka", "consumer", "stall"]


def test_context_that_is_itself_subjectless_stays_empty():
    assert ChatOrchestrator._troubleshooting_subject_words("What should I check next?", "Why did this fail?") == []
    assert ChatOrchestrator._troubleshooting_subject_words("Why did this request fail?", "Why did this fail?") == []
