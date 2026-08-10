"""Tests for ui/formatting.py's shared text helpers -- pure functions,
no Streamlit/API dependency, so these run like any other unit test
despite living under ui/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))

from formatting import ticket_number_from_tags, truncate_words  # noqa: E402


def test_truncate_words_returns_short_text_unchanged():
    assert truncate_words("RF Mesh issue", max_len=60) == "RF Mesh issue"


def test_truncate_words_breaks_on_word_boundary_with_ellipsis():
    text = "ATCO - RF MESH IP - CC 8.7 MR1 - Each Incremental interval extract job returns empty file"
    result = truncate_words(text, max_len=60)
    assert result.endswith("…")
    assert not result[:-1].endswith(" ")  # no dangling trailing space before the ellipsis
    # Never cuts mid-word: the character right before the ellipsis must
    # have been a real word boundary in the source text.
    assert text.startswith(result[:-1])
    kept = result[:-1]
    assert text[len(kept)] == " " or len(kept) == len(text)


def test_truncate_words_falls_back_to_hard_cut_when_no_reasonable_space():
    text = "a" * 100
    result = truncate_words(text, max_len=60)
    assert result == "a" * 60 + "…"


def test_truncate_words_handles_empty_and_none():
    assert truncate_words("") == ""
    assert truncate_words(None) == ""


def test_ticket_number_from_tags_string_form():
    assert ticket_number_from_tags("ticket:INC0012345, priority:High") == "INC0012345"


def test_ticket_number_from_tags_list_form():
    assert ticket_number_from_tags(["ticket:INC0012345", "priority:High"]) == "INC0012345"


def test_ticket_number_from_tags_absent():
    assert ticket_number_from_tags("priority:High, state:Closed") is None


def test_ticket_number_from_tags_none_input():
    assert ticket_number_from_tags(None) is None
    assert ticket_number_from_tags("") is None
