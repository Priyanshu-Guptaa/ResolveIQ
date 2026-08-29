"""Tests for ui/components/chat_attachments.py's pure state/formatting
helpers (Chat Assistant Phase 42) -- no Streamlit/API dependency, so
these run like any other unit test despite living under ui/, exactly
mirroring tests/test_ui_formatting.py's own pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ui"))

from components.chat_attachments import (  # noqa: E402
    MAX_TRACKED_LOG_ATTACHMENTS,
    format_enhancement_status,
    format_eviction_notice,
    is_enhancement_terminal,
    persistence_message,
    record_log_attachment,
)


def _upload(title: str, **overrides) -> dict:
    defaults = dict(title=title, event_count=4, entity_count=0, warnings=[])
    defaults.update(overrides)
    return defaults


# --- record_log_attachment --------------------------------------------------


def test_single_upload_is_recorded():
    updated, evicted = record_log_attachment([], _upload("a.log"), capped=True)
    assert [u["title"] for u in updated] == ["a.log"]
    assert evicted is None


def test_multiple_uploads_are_all_represented_in_order():
    attachments: list[dict] = []
    for name in ("a.log", "b.log", "c.log"):
        attachments, evicted = record_log_attachment(attachments, _upload(name), capped=True)
        assert evicted is None
    assert [u["title"] for u in attachments] == ["a.log", "b.log", "c.log"]


def test_retained_count_matches_max_when_capped():
    attachments: list[dict] = []
    for i in range(MAX_TRACKED_LOG_ATTACHMENTS):
        attachments, _ = record_log_attachment(attachments, _upload(f"f{i}.log"), capped=True)
    assert len(attachments) == MAX_TRACKED_LOG_ATTACHMENTS


def test_sixth_capped_upload_evicts_the_oldest():
    attachments: list[dict] = []
    for i in range(MAX_TRACKED_LOG_ATTACHMENTS):
        attachments, _ = record_log_attachment(attachments, _upload(f"f{i}.log"), capped=True)
    attachments, evicted = record_log_attachment(attachments, _upload("f5.log"), capped=True)
    assert len(attachments) == MAX_TRACKED_LOG_ATTACHMENTS
    assert evicted is not None
    assert evicted["title"] == "f0.log"  # the real, previously-recorded oldest entry -- never guessed
    assert [u["title"] for u in attachments] == ["f1.log", "f2.log", "f3.log", "f4.log", "f5.log"]


def test_uncapped_investigation_scoped_uploads_are_never_evicted():
    """Absolute Rule 18 -- must never fabricate an eviction that did not
    happen server-side. InvestigationEngine.add_file_evidence has no cap
    at all, so capped=False must retain every upload."""
    attachments: list[dict] = []
    for i in range(MAX_TRACKED_LOG_ATTACHMENTS + 3):
        attachments, evicted = record_log_attachment(attachments, _upload(f"f{i}.log"), capped=False)
        assert evicted is None
    assert len(attachments) == MAX_TRACKED_LOG_ATTACHMENTS + 3


def test_record_log_attachment_never_mutates_the_input_list():
    original: list[dict] = [_upload("a.log")]
    snapshot = list(original)
    record_log_attachment(original, _upload("b.log"), capped=True)
    assert original == snapshot


# --- format_eviction_notice --------------------------------------------------


def test_no_eviction_notice_when_nothing_evicted():
    assert format_eviction_notice(None) is None


def test_eviction_notice_names_the_real_evicted_file():
    notice = format_eviction_notice(_upload("oldest.log"))
    assert notice is not None
    assert "oldest.log" in notice
    assert str(MAX_TRACKED_LOG_ATTACHMENTS) in notice


def test_eviction_notice_never_fabricates_a_missing_title():
    notice = format_eviction_notice({"event_count": 1})  # no "title" key at all
    assert notice is not None
    assert "oldest attachment was removed" in notice.lower()


# --- persistence_message ----------------------------------------------------


def test_persistence_message_investigation_scoped():
    msg = persistence_message(investigation_scoped=True)
    assert "saved" in msg.lower() or "persist" in msg.lower()
    assert "temporary" not in msg.lower()


def test_persistence_message_standalone_mentions_temporary_and_cap():
    msg = persistence_message(investigation_scoped=False)
    assert "temporary" in msg.lower()
    assert str(MAX_TRACKED_LOG_ATTACHMENTS) in msg


def test_persistence_messages_are_distinct():
    assert persistence_message(investigation_scoped=True) != persistence_message(investigation_scoped=False)


# --- format_enhancement_status ----------------------------------------------


def test_none_response_is_pending():
    assert format_enhancement_status(None) == {"outcome": "pending", "message": None}


def test_pending_status_is_pending():
    assert format_enhancement_status({"status": "pending"})["outcome"] == "pending"


def test_running_status_is_pending():
    assert format_enhancement_status({"status": "running"})["outcome"] == "pending"


def test_completed_with_real_answer_is_completed():
    result = format_enhancement_status({"status": "completed", "answer_text": "Enhanced phrasing."})
    assert result == {"outcome": "completed", "answer_text": "Enhanced phrasing."}


def test_completed_with_empty_answer_text_does_not_overwrite_deterministic_answer():
    """A malformed/empty completion must never be reported as success --
    this is the exact 'empty enhancement' safety case Phase 42 names."""
    result = format_enhancement_status({"status": "completed", "answer_text": ""})
    assert result["outcome"] == "pending"


def test_completed_with_missing_answer_text_key_is_pending():
    result = format_enhancement_status({"status": "completed"})
    assert result["outcome"] == "pending"


def test_completed_with_whitespace_only_answer_text_is_pending():
    result = format_enhancement_status({"status": "completed", "answer_text": "   "})
    assert result["outcome"] == "pending"


def test_failed_status_is_unavailable_with_safe_message():
    result = format_enhancement_status({"status": "failed", "error": "Ollama request to http://x timed out."})
    assert result["outcome"] == "unavailable"
    assert result["message"] == "Ollama request to http://x timed out."


def test_rejected_status_is_unavailable():
    result = format_enhancement_status({"status": "rejected", "error": "Enhancement queue is full; the deterministic answer stands."})
    assert result["outcome"] == "unavailable"


def test_timed_out_status_is_unavailable():
    assert format_enhancement_status({"status": "timed_out", "error": "took too long"})["outcome"] == "unavailable"


def test_failed_status_with_no_error_field_gets_a_safe_generic_message():
    result = format_enhancement_status({"status": "failed"})
    assert result["outcome"] == "unavailable"
    assert result["message"]  # never empty/None -- always something safe to show


def test_unrecognized_status_is_treated_as_pending_not_success():
    assert format_enhancement_status({"status": "some_future_status"})["outcome"] == "pending"


def test_non_dict_response_is_pending():
    assert format_enhancement_status("not a dict")["outcome"] == "pending"
    assert format_enhancement_status(42)["outcome"] == "pending"


# --- is_enhancement_terminal -------------------------------------------------


def test_terminal_statuses():
    for status in ("completed", "failed", "rejected", "timed_out"):
        assert is_enhancement_terminal({"status": status}) is True


def test_non_terminal_statuses():
    for status in ("pending", "running"):
        assert is_enhancement_terminal({"status": status}) is False


def test_none_is_not_terminal():
    assert is_enhancement_terminal(None) is False
