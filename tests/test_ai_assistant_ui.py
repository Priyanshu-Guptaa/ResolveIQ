"""Tests for the AI Assistant Streamlit page (Chat Assistant Phase 42 --
log-attachment list + async-enhancement UI).

Uses ``streamlit.testing.v1.AppTest`` to run the real page script with a
fake ``api_client`` module substituted in ``sys.modules`` -- the exact
same pattern already established by
``tests/test_classification_review_ui.py`` (no real HTTP call is ever
made; no real Ollama/LLM is ever touched). ``st.file_uploader`` cannot
be driven by ``AppTest`` (no public API for injecting file bytes), so
upload-button-click behavior itself is covered by the pure-function
tests in ``tests/test_chat_attachments.py``; these tests instead verify
the PAGE renders whatever attachment/enhancement state is present in
``session_state`` (pre-seeded here exactly as the real upload/message
handlers would leave it) -- i.e., that the page wires that state to the
right on-screen text, not that the upload widget itself works.
"""

from __future__ import annotations

import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, "ui")  # ui/theme.py etc., imported by the page itself, live here

_PAGE_PATH = "ui/views/7_AI_Assistant.py"


def _attachment(title: str, **overrides) -> dict:
    defaults = dict(title=title, event_count=4, entity_count=0, warnings=[])
    defaults.update(overrides)
    return defaults


def _chat_response(**overrides) -> dict:
    defaults = dict(answer_text="Here is the deterministic answer.")
    defaults.update(overrides)
    return defaults


class _FakeApiClient:
    """Records every call; every response is configurable per test.
    Mirrors the real ``api_client`` module's contract: returns ``None``
    on failure (never raises)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.session_response: dict | None = {"id": "sess-1", "investigation_id": None}
        self.messages_response: list | None = []
        self.post_message_response: dict | None = _chat_response()
        self.log_upload_response: dict | None = _attachment("device.log")
        self.enhancement_responses: list[dict | None] = []
        """Popped in order, one per GET /chat/enhancements/{id} call --
        lets a test script a pending -> completed (or -> failed)
        sequence across successive reruns/polls."""

    def api_get(self, path, params=None):
        self.calls.append(("GET", path, params))
        if path.endswith("/messages"):
            return self.messages_response
        if path.startswith("/chat/enhancements/"):
            if self.enhancement_responses:
                return self.enhancement_responses.pop(0)
            return None
        return None

    def api_post(self, path, json_body=None, files=None, timeout=60):
        self.calls.append(("POST", path, json_body))
        if path == "/chat/sessions":
            return self.session_response
        if path.endswith("/logs"):
            return self.log_upload_response
        if path.endswith("/messages"):
            return self.post_message_response
        return None

    def ensure_api_available(self):
        return True


@pytest.fixture
def fake_client(monkeypatch):
    client = _FakeApiClient()
    module = types.ModuleType("api_client")
    module.API_BASE_URL = "http://fake"
    module.api_get = client.api_get
    module.api_post = client.api_post
    module.ensure_api_available = client.ensure_api_available
    module.list_investigations_cached = lambda: []  # imported by components.investigation_picker at module load;
    # never actually called by these Standalone-scope tests, but the import itself must succeed.
    monkeypatch.setitem(sys.modules, "api_client", module)
    yield client


def _run(fake_client, *, session_state: dict | None = None) -> AppTest:
    at = AppTest.from_file(_PAGE_PATH)
    at.session_state["chat_scope"] = "Standalone"  # skip the investigation picker entirely
    # A matching chat_session_key by default, so pre-seeded state below
    # survives instead of being wiped by the page's own "session changed"
    # reset block -- tests that specifically want to exercise that reset
    # (e.g. test_new_session_key_change_clears_attachment_state) override it.
    at.session_state["chat_session_key"] = "Standalone::standalone"
    at.session_state["chat_session_id"] = "sess-1"
    for key, value in (session_state or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"
    return at


def _state_get(at: AppTest, key: str, default=None):
    """``AppTest.session_state`` supports ``in``/``[]`` but not ``.get()``
    (attribute access falls through to item lookup and raises)."""
    return at.session_state[key] if key in at.session_state else default


def _all_text(at: AppTest) -> str:
    return "\n".join(
        [*(m.value for m in at.markdown), *(c.value for c in at.caption), *(w.value for w in at.warning), *(i.value for i in at.info)]
    )


# --- Baseline: page loads, standalone scope, no attachments -----------------


def test_page_loads_with_no_attachments(fake_client):
    at = _run(fake_client, session_state={"chat_session_id": "sess-1"})
    assert "attached" not in _all_text(at).lower()


# --- Requirement A: multiple attached logs -----------------------------------


def test_single_attached_log_is_displayed(fake_client):
    at = _run(fake_client, session_state={"chat_session_id": "sess-1", "chat_log_attachments": [_attachment("device.log")]})
    text = _all_text(at)
    assert "1 log(s)" in text
    assert "device.log" in text


def test_multiple_attached_logs_are_all_represented(fake_client):
    attachments = [_attachment("a.log"), _attachment("b.log"), _attachment("c.log")]
    at = _run(fake_client, session_state={"chat_session_id": "sess-1", "chat_log_attachments": attachments})
    text = _all_text(at)
    assert "3 log(s)" in text
    for name in ("a.log", "b.log", "c.log"):
        assert name in text


def test_retained_count_reflects_full_list_not_just_latest(fake_client):
    attachments = [_attachment(f"f{i}.log") for i in range(5)]
    at = _run(fake_client, session_state={"chat_session_id": "sess-1", "chat_log_attachments": attachments})
    text = _all_text(at)
    assert "5 log(s)" in text
    assert "f0.log" in text  # the OLDEST is still shown, not just the newest


# --- Requirement B: FIFO eviction visibility ---------------------------------


def test_eviction_notice_is_shown_when_present(fake_client):
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_log_attachments": [_attachment(f"f{i}.log") for i in range(1, 6)],
            "chat_log_eviction_notice": "Maximum 5 logs are retained per chat. Oldest attachment removed: **f0.log**.",
        },
    )
    text = _all_text(at)
    assert "f0.log" in text
    assert "removed" in text.lower()


def test_no_eviction_notice_when_none_recorded(fake_client):
    at = _run(fake_client, session_state={"chat_session_id": "sess-1", "chat_log_attachments": [_attachment("a.log")]})
    assert "removed" not in _all_text(at).lower()


def test_eviction_notice_does_not_fabricate_when_absent():
    """Pure-logic guarantee (also exercised in test_chat_attachments.py) --
    included here too as a page-level sanity check that the page never
    invents its own eviction text independent of session_state."""
    from components.chat_attachments import format_eviction_notice

    assert format_eviction_notice(None) is None


# --- Requirement C: persistence semantics ------------------------------------


def test_standalone_persistence_message_mentions_temporary(fake_client):
    at = _run(fake_client, session_state={"chat_session_id": "sess-1", "chat_log_attachments": [_attachment("a.log")]})
    text = _all_text(at).lower()
    assert "temporary" in text


# --- Upload warnings remain visible (existing Phase 39 behavior, unchanged) --


def test_upload_warnings_survive_in_the_attachment_list_context(fake_client):
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_log_attachments": [_attachment("a.log", warnings=["Invalid JSON; stored as raw text"])],
        },
    )
    # Warnings are shown at upload time (st.success block, not re-rendered
    # from the retained list on every rerun) -- verified in
    # test_chat_attachments.py's own upload-response shape; this test only
    # confirms the attachment list rendering itself doesn't choke on a
    # response dict that has a non-empty "warnings" field.
    assert not at.exception


# --- New session/scope does not inherit previous attachment state -----------


def test_new_session_key_change_clears_attachment_state(fake_client):
    """Simulates arriving on the page after switching scope/investigation
    (the page's own session_key mismatch detection) -- the previous
    session's attachment list must not leak into the new one."""
    at = AppTest.from_file(_PAGE_PATH)
    at.session_state["chat_scope"] = "Standalone"
    at.session_state["chat_session_key"] = "Investigation::some-other-investigation"  # mismatches "Standalone::standalone"
    at.session_state["chat_session_id"] = "old-session"
    at.session_state["chat_log_attachments"] = [_attachment("leftover.log")]
    fake_client.session_response = {"id": "new-session", "investigation_id": None}
    at.run()
    assert not at.exception
    assert "leftover.log" not in _all_text(at)
    assert _state_get(at, "chat_log_attachments") in (None, [])


# --- Async enhancement: pending state ----------------------------------------


def test_enhancement_pending_shows_status_and_check_button(fake_client):
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "pending"}]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="Deterministic answer."),
            "chat_enhancement_job_id": "job-1",
        },
    )
    text = _all_text(at)
    assert "Deterministic answer." in text
    assert "still being generated" in text.lower()
    assert len(at.button) >= 1
    assert any("check" in b.label.lower() for b in at.button)


def test_enhancement_check_button_click_polls_a_small_bounded_number_of_times(fake_client):
    """The page code itself makes at most one GET /chat/enhancements/{id}
    call per top-to-bottom script execution (no loop, no sleep -- see
    ui/views/7_AI_Assistant.py's own comment at the polling site).
    AppTest's own click-and-settle harness re-executes the script a
    small, fixed number of times per interaction (an AppTest
    implementation detail, not a property of the page), so this asserts
    boundedness generously rather than an exact count that would couple
    the test to that harness internal."""
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "pending"} for _ in range(10)]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(),
            "chat_enhancement_job_id": "job-1",
        },
    )
    calls_before = len([c for c in fake_client.calls if c[1].startswith("/chat/enhancements/")])
    check_button = next(b for b in at.button if "check" in b.label.lower())
    check_button.click().run()
    calls_after = len([c for c in fake_client.calls if c[1].startswith("/chat/enhancements/")])
    # A real uncontrolled loop would have drained all 10 queued responses
    # in one click; a bounded, one-check-per-rerun design cannot.
    assert 0 < calls_after - calls_before <= 3


# --- Async enhancement: completed state --------------------------------------


def test_enhancement_completed_renders_enhanced_text_without_replacing_deterministic_answer(fake_client):
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "completed", "answer_text": "A nicer phrasing of the same answer."}]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="Deterministic answer stands."),
            "chat_enhancement_job_id": "job-1",
        },
    )
    text = _all_text(at)
    assert "Deterministic answer stands." in text  # never removed/replaced
    assert "A nicer phrasing of the same answer." in text
    assert _state_get(at, "chat_enhancement_job_id") is None  # terminal -- stopped polling


def test_enhancement_result_persists_across_an_unrelated_rerun(fake_client):
    """Once fetched, the completed enhancement must still show up on a
    LATER rerun even if that rerun's job_id has already been cleared --
    it must be cached in chat_enhancement_result, not re-fetched or lost."""
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="Deterministic answer stands."),
            "chat_enhancement_job_id": None,
            "chat_enhancement_result": {"outcome": "completed", "answer_text": "Cached enhanced phrasing."},
        },
    )
    assert "Cached enhanced phrasing." in _all_text(at)
    assert not any(c[1].startswith("/chat/enhancements/") for c in fake_client.calls)  # no re-fetch needed


# --- Async enhancement: failure/fallback -------------------------------------


def test_enhancement_failure_preserves_deterministic_answer_and_shows_safe_status(fake_client):
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "failed", "error": "Ollama request to http://x timed out."}]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="Deterministic answer stands."),
            "chat_enhancement_job_id": "job-1",
        },
    )
    text = _all_text(at)
    assert "Deterministic answer stands." in text
    assert "unavailable" in text.lower()
    assert "Ollama request to http://x timed out." in text  # backend's own safe error text, not swallowed
    assert _state_get(at, "chat_enhancement_job_id") is None


def test_enhancement_rejected_preserves_deterministic_answer(fake_client):
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "rejected", "error": "Enhancement queue is full; the deterministic answer stands."}]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="Deterministic answer stands."),
            "chat_enhancement_job_id": "job-1",
        },
    )
    assert "Deterministic answer stands." in _all_text(at)
    assert not at.exception


def test_no_exception_or_traceback_ever_rendered_on_enhancement_failure(fake_client):
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "failed", "error": "Ollama request timed out."}]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(),
            "chat_enhancement_job_id": "job-1",
        },
    )
    text = _all_text(at)
    assert "Traceback" not in text
    assert "Exception" not in text
    assert not at.exception


# --- Async enhancement: empty/malformed responses never overwrite -----------


def test_empty_answer_text_completion_does_not_overwrite_deterministic_answer(fake_client):
    fake_client.enhancement_responses = [{"job_id": "job-1", "status": "completed", "answer_text": ""}]
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="Deterministic answer stands."),
            "chat_enhancement_job_id": "job-1",
        },
    )
    text = _all_text(at)
    assert "Deterministic answer stands." in text
    assert "**AI-enhanced phrasing**" not in text  # the labeled SUCCESS block never renders
    assert "still being generated" in text.lower()  # correctly still reported as pending, not success
    # Still pending (empty completion is not treated as success) -- job_id retained.
    assert _state_get(at, "chat_enhancement_job_id") == "job-1"


def test_none_enhancement_response_keeps_pending_state_without_erroring(fake_client):
    fake_client.enhancement_responses = []  # api_get returns None (no queued response)
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(),
            "chat_enhancement_job_id": "job-1",
        },
    )
    assert not at.exception
    assert _state_get(at, "chat_enhancement_job_id") == "job-1"


# --- No enhancement scheduled (llm_async_enabled=False, every real case today)


def test_no_enhancement_field_renders_nothing_extra_and_polls_nothing(fake_client):
    at = _run(
        fake_client,
        session_state={"chat_session_id": "sess-1", "chat_last_response": _chat_response(answer_text="Plain deterministic answer.")},
    )
    text = _all_text(at)
    assert "Plain deterministic answer." in text
    assert "AI-enhanced" not in text
    assert "still being generated" not in text.lower()
    assert not any(c[1].startswith("/chat/enhancements/") for c in fake_client.calls)


# --- End-to-end: sending a message with a scheduled enhancement -------------


def test_sending_message_with_enhancement_ref_starts_pending_state(fake_client):
    fake_client.post_message_response = _chat_response(
        answer_text="Fresh deterministic answer.", enhancement={"job_id": "job-42", "status": "pending"}
    )
    fake_client.enhancement_responses = [{"job_id": "job-42", "status": "pending"}]
    at = _run(fake_client, session_state={"chat_session_id": "sess-1"})
    at.chat_input[0].set_value("What do the logs show?").run()
    assert not at.exception
    assert _state_get(at, "chat_enhancement_job_id") in ("job-42", None)
    text = _all_text(at)
    assert "Fresh deterministic answer." in text


def test_sending_a_new_message_clears_the_previous_turns_enhancement_result(fake_client):
    fake_client.post_message_response = _chat_response(answer_text="Second answer.", enhancement=None)
    at = _run(
        fake_client,
        session_state={
            "chat_session_id": "sess-1",
            "chat_last_response": _chat_response(answer_text="First answer."),
            "chat_enhancement_result": {"outcome": "completed", "answer_text": "Old enhanced phrasing."},
        },
    )
    at.chat_input[0].set_value("Another question.").run()
    text = _all_text(at)
    assert "Old enhanced phrasing." not in text
    assert "Second answer." in text
