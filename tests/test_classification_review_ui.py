"""Tests for the Classification Review Streamlit page (2026-08-14,
Phase 5, extended same day with the title/sort/group UI improvement --
``ui/views/14_Classification_Review.py``).

Uses ``streamlit.testing.v1.AppTest`` to run the real page script with a
fake ``api_client`` module substituted in ``sys.modules`` (so no real
HTTP call is ever made) -- introduced in the initial Phase 5 pass as
this codebase's first UI-level test, extended here rather than
duplicated. Everything the page calls is a real, unchanged existing
endpoint (``app/api/routers/admin/classification.py``); this file never
talks to the classification engine, repository, or database -- it only
proves the PAGE wires user actions to the RIGHT api_client calls, and
that viewing/sorting/filtering/grouping never calls a mutating endpoint.

Fixture suggestions now include ``object_title``/``mention_count`` --
the two fields the UI-improvement pass added to the (real,
API-provided) suggestion shape (``PendingSuggestionView``,
``app/domain/classification.py``). This file never computes either
field itself, exactly mirroring the real contract: the page only ever
displays what the (faked) API response already contains.
"""

from __future__ import annotations

import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, "ui")  # ui/theme.py, imported by the page itself, lives here

_PAGE_PATH = "ui/views/14_Classification_Review.py"


def _suggestion(**overrides) -> dict:
    defaults = dict(
        id="sugg-1", object_type="document", object_id="doc-11112222", dimension="technology",
        suggested_value_id="tech-1", suggested_value_text="RF Mesh IP", confidence_tier="medium",
        evidence_snippet="RF Mesh IP mentioned twice in the body.", evidence_rule="body_repeated_mention",
        status="pending", reviewed_by=None, reviewed_at=None, created_at="2026-01-01T00:00:00Z",
        object_title="98-1803-Rev-AB-RF-Mesh-IP-ZigBee", mention_count=5,
    )
    defaults.update(overrides)
    return defaults


class _FakeApiClient:
    """Records every call; GET/POST responses are configurable per test.
    Mirrors the real ``api_client`` module's contract: returns ``None``
    on failure (never raises), matching what the real HTTP-backed
    version does for the page code to react to."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None]] = []
        self.pending_response: list[dict] | None = []
        self.post_responses: dict[str, dict | None] = {}
        self.default_post_response: dict | None = {"ok": True}

    def api_get(self, path, params=None):
        self.calls.append(("GET", path, params))
        if path == "/admin/classification/pending":
            return self.pending_response
        return None

    def api_post(self, path, json_body=None, files=None, timeout=60):
        self.calls.append(("POST", path, json_body))
        if path in self.post_responses:
            return self.post_responses[path]
        return self.default_post_response

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
    monkeypatch.setitem(sys.modules, "api_client", module)
    yield client


def _run(fake_client, pending):
    fake_client.pending_response = pending
    at = AppTest.from_file(_PAGE_PATH)
    at.run()
    assert not at.exception, f"Page raised: {at.exception}"
    return at


def _post_calls(fake_client):
    return [c for c in fake_client.calls if c[0] == "POST"]


def _all_text(at) -> str:
    return "\n".join(m.value for m in at.markdown) + "\n" + "\n".join(c.value for c in at.caption) + "\n" + "\n".join(w.value for w in at.warning)


# --- 1. Page registration --------------------------------------------------


def test_page_registered_in_navigation():
    home = open("ui/Home.py", encoding="utf-8").read()
    assert 'st.Page("views/14_Classification_Review.py"' in home


def test_page_file_exists():
    import os

    assert os.path.isfile(_PAGE_PATH)


# --- 2. Pending suggestions load from the existing API ----------------------


def test_pending_suggestions_load_from_api(fake_client):
    at = _run(fake_client, [_suggestion()])
    assert ("GET", "/admin/classification/pending", None) in fake_client.calls
    full_text = "\n".join(m.value for m in at.markdown)
    assert "1 pending suggestion" in full_text
    assert "RF Mesh IP" in full_text


def test_empty_queue_shows_no_pending_message(fake_client):
    at = _run(fake_client, [])
    full_text = "\n".join(m.value for m in at.markdown) + "\n".join(i.value for i in at.info)
    assert "0 pending suggestion" in full_text
    assert any("empty" in i.value.lower() for i in at.info)


# --- Object title display (new) ---------------------------------------------


def test_object_title_is_displayed_when_available(fake_client):
    at = _run(fake_client, [_suggestion(object_title="Real Document Title From The Record")])
    assert "Real Document Title From The Record" in _all_text(at)


def test_object_title_never_derived_from_evidence_snippet(fake_client):
    """The title shown must be exactly what the API supplied -- never a
    string built from evidence_snippet by the page itself."""
    at = _run(
        fake_client,
        [_suggestion(object_title="The Real Title", evidence_snippet="Completely different evidence text about NMS servers.")],
    )
    assert "The Real Title" in _all_text(at)


def test_uuid_fallback_and_unavailable_notice_when_title_is_none(fake_client):
    at = _run(fake_client, [_suggestion(object_title=None, object_id="deleted-object-id-123")])
    text = _all_text(at)
    assert "deleted-object-id-123" in text
    assert "unavailable" in text.lower()


# --- Mention count / evidence strength display (new) -------------------------


def test_mention_count_is_displayed(fake_client):
    at = _run(fake_client, [_suggestion(mention_count=17)])
    assert "17 mention" in _all_text(at)


def test_missing_mention_count_shown_as_not_available_not_zero(fake_client):
    at = _run(fake_client, [_suggestion(mention_count=None)])
    text = _all_text(at)
    assert "mentions: n/a" in text
    assert "0 mention" not in text


# --- 3. Filters (existing, preserved) ---------------------------------------


def test_dimension_filter_narrows_the_list(fake_client):
    items = [
        _suggestion(id="s1", dimension="technology", suggested_value_text="RF Mesh IP", object_title="Doc A", evidence_snippet="tech evidence"),
        _suggestion(id="s2", dimension="customer", suggested_value_text="TEPCO", object_id="hist-1", object_title="Doc B", evidence_snippet="cust evidence"),
    ]
    at = _run(fake_client, items)
    at.multiselect(key="classification_filter_dimension").set_value(["customer"]).run()
    assert not at.exception
    full_text = "\n".join(m.value for m in at.markdown)
    assert "Showing 1 of 2" in "\n".join(c.value for c in at.caption)
    assert "TEPCO" in full_text
    assert "RF Mesh IP" not in full_text


def test_object_type_filter_narrows_the_list(fake_client):
    items = [
        _suggestion(id="s1", object_type="document", object_id="doc-1"),
        _suggestion(id="s2", object_type="historical_investigation", object_id="hist-1"),
    ]
    at = _run(fake_client, items)
    at.multiselect(key="classification_filter_object_type").set_value(["historical_investigation"]).run()
    assert not at.exception
    assert "Showing 1 of 2" in "\n".join(c.value for c in at.caption)


def test_confidence_filter_still_present(fake_client):
    at = _run(fake_client, [_suggestion()])
    assert at.multiselect(key="classification_filter_tier") is not None


# --- Sorting (new) -----------------------------------------------------------


def test_default_sort_orders_by_dimension_then_value_then_mentions_desc(fake_client):
    items = [
        _suggestion(id="s1", dimension="technology", suggested_value_text="RF Mesh", object_title="TitleLow", mention_count=2),
        _suggestion(id="s2", dimension="technology", suggested_value_text="RF Mesh", object_title="TitleHigh", mention_count=20),
        _suggestion(id="s3", dimension="component", suggested_value_text="Scheduler", object_title="TitleComp", mention_count=99),
    ]
    at = _run(fake_client, items)
    names = ("TitleLow", "TitleHigh", "TitleComp")
    titles_in_order = [next(t for t in names if t in m.value) for m in at.markdown if any(t in m.value for t in names)]
    # component dimension sorts before technology alphabetically; within the
    # same (dimension, value) group, higher mention count comes first.
    assert titles_in_order == ["TitleComp", "TitleHigh", "TitleLow"]


def test_sort_by_mention_count_descending(fake_client):
    items = [
        _suggestion(id="s1", suggested_value_text="A", object_title="Weak", mention_count=2),
        _suggestion(id="s2", suggested_value_text="B", object_title="Strong", mention_count=50),
    ]
    at = _run(fake_client, items)
    at.selectbox(key="classification_sort_by").set_value("Mention count").run()
    assert not at.exception
    titles_in_order = [next(t for t in ("Weak", "Strong") if t in m.value) for m in at.markdown if any(t in m.value for t in ("Weak", "Strong"))]
    assert titles_in_order == ["Strong", "Weak"]  # descending is the default for mention count


def test_sort_by_object_title_ascending(fake_client):
    items = [
        _suggestion(id="s1", object_title="Zebra Document"),
        _suggestion(id="s2", object_title="Alpha Document"),
    ]
    at = _run(fake_client, items)
    at.selectbox(key="classification_sort_by").set_value("Object title").run()
    assert not at.exception
    names = ("Zebra Document", "Alpha Document")
    titles_in_order = [next(t for t in names if t in m.value) for m in at.markdown if any(t in m.value for t in names)]
    assert titles_in_order == ["Alpha Document", "Zebra Document"]


# --- Grouping (new) -----------------------------------------------------------


def test_group_by_dimension_and_value_creates_expanders_with_counts(fake_client):
    items = [
        _suggestion(id="s1", dimension="component", suggested_value_text="Scheduler", object_title="Doc A"),
        _suggestion(id="s2", dimension="component", suggested_value_text="Scheduler", object_title="Doc B"),
        _suggestion(id="s3", dimension="component", suggested_value_text="NMS", object_title="Doc C"),
    ]
    at = _run(fake_client, items)
    at.selectbox(key="classification_group_by").set_value("Dimension → Suggested value").run()
    assert not at.exception
    expander_labels = [e.label for e in at.expander if "→" in e.label or "Scheduler" in e.label or "NMS" in e.label]
    assert any("Scheduler" in lbl and "(2)" in lbl for lbl in expander_labels)
    assert any("NMS" in lbl and "(1)" in lbl for lbl in expander_labels)


def test_group_by_none_shows_flat_paginated_list(fake_client):
    at = _run(fake_client, [_suggestion(id="s1"), _suggestion(id="s2", object_id="doc-2")])
    assert at.selectbox(key="classification_group_by").value == "None (flat list)"
    assert at.number_input(key="classification_page") is not None


# --- Accept / Reject still use the existing endpoints, unchanged -------------


def test_accept_calls_accept_endpoint_with_correct_id_and_actor(fake_client):
    at = _run(fake_client, [_suggestion(id="sugg-42")])
    at.text_input(key="classification_review_actor").set_value("jdoe").run()

    at.button(key="classification_accept_sugg-42").click().run()
    assert not at.exception
    assert not _post_calls(fake_client)  # confirm state only -- no API call yet

    at.button(key="classification_confirm_sugg-42_yes").click().run()
    assert not at.exception
    assert _post_calls(fake_client) == [("POST", "/admin/classification/sugg-42/accept", {"actor": "jdoe"})]


def test_reject_calls_reject_endpoint_with_correct_id_and_actor(fake_client):
    at = _run(fake_client, [_suggestion(id="sugg-99")])
    at.text_input(key="classification_review_actor").set_value("jdoe").run()

    at.button(key="classification_reject_sugg-99").click().run()
    at.button(key="classification_confirm_sugg-99_yes").click().run()
    assert not at.exception
    assert _post_calls(fake_client) == [("POST", "/admin/classification/sugg-99/reject", {"actor": "jdoe"})]


def test_cancel_does_not_call_the_api(fake_client):
    at = _run(fake_client, [_suggestion(id="sugg-7")])
    at.button(key="classification_accept_sugg-7").click().run()
    at.button(key="classification_confirm_sugg-7_no").click().run()
    assert not at.exception
    assert not _post_calls(fake_client)
    assert at.button(key="classification_accept_sugg-7") is not None


def test_no_bulk_accept_or_reject_controls_exist(fake_client):
    """Explicit, negative regression: this UI pass must never add a
    bulk-action control."""
    at = _run(fake_client, [_suggestion(id="s1"), _suggestion(id="s2", object_id="doc-2")])
    button_labels = " ".join((b.label or "") for b in at.button)
    assert "bulk" not in button_labels.lower()
    assert "accept all" not in button_labels.lower()
    assert "reject all" not in button_labels.lower()


# --- Viewing/sorting/filtering/grouping never mutates ------------------------


def test_viewing_sorting_filtering_grouping_makes_no_mutating_call(fake_client):
    items = [
        _suggestion(id="s1", dimension="technology", suggested_value_text="RF Mesh IP", object_title="Doc A"),
        _suggestion(id="s2", dimension="component", suggested_value_text="Scheduler", object_title="Doc B", object_id="doc-2"),
    ]
    at = _run(fake_client, items)
    at.multiselect(key="classification_filter_dimension").set_value(["component"]).run()
    at.selectbox(key="classification_sort_by").set_value("Mention count").run()
    at.selectbox(key="classification_group_by").set_value("Dimension → Suggested value").run()
    assert not at.exception
    assert not _post_calls(fake_client)
    get_paths = [c[1] for c in fake_client.calls if c[0] == "GET"]
    assert all(p == "/admin/classification/pending" for p in get_paths)


# --- Refresh after accept/reject ---------------------------------------------


def test_queue_refreshes_after_accept(fake_client):
    at = _run(fake_client, [_suggestion(id="sugg-1", suggested_value_text="RF Mesh IP", object_title="Before")])
    at.button(key="classification_accept_sugg-1").click().run()

    fake_client.pending_response = [_suggestion(id="sugg-2", suggested_value_text="RF Mesh", object_title="After", object_id="doc-2")]
    at.button(key="classification_confirm_sugg-1_yes").click().run()
    assert not at.exception

    get_calls = [c for c in fake_client.calls if c[0] == "GET"]
    assert len(get_calls) >= 2
    full_text = "\n".join(m.value for m in at.markdown)
    assert "Before" not in full_text
    assert "After" in full_text


# --- Evidence snippet/rule displayed unchanged -------------------------------


def test_evidence_snippet_and_rule_displayed_verbatim(fake_client):
    at = _run(
        fake_client,
        [_suggestion(evidence_snippet="Exact evidence text, unmodified.", evidence_rule="body_repeated_mention")],
    )
    markdown_text = "\n".join(m.value for m in at.markdown)
    snippet_text = "\n".join(t.value for t in at.text)
    assert "body repeated mention" in markdown_text
    assert "Exact evidence text, unmodified." in snippet_text


def test_evidence_snippet_with_dollar_signs_rendered_literally_not_as_latex(fake_client):
    """Real evidence (PowerShell/regex snippets) can contain '$...$' --
    st.markdown would otherwise misrender that as LaTeX. Found live
    during this pass's own verification; the snippet must render via
    st.text (plain, literal), never st.markdown."""
    at = _run(fake_client, [_suggestion(evidence_snippet='$_.address="http://x"} $mode.configuration')])
    snippet_text = "\n".join(t.value for t in at.text)
    assert '$_.address="http://x"} $mode.configuration' in snippet_text


# --- API errors surfaced cleanly, never a crash ---------------------------


def test_pending_fetch_failure_does_not_crash(fake_client):
    fake_client.pending_response = None
    at = AppTest.from_file(_PAGE_PATH)
    at.run()
    assert not at.exception


def test_accept_failure_does_not_crash_and_shows_no_false_success(fake_client):
    at = _run(fake_client, [_suggestion(id="sugg-5")])
    fake_client.post_responses["/admin/classification/sugg-5/accept"] = None
    at.button(key="classification_accept_sugg-5").click().run()
    at.button(key="classification_confirm_sugg-5_yes").click().run()
    assert not at.exception
    assert not any("Accepted" in s.value for s in at.success)
