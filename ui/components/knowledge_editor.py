"""Knowledge Editor (Sprint 3, Phase 3.4) -- the shell every knowledge
object type is edited through: lifecycle buttons (Publish / Archive /
Restore / Deprecate / Delete) plus Metadata / Relationships / History /
Validation / Impact tabs, each a reusable panel. This is the piece that
proves the framework's brief -- "the same components should work across
all knowledge object types" -- since this single function is what
``11_Knowledge_Objects.py`` calls for every one of the nine types
without a single ``if object_type == ...`` branch.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_delete, api_post

from components.history_panel import render_history_panel
from components.impact_summary import render_impact_summary
from components.metadata_panel import render_metadata_panel
from components.object_status_badge import object_status_badge_html
from components.relationship_panel import render_relationship_panel
from components.validation_summary import render_validation_summary

_LIFECYCLE_ACTIONS = [
    ("📢 Publish", "publish"),
    ("📦 Archive", "archive"),
    ("↩️ Restore to Draft", "restore"),
    ("🚫 Deprecate", "deprecate"),
]

_VERIFICATION_URL_SEGMENT = {
    "historical_investigation": "historical-investigations",
    "known_bug": "known-bugs",
}
"""Object types with a Resolution Verification panel (2026-08-13, Phase
0) -- deliberately just these two, not a ninth generic tab: verification
only means something for a type that carries ``resolution_verified*``
fields at all (see app/domain/evidence.py), and this is a real,
consequence-bearing action (a CONFIRMED tier becomes visible to every
future recommendation on this record), not ordinary metadata editing --
it belongs in its own clearly-labeled panel, not folded into the
generic Metadata tab's field-agnostic form (which deliberately excludes
these four fields entirely, see metadata_panel.py)."""


def render_knowledge_editor(object_type: str, obj: dict, *, actor: str = "admin") -> None:
    header_cols = st.columns([4, 1])
    header_cols[0].markdown(
        f"## {obj.get('title') or obj.get('name') or obj['id']} &nbsp; "
        f"{object_status_badge_html(obj.get('status', ''))}",
        unsafe_allow_html=True,
    )

    action_cols = st.columns(len(_LIFECYCLE_ACTIONS) + 1)
    for col, (label, action) in zip(action_cols, _LIFECYCLE_ACTIONS):
        if col.button(label, key=f"lifecycle_{action}_{obj['id']}"):
            result = api_post(f"/admin/objects/{object_type}/{obj['id']}/{action}", {"actor": actor})
            if result is not None:
                st.success(f"{label.split(' ', 1)[1]} done.")
                st.rerun()

    if action_cols[-1].button("🗑️ Delete", key=f"lifecycle_delete_{obj['id']}"):
        if api_delete(f"/admin/objects/{object_type}/{obj['id']}"):
            st.success("Deleted.")
            st.session_state.pop(f"selected_{object_type}", None)
            st.rerun()
        # api_delete already surfaced the 409 "has dependents" detail via st.error.

    if object_type in _VERIFICATION_URL_SEGMENT:
        _render_resolution_verification_panel(object_type, obj, actor=actor)

    tab_metadata, tab_relationships, tab_history, tab_validation, tab_impact = st.tabs(
        ["📋 Metadata", "🕸️ Relationships", "🕒 History", "✅ Validation", "🎯 Impact"]
    )
    with tab_metadata:
        render_metadata_panel(obj, object_type=object_type, actor=actor)
    with tab_relationships:
        render_relationship_panel(object_type, obj["id"])
    with tab_history:
        render_history_panel(object_type, obj["id"])
    with tab_validation:
        render_validation_summary(object_type, obj["id"])
    with tab_impact:
        render_impact_summary(object_type, obj["id"])


def _render_resolution_verification_panel(object_type: str, obj: dict, *, actor: str) -> None:
    """The dedicated, audit-paired action for
    ``resolution_verified``/``resolution_verified_by``/
    ``resolution_verified_at``/``resolution_verification_note`` (2026-08-13,
    Phase 0) -- see this module's ``_VERIFICATION_URL_SEGMENT`` docstring
    for why this is separate from the generic Metadata tab. Backed by
    ``app/api/routers/admin/resolution_verification.py``; never the
    generic ``PATCH /admin/objects/...`` endpoint, which rejects these
    four field names outright."""
    segment = _VERIFICATION_URL_SEGMENT[object_type]
    is_verified = bool(obj.get("resolution_verified"))

    with st.expander("✅ Resolution Verification", expanded=is_verified):
        if is_verified:
            st.success(
                f"Verified by **{obj.get('resolution_verified_by') or 'unknown'}** "
                f"on {(obj.get('resolution_verified_at') or '').replace('T', ' ')[:19] or 'an unrecorded date'}."
            )
            if obj.get("resolution_verification_note"):
                st.caption(obj["resolution_verification_note"])
            st.caption(
                "This record can reach the CONFIRMED resolution-provenance tier for future recommendations "
                "while verified."
            )
            if st.button("↩️ Remove verification", key=f"unverify_{obj['id']}"):
                if api_post(f"/admin/{segment}/{obj['id']}/unverify", {"actor": actor}) is not None:
                    st.success("Verification removed.")
                    st.rerun()
        else:
            st.caption(
                "Not verified -- this record can still be recommended (Likely/Possible), but can never reach "
                "the CONFIRMED tier without either this or a real cross-source correlation. Only mark this "
                "once the resolution has genuinely been confirmed to work, not merely because it looks right."
            )
            with st.form(key=f"verify_form_{obj['id']}"):
                note = st.text_area(
                    "Verification note (recommended)",
                    placeholder='e.g. "Confirmed with the customer after applying the fix; issue did not recur."',
                    key=f"verify_note_{obj['id']}",
                )
                submitted = st.form_submit_button("✅ Mark resolution verified", type="primary")
            if submitted:
                payload = {"actor": actor, "note": note or None}
                if api_post(f"/admin/{segment}/{obj['id']}/verify", payload) is not None:
                    st.success("Marked verified.")
                    st.rerun()
