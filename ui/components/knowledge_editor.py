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
