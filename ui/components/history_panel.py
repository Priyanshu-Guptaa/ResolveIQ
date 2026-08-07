"""History Panel (Sprint 3, Phase 3.4) -- renders the shared
``EntityVersion`` list for any object type, newest first. One component,
generic across all nine types since the History table itself is generic
(``app/domain/entity_version.py``).
"""

from __future__ import annotations

import json

import streamlit as st
from api_client import api_get


def render_history_panel(object_type: str, object_id: str) -> None:
    versions = api_get(f"/admin/objects/{object_type}/{object_id}/history") or []
    if not versions:
        st.caption("No history yet.")
        return

    for version in versions:
        with st.container(border=True):
            cols = st.columns([1, 3, 3])
            cols[0].markdown(f"**v{version['version_number']}**")
            cols[1].markdown(version["change_summary"] or "—")
            cols[2].caption(f"{version.get('changed_by') or 'unknown'} · {version['changed_at'].replace('T', ' ')[:19]}")
            with st.expander("Snapshot"):
                st.code(json.dumps(version["snapshot"], indent=2), language="json")
