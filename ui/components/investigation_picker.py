"""Shared investigation selector.

Phase 1 had this dropdown copy-pasted (with minor variations) in
Workspace, Log Intelligence, and AI Assistant. This is the one
implementation every page uses instead -- it reads/writes the shared
Investigation Context, so picking an investigation on any page is picked
everywhere else too.
"""

from __future__ import annotations

import streamlit as st
from api_client import list_investigations_cached
from context import get_active_investigation_id, set_active_investigation_id


def render_investigation_picker(*, key: str, label: str = "Investigation") -> str | None:
    """Renders a selectbox defaulted to the shared active investigation
    (if any) and updates the shared context when changed. Returns the
    selected investigation id, or None if there are no investigations yet."""
    investigations = list_investigations_cached() or []
    if not investigations:
        st.info("No investigations yet -- start one from the Dashboard.")
        return None

    options = {f"{inv['title']} ({inv['id'][:8]})": inv["id"] for inv in investigations}
    labels = list(options.keys())

    active_id = get_active_investigation_id()
    default_index = None
    for i, investigation_id in enumerate(options.values()):
        if investigation_id == active_id:
            default_index = i
            break

    selected_label = st.selectbox(label, labels, index=default_index, key=key)
    if selected_label is None:
        return None

    selected_id = options[selected_label]
    set_active_investigation_id(selected_id)
    return selected_id
