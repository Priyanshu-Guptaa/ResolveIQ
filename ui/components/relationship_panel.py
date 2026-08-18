"""Relationship Panel (Sprint 3, Phase 3.4) -- every Knowledge Editor
shows Incoming/Outgoing relationships for the selected object, split
client-side from the same ``ResolvedRelationship`` list Phase 3.3's
Relationship Manager already fetches from ``/admin/relationships`` --
no new backend endpoint, no duplicated relationship logic.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_delete, api_get

from components.object_types import TYPE_LABELS

_RELATIONSHIP_TYPES = ["related_to", "documents", "fixes", "requires", "uses", "applies_to", "supersedes"]


def _ref_line(ref: dict) -> str:
    label = TYPE_LABELS.get(ref["type"], ref["type"])
    subtitle = f" — {ref['subtitle']}" if ref.get("subtitle") else ""
    return f"`{label}` **{ref['title']}**{subtitle}"


def render_relationship_panel(object_type: str, object_id: str) -> None:
    relationships = api_get("/admin/relationships", params={"object_type": object_type, "object_id": object_id}) or []
    incoming = [r for r in relationships if r["to_object"]["type"] == object_type and r["to_object"]["id"] == object_id]
    outgoing = [r for r in relationships if r["from_object"]["type"] == object_type and r["from_object"]["id"] == object_id]

    col_in, col_out = st.columns(2)
    with col_in:
        st.markdown(f"##### ⬅️ Incoming ({len(incoming)})")
        if not incoming:
            st.caption("Nothing points to this object yet.")
        for rel in incoming:
            with st.container(border=True):
                cols = st.columns([5, 1])
                cols[0].markdown(f"{_ref_line(rel['from_object'])} — *{rel['relationship']['relationship_type']}*")
                if cols[1].button("🗑", key=f"rel_in_{rel['relationship']['id']}", help="Remove relationship"):
                    if api_delete(f"/admin/relationships/{rel['relationship']['id']}"):
                        st.rerun()

    with col_out:
        st.markdown(f"##### ➡️ Outgoing ({len(outgoing)})")
        if not outgoing:
            st.caption("This object doesn't point to anything yet.")
        for rel in outgoing:
            with st.container(border=True):
                cols = st.columns([5, 1])
                cols[0].markdown(f"*{rel['relationship']['relationship_type']}* → {_ref_line(rel['to_object'])}")
                if cols[1].button("🗑", key=f"rel_out_{rel['relationship']['id']}", help="Remove relationship"):
                    if api_delete(f"/admin/relationships/{rel['relationship']['id']}"):
                        st.rerun()

    st.divider()
    st.caption("Full graph view: open Relationship Manager → Explorer, centered on this object.")
