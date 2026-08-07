"""Metadata Panel (Sprint 3, Phase 3.4 Knowledge Object Framework) --
one reusable component for every governed object's metadata, universal
fields only, per the phase's explicit "do not build object-specific
business rules yet."

Two halves:
 - Governance strip: the six ``GovernanceFields`` every one of the nine
   types shares (status, created/updated by+at, active) -- always
   read-only here, changed only via the lifecycle buttons in
   ``knowledge_editor.py``.
 - Field editor: every OTHER top-level field on the object, rendered
   generically from its current value's Python type. This is what lets
   one component edit a Known Bug's ``description`` and a Product's
   ``name`` without knowing either exists -- it never hardcodes a
   field list per type.
"""

from __future__ import annotations

import json

import streamlit as st
from api_client import api_patch

from components.object_status_badge import object_status_badge_html

_GOVERNANCE_FIELDS = {"id", "status", "created_at", "updated_at", "created_by", "updated_by", "is_active"}


def render_metadata_panel(obj: dict, *, object_type: str, actor: str = "admin") -> None:
    cols = st.columns([2, 2, 2, 1])
    cols[0].markdown(f"**Status** &nbsp; {object_status_badge_html(obj.get('status', ''))}", unsafe_allow_html=True)
    cols[1].caption(f"Created by {obj.get('created_by') or 'unknown'} · {_short(obj.get('created_at'))}")
    cols[2].caption(f"Updated by {obj.get('updated_by') or 'unknown'} · {_short(obj.get('updated_at'))}")
    cols[3].caption("Active" if obj.get("is_active", True) else "Inactive")

    st.divider()
    st.markdown("##### Fields")
    editable = {k: v for k, v in obj.items() if k not in _GOVERNANCE_FIELDS}
    with st.form(key=f"metadata_form_{object_type}_{obj['id']}"):
        new_values: dict = {}
        for field_name, value in editable.items():
            new_values[field_name] = _render_field_input(field_name, value, key_prefix=f"{object_type}_{obj['id']}")
        submitted = st.form_submit_button("💾 Save changes", type="primary")

    if submitted:
        changed = {k: v for k, v in new_values.items() if v != editable[k]}
        if not changed:
            st.info("No changes to save.")
        else:
            updated = api_patch(f"/admin/objects/{object_type}/{obj['id']}", {"fields": changed, "actor": actor})
            if updated is not None:
                st.success("Saved.")
                st.rerun()


def _short(iso_timestamp: str | None) -> str:
    if not iso_timestamp:
        return "—"
    return iso_timestamp.replace("T", " ")[:19]


def _render_field_input(field_name: str, value, *, key_prefix: str):
    label = field_name.replace("_", " ").title()
    key = f"{key_prefix}_{field_name}"
    if isinstance(value, bool):
        return st.checkbox(label, value=value, key=key)
    if isinstance(value, (int, float)):
        return st.number_input(label, value=value, key=key)
    if isinstance(value, (list, dict)):
        raw = st.text_area(f"{label} (JSON)", value=json.dumps(value, indent=2), key=key)
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            st.caption(f"⚠️ {label}: invalid JSON, keeping previous value.")
            return value
    if isinstance(value, str) and len(value) > 120:
        return st.text_area(label, value=value, key=key)
    return st.text_input(label, value="" if value is None else str(value), key=key)
