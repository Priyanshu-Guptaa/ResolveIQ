"""Searchable knowledge-object picker (Sprint 3, Phase 3.3) -- "replace
free-text references wherever possible... use searchable entity
pickers." Every relationship editor interaction goes through this: pick
a type, search, pick a real object. There is no free-text entry point
to a relationship anywhere in this module.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get

_TYPE_LABELS = {
    "component": "Component",
    "document": "Document",
    "known_bug": "Known Bug",
    "sql_template": "SQL Template",
    "historical_investigation": "Historical Investigation",
    "playbook": "Playbook",
    "product": "Product",
    "technology": "Technology",
    "version": "Version",
}
OBJECT_TYPES = list(_TYPE_LABELS.keys())


def render_object_picker(*, key: str, label: str = "Object", default_type: str | None = None) -> tuple[str, str, str] | None:
    """Returns ``(object_type, object_id, display_title)`` for the
    selected object, or ``None`` if the current search has no results."""
    col1, col2 = st.columns([1, 2])
    default_index = OBJECT_TYPES.index(default_type) if default_type in OBJECT_TYPES else 0
    selected_type = col1.selectbox(
        f"{label} type", OBJECT_TYPES, index=default_index, format_func=lambda t: _TYPE_LABELS[t], key=f"{key}_type"
    )
    search_text = col2.text_input(f"Search {_TYPE_LABELS[selected_type]}s", key=f"{key}_search")

    results = (
        api_get("/admin/relationships/search-objects", params={"q": search_text, "object_type": selected_type, "limit": 100})
        or []
    )
    if not results:
        st.caption(f"No {_TYPE_LABELS[selected_type].lower()}s match \"{search_text}\"." if search_text else f"No {_TYPE_LABELS[selected_type].lower()}s exist yet.")
        return None

    def _option_label(ref: dict) -> str:
        return f"{ref['title']} — {ref['subtitle']}" if ref.get("subtitle") else ref["title"]

    options = {_option_label(r): r["id"] for r in results}
    selected_label = st.selectbox(f"Select a {_TYPE_LABELS[selected_type].lower()}", list(options.keys()), key=f"{key}_object")
    selected_id = options[selected_label]
    return selected_type, selected_id, selected_label
