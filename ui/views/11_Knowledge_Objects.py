"""Knowledge Objects -- Sprint 3, Phase 3.4's Administration module,
and the demonstration that the Knowledge Object Framework actually
works: the exact same ``render_knowledge_editor`` shell, with zero
per-type branching, drives Create/Edit/Publish/Archive/Restore/
Deprecate/Delete/Relationships/History/Validation/Impact for all nine
object types below -- including the six (Known Bug, SQL Template,
Playbook, Product, Technology, Version) that had no Administration UI
at all before this phase.

Document and Component keep their specialized modules (Knowledge
Management's upload/extract pipeline; Product Intelligence's
Architecture Explorer) as the primary way to work with those two types
-- this page is still a fully working generic secondary view for them,
proving the framework doesn't special-case anything, but an admin
managing documents day-to-day should keep using Knowledge Management.

No role gate: see app/api/routers/admin/__init__.py.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, api_post, ensure_api_available

from components.knowledge_editor import render_knowledge_editor
from components.object_types import MANUAL_CREATE_UNSUPPORTED, OBJECT_TYPES, TYPE_LABELS
from theme import inject_theme

inject_theme()
st.title("🗃️ Knowledge Objects")
st.caption(
    "Administration · Create, edit, and manage the lifecycle of any governed knowledge object through one "
    "reusable framework -- Components, Documents, Known Bugs, SQL Templates, Historical Investigations, "
    "Playbooks, Products, Technologies, Versions, and the Log Intelligence Knowledge Base."
)
ensure_api_available()

col_type, col_new = st.columns([3, 1])
object_type = col_type.selectbox("Object type", OBJECT_TYPES, format_func=lambda t: TYPE_LABELS[t], key="ko_type")
show_create = col_new.toggle("➕ New", key="ko_show_create")

if show_create and object_type in MANUAL_CREATE_UNSUPPORTED:
    st.info(
        f"{TYPE_LABELS[object_type]} records are created by importing a wiki page -- see **Log Wiki Import** "
        "in the sidebar. This page is for browsing, editing, and managing their lifecycle once they exist."
    )
elif show_create:
    with st.form(key=f"ko_create_{object_type}"):
        st.markdown(f"##### New {TYPE_LABELS[object_type]}")
        title_field = "name" if object_type in ("product", "technology", "version") else "title"
        title_value = st.text_input(title_field.title())
        description_value = ""
        if object_type in ("known_bug", "historical_investigation"):
            description_value = st.text_area("Description")
        submitted = st.form_submit_button("Create", type="primary")
    if submitted:
        if not title_value.strip():
            st.error(f"{title_field.title()} is required.")
        else:
            fields = {title_field: title_value}
            if object_type == "known_bug":
                fields["description"] = description_value or "—"
            elif object_type == "historical_investigation":
                fields.update(description=description_value or "—", root_cause="—", resolution="—")
            elif object_type == "sql_template":
                fields.update(category="general", sql_text="-- TODO", explanation="—")
            created = api_post(f"/admin/objects/{object_type}", {"fields": fields, "actor": "admin"})
            if created is not None:
                st.success(f"Created {TYPE_LABELS[object_type]} \"{title_value}\".")
                st.session_state[f"selected_{object_type}"] = created["id"]
                st.rerun()

st.divider()

refs = api_get(f"/admin/objects/{object_type}") or []
if not refs:
    st.info(f"No {TYPE_LABELS[object_type].lower()}s yet -- use “New” above to create one.")
    st.stop()


def _option_label(ref: dict) -> str:
    return f"{ref['title']} — {ref['subtitle']}" if ref.get("subtitle") else ref["title"]


options = {_option_label(r): r["id"] for r in refs}
default_id = st.session_state.get(f"selected_{object_type}")
labels = list(options.keys())
default_index = 0
if default_id in options.values():
    default_index = list(options.values()).index(default_id)

selected_label = st.selectbox(f"{TYPE_LABELS[object_type]}s ({len(refs)})", labels, index=default_index, key=f"ko_pick_{object_type}")
selected_id = options[selected_label]
st.session_state[f"selected_{object_type}"] = selected_id

obj = api_get(f"/admin/objects/{object_type}/{selected_id}")
if obj:
    render_knowledge_editor(object_type, obj)
