"""Document Details / Preview / Metadata / Validate / Lifecycle panel --
Knowledge Management's per-document view (Sprint 3, Phase 3.2). Shared
between the Library (an admin opens an existing document) and Upload
(review a just-created Draft before publishing).
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, api_patch, api_post, get_component_profiles_cached

from components.document_status_badge import document_status_badge_html


def render_document_detail(document_id: str, *, key_prefix: str | None = None) -> None:
    prefix = key_prefix or document_id
    document = api_get(f"/admin/knowledge/documents/{document_id}")
    if document is None:
        st.warning("This document no longer exists.")
        return

    st.markdown(
        f"### {document['title']} &nbsp; {document_status_badge_html(document['status'])}",
        unsafe_allow_html=True,
    )
    meta_cols = st.columns(4)
    meta_cols[0].caption(f"Source: **{document['source']}**")
    meta_cols[1].caption(f"Created by: **{document['created_by'] or '—'}**")
    meta_cols[2].caption(f"Created: **{document['created_at'][:10]}**")
    meta_cols[3].caption(f"Updated: **{document['updated_at'][:10]}**")
    if document.get("original_filename"):
        st.caption(f"Uploaded from: `{document['original_filename']}` ({document.get('file_type', '—')})")

    tab_preview, tab_metadata, tab_validate = st.tabs(["Preview", "Metadata", "Validate"])

    with tab_preview:
        if document["content"].strip():
            st.text_area(
                "Extracted text",
                value=document["content"],
                height=280,
                disabled=True,
                key=f"{prefix}_preview",
                label_visibility="collapsed",
            )
        else:
            st.info("No extracted text -- this file type may not have been recognized, or the source page had none.")

    with tab_metadata:
        components = get_component_profiles_cached() or []
        component_names = [c["name"] for c in components]
        with st.form(f"{prefix}_metadata_form"):
            title = st.text_input("Title", value=document["title"], key=f"{prefix}_title")
            tags_text = st.text_input(
                "Tags (comma-separated)", value=", ".join(document["tags"]), key=f"{prefix}_tags"
            )
            col1, col2, col3 = st.columns(3)
            product = col1.text_input("Product", value=document.get("product") or "", key=f"{prefix}_product")
            version = col2.text_input("Version", value=document.get("version") or "", key=f"{prefix}_version")
            technology = col3.text_input(
                "Technology", value=document.get("technology") or "", key=f"{prefix}_technology"
            )
            related = st.multiselect(
                "Related components",
                options=component_names,
                default=[c for c in document.get("related_components", []) if c in component_names],
                key=f"{prefix}_components",
            )
            saved = st.form_submit_button("Save metadata")
        if saved:
            payload = {
                "title": title,
                "tags": [t.strip() for t in tags_text.split(",") if t.strip()],
                "product": product or None,
                "version": version or None,
                "technology": technology or None,
                "related_components": related,
            }
            result = api_patch(f"/admin/knowledge/documents/{document_id}", payload)
            if result is not None:
                st.success("Metadata saved.")
                st.rerun()

    with tab_validate:
        if st.button("Run validation", key=f"{prefix}_validate_btn"):
            warnings = api_get(f"/admin/knowledge/documents/{document_id}/validate")
            st.session_state[f"{prefix}_validation_result"] = warnings
        warnings = st.session_state.get(f"{prefix}_validation_result")
        if warnings is not None:
            if not warnings:
                st.success("No issues found.")
            else:
                for warning in warnings:
                    st.warning(warning)

    st.divider()
    status = document["status"]
    action_cols = st.columns(3)
    if status in ("draft", "under_review"):
        if action_cols[0].button("📤 Publish", key=f"{prefix}_publish", type="primary", use_container_width=True):
            if api_post(f"/admin/knowledge/documents/{document_id}/publish") is not None:
                st.success("Published -- now searchable.")
                st.rerun()
        if action_cols[1].button("🗄 Archive", key=f"{prefix}_archive_from_draft", use_container_width=True):
            if api_post(f"/admin/knowledge/documents/{document_id}/archive") is not None:
                st.rerun()
    elif status == "published":
        if action_cols[0].button("🗄 Archive", key=f"{prefix}_archive", use_container_width=True):
            if api_post(f"/admin/knowledge/documents/{document_id}/archive") is not None:
                st.success("Archived -- removed from search.")
                st.rerun()
    elif status == "archived":
        if action_cols[0].button("↩ Restore to Draft", key=f"{prefix}_restore", use_container_width=True):
            if api_post(f"/admin/knowledge/documents/{document_id}/restore") is not None:
                st.rerun()
