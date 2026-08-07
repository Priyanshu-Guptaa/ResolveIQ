"""Knowledge Management -- Sprint 3, Phase 3.2's first Administration
module (RFC-003). The primary interface administrators use to grow
ResolveIQ's knowledge base: upload -> preview -> extract -> metadata ->
validate -> publish, plus a searchable/filterable/sortable library and
a status dashboard.

No role gate: authentication (RFC-003's User Management) is a later,
not-yet-scheduled phase. See app/api/routers/admin/__init__.py.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, api_post, ensure_api_available

from components.document_detail import render_document_detail
from components.document_status_badge import document_status_badge_html
from theme import inject_theme

inject_theme()
st.title("🗂️ Knowledge Management")
st.caption(
    "Administration · this is what grows the Knowledge Center search, the Recommendation Engine, "
    "and the Architecture Explorer over time -- not hardcoded data."
)
ensure_api_available()

tab_dashboard, tab_library, tab_upload = st.tabs(["📊 Dashboard", "📚 Library", "⬆ Upload"])

# === Dashboard ===============================================================
with tab_dashboard:
    stats = api_get("/admin/knowledge/dashboard")
    if stats:
        cols = st.columns(5)
        cols[0].metric("Total Documents", stats["total_count"])
        cols[1].metric("Published", stats["published_count"])
        cols[2].metric("Draft", stats["draft_count"])
        cols[3].metric("Under Review", stats["under_review_count"])
        cols[4].metric("Archived", stats["archived_count"])

        st.markdown("##### Recently Added")
        if not stats["recently_added"]:
            st.caption("No documents yet -- upload one from the Upload tab to get started.")
        for doc in stats["recently_added"]:
            st.markdown(
                f"{document_status_badge_html(doc['status'])} &nbsp; **{doc['title']}** "
                f"&nbsp; <span style='opacity:0.6'>{doc['created_at'][:10]}</span>",
                unsafe_allow_html=True,
            )

# === Library (search / filter / sort / pagination) ==========================
with tab_library:
    filter_cols = st.columns([2, 1, 1, 1])
    search = filter_cols[0].text_input("Search title", key="km_search", placeholder="e.g. GLP, Kafka, blocking…")
    status_filter = filter_cols[1].selectbox(
        "Status", ["All", "draft", "under_review", "published", "archived"], key="km_status"
    )
    sort_by = filter_cols[2].selectbox("Sort by", ["updated_at", "created_at", "title", "status"], key="km_sort")
    sort_desc = filter_cols[3].checkbox("Newest first", value=True, key="km_sort_desc")

    filter_cols2 = st.columns(2)
    product_filter = filter_cols2[0].text_input("Product", key="km_product")
    technology_filter = filter_cols2[1].text_input("Technology", key="km_technology")

    if "km_page" not in st.session_state:
        st.session_state["km_page"] = 1

    result = api_get(
        "/admin/knowledge/documents",
        params={
            "search": search or None,
            "status": None if status_filter == "All" else status_filter,
            "product": product_filter or None,
            "technology": technology_filter or None,
            "sort_by": sort_by,
            "sort_desc": sort_desc,
            "page": st.session_state["km_page"],
            "page_size": 10,
        },
    )

    if result:
        st.caption(f"{result['total_count']} document(s) · page {result['page']} of {result['total_pages']}")
        if not result["items"]:
            st.info("No documents match these filters.")
        for item in result["items"]:
            with st.container(border=True):
                row = st.columns([4, 1, 1, 1])
                row[0].markdown(
                    f"{document_status_badge_html(item['status'])} &nbsp; **{item['title']}**",
                    unsafe_allow_html=True,
                )
                row[1].caption(item.get("product") or "—")
                row[2].caption(item.get("technology") or "—")
                if row[3].button("Open →", key=f"km_open_{item['id']}", use_container_width=True):
                    st.session_state["km_selected_document"] = item["id"]

        nav_cols = st.columns([1, 1, 6])
        if nav_cols[0].button("← Previous", disabled=result["page"] <= 1, key="km_prev"):
            st.session_state["km_page"] -= 1
            st.rerun()
        if nav_cols[1].button("Next →", disabled=result["page"] >= result["total_pages"], key="km_next"):
            st.session_state["km_page"] += 1
            st.rerun()

    selected_id = st.session_state.get("km_selected_document")
    if selected_id:
        st.divider()
        render_document_detail(selected_id, key_prefix="library")

# === Upload (Upload -> Preview -> Extract -> Metadata -> Validate -> Publish) ===
with tab_upload:
    st.markdown(
        "Upload PDF, DOCX, PPTX, XLSX, TXT, LOG, JSON, CSV, or XML files -- or a ZIP of any of those "
        "(it fans out into one document per entry). Runs the same Evidence Ingestion Pipeline "
        "that parses investigation evidence; every document starts as **Draft** and isn't "
        "searchable until you publish it below."
    )
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "txt", "log", "json", "csv", "xml", "zip"],
        accept_multiple_files=True,
        key="km_uploader",
        label_visibility="collapsed",
    )
    if uploaded_files and st.button("Upload & Extract", type="primary"):
        files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        results = api_post("/admin/knowledge/documents/upload", files=files_payload)
        if results is not None:
            st.success(f"Created {len(results)} draft document(s) -- review them below before publishing.")
            st.session_state["km_just_uploaded"] = [r["document"]["id"] for r in results]
            st.rerun()

    just_uploaded = st.session_state.get("km_just_uploaded") or []
    if just_uploaded:
        st.divider()
        st.markdown("##### Just uploaded -- review before publishing")
        for document_id in just_uploaded:
            render_document_detail(document_id, key_prefix=f"upload_{document_id}")
            st.divider()
        if st.button("Clear this list", key="km_clear_uploaded"):
            st.session_state["km_just_uploaded"] = []
            st.rerun()
