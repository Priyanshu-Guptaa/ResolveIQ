"""Task Import -- decomposes a ServiceNow task export (JSON, XLSX, or
CSV) directly into individual Historical Investigation records, one
per row, instead of one searchable blob document. See
``app/engines/task_import/importer.py`` for why this is a separate
pipeline from Knowledge Management's document upload.

"Archive Original Source" is a deliberately separate, explicit action
(never automatic) -- the admin picks which already-uploaded blob
document to archive, independent of running an import here.

No role gate: see app/api/routers/admin/__init__.py.
"""

from __future__ import annotations

import streamlit as st
from api_client import API_BASE_URL, api_get, api_post, auth_headers, ensure_api_available
import requests

from theme import inject_theme

inject_theme()
st.title("📥 Task Import")
st.caption(
    "Administration · imports a ServiceNow task export (.json, .xlsx, or .csv) as individual Historical "
    "Investigation records -- each row becomes its own searchable, linkable record, not one giant blob "
    "document."
)
ensure_api_available()

tab_import, tab_archive = st.tabs(["📥 Import", "🗄️ Archive Original Source"])

# === Import ================================================================
with tab_import:
    st.markdown(
        "Upload a **.json**, **.xlsx**, or **.csv** export. Only rows with a captured resolution are "
        "imported -- everything else is skipped (nothing to teach the Recommendation Engine without one). "
        "Re-uploading the same file, or a file with overlapping tickets, is safe: already-imported tickets "
        "are detected by number and only updated if their content actually changed -- upload your full "
        "running case list each time and only genuinely new or changed tickets are added."
    )
    uploaded = st.file_uploader("Task export", type=["json", "xlsx", "xlsm", "csv"], key="task_import_uploader")
    actor = st.text_input("Imported by", value="admin", key="task_import_actor")

    if uploaded and st.button("Import", type="primary"):
        with st.spinner(f"Importing {uploaded.name}..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/admin/task-import",
                    params={"actor": actor},
                    files={"file": (uploaded.name, uploaded.getvalue())},
                    headers=auth_headers(),
                    timeout=300,
                )
                response.raise_for_status()
                summary = response.json()
            except requests.RequestException as exc:
                detail = ""
                if getattr(exc, "response", None) is not None:
                    detail = f" -- {exc.response.text}"
                st.error(f"Import failed: {exc}{detail}")
                summary = None

        if summary is not None:
            st.session_state["task_import_last_summary"] = summary

    summary = st.session_state.get("task_import_last_summary")
    if summary:
        st.divider()
        st.markdown(f"##### Import summary -- {summary['source_label']}")
        cols = st.columns(6)
        cols[0].metric("Total rows", summary["total_rows"])
        cols[1].metric("Imported", summary["imported"])
        cols[2].metric("Updated", summary["updated"])
        cols[3].metric("Duplicates", summary["duplicates"])
        cols[4].metric("Failed", summary["failed"])
        cols[5].metric("Relationships created", summary["relationships_created"])

        if summary["failed"]:
            with st.expander(f"⚠️ {summary['failed']} row(s) failed"):
                for err in summary["errors"]:
                    st.caption(err)

        if summary["imported"] or summary["updated"]:
            st.success(
                f"{summary['imported']} new and {summary['updated']} updated Historical Investigation "
                "record(s) are now indexed and searchable in Knowledge Center."
            )

# === Archive Original Source ================================================
with tab_archive:
    st.markdown(
        "If you separately uploaded the raw export as a Document through Knowledge Management, it stays "
        "published (and searchable as one large blob) until you explicitly archive it here -- this app "
        "never does that automatically."
    )
    documents = api_get("/admin/knowledge/documents", params={"page_size": 200, "sort_by": "updated_at"})
    candidates = [d for d in (documents or {}).get("items", []) if d["status"] != "archived"]

    if not candidates:
        st.caption("No non-archived documents found.")
    else:
        options = {f"{d['title']} ({d['status']}) -- {d['id'][:8]}": d["id"] for d in candidates}
        selected_label = st.selectbox("Document to archive", list(options.keys()), key="archive_source_pick")
        selected_id = options[selected_label]
        if st.button("🗄️ Archive Original Source", key="archive_source_button"):
            result = api_post(f"/admin/objects/document/{selected_id}/archive", {"actor": "admin"})
            if result is not None:
                st.success("Archived. It's removed from search but not deleted -- restorable from Knowledge Objects.")
                st.rerun()
