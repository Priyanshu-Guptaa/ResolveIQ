"""Log Wiki Import -- turns an exported internal wiki page (log
repositories, message-flow workflows) into governed Log Intelligence
Knowledge Base records (``LogSourceApplication`` / ``LogCollectionScenario``).

Deliberately mirrors Task Import's page shape (upload -> summary), for
the same reason: one consistent "run an import, see what happened"
pattern across every bulk-import feature in the Administration Portal.

Generic Knowledge Object CRUD for the resulting records already exists
in Knowledge Objects (Phase 3.4) -- nothing type-specific is
duplicated on this page.

No role gate: see app/api/routers/admin/__init__.py.
"""

from __future__ import annotations

import requests
import streamlit as st
from api_client import API_BASE_URL, auth_headers, ensure_api_available

from theme import inject_theme

inject_theme()
st.title("📚 Log Wiki Import")
st.caption(
    "Administration · imports an exported internal wiki page (.pdf/.html/.txt) describing log "
    "repositories and message-flow workflows into the Log Intelligence Knowledge Base -- what powers "
    "the Investigation Workspace's Recommended Log Collection guidance."
)
ensure_api_available()

st.markdown(
    "Upload an exported wiki page (e.g. a Confluence \"Logs Repository\" page saved as PDF). Every "
    "application and message-flow scenario it describes becomes a governed record -- log repository "
    "location, technology, and collection order are extracted deterministically (no AI reasoning); "
    "fields the page doesn't mention are left empty rather than guessed. Re-uploading the same page, or "
    "a later page that describes the same applications, **enriches** those existing records instead of "
    "duplicating them."
)

uploaded = st.file_uploader("Wiki page export", type=["pdf", "html", "htm", "txt"], key="log_wiki_uploader")
col1, col2 = st.columns(2)
product = col1.text_input("Product", value="Command Center", key="log_wiki_product")
actor = col2.text_input("Imported by", value="admin", key="log_wiki_actor")

if uploaded and st.button("Import", type="primary"):
    with st.spinner(f"Extracting and importing {uploaded.name}..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/admin/log-knowledge/import",
                params={"product": product, "actor": actor},
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
        st.session_state["log_wiki_last_summary"] = summary

summary = st.session_state.get("log_wiki_last_summary")
if summary:
    st.divider()
    st.markdown(f"##### Import summary -- {summary['source_wiki_page']}")
    cols = st.columns(5)
    cols[0].metric("Log sources created", summary["sources_created"])
    cols[1].metric("Log sources enriched", summary["sources_enriched"])
    cols[2].metric("Scenarios created", summary["scenarios_created"])
    cols[3].metric("Scenarios updated", summary["scenarios_updated"])
    cols[4].metric("Component links created", summary["component_relationships_created"])

    if summary["errors"]:
        with st.expander(f"⚠️ {len(summary['errors'])} record(s) failed"):
            for err in summary["errors"]:
                st.caption(err)

    if summary["sources_created"] or summary["sources_enriched"]:
        st.success(
            "Knowledge base updated. Browse the resulting records in Knowledge Objects "
            "(Log Source Application / Log Collection Scenario), or open an investigation to see "
            "matching scenarios surfaced as Recommended Log Collection guidance."
        )
