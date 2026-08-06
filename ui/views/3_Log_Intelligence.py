"""Log Intelligence.

Not one of Sprint 2's 9 numbered phases -- built here only far enough to
avoid a dead nav link. Shows real parsed log/document evidence and
extracted entities for the shared active investigation. The dedicated
explorer (thread reconstruction, cross-file correlation, error/exception/
warning views) is out of scope until a future sprint picks it up.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, ensure_api_available
from components.investigation_picker import render_investigation_picker
from theme import inject_theme

inject_theme()
st.title("📂 Log Intelligence")
ensure_api_available()

investigation_id = render_investigation_picker(key="log_intel_picker")
if not investigation_id:
    st.stop()

investigation = api_get(f"/investigations/{investigation_id}")
if investigation is None:
    st.stop()

file_evidence = [e for e in investigation["evidence"] if e["evidence_type"] == "log_file"]
if not file_evidence:
    st.caption("No files uploaded to this investigation yet.")
    st.stop()

for evidence in file_evidence:
    file_kind = evidence.get("metadata", {}).get("file_kind", "text")
    with st.expander(f"📋 {evidence['title']} · {file_kind} · {len(evidence['log_events'])} events", expanded=True):
        for warning in evidence.get("metadata", {}).get("warnings") or []:
            icon = "⚠️" if warning["severity"] == "warning" else "🛑"
            st.caption(f"{icon} {warning['message']}")
        if evidence["log_events"]:
            for event in evidence["log_events"]:
                level = event["level"]
                color = {"ERROR": "🔴", "WARN": "🟡", "FATAL": "🔴"}.get(level, "⚪")
                st.markdown(f"{color} `{event.get('timestamp') or '--'}` **{level}** {event['message'][:200]}")
        else:
            st.text(evidence["raw_content"][:2000])
        if evidence["extracted_entities"]:
            st.caption(
                "Entities: "
                + ", ".join(f"{e['entity_type']}={e['value']}" for e in evidence["extracted_entities"])
            )
