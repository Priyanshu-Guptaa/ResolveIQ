"""Log Intelligence.

Not one of Sprint 2's 9 numbered phases -- built here only far enough to
avoid a dead nav link. Shows real parsed log/document evidence and
extracted entities for the shared active investigation. The dedicated
explorer (thread reconstruction, cross-file correlation, error/exception/
warning views) is out of scope until a future sprint picks it up.

Investigation loading redesign: this page used to render every log-file
evidence item's full parsed content (log_events, extracted_entities,
raw_content) in an already-expanded expander, all at once -- the same
"opening an investigation loads every uploaded file" problem the
Workspace had, just here instead. Now: the file list is metadata-only
(from GET .../evidence), and full content is fetched for exactly one
file at a time -- whichever the engineer picks -- via
GET .../evidence/{evidence_id}.
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

evidence_list = api_get(f"/investigations/{investigation_id}/evidence") or []
file_summaries = [e for e in evidence_list if e["evidence_type"] == "log_file"]
if not file_summaries:
    st.caption("No files uploaded to this investigation yet.")
    st.stop()


def _option_label(ev: dict) -> str:
    kind = ev.get("file_kind") or "text"
    return f"{ev['title']} · {kind} · {ev['log_event_count']} events"


options = {_option_label(ev): ev["id"] for ev in file_summaries}
selected_label = st.selectbox(f"Files ({len(file_summaries)})", list(options.keys()), key="log_intel_file_pick")
selected_id = options[selected_label]

evidence = api_get(f"/investigations/{investigation_id}/evidence/{selected_id}")
if evidence is None:
    st.stop()

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
            "Entities: " + ", ".join(f"{e['entity_type']}={e['value']}" for e in evidence["extracted_entities"])
        )
