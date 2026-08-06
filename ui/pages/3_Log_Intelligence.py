"""Log Intelligence.

Not one of Sprint 2's 9 numbered phases -- built here only far enough to
avoid a dead nav link (RFC rev 3 "no fake navigation"). Shows real parsed
log evidence and extracted entities for a chosen investigation. The
dedicated explorer (thread reconstruction, cross-file correlation, error/
exception/warning views) is out of scope until a future sprint picks it
up explicitly.
"""

from __future__ import annotations

from api_client import api_available, api_get
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("📂 Log Intelligence")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

investigations = api_get("/investigations") or []
if not investigations:
    st.info("No investigations yet -- start one from the Dashboard.")
    st.stop()

options = {f"{inv['title']} ({inv['id'][:8]})": inv["id"] for inv in investigations}
selected = st.selectbox("Investigation", list(options.keys()))
investigation = api_get(f"/investigations/{options[selected]}")
if investigation is None:
    st.stop()

log_evidence = [e for e in investigation["evidence"] if e["evidence_type"] == "log_file"]
if not log_evidence:
    st.caption("No log files uploaded to this investigation yet.")
    st.stop()

for evidence in log_evidence:
    with st.expander(f"📋 {evidence['title']} -- {len(evidence['log_events'])} events", expanded=True):
        for event in evidence["log_events"]:
            level = event["level"]
            color = {"ERROR": "🔴", "WARN": "🟡", "FATAL": "🔴"}.get(level, "⚪")
            st.markdown(f"{color} `{event.get('timestamp') or '--'}` **{level}** {event['message'][:200]}")
        if evidence["extracted_entities"]:
            st.caption(
                "Entities: "
                + ", ".join(f"{e['entity_type']}={e['value']}" for e in evidence["extracted_entities"])
            )
