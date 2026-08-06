"""Investigation Workspace.

Phase 1 scope: the full Sprint 1 working flow (select/create an
investigation, add evidence, get a recommendation), carried over from
streamlit_app.py, restructured for the multi-page shell. This is NOT yet
the three-pane console -- that is Phase 2's job. Every control here is
real and backend-wired; nothing is a placeholder for the Phase 2 layout.
"""

from __future__ import annotations

from api_client import api_available, api_get, api_post
from components.status_badge import status_badge_html
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("🔍 Investigation Workspace")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

# --- Investigation selection ---------------------------------------------

with st.sidebar:
    st.subheader("Start a new investigation")
    with st.form("new_investigation_form", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. ServiceNow task INC0012345")
        description = st.text_area("Paste the task description", height=150)
        if st.form_submit_button("Start Investigation", type="primary") and title.strip():
            result = api_post("/investigations", {"title": title, "description": description})
            if result:
                st.session_state["investigation_id"] = result["id"]
                st.rerun()

    st.divider()
    st.subheader("Or resume an existing investigation")
    investigations = api_get("/investigations") or []
    if investigations:
        options = {f"{inv['title']} ({inv['evidence_count']} evidence)": inv["id"] for inv in investigations}
        selected_label = st.selectbox("Investigations", list(options.keys()), index=None)
        if selected_label:
            st.session_state["investigation_id"] = options[selected_label]
    else:
        st.caption("No investigations yet.")

investigation_id = st.session_state.get("investigation_id")
if not investigation_id:
    st.info("👈 Start a new investigation or select an existing one from the sidebar.")
    st.stop()

investigation = api_get(f"/investigations/{investigation_id}")
if investigation is None:
    st.stop()

st.header(investigation["title"])
st.markdown(status_badge_html(investigation["status"]), unsafe_allow_html=True)
st.caption(f"{len(investigation['evidence'])} evidence item(s)")

tab_evidence, tab_recommendations = st.tabs(["Evidence", "Recommendations"])

with tab_evidence:
    entity_counts: dict[str, int] = {}
    for evidence in investigation["evidence"]:
        for entity in evidence.get("extracted_entities", []):
            entity_counts[entity["entity_type"]] = entity_counts.get(entity["entity_type"], 0) + 1

    with st.expander(f"Evidence ({len(investigation['evidence'])})", expanded=True):
        for evidence in investigation["evidence"]:
            st.markdown(
                f"**{evidence['title']}** · `{evidence['evidence_type']}` · source: {evidence['source']}"
            )
            st.text(evidence["raw_content"][:1000])
            if evidence.get("extracted_entities"):
                st.caption(
                    "Entities: "
                    + ", ".join(f"{e['entity_type']}={e['value']}" for e in evidence["extracted_entities"][:15])
                )
            st.divider()

    if entity_counts:
        st.caption("Entity summary across all evidence:")
        st.json(entity_counts, expanded=False)

    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader("Upload log files", accept_multiple_files=True, key="log_uploader")
        if uploaded_files and st.button("Upload logs"):
            files_payload = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files]
            result = api_post(f"/investigations/{investigation_id}/evidence/logs", files=files_payload)
            if result is not None:
                st.success(f"Uploaded {len(result)} log file(s) and extracted entities.")
                st.rerun()
    with col2:
        note = st.text_area("Add a manual note")
        if st.button("Add note") and note.strip():
            result = api_post(f"/investigations/{investigation_id}/evidence/notes", {"text": note})
            if result is not None:
                st.success("Note added.")
                st.rerun()

with tab_recommendations:
    if st.button("🔍 Analyze / Refresh recommendation", type="primary"):
        st.session_state["recommendation"] = api_get(f"/investigations/{investigation_id}/recommendations")

    recommendation = st.session_state.get("recommendation")
    if not recommendation:
        st.info("Click Analyze to generate a recommendation from the evidence gathered so far.")
    else:
        st.markdown(f"### ➡️ Next best step\n{recommendation['next_best_step']}")
        st.progress(
            recommendation["overall_confidence"],
            text=f"Overall confidence: {recommendation['overall_confidence']:.0%}",
        )
        if recommendation["root_causes"]:
            st.markdown("#### Likely root causes")
            for rc in recommendation["root_causes"]:
                st.markdown(f"- **{rc['description']}** _(confidence {rc['confidence']:.0%})_")
                st.caption(rc["rationale"])

        result_tabs = st.tabs(["Similar investigations", "Documentation", "Known bugs", "Suggested actions"])
        with result_tabs[0]:
            for m in recommendation["similar_investigations"] or [None]:
                if m is None:
                    st.caption("No matches.")
                    break
                st.markdown(f"**{m['title']}** _({m['score']:.0%})_")
                st.caption(m["snippet"])
        with result_tabs[1]:
            for m in recommendation["relevant_documentation"] or [None]:
                if m is None:
                    st.caption("No matches.")
                    break
                st.markdown(f"**{m['title']}** _({m['score']:.0%})_")
                st.caption(m["snippet"])
        with result_tabs[2]:
            for m in recommendation["known_bugs"] or [None]:
                if m is None:
                    st.caption("No matches.")
                    break
                st.markdown(f"**{m['title']}** _({m['score']:.0%})_")
                st.caption(m["snippet"])
        with result_tabs[3]:
            for item in recommendation["suggested_logs"]:
                st.markdown(f"- {item}")
            for sql in recommendation["suggested_sql"]:
                st.code(sql, language="sql")
