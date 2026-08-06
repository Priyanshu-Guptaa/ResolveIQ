"""ResolveIQ Sprint 1 UI: a thin Streamlit client over the FastAPI backend.

This is deliberately NOT where any investigation logic lives -- it only
calls the API and renders what comes back, so the UI can be swapped for a
richer frontend later without touching any engine code.

Run with:

    streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("RESOLVEIQ_API_URL", "http://localhost:8000")

st.set_page_config(page_title="ResolveIQ", page_icon="🔎", layout="wide")


# --- API client helpers ------------------------------------------------------


def api_get(path: str) -> dict | list | None:
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"API request failed: GET {path} -- {exc}")
        return None


def api_post(path: str, json_body: dict | None = None, files=None) -> dict | list | None:
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=json_body, files=files, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            detail = f" -- {exc.response.text}"
        st.error(f"API request failed: POST {path} -- {exc}{detail}")
        return None


# --- Sidebar: investigation selection / creation -----------------------------


def render_sidebar() -> None:
    st.sidebar.title("🔎 ResolveIQ")
    st.sidebar.caption("Investigation Intelligence Platform")

    st.sidebar.subheader("Start a new investigation")
    with st.sidebar.form("new_investigation_form", clear_on_submit=True):
        title = st.text_input("Title", placeholder="e.g. ServiceNow task INC0012345")
        description = st.text_area(
            "Paste the task description",
            placeholder="Paste the ServiceNow task / issue description here...",
            height=150,
        )
        submitted = st.form_submit_button("Start Investigation", type="primary")
        if submitted and title.strip():
            result = api_post("/investigations", {"title": title, "description": description})
            if result:
                st.session_state["investigation_id"] = result["id"]
                st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Or resume an existing investigation")
    investigations = api_get("/investigations") or []
    if investigations:
        options = {f"{inv['title']} ({inv['evidence_count']} evidence)": inv["id"] for inv in investigations}
        selected_label = st.sidebar.selectbox("Investigations", list(options.keys()), index=None)
        if selected_label:
            st.session_state["investigation_id"] = options[selected_label]
    else:
        st.sidebar.caption("No investigations yet.")


# --- Main panel ---------------------------------------------------------------


def render_evidence_section(investigation_id: str) -> None:
    st.subheader("2. Add evidence")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Upload log files", accept_multiple_files=True, key="log_uploader"
        )
        if uploaded_files and st.button("Upload logs", key="upload_logs_btn"):
            files_payload = [
                ("files", (f.name, f.getvalue(), "text/plain")) for f in uploaded_files
            ]
            result = api_post(f"/investigations/{investigation_id}/evidence/logs", files=files_payload)
            if result is not None:
                st.success(f"Uploaded {len(result)} log file(s) and extracted entities.")
                st.rerun()

    with col2:
        note = st.text_area("Add a manual note", placeholder="Additional context, findings so far, ...")
        if st.button("Add note", key="add_note_btn") and note.strip():
            result = api_post(f"/investigations/{investigation_id}/evidence/notes", {"text": note})
            if result is not None:
                st.success("Note added.")
                st.rerun()


def render_investigation_state(investigation: dict) -> None:
    st.subheader("Investigation state")
    st.write(f"**Status:** {investigation['status']}  |  **Evidence items:** {len(investigation['evidence'])}")

    entity_counts: dict[str, int] = {}
    for evidence in investigation["evidence"]:
        for entity in evidence.get("extracted_entities", []):
            entity_counts[entity["entity_type"]] = entity_counts.get(entity["entity_type"], 0) + 1

    with st.expander(f"Evidence ({len(investigation['evidence'])})", expanded=False):
        for evidence in investigation["evidence"]:
            st.markdown(f"**{evidence['title']}** &nbsp;·&nbsp; `{evidence['evidence_type']}` &nbsp;·&nbsp; source: {evidence['source']}")
            st.text(evidence["raw_content"][:1000])
            if evidence.get("extracted_entities"):
                st.caption(
                    "Entities: "
                    + ", ".join(f"{e['entity_type']}={e['value']}" for e in evidence["extracted_entities"][:15])
                )
            st.divider()

    if entity_counts:
        st.caption("Entity summary across all evidence:")
        st.write(entity_counts)


def render_recommendations(investigation_id: str) -> None:
    st.subheader("3. Recommendation")

    if st.button("🔍 Analyze / Refresh recommendation", type="primary"):
        st.session_state["recommendation"] = api_get(f"/investigations/{investigation_id}/recommendations")

    recommendation = st.session_state.get("recommendation")
    if not recommendation:
        st.info("Click **Analyze** to generate a recommendation from the evidence gathered so far.")
        return

    st.markdown(f"### ➡️ Next best step\n{recommendation['next_best_step']}")
    st.progress(recommendation["overall_confidence"], text=f"Overall confidence: {recommendation['overall_confidence']:.0%}")

    if recommendation["root_causes"]:
        st.markdown("#### Likely root causes")
        for rc in recommendation["root_causes"]:
            st.markdown(f"- **{rc['description']}** &nbsp; _(confidence {rc['confidence']:.0%})_")
            st.caption(rc["rationale"])

    tabs = st.tabs(["Similar investigations", "Documentation", "Known bugs", "Suggested next actions"])

    with tabs[0]:
        _render_matches(recommendation["similar_investigations"])
    with tabs[1]:
        _render_matches(recommendation["relevant_documentation"])
    with tabs[2]:
        _render_matches(recommendation["known_bugs"])
    with tabs[3]:
        if recommendation["suggested_logs"]:
            st.markdown("**Suggested logs to pull:**")
            for item in recommendation["suggested_logs"]:
                st.markdown(f"- {item}")
        if recommendation["suggested_sql"]:
            st.markdown("**Suggested SQL:**")
            for query in recommendation["suggested_sql"]:
                st.code(query, language="sql")
        if not recommendation["suggested_logs"] and not recommendation["suggested_sql"]:
            st.caption("No specific log/SQL suggestions for this evidence yet.")


def _render_matches(matches: list[dict]) -> None:
    if not matches:
        st.caption("No matches found.")
        return
    for match in matches:
        st.markdown(f"**{match['title']}** &nbsp; _(similarity {match['score']:.0%})_")
        st.caption(match["snippet"])
        st.divider()


# --- Page ------------------------------------------------------------------


def main() -> None:
    render_sidebar()

    st.title("ResolveIQ")
    st.caption("What should I do next?")

    investigation_id = st.session_state.get("investigation_id")
    if not investigation_id:
        st.info("👈 Start a new investigation or select an existing one from the sidebar.")
        return

    investigation = api_get(f"/investigations/{investigation_id}")
    if investigation is None:
        return

    st.header(f"1. {investigation['title']}")
    render_investigation_state(investigation)
    st.divider()
    render_evidence_section(investigation_id)
    st.divider()
    render_recommendations(investigation_id)


if __name__ == "__main__":
    main()
