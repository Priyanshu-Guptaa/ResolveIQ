"""Investigation Workspace -- the three-pane console (RFC rev 3, Phase 2A).

Layout:
  Persistent Summary Card (top, always visible)
  LEFT   Evidence explorer: Evidence / Logs / Documents / Screenshots / Timeline
  CENTER Tabs: Overview / Findings / Recommendations / Playbook / Notes
  RIGHT  Collapsible assist rail: AI Assistant / Historical Matches /
         Product Intelligence / Known Bugs / Suggested SQL / Related Docs

Scope discipline for this phase: every panel here displays EXISTING
computed data (Log Intelligence's merged_entities, the Recommendation
Engine's existing output) -- no new recommendation logic, per "do not
expand Recommendation Engine logic yet." Product Intelligence's right-rail
section (Phase 2B, incremental) renders real Component Registry data via
a plain manual picker -- no auto-matching of evidence to a component,
that's Recommendation Engine V2's job.

Design note on the left/center relationship: the RFC's wireframe shows
the left explorer and center tabs as loosely coupled (clicking an
explorer item can imply a center view). This implementation keeps them
independent for clarity -- the left panel filters and displays its own
evidence list in place; the center tabs are a separate, always-available
structure. Documented here rather than silently deviating.
"""

from __future__ import annotations

import streamlit as st
from api_client import (
    api_get,
    api_patch,
    api_post,
    ensure_api_available,
    get_component_profiles_cached,
    list_investigations_cached,
)
from components.component_profile import render_component_profile
from components.investigation_picker import render_investigation_picker
from components.summary_card import render_details_editor, render_summary_card
from context import get_active_investigation_id, set_active_investigation_id
from theme import inject_theme

inject_theme()
st.title("🔍 Investigation Workspace")
ensure_api_available()

# --- Investigation selection --------------------------------------------

with st.sidebar:
    st.subheader("Start a new investigation")
    with st.form("new_investigation_form", clear_on_submit=True):
        new_title = st.text_input("Title", placeholder="e.g. ServiceNow task INC0012345")
        new_description = st.text_area("Paste the task description", height=150)
        if st.form_submit_button("Start Investigation", type="primary") and new_title.strip():
            created = api_post("/investigations", {"title": new_title, "description": new_description})
            if created:
                list_investigations_cached.clear()
                set_active_investigation_id(created["id"])
                st.rerun()

    st.divider()
    st.subheader("Or resume an existing investigation")
    render_investigation_picker(key="workspace_picker")

investigation_id = get_active_investigation_id()
if not investigation_id:
    st.info("👈 Start a new investigation or select an existing one from the sidebar.")
    st.stop()

investigation = api_get(f"/investigations/{investigation_id}")
if investigation is None:
    st.stop()

# Recommendation is fetched once (on demand, via the button in the
# Recommendations tab) and reused by every panel that needs it -- the
# Findings/right-rail panels never trigger their own recommendation call.
if st.session_state.get("workspace_recommendation_for") != investigation_id:
    st.session_state["workspace_recommendation"] = None
    st.session_state["workspace_recommendation_for"] = investigation_id
recommendation = st.session_state.get("workspace_recommendation")

# --- Persistent Summary Card ---------------------------------------------

render_summary_card(investigation, recommendation)

# --- Three panes -----------------------------------------------------------

left, center, right = st.columns([1.1, 2.6, 1.6], gap="medium")

# === LEFT: Evidence explorer ===============================================
with left:
    st.markdown('<div class="riq-panel-title">Explorer</div>', unsafe_allow_html=True)

    nav_options = ["Evidence", "Logs", "Documents", "Screenshots", "Timeline"]
    nav_key = f"left_nav_{investigation_id}"
    if nav_key not in st.session_state:
        st.session_state[nav_key] = "Evidence"
    for option in nav_options:
        is_active = st.session_state[nav_key] == option
        if st.button(option, key=f"nav_{option}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state[nav_key] = option
            st.rerun()

    st.divider()
    selected_nav = st.session_state[nav_key]
    evidence_list = investigation["evidence"]

    def _file_kind(ev: dict) -> str:
        return ev.get("metadata", {}).get("file_kind", "text" if ev["evidence_type"] != "manual_note" else "note")

    if selected_nav == "Evidence":
        shown = evidence_list
    elif selected_nav == "Logs":
        shown = [e for e in evidence_list if _file_kind(e) == "text" and e["evidence_type"] == "log_file"]
    elif selected_nav == "Documents":
        shown = [e for e in evidence_list if _file_kind(e) in ("docx", "xlsx", "pdf")]
    elif selected_nav == "Screenshots":
        shown = [e for e in evidence_list if _file_kind(e) == "image"]
    else:  # Timeline
        shown = None

    if selected_nav == "Timeline":
        timeline = api_get(f"/investigations/{investigation_id}/timeline") or []
        for item in timeline:
            st.caption(f"`{item['occurred_at'][:16].replace('T', ' ')}`  {item['label'][:60]}")
    elif not shown:
        st.caption(f"No {selected_nav.lower()} yet.")
    else:
        for ev in shown:
            st.markdown(f"**{ev['title'][:28]}**")
            st.caption(f"{_file_kind(ev)} · {ev['source']}")

    st.divider()
    st.markdown('<div class="riq-panel-title">Add evidence</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload (log, txt, csv, json, xml, docx, xlsx, pdf, jpg, png, evtx, zip)",
        accept_multiple_files=True,
        key="workspace_uploader",
        label_visibility="collapsed",
    )
    if uploaded_files and st.button("Upload", key="workspace_upload_btn", use_container_width=True):
        files_payload = [("files", (f.name, f.getvalue())) for f in uploaded_files]
        result = api_post(f"/investigations/{investigation_id}/evidence/logs", files=files_payload)
        if result is not None:
            st.success(f"Parsed {len(result)} item(s).")
            st.rerun()

# === CENTER: tabs ============================================================
with center:
    tab_overview, tab_findings, tab_recs, tab_playbook, tab_notes = st.tabs(
        ["Overview", "Findings", "Recommendations", "Playbook", "Notes"]
    )

    with tab_overview:
        st.markdown(f"**{investigation['title']}**")
        st.caption(f"{len(investigation['evidence'])} evidence item(s) · created {investigation['created_at'][:16].replace('T', ' ')}")
        task_desc = next((e for e in investigation["evidence"] if e["evidence_type"] == "task_description"), None)
        if task_desc:
            st.text(task_desc["raw_content"][:1500])
        st.divider()
        st.markdown("**Case details**")

        def _save_details(fields: dict) -> None:
            updated = api_patch(f"/investigations/{investigation_id}/details", fields)
            if updated is not None:
                st.success("Saved.")
                st.rerun()

        render_details_editor(investigation, on_save=_save_details)

    with tab_findings:
        entity_counts: dict[str, int] = {}
        all_entities: list[dict] = []
        for ev in investigation["evidence"]:
            for entity in ev.get("extracted_entities", []):
                entity_counts[entity["entity_type"]] = entity_counts.get(entity["entity_type"], 0) + 1
                all_entities.append(entity)
        if not entity_counts:
            st.caption("No entities extracted yet -- upload evidence to populate this.")
        else:
            st.caption("Extracted entities across all evidence, most-frequent first:")
            for entity_type, count in sorted(entity_counts.items(), key=lambda kv: -kv[1]):
                values = sorted({e["value"] for e in all_entities if e["entity_type"] == entity_type})
                st.markdown(f"**{entity_type}** ({count})")
                st.caption(", ".join(values[:10]))

    with tab_recs:
        if st.button("🔍 Analyze / Refresh recommendation", type="primary", key="analyze_btn"):
            st.session_state["workspace_recommendation"] = api_get(f"/investigations/{investigation_id}/recommendations")
            st.rerun()

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

    with tab_playbook:
        st.caption(
            "Derived from the current recommendation's suggested logs/SQL -- a dedicated "
            "Playbook Engine with persisted checklist state is future work. Checkbox state "
            "here is session-only and resets on reload."
        )
        if not recommendation:
            st.info("Run Analyze in the Recommendations tab first.")
        else:
            steps = list(recommendation.get("suggested_logs", []))
            steps += [f"Run: {sql.splitlines()[0][:60]}" for sql in recommendation.get("suggested_sql", [])]
            if not steps:
                st.caption("No suggested steps for this evidence yet.")
            for i, step in enumerate(steps):
                st.checkbox(step, key=f"playbook_{investigation_id}_{i}")

    with tab_notes:
        notes = [e for e in investigation["evidence"] if e["evidence_type"] == "manual_note"]
        st.caption(
            "General notes. Categorized types (Customer Update, L2 Notes, L3 Escalation, "
            "RCA) arrive with Documentation Generators in a later phase."
        )
        if not notes:
            st.caption("No notes yet.")
        for note in notes:
            st.markdown(f"`{note['created_at'][:16].replace('T', ' ')}`")
            st.text(note["raw_content"][:800])
            st.divider()

        # A plain text_area + button (no form) keeps its typed value in
        # session_state across the st.rerun() below -- after successfully
        # adding a note, the box would still show the old text, reading
        # as "did that actually save?" clear_on_submit=True is exactly
        # for this: the widget resets once the form submits successfully.
        with st.form(f"add_note_form_{investigation_id}", clear_on_submit=True):
            new_note = st.text_area("Add a note")
            submitted = st.form_submit_button("Add note")
        if submitted and new_note.strip():
            result = api_post(f"/investigations/{investigation_id}/evidence/notes", {"text": new_note})
            if result is not None:
                st.success("Note added.")
                st.rerun()

# === RIGHT: collapsible assist rail =========================================
with right:
    st.markdown('<div class="riq-panel-title">Assist</div>', unsafe_allow_html=True)

    with st.expander("🤖 AI Assistant", expanded=False):
        if recommendation:
            st.caption(recommendation["next_best_step"][:200])
        else:
            st.caption("Run Analyze to populate this.")
        if st.button("Open full AI Assistant →", key="open_ai_assistant"):
            st.switch_page("views/7_AI_Assistant.py")

    with st.expander("📊 Historical Matches", expanded=True):
        if not recommendation or not recommendation["similar_investigations"]:
            st.caption("Run Analyze to see similar past investigations.")
        else:
            for m in recommendation["similar_investigations"][:5]:
                st.markdown(f"**{m['title'][:40]}** _({m['score']:.0%})_")
            st.caption("Detailed match reasoning (component/firmware/version) arrives with Recommendation Engine V2 -- Phase 2C.")

    with st.expander("🧩 Product Intelligence", expanded=False):
        components = get_component_profiles_cached() or []
        if not components:
            st.caption("No component profiles loaded.")
        else:
            names = [c["name"] for c in components]
            component_key = f"pi_component_{investigation_id}"
            # A related-component click sets this *pending* key instead of
            # component_key directly -- Streamlit forbids writing to a
            # widget's session_state key after that widget has already
            # been instantiated in the same run, and by the time we know
            # what was clicked (inside render_component_profile, below)
            # the selectbox has already been created. Applying the
            # pending value here, before the selectbox exists, and then
            # clearing it, is the standard workaround.
            pending_key = f"pi_component_pending_{investigation_id}"
            if pending_key in st.session_state:
                st.session_state[component_key] = st.session_state.pop(pending_key)
            selected_name = st.selectbox("Component", names, key=component_key)
            selected_profile = next(c for c in components if c["name"] == selected_name)
            navigate_to = render_component_profile(selected_profile, known_component_names=set(names))
            if navigate_to and navigate_to != selected_name:
                st.session_state[pending_key] = navigate_to
                st.rerun()
            st.caption("Automatic evidence-to-component matching arrives with Recommendation Engine V2 -- Phase 2C.")

    with st.expander("🐞 Known Bugs", expanded=False):
        if not recommendation or not recommendation["known_bugs"]:
            st.caption("Run Analyze to check for known bugs.")
        else:
            for b in recommendation["known_bugs"][:5]:
                st.markdown(f"**{b['title'][:40]}**")

    with st.expander("🗄 Suggested SQL", expanded=False):
        if not recommendation or not recommendation["suggested_sql"]:
            st.caption("Run Analyze to see suggested queries.")
        else:
            for sql in recommendation["suggested_sql"]:
                st.code(sql, language="sql")

    with st.expander("📚 Related Documentation", expanded=False):
        if not recommendation or not recommendation["relevant_documentation"]:
            st.caption("Run Analyze to see related documentation.")
        else:
            for d in recommendation["relevant_documentation"][:5]:
                st.markdown(f"**{d['title'][:40]}**")
