"""Investigation Strategy (Recommendation Engine V2) -- the primary
recommendation experience: one guided, explainable hierarchy instead of
independent panels. Renders exactly the sections
``InvestigationStrategy`` carries, in the approved order: stage,
progress, next action, decision checkpoint, required/missing evidence,
ordered log collection, suggested SQL, related Product Intelligence,
historical investigations, known bugs, documentation.

Reuses existing renderers where the underlying data is itself reused
(not duplicated) from another module -- ``render_recommended_log_collection``
for the ordered log collection section is the same component the
Recommendations tab used before this phase, now called with the
Strategy's ``ordered_log_collection`` (the exact same list as the
legacy ``recommended_logs`` field, per the Recommendation Engine's own
"reuse, don't duplicate" design).
"""

from __future__ import annotations

import streamlit as st

from components.recommended_log_collection import render_recommended_log_collection

_STAGE_DISPLAY = {
    "triage": ("🆕", "Triage"),
    "evidence_collection": ("📥", "Evidence Collection"),
    "analysis": ("🔎", "Analysis"),
    "root_cause_identified": ("✅", "Root Cause Identified"),
}


def render_investigation_strategy(strategy: dict, *, investigation_id: str) -> str | None:
    """Returns the linked component name the engineer clicked through to
    Product Intelligence for, if any -- same cross-navigation contract
    ``render_recommended_log_collection`` already used; the caller
    applies it via the existing pending-key + rerun pattern."""
    icon, label = _STAGE_DISPLAY.get(strategy["current_stage"], ("❓", strategy["current_stage"]))
    st.markdown(f"### {icon} {label}")
    st.caption(strategy["stage_rationale"])
    st.progress(strategy["progress"], text=strategy["progress_summary"])

    st.markdown("#### ➡️ Recommended next action")
    st.markdown(strategy["recommended_next_action"])
    st.caption(strategy["next_action_rationale"])

    if strategy["decision_checkpoint"]:
        st.info(f"🧭 **Decision checkpoint:** {strategy['decision_checkpoint']}")

    navigate_to_component: str | None = None

    required = strategy["required_evidence"]
    missing = strategy["missing_evidence"]
    if required:
        collected = len(required) - len(missing)
        with st.expander(f"📋 Required evidence ({collected}/{len(required)} collected)", expanded=bool(missing)):
            for item in required:
                mark = "✅" if item["satisfied"] else "⬜"
                st.markdown(f"{mark} {item['description']}")

    nav = render_recommended_log_collection(strategy["ordered_log_collection"], investigation_id=investigation_id)
    if nav:
        navigate_to_component = nav

    if strategy["suggested_sql"]:
        with st.expander("🗄 Suggested SQL", expanded=False):
            for item in strategy["suggested_sql"]:
                title = item["title"]
                if item["source"] == "sql_library":
                    title += " _(SQL Library)_"
                st.markdown(f"**{title}**")
                if item["explanation"]:
                    st.caption(item["explanation"])
                st.code(item["sql_text"], language="sql")

    matched_component = strategy["matched_component"]
    if matched_component:
        header = f"🧩 Related Product Intelligence -- {matched_component['component_name']} ({matched_component['confidence']:.0%})"
        with st.expander(header, expanded=False):
            st.caption(matched_component["match_reason"])
            if st.button(
                f"View {matched_component['component_name']} in Product Intelligence →",
                key=f"strategy_pi_link_{investigation_id}",
            ):
                navigate_to_component = matched_component["component_name"]

    historical = strategy["historical_investigations"]
    if historical:
        with st.expander(f"📊 Historical investigations ({len(historical)})", expanded=False):
            for match in historical[:5]:
                st.markdown(f"**{match['title'][:60]}** _({match['score']:.0%})_")

    known_bugs = strategy["known_bugs"]
    if known_bugs:
        with st.expander(f"🐞 Known bugs ({len(known_bugs)})", expanded=False):
            for bug in known_bugs[:5]:
                st.markdown(f"**{bug['title'][:60]}**")

    documentation = strategy["documentation"]
    if documentation:
        with st.expander(f"📚 Documentation ({len(documentation)})", expanded=False):
            for doc in documentation[:5]:
                st.markdown(f"**{doc['title'][:60]}**")

    return navigate_to_component
