"""Investigation Strategy (Recommendation Engine V2) -- the primary
recommendation experience: one guided, explainable hierarchy instead of
independent panels. Render order (revised for External Knowledge):
stage, progress, next action, required/missing evidence, ordered log
collection, local KB matches (historical investigations/known bugs/
documentation), external knowledge (TFS/Wiki, live), suggested SQL,
related Product Intelligence, decision checkpoint. Decision checkpoint
moved from just-after-next-action to last -- everything else is
additive, nothing removed.

Reuses existing renderers where the underlying data is itself reused
(not duplicated) from another module -- ``render_recommended_log_collection``
for the ordered log collection section is the same component the
Recommendations tab used before this phase, now called with the
Strategy's ``ordered_log_collection`` (the exact same list as the
legacy ``recommended_logs`` field, per the Recommendation Engine's own
"reuse, don't duplicate" design). ``render_external_knowledge`` is the
same discipline applied to the live TFS/Wiki results.
"""

from __future__ import annotations

import streamlit as st

from components.external_knowledge import render_external_knowledge
from components.historical_match import render_historical_match
from components.recommended_log_collection import render_recommended_log_collection
from formatting import truncate_words

_STAGE_DISPLAY = {
    "triage": ("🆕", "Triage"),
    "evidence_collection": ("📥", "Evidence Collection"),
    "analysis": ("🔎", "Analysis"),
    "root_cause_identified": ("✅", "Root Cause Identified"),
}


def _render_recommended_solution(solution: dict | None) -> None:
    """The synthesized, cross-source answer -- correlates local KB +
    live TFS + live Wiki (RecommendationEngine._synthesize_recommendation)
    into one recommendation, rendered first/most prominently since
    it's meant to answer "what's likely happening and what do I do"
    directly. Every source that contributed is tagged explicitly
    (never a blended, unattributed claim), and an honest
    "additional investigation is required" replaces a resolution when
    the engine didn't find enough real evidence -- never a fabricated
    fix."""
    if solution is None:
        return

    st.markdown("#### 💡 Recommended Solution")
    st.markdown(f"**Likely issue:** {solution['likely_issue']}")
    st.caption(f"Why: {solution['rationale']}")

    source_tags = []
    if solution["source_local"]:
        source_tags.append("📚 Local KB")
    if solution["source_tfs"]:
        source_tags.append("🔧 TFS")
    if solution["source_wiki"]:
        source_tags.append("🌐 Wiki")
    if source_tags:
        st.caption("Sources: " + " · ".join(source_tags))

    if solution["what_to_check"]:
        st.markdown("**What to check:**")
        for item in solution["what_to_check"]:
            st.markdown(f"- {item}")

    if solution["insufficient_evidence"]:
        st.warning(
            "⚠️ Additional investigation is required -- no source currently has enough evidence to "
            "recommend a specific resolution."
        )
    else:
        st.success(f"**Recommended resolution** _(confidence: {solution['confidence']})_")
        st.markdown(solution["recommended_resolution"])

    link_cols = []
    if solution.get("supporting_tfs_url"):
        link_cols.append(f"[TFS-{solution['supporting_tfs_id']} →]({solution['supporting_tfs_url']})")
    if solution.get("supporting_wiki_url"):
        link_cols.append(f"[{solution['supporting_wiki_title']} (Wiki) →]({solution['supporting_wiki_url']})")
    if link_cols:
        st.caption(" · ".join(link_cols))
    st.markdown("---")


def render_investigation_strategy(strategy: dict, *, investigation_id: str) -> str | None:
    """Returns the linked component name the engineer clicked through to
    Product Intelligence for, if any -- same cross-navigation contract
    ``render_recommended_log_collection`` already used; the caller
    applies it via the existing pending-key + rerun pattern."""
    icon, label = _STAGE_DISPLAY.get(strategy["current_stage"], ("❓", strategy["current_stage"]))
    st.markdown(f"### {icon} {label}")
    st.caption(strategy["stage_rationale"])
    st.progress(strategy["progress"], text=strategy["progress_summary"])

    _render_recommended_solution(strategy.get("recommended_solution"))

    st.markdown("#### ➡️ Recommended next action")
    st.markdown(strategy["recommended_next_action"])
    st.caption(strategy["next_action_rationale"])

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

    historical = strategy["historical_investigations"]
    if historical:
        with st.expander(f"📊 Historical investigations ({len(historical)})", expanded=False):
            for i, match in enumerate(historical[:5]):
                render_historical_match(match, key=f"strategy_hist_{investigation_id}_{i}")

    known_bugs = strategy["known_bugs"]
    if known_bugs:
        with st.expander(f"🐞 Known bugs ({len(known_bugs)})", expanded=False):
            for bug in known_bugs[:5]:
                st.markdown(f"**{truncate_words(bug['title'])}**")

    documentation = strategy["documentation"]
    if documentation:
        with st.expander(f"📚 Documentation ({len(documentation)})", expanded=False):
            for doc in documentation[:5]:
                st.markdown(f"**{truncate_words(doc['title'])}**")

    render_external_knowledge(strategy.get("tfs_matches"), strategy.get("wiki_matches"))

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

    if strategy["decision_checkpoint"]:
        st.info(f"🧭 **Decision checkpoint:** {strategy['decision_checkpoint']}")

    return navigate_to_component
