"""AI Assistant / Chat (2026-08-14, Phase 4 -- Chat Orchestrator + API +
UI). Replaces the Phase 1 static-recommendation body per the approved
architecture (§30) -- deterministic, evidence-backed chat over the
exact same RecommendationEngine/StructuredResolution pipeline the
Investigation Workspace already uses. No LLM/Copilot/external AI
anywhere: every statement in an answer is quoted or directly derived
from real ResolveIQ evidence, shown alongside the answer below it,
never hidden and never fabricated.

Investigation-scoped (pick an investigation, same as before) and
standalone (no investigation) chat are both supported -- see
``ChatOrchestrator``/``ConversationStateEngine`` for how each resolves
retrieval context.

Only the most recent assistant turn's full evidence panel is rendered
in detail (historical matches, provenance, validation steps, ...) --
older turns in the transcript show their plain answer text only.
Per this phase's explicit constraint ("do not persist generated answer
text as authoritative knowledge... no chat persistence beyond the
Phase 3 implementation"), the backend itself only ever persists a
message's plain text, never the full structured response -- so a page
reload cannot reconstruct evidence for turns older than the current
one. This is a real, reported UI limitation (see this phase's report),
not an oversight.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, api_post, ensure_api_available
from components.external_knowledge import render_external_knowledge
from components.historical_match import render_historical_match
from components.investigation_picker import render_investigation_picker
from formatting import truncate_words
from theme import inject_theme

inject_theme()
st.title("🤖 AI Assistant")
st.caption(
    "Deterministic, evidence-backed chat -- every answer is generated from ResolveIQ's own knowledge "
    "base, TFS/Wiki, and applicability rules, not an AI model. Every claim below the answer traces to "
    "real matched evidence."
)
ensure_api_available()

_PROVENANCE_BADGE = {
    "confirmed": "🟢 **Confirmed**",
    "likely": "🟡 **Likely**",
    "possible": "🟠 **Possible**",
    "unknown": "⚪ **Unknown**",
}
_EVIDENCE_KIND_LABEL = {
    "historical_investigation": "📚 Historical Investigation",
    "known_bug": "🐞 Known Bug",
    "documentation": "📄 Documentation",
    "tfs_case": "🔧 TFS",
    "wiki_page": "🌐 Wiki",
    "sql_template": "🗄️ SQL Library",
    "log_collection_step": "🪵 Log Collection",
    "entity_heuristic": "🔍 Extracted Evidence",
    "component_match": "🧩 Component Match",
}

scope = st.radio("Scope", ["Investigation", "Standalone"], horizontal=True, key="chat_scope")

investigation_id: str | None = None
if scope == "Investigation":
    investigation_id = render_investigation_picker(key="chat_investigation_picker")
    if not investigation_id:
        st.stop()

session_key = f"{scope}::{investigation_id or 'standalone'}"
if st.session_state.get("chat_session_key") != session_key:
    session = api_post("/chat/sessions", {"investigation_id": investigation_id})
    if session is None:
        st.stop()
    st.session_state["chat_session_key"] = session_key
    st.session_state["chat_session_id"] = session["id"]
    st.session_state.pop("chat_last_response", None)

chat_session_id = st.session_state["chat_session_id"]

messages = api_get(f"/chat/sessions/{chat_session_id}/messages") or []
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_text = st.chat_input('Ask a question, e.g. "Has this happened before?"')
if user_text:
    with st.chat_message("user"):
        st.write(user_text)
    with st.spinner("Searching ResolveIQ's knowledge base, TFS, and Wiki..."):
        response = api_post(f"/chat/sessions/{chat_session_id}/messages", {"message": user_text})
    if response is not None:
        st.session_state["chat_last_response"] = response
    st.rerun()

response = st.session_state.get("chat_last_response")
if response is None:
    st.stop()

with st.chat_message("assistant"):
    st.write(response["answer_text"])

    ctx = response.get("active_context") or {}
    context_bits = [
        f"**{label}:** {ctx[field]}"
        for label, field in (("Customer", "customer_name"), ("Region", "region_name"), ("Technology", "technology_name"))
        if ctx.get(field)
    ]
    if context_bits:
        st.caption(" · ".join(context_bits))

    if response.get("resolution_provenance"):
        st.markdown(_PROVENANCE_BADGE.get(response["resolution_provenance"], response["resolution_provenance"]))

    if response.get("follow_up_question"):
        st.info(f"❓ {response['follow_up_question']}")

    ambiguity = response.get("ambiguity") or {}
    if ambiguity.get("kind") not in (None, "none"):
        candidates = ambiguity.get("candidates") or []
        st.warning("Multiple possible matches found: " + (", ".join(candidates) if candidates else "see follow-up question above"))

    structured = response.get("structured_resolution")
    if structured:
        if structured.get("root_cause"):
            st.markdown(f"**Root cause:** {structured['root_cause']}")
        if structured.get("confidence_rationale"):
            st.caption(structured["confidence_rationale"])

        candidates = structured.get("resolution_candidates") or []
        if candidates:
            st.markdown("**Resolution candidates**" + (" _(all attributed sources, not just the primary one)_" if len(candidates) > 1 else ""))
            for candidate in candidates:
                tag = " · _primary_" if candidate["is_primary"] else " · _also seen_"
                evidence = candidate["evidence"]
                kind_label = _EVIDENCE_KIND_LABEL.get(evidence["kind"], evidence["kind"])
                st.markdown(f"- {candidate['text']}{tag}")
                st.caption(f"{kind_label} · {evidence['title']} -- {evidence['reason']}")

        validation_steps = structured.get("validation_steps") or []
        if validation_steps:
            st.markdown("**Validation steps** _(how to check, not a claim that this was already checked)_")
            for step in validation_steps:
                st.markdown(f"- {step['instruction']}")
                if step.get("expected_result"):
                    st.caption(f"Expected result: {step['expected_result']}")
                if step.get("disproving_result"):
                    st.caption(f"Would disprove this: {step['disproving_result']}")

    historical = response.get("historical_investigations") or []
    if historical:
        with st.expander(f"📚 Historical investigations ({len(historical)})"):
            for i, match in enumerate(historical):
                render_historical_match(match, key=f"chat_hist_{i}")

    known_bugs = response.get("known_bugs") or []
    if known_bugs:
        with st.expander(f"🐞 Known bugs ({len(known_bugs)})"):
            for i, match in enumerate(known_bugs):
                render_historical_match(match, key=f"chat_bug_{i}")

    documentation = response.get("documentation") or []
    if documentation:
        with st.expander(f"📄 Documentation ({len(documentation)})"):
            for match in documentation:
                st.markdown(f"**{truncate_words(match['title'])}** _({match['score']:.0%} similarity)_")
                if match.get("snippet"):
                    st.caption(match["snippet"])
                if match.get("reason"):
                    st.caption(f"Why: {match['reason']}")

    recommended_logs = response.get("recommended_logs") or []
    if recommended_logs:
        with st.expander(f"🪵 Recommended logs ({len(recommended_logs)})"):
            for item in recommended_logs:
                st.markdown(f"**{item['priority_label']} · {item['component_name']}** ({item['scenario_technology']} / {item['scenario_type']})")
                st.caption(item["explanation"])
                st.caption(f"Why: {item['match_reason']}")

    suggested_sql = response.get("suggested_sql") or []
    if suggested_sql:
        with st.expander(f"🗄️ Suggested SQL ({len(suggested_sql)})"):
            for item in suggested_sql:
                st.markdown(f"**{item['title']}**")
                st.code(item["sql_text"], language="sql")
                if item.get("match_reason"):
                    st.caption(f"Why: {item['match_reason']}")

    if response.get("tfs_matches") or response.get("wiki_matches"):
        with st.expander("🔧 TFS / 🌐 Wiki (live)"):
            render_external_knowledge(response.get("tfs_matches"), response.get("wiki_matches"))
