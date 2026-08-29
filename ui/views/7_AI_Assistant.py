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
from components.chat_attachments import (
    format_enhancement_status,
    format_eviction_notice,
    persistence_message,
    record_log_attachment,
)
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
    st.session_state.pop("chat_log_attachments", None)
    st.session_state.pop("chat_log_eviction_notice", None)
    st.session_state.pop("chat_enhancement_job_id", None)
    st.session_state.pop("chat_enhancement_result", None)

chat_session_id = st.session_state["chat_session_id"]

messages = api_get(f"/chat/sessions/{chat_session_id}/messages") or []
for msg in messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat Assistant Phase 39 -- chat-side log upload. Mirrors the
# Investigation Workspace's own upload pattern (files= via api_post);
# the uploaded content flows through the existing, unmodified
# LogIntelligenceEngine, never a second parser. Restricted to the four
# plain-text log formats the chat-upload endpoint accepts.
#
# Chat Assistant Phase 42 -- shows the FULL retained attachment list
# (up to 5), not just the most recent upload (Phase 41 audit finding).
# The FIFO cap (`app.engines.chat.log_upload._MAX_LOGS_PER_SESSION`)
# only applies server-side to standalone sessions -- an
# investigation-scoped upload goes through
# InvestigationEngine.add_file_evidence, which has no cap at all -- so
# `capped` below must track the real backend contract per session type,
# never fabricate an eviction that did not actually happen.
with st.expander("📎 Attach a log file", expanded=False):
    uploaded_log = st.file_uploader(
        "Upload a log (.log, .txt, .csv, .json)",
        type=["log", "txt", "csv", "json"],
        key="chat_log_uploader",
    )
    if uploaded_log and st.button("Attach log", key="chat_log_attach_btn"):
        with st.spinner("Analyzing log..."):
            result = api_post(
                f"/chat/sessions/{chat_session_id}/logs",
                files=[("file", (uploaded_log.name, uploaded_log.getvalue()))],
                timeout=120,
            )
        if result is not None:
            existing_attachments = st.session_state.get("chat_log_attachments", [])
            updated_attachments, evicted = record_log_attachment(
                existing_attachments, result, capped=(scope == "Standalone")
            )
            st.session_state["chat_log_attachments"] = updated_attachments
            # Overwritten on every upload attempt (never left stale from an
            # earlier upload) -- always None when this upload didn't evict
            # anything, so the notice can never outlive its own cause.
            st.session_state["chat_log_eviction_notice"] = format_eviction_notice(evicted)
            entity_word = "entity" if result["entity_count"] == 1 else "entities"
            st.success(f"Attached **{result['title']}** -- {result['event_count']} event(s), {result['entity_count']} recognized {entity_word}.")
            for warning in result.get("warnings") or []:
                st.caption(f"⚠️ {warning}")

    attachments = st.session_state.get("chat_log_attachments", [])
    if attachments:
        st.markdown(f"**{len(attachments)} log(s) currently attached to this chat:**")
        for item in attachments:
            item_entity_word = "entity" if item.get("entity_count") == 1 else "entities"
            st.caption(
                f"📎 {item['title']} -- {item.get('event_count', 0)} event(s), "
                f"{item.get('entity_count', 0)} recognized {item_entity_word}"
            )
        st.caption(persistence_message(investigation_scoped=(scope == "Investigation")))

eviction_notice = st.session_state.get("chat_log_eviction_notice")
if eviction_notice:
    st.warning(eviction_notice)

if attachments:
    attachment_names = ", ".join(item["title"] for item in attachments)
    st.caption(f"📎 {len(attachments)} log(s) attached to this conversation: {attachment_names}")

user_text = st.chat_input('Ask a question, e.g. "Has this happened before?" or "Analyze this log"')
if user_text:
    with st.chat_message("user"):
        st.write(user_text)
    with st.spinner("Searching ResolveIQ's knowledge base, TFS, and Wiki..."):
        response = api_post(f"/chat/sessions/{chat_session_id}/messages", {"message": user_text})
    if response is not None:
        st.session_state["chat_last_response"] = response
        # Chat Assistant Phase 42 -- a new turn always starts a fresh
        # enhancement lifecycle: any earlier turn's job/result belonged to
        # a different answer and must never be shown attached to this one.
        enhancement_ref = response.get("enhancement")
        st.session_state["chat_enhancement_job_id"] = enhancement_ref["job_id"] if enhancement_ref else None
        st.session_state["chat_enhancement_result"] = None
    st.rerun()

response = st.session_state.get("chat_last_response")
if response is None:
    st.stop()

with st.chat_message("assistant"):
    st.write(response["answer_text"])

    # Chat Assistant Phase 42 -- surface the existing async LLM
    # enhancement (Phase 37/38 backend, previously never rendered here).
    # The deterministic answer above is ALWAYS shown and is never
    # replaced or reworded by anything below -- this is presentation
    # enhancement only, never a second source of truth for confidence,
    # provenance, or resolution. Bounded polling: at most one
    # GET /chat/enhancements/{job_id} call per script rerun, no loop, no
    # sleep -- a still-pending job is checked again only on the next
    # natural rerun (a new message, an upload, or the button below).
    enhancement_job_id = st.session_state.get("chat_enhancement_job_id")
    if enhancement_job_id:
        poll_result = api_get(f"/chat/enhancements/{enhancement_job_id}")
        outcome = format_enhancement_status(poll_result)
        if outcome["outcome"] != "pending":
            st.session_state["chat_enhancement_result"] = outcome
            st.session_state["chat_enhancement_job_id"] = None  # terminal -- stop polling this job

    enhancement_result = st.session_state.get("chat_enhancement_result")
    if enhancement_result is not None:
        if enhancement_result["outcome"] == "completed":
            st.info(
                "✨ **AI-enhanced phrasing** _(same evidence, confidence, and root cause as above -- "
                f"this only rewords it)_:\n\n{enhancement_result['answer_text']}"
            )
        else:
            st.caption(f"AI-enhanced phrasing unavailable: {enhancement_result['message']}")
    elif st.session_state.get("chat_enhancement_job_id"):
        st.caption("⏳ AI-enhanced phrasing is still being generated...")
        if st.button("🔄 Check for AI-enhanced answer", key="chat_enhancement_check_btn"):
            st.rerun()

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
