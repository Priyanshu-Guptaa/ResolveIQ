"""Log Intelligence.

Three views over an investigation's uploaded logs:
 - Files: per-file raw event view (original behavior) -- pick one file,
   see its parsed events. Useful for a quick skim, not for finding
   anything specific across a large upload.
 - Search: a real cross-file filter (entity type/value, keyword,
   level) -- replaces "pick one file, see every line in it" for the
   actual "find X" case. Works with just an uploaded log file and no
   task description at all.
 - Command Flow: reconstructs the ordered, directional flow (Command
   Request Outbound -> ... -> Meter, or Meter -> ... -> Command
   Response Inbound) for one correlating value (Command Log ID or
   Meter/Endpoint Number), using the wiki-derived Log Intelligence
   Knowledge Base to resolve which component produced each matching
   log line and where it belongs in the real flow -- see
   app/engines/log_intelligence/flow.py.

Investigation loading redesign: the Files tab still fetches full
content for exactly one file at a time (unchanged from before). Search
and Command Flow necessarily scan every log file's already-parsed
content server-side (the same full-hydration path
GET .../recommendations already uses) -- only the filtered/matched
results are ever sent back to this page, never the full dump.
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

# Search/flow results are fetched on demand (button click), not on every
# render -- but must not silently carry over from a *different*
# investigation when the sidebar picker changes, same guard the
# Workspace's recommendation cache already uses.
if st.session_state.get("log_intel_results_for") != investigation_id:
    st.session_state["log_search_result"] = None
    st.session_state["log_flow_result"] = None
    st.session_state["log_intel_results_for"] = investigation_id

tab_files, tab_search, tab_flow = st.tabs(["📄 Files", "🔎 Search", "🔀 Command Flow"])

# --- Files (original per-file view) -----------------------------------------
with tab_files:

    def _option_label(ev: dict) -> str:
        kind = ev.get("file_kind") or "text"
        return f"{ev['title']} · {kind} · {ev['log_event_count']} events"

    options = {_option_label(ev): ev["id"] for ev in file_summaries}
    selected_label = st.selectbox(f"Files ({len(file_summaries)})", list(options.keys()), key="log_intel_file_pick")
    selected_id = options[selected_label]

    evidence = api_get(f"/investigations/{investigation_id}/evidence/{selected_id}")
    if evidence is not None:
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

# --- Search (cross-file filter) ---------------------------------------------
with tab_search:
    st.caption(
        "Filters across every uploaded log file at once -- works with just uploaded logs, no task "
        "description required. Leave a field blank to skip that filter."
    )
    _ENTITY_TYPES = [
        "(any)", "command_log_id", "meter_number", "endpoint_id", "request_id", "correlation_id",
        "serial_number", "session_id", "thread_id", "process_id", "sql_session", "pod_name", "host_name",
        "ip_address", "service_name", "kafka_topic", "rabbitmq_queue", "consumer_group", "exception_type",
        "event_id", "user_id",
    ]
    col1, col2 = st.columns(2)
    search_entity_type = col1.selectbox("Entity type", _ENTITY_TYPES, key="log_search_entity_type")
    search_entity_value = col2.text_input("Entity value", key="log_search_entity_value")
    col3, col4 = st.columns(2)
    search_keyword = col3.text_input("Keyword (message contains)", key="log_search_keyword")
    search_level = col4.selectbox("Level", ["(any)", "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"], key="log_search_level")

    if st.button("🔎 Search", type="primary", key="log_search_btn"):
        params = {}
        if search_entity_type != "(any)":
            params["entity_type"] = search_entity_type
        if search_entity_value.strip():
            params["entity_value"] = search_entity_value.strip()
        if search_keyword.strip():
            params["keyword"] = search_keyword.strip()
        if search_level != "(any)":
            params["level"] = search_level
        if not params:
            st.warning("Set at least one filter before searching.")
        else:
            st.session_state["log_search_result"] = api_get(f"/investigations/{investigation_id}/log-search", params=params)

    result = st.session_state.get("log_search_result")
    if result:
        st.markdown(f"##### {result['total_matches']} match(es) for {result['query_summary']}")
        if result["truncated"]:
            st.caption(f"Showing the first {len(result['hits'])} -- narrow the filter to see the rest.")
        for hit in result["hits"]:
            level = hit["level"]
            color = {"ERROR": "🔴", "WARN": "🟡", "FATAL": "🔴"}.get(level, "⚪")
            st.markdown(
                f"{color} `{hit.get('timestamp') or '--'}` **{level}** · *{hit['evidence_title']}* -- {hit['message'][:300]}"
            )

# --- Command Flow (reconstructed, directional) -------------------------------
with tab_flow:
    st.caption(
        "Reconstructs the ordered flow for one Command Log ID or Meter/Endpoint Number, across every "
        "uploaded log file -- direction and order come from the real wiki-derived message flow (Log "
        "Intelligence Knowledge Base), not guessed from log text. A step with no matching log entry is "
        "flagged, not hidden -- that's exactly where the trail goes cold."
    )
    col1, col2 = st.columns(2)
    flow_entity_type = col1.selectbox(
        "Correlating ID type",
        ["command_log_id", "meter_number", "endpoint_id", "request_id", "correlation_id"],
        key="log_flow_entity_type",
    )
    flow_entity_value = col2.text_input("Value", key="log_flow_entity_value")

    if st.button("🔀 Reconstruct Flow", type="primary", key="log_flow_btn"):
        if not flow_entity_value.strip():
            st.warning("Enter a value to reconstruct the flow for.")
        else:
            st.session_state["log_flow_result"] = api_get(
                f"/investigations/{investigation_id}/log-flow",
                params={"entity_type": flow_entity_type, "entity_value": flow_entity_value.strip()},
            )

    flow = st.session_state.get("log_flow_result")
    if flow:
        if flow["matched_component_count"] == 0 and not flow["unresolved_events"]:
            st.info(f"No log entries found for {flow['correlating_entity_type']}={flow['correlating_value']}.")
        else:
            if flow["scenario_type"]:
                st.markdown(f"##### {flow['scenario_technology']} / {flow['scenario_type']}")
            else:
                st.caption(
                    "No wiki scenario recognized the component(s) that logged this value -- showing raw "
                    "matches below without flow ordering."
                )

            def _render_step(step: dict) -> None:
                icon = "✅" if step["has_log_entry"] else "⬜"
                with st.expander(f"{icon} {step['order']}. {step['component_name']}", expanded=step["has_log_entry"]):
                    if not step["has_log_entry"]:
                        st.warning("No log entry found for this step -- possible gap in the collected evidence.")
                    for event in step["events"]:
                        st.markdown(f"`{event.get('timestamp') or '--'}` **{event['level']}** · *{event['evidence_title']}* -- {event['message'][:300]}")

            if flow["outbound_steps"]:
                st.markdown("**➡️ Sent**")
                for step in flow["outbound_steps"]:
                    _render_step(step)
            if flow["inbound_steps"]:
                st.markdown("**⬅️ Response**")
                for step in flow["inbound_steps"]:
                    _render_step(step)
            if flow["unresolved_events"]:
                with st.expander(f"❔ Unplaced matches ({len(flow['unresolved_events'])})", expanded=not flow["outbound_steps"] and not flow["inbound_steps"]):
                    st.caption("Matched this value, but the producing component isn't part of the resolved flow.")
                    for event in flow["unresolved_events"]:
                        st.markdown(f"`{event.get('timestamp') or '--'}` **{event['level']}** · *{event['evidence_title']}* -- {event['message'][:300]}")
