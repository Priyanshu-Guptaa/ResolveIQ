"""AI Assistant.

Phase 1 scope: shows the Recommendation Engine's real output (root causes,
next best step) for the shared active investigation -- genuine data, not
a chat transcript. The Documentation Generator buttons (Customer Update,
L2 Notes, L3 Escalation, RCA, Resolution Summary, Internal Notes) are
Phase 4. No LLM calls anywhere in this module, per RFC rev 3 sign-off.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, ensure_api_available
from components.investigation_picker import render_investigation_picker
from theme import inject_theme

inject_theme()
st.title("🤖 AI Assistant")
st.caption("Scoped to one investigation at a time. Generator buttons (RCA, Customer Update, ...) arrive in Phase 4.")
ensure_api_available()

investigation_id = render_investigation_picker(key="ai_assistant_picker")
if not investigation_id:
    st.stop()

recommendation = api_get(f"/investigations/{investigation_id}/recommendations")
if recommendation is None:
    st.stop()

st.markdown(f"### Next best step\n{recommendation['next_best_step']}")
if recommendation["root_causes"]:
    st.markdown("### Root cause hypotheses")
    for rc in recommendation["root_causes"]:
        st.markdown(f"- **{rc['description']}** _(confidence {rc['confidence']:.0%})_ -- {rc['rationale']}")
