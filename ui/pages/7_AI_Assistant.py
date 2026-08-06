"""AI Assistant.

Phase 1 scope: shows the Recommendation Engine's real output (root causes,
next best step) for a chosen investigation -- genuine data, not a chat
transcript. The Documentation Generator buttons (Customer Update, L2
Notes, L3 Escalation, RCA, Resolution Summary, Internal Notes) are Phase
4. No LLM calls anywhere in this module, per RFC rev 3 sign-off.
"""

from __future__ import annotations

from api_client import api_available, api_get
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("🤖 AI Assistant")
st.caption("Scoped to one investigation at a time. Generator buttons (RCA, Customer Update, ...) arrive in Phase 4.")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

investigations = api_get("/investigations") or []
if not investigations:
    st.info("No investigations yet -- start one from the Dashboard.")
    st.stop()

options = {f"{inv['title']} ({inv['id'][:8]})": inv["id"] for inv in investigations}
selected = st.selectbox("Investigation", list(options.keys()))
investigation_id = options[selected]

recommendation = api_get(f"/investigations/{investigation_id}/recommendations")
if recommendation is None:
    st.stop()

st.markdown(f"### Next best step\n{recommendation['next_best_step']}")
if recommendation["root_causes"]:
    st.markdown("### Root cause hypotheses")
    for rc in recommendation["root_causes"]:
        st.markdown(f"- **{rc['description']}** _(confidence {rc['confidence']:.0%})_ -- {rc['rationale']}")
