"""Settings -- read-only view of the running configuration (RFC rev 3)."""

from __future__ import annotations

from api_client import api_available, api_get
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("⚙ Settings")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

settings = api_get("/settings")
if settings is None:
    st.stop()

st.json(settings)
st.caption("Read-only for Sprint 2 -- editable settings are a future sprint's scope.")
