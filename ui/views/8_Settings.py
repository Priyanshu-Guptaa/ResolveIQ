"""Settings -- read-only view of the running configuration (RFC rev 3)."""

from __future__ import annotations

import streamlit as st
from api_client import ensure_api_available, get_settings_cached
from theme import inject_theme

inject_theme()
st.title("⚙ Settings")
ensure_api_available()

settings = get_settings_cached()
if settings is None:
    st.stop()

st.json(settings)
st.caption("Read-only for Sprint 2 -- editable settings are a future sprint's scope.")
