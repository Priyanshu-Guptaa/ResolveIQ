"""ResolveIQ multi-page entrypoint (RFC rev 3).

Replaces Sprint 1's single-page ``streamlit_app.py``. Run with:

    streamlit run ui/Home.py

Every page below is a genuinely working screen wired to the FastAPI
backend -- pages not yet reached by their assigned Sprint 2 phase are
intentionally minimal, never fake: real data, real requests, honestly
scoped down until their phase lands (see each page's module docstring).
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="ResolveIQ", page_icon="🔎", layout="wide")

pages = [
    st.Page("pages/1_Dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("pages/2_Investigation_Workspace.py", title="Investigation Workspace", icon="🔍"),
    st.Page("pages/3_Log_Intelligence.py", title="Log Intelligence", icon="📂"),
    st.Page("pages/4_Knowledge_Center.py", title="Knowledge Center", icon="📚"),
    st.Page("pages/5_Historical_Investigations.py", title="Historical Investigations", icon="📊"),
    st.Page("pages/6_SQL_Studio.py", title="SQL Studio", icon="🗄"),
    st.Page("pages/7_AI_Assistant.py", title="AI Assistant", icon="🤖"),
    st.Page("pages/8_Settings.py", title="Settings", icon="⚙"),
]

navigation = st.navigation(pages)
navigation.run()
