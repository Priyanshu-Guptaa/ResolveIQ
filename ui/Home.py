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

# Sprint 3, Phase 3.2 introduces the Administration Portal's first
# module (Knowledge Management) -- grouped nav sections now, per
# RFC-003. Access control is enforced by the API, not by hiding pages:
# with RESOLVEIQ_AUTH_ENABLED the sign-in gate lives in
# api_client.ensure_api_available(), and every /admin/* endpoint
# requires the admin role (an engineer opening an Administer page gets
# a 403 from the API). With auth disabled (local use) every page is open.
navigation = st.navigation(
    {
        "Investigate": [
            st.Page("views/1_Dashboard.py", title="Dashboard", icon="🏠", default=True),
            st.Page("views/2_Investigation_Workspace.py", title="Investigation Workspace", icon="🔍"),
            st.Page("views/3_Log_Intelligence.py", title="Log Intelligence", icon="📂"),
            st.Page("views/4_Knowledge_Center.py", title="Knowledge Center", icon="📚"),
            st.Page("views/5_Historical_Investigations.py", title="Historical Investigations", icon="📊"),
            st.Page("views/6_SQL_Studio.py", title="SQL Studio", icon="🗄"),
            st.Page("views/7_AI_Assistant.py", title="AI Assistant", icon="🤖"),
            st.Page("views/8_Settings.py", title="Settings", icon="⚙"),
        ],
        "Administer": [
            st.Page("views/9_Knowledge_Management.py", title="Knowledge Management", icon="🗂️"),
            st.Page("views/10_Relationship_Manager.py", title="Relationship Manager", icon="🕸️"),
            st.Page("views/11_Knowledge_Objects.py", title="Knowledge Objects", icon="🗃️"),
            st.Page("views/12_Task_Import.py", title="Task Import", icon="📥"),
            st.Page("views/13_Log_Wiki_Import.py", title="Log Wiki Import", icon="📚"),
            st.Page("views/14_Classification_Review.py", title="Classification Review", icon="🏷️"),
        ],
    }
)
navigation.run()
