"""Dashboard -- the operational hub (RFC rev 3, §08).

Every panel reads from GET /dashboard (cached 15s, Phase 1.5 -- see
api_client.get_dashboard_cached), which is backed entirely by real
queries. No panel shows illustrative/placeholder numbers -- where a
metric has no data yet, it shows an honest empty state.

Phase 1.5: clicking "Open" on an active investigation sets the shared
Investigation Context and jumps straight to Workspace, instead of
requiring you to re-select it there.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_post, ensure_api_available, get_dashboard_cached, list_investigations_cached
from components.cards import render_card, render_kpi, render_panel_title
from components.status_badge import status_badge_html
from context import set_active_investigation_id
from theme import inject_theme

inject_theme()

header_col, refresh_col = st.columns([6, 1])
with header_col:
    st.title("🏠 Dashboard")
with refresh_col:
    st.write("")
    if st.button("🔄 Refresh"):
        get_dashboard_cached.clear()
        st.rerun()

ensure_api_available()

dashboard = get_dashboard_cached()
if dashboard is None:
    st.stop()

stats = dashboard["stats"]

# --- KPI row -----------------------------------------------------------

kpi_cols = st.columns(4)
with kpi_cols[0]:
    render_kpi(str(stats["active_count"]), "Active")
with kpi_cols[1]:
    avg = stats["avg_resolution_hours"]
    render_kpi(f"{avg}h" if avg is not None else "—", "Avg resolution")
with kpi_cols[2]:
    render_kpi(str(stats["resolved_count"]), "Resolved")
with kpi_cols[3]:
    render_kpi(str(stats["total_count"]), "Total investigations")

st.divider()

# --- Quick Actions -------------------------------------------------------

render_panel_title("Quick actions")
qa_cols = st.columns(4)
with qa_cols[0]:
    with st.popover("➕ New Investigation", use_container_width=True):
        title = st.text_input("Title", key="qa_new_title")
        description = st.text_area("Task description", key="qa_new_desc", height=100)
        if st.button("Create", key="qa_new_submit") and title.strip():
            created = api_post("/investigations", {"title": title, "description": description})
            if created:
                get_dashboard_cached.clear()
                list_investigations_cached.clear()
                set_active_investigation_id(created["id"])
                st.success(f"Created {created['id']}. Open it from Investigation Workspace.")
with qa_cols[1]:
    if st.button("🔎 Search Knowledge", use_container_width=True):
        st.switch_page("views/4_Knowledge_Center.py")
with qa_cols[2]:
    if st.button("🗄 Open SQL Studio", use_container_width=True):
        st.switch_page("views/6_SQL_Studio.py")
with qa_cols[3]:
    if st.button("📚 Knowledge Center", use_container_width=True):
        st.switch_page("views/4_Knowledge_Center.py")

st.divider()

# --- Investigations row ---------------------------------------------------

col_active, col_viewed, col_activity = st.columns(3)

with col_active:
    render_panel_title("My Active Investigations")
    active = dashboard["active_investigations"]
    if not active:
        st.caption("No active investigations. Start one above.")
    for inv in active:
        render_card(
            inv["title"],
            f'{inv["id"][:8]} · {inv["evidence_count"]} evidence',
            badge_html=status_badge_html(inv["status"]),
        )
        if st.button("Open →", key=f"open_{inv['id']}"):
            set_active_investigation_id(inv["id"])
            st.switch_page("views/2_Investigation_Workspace.py")

with col_viewed:
    render_panel_title("Recently Viewed")
    viewed = dashboard["recently_viewed"]
    if not viewed:
        st.caption("Nothing viewed yet -- open an investigation to populate this.")
    for inv in viewed:
        render_card(inv["title"], f'viewed {inv["last_viewed_at"][:16].replace("T", " ")}')

with col_activity:
    render_panel_title("Recent Activity")
    activity = dashboard["recent_activity"]
    if not activity:
        st.caption("No activity yet.")
    for item in activity:
        render_card(item["label"], item["occurred_at"][:16].replace("T", " "))

st.divider()

# --- Knowledge / documents / SQL row --------------------------------------

col_docs, col_kb, col_sql = st.columns(3)

with col_docs:
    render_panel_title("Recently Added Knowledge")
    docs = dashboard["recent_knowledge"]
    if not docs:
        st.caption("No documentation imported yet.")
    for doc in docs:
        render_card(doc["title"], "documentation")

with col_kb:
    render_panel_title("Recently Added Known Bugs")
    bugs = dashboard["recent_known_bugs"]
    if not bugs:
        st.caption("No known bugs recorded yet.")
    for bug in bugs:
        render_card(bug["title"], "known bug")

with col_sql:
    render_panel_title("Query Library")
    st.caption("Frequency ranking arrives with SQL Studio (Phase 8) usage tracking.")
    for q in dashboard["query_library_preview"]:
        render_card(q["title"], q["category"])
