"""Dashboard -- the operational hub (RFC rev 3, §08).

Every panel below reads from GET /dashboard, which is backed entirely by
real queries (see app/api/routers/dashboard.py and its engine methods).
No panel shows illustrative/placeholder numbers -- where a metric has no
data yet (e.g. avg resolution time before anything is resolved), it shows
an honest empty state instead of a fabricated figure.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_available, api_get, api_post
from components.status_badge import status_badge_html
from theme import inject_theme

inject_theme()

st.title("🏠 Dashboard")

if not api_available():
    st.error(
        "Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload` "
        "and reload this page."
    )
    st.stop()

dashboard = api_get("/dashboard")
if dashboard is None:
    st.stop()

stats = dashboard["stats"]

# --- KPI row -----------------------------------------------------------

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(
        f'<div class="riq-kpi"><div class="riq-kpi-value">{stats["active_count"]}</div>'
        '<div class="riq-kpi-label">Active</div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    avg = stats["avg_resolution_hours"]
    display = f"{avg}h" if avg is not None else "—"
    st.markdown(
        f'<div class="riq-kpi"><div class="riq-kpi-value">{display}</div>'
        '<div class="riq-kpi-label">Avg resolution</div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    st.markdown(
        f'<div class="riq-kpi"><div class="riq-kpi-value">{stats["resolved_count"]}</div>'
        '<div class="riq-kpi-label">Resolved</div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[3]:
    st.markdown(
        f'<div class="riq-kpi"><div class="riq-kpi-value">{stats["total_count"]}</div>'
        '<div class="riq-kpi-label">Total investigations</div></div>',
        unsafe_allow_html=True,
    )

st.divider()

# --- Quick Actions -------------------------------------------------------

st.markdown('<div class="riq-panel-title">Quick actions</div>', unsafe_allow_html=True)
qa_cols = st.columns(4)
with qa_cols[0]:
    with st.popover("➕ New Investigation", use_container_width=True):
        title = st.text_input("Title", key="qa_new_title")
        description = st.text_area("Task description", key="qa_new_desc", height=100)
        if st.button("Create", key="qa_new_submit") and title.strip():
            created = api_post("/investigations", {"title": title, "description": description})
            if created:
                st.success(f"Created {created['id']}. Open it from Investigation Workspace.")
with qa_cols[1]:
    if st.button("🔎 Search Knowledge", use_container_width=True):
        st.switch_page("pages/4_Knowledge_Center.py")
with qa_cols[2]:
    if st.button("🗄 Open SQL Studio", use_container_width=True):
        st.switch_page("pages/6_SQL_Studio.py")
with qa_cols[3]:
    if st.button("📚 Knowledge Center", use_container_width=True):
        st.switch_page("pages/4_Knowledge_Center.py")

st.divider()

# --- Investigations row ---------------------------------------------------

col_active, col_viewed, col_activity = st.columns(3)

with col_active:
    st.markdown('<div class="riq-panel-title">My Active Investigations</div>', unsafe_allow_html=True)
    active = dashboard["active_investigations"]
    if not active:
        st.caption("No active investigations. Start one above.")
    for inv in active:
        st.markdown(
            f'<div class="riq-card"><div class="riq-card-title">{inv["title"]}</div>'
            f'<div class="riq-card-meta">{inv["id"][:8]} · {inv["evidence_count"]} evidence</div>'
            f'{status_badge_html(inv["status"])}</div>',
            unsafe_allow_html=True,
        )

with col_viewed:
    st.markdown('<div class="riq-panel-title">Recently Viewed</div>', unsafe_allow_html=True)
    viewed = dashboard["recently_viewed"]
    if not viewed:
        st.caption("Nothing viewed yet -- open an investigation to populate this.")
    for inv in viewed:
        st.markdown(
            f'<div class="riq-card"><div class="riq-card-title">{inv["title"]}</div>'
            f'<div class="riq-card-meta">viewed {inv["last_viewed_at"][:16].replace("T", " ")}</div></div>',
            unsafe_allow_html=True,
        )

with col_activity:
    st.markdown('<div class="riq-panel-title">Recent Activity</div>', unsafe_allow_html=True)
    activity = dashboard["recent_activity"]
    if not activity:
        st.caption("No activity yet.")
    for item in activity:
        st.markdown(
            f'<div class="riq-card"><div class="riq-card-title">{item["label"]}</div>'
            f'<div class="riq-card-meta">{item["occurred_at"][:16].replace("T", " ")}</div></div>',
            unsafe_allow_html=True,
        )

st.divider()

# --- Knowledge / documents / SQL row --------------------------------------

col_docs, col_kb, col_sql = st.columns(3)

with col_docs:
    st.markdown('<div class="riq-panel-title">Recently Added Knowledge</div>', unsafe_allow_html=True)
    docs = dashboard["recent_knowledge"]
    if not docs:
        st.caption("No documentation imported yet.")
    for doc in docs:
        st.markdown(
            f'<div class="riq-card"><div class="riq-card-title">{doc["title"]}</div>'
            f'<div class="riq-card-meta">documentation</div></div>',
            unsafe_allow_html=True,
        )

with col_kb:
    st.markdown('<div class="riq-panel-title">Recently Added Known Bugs</div>', unsafe_allow_html=True)
    bugs = dashboard["recent_known_bugs"]
    if not bugs:
        st.caption("No known bugs recorded yet.")
    for bug in bugs:
        st.markdown(
            f'<div class="riq-card"><div class="riq-card-title">{bug["title"]}</div>'
            f'<div class="riq-card-meta">known bug</div></div>',
            unsafe_allow_html=True,
        )

with col_sql:
    st.markdown('<div class="riq-panel-title">Query Library</div>', unsafe_allow_html=True)
    st.caption("Frequency ranking arrives with SQL Studio (Phase 8) usage tracking.")
    for q in dashboard["query_library_preview"]:
        st.markdown(
            f'<div class="riq-card"><div class="riq-card-title">{q["title"]}</div>'
            f'<div class="riq-card-meta">{q["category"]}</div></div>',
            unsafe_allow_html=True,
        )
