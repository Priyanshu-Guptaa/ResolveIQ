"""Relationship Manager -- Sprint 3, Phase 3.3's Administration module.
ResolveIQ is "no longer simply storing documents. It is maintaining a
knowledge graph" -- this page is where an administrator views, adds,
removes, searches, and validates the relationships between all nine
knowledge object types, browses them via the Explorer, and checks the
Knowledge Health report and Impact Analysis before changing anything.

No role gate: see app/api/routers/admin/__init__.py.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_delete, api_get, api_post, ensure_api_available

from components.object_picker import render_object_picker
from components.object_types import TYPE_LABELS as _TYPE_LABELS
from theme import inject_theme

RELATIONSHIP_TYPES = ["related_to", "documents", "fixes", "requires", "uses", "applies_to", "supersedes"]

inject_theme()
st.title("🕸️ Relationship Manager")
st.caption(
    "Administration · the knowledge graph connecting Components, Documents, Known Bugs, SQL Templates, "
    "Historical Investigations, Playbooks, Products, Technologies, and Versions."
)
ensure_api_available()


def _ref_line(ref: dict) -> str:
    label = _TYPE_LABELS.get(ref["type"], ref["type"])
    subtitle = f" — {ref['subtitle']}" if ref.get("subtitle") else ""
    return f"`{label}` **{ref['title']}**{subtitle}"


tab_explorer, tab_manage, tab_validate, tab_health, tab_impact = st.tabs(
    ["🔎 Explorer", "🔗 Manage Relationships", "✅ Validation", "📊 Knowledge Health", "🎯 Impact Analysis"]
)

# === Explorer =============================================================
with tab_explorer:
    st.markdown("Select any object to immediately see everything connected to it, grouped by type.")
    picked = render_object_picker(key="explorer", label="Center on")
    if picked:
        object_type, object_id, _ = picked
        view = api_get(f"/admin/relationships/explorer/{object_type}/{object_id}")
        if view:
            st.markdown(f"### {_ref_line(view['center'])}")
            if not view["groups"]:
                st.info("Nothing connected to this object yet -- add a relationship from the Manage tab.")
            for group in view["groups"]:
                label = _TYPE_LABELS.get(group["object_type"], group["object_type"])
                with st.expander(f"{label} ({len(group['objects'])})", expanded=True):
                    for obj in group["objects"]:
                        st.markdown(f"- {_ref_line(obj)}")

# === Manage Relationships ==================================================
with tab_manage:
    st.markdown("##### Add a relationship")
    st.caption("Both ends are picked from real, searchable objects -- never typed as free text.")
    from_picked = render_object_picker(key="manage_from", label="From")
    st.markdown("↓")
    to_picked = render_object_picker(key="manage_to", label="To")
    relationship_type = st.selectbox("Relationship type", RELATIONSHIP_TYPES, key="manage_rel_type")

    if from_picked and to_picked:
        from_type, from_id, from_label = from_picked
        to_type, to_id, to_label = to_picked
        if st.button("➕ Add relationship", type="primary"):
            if from_type == to_type and from_id == to_id:
                st.error("An object can't be related to itself.")
            else:
                result = api_post(
                    "/admin/relationships",
                    {
                        "from_type": from_type,
                        "from_id": from_id,
                        "to_type": to_type,
                        "to_id": to_id,
                        "relationship_type": relationship_type,
                    },
                )
                if result is not None:
                    st.success(f"Related {from_label} → {to_label}.")
                    st.rerun()

    st.divider()
    st.markdown("##### View / remove relationships for an object")
    view_picked = render_object_picker(key="manage_view", label="Object")
    if view_picked:
        object_type, object_id, _ = view_picked
        relationships = api_get("/admin/relationships", params={"object_type": object_type, "object_id": object_id}) or []
        if not relationships:
            st.caption("No relationships yet.")
        for rel in relationships:
            with st.container(border=True):
                cols = st.columns([5, 1])
                cols[0].markdown(
                    f"{_ref_line(rel['from_object'])} — *{rel['relationship']['relationship_type']}* → {_ref_line(rel['to_object'])}"
                )
                if cols[1].button("🗑 Remove", key=f"remove_{rel['relationship']['id']}"):
                    if api_delete(f"/admin/relationships/{rel['relationship']['id']}"):
                        st.rerun()

# === Validation ==========================================================
with tab_validate:
    st.markdown("Checks: broken links, duplicate relationships, circular references.")
    if st.button("Run validation", type="primary"):
        st.session_state["rm_validation"] = api_get("/admin/relationships/validate")
    issues = st.session_state.get("rm_validation")
    if issues is not None:
        if not issues:
            st.success("No issues found -- the relationship graph is consistent.")
        else:
            for issue in issues:
                renderer = st.error if issue["severity"] == "error" else st.warning
                renderer(f"**{issue['issue_type'].replace('_', ' ').title()}**: {issue['description']}")

# === Knowledge Health =====================================================
with tab_health:
    report = api_get("/admin/relationships/health")
    if report:
        cols = st.columns(4)
        cols[0].metric("Total Relationships", report["total_relationships"])
        cols[1].metric("Components", report["component_count"])
        cols[2].metric("Coverage", f"{report['component_relationship_coverage']:.0%}")
        cols[3].metric(
            "Integrity Issues",
            len(report["broken_relationships"]) + len(report["duplicate_relationships"]) + len(report["circular_references"]),
        )

        issue_cols = st.columns(3)
        with issue_cols[0]:
            st.markdown(f"**Broken links** ({len(report['broken_relationships'])})")
            for issue in report["broken_relationships"]:
                st.caption(issue["description"])
        with issue_cols[1]:
            st.markdown(f"**Duplicates** ({len(report['duplicate_relationships'])})")
            for issue in report["duplicate_relationships"]:
                st.caption(issue["description"])
        with issue_cols[2]:
            st.markdown(f"**Circular references** ({len(report['circular_references'])})")
            for issue in report["circular_references"]:
                st.caption(issue["description"])

        st.divider()
        unused_cols = st.columns(3)
        with unused_cols[0]:
            st.markdown(f"**Unused documents** ({len(report['unused_documents'])})")
            for ref in report["unused_documents"][:10]:
                st.caption(ref["title"])
        with unused_cols[1]:
            st.markdown(f"**Unused SQL templates** ({len(report['unused_sql_templates'])})")
            for ref in report["unused_sql_templates"][:10]:
                st.caption(ref["title"])
        with unused_cols[2]:
            st.markdown(f"**Unused components** ({len(report['unused_components'])})")
            for ref in report["unused_components"][:10]:
                st.caption(ref["title"])

        st.divider()
        missing_cols = st.columns(3)
        with missing_cols[0]:
            st.markdown(f"**Components missing playbooks** ({len(report['components_missing_playbooks'])})")
            for ref in report["components_missing_playbooks"][:10]:
                st.caption(ref["title"])
        with missing_cols[1]:
            st.markdown(f"**Components missing documentation** ({len(report['components_missing_documentation'])})")
            for ref in report["components_missing_documentation"][:10]:
                st.caption(ref["title"])
        with missing_cols[2]:
            st.markdown(
                f"**Components missing historical investigations** "
                f"({len(report['components_missing_historical_investigations'])})"
            )
            for ref in report["components_missing_historical_investigations"][:10]:
                st.caption(ref["title"])

# === Impact Analysis ========================================================
with tab_impact:
    st.markdown("Before changing or removing an object, see everything that depends on it.")
    impact_picked = render_object_picker(key="impact", label="Object to analyze")
    if impact_picked:
        object_type, object_id, _ = impact_picked
        impact = api_get(f"/admin/relationships/impact/{object_type}/{object_id}")
        if impact:
            st.markdown(f"### {_ref_line(impact['object'])}")
            if impact["total_dependents"] == 0:
                st.success("Nothing depends on this -- safe to change or remove without affecting other objects.")
            else:
                st.warning(f"{impact['total_dependents']} object(s) depend on this. Review before changing it.")
                for group in impact["dependents"]:
                    label = _TYPE_LABELS.get(group["object_type"], group["object_type"])
                    with st.expander(f"{label} ({len(group['objects'])})", expanded=True):
                        for obj in group["objects"]:
                            st.markdown(f"- {_ref_line(obj)}")
