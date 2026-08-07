"""Renders one ComponentProfile as an Architecture Explorer panel --
grouped, navigable knowledge, not a flat documentation dump.

Still zero reasoning, per Domain Intelligence's scope discipline: the
only "intelligence" here is a plain string-equality check against known
component names, used to decide whether a relationship value should
render as a clickable link to that component's own profile. Everything
else is direct display of what's already in the registry. Deciding
*which* component matters for a given investigation is still
Recommendation Engine V2's job (Phase 2C), not this module's.
"""

from __future__ import annotations

import streamlit as st

_RELATIONSHIP_FIELDS = [
    ("related_components", "Related Components"),
    ("dependencies", "Dependencies"),
    ("consumes", "Consumes"),
    ("produces", "Produces"),
    ("related_services", "Related Services"),
    ("related_queues", "Related Queues"),
    ("database_tables", "Database Tables"),
]

_OPERATIONAL_FIELDS = [
    ("configuration", "Configuration"),
    ("events", "Events"),
    ("message_flows", "Message Flows"),
    ("data_flows", "Data Flows"),
]

_SUPPORT_FIELDS = [
    ("required_logs", "Required Logs"),
    ("known_bugs", "Known Bugs"),
    ("documentation_links", "Documentation"),
    ("playbooks", "Investigation Playbooks"),
    ("historical_investigations", "Historical Investigations"),
]


def _render_list_field(label: str, values: list[str]) -> None:
    if not values:
        return
    st.markdown(f"**{label}**")
    for value in values:
        st.caption(f"- {value}")


def _render_relationship_field(
    label: str,
    values: list[str],
    known_names: set[str],
    field_key: str,
    widget_prefix: str,
) -> str | None:
    """Same as _render_list_field, except a value that exactly matches
    another component's name renders as a clickable button -- the
    Architecture Explorer's navigation. Returns the clicked component
    name, if any (only one click is possible per Streamlit run)."""
    if not values:
        return None
    st.markdown(f"**{label}**")
    clicked = None
    for i, value in enumerate(values):
        if value in known_names:
            if st.button(f"→ {value}", key=f"{widget_prefix}_{field_key}_{i}", use_container_width=True):
                clicked = value
        else:
            st.caption(f"- {value}")
    return clicked


def render_component_profile(profile: dict, known_component_names: set[str] | None = None) -> str | None:
    """Renders the full profile in grouped sections.

    Returns the name of a related component the user clicked to
    navigate to, or None if nothing was clicked this run -- the caller
    (the Workspace page) is responsible for switching the selection and
    rerunning, since only it knows the selectbox's session_state key.
    """
    known_component_names = known_component_names or set()
    widget_prefix = f"pi_{profile['id']}"
    navigate_to: str | None = None

    st.caption(profile["product"])
    _render_list_field("Responsibilities", profile.get("responsibilities") or [])

    if any(profile.get(field_key) for field_key, _ in _RELATIONSHIP_FIELDS):
        st.divider()
        st.markdown("##### 🔗 Relationships")
        for field_key, label in _RELATIONSHIP_FIELDS:
            values = profile.get(field_key) or []
            clicked = _render_relationship_field(label, values, known_component_names, field_key, widget_prefix)
            navigate_to = navigate_to or clicked

    if any(profile.get(field_key) for field_key, _ in _OPERATIONAL_FIELDS):
        st.divider()
        st.markdown("##### ⚙️ Operational Shape")
        for field_key, label in _OPERATIONAL_FIELDS:
            _render_list_field(label, profile.get(field_key) or [])

    version_differences = profile.get("version_differences") or []
    if version_differences:
        st.divider()
        st.markdown("##### 🕒 Version Differences")
        for vd in version_differences:
            st.caption(f"**{vd['version']}** -- {vd['change']}")

    typical_failures = profile.get("typical_failures") or []
    if typical_failures:
        st.divider()
        _render_list_field("Typical Failures", typical_failures)

    common_sql = profile.get("common_sql") or []
    has_support_content = common_sql or any(profile.get(field_key) for field_key, _ in _SUPPORT_FIELDS)
    if has_support_content:
        st.divider()
        st.markdown("##### 📚 Support Knowledge")
        if common_sql:
            st.markdown("**Common SQL Queries**")
            for query in common_sql:
                st.code(query, language="sql")
        for field_key, label in _SUPPORT_FIELDS:
            _render_list_field(label, profile.get(field_key) or [])

    return navigate_to
