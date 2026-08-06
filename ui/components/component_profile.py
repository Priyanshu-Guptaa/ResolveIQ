"""Renders one ComponentProfile. Plain lookup/display only -- no
reasoning about which component is relevant to anything (that's
Recommendation Engine V2's job, a later phase).
"""

from __future__ import annotations

import streamlit as st

_FIELDS = [
    ("responsibilities", "Responsibilities"),
    ("related_components", "Related Components"),
    ("typical_failures", "Typical Failures"),
    ("required_logs", "Required Logs"),
    ("common_sql", "Common SQL Queries"),
    ("known_bugs", "Known Bugs"),
    ("documentation_links", "Documentation"),
    ("playbooks", "Investigation Playbooks"),
]


def render_component_profile(profile: dict) -> None:
    st.caption(profile["product"])
    for field_key, label in _FIELDS:
        values = profile.get(field_key) or []
        if not values:
            continue
        st.markdown(f"**{label}**")
        for value in values:
            if field_key == "common_sql":
                st.code(value, language="sql")
            else:
                st.caption(f"- {value}")
