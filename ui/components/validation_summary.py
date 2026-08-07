"""Validation Summary (Sprint 3, Phase 3.4) -- renders the generic,
universal-only warnings from ``KnowledgeObjectService.validate``. Not to
be confused with Phase 3.3's relationship-graph validation
(broken/duplicate/circular links across the whole graph, still on the
Relationship Manager page) -- this is per-object.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get


def render_validation_summary(object_type: str, object_id: str) -> None:
    warnings = api_get(f"/admin/objects/{object_type}/{object_id}/validate")
    if warnings is None:
        return
    if not warnings:
        st.success("No issues found.")
        return
    for warning in warnings:
        st.warning(warning)
