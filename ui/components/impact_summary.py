"""Impact Summary (Sprint 3, Phase 3.4) -- "before deleting or modifying
any knowledge object, show every downstream dependency," rendered as a
reusable panel rather than only living on the Relationship Manager's
dedicated tab. Same ``ImpactAnalysis`` endpoint Phase 3.3 already built
(``/admin/relationships/impact/...``) -- no new backend logic.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get

from components.object_types import TYPE_LABELS


def render_impact_summary(object_type: str, object_id: str) -> None:
    impact = api_get(f"/admin/relationships/impact/{object_type}/{object_id}")
    if not impact:
        return
    if impact["total_dependents"] == 0:
        st.success("Nothing depends on this -- safe to change or delete.")
        return

    st.warning(f"{impact['total_dependents']} object(s) depend on this. Review before changing or deleting it.")
    for group in impact["dependents"]:
        label = TYPE_LABELS.get(group["object_type"], group["object_type"])
        with st.expander(f"{label} ({len(group['objects'])})"):
            for obj in group["objects"]:
                subtitle = f" — {obj['subtitle']}" if obj.get("subtitle") else ""
                st.markdown(f"- **{obj['title']}**{subtitle}")
