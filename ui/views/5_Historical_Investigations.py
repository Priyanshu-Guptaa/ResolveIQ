"""Historical Investigations.

Phase 1 scope: real semantic search scoped to the historical_investigations
collection (cached, Phase 1.5). The 10-filter enterprise panel (Product,
Version, Technology, Component, Customer, Issue Type, Severity, Status,
Tags, Date Range) is Phase 7 -- those fields don't exist on
HistoricalInvestigationRecord yet, so building the filter UI now would be
decorative, not functional.

Phase 1.5: if an investigation is active in the shared context, shows its
already-computed ``similar_investigations`` (Recommendation Engine,
Sprint 1) above the manual search -- real reuse, not new matching logic.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, ensure_api_available, search_knowledge_cached
from context import get_active_investigation_id
from theme import inject_theme

inject_theme()
st.title("📊 Historical Investigations")
st.caption("Enterprise filters (Product, Version, Technology, Component, Customer, ...) arrive in Phase 7.")
ensure_api_available()

investigation_id = get_active_investigation_id()
if investigation_id:
    investigation = api_get(f"/investigations/{investigation_id}")
    if investigation:
        st.info(f"Scoped to: **{investigation['title']}**")
        recommendation = api_get(f"/investigations/{investigation_id}/recommendations")
        if recommendation and recommendation["similar_investigations"]:
            st.subheader("Similar to this investigation")
            for m in recommendation["similar_investigations"]:
                st.markdown(f"**{m['title']}** _({m['score']:.0%} similarity)_")
                st.caption(m["snippet"])
            st.divider()

st.subheader("Search all historical investigations")
query = st.text_input("🔎 Search")
if query.strip():
    results = search_knowledge_cached(query, collection="historical_investigations", top_k=10) or []
    if not results:
        st.caption("No matches.")
    for r in results:
        st.markdown(f"**{r['title']}** _({r['score']:.0%} similarity)_")
        st.caption(r["snippet"])
        if r["metadata"].get("root_cause"):
            st.caption(f"Root cause: {r['metadata']['root_cause']}")
        st.divider()
else:
    st.info("Type a search term above.")
