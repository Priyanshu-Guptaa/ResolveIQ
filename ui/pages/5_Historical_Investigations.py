"""Historical Investigations.

Phase 1 scope: real semantic search scoped to the historical_investigations
collection. The 10-filter enterprise panel (Product, Version, Technology,
Component, Customer, Issue Type, Severity, Status, Tags, Date Range) is
Phase 7 -- those fields don't exist on HistoricalInvestigationRecord yet,
so building the filter UI now would be decorative, not functional. This
page stays a working search box until Phase 7 adds the real fields.
"""

from __future__ import annotations

from api_client import api_available, api_get
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("📊 Historical Investigations")
st.caption("Enterprise filters (Product, Version, Technology, Component, Customer, ...) arrive in Phase 7.")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

query = st.text_input("🔎 Search historical investigations")
if query.strip():
    results = (
        api_get(
            "/knowledge/search",
            params={"q": query, "collection": "historical_investigations", "top_k": 10},
        )
        or []
    )
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
