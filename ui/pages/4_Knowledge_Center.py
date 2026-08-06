"""Knowledge Center.

Phase 1 scope: real semantic search across all three Knowledge Engine
collections via GET /knowledge/search. The 10-category browsing rail
(Documentation, Product Components, Technologies, Firmware/DCW, ...) is
Phase 6 -- this page is the search primitive Phase 6 builds the category
UI on top of, not a placeholder for it.
"""

from __future__ import annotations

from api_client import api_available, api_get
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("📚 Knowledge Center")
st.caption("Category browsing (Documentation, Product Components, Technologies, ...) arrives in Phase 6.")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

query = st.text_input("🔎 Search across documentation, historical investigations, and known bugs")
if query.strip():
    results = api_get("/knowledge/search", params={"q": query, "top_k": 10}) or []
    if not results:
        st.caption("No matches.")
    for r in results:
        st.markdown(f"**{r['title']}** _(`{r['collection']}` · {r['score']:.0%})_")
        st.caption(r["snippet"])
        st.divider()
else:
    st.info("Type a search term above.")
