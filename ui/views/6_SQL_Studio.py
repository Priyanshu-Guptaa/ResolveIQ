"""SQL Studio.

Phase 1 scope: the read-only Query Library via GET /sql/library (cached,
Phase 1.5). Saved / Recent / Favourite queries and the Query Generator
are Phase 8. The Run button is disabled everywhere in Sprint 2, per RFC
rev 3 §15 -- explicitly out of scope, not a placeholder.

Phase 1.5: reads the shared Investigation Context (context.py). If an
investigation is active, shows its already-computed ``suggested_sql``
(from the Recommendation Engine, Sprint 1) as a "Suggested for this
investigation" section above the static library -- real reuse of data
already computed elsewhere, not new logic.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, ensure_api_available, get_query_library_cached
from context import get_active_investigation_id
from theme import inject_theme

inject_theme()
st.title("🗄 SQL Studio")
st.caption("Saved / Recent / Favourite queries arrive in Phase 8. Query execution is out of scope for Sprint 2.")
ensure_api_available()

investigation_id = get_active_investigation_id()
if investigation_id:
    investigation = api_get(f"/investigations/{investigation_id}")
    if investigation:
        st.info(f"Scoped to: **{investigation['title']}**")
        recommendation = api_get(f"/investigations/{investigation_id}/recommendations")
        if recommendation and recommendation["suggested_sql"]:
            st.subheader("Suggested for this investigation")
            for sql in recommendation["suggested_sql"]:
                st.code(sql, language="sql")
            st.divider()

library = get_query_library_cached() or []

st.subheader("Query Library")
for q in library:
    with st.expander(f"{q['title']} · {q['category']}"):
        st.code(q["sql_text"], language="sql")
        st.caption(q["explanation"])
        st.button("▶ Run", disabled=True, key=f"run_{q['id']}", help="Execution ships in a later sprint.")
