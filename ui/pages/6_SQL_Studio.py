"""SQL Studio.

Phase 1 scope: the read-only Query Library via GET /sql/library. Saved /
Recent / Favourite queries and the Query Generator are Phase 8. The Run
button is disabled everywhere in Sprint 2, per RFC rev 3 §15 -- this is
not a placeholder, it's the specified behavior: execution is explicitly
out of scope until a future sprint picks a connection/security model.
"""

from __future__ import annotations

from api_client import api_available, api_get
from theme import inject_theme

import streamlit as st

inject_theme()
st.title("🗄 SQL Studio")
st.caption("Saved / Recent / Favourite queries arrive in Phase 8. Query execution is out of scope for Sprint 2.")

if not api_available():
    st.error("Can't reach the ResolveIQ API. Start it with `uvicorn app.api.main:app --reload`.")
    st.stop()

library = api_get("/sql/library") or []

st.subheader("Query Library")
for q in library:
    with st.expander(f"{q['title']} · {q['category']}"):
        st.code(q["sql_text"], language="sql")
        st.caption(q["explanation"])
        st.button("▶ Run", disabled=True, key=f"run_{q['id']}", help="Execution ships in a later sprint.")
