"""Historical Investigation match rendering -- one shared renderer for
the Recommendations panel's "Historical investigations" section and
the dedicated Historical Investigations search page, so both surfaces
show the real ticket number, a correctly word-wrapped title, and a
click-through detail view instead of each independently re-deriving
(or, before this, not deriving at all) the same thing.

Fixes two real, reported bugs: title truncation was a bare
``title[:60]`` slice (cuts mid-word, no ellipsis -- read as garbled
text, e.g. "...Each Incremental interval e"), and the ticket number
was captured on import (``tags: ["ticket:<number>"]``, see
``app/engines/task_import/importer.py``) but never surfaced anywhere,
so an engineer with a strong historical match had no way to look the
ticket up in ServiceNow or see its full resolution.

The detail view is a toggle button + inline container, not
``st.expander`` -- the Recommendations panel's caller already renders
this inside its own "Historical investigations" expander, and
Streamlit does not allow nesting expanders.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get
from formatting import ticket_number_from_tags, truncate_words


def render_historical_match(match: dict, *, key: str) -> None:
    """``match`` is a ``KnowledgeMatch`` dict (has ``record_id``,
    ``title``, ``score``, ``snippet``, ``metadata``). ``key`` must be
    unique across the whole page for this one call -- the same record
    can legitimately appear in more than one list on the same page
    (e.g. both "similar to this investigation" and a manual search),
    so the caller supplies it rather than this function guessing at
    uniqueness from ``record_id`` alone.

    The detail view fetches the full governed record on demand via the
    existing generic Knowledge Object endpoint -- no new backend
    surface for this, since every field it shows (description,
    root_cause, resolution, next_step, tags) is already there
    untruncated."""
    ticket = ticket_number_from_tags(match.get("metadata", {}).get("tags"))
    ticket_suffix = f" · 🎫 `{ticket}`" if ticket else ""
    st.markdown(f"**{truncate_words(match['title'])}** _({match['score']:.0%} similarity)_{ticket_suffix}")
    if match.get("snippet"):
        st.caption(match["snippet"])

    state_key = f"{key}_open"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    button_label = "▲ Hide full record" if st.session_state[state_key] else "▼ View full record"
    if st.button(button_label, key=f"{key}_toggle"):
        st.session_state[state_key] = not st.session_state[state_key]

    if not st.session_state[state_key]:
        return

    with st.container(border=True):
        record = api_get(f"/admin/objects/historical_investigation/{match['record_id']}")
        if record is None:
            st.caption("Record not found -- it may have been deleted since this match was computed.")
            return
        record_ticket = ticket_number_from_tags(record.get("tags"))
        if record_ticket:
            st.caption(f"Ticket: `{record_ticket}`  ·  Search this number in ServiceNow for the full ticket.")
        st.markdown(f"#### {record['title']}")
        if record.get("description"):
            st.markdown("**Description**")
            st.write(record["description"])
        if record.get("root_cause"):
            st.markdown("**Root cause**")
            st.write(record["root_cause"])
        if record.get("resolution"):
            st.markdown("**Resolution**")
            st.write(record["resolution"])
        if record.get("next_step"):
            st.markdown("**Next step**")
            st.write(record["next_step"])
        if record.get("tags"):
            st.caption("Tags: " + ", ".join(record["tags"]))
