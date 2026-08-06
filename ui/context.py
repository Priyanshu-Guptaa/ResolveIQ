"""Shared Investigation Context.

Every page that operates on "the investigation the engineer is currently
looking at" reads/writes the same session_state key through this module,
instead of each page keeping its own independent selection. Because
Streamlit's multi-page shell (st.navigation) shares one session per
browser tab across all pages, setting the active investigation on one
page makes every other page that reads this module see it immediately on
its next render -- no cross-page messaging needed, just a shared key.
"""

from __future__ import annotations

import streamlit as st

_KEY = "active_investigation_id"


def get_active_investigation_id() -> str | None:
    return st.session_state.get(_KEY)


def set_active_investigation_id(investigation_id: str) -> None:
    st.session_state[_KEY] = investigation_id


def clear_active_investigation() -> None:
    st.session_state.pop(_KEY, None)
