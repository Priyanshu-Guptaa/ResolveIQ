"""Injects ui/theme.css into the current Streamlit page.

Call once near the top of every page (Home.py and each ui/pages/*.py).
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent / "theme.css"


def inject_theme() -> None:
    css = _CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
