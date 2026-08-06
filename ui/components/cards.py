"""Shared card/KPI rendering.

Phase 1 built the ``.riq-card`` / ``.riq-kpi`` HTML strings inline,
repeated across Dashboard's 6+ panels. These two functions are the single
place that HTML gets built now.
"""

from __future__ import annotations

import streamlit as st


def render_kpi(value: str, label: str) -> None:
    st.markdown(
        f'<div class="riq-kpi"><div class="riq-kpi-value">{value}</div>'
        f'<div class="riq-kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


def render_card(title: str, meta: str = "", *, badge_html: str = "") -> None:
    meta_html = f'<div class="riq-card-meta">{meta}</div>' if meta else ""
    st.markdown(
        f'<div class="riq-card"><div class="riq-card-title">{title}</div>{meta_html}{badge_html}</div>',
        unsafe_allow_html=True,
    )


def render_panel_title(title: str) -> None:
    st.markdown(f'<div class="riq-panel-title">{title}</div>', unsafe_allow_html=True)
