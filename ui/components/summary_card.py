"""Persistent Investigation Summary card (RFC rev 3, Phase 2A).

Every field is either real backend data or an honest "--" when unset --
nothing here is illustrative. Confidence and Hypothesis come from the
Recommendation Engine's *existing* output (no new recommendation logic,
per the Phase 2A scope: "do not expand Recommendation Engine logic yet").
"""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st
from components.status_badge import status_badge_html


def _elapsed_since(iso_timestamp: str) -> str:
    try:
        created = datetime.fromisoformat(iso_timestamp)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created
    except ValueError:
        return "—"
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def render_summary_card(investigation: dict, recommendation: dict | None) -> None:
    confidence = f"{recommendation['overall_confidence']:.0%}" if recommendation else "—"
    hypothesis = "—"
    if recommendation and recommendation.get("root_causes"):
        hypothesis = recommendation["root_causes"][0]["description"]
        if len(hypothesis) > 70:
            hypothesis = hypothesis[:70] + "…"

    product = investigation.get("product") or "—"
    version = investigation.get("version") or "—"

    cells = [
        ("Customer", investigation.get("customer") or "—"),
        ("Issue", (investigation["title"][:50] + "…") if len(investigation["title"]) > 50 else investigation["title"]),
        ("Product · Version", f"{product} · {version}" if product != "—" or version != "—" else "—"),
        ("Technology", investigation.get("technology") or "—"),
        ("Status", status_badge_html(investigation["status"])),
        ("Engineer", investigation.get("assigned_engineer") or "—"),
        ("Confidence", confidence),
        ("Time Spent", _elapsed_since(investigation["created_at"])),
        ("Hypothesis", hypothesis),
    ]

    cell_html = "".join(
        f'<div class="riq-summary-cell"><div class="riq-summary-key">{key}</div>'
        f'<div class="riq-summary-value">{value}</div></div>'
        for key, value in cells
    )
    st.markdown(f'<div class="riq-summary-card">{cell_html}</div>', unsafe_allow_html=True)


def render_details_editor(investigation: dict, *, on_save) -> None:
    """Small form for the engineer-entered fields. ``on_save`` receives a
    dict of the four text fields plus assigned_engineer and is
    responsible for calling the API (kept out of this component so it
    stays UI-only, no HTTP knowledge)."""
    with st.form("details_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            customer = st.text_input("Customer", value=investigation.get("customer") or "")
            product = st.text_input("Product", value=investigation.get("product") or "")
            version = st.text_input("Version", value=investigation.get("version") or "")
        with col2:
            technology = st.text_input("Technology", value=investigation.get("technology") or "")
            assigned_engineer = st.text_input("Assigned Engineer", value=investigation.get("assigned_engineer") or "")
        if st.form_submit_button("Save details"):
            on_save(
                {
                    "customer": customer or None,
                    "product": product or None,
                    "version": version or None,
                    "technology": technology or None,
                    "assigned_engineer": assigned_engineer or None,
                }
            )
