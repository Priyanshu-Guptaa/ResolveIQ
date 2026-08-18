"""Classification Review (2026-08-14, Phase 5; UI improvement pass
same day) -- admin queue for the Medium-confidence classification
suggestions ``DocumentClassificationEngine`` already produces
(``app/engines/knowledge/classification.py``, unchanged by this page).

Pure API consumer -- this page never talks to the classification
engine, ``ClassificationRepository``, or the database directly. Every
accept/reject goes through the existing, unchanged admin endpoints
(``app/api/routers/admin/classification.py``): ``POST .../run``,
``GET .../pending``, ``POST .../{id}/accept``, ``POST .../{id}/reject``.
No second classification/matching implementation, no raw DB mutation
control, nothing bypasses the existing repository/API validation, no
bulk accept/reject, no auto-decisions -- every action is still one
explicit human confirmation per suggestion.

UI improvement pass (2026-08-14, same day as the initial page):
``GET /admin/classification/pending`` now returns ``PendingSuggestionView``
(``app/domain/classification.py``) -- each suggestion enriched with its
real object title and a re-derived real mention count, both resolved
read-only by ``DocumentClassificationEngine.list_pending_with_context()``.
This page adds sorting, grouping, and richer evidence display on top of
that -- still a pure, read-only consumer of the API response; nothing
here computes a title or a mention count itself.

High-confidence suggestions already auto-applied (never appear here);
Low-confidence suggestions are recorded for traceability only and also
never appear here. The queue size is always fetched live, never
assumed or hardcoded.

No role gate: see ``app/api/routers/admin/__init__.py``.
"""

from __future__ import annotations

import streamlit as st
from api_client import api_get, api_post, ensure_api_available

from theme import inject_theme

inject_theme()
st.title("🏷️ Classification Review")
st.caption(
    "Administration · review queue for Medium-confidence classification suggestions. High-confidence "
    "matches already applied automatically; Low-confidence matches are recorded for traceability only and "
    "never appear here. Every suggestion is extractive against an already-governed Customer/Region/"
    "Technology/Component/Product value -- see each suggestion's own evidence below, never a guess."
)
ensure_api_available()

_DIMENSION_ICON = {
    "product": "📦", "technology": "🔧", "component": "🧩", "customer": "🏢", "region": "🌍", "version": "🏷️",
}
_TIER_STYLE = {
    "high": ("#E3F0E7", "#3E8B5C"), "medium": ("#F5E9D6", "#B8791A"), "low": ("#EEF1F4", "#8892A0"),
}
_SORT_RECOMMENDED = "Recommended (Dimension → Value → Mentions)"
_SORT_OPTIONS = [
    _SORT_RECOMMENDED, "Suggested value", "Object title", "Dimension", "Object type",
    "Mention count", "Confidence / status",
]
_GROUP_OPTIONS = ["None (flat list)", "Dimension → Suggested value", "Dimension", "Object type"]


def _pill(label: str, bg: str, fg: str) -> str:
    return (
        f'<span style="display:inline-flex;align-items:center;gap:5px;'
        f'font-family:ui-monospace,Consolas,monospace;font-size:0.72rem;'
        f'padding:3px 9px;border-radius:99px;font-weight:600;'
        f'background:{bg};color:{fg};">{label}</span>'
    )


def _tier_badge(tier: str) -> str:
    bg, fg = _TIER_STYLE.get(tier, ("#EEF1F4", "#545F6E"))
    return _pill(tier.title(), bg, fg)


def _sort_key(option: str):
    """Every key is built only from fields already on the API response
    -- no new computation, just a read of what's there."""
    if option == _SORT_RECOMMENDED:
        return lambda s: (s["dimension"], s["suggested_value_text"], -(s["mention_count"] if s["mention_count"] is not None else -1))
    if option == "Suggested value":
        return lambda s: s["suggested_value_text"].lower()
    if option == "Object title":
        return lambda s: (s["object_title"] is None, (s["object_title"] or "").lower())
    if option == "Dimension":
        return lambda s: s["dimension"]
    if option == "Object type":
        return lambda s: s["object_type"]
    if option == "Mention count":
        return lambda s: s["mention_count"] if s["mention_count"] is not None else -1
    if option == "Confidence / status":
        return lambda s: (s["confidence_tier"], s["status"])
    return lambda s: 0


def _render_evidence_card(suggestion: dict, *, actor: str) -> None:
    """One suggestion's full review card -- object identity, evidence,
    and the accept/reject confirm flow. Shared by both the flat list
    and every grouped view so the review action is identical either
    way."""
    suggestion_id = suggestion["id"]
    icon = _DIMENSION_ICON.get(suggestion["dimension"], "🏷️")

    with st.container(border=True):
        header_col, badge_col = st.columns([5, 2])
        header_col.markdown(f"{icon} **{suggestion['dimension'].title()} → {suggestion['suggested_value_text']}**")
        mentions = suggestion.get("mention_count")
        mention_label = f"{mentions} mention{'s' if mentions != 1 else ''}" if mentions is not None else "mentions: n/a"
        badge_col.markdown(
            _tier_badge(suggestion["confidence_tier"]) + " " + _pill(suggestion["status"].title(), "#EEF1F4", "#545F6E")
            + " " + _pill(mention_label, "#E4EDF1", "#1F5772"),
            unsafe_allow_html=True,
        )

        title = suggestion.get("object_title")
        if title:
            st.markdown(f"**{title}**")
            st.caption(f"{suggestion['object_type'].replace('_', ' ').title()} · `{suggestion['object_id'][:8]}…`")
        else:
            st.warning(f"⚠️ Title unavailable -- object may have been deleted. `{suggestion['object_type']}:{suggestion['object_id']}`")

        st.markdown(f"**Evidence** _({suggestion['evidence_rule'].replace('_', ' ')})_:")
        # st.text (not st.markdown) for the raw snippet itself -- some real
        # evidence contains "$...$" (e.g. PowerShell variable syntax),
        # which st.markdown's LaTeX support would otherwise misrender.
        # This changes nothing about what text is shown, only how it's
        # rendered -- found live in this pass's own verification.
        st.text(suggestion["evidence_snippet"])

        confirm_key = f"classification_confirm_{suggestion_id}"
        pending_action = st.session_state.get(confirm_key)

        if pending_action is None:
            accept_col, reject_col, _spacer = st.columns([1, 1, 4])
            if accept_col.button("✅ Accept", key=f"classification_accept_{suggestion_id}"):
                st.session_state[confirm_key] = "accept"
                st.rerun()
            if reject_col.button("❌ Reject", key=f"classification_reject_{suggestion_id}"):
                st.session_state[confirm_key] = "reject"
                st.rerun()
        else:
            verb = "Accept" if pending_action == "accept" else "Reject"
            st.warning(
                f"Confirm: **{verb}** {suggestion['dimension']} → {suggestion['suggested_value_text']} on "
                f"{suggestion['object_type']} `{suggestion_id[:8]}`?"
            )
            confirm_col, cancel_col, _spacer = st.columns([1, 1, 4])
            if confirm_col.button(f"Confirm {verb}", key=f"{confirm_key}_yes", type="primary"):
                endpoint = f"/admin/classification/{suggestion_id}/{'accept' if pending_action == 'accept' else 'reject'}"
                result = api_post(endpoint, {"actor": actor})
                st.session_state.pop(confirm_key, None)
                if result is not None:
                    st.success(f"{verb}ed.")
                    st.rerun()
            if cancel_col.button("Cancel", key=f"{confirm_key}_no"):
                st.session_state.pop(confirm_key, None)
                st.rerun()


actor = st.text_input(
    "Reviewing as",
    value=st.session_state.get("classification_review_actor", "admin"),
    key="classification_review_actor",
    help="Recorded on every accept/reject as reviewed_by, and on a run as its actor -- the same actor "
    "identity every other admin action in this app already requires.",
)

with st.expander("⚙️ Run classification scan"):
    st.caption(
        "Re-scans Documents, Historical Investigations, and Known Bugs for new suggestions. Safe to "
        "re-run -- idempotent: an (object, dimension, value) pair that already has a suggestion on file is "
        "never duplicated or re-decided (app.engines.knowledge.classification.DocumentClassificationEngine.run)."
    )
    limit_input = st.number_input(
        "Limit (0 = scan everything)", min_value=0, value=0, step=50, key="classification_run_limit"
    )
    if st.button("Run classification scan", key="classification_run_button"):
        with st.spinner("Running classification scan..."):
            summary = api_post(
                "/admin/classification/run", {"actor": actor, "limit": int(limit_input) or None}
            )
        if summary is not None:
            st.success(
                f"Scanned {summary['documents_scanned']} object(s) — {summary['suggestions_created']} new "
                f"suggestion(s) ({summary['auto_accepted']} auto-accepted, {summary['pending_review']} "
                f"pending review, {summary['recorded_low_confidence']} low-confidence recorded)."
            )
            if summary["errors"]:
                with st.expander(f"⚠️ {len(summary['errors'])} error(s) during the scan"):
                    for err in summary["errors"]:
                        st.caption(err)
            st.rerun()

st.divider()

pending = api_get("/admin/classification/pending")
if pending is None:
    st.stop()

st.markdown(f"### {len(pending)} pending suggestion(s)")

if not pending:
    st.info("The review queue is empty — nothing awaiting a decision right now.")
    st.stop()

# --- Filters (unchanged from the initial Phase 5 page) ----------------------

filter_col1, filter_col2, filter_col3 = st.columns(3)
dimensions = sorted({s["dimension"] for s in pending})
object_types = sorted({s["object_type"] for s in pending})
tiers = sorted({s["confidence_tier"] for s in pending})

dimension_filter = filter_col1.multiselect("Dimension", dimensions, default=[], key="classification_filter_dimension")
object_type_filter = filter_col2.multiselect("Object type", object_types, default=[], key="classification_filter_object_type")
tier_filter = filter_col3.multiselect(
    "Confidence", tiers, default=[], key="classification_filter_tier",
    help="Every item in this queue is Medium confidence by construction (High auto-applies, Low never "
    "queues) -- this filter is here for completeness and stays ready if that ever changes.",
)

filtered = [
    s
    for s in pending
    if (not dimension_filter or s["dimension"] in dimension_filter)
    and (not object_type_filter or s["object_type"] in object_type_filter)
    and (not tier_filter or s["confidence_tier"] in tier_filter)
]

# --- Sorting + grouping (new) -----------------------------------------------

sort_col, dir_col, group_col = st.columns([2, 1, 2])
sort_by = sort_col.selectbox("Sort by", _SORT_OPTIONS, index=0, key="classification_sort_by")
descending = False
if sort_by != _SORT_RECOMMENDED:
    descending = dir_col.checkbox("Descending", value=(sort_by == "Mention count"), key="classification_sort_desc")
group_by = group_col.selectbox("Group by", _GROUP_OPTIONS, index=0, key="classification_group_by")

filtered = sorted(filtered, key=_sort_key(sort_by), reverse=descending)

st.caption(f"Showing {len(filtered)} of {len(pending)} pending suggestions.")

if not filtered:
    st.info("No pending suggestions match the current filters.")
    st.stop()

# --- Render: grouped or flat -------------------------------------------------

if group_by == "None (flat list)":
    _PAGE_SIZE = 25
    total_pages = max(1, (len(filtered) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="classification_page")
    st.caption(f"Page {page} of {total_pages}")
    start = (int(page) - 1) * _PAGE_SIZE
    for suggestion in filtered[start : start + _PAGE_SIZE]:
        _render_evidence_card(suggestion, actor=actor)
else:
    if group_by == "Dimension → Suggested value":
        key_fn = lambda s: (s["dimension"], s["suggested_value_text"])
        label_fn = lambda k: f"{_DIMENSION_ICON.get(k[0], '🏷️')} {k[0].title()} → {k[1]}"
    elif group_by == "Dimension":
        key_fn = lambda s: (s["dimension"],)
        label_fn = lambda k: f"{_DIMENSION_ICON.get(k[0], '🏷️')} {k[0].title()}"
    else:  # "Object type"
        key_fn = lambda s: (s["object_type"],)
        label_fn = lambda k: k[0].replace("_", " ").title()

    groups: dict[tuple, list[dict]] = {}
    for s in filtered:
        groups.setdefault(key_fn(s), []).append(s)
    # Largest groups first -- this is exactly what surfaces a systematic
    # pattern (e.g. many "Scheduler" suggestions) at a glance.
    ordered_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    for key, items in ordered_groups:
        with st.expander(f"{label_fn(key)}  ({len(items)})"):
            for suggestion in items:
                _render_evidence_card(suggestion, actor=actor)
