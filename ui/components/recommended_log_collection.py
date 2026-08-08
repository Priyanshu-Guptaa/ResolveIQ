"""Recommended Log Collection (Log Intelligence) -- renders
``Recommendation.recommended_logs`` as a guided, step-by-step
collection workflow: which log, why (both the structural explanation
and *why this scenario matched this investigation*), whether it looks
already collected, and a jump-off point into Product Intelligence when
the underlying log source is a known architecture Component.

One reusable renderer (not duplicated between the Recommendations tab
and the Playbook tab) -- nothing here re-derives matching, priority,
already-collected detection, or component linking; that's entirely the
Recommendation Engine's job (``app/engines/recommendation/engine.py``).
This module only presents what it's given.
"""

from __future__ import annotations

import streamlit as st

_PRIORITY_ICON = {"Critical": "🔴", "Recommended": "🟡", "Optional": "⚪"}
_PRIORITY_ORDER = {"Critical": 0, "Recommended": 1, "Optional": 2}


def _group_by_scenario(items: list[dict]) -> dict[tuple[str, str, str | None], list[dict]]:
    grouped: dict[tuple[str, str, str | None], list[dict]] = {}
    for item in items:
        key = (item["scenario_technology"], item["scenario_type"], item.get("scenario_region"))
        grouped.setdefault(key, []).append(item)
    return grouped


def _checklist_key(investigation_id: str, item: dict) -> str:
    return f"logcol_{investigation_id}_{item['scenario_id']}_{item['log_source_id']}_{item['order']}"


def render_recommended_log_collection(recommended_logs: list[dict], *, investigation_id: str) -> str | None:
    """Grouped by matched scenario, most-confident match first; each
    scenario's own steps stay in the wiki-preserved collection order.

    Collection state is session-only (resets on reload), same as every
    other checklist in this workspace -- see the Playbook tab's own
    note on this; a persisted Playbook Engine is separate future work,
    not something this render function silently takes on.

    Returns the linked component name the engineer clicked through to
    Product Intelligence for, if any -- the caller (Investigation
    Workspace) applies it via the same pending-key + rerun pattern
    already used for Architecture Explorer cross-navigation.
    """
    if not recommended_logs:
        return None

    st.markdown("#### 📂 Recommended Log Collection")
    st.caption(
        "Exact logs to collect for this investigation's matched scenario(s), in order -- sourced from "
        "the Log Intelligence Knowledge Base, not guessed. Pre-checked items were detected from evidence "
        "already uploaded to this investigation; toggle any item by hand as you actually collect it."
    )

    # Seed each item's checkbox from the engine's auto-detection exactly
    # once -- before the widget exists, per Streamlit's rule that a
    # widget's session_state key can't be written after instantiation.
    for item in recommended_logs:
        key = _checklist_key(investigation_id, item)
        if key not in st.session_state:
            st.session_state[key] = item["already_collected"]

    total = len(recommended_logs)
    collected = sum(1 for item in recommended_logs if st.session_state.get(_checklist_key(investigation_id, item)))
    st.progress(collected / total if total else 0.0, text=f"{collected} of {total} logs collected")

    ordered_items = sorted(
        recommended_logs, key=lambda i: (_PRIORITY_ORDER.get(i["priority_label"], 9), i["order"])
    )
    next_item = next(
        (i for i in ordered_items if not st.session_state.get(_checklist_key(investigation_id, i))), None
    )
    if next_item:
        region_suffix = f" ({next_item['scenario_region']})" if next_item.get("scenario_region") else ""
        st.info(
            f"➡️ Next: collect **{next_item['component_name']}** logs "
            f"({next_item['scenario_technology']} / {next_item['scenario_type']}{region_suffix})"
        )
    else:
        st.success("✅ All recommended logs are collected.")

    navigate_to_component: str | None = None
    grouped = _group_by_scenario(recommended_logs)
    ordered_keys = sorted(grouped.keys(), key=lambda k: _PRIORITY_ORDER.get(grouped[k][0]["priority_label"], 9))

    for technology, scenario_type, region in ordered_keys:
        items = sorted(grouped[(technology, scenario_type, region)], key=lambda i: i["order"])
        label = items[0]["priority_label"]
        icon = _PRIORITY_ICON.get(label, "⚪")
        region_suffix = f" ({region})" if region else ""
        with st.expander(f"{icon} {label} -- {technology} / {scenario_type}{region_suffix}", expanded=(label == "Critical")):
            st.caption(f"🧭 {items[0]['match_reason']}")
            for item in items:
                key = _checklist_key(investigation_id, item)
                check_col, detail_col = st.columns([0.06, 0.94])
                check_col.checkbox("Collected", key=key, label_visibility="collapsed")
                with detail_col:
                    st.markdown(f"**{item['order']}. {item['component_name']}**")
                    st.caption(item["explanation"])
                    path = item["repository_root_path"]
                    if item.get("repository_subdirectory"):
                        path = f"{path}/{item['repository_subdirectory']}"
                    filenames = ", ".join(item.get("filename_patterns") or []) or "(filename not captured)"
                    st.code(f"[{item['repository_platform']}] {path}\nfiles: {filenames}", language=None)
                    if item.get("linked_component_name"):
                        if st.button(
                            f"🧩 View {item['linked_component_name']} in Product Intelligence",
                            key=f"{key}_pi_link",
                        ):
                            navigate_to_component = item["linked_component_name"]

    return navigate_to_component


def flatten_for_checklist(recommended_logs: list[dict]) -> list[str]:
    """The Playbook tab's plain checkbox list wants one short line per
    log, not the full guided layout -- same underlying data, a
    simpler, printable presentation. Already-collected items are
    marked so the engineer can see at a glance what's outstanding even
    before manually checking the Playbook's own (separately-stateful,
    session-only) box for it."""
    grouped = _group_by_scenario(recommended_logs)
    lines: list[str] = []
    for (technology, scenario_type, region), items in sorted(
        grouped.items(), key=lambda kv: _PRIORITY_ORDER.get(kv[1][0]["priority_label"], 9)
    ):
        region_suffix = f" ({region})" if region else ""
        for item in sorted(items, key=lambda i: i["order"]):
            marker = "✅ already collected -- " if item.get("already_collected") else ""
            lines.append(
                f"[{item['priority_label']}] {marker}Collect {item['component_name']} logs "
                f"({technology} / {scenario_type}{region_suffix}, step {item['order']})"
            )
    return lines
