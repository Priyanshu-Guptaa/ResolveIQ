"""Recommended Log Collection (Log Intelligence) -- renders
``Recommendation.recommended_logs``: the structured "collect exactly
these logs, in this order, and here's why" guidance that replaces the
generic "upload logs" hint whenever a Log Collection Scenario matched
the investigation.

One reusable renderer (not duplicated between the Recommendations tab
and the Playbook tab) -- both call this; nothing here re-derives
matching or priority, that's entirely the Recommendation Engine's job
(``app/engines/recommendation/engine.py``).
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


def render_recommended_log_collection(recommended_logs: list[dict]) -> None:
    """Grouped by matched scenario, most-confident match first; each
    scenario's own steps stay in the wiki-preserved collection order."""
    if not recommended_logs:
        return

    st.markdown("#### 📂 Recommended Log Collection")
    st.caption(
        "Exact logs to collect for this investigation's matched scenario(s), in order -- sourced from "
        "the Log Intelligence Knowledge Base, not guessed."
    )

    grouped = _group_by_scenario(recommended_logs)
    ordered_keys = sorted(
        grouped.keys(), key=lambda k: _PRIORITY_ORDER.get(grouped[k][0]["priority_label"], 9)
    )

    for technology, scenario_type, region in ordered_keys:
        items = sorted(grouped[(technology, scenario_type, region)], key=lambda i: i["order"])
        label = items[0]["priority_label"]
        icon = _PRIORITY_ICON.get(label, "⚪")
        region_suffix = f" ({region})" if region else ""
        with st.expander(f"{icon} {label} -- {technology} / {scenario_type}{region_suffix}", expanded=(label == "Critical")):
            for item in items:
                st.markdown(f"**{item['order']}. {item['component_name']}**")
                st.caption(item["explanation"])
                path = item["repository_root_path"]
                if item.get("repository_subdirectory"):
                    path = f"{path}/{item['repository_subdirectory']}"
                filenames = ", ".join(item.get("filename_patterns") or []) or "(filename not captured)"
                st.code(f"[{item['repository_platform']}] {path}\nfiles: {filenames}", language=None)


def flatten_for_checklist(recommended_logs: list[dict]) -> list[str]:
    """The Playbook tab's checkbox list wants one short line per log,
    not the full expander layout -- same underlying data, different
    presentation."""
    grouped = _group_by_scenario(recommended_logs)
    lines: list[str] = []
    for (technology, scenario_type, region), items in sorted(
        grouped.items(), key=lambda kv: _PRIORITY_ORDER.get(kv[1][0]["priority_label"], 9)
    ):
        region_suffix = f" ({region})" if region else ""
        for item in sorted(items, key=lambda i: i["order"]):
            lines.append(
                f"[{item['priority_label']}] Collect {item['component_name']} logs "
                f"({technology} / {scenario_type}{region_suffix}, step {item['order']})"
            )
    return lines
