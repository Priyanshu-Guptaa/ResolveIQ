"""External Knowledge -- live TFS + Wiki results, rendered exactly per
the approved layout: a status line per source ("✓ N relevant results
found" / "No relevant ... found" / an unavailable message), then one
card per match with its match reasons, extracted resolution, and a
real link back to the source.

Deliberately a thin renderer only -- every value here already came
from ``ExternalKnowledgeResult``/``ExternalMatch`` (see
app/domain/external_knowledge.py); nothing is computed or invented
here. Every TFS/Wiki-sourced string on screen is visually tagged with
its source, per the explicit "clearly identify TFS-derived
information" requirement.
"""

from __future__ import annotations

import streamlit as st

from formatting import truncate_words


def _render_tfs_match(match: dict) -> None:
    case = match["tfs_case"]
    with st.container(border=True):
        st.markdown(f"**TFS-{case['tfs_id']}** · {truncate_words(case['title'], 90)} · _{match['confidence']} confidence_")
        if case.get("root_cause"):
            st.markdown(f"**TFS reported root cause:** {case['root_cause']}")
        if case.get("resolution_text"):
            st.markdown(f"**TFS reported resolution:** {case['resolution_text']}")
        else:
            st.caption("TFS has no resolution text recorded for this case.")
        if match.get("recommended_action"):
            st.info(f"**ResolveIQ recommendation:** {match['recommended_action']}")
        if match["match_reasons"]:
            st.caption("Why relevant: " + "; ".join(match["match_reasons"]))
        meta_bits = [f"State: {case['state']}"]
        if case.get("target_release"):
            meta_bits.append(f"Release: {case['target_release']}")
        if case.get("crm_id"):
            meta_bits.append(f"CRM: {case['crm_id']}")
        st.caption(" · ".join(meta_bits))
        st.markdown(f"[Open in TFS →]({case['url']})  ·  Source: TFS")


def _render_wiki_match(match: dict) -> None:
    page = match["wiki_page"]
    with st.container(border=True):
        st.markdown(f"**{truncate_words(page['title'], 90)}** · _{match['confidence']} confidence_")
        if page.get("excerpt"):
            st.markdown(page["excerpt"])
        if match["match_reasons"]:
            st.caption("Why relevant: " + "; ".join(match["match_reasons"]))
        st.caption(f"Space: {page['space_key']}" if page.get("space_key") else "")
        st.markdown(f"[Open in Wiki →]({page['url']})  ·  Source: Landis+Gyr Wiki")


def render_external_knowledge(tfs_result: dict | None, wiki_result: dict | None) -> None:
    if tfs_result is None and wiki_result is None:
        return

    st.markdown("#### 🌐 External Knowledge")
    st.caption(
        "Live results from TFS and the Landis+Gyr Wiki, ranked against this investigation -- never imported "
        "or stored, fetched fresh (or from a short-lived cache) each time. TFS/Wiki remain the source of "
        "truth; use \"Open in TFS/Wiki\" to verify anything shown here."
    )

    if tfs_result is not None:
        st.markdown("**TFS**")
        if not tfs_result["available"]:
            st.warning(f"⚠️ {tfs_result['error'] or 'TFS was unavailable.'} Local ResolveIQ knowledge was used instead.")
        elif not tfs_result["matches"]:
            st.caption("No relevant historical issues found.")
        else:
            _render_result_status(tfs_result["matches"], noun="historical issue", cached=bool(tfs_result.get("from_cache")))
            for match in tfs_result["matches"]:
                _render_tfs_match(match)

    if wiki_result is not None:
        st.markdown("**Wiki**")
        if not wiki_result["available"]:
            st.warning(f"⚠️ {wiki_result['error'] or 'Wiki was unavailable.'} Local ResolveIQ knowledge was used instead.")
        elif not wiki_result["matches"]:
            st.caption("No relevant documentation found.")
        else:
            _render_result_status(wiki_result["matches"], noun="page", cached=bool(wiki_result.get("from_cache")))
            for match in wiki_result["matches"]:
                _render_wiki_match(match)


def _render_result_status(matches: list[dict], *, noun: str, cached: bool) -> None:
    """A green "relevant" checkmark overclaims when every match is
    Low confidence -- found live: 5 real TFS matches, all Low
    confidence (technology-only overlap), were shown as "5 relevant
    historical issue(s) found" with a success banner even though none
    of them addressed the investigation's actual problem. Only claim
    "relevant" when at least one match genuinely cleared the bar;
    otherwise say plainly that these share only the technology and
    need manual review."""
    cache_note = " (cached)" if cached else ""
    if any(m["confidence"] != "Low" for m in matches):
        st.success(f"✓ {len(matches)} relevant {noun}(s) found{cache_note}")
    else:
        st.caption(
            f"{len(matches)} {noun}(s) found with overlapping technology/keywords{cache_note}, but none closely "
            "match this investigation's specific problem -- review manually before relying on any of them."
        )
