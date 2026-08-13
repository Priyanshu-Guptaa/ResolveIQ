"""Generic parent/child specificity resolution (Context Dimensions
phase, RF-Mesh-vs-Mesh-IP classification fix, 2026-08-13).

Shared by any matcher that needs "when multiple candidates match the
same text, prefer the most specific one" over a real, already-governed
hierarchy -- ``Technology.parent_technology_id`` today, any future
governed hierarchy tomorrow (Protocol, Component sub-types, ...).

Pure data-in/data-out: no domain types, no hardcoded names, no
assumption about hierarchy depth. This module does not decide *whether*
two candidates match -- that's the caller's job (e.g. a title/body text
match) -- it only resolves *which of the already-matched candidates* to
prefer once more than one has.
"""

from __future__ import annotations


def most_specific(matched_ids: set[str], parent_of: dict[str, str | None]) -> set[str]:
    """Given a set of ids that all matched the same query, and a
    complete id -> parent_id map for the hierarchy they belong to
    (``parent_of`` may include ids outside ``matched_ids`` -- ancestors
    that didn't themselves match still need to be walked through to
    resolve a deeper hierarchy correctly), returns the subset that are
    not an ancestor of any other matched id.

    A no-op (returns ``matched_ids`` unchanged) whenever nothing in the
    hierarchy relates any two matched ids to each other -- correct
    behavior for a flat, non-hierarchical candidate set (nothing to
    prefer), and for a hierarchical set where the matches happen to be
    unrelated siblings (both are equally specific, both are kept; the
    caller's own tie-break, e.g. "first in a pre-sorted candidate
    list," decides between genuine ties -- this function only removes
    candidates that are strictly less specific than another real
    match, it never itself picks a single winner).

    Cycle-safe: a malformed ``parent_of`` map with a cycle simply stops
    walking once it revisits a node, rather than looping forever.
    """

    def ancestors(node_id: str) -> set[str]:
        seen: set[str] = set()
        current = parent_of.get(node_id)
        while current is not None and current not in seen:
            seen.add(current)
            current = parent_of.get(current)
        return seen

    ancestors_of_any_match: set[str] = set()
    for node_id in matched_ids:
        ancestors_of_any_match |= ancestors(node_id)

    return {node_id for node_id in matched_ids if node_id not in ancestors_of_any_match}
