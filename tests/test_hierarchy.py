"""Unit tests for the generic specificity resolver (app.engines.shared.
hierarchy) -- the RF-Mesh-vs-Mesh-IP classification fix, 2026-08-13.
Pure, no domain types, no hardcoded names -- covers hierarchies of
several shapes to prove genericity.
"""

from __future__ import annotations

from app.engines.shared.hierarchy import most_specific


def test_child_wins_over_matched_parent():
    parent_of = {"rf-mesh": None, "rf-mesh-ip": "rf-mesh"}
    result = most_specific({"rf-mesh", "rf-mesh-ip"}, parent_of)
    assert result == {"rf-mesh-ip"}


def test_no_hierarchy_relationship_is_a_noop():
    """Two matched, unrelated ids (no parent/child relationship at
    all) -- nothing to prefer, both survive."""
    parent_of = {"wi-sun": None, "kafka-broker": None}
    result = most_specific({"wi-sun", "kafka-broker"}, parent_of)
    assert result == {"wi-sun", "kafka-broker"}


def test_single_match_is_unaffected():
    parent_of = {"rf-mesh": None, "rf-mesh-ip": "rf-mesh"}
    result = most_specific({"rf-mesh"}, parent_of)
    assert result == {"rf-mesh"}


def test_grandchild_wins_over_grandparent_and_parent_arbitrary_depth():
    """Generic, hierarchy-depth-agnostic -- proves this is not special-
    cased to one level (the real RF Mesh hierarchy is only one level
    deep, but the algorithm must not assume that)."""
    parent_of = {"a": None, "b": "a", "c": "b"}
    result = most_specific({"a", "b", "c"}, parent_of)
    assert result == {"c"}


def test_deep_ancestor_not_itself_matched_is_still_resolved_correctly():
    """A hierarchy deeper than what actually matched -- the unmatched
    middle node must still be walked through to resolve the real
    relationship between the two matched (grandparent/grandchild) ids."""
    parent_of = {"a": None, "b": "a", "c": "b"}
    result = most_specific({"a", "c"}, parent_of)
    assert result == {"c"}


def test_sibling_technologies_both_survive_specificity_filter():
    """RF Mesh IP and RF Mesh (DAS implementation) are siblings (both
    children of RF Mesh) -- if somehow both matched, neither is an
    ancestor of the other, so both survive; picking between them is the
    caller's tie-break, not this function's job."""
    parent_of = {"rf-mesh": None, "rf-mesh-ip": "rf-mesh", "rf-mesh-das": "rf-mesh"}
    result = most_specific({"rf-mesh-ip", "rf-mesh-das"}, parent_of)
    assert result == {"rf-mesh-ip", "rf-mesh-das"}


def test_future_hierarchy_needs_no_code_change():
    """A hypothetical, entirely different future hierarchy (e.g. a
    Protocol dimension) works identically -- proves genericity, no
    RF-Mesh-specific logic anywhere in this module."""
    parent_of = {"dlms": None, "dlms-cosem": "dlms", "dlms-cosem-hdlc": "dlms-cosem"}
    result = most_specific({"dlms", "dlms-cosem", "dlms-cosem-hdlc"}, parent_of)
    assert result == {"dlms-cosem-hdlc"}


def test_empty_input():
    assert most_specific(set(), {}) == set()


def test_cycle_does_not_infinite_loop():
    """Defensive: a malformed parent_of map with a cycle must not hang."""
    parent_of = {"a": "b", "b": "a"}
    result = most_specific({"a", "b"}, parent_of)
    assert result == set()  # both are "ancestors" of each other in a cycle -- neither is more specific
