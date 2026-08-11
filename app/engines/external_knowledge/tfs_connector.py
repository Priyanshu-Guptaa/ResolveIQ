"""TFS connector interface -- the boundary the rest of ResolveIQ talks
to, never the REST implementation directly (dependency injection, same
discipline as every other engine in this codebase: e.g.
``LogKnowledgeRepository`` is a Protocol with one real SQLAlchemy
implementation and fakes in tests). Lets a future provider (Azure
DevOps Services cloud, a different tracker entirely) implement the same
three methods without anything else in ResolveIQ changing.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.external_knowledge import TfsCase


class TfsConnector(Protocol):
    def is_configured(self) -> bool:
        """False when tfs_base_url/tfs_project aren't set -- callers
        must treat this the same as "unavailable", not an error."""
        ...

    def search_work_items(
        self, *, anchor_terms: list[str], supporting_terms: list[str], work_item_types: list[str], max_results: int
    ) -> list[TfsCase]:
        """Live query -- never returns cached/stale data itself (the
        caller, ``service.py``, owns caching). Raises on network/auth
        failure; callers must catch and degrade gracefully, never let
        this take down the rest of an Analyze request.

        ``anchor_terms`` (component/technology) scope the query when
        present -- a real candidate must match at least one;
        ``supporting_terms`` (title keywords, entities) broaden the
        query only when there's no anchor to scope by at all. Keeping
        these separate (not one flat OR'd bag) is a real, live-observed
        fix: OR-ing every term equally let generic title words flood a
        recency-ordered candidate window with noise, burying genuinely
        relevant matches under unrelated tickets that merely shared one
        common word."""
        ...

    def get_work_item(self, tfs_id: int) -> TfsCase | None:
        """One work item by id, full detail -- for a "view full case"
        UI action independent of search, e.g. a link followed back
        from a cached match."""
        ...

    def get_work_item_history(self, tfs_id: int) -> str | None:
        """Raw (HTML) System.History field for one work item -- kept
        separate from ``get_work_item`` since it's a materially
        larger payload most callers don't need."""
        ...
