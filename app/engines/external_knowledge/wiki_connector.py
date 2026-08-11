"""Wiki connector interface -- same boundary discipline as
``tfs_connector.py``: the rest of ResolveIQ depends on this Protocol,
never on the Confluence REST implementation directly.
"""

from __future__ import annotations

from typing import Protocol

from app.domain.external_knowledge import WikiPage


class WikiConnector(Protocol):
    def is_configured(self) -> bool:
        """False when wiki_base_url/wiki_api_token aren't set --
        callers must treat this the same as "unavailable"."""
        ...

    def search(self, *, anchor_terms: list[str], supporting_terms: list[str], max_results: int) -> list[WikiPage]:
        """Live query. Raises on network/auth failure; callers must
        catch and degrade gracefully. See ``TfsConnector.search_work_items``'s
        docstring for why anchor/supporting terms are kept separate
        rather than one flat OR'd bag."""
        ...

    def get_page(self, page_id: str) -> WikiPage | None:
        """One page by id, full body -- for a "view full page" UI
        action independent of search."""
        ...
