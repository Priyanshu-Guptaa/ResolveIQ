"""SQL Library Engine: thin facade over the SQL Template repository.

Same pattern as ProductIntelligenceEngine/ComponentRegistry -- kept as a
separate engine class (not the router depending on the repository
directly) so SQL Studio follows the same UI -> API -> Engine ->
Repository -> SQLite layering as every other module, per Sprint 3 Phase
3.1's explicit requirement. Replaces the ``QUERY_LIBRARY`` Python
constant as the runtime source; ``QUERY_LIBRARY`` itself now only feeds
the one-time migration (``seed_migration.py``).
"""

from __future__ import annotations

from app.domain.sql_studio import QueryTemplate
from app.infrastructure.db.sql_template_repository import SqlTemplateRepository


class SqlLibraryEngine:
    def __init__(self, repository: SqlTemplateRepository) -> None:
        self._repository = repository

    def list_templates(self) -> list[QueryTemplate]:
        return self._repository.list_all()

    def get_template(self, template_id: str) -> QueryTemplate | None:
        return self._repository.get(template_id)
