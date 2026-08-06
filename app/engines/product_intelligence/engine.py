"""Product Intelligence Engine: thin facade over the Component Registry.

Same pattern as IngestionEngine/FileTypeRegistry -- kept as a separate
engine class (not just the registry directly) for consistency with every
other engine's DI/testability shape, even though it's a pass-through
today. Recommendation Engine V2 (later phase) is the intended second
caller of this engine, alongside the Workspace UI.
"""

from __future__ import annotations

from app.domain.product_intelligence import ComponentProfile
from app.engines.product_intelligence.component_registry import ComponentRegistry


class ProductIntelligenceEngine:
    def __init__(self, registry: ComponentRegistry) -> None:
        self._registry = registry

    def list_components(self) -> list[ComponentProfile]:
        return self._registry.list_all()

    def get_component(self, name: str) -> ComponentProfile | None:
        return self._registry.get(name)

    def search_components(self, query: str) -> list[ComponentProfile]:
        return self._registry.search(query)
