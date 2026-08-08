"""Knowledge Relationship Engine (Sprint 3, Phase 3.3) -- view/add/
remove/search/filter/validate relationships across all nine knowledge
object types, the Relationship Explorer, the Knowledge Health report,
and Impact Analysis.

Dispatches to each object type's *existing* repository (from Phase
3.1/3.2/3.3) rather than owning storage itself -- this engine's only
new persistence is the relationship edges themselves
(RelationshipRepository). Every object it can relate must already be
governed and readable through a repository; nothing here reads JSON or
introduces a second source of truth for anything.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from typing import TYPE_CHECKING

from app.domain.enums import DocumentStatus
from app.domain.knowledge_relationships import (
    ExplorerGroup,
    ExplorerView,
    ImpactAnalysis,
    KnowledgeHealthReport,
    KnowledgeObjectRef,
    KnowledgeObjectType,
    KnowledgeRelationship,
    RelationshipType,
    RelationshipValidationIssue,
    ResolvedRelationship,
)
from app.domain.lookup_entities import Product, Technology, Version
from app.domain.playbook import Playbook
from app.engines.knowledge_object_framework.adapters import build_adapters

if TYPE_CHECKING:
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.knowledge_repository import KnowledgeRepository
    from app.infrastructure.db.log_knowledge_repository import LogKnowledgeRepository
    from app.infrastructure.db.lookup_repository import LookupRepository
    from app.infrastructure.db.playbook_repository import PlaybookRepository
    from app.infrastructure.db.relationship_repository import RelationshipRepository
    from app.infrastructure.db.sql_template_repository import SqlTemplateRepository

logger = logging.getLogger(__name__)

# Display order for grouped views, matching the brief's explicit
# "Components -> Known Bugs -> SQL -> Playbooks -> Documents ->
# Historical Investigations" ordering, with the three new lookup types
# appended.
_GROUP_ORDER = [
    KnowledgeObjectType.COMPONENT,
    KnowledgeObjectType.KNOWN_BUG,
    KnowledgeObjectType.SQL_TEMPLATE,
    KnowledgeObjectType.PLAYBOOK,
    KnowledgeObjectType.DOCUMENT,
    KnowledgeObjectType.HISTORICAL_INVESTIGATION,
    KnowledgeObjectType.PRODUCT,
    KnowledgeObjectType.TECHNOLOGY,
    KnowledgeObjectType.VERSION,
]


class KnowledgeObjectNotFoundError(Exception):
    def __init__(self, object_type: KnowledgeObjectType, object_id: str) -> None:
        self.object_type = object_type
        self.object_id = object_id
        super().__init__(f"{object_type.value} not found: {object_id}")


class DuplicateRelationshipError(Exception):
    def __init__(self, existing: KnowledgeRelationship) -> None:
        self.existing = existing
        super().__init__(f"That relationship already exists ({existing.id})")


class KnowledgeRelationshipEngine:
    def __init__(
        self,
        relationship_repo: "RelationshipRepository",
        component_repo: "ComponentProfileRepository",
        knowledge_repo: "KnowledgeRepository",
        sql_repo: "SqlTemplateRepository",
        playbook_repo: "PlaybookRepository",
        lookup_repo: "LookupRepository",
        log_knowledge_repo: "LogKnowledgeRepository | None" = None,
    ) -> None:
        self._relationships = relationship_repo
        self._components = component_repo
        self._knowledge = knowledge_repo
        self._sql = sql_repo
        self._playbooks = playbook_repo
        self._lookups = lookup_repo
        # Phase 3.4: the per-type if/elif dispatch this engine used to
        # own directly is now the shared adapter registry -- see
        # app/engines/knowledge_object_framework/adapters.py's module
        # docstring. Behavior-preserving: same inputs/outputs as before,
        # this engine's own Phase 3.3 tests are unchanged by the move.
        # log_knowledge_repo (Log Intelligence follow-up) is optional and
        # additive: LogSourceApplication/LogCollectionScenario only join
        # the resolvable object types when it's supplied.
        self._adapters = build_adapters(
            component_repo, knowledge_repo, sql_repo, playbook_repo, lookup_repo, log_knowledge_repo
        )

    # --- Object resolution (dispatch via the shared adapter registry) ------

    def get_object(self, object_type: KnowledgeObjectType, object_id: str) -> KnowledgeObjectRef | None:
        adapter = self._adapters.get(object_type)
        if adapter is None:
            return None
        obj = adapter.get(object_id)
        return adapter.to_ref(obj) if obj is not None else None

    def _resolve_or_placeholder(self, object_type: KnowledgeObjectType, object_id: str) -> KnowledgeObjectRef:
        """Never silently drops a broken link -- a relationship whose
        target no longer resolves still needs to be visible (and
        removable) in the UI, not just reported once by validate()."""
        ref = self.get_object(object_type, object_id)
        if ref is not None:
            return ref
        return KnowledgeObjectRef(type=object_type, id=object_id, title="(missing)", subtitle="broken link")

    def list_objects(self, object_type: KnowledgeObjectType) -> list[KnowledgeObjectRef]:
        """Bounded -- every one of these tables is small (tens to low
        hundreds of rows at this platform's scale), so a full list plus
        Python-side filtering (see search_objects) is simpler and
        sufficient, the same reasoning already applied to Chroma's
        list_recent()."""
        adapter = self._adapters.get(object_type)
        if adapter is None:
            return []
        return [adapter.to_ref(obj) for obj in adapter.list_all()]

    def search_objects(
        self, query: str, *, object_type: KnowledgeObjectType | None = None, limit: int = 50
    ) -> list[KnowledgeObjectRef]:
        """Powers the Relationship Editor's searchable picker -- title/
        subtitle substring match, no free-text entry ever reaches a
        relationship: every result here is a real object with a real
        id."""
        types = [object_type] if object_type else list(KnowledgeObjectType)
        q = query.strip().lower()
        results: list[KnowledgeObjectRef] = []
        for t in types:
            for ref in self.list_objects(t):
                if not q or q in ref.title.lower() or q in ref.subtitle.lower():
                    results.append(ref)
                if len(results) >= limit:
                    return results
        return results

    # --- Minimal creation for the graph's newest node types -----------------
    # Full management (edit, archive, ...) for these is a later
    # Administration module (Playbook Management, Product/Technology/
    # Version Management) -- this phase only needs a way to give the
    # Relationship Editor's picker a real object to point at when one
    # doesn't exist yet.

    def create_playbook(
        self, title: str, *, product: str | None = None, description: str = "", steps: list[str] | None = None, created_by: str | None = None
    ) -> Playbook:
        playbook = Playbook(
            id=str(uuid.uuid4()),
            title=title,
            product=product,
            description=description,
            steps=steps or [],
            created_by=created_by,
            updated_by=created_by,
        )
        self._playbooks.save(playbook)
        return playbook

    def create_product(self, name: str, *, created_by: str | None = None) -> Product:
        product = Product(id=str(uuid.uuid4()), name=name, created_by=created_by, updated_by=created_by)
        self._lookups.save_product(product)
        return product

    def create_technology(self, name: str, *, created_by: str | None = None) -> Technology:
        technology = Technology(id=str(uuid.uuid4()), name=name, created_by=created_by, updated_by=created_by)
        self._lookups.save_technology(technology)
        return technology

    def create_version(self, name: str, *, product_id: str | None = None, created_by: str | None = None) -> Version:
        version = Version(id=str(uuid.uuid4()), name=name, product_id=product_id, created_by=created_by, updated_by=created_by)
        self._lookups.save_version(version)
        return version

    def list_playbooks(self) -> list[Playbook]:
        return self._playbooks.list_all()

    def list_products(self) -> list[Product]:
        return self._lookups.list_products()

    def list_technologies(self) -> list[Technology]:
        return self._lookups.list_technologies()

    def list_versions(self) -> list[Version]:
        return self._lookups.list_versions()

    # --- Relationship CRUD -----------------------------------------------

    def add_relationship(
        self,
        from_type: KnowledgeObjectType,
        from_id: str,
        to_type: KnowledgeObjectType,
        to_id: str,
        relationship_type: RelationshipType = RelationshipType.RELATED_TO,
        created_by: str | None = None,
    ) -> ResolvedRelationship:
        """"No orphan relationships": both ends must already exist as
        real objects before an edge is written -- this is the
        integrity guarantee a free-text field could never give."""
        from_ref = self.get_object(from_type, from_id)
        if from_ref is None:
            raise KnowledgeObjectNotFoundError(from_type, from_id)
        to_ref = self.get_object(to_type, to_id)
        if to_ref is None:
            raise KnowledgeObjectNotFoundError(to_type, to_id)

        existing = self._relationships.find_pair(from_type, from_id, to_type, to_id, relationship_type)
        if existing is not None:
            raise DuplicateRelationshipError(existing)

        relationship = self._relationships.add(from_type, from_id, to_type, to_id, relationship_type, created_by)
        return ResolvedRelationship(relationship=relationship, from_object=from_ref, to_object=to_ref)

    def remove_relationship(self, relationship_id: str) -> None:
        self._relationships.remove(relationship_id)

    def list_relationships(self, object_type: KnowledgeObjectType, object_id: str) -> list[ResolvedRelationship]:
        relationships = self._relationships.list_for_object(object_type, object_id)
        resolved = []
        for rel in relationships:
            resolved.append(
                ResolvedRelationship(
                    relationship=rel,
                    from_object=self._resolve_or_placeholder(rel.from_type, rel.from_id),
                    to_object=self._resolve_or_placeholder(rel.to_type, rel.to_id),
                )
            )
        return resolved

    # --- Explorer / Impact Analysis -----------------------------------------

    def get_explorer_view(self, object_type: KnowledgeObjectType, object_id: str) -> ExplorerView:
        """"A user selecting CommandProcessorHost should immediately
        see every connected object" -- grouped by the *other* object's
        type. Folds in Phase 3.1's Component-anchored relationships
        (known_bug_components etc.) alongside this phase's generic
        edges when centered on a Component, so this is a genuinely
        complete view, not just what's new since this phase -- those
        four legacy tables remain the source of truth for those
        specific links; nothing here duplicates or migrates them."""
        center = self.get_object(object_type, object_id)
        if center is None:
            raise KnowledgeObjectNotFoundError(object_type, object_id)

        grouped: dict[KnowledgeObjectType, list[KnowledgeObjectRef]] = defaultdict(list)
        seen: set[tuple[KnowledgeObjectType, str]] = set()

        def _add(ref: KnowledgeObjectRef) -> None:
            key = (ref.type, ref.id)
            if key not in seen:
                seen.add(key)
                grouped[ref.type].append(ref)

        for rel in self._relationships.list_for_object(object_type, object_id):
            is_from = rel.from_type == object_type and rel.from_id == object_id
            other_type, other_id = (rel.to_type, rel.to_id) if is_from else (rel.from_type, rel.from_id)
            _add(self._resolve_or_placeholder(other_type, other_id))

        if object_type == KnowledgeObjectType.COMPONENT and center is not None:
            component_name = center.title
            for bug in self._knowledge.list_known_bugs():
                if component_name in bug.related_components:
                    _add(
                        KnowledgeObjectRef(
                            type=KnowledgeObjectType.KNOWN_BUG, id=bug.id, title=bug.title, subtitle=bug.bug_status
                        )
                    )
            for template in self._sql.list_all():
                if component_name in template.related_components:
                    _add(
                        KnowledgeObjectRef(
                            type=KnowledgeObjectType.SQL_TEMPLATE, id=template.id, title=template.title, subtitle=template.category
                        )
                    )
            for investigation in self._knowledge.list_historical_investigations():
                if component_name in investigation.related_components:
                    _add(
                        KnowledgeObjectRef(
                            type=KnowledgeObjectType.HISTORICAL_INVESTIGATION,
                            id=investigation.id,
                            title=investigation.title,
                            subtitle=investigation.domain,
                        )
                    )
            for document in self._knowledge.list_all_documentation():
                if component_name in document.related_components:
                    _add(
                        KnowledgeObjectRef(
                            type=KnowledgeObjectType.DOCUMENT, id=document.id, title=document.title, subtitle=_status_value(document.status)
                        )
                    )

        groups = [ExplorerGroup(object_type=t, objects=refs) for t, refs in grouped.items()]
        groups.sort(key=lambda g: _GROUP_ORDER.index(g.object_type) if g.object_type in _GROUP_ORDER else 99)
        return ExplorerView(center=center, groups=groups)

    def get_impact_analysis(self, object_type: KnowledgeObjectType, object_id: str) -> ImpactAnalysis:
        """"Before deleting or modifying any knowledge object, show
        every downstream dependency" -- structurally the same question
        the Explorer answers ("what's connected to this"), just framed
        for a different moment (about to change something) rather than
        browsing, so it reuses the same computation instead of a
        second implementation."""
        view = self.get_explorer_view(object_type, object_id)
        total = sum(len(g.objects) for g in view.groups)
        return ImpactAnalysis(object=view.center, dependents=view.groups, total_dependents=total)

    # --- Validation ------------------------------------------------------

    def validate_relationships(self) -> list[RelationshipValidationIssue]:
        issues: list[RelationshipValidationIssue] = []
        all_relationships = self._relationships.list_all()
        seen_pairs: set[tuple[frozenset, RelationshipType]] = set()

        for rel in all_relationships:
            if self.get_object(rel.from_type, rel.from_id) is None:
                issues.append(
                    RelationshipValidationIssue(
                        issue_type="broken_link",
                        severity="error",
                        description=f"References a missing {rel.from_type.value} ({rel.from_id})",
                        relationship_id=rel.id,
                    )
                )
            if self.get_object(rel.to_type, rel.to_id) is None:
                issues.append(
                    RelationshipValidationIssue(
                        issue_type="broken_link",
                        severity="error",
                        description=f"References a missing {rel.to_type.value} ({rel.to_id})",
                        relationship_id=rel.id,
                    )
                )

            pair_key = (frozenset([(rel.from_type, rel.from_id), (rel.to_type, rel.to_id)]), rel.relationship_type)
            if pair_key in seen_pairs:
                issues.append(
                    RelationshipValidationIssue(
                        issue_type="duplicate",
                        severity="warning",
                        description=(
                            f"Duplicate {rel.relationship_type.value} relationship between "
                            f"{rel.from_type.value}:{rel.from_id} and {rel.to_type.value}:{rel.to_id}"
                        ),
                        relationship_id=rel.id,
                    )
                )
            else:
                seen_pairs.add(pair_key)

        issues.extend(self._detect_cycles(all_relationships))
        return issues

    def _detect_cycles(self, relationships: list[KnowledgeRelationship]) -> list[RelationshipValidationIssue]:
        """Standard DFS white/gray/black cycle detection over the
        stored edge direction. Cheap at this graph's scale (dozens to
        low hundreds of edges) -- O(V+E), no recursion-depth concern
        for a graph this small."""
        graph: dict[tuple, list[tuple]] = defaultdict(list)
        for rel in relationships:
            graph[(rel.from_type, rel.from_id)].append((rel.to_type, rel.to_id))

        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[tuple, int] = defaultdict(int)
        issues: list[RelationshipValidationIssue] = []
        found_cycle_keys: set[frozenset] = set()

        def dfs(node: tuple, path: list[tuple]) -> None:
            color[node] = GRAY
            path.append(node)
            for neighbor in graph.get(node, []):
                if color[neighbor] == GRAY:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycle_key = frozenset(cycle)
                    if cycle_key not in found_cycle_keys:
                        found_cycle_keys.add(cycle_key)
                        description = " -> ".join(f"{t.value}:{i}" for t, i in cycle)
                        issues.append(
                            RelationshipValidationIssue(
                                issue_type="circular_reference", severity="error", description=f"Circular reference: {description}"
                            )
                        )
                elif color[neighbor] == WHITE:
                    dfs(neighbor, path)
            path.pop()
            color[node] = BLACK

        for node in list(graph.keys()):
            if color[node] == WHITE:
                dfs(node, [])
        return issues

    # --- Knowledge Health --------------------------------------------------

    def get_health_report(self) -> KnowledgeHealthReport:
        components = self._components.list_all()
        documents = self._knowledge.list_all_documentation(status=DocumentStatus.PUBLISHED.value)
        sql_templates = self._sql.list_all()
        all_relationships = self._relationships.list_all()
        validation_issues = self.validate_relationships()

        name_to_id = {c.name: c.id for c in components}

        # component_id -> {object_type -> set(connected object ids)},
        # combining this phase's generic edges with Phase 3.1's legacy
        # Component-anchored links (matched by name -> id).
        connections: dict[str, dict[KnowledgeObjectType, set[str]]] = defaultdict(lambda: defaultdict(set))
        document_users: set[str] = set()
        sql_users: set[str] = set()

        for rel in all_relationships:
            pairs = [(rel.from_type, rel.from_id, rel.to_type, rel.to_id), (rel.to_type, rel.to_id, rel.from_type, rel.from_id)]
            for this_type, this_id, other_type, other_id in pairs:
                if this_type == KnowledgeObjectType.COMPONENT:
                    connections[this_id][other_type].add(other_id)
                    if other_type == KnowledgeObjectType.DOCUMENT:
                        document_users.add(other_id)
                    if other_type == KnowledgeObjectType.SQL_TEMPLATE:
                        sql_users.add(other_id)

        for bug in self._knowledge.list_known_bugs():
            for name in bug.related_components:
                if name in name_to_id:
                    connections[name_to_id[name]][KnowledgeObjectType.KNOWN_BUG].add(bug.id)
        for template in sql_templates:
            for name in template.related_components:
                if name in name_to_id:
                    connections[name_to_id[name]][KnowledgeObjectType.SQL_TEMPLATE].add(template.id)
                    sql_users.add(template.id)
        for investigation in self._knowledge.list_historical_investigations():
            for name in investigation.related_components:
                if name in name_to_id:
                    connections[name_to_id[name]][KnowledgeObjectType.HISTORICAL_INVESTIGATION].add(investigation.id)
        for document in documents:
            for name in document.related_components:
                if name in name_to_id:
                    connections[name_to_id[name]][KnowledgeObjectType.DOCUMENT].add(document.id)
                    document_users.add(document.id)

        unused_documents = [
            KnowledgeObjectRef(type=KnowledgeObjectType.DOCUMENT, id=d.id, title=d.title, subtitle=_status_value(d.status))
            for d in documents
            if d.id not in document_users
        ]
        unused_sql_templates = [
            KnowledgeObjectRef(type=KnowledgeObjectType.SQL_TEMPLATE, id=t.id, title=t.title, subtitle=t.category)
            for t in sql_templates
            if t.id not in sql_users
        ]

        connected_component_count = 0
        unused_components: list[KnowledgeObjectRef] = []
        components_missing_playbooks: list[KnowledgeObjectRef] = []
        components_missing_documentation: list[KnowledgeObjectRef] = []
        components_missing_historical_investigations: list[KnowledgeObjectRef] = []

        for component in components:
            comp_connections = connections.get(component.id, {})
            has_any_relationship = any(comp_connections.values())
            has_self_referential = bool(component.related_components or component.dependencies)
            ref = KnowledgeObjectRef(
                type=KnowledgeObjectType.COMPONENT, id=component.id, title=component.name, subtitle=component.product
            )

            if has_any_relationship or has_self_referential:
                connected_component_count += 1
            else:
                unused_components.append(ref)

            if not comp_connections.get(KnowledgeObjectType.PLAYBOOK):
                components_missing_playbooks.append(ref)
            if not comp_connections.get(KnowledgeObjectType.DOCUMENT):
                components_missing_documentation.append(ref)
            if not comp_connections.get(KnowledgeObjectType.HISTORICAL_INVESTIGATION):
                components_missing_historical_investigations.append(ref)

        coverage = (connected_component_count / len(components)) if components else 0.0

        return KnowledgeHealthReport(
            total_relationships=len(all_relationships),
            broken_relationships=[i for i in validation_issues if i.issue_type == "broken_link"],
            duplicate_relationships=[i for i in validation_issues if i.issue_type == "duplicate"],
            circular_references=[i for i in validation_issues if i.issue_type == "circular_reference"],
            unused_documents=unused_documents,
            unused_sql_templates=unused_sql_templates,
            unused_components=unused_components,
            components_missing_playbooks=components_missing_playbooks,
            components_missing_documentation=components_missing_documentation,
            components_missing_historical_investigations=components_missing_historical_investigations,
            component_count=len(components),
            component_relationship_coverage=coverage,
        )


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)
