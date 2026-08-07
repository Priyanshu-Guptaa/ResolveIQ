"""The reusable dispatch mechanism at the heart of the Knowledge Object
Framework (Sprint 3, Phase 3.4).

``KnowledgeRelationshipEngine`` (Phase 3.3) already had to solve "given
a ``KnowledgeObjectType``, which repository/method resolves it" once,
as an if/elif chain private to itself. Phase 3.4 needs that exact same
dispatch for create/edit/archive/restore/delete/history/validate too --
duplicating the if/elif a second time is precisely what "avoid
duplicated repository patterns" rules out. ``build_adapters`` is that
dispatch, extracted once and shared: ``KnowledgeRelationshipEngine`` was
refactored to use it (behavior-preserving -- same inputs/outputs, its
own Phase 3.3 tests are unchanged), and ``KnowledgeObjectService``
(this phase) is its second, new caller.

Each adapter is a thin, uniform Protocol-shaped wrapper around a
repository that already exists (from Phase 3.1/3.2/3.3) -- nothing here
introduces a new persistence mechanism or bypasses the existing
repositories. It exists purely to paper over their genuinely different
method names (``save`` vs ``save_known_bug`` vs ``save_documentation``
vs ``save_product``, ...), which is the actual duplication this module
avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

from app.domain.evidence import DocumentationRecord, HistoricalInvestigationRecord, KnownBugRecord
from app.domain.knowledge_relationships import KnowledgeObjectRef, KnowledgeObjectType
from app.domain.lookup_entities import Product, Technology, Version
from app.domain.playbook import Playbook
from app.domain.product_intelligence import ComponentProfile
from app.domain.sql_studio import QueryTemplate

if TYPE_CHECKING:
    from app.infrastructure.db.component_repository import ComponentProfileRepository
    from app.infrastructure.db.knowledge_repository import KnowledgeRepository
    from app.infrastructure.db.lookup_repository import LookupRepository
    from app.infrastructure.db.playbook_repository import PlaybookRepository
    from app.infrastructure.db.sql_template_repository import SqlTemplateRepository


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


@dataclass
class KnowledgeObjectAdapter:
    """Everything a generic caller needs for one object type, without
    needing to know that type's underlying repository's actual method
    names."""

    get: Callable[[str], BaseModel | None]
    list_all: Callable[[], list[BaseModel]]
    save: Callable[[BaseModel], None]
    delete: Callable[[str], None]
    to_ref: Callable[[BaseModel], KnowledgeObjectRef]
    model_cls: type[BaseModel]
    """The Pydantic class ``KnowledgeObjectService.create`` instantiates
    for this type -- Pydantic's own validation enforces whatever fields
    that type requires; the framework doesn't hardcode a per-type
    required-field list anywhere."""


def build_adapters(
    component_repo: "ComponentProfileRepository",
    knowledge_repo: "KnowledgeRepository",
    sql_repo: "SqlTemplateRepository",
    playbook_repo: "PlaybookRepository",
    lookup_repo: "LookupRepository",
) -> dict[KnowledgeObjectType, KnowledgeObjectAdapter]:
    T = KnowledgeObjectType
    return {
        T.COMPONENT: KnowledgeObjectAdapter(
            get=component_repo.get,
            list_all=component_repo.list_all,
            save=component_repo.save,
            delete=component_repo.delete,
            to_ref=lambda o: KnowledgeObjectRef(type=T.COMPONENT, id=o.id, title=o.name, subtitle=o.product),
            model_cls=ComponentProfile,
        ),
        T.DOCUMENT: KnowledgeObjectAdapter(
            get=knowledge_repo.get_documentation,
            list_all=lambda: knowledge_repo.list_documentation_summaries(page_size=1000).items,
            save=knowledge_repo.save_documentation,
            delete=knowledge_repo.delete_documentation,
            to_ref=lambda o: KnowledgeObjectRef(type=T.DOCUMENT, id=o.id, title=o.title, subtitle=_status_value(o.status)),
            model_cls=DocumentationRecord,
        ),
        T.KNOWN_BUG: KnowledgeObjectAdapter(
            get=knowledge_repo.get_known_bug,
            list_all=knowledge_repo.list_known_bugs,
            save=knowledge_repo.save_known_bug,
            delete=knowledge_repo.delete_known_bug,
            to_ref=lambda o: KnowledgeObjectRef(type=T.KNOWN_BUG, id=o.id, title=o.title, subtitle=o.bug_status),
            model_cls=KnownBugRecord,
        ),
        T.SQL_TEMPLATE: KnowledgeObjectAdapter(
            get=sql_repo.get,
            list_all=sql_repo.list_all,
            save=sql_repo.save,
            delete=sql_repo.delete,
            to_ref=lambda o: KnowledgeObjectRef(type=T.SQL_TEMPLATE, id=o.id, title=o.title, subtitle=o.category),
            model_cls=QueryTemplate,
        ),
        T.HISTORICAL_INVESTIGATION: KnowledgeObjectAdapter(
            get=knowledge_repo.get_historical_investigation,
            list_all=knowledge_repo.list_historical_investigations,
            save=knowledge_repo.save_historical_investigation,
            delete=knowledge_repo.delete_historical_investigation,
            to_ref=lambda o: KnowledgeObjectRef(
                type=T.HISTORICAL_INVESTIGATION, id=o.id, title=o.title, subtitle=o.domain
            ),
            model_cls=HistoricalInvestigationRecord,
        ),
        T.PLAYBOOK: KnowledgeObjectAdapter(
            get=playbook_repo.get,
            list_all=playbook_repo.list_all,
            save=playbook_repo.save,
            delete=playbook_repo.delete,
            to_ref=lambda o: KnowledgeObjectRef(type=T.PLAYBOOK, id=o.id, title=o.title, subtitle=o.product or ""),
            model_cls=Playbook,
        ),
        T.PRODUCT: KnowledgeObjectAdapter(
            get=lookup_repo.get_product,
            list_all=lookup_repo.list_products,
            save=lookup_repo.save_product,
            delete=lookup_repo.delete_product,
            to_ref=lambda o: KnowledgeObjectRef(type=T.PRODUCT, id=o.id, title=o.name),
            model_cls=Product,
        ),
        T.TECHNOLOGY: KnowledgeObjectAdapter(
            get=lookup_repo.get_technology,
            list_all=lookup_repo.list_technologies,
            save=lookup_repo.save_technology,
            delete=lookup_repo.delete_technology,
            to_ref=lambda o: KnowledgeObjectRef(type=T.TECHNOLOGY, id=o.id, title=o.name),
            model_cls=Technology,
        ),
        T.VERSION: KnowledgeObjectAdapter(
            get=lookup_repo.get_version,
            list_all=lookup_repo.list_versions,
            save=lookup_repo.save_version,
            delete=lookup_repo.delete_version,
            to_ref=lambda o: KnowledgeObjectRef(type=T.VERSION, id=o.id, title=o.name),
            model_cls=Version,
        ),
    }
