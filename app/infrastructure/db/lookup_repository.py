"""Repository for Product / Technology / Version (Sprint 3, Phase 3.3)
-- grouped into one repository, same reasoning as ``KnowledgeRepository``
grouping known bugs/historical investigations/documentation in Phase
3.1: three small, related lookup entities that are always seeded and
queried together, not three near-identical files.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.lookup_entities import Product, Technology, Version
from app.infrastructure.db.models import ProductModel, TechnologyModel, VersionModel

logger = logging.getLogger(__name__)


class LookupRepository(Protocol):
    def save_product(self, product: Product) -> None:
        ...

    def get_product(self, product_id: str) -> Product | None:
        ...

    def get_product_by_name(self, name: str) -> Product | None:
        ...

    def list_products(self, *, active_only: bool = True) -> list[Product]:
        ...

    def save_technology(self, technology: Technology) -> None:
        ...

    def get_technology(self, technology_id: str) -> Technology | None:
        ...

    def get_technology_by_name(self, name: str) -> Technology | None:
        ...

    def list_technologies(self, *, active_only: bool = True) -> list[Technology]:
        ...

    def save_version(self, version: Version) -> None:
        ...

    def get_version(self, version_id: str) -> Version | None:
        ...

    def get_version_by_name(self, name: str) -> Version | None:
        ...

    def list_versions(self, *, active_only: bool = True) -> list[Version]:
        ...


class SqlAlchemyLookupRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    # --- Products ----------------------------------------------------------

    def save_product(self, product: Product) -> None:
        with self._session_factory() as session:
            model = session.get(ProductModel, product.id)
            if model is None:
                model = ProductModel(id=product.id)
                session.add(model)
            model.name = product.name
            model.created_at = product.created_at
            model.updated_at = product.updated_at
            model.created_by = product.created_by
            model.updated_by = product.updated_by
            model.is_active = product.is_active
            session.commit()

    def get_product(self, product_id: str) -> Product | None:
        with self._session_factory() as session:
            model = session.get(ProductModel, product_id)
            return Product(**_row_to_kwargs(model)) if model is not None else None

    def get_product_by_name(self, name: str) -> Product | None:
        with self._session_factory() as session:
            model = session.query(ProductModel).filter(ProductModel.name == name).first()
            return Product(**_row_to_kwargs(model)) if model is not None else None

    def list_products(self, *, active_only: bool = True) -> list[Product]:
        with self._session_factory() as session:
            query = session.query(ProductModel)
            if active_only:
                query = query.filter(ProductModel.is_active.is_(True))
            return [Product(**_row_to_kwargs(m)) for m in query.order_by(ProductModel.name).all()]

    # --- Technologies ----------------------------------------------------

    def save_technology(self, technology: Technology) -> None:
        with self._session_factory() as session:
            model = session.get(TechnologyModel, technology.id)
            if model is None:
                model = TechnologyModel(id=technology.id)
                session.add(model)
            model.name = technology.name
            model.created_at = technology.created_at
            model.updated_at = technology.updated_at
            model.created_by = technology.created_by
            model.updated_by = technology.updated_by
            model.is_active = technology.is_active
            session.commit()

    def get_technology(self, technology_id: str) -> Technology | None:
        with self._session_factory() as session:
            model = session.get(TechnologyModel, technology_id)
            return Technology(**_row_to_kwargs(model)) if model is not None else None

    def get_technology_by_name(self, name: str) -> Technology | None:
        with self._session_factory() as session:
            model = session.query(TechnologyModel).filter(TechnologyModel.name == name).first()
            return Technology(**_row_to_kwargs(model)) if model is not None else None

    def list_technologies(self, *, active_only: bool = True) -> list[Technology]:
        with self._session_factory() as session:
            query = session.query(TechnologyModel)
            if active_only:
                query = query.filter(TechnologyModel.is_active.is_(True))
            return [Technology(**_row_to_kwargs(m)) for m in query.order_by(TechnologyModel.name).all()]

    # --- Versions ----------------------------------------------------------

    def save_version(self, version: Version) -> None:
        with self._session_factory() as session:
            model = session.get(VersionModel, version.id)
            if model is None:
                model = VersionModel(id=version.id)
                session.add(model)
            model.name = version.name
            model.product_id = version.product_id
            model.created_at = version.created_at
            model.updated_at = version.updated_at
            model.created_by = version.created_by
            model.updated_by = version.updated_by
            model.is_active = version.is_active
            session.commit()

    def get_version(self, version_id: str) -> Version | None:
        with self._session_factory() as session:
            model = session.get(VersionModel, version_id)
            return _version_to_domain(model) if model is not None else None

    def get_version_by_name(self, name: str) -> Version | None:
        with self._session_factory() as session:
            model = session.query(VersionModel).filter(VersionModel.name == name).first()
            return _version_to_domain(model) if model is not None else None

    def list_versions(self, *, active_only: bool = True) -> list[Version]:
        with self._session_factory() as session:
            query = session.query(VersionModel)
            if active_only:
                query = query.filter(VersionModel.is_active.is_(True))
            return [_version_to_domain(m) for m in query.order_by(VersionModel.name).all()]


def _row_to_kwargs(model) -> dict:
    return {
        "id": model.id,
        "name": model.name,
        "created_at": model.created_at,
        "updated_at": model.updated_at,
        "created_by": model.created_by,
        "updated_by": model.updated_by,
        "is_active": model.is_active,
    }


def _version_to_domain(model: VersionModel) -> Version:
    return Version(product_id=model.product_id, **_row_to_kwargs(model))
