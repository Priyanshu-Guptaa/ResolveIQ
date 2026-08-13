"""Repository for Product / Technology / Version / Customer / Region
(Sprint 3, Phase 3.3; Customer/Region added in the Context Dimensions
phase, 2026-08-12) -- grouped into one repository, same reasoning as
``KnowledgeRepository`` grouping known bugs/historical investigations/
documentation in Phase 3.1: small, related lookup entities that are
always seeded and queried together, not five near-identical files.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.lookup_entities import Customer, Product, Region, Technology, Version
from app.infrastructure.db.models import CustomerModel, ProductModel, RegionModel, TechnologyModel, VersionModel

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

    def delete_product(self, product_id: str) -> None:
        ...

    def save_technology(self, technology: Technology) -> None:
        ...

    def get_technology(self, technology_id: str) -> Technology | None:
        ...

    def get_technology_by_name(self, name: str) -> Technology | None:
        ...

    def list_technologies(self, *, active_only: bool = True) -> list[Technology]:
        ...

    def delete_technology(self, technology_id: str) -> None:
        ...

    def save_version(self, version: Version) -> None:
        ...

    def get_version(self, version_id: str) -> Version | None:
        ...

    def get_version_by_name(self, name: str) -> Version | None:
        ...

    def list_versions(self, *, active_only: bool = True) -> list[Version]:
        ...

    def delete_version(self, version_id: str) -> None:
        ...

    def save_customer(self, customer: Customer) -> None:
        ...

    def get_customer(self, customer_id: str) -> Customer | None:
        ...

    def get_customer_by_name(self, name: str) -> Customer | None:
        """Matches ``name`` or any of ``aliases`` -- see
        ``Customer.aliases``'s docstring."""
        ...

    def list_customers(self, *, active_only: bool = True) -> list[Customer]:
        ...

    def delete_customer(self, customer_id: str) -> None:
        ...

    def save_region(self, region: Region) -> None:
        ...

    def get_region(self, region_id: str) -> Region | None:
        ...

    def get_region_by_name(self, name: str) -> Region | None:
        ...

    def list_regions(self, *, active_only: bool = True) -> list[Region]:
        ...

    def delete_region(self, region_id: str) -> None:
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
            model.status = product.status.value
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

    def delete_product(self, product_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(ProductModel, product_id)
            if model is not None:
                session.delete(model)
                session.commit()

    # --- Technologies ----------------------------------------------------

    def save_technology(self, technology: Technology) -> None:
        with self._session_factory() as session:
            model = session.get(TechnologyModel, technology.id)
            if model is None:
                model = TechnologyModel(id=technology.id)
                session.add(model)
            model.name = technology.name
            model.parent_technology_id = technology.parent_technology_id
            model.created_at = technology.created_at
            model.updated_at = technology.updated_at
            model.created_by = technology.created_by
            model.updated_by = technology.updated_by
            model.is_active = technology.is_active
            model.status = technology.status.value
            session.commit()

    def get_technology(self, technology_id: str) -> Technology | None:
        with self._session_factory() as session:
            model = session.get(TechnologyModel, technology_id)
            return _technology_to_domain(model) if model is not None else None

    def get_technology_by_name(self, name: str) -> Technology | None:
        with self._session_factory() as session:
            model = session.query(TechnologyModel).filter(TechnologyModel.name == name).first()
            return _technology_to_domain(model) if model is not None else None

    def list_technologies(self, *, active_only: bool = True) -> list[Technology]:
        with self._session_factory() as session:
            query = session.query(TechnologyModel)
            if active_only:
                query = query.filter(TechnologyModel.is_active.is_(True))
            return [_technology_to_domain(m) for m in query.order_by(TechnologyModel.name).all()]

    def delete_technology(self, technology_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(TechnologyModel, technology_id)
            if model is not None:
                session.delete(model)
                session.commit()

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
            model.status = version.status.value
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

    def delete_version(self, version_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(VersionModel, version_id)
            if model is not None:
                session.delete(model)
                session.commit()

    # --- Customers (Context Dimensions phase) -------------------------------

    def save_customer(self, customer: Customer) -> None:
        with self._session_factory() as session:
            model = session.get(CustomerModel, customer.id)
            if model is None:
                model = CustomerModel(id=customer.id)
                session.add(model)
            model.name = customer.name
            model.aliases = customer.aliases
            model.verified = customer.verified
            model.source_type = customer.source_type
            model.source_reference = customer.source_reference
            model.created_at = customer.created_at
            model.updated_at = customer.updated_at
            model.created_by = customer.created_by
            model.updated_by = customer.updated_by
            model.is_active = customer.is_active
            model.status = customer.status.value
            session.commit()

    def get_customer(self, customer_id: str) -> Customer | None:
        with self._session_factory() as session:
            model = session.get(CustomerModel, customer_id)
            return _customer_to_domain(model) if model is not None else None

    def get_customer_by_name(self, name: str) -> Customer | None:
        """Matches ``name`` exactly first, then falls back to scanning
        ``aliases`` (a JSON column -- SQLite can't index into it, and
        the customer list is small enough that a Python-side scan over
        every row is fine at this scale; revisit if it ever isn't)."""
        needle = name.strip().lower()
        with self._session_factory() as session:
            model = session.query(CustomerModel).filter(CustomerModel.name == name).first()
            if model is not None:
                return _customer_to_domain(model)
            for model in session.query(CustomerModel).all():
                if any(alias.strip().lower() == needle for alias in (model.aliases or [])):
                    return _customer_to_domain(model)
        return None

    def list_customers(self, *, active_only: bool = True) -> list[Customer]:
        with self._session_factory() as session:
            query = session.query(CustomerModel)
            if active_only:
                query = query.filter(CustomerModel.is_active.is_(True))
            return [_customer_to_domain(m) for m in query.order_by(CustomerModel.name).all()]

    def delete_customer(self, customer_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(CustomerModel, customer_id)
            if model is not None:
                session.delete(model)
                session.commit()

    # --- Regions (Context Dimensions phase) ---------------------------------

    def save_region(self, region: Region) -> None:
        with self._session_factory() as session:
            model = session.get(RegionModel, region.id)
            if model is None:
                model = RegionModel(id=region.id)
                session.add(model)
            model.name = region.name
            model.aliases = region.aliases
            model.source_type = region.source_type
            model.source_reference = region.source_reference
            model.created_at = region.created_at
            model.updated_at = region.updated_at
            model.created_by = region.created_by
            model.updated_by = region.updated_by
            model.is_active = region.is_active
            model.status = region.status.value
            session.commit()

    def get_region(self, region_id: str) -> Region | None:
        with self._session_factory() as session:
            model = session.get(RegionModel, region_id)
            return _region_to_domain(model) if model is not None else None

    def get_region_by_name(self, name: str) -> Region | None:
        needle = name.strip().lower()
        with self._session_factory() as session:
            model = session.query(RegionModel).filter(RegionModel.name == name).first()
            if model is not None:
                return _region_to_domain(model)
            for model in session.query(RegionModel).all():
                if any(alias.strip().lower() == needle for alias in (model.aliases or [])):
                    return _region_to_domain(model)
        return None

    def list_regions(self, *, active_only: bool = True) -> list[Region]:
        with self._session_factory() as session:
            query = session.query(RegionModel)
            if active_only:
                query = query.filter(RegionModel.is_active.is_(True))
            return [_region_to_domain(m) for m in query.order_by(RegionModel.name).all()]

    def delete_region(self, region_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(RegionModel, region_id)
            if model is not None:
                session.delete(model)
                session.commit()


def _row_to_kwargs(model) -> dict:
    return {
        "id": model.id,
        "name": model.name,
        "created_at": model.created_at,
        "updated_at": model.updated_at,
        "created_by": model.created_by,
        "updated_by": model.updated_by,
        "is_active": model.is_active,
        "status": model.status,
    }


def _technology_to_domain(model: TechnologyModel) -> Technology:
    return Technology(parent_technology_id=model.parent_technology_id, **_row_to_kwargs(model))


def _version_to_domain(model: VersionModel) -> Version:
    return Version(product_id=model.product_id, **_row_to_kwargs(model))


def _customer_to_domain(model: CustomerModel) -> Customer:
    return Customer(
        aliases=model.aliases or [],
        verified=model.verified,
        source_type=model.source_type,
        source_reference=model.source_reference,
        **_row_to_kwargs(model),
    )


def _region_to_domain(model: RegionModel) -> Region:
    return Region(
        aliases=model.aliases or [],
        source_type=model.source_type,
        source_reference=model.source_reference,
        **_row_to_kwargs(model),
    )
