"""Repository for the Log Intelligence Knowledge Base (LogSourceApplication
+ LogCollectionScenario). Same Protocol + SQLAlchemy-implementation shape
every other governed entity in this app uses.
"""

from __future__ import annotations

import logging
from typing import Protocol

from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.domain.log_intelligence_kb import (
    LogCollectionScenario,
    LogCollectionStep,
    LogRepositoryLocation,
    LogSourceApplication,
)
from app.infrastructure.db.models import LogCollectionScenarioModel, LogSourceApplicationModel

logger = logging.getLogger(__name__)


class LogKnowledgeRepository(Protocol):
    def save_log_source(self, source: LogSourceApplication) -> None:
        ...

    def get_log_source(self, source_id: str) -> LogSourceApplication | None:
        ...

    def find_log_source_by_name(self, name: str) -> LogSourceApplication | None:
        ...

    def list_log_sources(self, *, active_only: bool = True) -> list[LogSourceApplication]:
        ...

    def delete_log_source(self, source_id: str) -> None:
        ...

    def save_scenario(self, scenario: LogCollectionScenario) -> None:
        ...

    def get_scenario(self, scenario_id: str) -> LogCollectionScenario | None:
        ...

    def find_scenario(
        self, *, product: str, technology: str, scenario_type: str, region: str | None
    ) -> LogCollectionScenario | None:
        ...

    def list_scenarios(self, *, active_only: bool = True) -> list[LogCollectionScenario]:
        ...

    def delete_scenario(self, scenario_id: str) -> None:
        ...


class SqlAlchemyLogKnowledgeRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    # --- LogSourceApplication --------------------------------------------

    def save_log_source(self, source: LogSourceApplication) -> None:
        with self._session_factory() as session:
            model = session.get(LogSourceApplicationModel, source.id)
            if model is None:
                model = LogSourceApplicationModel(id=source.id)
                session.add(model)
            model.name = source.name
            model.location_platform = source.location.platform
            model.location_root_path = source.location.root_path
            model.location_subdirectory = source.location.subdirectory
            model.location_filename_patterns = source.location.filename_patterns
            model.location_raw_paths = source.location.raw_paths
            model.technology = source.technology
            model.product = source.product
            model.purpose = source.purpose
            model.log_level_support = source.log_level_support
            model.typical_issues = source.typical_issues
            model.common_errors = source.common_errors
            model.related_sql = source.related_sql
            model.related_documentation = source.related_documentation
            model.related_known_bugs = source.related_known_bugs
            model.related_playbooks = source.related_playbooks
            model.notes = source.notes
            model.source_wiki_pages = source.source_wiki_pages
            model.created_at = source.created_at
            model.updated_at = source.updated_at
            model.created_by = source.created_by
            model.updated_by = source.updated_by
            model.is_active = source.is_active
            model.status = source.status.value
            session.commit()
            logger.debug("Saved log source application %s (%s)", source.id, source.name)

    def get_log_source(self, source_id: str) -> LogSourceApplication | None:
        with self._session_factory() as session:
            model = session.get(LogSourceApplicationModel, source_id)
            return _log_source_to_domain(model) if model is not None else None

    def find_log_source_by_name(self, name: str) -> LogSourceApplication | None:
        with self._session_factory() as session:
            model = session.query(LogSourceApplicationModel).filter(LogSourceApplicationModel.name == name).first()
            return _log_source_to_domain(model) if model is not None else None

    def list_log_sources(self, *, active_only: bool = True) -> list[LogSourceApplication]:
        with self._session_factory() as session:
            query = session.query(LogSourceApplicationModel)
            if active_only:
                query = query.filter(LogSourceApplicationModel.is_active.is_(True))
            return [_log_source_to_domain(m) for m in query.order_by(LogSourceApplicationModel.name).all()]

    def delete_log_source(self, source_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(LogSourceApplicationModel, source_id)
            if model is not None:
                session.delete(model)
                session.commit()

    # --- LogCollectionScenario ---------------------------------------------

    def save_scenario(self, scenario: LogCollectionScenario) -> None:
        with self._session_factory() as session:
            model = session.get(LogCollectionScenarioModel, scenario.id)
            if model is None:
                model = LogCollectionScenarioModel(id=scenario.id)
                session.add(model)
            model.product = scenario.product
            model.technology = scenario.technology
            model.version = scenario.version
            model.scenario_type = scenario.scenario_type
            model.region = scenario.region
            model.steps = [step.model_dump(mode="json") for step in scenario.steps]
            model.notes = scenario.notes
            model.source_wiki_page = scenario.source_wiki_page
            model.created_at = scenario.created_at
            model.updated_at = scenario.updated_at
            model.created_by = scenario.created_by
            model.updated_by = scenario.updated_by
            model.is_active = scenario.is_active
            model.status = scenario.status.value
            session.commit()
            logger.debug(
                "Saved log collection scenario %s (%s / %s / %s)",
                scenario.id,
                scenario.product,
                scenario.technology,
                scenario.scenario_type,
            )

    def get_scenario(self, scenario_id: str) -> LogCollectionScenario | None:
        with self._session_factory() as session:
            model = session.get(LogCollectionScenarioModel, scenario_id)
            return _scenario_to_domain(model) if model is not None else None

    def find_scenario(
        self, *, product: str, technology: str, scenario_type: str, region: str | None
    ) -> LogCollectionScenario | None:
        with self._session_factory() as session:
            model = (
                session.query(LogCollectionScenarioModel)
                .filter(
                    LogCollectionScenarioModel.product == product,
                    LogCollectionScenarioModel.technology == technology,
                    LogCollectionScenarioModel.scenario_type == scenario_type,
                    LogCollectionScenarioModel.region == region,
                )
                .first()
            )
            return _scenario_to_domain(model) if model is not None else None

    def list_scenarios(self, *, active_only: bool = True) -> list[LogCollectionScenario]:
        with self._session_factory() as session:
            query = session.query(LogCollectionScenarioModel)
            if active_only:
                query = query.filter(LogCollectionScenarioModel.is_active.is_(True))
            return [
                _scenario_to_domain(m)
                for m in query.order_by(LogCollectionScenarioModel.technology, LogCollectionScenarioModel.scenario_type).all()
            ]

    def delete_scenario(self, scenario_id: str) -> None:
        with self._session_factory() as session:
            model = session.get(LogCollectionScenarioModel, scenario_id)
            if model is not None:
                session.delete(model)
                session.commit()


def _log_source_to_domain(model: LogSourceApplicationModel) -> LogSourceApplication:
    return LogSourceApplication(
        id=model.id,
        name=model.name,
        location=LogRepositoryLocation(
            platform=model.location_platform,
            root_path=model.location_root_path,
            subdirectory=model.location_subdirectory,
            filename_patterns=model.location_filename_patterns,
            raw_paths=model.location_raw_paths,
        ),
        technology=model.technology,
        product=model.product,
        purpose=model.purpose,
        log_level_support=model.log_level_support,
        typical_issues=model.typical_issues,
        common_errors=model.common_errors,
        related_sql=model.related_sql,
        related_documentation=model.related_documentation,
        related_known_bugs=model.related_known_bugs,
        related_playbooks=model.related_playbooks,
        notes=model.notes,
        source_wiki_pages=model.source_wiki_pages,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
        status=model.status,
    )


def _scenario_to_domain(model: LogCollectionScenarioModel) -> LogCollectionScenario:
    return LogCollectionScenario(
        id=model.id,
        product=model.product,
        technology=model.technology,
        version=model.version,
        scenario_type=model.scenario_type,
        region=model.region,
        steps=[LogCollectionStep(**step) for step in model.steps],
        notes=model.notes,
        source_wiki_page=model.source_wiki_page,
        created_at=model.created_at,
        updated_at=model.updated_at,
        created_by=model.created_by,
        updated_by=model.updated_by,
        is_active=model.is_active,
        status=model.status,
    )
