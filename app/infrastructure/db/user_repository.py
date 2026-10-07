"""Repository for the ``users`` table (hosted-deployment authentication)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from app.infrastructure.db.models import UserModel


@dataclass(frozen=True)
class UserRecord:
    username: str
    password_hash: str
    role: str
    is_active: bool


class UserRepository(Protocol):
    def get(self, username: str) -> UserRecord | None:
        ...

    def create(self, username: str, password_hash: str, role: str) -> None:
        ...

    def count(self) -> int:
        ...


class SqlAlchemyUserRepository:
    def __init__(self, session_factory: sessionmaker[OrmSession]) -> None:
        self._session_factory = session_factory

    def get(self, username: str) -> UserRecord | None:
        with self._session_factory() as session:
            model = session.get(UserModel, username)
            if model is None:
                return None
            return UserRecord(model.username, model.password_hash, model.role, model.is_active)

    def create(self, username: str, password_hash: str, role: str) -> None:
        with self._session_factory() as session:
            session.add(UserModel(username=username, password_hash=password_hash, role=role, is_active=True))
            session.commit()

    def count(self) -> int:
        with self._session_factory() as session:
            return session.scalar(select(func.count()).select_from(UserModel)) or 0
