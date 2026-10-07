"""Authentication/authorization for the hosted deployment.

Auth is off by default (local use + every other test unchanged); these
tests turn it on via dependency overrides against a temp SQLite DB.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api.dependencies import get_auth_service, get_settings_dep
from app.api.main import app
from app.auth.security import (
    InvalidTokenError,
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from app.auth.service import AuthService
from app.config import Settings
from app.infrastructure.db.session import get_session_factory
from app.infrastructure.db.user_repository import SqlAlchemyUserRepository

SECRET = "x" * 40


# --- primitives --------------------------------------------------------------


def test_password_hash_roundtrip_and_salting():
    first, second = hash_password("s3cret"), hash_password("s3cret")
    assert first != second  # per-hash salt
    assert verify_password("s3cret", first)
    assert not verify_password("wrong", first)
    assert not verify_password("s3cret", "garbage")
    assert not verify_password("s3cret", "md5$1$a$b")


def test_token_roundtrip():
    token = create_access_token(username="amy", role="admin", secret_key=SECRET, ttl_minutes=5)
    claims = decode_access_token(token, secret_key=SECRET)
    assert claims["sub"] == "amy" and claims["role"] == "admin"


def test_token_rejected_with_wrong_key_expiry_or_tampering():
    token = create_access_token(username="amy", role="admin", secret_key=SECRET, ttl_minutes=5)
    with pytest.raises(InvalidTokenError):
        decode_access_token(token, secret_key="y" * 40)
    with pytest.raises(InvalidTokenError):
        decode_access_token(token[:-2] + "xx", secret_key=SECRET)
    expired = create_access_token(username="amy", role="admin", secret_key=SECRET, ttl_minutes=-1)
    with pytest.raises(InvalidTokenError):
        decode_access_token(expired, secret_key=SECRET)


def test_unsigned_alg_none_token_rejected():
    import jwt

    forged = jwt.encode({"sub": "amy", "role": "admin", "exp": time.time() + 600}, key="", algorithm="none")
    with pytest.raises(InvalidTokenError):
        decode_access_token(forged, secret_key=SECRET)


# --- settings validation -----------------------------------------------------


def test_auth_disabled_by_default():
    assert Settings().auth_enabled is False


def test_auth_enabled_requires_strong_secret():
    with pytest.raises(ValidationError):
        Settings(auth_enabled=True)
    with pytest.raises(ValidationError):
        Settings(auth_enabled=True, auth_secret_key="short")
    assert Settings(auth_enabled=True, auth_secret_key=SECRET).auth_enabled is True


def test_bootstrap_credentials_must_be_paired():
    with pytest.raises(ValidationError):
        Settings(auth_bootstrap_admin_username="a")


# --- service -----------------------------------------------------------------


@pytest.fixture
def auth_settings(tmp_path):
    return Settings(
        sqlite_path=tmp_path / "t.db",
        chroma_persist_dir=tmp_path / "chroma",
        auth_enabled=True,
        auth_secret_key=SECRET,
        auth_bootstrap_admin_username="admin",
        auth_bootstrap_admin_password="admin-pass",
        auto_seed_knowledge=False,
    )


@pytest.fixture
def service(auth_settings):
    users = SqlAlchemyUserRepository(get_session_factory(auth_settings.sqlite_url))
    return AuthService(users, auth_settings), users


def test_bootstrap_creates_admin_once(service):
    svc, users = service
    assert svc.bootstrap_admin() is True
    assert svc.bootstrap_admin() is False
    assert users.count() == 1
    assert users.get("admin").role == "admin"
    assert users.get("admin").password_hash != "admin-pass"


def test_authenticate_success_and_uniform_failure(service):
    svc, users = service
    svc.bootstrap_admin()
    assert svc.authenticate("admin", "admin-pass")
    assert svc.authenticate("admin", "nope") is None
    assert svc.authenticate("ghost", "admin-pass") is None


def test_disabled_account_cannot_log_in(service):
    svc, users = service
    users.create("eve", hash_password("pw"), "engineer")
    assert svc.authenticate("eve", "pw")
    from app.infrastructure.db.models import UserModel

    with users._session_factory() as session:
        session.get(UserModel, "eve").is_active = False
        session.commit()
    assert svc.authenticate("eve", "pw") is None


# --- HTTP enforcement --------------------------------------------------------


@pytest.fixture
def client(auth_settings, service):
    svc, users = service
    svc.bootstrap_admin()
    users.create("eng", hash_password("eng-pass"), "engineer")
    app.dependency_overrides[get_settings_dep] = lambda: auth_settings
    app.dependency_overrides[get_auth_service] = lambda: svc
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_settings_dep, None)
    app.dependency_overrides.pop(get_auth_service, None)


def _login(client, username, password):
    return client.post("/auth/login", json={"username": username, "password": password})


def _headers(client, username, password):
    token = _login(client, username, password).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_and_login_are_open(client):
    assert client.get("/health").status_code == 200
    assert _login(client, "admin", "admin-pass").status_code == 200


def test_bad_login_is_401(client):
    assert _login(client, "admin", "wrong").status_code == 401
    assert _login(client, "ghost", "x").status_code == 401


def test_protected_routes_reject_missing_and_garbage_tokens(client):
    assert client.get("/dashboard").status_code == 401
    assert client.get("/dashboard", headers={"Authorization": "Bearer not.a.token"}).status_code == 401
    assert client.get("/admin/knowledge/documents").status_code == 401


def test_engineer_can_use_app_but_not_admin(client):
    headers = _headers(client, "eng", "eng-pass")
    assert client.get("/dashboard", headers=headers).status_code == 200
    assert client.get("/auth/me", headers=headers).json()["role"] == "engineer"
    assert client.get("/admin/knowledge/documents", headers=headers).status_code == 403


def test_admin_can_reach_admin_routes(client):
    headers = _headers(client, "admin", "admin-pass")
    assert client.get("/dashboard", headers=headers).status_code == 200
    assert client.get("/admin/knowledge/documents", headers=headers).status_code != 403


def test_login_404_when_auth_disabled():
    app.dependency_overrides[get_settings_dep] = lambda: Settings()
    try:
        with TestClient(app) as c:
            assert c.post("/auth/login", json={"username": "a", "password": "b"}).status_code == 404
            assert c.get("/auth/me").json()["auth_enabled"] is False
    finally:
        app.dependency_overrides.pop(get_settings_dep, None)
