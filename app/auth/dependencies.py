"""FastAPI dependencies enforcing authentication/roles.

With ``Settings.auth_enabled=False`` (the default) both dependencies are
no-ops that return a local admin principal -- local use and the existing
test-suite are unaffected. With it enabled, a valid bearer token is
required, and ``require_admin`` additionally requires the ``admin`` role.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies import get_settings_dep
from app.auth.roles import ROLE_ADMIN
from app.auth.security import InvalidTokenError, decode_access_token
from app.config import Settings

_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    username: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == ROLE_ADMIN


_LOCAL_PRINCIPAL = Principal(username="local", role=ROLE_ADMIN)


def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings_dep),
) -> Principal:
    if not settings.auth_enabled:
        return _LOCAL_PRINCIPAL
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        claims = decode_access_token(credentials.credentials, secret_key=settings.auth_secret_key or "")
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    return Principal(username=claims["sub"], role=claims["role"])


def require_admin(principal: Principal = Depends(require_user)) -> Principal:
    if not principal.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required.")
    return principal
