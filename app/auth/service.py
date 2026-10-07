"""Login and first-run bootstrap."""

from __future__ import annotations

import logging

from app.auth.roles import ROLE_ADMIN
from app.auth.security import create_access_token, hash_password, verify_password
from app.config import Settings
from app.infrastructure.db.user_repository import UserRepository

logger = logging.getLogger(__name__)

# Verified against when the username doesn't exist so a failed login costs
# the same PBKDF2 time either way (no username-enumeration timing signal).
_DUMMY_HASH = hash_password("not-a-real-password")


class AuthService:
    def __init__(self, users: UserRepository, settings: Settings) -> None:
        self._users = users
        self._settings = settings

    def authenticate(self, username: str, password: str) -> str | None:
        """Returns an access token, or None for any failure (unknown user,
        wrong password, disabled account) -- callers must not distinguish."""
        user = self._users.get(username)
        stored = user.password_hash if user else _DUMMY_HASH
        password_ok = verify_password(password, stored)
        if user is None or not password_ok or not user.is_active:
            return None
        return create_access_token(
            username=user.username,
            role=user.role,
            secret_key=self._settings.auth_secret_key or "",
            ttl_minutes=self._settings.auth_token_ttl_minutes,
        )

    def bootstrap_admin(self) -> bool:
        """Creates the configured admin if (and only if) no user exists yet."""
        username = self._settings.auth_bootstrap_admin_username
        password = self._settings.auth_bootstrap_admin_password
        if not username or not password or self._users.count() > 0:
            return False
        self._users.create(username, hash_password(password), ROLE_ADMIN)
        logger.info("Created bootstrap admin user '%s'", username)
        return True
