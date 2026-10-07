"""Password hashing and signed-token primitives.

Stdlib PBKDF2-HMAC-SHA256 for passwords (no extra native dependency) and
PyJWT (HS256, algorithm pinned on decode) for access tokens.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

_PBKDF2_ITERATIONS = 600_000
_ALGORITHM = "HS256"


class InvalidTokenError(Exception):
    """Token is malformed, tampered with, or expired."""


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored_hash.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)


def create_access_token(*, username: str, role: str, secret_key: str, ttl_minutes: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": username, "role": role, "iat": now, "exp": now + timedelta(minutes=ttl_minutes)}
    return jwt.encode(payload, secret_key, algorithm=_ALGORITHM)


def decode_access_token(token: str, *, secret_key: str) -> dict:
    try:
        return jwt.decode(token, secret_key, algorithms=[_ALGORITHM], options={"require": ["exp", "sub", "role"]})
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
