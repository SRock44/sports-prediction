"""Auth primitives: Argon2 key hashing, JWT encode/decode, API key generation."""
from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError
from jose import JWTError, jwt

from src.core.config import settings
from src.core.time import utc_now

_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=65536,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

# ── API key utilities ─────────────────────────────────────────────────────────

def generate_api_key() -> str:
    """Return a new plaintext API key (caller is responsible for hashing and storing)."""
    return secrets.token_urlsafe(32)


def hash_api_key(plaintext: str) -> str:
    return _hasher.hash(plaintext)


def verify_api_key(plaintext: str, hashed: str) -> bool:
    try:
        return _hasher.verify(hashed, plaintext)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(hashed: str) -> bool:
    return _hasher.check_needs_rehash(hashed)


# ── JWT utilities ─────────────────────────────────────────────────────────────

def create_access_token(subject: str, scopes: list[str], extra: dict[str, Any] | None = None) -> str:
    now = utc_now()
    expire = now + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "scopes": scopes,
        "exp": int(expire.timestamp()),
        "iat": int(now.timestamp()),
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and validate. Raises JWTError on any failure."""
    return jwt.decode(  # type: ignore[no-any-return]
        token,
        settings.jwt_secret.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
    )


def token_has_scope(payload: dict[str, Any], required: str) -> bool:
    scopes: list[str] = payload.get("scopes", [])
    return required in scopes or "admin" in scopes
