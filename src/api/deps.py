"""FastAPI dependency injectors: auth, DB session, rate limiting."""
from __future__ import annotations

from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.security import decode_access_token, token_has_scope
from src.db.session import get_async_session

# ── Rate limiter ──────────────────────────────────────────────────────────────

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=settings.redis_url,
    default_limits=[
        f"{settings.rate_limit_per_minute}/minute",
        f"{settings.rate_limit_per_hour}/hour",
    ],
)

# ── Auth dependencies ─────────────────────────────────────────────────────────

bearer_scheme = HTTPBearer()


async def get_current_payload(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    request: Request,
) -> dict:
    try:
        payload = decode_access_token(credentials.credentials)
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Attach key_id to request state for audit logging
    request.state.api_key_id = payload.get("sub")

    # Check JWT blacklist (revoked tokens in Redis)
    jti = credentials.credentials[-8:]  # last 8 chars as cheap jti proxy
    redis_client = await _get_redis()
    if await redis_client.get(f"revoked:{jti}"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
        )

    return payload


def require_scope(scope: str):
    """Dependency factory: raises 403 if the JWT lacks the required scope."""
    async def _check(payload: dict = Depends(get_current_payload)) -> dict:
        if not token_has_scope(payload, scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires scope: {scope}",
            )
        return payload
    return _check


async def require_admin_ip(request: Request) -> None:
    """Reject non-allowlisted IPs on admin endpoints."""
    client_ip = request.client.host if request.client else "unknown"
    if settings.admin_ip_allowlist and client_ip not in settings.admin_ip_allowlist:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin endpoint: IP not allowlisted",
        )


# ── Redis client ──────────────────────────────────────────────────────────────

_redis_pool: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await aioredis.from_url(settings.redis_url, decode_responses=True)
    return _redis_pool
