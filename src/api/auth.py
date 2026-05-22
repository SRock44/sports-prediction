"""API key → JWT exchange and key management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import create_access_token, hash_api_key, needs_rehash, verify_api_key
from src.db.models.auth import ApiKey
from src.db.session import get_async_session

router = APIRouter(tags=["auth"])


class TokenRequest(BaseModel):
    api_key: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


@router.post("/auth/token", response_model=TokenResponse)
async def exchange_token(
    body: TokenRequest,
    session: AsyncSession = Depends(get_async_session),
) -> TokenResponse:
    """Exchange a long-lived API key for a short-lived JWT."""
    from sqlalchemy import select

    # Narrow candidates by the 8-char key_prefix before running slow Argon2 verify.
    # Keys created before this column existed have key_prefix=NULL; fall back to
    # full scan only for those rows so legacy keys keep working after migration.
    prefix = body.api_key[:8]
    result = await session.execute(
        select(ApiKey).where(
            ApiKey.revoked_at.is_(None),
            (ApiKey.key_prefix == prefix) | ApiKey.key_prefix.is_(None),
        )
    )
    candidates = result.scalars().all()

    matched_key: ApiKey | None = None
    for key in candidates:
        if verify_api_key(body.api_key, key.key_hash):
            if not key.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key is expired or revoked",
                )
            matched_key = key
            break

    if matched_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    # Rehash if cost parameters have changed
    if needs_rehash(matched_key.key_hash):
        matched_key.key_hash = hash_api_key(body.api_key)

    from src.core.config import settings

    token = create_access_token(
        subject=str(matched_key.id),
        scopes=list(matched_key.scopes or []),
    )
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_access_token_expire_minutes * 60,
    )
