"""SQLAlchemy engine and session factories.

- async_session_factory  → used by FastAPI (request-scoped).
- sync_session_factory   → used by Celery workers (thread-safe sync).
"""
from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings

# ── Async engine (FastAPI) ────────────────────────────────────────────────────
async_engine = create_async_engine(
    settings.database_url_async,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    pool_pre_ping=True,
    echo=not settings.is_production,
)

async_session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yields a session, rolls back on exception."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Sync engine (Celery workers) ──────────────────────────────────────────────
sync_engine = create_engine(
    settings.database_url_sync,
    pool_size=settings.postgres_pool_size,
    max_overflow=settings.postgres_max_overflow,
    pool_pre_ping=True,
)

sync_session_factory: sessionmaker[Session] = sessionmaker(
    sync_engine,
    autocommit=False,
    autoflush=False,
)


@contextmanager
def get_sync_session() -> Generator[Session, None, None]:
    """Context-manager-style session for Celery tasks."""
    with sync_session_factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
