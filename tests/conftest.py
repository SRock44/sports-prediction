"""Shared pytest fixtures: ephemeral Postgres + Redis via testcontainers."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

# testcontainers is only needed in integration tests; guard the import
try:
    from testcontainers.postgres import PostgresContainer
    from testcontainers.redis import RedisContainer

    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False


# ── Unit-test session (SQLite in-memory) ──────────────────────────────────────


@pytest.fixture(scope="session")
def sqlite_engine():
    """Lightweight in-memory engine for unit tests that need ORM models."""
    import src.db.models  # noqa: F401 — trigger all model registrations
    from src.db.models.base import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def sqlite_session(sqlite_engine) -> Session:
    conn = sqlite_engine.connect()
    tx = conn.begin()
    session = sessionmaker(bind=conn)()
    yield session
    session.close()
    tx.rollback()
    conn.close()


# ── Integration-test containers ───────────────────────────────────────────────


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: mark test as requiring live Postgres + Redis containers",
    )


@pytest.fixture(scope="session")
def pg_container():
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed")
    with PostgresContainer("timescale/timescaledb:latest-pg16") as pg:
        yield pg


@pytest.fixture(scope="session")
def redis_container():
    if not HAS_TESTCONTAINERS:
        pytest.skip("testcontainers not installed")
    with RedisContainer("redis:7-alpine") as r:
        yield r


@pytest.fixture(scope="session")
def pg_engine(pg_container):
    import src.db.models  # noqa: F401
    from src.db.models.base import Base

    engine = create_engine(pg_container.get_connection_url())
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        conn.commit()
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def pg_session(pg_engine) -> Session:
    conn = pg_engine.connect()
    tx = conn.begin()
    session = sessionmaker(bind=conn)()
    yield session
    session.close()
    tx.rollback()
    conn.close()


@pytest.fixture(scope="session")
def redis_client(redis_container):
    import redis as redis_lib

    return redis_lib.from_url(redis_container.get_connection_url(), decode_responses=True)
