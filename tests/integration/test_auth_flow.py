"""Integration test: full API key → JWT → protected endpoint flow."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.core.security import generate_api_key, hash_api_key
from src.db.models.auth import ApiKey


@pytest.fixture()
def test_client(pg_session, monkeypatch):
    """FastAPI TestClient with a real Postgres session injected."""
    from contextlib import contextmanager

    from src.api.main import app

    @contextmanager
    def _override():
        yield pg_session

    monkeypatch.setattr("src.db.session.get_sync_session", _override)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture()
def api_key_pair(pg_session) -> tuple[str, str]:
    """Insert a live API key and return (plaintext, key_id)."""
    plaintext = generate_api_key()
    k = ApiKey(
        name="test-key",
        key_hash=hash_api_key(plaintext),
        scopes=["predictions:read", "models:read"],
    )
    pg_session.add(k)
    pg_session.flush()
    return plaintext, str(k.id)


@pytest.mark.integration
def test_health_no_auth(test_client):
    resp = test_client.get("/v1/health")
    assert resp.status_code == 200


@pytest.mark.integration
def test_auth_token_flow(test_client, api_key_pair):
    plaintext, _ = api_key_pair
    resp = test_client.post("/v1/auth/token", json={"api_key": plaintext})
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert token

    # Use token to access protected route
    resp2 = test_client.get(
        "/v1/games/upcoming?sport=nba&hours=48",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp2.status_code == 200


@pytest.mark.integration
def test_no_token_returns_401(test_client):
    resp = test_client.get("/v1/games/upcoming?sport=nba")
    assert resp.status_code == 401


@pytest.mark.integration
def test_wrong_token_returns_401(test_client):
    resp = test_client.get(
        "/v1/games/upcoming?sport=nba",
        headers={"Authorization": "Bearer totally.invalid.token"},
    )
    assert resp.status_code == 401


@pytest.mark.integration
def test_wrong_api_key_returns_401(test_client):
    resp = test_client.post("/v1/auth/token", json={"api_key": "not-a-real-key"})
    assert resp.status_code == 401
