"""Unit tests for security helpers: API key hashing, JWT encode/decode, scopes."""
from __future__ import annotations

import time

import pytest

from src.core.security import (
    generate_api_key,
    hash_api_key,
    verify_api_key,
    create_access_token,
    decode_access_token,
    token_has_scope,
)


class TestApiKey:
    def test_generate_is_urlsafe_string(self):
        key = generate_api_key()
        assert isinstance(key, str)
        assert len(key) >= 40  # 32 bytes base64url

    def test_hash_and_verify_round_trip(self):
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert verify_api_key(key, hashed)

    def test_wrong_key_fails_verify(self):
        key = generate_api_key()
        hashed = hash_api_key(key)
        assert not verify_api_key("wrong_key", hashed)

    def test_different_keys_different_hashes(self):
        h1 = hash_api_key("abc")
        h2 = hash_api_key("abc")
        # Argon2 salts each hash separately
        assert h1 != h2

    def test_two_keys_independently_correct(self):
        k1, k2 = generate_api_key(), generate_api_key()
        assert k1 != k2
        h1, h2 = hash_api_key(k1), hash_api_key(k2)
        assert verify_api_key(k1, h1)
        assert verify_api_key(k2, h2)
        assert not verify_api_key(k1, h2)


class TestJWT:
    def test_encode_decode_round_trip(self):
        token = create_access_token("user-42", ["predictions:read"])
        payload = decode_access_token(token)
        assert payload["sub"] == "user-42"
        assert "predictions:read" in payload["scopes"]

    def test_expired_token_raises(self):
        # Create a token that expires in -1 second (already expired)
        from datetime import datetime, timezone, timedelta
        from jose import jwt
        from src.core.config import get_settings

        settings = get_settings()
        exp = datetime.now(timezone.utc) - timedelta(seconds=1)
        payload = {"sub": "user-1", "scopes": [], "exp": int(exp.timestamp())}
        token = jwt.encode(payload, settings.secret_key, algorithm="HS256")

        with pytest.raises(Exception):
            decode_access_token(token)

    def test_tampered_token_raises(self):
        token = create_access_token("user-1", [])
        tampered = token[:-5] + "XXXXX"
        with pytest.raises(Exception):
            decode_access_token(tampered)

    def test_extra_claims_preserved(self):
        token = create_access_token("u", [], extra={"api_key_id": 99})
        payload = decode_access_token(token)
        assert payload.get("api_key_id") == 99


class TestScopeCheck:
    def test_has_required_scope(self):
        payload = {"scopes": ["predictions:read", "models:read"]}
        assert token_has_scope(payload, "predictions:read")

    def test_missing_scope_returns_false(self):
        payload = {"scopes": ["predictions:read"]}
        assert not token_has_scope(payload, "admin")

    def test_empty_scopes_returns_false(self):
        assert not token_has_scope({"scopes": []}, "predictions:read")

    def test_admin_has_all_scopes(self):
        payload = {"scopes": ["admin"]}
        assert token_has_scope(payload, "admin")
