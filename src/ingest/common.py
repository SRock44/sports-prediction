"""Shared ingest infrastructure: rate-limited HTTP client, bulk upserter, result type."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import Semaphore
from typing import Any, TypeVar

import httpx
from sqlalchemy import literal_column
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# ── Result type ───────────────────────────────────────────────────────────────


@dataclass
class IngestResult:
    rows_inserted: int = 0
    rows_updated: int = 0
    rows_skipped: int = 0
    last_external_id: str | None = None
    errors: list[str] = field(default_factory=list)

    def __iadd__(self, other: IngestResult) -> IngestResult:
        self.rows_inserted += other.rows_inserted
        self.rows_updated += other.rows_updated
        self.rows_skipped += other.rows_skipped
        self.errors.extend(other.errors)
        return self


# ── Rate-limited HTTP client ──────────────────────────────────────────────────

_NBA_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
]


class RateLimitedClient:
    """Token-bucket HTTP client with retry and UA rotation.

    requests_per_second: enforced across all calls (thread-safe via Semaphore).
    """

    def __init__(
        self,
        base_url: str = "",
        requests_per_second: float = 1.0,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        ua_pool: list[str] | None = None,
    ) -> None:
        self._rps = requests_per_second
        self._min_interval = 1.0 / requests_per_second
        self._last_call = 0.0
        self._sem = Semaphore(1)
        self._ua_pool = ua_pool or []
        self._ua_idx = 0

        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers or {},
            follow_redirects=True,
        )

    def _next_ua(self) -> str | None:
        if not self._ua_pool:
            return None
        ua = self._ua_pool[self._ua_idx % len(self._ua_pool)]
        self._ua_idx += 1
        return ua

    def _throttle(self) -> None:
        with self._sem:
            elapsed = time.monotonic() - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        reraise=True,
    )
    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        self._throttle()
        ua = self._next_ua()
        if ua:
            kwargs.setdefault("headers", {})["User-Agent"] = ua
        response = self._client.get(url, **kwargs)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> RateLimitedClient:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


# ── Bulk upsert helper ────────────────────────────────────────────────────────


class Upserter:
    """Postgres INSERT … ON CONFLICT DO UPDATE wrapper."""

    def __init__(self, session: Session, model: type, conflict_columns: list[str]) -> None:
        self._session = session
        self._model = model
        self._conflict_columns = conflict_columns
        self._table = model.__table__  # type: ignore[attr-defined]

    def upsert_many(self, rows: list[dict[str, Any]], chunk_size: int = 500) -> IngestResult:
        result = IngestResult()
        if not rows:
            return result

        for i in range(0, len(rows), chunk_size):
            chunk = rows[i : i + chunk_size]
            stmt = pg_insert(self._table).values(chunk)

            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in self._table.columns
                if col.name not in self._conflict_columns and col.name != "id"
            }

            stmt = stmt.on_conflict_do_update(  # type: ignore[assignment]
                index_elements=self._conflict_columns,
                set_=update_cols,
                # xmax = 0 means the row was freshly inserted; non-zero means it was updated.
            ).returning(literal_column("(xmax = 0)::bool").label("was_inserted"))

            rows_returned = self._session.execute(stmt).fetchall()
            for row_result in rows_returned:
                if row_result.was_inserted:
                    result.rows_inserted += 1
                else:
                    result.rows_updated += 1

        return result


# ── Checksum helper ───────────────────────────────────────────────────────────


def dict_hash(d: dict[str, Any]) -> str:
    """Stable SHA-256 of a JSON-serialisable dict. Used for features_hash."""
    import json

    serialised = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()
