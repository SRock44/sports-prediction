"""API key storage and request audit log."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from src.db.models.base import Base


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    @property
    def is_active(self) -> bool:
        from src.core.time import utc_now
        now = utc_now()
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at < now:
            return False
        return True


class ApiRequest(Base):
    """Audit trail. Inserted asynchronously — never blocks a response."""
    __tablename__ = "api_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"), index=True)
    route: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    ip: Mapped[str | None] = mapped_column(String(64))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
