from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Postgres ──────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "prediction"
    postgres_user: str = "prediction_app"
    postgres_password: SecretStr = SecretStr("changeme")
    postgres_migration_user: str = "prediction_migration"
    postgres_migration_password: SecretStr = SecretStr("changeme")
    postgres_pool_size: int = 10
    postgres_max_overflow: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: SecretStr | None = None

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret: SecretStr = SecretStr("changeme_use_32_random_bytes_in_production")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 15

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    allowed_origins: list[str] = []
    admin_ip_allowlist: list[str] = ["127.0.0.1", "::1"]

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "file:./mlruns"
    mlflow_experiment_name: str = "prediction"

    # ── Notifications ─────────────────────────────────────────────────────────
    discord_webhook_nba: str | None = None
    discord_webhook_mlb: str | None = None
    discord_webhook_ops: str | None = None
    telegram_bot_token: SecretStr | None = None
    telegram_chat_id_nba: str | None = None
    telegram_chat_id_mlb: str | None = None

    # ── Discord interactive bot ────────────────────────────────────────────────
    discord_bot_token: SecretStr | None = None
    discord_guild_id: str | None = None  # for instant slash-command sync; global if unset

    # ── Backfill ──────────────────────────────────────────────────────────────
    nba_backfill_seasons: int = 5
    mlb_backfill_seasons: int = 5

    # ── Model freshness thresholds ────────────────────────────────────────────
    # Drift detection: trigger unscheduled retrain if 30-game rolling log-loss
    # degrades by this fraction relative to the backtest baseline.
    drift_logloss_threshold: float = 0.10
    drift_ece_threshold: float = 0.07
    drift_psi_warn: float = 0.25
    drift_psi_retrain: float = 0.50

    # Promotion gate: challenger must beat champion by at least this much.
    promote_logloss_improvement: float = 0.01
    promote_ece_margin: float = 0.02

    # ── Environment ───────────────────────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    # ── Derived properties (not env vars) ────────────────────────────────────
    @property
    def database_url_async(self) -> str:
        pw = self.postgres_password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        pw = self.postgres_password.get_secret_value()
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{pw}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            pw = self.redis_password.get_secret_value()
            return f"redis://:{pw}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_broker_url(self) -> str:
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        return self.redis_url

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @field_validator("allowed_origins", "admin_ip_allowlist", mode="before")
    @classmethod
    def split_csv(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v  # type: ignore[no-any-return]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
