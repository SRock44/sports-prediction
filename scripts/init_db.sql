-- Runs once on first Postgres container start.
-- Creates the migration user and enables TimescaleDB.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for fuzzy player name search

-- The app user is created by docker-compose env vars (POSTGRES_USER).
-- We need a separate migration user with DDL rights.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = current_setting('migration_user', true)) THEN
        NULL;  -- handled below via env substitution at deploy time
    END IF;
END$$;

-- Alembic migration user (created by the init script at deploy; see scripts/create_migration_user.sh)
-- App user gets connect + DML only; no DDL.
GRANT CONNECT ON DATABASE prediction TO prediction_app;
