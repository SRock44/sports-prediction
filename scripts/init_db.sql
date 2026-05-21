-- Runs once on first Postgres container start (docker-entrypoint-initdb.d).
-- Executed as the POSTGRES_USER superuser.

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for fuzzy player name search

-- ── Migration user (Alembic DDL) ──────────────────────────────────────────────
-- Creates prediction_migrate if absent. Override the password via the
-- MIGRATION_PASSWORD environment variable before production deployment;
-- the default 'change_me' must never reach a real environment.
DO $$
DECLARE
    mig_pass text := coalesce(nullif(current_setting('app.migration_password', true), ''), 'change_me');
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'prediction_migrate') THEN
        EXECUTE format('CREATE ROLE prediction_migrate WITH LOGIN PASSWORD %L', mig_pass);
    END IF;
END$$;

DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO prediction_migrate', current_database());
END$$;
GRANT ALL ON SCHEMA public TO prediction_migrate;

-- ── App user (DML only, no DDL) ───────────────────────────────────────────────
-- prediction_app is created by the POSTGRES_USER env var in docker-compose.
-- Strip superuser-equivalent rights it inherited by default and grant only
-- what the application needs.
DO $$
BEGIN
    EXECUTE format('GRANT CONNECT ON DATABASE %I TO prediction_app', current_database());
END$$;

GRANT USAGE ON SCHEMA public TO prediction_app;

-- Existing tables (in case migrations already ran before this grant was applied)
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO prediction_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO prediction_app;

-- Future tables created by prediction_migrate (Alembic)
ALTER DEFAULT PRIVILEGES FOR ROLE prediction_migrate IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO prediction_app;
ALTER DEFAULT PRIVILEGES FOR ROLE prediction_migrate IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO prediction_app;
