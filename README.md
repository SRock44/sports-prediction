# prediction

A production-grade sports match prediction system for **NBA** and **MLB**. Trains its own ML models on free official data, serves predictions through a secure REST API, and pushes rich notifications to Discord and Telegram.

No paid data feeds. No LLM inference. Fully self-hosted.

---

## What it does

- **Ingests** schedules, box scores, play-by-play, rosters, and injury reports from `stats.nba.com` and `statsapi.mlb.com` — both free, official sources
- **Trains** XGBoost game-winner models and LightGBM player-prop models on up to 5 seasons of history
- **Scores** upcoming games automatically each day; re-scores when confirmed lineups arrive
- **Publishes** predictions via Discord webhook embeds and Telegram bot messages
- **Retrains nightly** with a champion/challenger gate — fresh data in, regressions blocked out
- **Detects drift** in model performance and feature distributions; triggers emergency retrains when needed
- **Exposes** a JWT-authenticated REST API with per-key rate limiting, Argon2-hashed keys, and a full audit log

---

## Predictions

### Game winner (moneyline)
- **NBA**: calibrated win probability for each team; realistic accuracy ~67–69%
- **MLB**: calibrated win probability; realistic accuracy ~56–58%

### Player props
- **NBA**: PTS, REB, AST, 3PM, PRA
- **MLB hitters**: H, HR, TB, RBI, K
- **MLB pitchers**: K, ER, OUTS

Props output a full predictive distribution (10th / 25th / 50th / 75th / 90th percentile) so any over/under line can be evaluated.

---

## Architecture

```
Celery Workers (Redis broker)
  ingest  →  features  →  train  →  score  →  pg_notify
                                                    ↓
                                           PostgreSQL 16 + TimescaleDB
                                                    ↑
                                           FastAPI (gunicorn + uvicorn)
                                                    ↑
                                           Caddy (auto-HTTPS, HSTS)
                                                    ↑
                                           Discord / Telegram notifier
                                           (PG LISTEN on predictions)
```

All services run in Docker Compose. Same image serves API, worker, beat scheduler, and notifier.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Web | FastAPI + uvicorn + gunicorn |
| Database | PostgreSQL 16 + TimescaleDB |
| Cache / broker | Redis 7 |
| Queue | Celery 5 + celery-beat |
| ORM / migrations | SQLAlchemy 2 + Alembic |
| ML — winner | XGBoost + isotonic calibration |
| ML — props | LightGBM quantile regression |
| Hyperparameter search | Optuna (50 trials, walk-forward objective) |
| Model registry | MLflow (local file backend) |
| Auth | Argon2id API keys → JWT HS256 (15 min TTL) |
| Rate limiting | slowapi + Redis sliding window |
| Reverse proxy | Caddy (auto Let's Encrypt, TLS 1.3) |
| Notifications | Discord webhooks + Telegram Bot API |

---

## Quickstart

### Prerequisites

- Docker + Docker Compose
- A Discord webhook URL and/or Telegram bot token (optional but recommended)

### 1. Clone and configure

```bash
git clone <your-repo-url> prediction
cd prediction
cp .env.example .env
```

Edit `.env` and fill in at minimum:

```env
SECRET_KEY=<64-char random hex>           # openssl rand -hex 32
POSTGRES_PASSWORD=<strong password>
DISCORD_WEBHOOK_NBA=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_MLB=https://discord.com/api/webhooks/...
```

### 2. Start services

```bash
docker compose up -d
```

All services start healthy in under 90 seconds. Check status:

```bash
curl http://localhost:8000/v1/health
```

### 3. Run the initial migration

```bash
docker compose exec api alembic upgrade head
```

### 4. Backfill historical data and train initial models

```bash
docker compose exec api bash scripts/bootstrap_backfill.sh
```

This runs sequentially:
1. Backfill 5 seasons of NBA + MLB data (~4–8 hours on a typical VPS)
2. Train winner models for both sports (Optuna tuning, ~3–10 min each)
3. Train player prop models
4. Score the next 48 hours of upcoming games

Progress is logged to stdout. The process is idempotent — safe to re-run.

### 5. Create your first API key

```bash
docker compose exec api python -m src.cli keys create --name "discord-bot" --scopes "predictions:read"
```

The plaintext key is shown once. Store it securely.

### 6. Make your first API call

```bash
# Exchange key for JWT
TOKEN=$(curl -s -X POST http://localhost:8000/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"api_key": "YOUR_KEY_HERE"}' | jq -r .access_token)

# Fetch upcoming games
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/v1/games/upcoming?sport=nba&hours=48"

# Fetch predictions for a game
curl -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/v1/predictions/game/GAME_ID"
```

---

## API reference

Full OpenAPI docs at `/docs` (auth-gated in production; set `ENVIRONMENT=development` to open).

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/auth/token` | Exchange API key for JWT |
| `GET` | `/v1/health` | System health + SLO status |
| `GET` | `/v1/sports` | Available sports and active model versions |
| `GET` | `/v1/games/upcoming` | Scheduled games (`?sport=nba&hours=48`) |
| `GET` | `/v1/games/{id}` | Game detail and current status |
| `GET` | `/v1/predictions/game/{id}` | Win probability prediction for a game |
| `GET` | `/v1/predictions/props/{id}` | All player props for a game |
| `GET` | `/v1/predictions/player/{id}` | Single player props (`?game_id=...`) |
| `GET` | `/v1/models/active` | Active model versions by sport |

All responses include `model_version`, `as_of_utc`, and `features_hash` for staleness detection.

---

## CLI reference

```bash
python -m src.cli backfill --sport nba --seasons 5   # Ingest historical data
python -m src.cli train --sport nba --kind winner     # Train + optionally promote
python -m src.cli eval --sport nba --kind winner      # Walk-forward backtest report
python -m src.cli score --sport nba --hours 48        # Score upcoming games now

python -m src.cli keys create --name "my-bot"         # Generate API key
python -m src.cli keys list                           # List all keys
python -m src.cli keys revoke 3                       # Revoke key by ID

python -m src.cli model list --sport nba              # List model versions
python -m src.cli model rollback --sport nba          # Roll back to previous champion
python -m src.cli model promote <run-id>              # Force-promote an MLflow run
```

---

## Scheduled pipeline

All jobs run automatically via Celery beat once the system is running.

| Time (UTC) | Job | What it does |
|---|---|---|
| 09:00 daily | `ingest_yesterday` | Box scores + play-by-play for completed games |
| 11:00 daily | `refresh_injuries` | NBA injury PDF + MLB IL transactions |
| 12:00 daily | `rebuild_features` | Matchup features for upcoming 48h games |
| 13:00 daily | `score_upcoming` | Inference → predictions → Discord/Telegram |
| Every 15 min | `rescore_lineup_change` | Re-score when confirmed lineups arrive |
| Every 2 min (in-window) | `poll_live` | Live score updates during active games |
| 02:00 daily | `train_challenger` | Full retrain from scratch, log to MLflow |
| 03:00 daily | `drift_monitor` | Performance, calibration, and PSI drift checks |
| Mon 04:00 | `evaluate_and_promote` | Promote best nightly challenger if it passes all gates |
| Sun 03:00 | `backtest_report` | Regenerate full evaluation report |
| 1st of month 05:00 | `hyperparam_search` | Optuna re-tune of all hyperparameters |

---

## Continuous learning

The model stays fresh without forgetting old data through three mechanisms:

1. **Rolling training window** — winner models train on the last 5 seasons; prop models on the last 3. The window slides forward with each nightly retrain.

2. **Recency sample weighting** — `weight = exp(-λ × days_since_game / 365)`. Recent games dominate the loss function. λ is tuned per sport and target during walk-forward CV.

3. **Nightly challenger + weekly promotion gate** — every night a new model is trained from scratch and logged as a challenger. Every Monday the best challenger from the past week is compared to the current champion. Promotion requires all four gates to pass:
   - Log-loss improvement ≥ 1%
   - Expected Calibration Error ≤ champion ECE + 0.02
   - No feature importance shift > 50%
   - Brier score not worse than champion

**Drift detection** runs daily and triggers a priority retrain if:
- Rolling 30-game log-loss degrades > 10% vs. baseline
- Expected Calibration Error exceeds 0.07
- Any feature's Population Stability Index exceeds 0.50

---

## Anti-leakage guarantee

Every feature is computed using only data that existed at `scheduled_utc - 1 hour` (the as-of timestamp). This invariant is enforced in every SQL query in the feature pipeline. The same code path is used at training time and inference time — there is no separate "training feature builder."

The integration test suite includes:
- **Feature parity test**: features at training time must be bit-identical to features recomputed at inference time for the same game and as-of timestamp
- **Leakage test**: features must be unchanged when post-game data is visible (as-of anchors them to pre-game state)

---

## Security

- **Transport**: Caddy enforces HTTPS, HSTS preload, TLS 1.3 only. HTTP → 308 redirect.
- **API keys**: 32-byte random, stored Argon2id-hashed. Never logged.
- **Auth flow**: API key → JWT (HS256, 15-min TTL). Revocation is instant via Redis blacklist.
- **Rate limiting**: 60 req/min, 1000 req/hour per key (configurable). Redis sliding window.
- **Input validation**: all request parameters through Pydantic strict types; unknown fields rejected.
- **Security headers**: `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, no `Server` header.
- **Audit log**: every authenticated request written to `api_requests` (route, status, latency, key ID, IP).
- **DB roles**: app user has DML only; migrations run from a separate DDL role.
- **Containers**: non-root user, read-only root filesystem, all Linux capabilities dropped.

---

## Repository layout

```
prediction/
├── docker-compose.yml
├── Dockerfile
├── Caddyfile
├── pyproject.toml
├── .env.example
├── alembic/
│   └── versions/0001_initial_schema.py
├── scripts/
│   ├── bootstrap_backfill.sh
│   └── init_db.sql
├── src/
│   ├── cli.py                        # Typer CLI
│   ├── core/                         # Config, logging, security, time
│   ├── db/                           # SQLAlchemy models, session, repositories
│   ├── ingest/
│   │   ├── nba/                      # Schedule, box scores, injuries, live
│   │   └── mlb/                      # Schedule, box scores, IL, live
│   ├── features/
│   │   ├── common.py                 # Elo, rolling windows, haversine
│   │   ├── nba/                      # Team, matchup, player features
│   │   └── mlb/                      # Team, matchup, player features
│   ├── models/
│   │   ├── train_winner.py           # XGBoost + isotonic calibration
│   │   ├── train_props.py            # LightGBM quantile regression
│   │   ├── score.py                  # Inference + pg_notify
│   │   ├── registry.py               # MLflow wrapper, promotion, rollback
│   │   └── eval/                     # Walk-forward CV, metrics, reports
│   ├── api/                          # FastAPI app, auth, routes, schemas
│   ├── tasks/                        # Celery tasks + beat schedule
│   └── notify/                       # Discord, Telegram, PG LISTEN loop
└── tests/
    ├── unit/                         # Feature math, security, dedup, time
    └── integration/                  # Feature parity, leakage, auth flow
```

---

## Running tests

```bash
# Unit tests only (no containers needed)
docker compose exec api pytest tests/unit -q

# All tests including integration (requires containers)
docker compose exec api pytest -q

# Skip integration tests in CI without containers
docker compose exec api pytest tests/unit -q -m "not integration"
```

---

## Environment variables

See `.env.example` for the full list with descriptions. Required variables:

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT signing secret (32+ random bytes) |
| `POSTGRES_PASSWORD` | Database password |
| `POSTGRES_MIGRATION_PASSWORD` | DDL-role password for Alembic |
| `DISCORD_WEBHOOK_NBA` | Discord webhook for NBA predictions |
| `DISCORD_WEBHOOK_MLB` | Discord webhook for MLB predictions |
| `DISCORD_WEBHOOK_OPS` | Discord webhook for ops/drift alerts |

Optional:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID_NBA` | Telegram chat ID for NBA |
| `TELEGRAM_CHAT_ID_MLB` | Telegram chat ID for MLB |
| `ADMIN_IP_ALLOWLIST` | Comma-separated IPs for admin endpoints |

---

## Backtest reports

After `python -m src.cli eval --sport nba --kind winner`, a Markdown report is written to `reports/`. It includes:

- Per-season log-loss, Brier score, accuracy, and calibration error
- Calibration curve (predicted probability vs. actual win rate by bucket)
- Feature importance rankings
- Sanity checks: shuffled-label baseline, future-data mask test

Realistic performance targets: NBA winner ~67–69% accuracy, MLB winner ~56–58%. Any backtest claiming significantly higher is a leakage red flag — the eval harness explicitly checks for it.

---

## Adding a new sport

1. Add ingest fetchers in `src/ingest/<sport>/`
2. Add feature builders in `src/features/<sport>/`
3. Add a feature config list in `src/models/configs/<sport>_winner.py`
4. Register Celery tasks in `src/tasks/ingest_tasks.py` and `src/tasks/schedule.py`
5. Run `alembic revision --autogenerate` if schema changes are needed
6. Run `python -m src.cli backfill --sport <sport>` and then `train`

---

## License

MIT
