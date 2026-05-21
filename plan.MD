# Sports Prediction Agent — System Plan

## Context

Goal: build a production-grade sports match prediction service for **NBA** and **MLB** that produces **moneyline (win/loss)** picks and **player props** without paying for live data and without using LLM API calls for inference. The system trains its own models on bulk historical data, ingests fresh data continuously from free official sources, and serves predictions through a **secure REST API** consumed by Discord/Telegram bots downstream.

Why this shape:
- **No paid APIs** — `stats.nba.com` (used by NBA.com itself) and `statsapi.mlb.com` (MLB's official public stats endpoint) provide everything we need for free, including box scores, play-by-play, lineups, and live game state. Both have mature Python wrappers (`nba_api`, `MLB-StatsAPI`, `pybaseball`).
- **No LLM inference** — gradient-boosted trees (XGBoost / LightGBM) dominate tabular sports prediction. Inference is microseconds and free after training.
- **Worker + queue (Celery + Redis)** — keeps ingest, training, and scoring off the API hot path; horizontally scalable when needed.
- **Full system upfront** — single coherent design, no throwaway scaffolding.
- **5 seasons backfill** — enough for robust training without era-shift contamination.
- **Backtesting from day one** — walk-forward CV, calibration, per-season metrics. Without it we cannot honestly call predictions "accurate."

Deployment is intentionally undecided; everything ships in Docker Compose so it can run on a laptop, a Hetzner VPS, or a cloud VM unchanged.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                       Celery Workers (queue: Redis)              │
│                                                                  │
│  ingest_nba   ingest_mlb   build_features   train_models   score │
└─────────┬────────┬─────────────┬───────────────┬────────────┬────┘
          │        │             │               │            │
          ▼        ▼             ▼               ▼            ▼
┌──────────────────────────────────────────────────────────────────┐
│         PostgreSQL 16 + TimescaleDB extension                    │
│  raw_* tables → feature_* tables → predictions, models, audit    │
└──────────────────────────────────────────────────────────────────┘
                                                  ▲
                                                  │
                          ┌───────────────────────┴────────────────┐
                          │      FastAPI (gunicorn + uvicorn)      │
                          │  API-key → JWT, slowapi rate limit     │
                          └───────────────────────┬────────────────┘
                                                  │
                                                  ▼
                          ┌────────────────────────────────────────┐
                          │  Notifier worker → Discord / Telegram  │
                          │  (PG LISTEN on predictions)            │
                          └────────────────────────────────────────┘

   Caddy (reverse proxy, auto-HTTPS) sits in front of FastAPI.
```

All services are containerised. `docker-compose.yml` declares: `postgres`, `redis`, `api`, `worker`, `beat` (Celery scheduler), `notifier`, `caddy`.

---

## Tech Stack

| Layer            | Choice                                          | Why                                                            |
|------------------|-------------------------------------------------|----------------------------------------------------------------|
| Language         | Python 3.11+                                    | Required for `nba_api`, `pybaseball`, XGBoost/LightGBM.        |
| Web framework    | FastAPI + uvicorn (gunicorn workers)            | Async, Pydantic validation, OpenAPI auto-docs.                 |
| DB               | PostgreSQL 16 + TimescaleDB ext                 | Strong relational + time-series hypertables for game stats.    |
| Cache / broker   | Redis 7                                         | Celery broker + slowapi rate-limit backend + dedup cache.      |
| Queue            | Celery 5 + celery-beat                          | Cron-like scheduling, retries, distributed workers.            |
| ORM / migrations | SQLAlchemy 2 + Alembic                          | Mature, async-capable.                                         |
| ML               | XGBoost (winner), LightGBM (props), scikit-learn| Best-in-class for tabular; quantile regression for props.      |
| Experiment track | MLflow (local file backend)                     | Versioned model registry, no external deps.                    |
| Auth             | API key (Argon2 hashed) → JWT HS256 (15-min)    | Standard secure pattern; revocable keys.                       |
| Rate limit       | slowapi + Redis                                 | Per-key sliding window.                                        |
| Reverse proxy    | Caddy                                           | Auto Let's Encrypt, HSTS, HTTP→HTTPS redirect out of the box.  |
| Secrets          | env vars + `.env` (gitignored)                  | Vault-pluggable later.                                         |
| Tests            | pytest + pytest-asyncio + Hypothesis            | Unit + property-based for feature math.                        |
| Lint / format    | ruff + mypy (strict on `src/core`, `src/api`)   | Fast, opinionated.                                             |

---

## Repository Layout

```
prediction/
├── docker-compose.yml
├── Dockerfile                    # multi-stage; same image runs api / worker / beat
├── Caddyfile
├── pyproject.toml                # ruff, mypy, pytest config
├── .env.example
├── alembic/
│   └── versions/
├── src/
│   ├── core/
│   │   ├── config.py             # Pydantic Settings — all env vars
│   │   ├── logging.py            # structlog JSON logs
│   │   ├── security.py           # Argon2, JWT encode/decode, API-key hashing
│   │   └── time.py               # tz-aware helpers; everything stored UTC
│   ├── db/
│   │   ├── session.py            # async + sync engines
│   │   ├── models/               # SQLAlchemy ORM models (see Schema below)
│   │   └── repositories/         # query helpers (no business logic)
│   ├── ingest/
│   │   ├── common.py             # rate-limited HTTP client, retry, UA rotation
│   │   ├── nba/
│   │   │   ├── client.py         # thin wrapper around nba_api endpoints
│   │   │   ├── games.py          # schedule + box score + play-by-play
│   │   │   ├── players.py        # roster + injury report PDF parser
│   │   │   └── live.py           # in-game updates during a game window
│   │   └── mlb/
│   │       ├── client.py         # statsapi + pybaseball (Statcast)
│   │       ├── games.py
│   │       ├── players.py        # IL transactions, lineups
│   │       └── live.py
│   ├── features/
│   │   ├── common.py             # as-of join utilities, rolling windows, Elo
│   │   ├── nba/
│   │   │   ├── team.py           # net-rating, pace, rest, travel, Elo
│   │   │   ├── matchup.py        # H2H, positional defense, lineup strength
│   │   │   └── player.py         # rolling avgs, usage when teammate in/out
│   │   └── mlb/
│   │       ├── team.py           # bullpen state, lineup wOBA, park factor
│   │       ├── matchup.py        # starter vs lineup, handedness, umpire
│   │       └── player.py         # batter splits, pitcher vs opp handedness
│   ├── models/
│   │   ├── registry.py           # MLflow wrapper; promote/rollback by metric
│   │   ├── train_winner.py       # XGBoost binary, isotonic calibration
│   │   ├── train_props.py        # LightGBM quantile regression per stat
│   │   ├── score.py              # load active model, batch score upcoming games
│   │   └── eval/
│   │       ├── walk_forward.py   # chronological CV split generator
│   │       ├── metrics.py        # log-loss, Brier, calibration error, MAE
│   │       └── report.py         # writes Markdown + plots to /reports/
│   ├── api/
│   │   ├── main.py               # FastAPI app, middleware
│   │   ├── deps.py               # auth dep, db session dep
│   │   ├── auth.py               # /v1/auth/token, key verification
│   │   ├── routes/
│   │   │   ├── games.py
│   │   │   ├── predictions.py
│   │   │   ├── models.py
│   │   │   └── health.py
│   │   └── schemas/              # Pydantic response/request models
│   ├── notify/
│   │   ├── listener.py           # PG LISTEN on `predictions_channel`
│   │   ├── discord.py            # webhook embed formatter
│   │   ├── telegram.py           # bot API, markdown formatter
│   │   └── dedup.py              # skip near-duplicate re-posts
│   ├── tasks/                    # Celery task definitions
│   │   ├── celery_app.py
│   │   ├── ingest_tasks.py
│   │   ├── feature_tasks.py
│   │   ├── train_tasks.py
│   │   ├── score_tasks.py
│   │   └── schedule.py           # beat_schedule dict
│   └── cli.py                    # Typer CLI: backfill, retrain, eval, keygen
├── tests/
│   ├── unit/
│   ├── integration/              # spins up a test PG + Redis via testcontainers
│   └── fixtures/                 # frozen JSON snapshots of NBA/MLB responses
├── reports/                      # generated backtest reports (gitignored)
├── scripts/
│   └── bootstrap_backfill.sh
└── docs/
    ├── api.md
    ├── data_sources.md
    └── ops.md
```

---

## Database Schema (key tables)

All times stored UTC. Use Timescale hypertables on `*_game_stats` for fast windowed aggregates.

```sql
-- Canonical entities
sports(id, code UNIQUE)  -- 'nba', 'mlb'
teams(id, sport_id, external_id UNIQUE, name, abbrev, conference, division, meta jsonb)
players(id, sport_id, external_id UNIQUE, full_name, primary_position, bats, throws, birthdate, meta jsonb)
venues(id, sport_id, name, city, lat, lon, indoor bool, meta jsonb)

-- Games + stats
games(id, sport_id, external_id UNIQUE, season, scheduled_utc, status,
      home_team_id, away_team_id, venue_id, home_score, away_score, meta jsonb,
      UNIQUE(sport_id, external_id))
team_game_stats(game_id, team_id, stats jsonb, PRIMARY KEY(game_id, team_id))  -- hypertable on scheduled_utc
player_game_stats(game_id, player_id, team_id, stats jsonb, PRIMARY KEY(game_id, player_id))  -- hypertable
plays(game_id, sequence, period, clock, event_type, payload jsonb)  -- play-by-play, hypertable
lineups(game_id, team_id, source, players jsonb, fetched_at)
injuries(player_id, status, reason, reported_at, expected_return_date, source)

-- Derived (rebuilt incrementally; never trust them to be present for any given as-of date)
team_features(team_id, as_of_utc, sport_id, features jsonb, PRIMARY KEY(team_id, as_of_utc))
player_features(player_id, as_of_utc, features jsonb, PRIMARY KEY(player_id, as_of_utc))
matchup_features(game_id, features jsonb, computed_at, PRIMARY KEY(game_id))

-- Model registry mirror (MLflow is source of truth; this is for joins)
models(id, sport_id, kind, target, version, mlflow_run_id, trained_at,
       active bool, metrics jsonb, feature_spec_hash)
predictions(id, game_id, model_id, target, value numeric, probability numeric,
            quantiles jsonb, features_hash, created_at,
            UNIQUE(game_id, model_id, target, COALESCE(player_id, 0)))
predictions_audit(prediction_id, raw_features jsonb, model_version, created_at)

-- Auth + audit
api_keys(id, name, key_hash, scopes text[], created_at, expires_at, revoked_at)
api_requests(id, api_key_id, route, status, latency_ms, ip, ts) -- partitioned by month
```

**Critical invariant**: every feature row has an `as_of_utc` and is built **only** from data whose `scheduled_utc < as_of_utc` (or `reported_at < as_of_utc` for injuries). This prevents lookahead leakage in training and backtests — the single biggest source of fake "accuracy" in sports models.

---

## Data Ingestion

### NBA — `stats.nba.com` via `nba_api`
- **Schedule + scores**: `LeagueGameFinder`, `ScoreboardV3`.
- **Box scores**: `BoxScoreTraditionalV3`, `BoxScoreAdvancedV3`.
- **Play-by-play**: `PlayByPlayV3`.
- **Injuries**: NBA daily injury report is published as PDF — parse with `pdfplumber`. (No JSON endpoint.)
- **Live**: `LiveScoreboard` polled every 30s during game windows.
- **Throttling**: 1 req/sec cap, rotating UA headers from a small allowlist, exponential backoff on 429. `nba_api` enforces this by default but we wrap it.

### MLB — `statsapi.mlb.com` + Baseball Savant
- **Schedule, lineups, live**: `MLB-StatsAPI` (`schedule`, `boxscore`, `linescore`, `live_feed`).
- **Statcast pitch-by-pitch**: `pybaseball.statcast` (queries Baseball Savant).
- **IL transactions**: `transactions` endpoint filtered on injured list moves.
- **Live**: GUMBO feed (`live_feed`) polled every 15s during game windows.
- **Throttling**: 5 req/sec cap; pybaseball Statcast queries chunked by date and cached on disk.

### Common ingest contract
`src/ingest/common.py` provides:
- `RateLimitedClient` — token-bucket HTTP wrapper with retry + jitter.
- `Upserter` — bulk PG upsert by `(sport_id, external_id)`.
- Every fetcher returns `IngestResult(rows_inserted, rows_updated, last_external_id)` so the scheduler can checkpoint.

### Scheduled jobs (Celery beat)
| Time (UTC)        | Task                          | Purpose                                    |
|-------------------|-------------------------------|--------------------------------------------|
| 09:00 daily       | `ingest_yesterday_results`    | Final box scores + play-by-play            |
| 11:00 daily       | `refresh_injuries`            | NBA injury PDF + MLB IL moves              |
| 12:00 daily       | `rebuild_features_incremental`| Feature rows for upcoming-game teams       |
| 13:00 daily       | `score_upcoming_games`        | Run inference, write predictions, NOTIFY   |
| Every 15 min      | `score_lineup_changes`        | Re-score if confirmed lineups arrived      |
| Every 30s in-window | `live_polling_nba/mlb`       | Live score updates only (no re-score v1)   |
| 02:00 daily       | `train_challenger`            | Nightly retrain from scratch, log as candidate |
| 03:00 daily       | `drift_monitor`               | Performance/calibration/PSI checks; may trigger priority retrain |
| Mon 04:00 weekly  | `evaluate_and_promote`        | Gate last week's challengers; promote best if it passes |
| Sun 03:00 weekly  | `full_backtest_report`        | Regenerate evaluation report               |
| 1st of month 05:00| `full_hyperparam_search`      | Optuna re-tune of hyperparams + recency λ  |

---

## Feature Engineering

Features are recomputed **as-of** `(game.scheduled_utc - 1 hour)` for training and inference alike. Same code path both ways — no train/serve skew.

### NBA — game-winner features (~60)
- Elo with home-court adjustment (HCA tuned per season).
- Rolling **net rating** windows: last 5 / 10 / 20 / season-to-date.
- Pace-adjusted offensive & defensive efficiency.
- Rest days; back-to-back flag; **3-in-4** and **4-in-6** flags (high signal).
- Travel distance from previous game venue (haversine on `venues.lat/lon`).
- Strength-of-schedule-adjusted last-N (SRS-style).
- Head-to-head last 5 meetings.
- Home/away splits.
- Starting-five availability ratio: `sum(usage of available starters) / sum(usage of full starting 5)` derived from injury report + recent lineup data — captures Kawhi-load-management style absences.

### NBA — player props features (~40 per player)
- Rolling means (5/10/20) for the prop stat.
- Rolling **per-minute** rate (decouples from minutes).
- Opponent defensive rating **at player's position** (DvP).
- Usage when key teammate is **in vs. out** (e.g., your usage with/without your star) — biggest signal in props.
- Projected minutes: estimated from rotation patterns + injury-adjusted depth chart.
- Home/away split.
- Days rest.

### MLB — game-winner features (~70)
- **Starting pitcher quality**: xFIP, K-BB%, SIERA, last-3-start rolling.
- **Bullpen state**: aggregate WAR available × rest (pitchers used in last 2 days flagged).
- Team **wOBA vs. handedness** of opposing starter (L/R splits).
- Park factor (run environment) from FanGraphs reference table.
- Days rest, travel distance.
- Lineup projected wOBA (sum of confirmed-lineup batters' rolling wOBA).
- Elo with starter-quality adjustment.
- Umpire strike-zone tendency (called K% above/below avg).
- Weather: wind speed/direction, temperature for outdoor venues (NWS free API).

### MLB — player props features (~50 per player)
- Batter splits vs. pitcher handedness (regressed toward league mean if PA < 50).
- Recent form: 14-day rolling Statcast (xwOBA, barrel%, hard-hit%).
- Ballpark factor for the specific stat (HR factor ≠ hit factor).
- Lineup spot (drives plate-appearance volume).
- Wind direction relative to ballpark orientation (HR props only).
- Pitcher K% vs. opposing-handedness lineup (for pitcher K props).
- Opposing bullpen K% (late-game PAs).

---

## Models

### Winner model (per sport)
- **Algorithm**: XGBoost binary classifier → isotonic calibration on holdout fold.
- **Target**: `home_team_won`.
- **Output**: `P(home_win)` with calibration error < 5% on holdout.
- **Loss**: log-loss; secondary metric Brier score (proper scoring rule).
- **Hyperparameters**: tuned via Optuna with walk-forward objective (50 trials, time budget cap).
- **Realistic accuracy bar**: NBA ~67–69% (Vegas ~70%), MLB ~56–58% (Vegas ~58%). Anything claiming higher in backtest is a leakage red flag — the eval harness explicitly checks for this.

### Player props (per stat per sport)
- **Algorithm**: LightGBM **quantile regression** at q = [0.10, 0.25, 0.50, 0.75, 0.90]. Gives full predictive distribution, not just a point estimate.
- **Stats v1**:
  - NBA: PTS, REB, AST, 3PM, PRA combined.
  - MLB hitter: H, HR, TB, RBI, K. Pitcher: K, OUTS, ER.
- **Output**: predicted median + quantile band. Implied `P(over X.5)` for any line interpolated from quantiles.
- **Metric**: pinball loss per quantile; MAE on median; coverage check (does the 80% band contain the truth ~80% of the time?).

### Model registry & promotion
- MLflow tracks every training run with full metric set, feature spec hash, training data range.
- `train_winner.py --promote` only sets `models.active = true` if **new model beats current on out-of-time holdout by ≥ 1% log-loss reduction** AND calibration stays within bounds.
- One-command rollback (`cli.py model rollback <sport> <kind>`).

### Continuous Learning Strategy — staying current without forgetting

**The honest framing**: XGBoost and LightGBM are *not* online learners. There is no good way to "stream new games into the existing trees." Their `xgb_model=<prev>` continuation mode exists but is fragile (schema drift, calibration drift, no way to down-weight old trees). Every serious sports-prediction shop solves freshness the same way: **scheduled retraining from scratch on a rolling window, with sample weights that favor recent games**, plus a champion/challenger workflow so freshness never costs accuracy.

This gives both properties at once:
- **"Remembers old data"** — old seasons stay in the training window, just at lower weight, so the model still learns long-horizon patterns (park factors, persistent team identities, era-level Elo priors).
- **"Stays current"** — recent games dominate the loss, weekly retrains pick up roster moves and tactical shifts, and drift detection forces unscheduled retrains when reality changes faster than the schedule.

#### 1. Rolling-window training data
- **Winner models**: train on the last **5 seasons + current season to date**. Each new training run slides the window forward; oldest season drops off naturally.
- **Props models**: tighter window — **3 seasons + current** — because individual-player behavior changes faster than team-level dynamics (aging, role changes, injuries).
- Configurable per model in `src/models/configs/`.

#### 2. Recency sample weighting
- `sample_weight = exp(-λ · days_since_game / 365)` passed to XGBoost/LightGBM.
- λ tuned per sport/target during walk-forward CV (typical: NBA winner ~0.3, MLB winner ~0.2, NBA props ~0.5, MLB props ~0.4 — higher = forgets faster).
- This is the actual mechanism that makes "old data remembered, new data prioritized" mathematically true rather than vibes.

#### 3. Retraining cadence (driven by Celery beat)
| Cadence              | Job                                | Purpose                                                              |
|----------------------|------------------------------------|----------------------------------------------------------------------|
| **Nightly**          | `train_challenger`                 | Retrain from scratch on latest window. Logs to MLflow as candidate.  |
| **Weekly (Mon 04:00)** | `evaluate_and_promote`           | Compare last 7 nightly challengers vs. champion on rolling holdout. Promote if best beats champion by threshold AND calibration holds. |
| **On drift signal**  | `train_challenger` (priority queue)| Triggered by drift detector (see #4). Out-of-band retrain.           |
| **Monthly**          | `full_hyperparam_search`           | Optuna re-tune of hyperparameters + recency λ on full walk-forward.  |
| **Off-season transition** | `cold_start_retrain`          | Hard retrain when new season opens; resets some priors (rookies, rule changes). |

Nightly retrains are cheap (~3–10 min per model on a modest CPU; GBDTs are fast). Producing many challengers and promoting selectively is the central trick: **freshness is free, regressions are gated**.

#### 4. Drift detection (when scheduled cadence isn't enough)
A separate `drift_monitor` Celery task runs daily after scoring:
- **Performance drift**: rolling 30-game log-loss of the live champion vs. its backtest baseline. > 10% degradation → fire `train_challenger` immediately with `priority=high`.
- **Calibration drift**: bucket recent predictions by predicted probability, compare actual win rates to expected. Expected Calibration Error > 0.07 → priority retrain.
- **Feature distribution drift**: per-feature **Population Stability Index** (PSI) between current week and training window. Any feature with PSI > 0.25 → log + flag; > 0.5 → priority retrain.
- **Concept drift**: when the *relationship* between features and outcomes shifts (rule changes, new 3-point era, dead-ball era in MLB). Detected by comparing partial-dependence of top features to a baseline snapshot.

All drift events are written to a `drift_events` table and surfaced via `/v1/health` and the Discord ops channel.

#### 5. Champion/challenger promotion gate (the safety valve)
A nightly challenger is **only** promoted if all four hold on the rolling out-of-time holdout (last 4 weeks of completed games, never used in training):
1. Log-loss reduction ≥ 1% vs. current champion.
2. Expected Calibration Error ≤ champion's ECE + 0.02.
3. No feature has shifted importance by > 50% (catches "model learned a noisy signal").
4. Brier score not worse than champion.

If a challenger fails the gate, it's archived for inspection but never serves traffic. If three consecutive nightly challengers fail, the system pages the ops Discord channel — usually means feature pipeline is broken, not the model.

#### 6. Player-specific cold-start handling (props)
Props models hit a real cold-start problem: rookies have no history, traded players have unstable usage, returning-from-injury players are different than they were. Strategy:
- **Hierarchical fallback**: if player has < 20 games this season, blend prediction toward `(positional_archetype_prior, player_history)` with weight = `min(1, games_played / 20)`.
- **Trade / injury-return reset**: when a player changes teams or returns from IL, recency weight λ for that player is temporarily boosted (forget pre-event data faster) for the next 15 games.
- These rules live in `src/features/{nba,mlb}/player.py` and are part of the feature, not the model — keeps the model simple.

#### 7. Why not actually online?
For completeness — we considered and rejected:
- **`xgb_model=` continuation**: doesn't down-weight old trees, calibration drifts, schema changes break it.
- **River / online SGD**: works, but ~5–10% accuracy hit on tabular sports data vs. GBDT. Not worth it.
- **Small neural net with online fine-tuning**: even larger accuracy hit, more ops complexity, no calibration story.
- **Bayesian online learning (e.g., dynamic Elo only)**: we *do* use this for the Elo *feature*, but the full model needs the GBDT interaction-finding power.

The chosen pattern (frequent rolling-window retrains + recency weights + champion/challenger) is what FiveThirtyEight, Pinnacle's research papers, and basketball-reference's RAPM-based models all converge on. It's the production answer for tabular sports prediction in 2026.

### Backtesting harness (`src/models/eval/`)
- **Walk-forward CV**: train on seasons [N…N+k], validate on N+k+1, slide forward. No shuffling, ever.
- Outputs to `reports/{sport}_{kind}_{date}.md`:
  - Per-season log-loss, Brier, accuracy, calibration curve.
  - Performance vs. closing Vegas line if available (we'll scrape closing odds from `the-odds-api.com` free tier just for evaluation, never for training).
  - Feature importance + SHAP summary.
  - "Sanity checks": predictions on shuffled labels (should be ~50%), predictions with future data masked (should match live).

---

## REST API

OpenAPI auto-generated at `/docs` (auth-gated in production).

### Endpoints
```
POST /v1/auth/token                       # body: {api_key} → {access_token, expires_in}
GET  /v1/sports                            # list available sports + last-trained model versions
GET  /v1/games/upcoming?sport=nba&hours=48 # scheduled games in window
GET  /v1/games/{game_id}                   # game detail + current status
GET  /v1/predictions/game/{game_id}        # win/loss prediction + features hash
GET  /v1/predictions/props/{game_id}       # all player props for a game
GET  /v1/predictions/player/{player_id}?game_id=...  # single player props
GET  /v1/models/active?sport=nba           # which model versions are serving
GET  /v1/health                            # 200 if DB + Redis + last-ingest within SLO
```

All responses include `model_version`, `as_of_utc`, and `features_hash` so clients can detect stale predictions.

### Security (defence in depth)
1. **Transport**: Caddy enforces HTTPS, HSTS preload, TLS 1.3 only. HTTP → 308 redirect.
2. **Auth**: long random API keys (32 bytes, base64). Stored **Argon2id** hashed. Keys exchanged for **JWT (HS256, 15-min TTL)**. Refresh requires re-auth.
3. **Authorization**: scopes per key (`predictions:read`, `models:read`, `admin`). Admin endpoints require an additional IP allowlist check.
4. **Rate limit**: slowapi with Redis backend; sliding window per key. Defaults: 60 req/min, 1000 req/hour, configurable per key.
5. **Validation**: every request body/query through Pydantic with strict types; reject unknown fields.
6. **Headers**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, restrictive CSP, no `Server` leak.
7. **CORS**: deny by default; allowlist explicit origins via env.
8. **Audit log**: every authenticated request → `api_requests` (route, status, latency, key id, IP). Async write, never blocks response.
9. **Secrets**: `.env` for dev, real env vars in prod. Never logged. Pydantic Settings marks them `SecretStr`.
10. **DB**: app role has no DDL; migrations run from a separate role.
11. **Container hardening**: non-root user, read-only root FS, drop all Linux capabilities, healthchecks.
12. **Dependency scanning**: `pip-audit` in CI; renovate-style updates monthly.
13. **Key management**: `cli.py keys create/revoke/list`; revocation is instant (JWT blacklist via Redis with TTL = JWT TTL).

---

## Notifier (Discord + Telegram)

Standalone process. Uses Postgres `LISTEN/NOTIFY` on a `predictions_channel` published by an `AFTER INSERT` trigger on `predictions`. No polling.

- **Discord**: outbound webhook(s) per channel/sport. Rich embeds: game header, win prob with progress bar, top 3 props with confidence (90% band width).
- **Telegram**: bot token + chat id(s) via `python-telegram-bot`. Markdown V2 formatting.
- **Dedup** (`notify/dedup.py`): skip if same `(game_id, target)` was posted within last 6h **and** probability shifted < 3pp. Confirmed-lineup updates always re-post (tagged "LINEUP UPDATE").
- **Outbound is fire-and-forget with retry**: failures logged, never block the prediction pipeline.
- Webhook URLs / bot tokens are config — multiple destinations per sport supported.

---

## Operational Concerns

- **Logging**: structlog JSON to stdout → Docker → host log aggregator (Loki-friendly).
- **Metrics**: Prometheus `/metrics` endpoint on API + worker. Track: ingest lag, last-trained-at, prediction count, API latency p50/p95/p99, model calibration drift (weekly batch).
- **Backups**: nightly `pg_dump` to local + (optional) S3-compatible bucket. Retention 30d.
- **Health SLOs** (surfaced by `/v1/health`):
  - Last NBA ingest < 25h.
  - Last MLB ingest < 25h (in season).
  - Active model trained within last 14d.
  - DB + Redis reachable within 100ms.
- **Disaster recovery**: full system reproducible from `git clone && docker compose up` + restoring PG dump + MLflow artifact dir.

---

## Bootstrap / Backfill

Single command: `python -m src.cli backfill --sport nba --seasons 5 && python -m src.cli backfill --sport mlb --seasons 5`.

- Idempotent: resumes from last successful date per sport.
- Rate-limited and chunked (chunks per month).
- Estimated runtime on a typical VPS: NBA ~3–4h (5 seasons), MLB ~6–8h (5 seasons including Statcast). Runs as Celery tasks so progress is observable.
- After backfill: `python -m src.cli train --sport nba --kind winner` → initial models → `cli.py model promote` if eval passes.

---

## Honesty Bar

This plan does **not** promise to beat Vegas. It promises:
- Predictions calibrated within 5% of true frequencies.
- Accuracy in the realistic ranges above (NBA 67–69%, MLB 56–58% on moneyline).
- A backtest report you can read to decide whether the model is actually any good before trusting it.
- Architecture that lets us add: ensembles, in-game re-scoring, line-shopping evaluation, additional sports — all without rework.

---

## Verification Plan

End-to-end checks before declaring the system ready:

1. **`docker compose up` from clean clone** brings up all services healthy in < 90s; `curl /v1/health` returns 200 with subsystem detail.
2. **Backfill smoke test**: `cli.py backfill --sport nba --seasons 1 --dry-run` exits 0 and reports row counts; full 5-season backfill completes without rate-limit errors.
3. **Feature parity test** (`tests/integration/test_feature_parity.py`): features computed at training time for game G equal features computed at inference time for game G when both use the same as-of timestamp. **Hard failure** if not bit-identical — guarantees no train/serve skew.
4. **Leakage test** (`tests/integration/test_no_leakage.py`): for a random sample of 100 historical games, recompute features with all post-game data scrubbed; assert features are bit-identical to the originals.
5. **Backtest report** runs end-to-end and produces `reports/nba_winner_<date>.md` with calibration plot and per-season metrics; metrics fall in the expected ranges.
6. **API auth flow**: `cli.py keys create` → POST `/v1/auth/token` → GET `/v1/predictions/game/{id}` with bearer → 200. Same call without token → 401. Expired token → 401. Wrong scope → 403.
7. **Rate limit**: 70 requests/min from one key triggers 429 by request 61.
8. **Security headers**: `curl -I` shows HSTS, X-Content-Type-Options, no Server header.
9. **Notifier dry-run** mode posts to a test Discord channel and a test Telegram chat with realistic payloads.
10. **Disaster recovery**: drop DB volume, restore from last `pg_dump`, all predictions for upcoming games still resolvable; new predictions resume on next beat tick.
11. **Continuous-learning loop**: nightly `train_challenger` produces a MLflow run with metrics; `evaluate_and_promote` correctly refuses to promote a deliberately-broken challenger (test injects one with shuffled labels) and correctly promotes one trained on +7 days of new data when it beats champion. `drift_monitor` fires a priority retrain when fed synthetic feature-drift data (test fixture).

The plan succeeds when all eleven of the above pass on a clean machine.
