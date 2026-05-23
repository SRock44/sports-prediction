# Prediction Engine — Architecture Diagrams

> Custom-built end-to-end ML system for NBA & MLB game outcome and player prop prediction.
> No third-party prediction API. Every component — data ingest, feature engineering, model training, serving, and drift monitoring — is built in-house.

---

## 1. System Overview

```mermaid
graph TB
    subgraph External["External Data Sources"]
        NBA_API["NBA Official API<br/>(stats.nba.com)"]
        MLB_API["MLB StatsAPI<br/>(statsapi.mlb.com)"]
        DK["DraftKings<br/>(web scrape)"]
        FD["FanDuel<br/>(web scrape)"]
        FAN["Fanatics<br/>(web scrape)"]
        KALSHI["Kalshi<br/>(public API)"]
    end

    subgraph Ingest["Ingest Layer (src/ingest/)"]
        NBA_ING["NBA Ingest<br/>games, players, live"]
        MLB_ING["MLB Ingest<br/>games, players, weather"]
        ODDS_ING["Odds Ingest<br/>consensus lines + Kalshi"]
    end

    subgraph Storage["Storage Layer"]
        PG[("PostgreSQL<br/>TimescaleDB<br/>20+ tables")]
        REDIS[("Redis<br/>Celery broker<br/>+ cache")]
    end

    subgraph Features["Feature Engineering (src/features/)"]
        FEAT_NBA["NBA Features<br/>matchup.py · team.py · player.py"]
        FEAT_MLB["MLB Features<br/>matchup.py · team.py · player.py"]
        ELO["Elo Rating<br/>compute_elo_series()"]
    end

    subgraph Training["Training Pipeline (src/models/)"]
        WINNER["Winner Model<br/>XGBoost + LightGBM<br/>Isotonic Calibration"]
        PROPS["Props Model<br/>LightGBM Quantile<br/>QuantileBundle"]
        OPTUNA["Optuna HPO<br/>200–500 trials<br/>Walk-Forward CV"]
        MLFLOW[("MLflow<br/>Model Registry")]
    end

    subgraph Serving["Scoring & Serving (src/models/score.py + src/api/)"]
        BATCH["Batch Scorer<br/>score_upcoming_games()"]
        API["FastAPI<br/>REST API<br/>/v1/predictions/*"]
    end

    subgraph Ops["Operations (src/tasks/ + src/notify/)"]
        CELERY["Celery Beat<br/>Scheduled Tasks"]
        NOTIFY["Notifier<br/>Discord · Telegram"]
        DRIFT["Drift Monitor<br/>LogLoss · ECE · PSI"]
    end

    subgraph Deploy["Deployment"]
        DOCKER["Docker Compose<br/>api · worker · beat · notifier · caddy"]
        CADDY["Caddy<br/>Reverse Proxy<br/>TLS termination"]
        PROM["Prometheus<br/>/metrics endpoint"]
    end

    NBA_API --> NBA_ING
    MLB_API --> MLB_ING
    DK & FD & FAN & KALSHI --> ODDS_ING

    NBA_ING & MLB_ING & ODDS_ING --> PG

    PG --> FEAT_NBA & FEAT_MLB
    ELO --> FEAT_NBA & FEAT_MLB
    FEAT_NBA & FEAT_MLB --> PG

    PG --> WINNER & PROPS
    OPTUNA --> WINNER & PROPS
    WINNER & PROPS --> MLFLOW

    MLFLOW --> BATCH
    PG --> BATCH
    BATCH --> PG

    PG --> API
    MLFLOW --> API

    CELERY --> NBA_ING & MLB_ING & ODDS_ING
    CELERY --> FEAT_NBA & FEAT_MLB
    CELERY --> BATCH
    CELERY --> DRIFT & WINNER

    PG -->|LISTEN/NOTIFY| NOTIFY
    DRIFT --> REDIS

    DOCKER --> CADDY --> API
    DOCKER --> PROM

    style External fill:#f0f0f0
    style Training fill:#e8f4fd
    style Serving fill:#e8fde8
    style Ops fill:#fdf8e8
```

---

## 2. Data Ingestion Pipeline

```mermaid
flowchart TD
    subgraph Sources["External Sources"]
        N1["nba_api<br/>(stats.nba.com)"]
        M1["MLB-StatsAPI<br/>(statsapi.mlb.com)"]
        O1["DraftKings API"]
        O2["FanDuel API"]
        O3["Fanatics API"]
        O4["Kalshi Public API"]
    end

    subgraph RateLimit["src/ingest/common.py"]
        RL["RateLimitedClient<br/>1 req/sec · UA rotation<br/>Tenacity retry"]
        IR["IngestResult<br/>rows_inserted · errors"]
    end

    subgraph NBA["src/ingest/nba/"]
        NS["ingest_season_schedule()<br/>games table"]
        NB["ingest_box_scores()<br/>team_game_stats<br/>player_game_stats"]
        NP["sync_teams() / sync_players()<br/>teams · players tables"]
        NL["poll_live_nba()<br/>every 2 min · live games"]
    end

    subgraph MLB["src/ingest/mlb/"]
        MS["ingest_season_schedule()"]
        MB["ingest_box_scores()"]
        MP["sync_teams() / sync_players()"]
        MW["ingest_mlb_weather()"]
        ML["poll_live_mlb()"]
    end

    subgraph Odds["src/ingest/odds/"]
        OC["get_consensus_lines()<br/>average DK + FD + Fanatics"]
        OK["get_kalshi_probs()<br/>prediction market"]
        OA["american_to_implied_prob()<br/>moneyline ↔ probability"]
    end

    subgraph DB["PostgreSQL"]
        G["games"]
        TGS["team_game_stats<br/>⏱ hypertable"]
        PGS["player_game_stats<br/>⏱ hypertable"]
        T["teams"]
        P["players"]
        GO["game_odds"]
    end

    subgraph Schedule["Celery Beat Schedule"]
        S1["09:00 UTC — box scores"]
        S2["11:00 UTC — injuries"]
        S3["14:00 + 22:00 UTC — odds"]
        S4["18:00 UTC — MLB weather"]
        S5["every 2 min — live poll"]
    end

    N1 --> RL --> NS & NB & NP
    N1 --> NL
    M1 --> RL --> MS & MB & MP & MW
    M1 --> ML
    O1 & O2 & O3 --> OC --> OA
    O4 --> OK

    NS --> G
    NB --> TGS & PGS
    NP --> T & P
    MS --> G
    MB --> TGS & PGS
    MP --> T & P
    OA & OK --> GO

    IR -.->|accumulates| NS & NB & NP & MS & MB

    S1 -->|triggers| NB & MB
    S2 -->|triggers| NP & MP
    S3 -->|triggers| OC & OK
    S4 -->|triggers| MW
    S5 -->|triggers| NL & ML

    style DB fill:#e8f4fd
    style Schedule fill:#fdf8e8
```

---

## 3. Feature Engineering Pipeline

```mermaid
flowchart TD
    subgraph DB["PostgreSQL"]
        G["games<br/>(scheduled_utc, home/away team)"]
        TGS["team_game_stats<br/>⏱ last 30 games per team"]
        PGS["player_game_stats<br/>⏱ last 30 games per player"]
        T["teams · venues"]
        GO["game_odds<br/>(moneylines, spreads)"]
    end

    subgraph Common["src/features/common.py"]
        LB["load_team_game_stats_before()<br/>as_of_utc cutoff — anti-leakage"]
        LP["load_player_game_stats_before()<br/>as_of_utc cutoff — anti-leakage"]
        ELO["compute_elo_series(games_df)<br/>chronological replay<br/>K=20, HCA=+50 pts"]
        RW["rolling_mean(values, window)<br/>5, 10, 20 game windows"]
        EXP["exponential_decay_weight(days_ago, λ)<br/>λ=0.55 NBA · λ=0.30 MLB"]
        HAV["haversine(lat1,lon1,lat2,lon2)<br/>travel distance km"]
    end

    subgraph TeamFeat["src/features/nba/team.py<br/>build_team_features()"]
        TF1["Efficiency<br/>net_rtg · off_rtg · def_rtg<br/>last 5/10/20 games"]
        TF2["Schedule Load<br/>rest_days · b2b<br/>3-in-4 · 4-in-6"]
        TF3["Form<br/>win_pct last 5/10/20<br/>streak · sos_diff"]
        TF4["Advanced<br/>pace · ts_pct · tov_rate<br/>oreb_pct · starter_availability"]
        TF5["Elo Rating<br/>from compute_elo_series()"]
    end

    subgraph MatchFeat["src/features/nba/matchup.py<br/>build_matchup_features()"]
        MF1["Differentials<br/>elo_diff · net_rtg_diff<br/>rest_diff · sos_diff"]
        MF2["Head-to-Head<br/>h2h_home_wins (last 5)<br/>h2h_home_win_pct"]
        MF3["Geography<br/>travel_km · tz_change_away<br/>venue_hca"]
        MF4["Market Signals<br/>odds_implied_home<br/>odds_line_move"]
        MF5["Roster<br/>roster_star_pts_diff<br/>roster_depth_diff"]
        MF6["Elo Probability<br/>elo_home_win_prob"]
    end

    subgraph PlayerFeat["src/features/nba/player.py<br/>build_player_features()"]
        PF1["Recent Form<br/>stat_last_5 · stat_last_10<br/>stat_season_avg"]
        PF2["Usage<br/>minutes · usage_rate<br/>home_away split"]
        PF3["Opponent Defense<br/>opp_def_rtg_vs_position"]
        PF4["Context<br/>game_pace · injury_status<br/>rest_days"]
    end

    subgraph Output["Feature Store (PostgreSQL)"]
        MFT["matchup_features table<br/>JSONB · 70+ features per game"]
        TFT["team_features table<br/>JSONB · per team per as_of"]
        PFT["player_features table<br/>JSONB · per player per as_of"]
    end

    G -->|game metadata| MatchFeat
    TGS -->|via LB| TeamFeat
    PGS -->|via LP| PlayerFeat
    T -->|venue coords| HAV
    GO -->|implied prob| MF4

    LB --> TF1 & TF2 & TF3 & TF4
    ELO --> TF5
    RW --> TF1 & TF3
    HAV --> MF3
    EXP -->|sample weights| Output

    TeamFeat -->|home + away features| MatchFeat
    MatchFeat --> MFT
    TeamFeat --> TFT
    PlayerFeat --> PFT

    style Output fill:#e8fde8
    style Common fill:#fdf8e8
```

---

## 4. Training Pipeline

```mermaid
flowchart TD
    subgraph Input["Training Data (from PostgreSQL)"]
        MF["matchup_features JSONB<br/>70+ features per game"]
        GT["games table<br/>home_score · away_score → home_won label"]
        SPLIT["Hold-out Split<br/>most recent season = holdout<br/>remaining = training"]
    end

    subgraph WalkForward["Walk-Forward Cross-Validation<br/>src/models/eval/walk_forward.py"]
        WF1["Fold 1: seasons 1-2 → season 3"]
        WF2["Fold 2: seasons 1-3 → season 4"]
        WF3["Fold 3: seasons 1-4 → season 5"]
        WFN["Fold N: seasons 1-(N-1) → season N"]
        GUARD["Anti-leakage assertion:<br/>max(train_date) < min(val_date)"]
    end

    subgraph Optuna["Optuna Hyperparameter Search<br/>200–500 trials (winner) · 30 trials (props)"]
        OBJ["Objective: weighted log-loss<br/>5-fold CV score"]
        SEARCH["Search Space<br/>n_estimators: 200–3000<br/>max_depth: 3–8<br/>learning_rate: 0.005–0.3<br/>subsample: 0.5–1.0<br/>colsample_bytree: 0.5–1.0<br/>reg_alpha / reg_lambda<br/>min_child_weight: 1–50"]
        PRUNE["Median Pruner<br/>stop bad trials early"]
    end

    subgraph XGB["XGBoost Training"]
        XFull["Train on full training set<br/>with best Optuna params"]
        XES["Early Stopping<br/>vs. calibration set<br/>max 5000 trees"]
        XModel["XGBClassifier<br/>eval_metric='logloss'"]
    end

    subgraph LGB["LightGBM Training (optional)"]
        LFull["Parallel Optuna search<br/>30-200 trials"]
        LModel["LGBMClassifier<br/>40% ensemble weight"]
    end

    subgraph Ensemble["IsotonicCalibratedEnsemble<br/>train_winner.py"]
        ENS["predict_proba(X):<br/>0.60 × XGB_prob + 0.40 × LGB_prob"]
        ISO["Isotonic Regression<br/>fit on held-out calibration set<br/>clips to 1e-6 .. 1.0-1e-6"]
        FINAL["Calibrated probability<br/>P(home_win)"]
    end

    subgraph Props["QuantileBundle (train_props.py)"]
        QMOD["5 × LightGBM Regressors<br/>quantile loss (α=0.10,0.25,0.50,0.75,0.90)"]
        QOPT["Optuna: minimize MAE on q=0.50"]
        QINF["implied_over_probability(line)<br/>interpolate P(stat > line) from CDF"]
    end

    subgraph Metrics["Evaluation Metrics<br/>src/models/eval/metrics.py"]
        M1["logloss (primary)"]
        M2["Brier score"]
        M3["Accuracy"]
        M4["ECE — Expected Calibration Error"]
        M5["Pinball loss (props)"]
        M6["Coverage — 80% interval"]
    end

    subgraph Promote["Promotion Gate<br/>should_promote()"]
        P1["logloss improvement ≥ 0.005"]
        P2["ECE increase ≤ 0.02"]
        P3["Brier score not worse"]
        PASS["All gates pass → promote"]
        FAIL["Any gate fails → keep champion"]
    end

    subgraph Registry["MLflow Model Registry<br/>src/models/registry.py"]
        LOG["log_model_run()<br/>model artifact + feature_names.json<br/>params + metrics + tags"]
        PROM["promote_model()<br/>deactivate champion<br/>activate challenger (ModelRecord)"]
        ROLL["rollback_model()<br/>revert to previous version"]
    end

    MF & GT --> SPLIT
    SPLIT -->|training seasons| WalkForward
    SPLIT -->|calibration set| ISO

    WalkForward --> Optuna
    GUARD -.->|validates| WalkForward

    Optuna --> OBJ
    OBJ --> SEARCH
    PRUNE --> Optuna

    SEARCH -->|best params| XFull
    XFull --> XES --> XModel
    SEARCH -->|best params| LFull --> LModel

    XModel & LModel --> ENS
    ENS --> ISO --> FINAL

    Optuna -->|best params| QMOD
    QOPT --> Optuna
    QMOD --> QINF

    FINAL & QINF --> Metrics

    Metrics --> Promote
    P1 & P2 & P3 --> PASS & FAIL
    PASS --> PROM
    FAIL -.->|challenger archived| LOG

    FINAL --> LOG
    PROM --> Registry
    ROLL --> Registry

    style Optuna fill:#fff3e0
    style Ensemble fill:#e8f4fd
    style Props fill:#e8fde8
    style Registry fill:#f3e8fd
```

---

## 5. Live Inference & API Flow

```mermaid
sequenceDiagram
    participant CB as Celery Beat<br/>(13:00 UTC)
    participant SC as score.py<br/>score_upcoming_games()
    participant FE as features/nba/<br/>matchup.py
    participant PG as PostgreSQL
    participant MF as MLflow<br/>Model Registry
    participant NT as Notifier<br/>(Discord/Telegram)
    participant CL as API Client

    Note over CB,SC: Batch scoring triggered daily
    CB->>SC: score_nba_upcoming task
    SC->>PG: query games WHERE scheduled_utc ∈ [now, +48h]<br/>AND status = 'scheduled'
    PG-->>SC: list of upcoming games

    loop For each game
        SC->>FE: build_matchup_features(session, game,<br/>as_of = scheduled_utc - 1h)
        FE->>PG: load_team_game_stats_before(team_id, as_of)
        PG-->>FE: last 30 games (before cutoff)
        FE->>FE: compute rolling stats, Elo,<br/>rest days, travel, odds
        FE-->>SC: feature dict (70+ keys)

        SC->>PG: load active ModelRecord<br/>WHERE sport='nba' AND kind='winner' AND active=True
        PG-->>SC: mlflow_run_id

        SC->>MF: load_model(run_id)
        MF-->>SC: IsotonicCalibratedEnsemble

        SC->>SC: X = feature_dict → numpy array<br/>proba = model.predict_proba(X)[0,1]

        SC->>PG: UPSERT predictions<br/>(game_id, model_id, value=proba,<br/>features_hash, created_at)
        SC->>PG: NOTIFY predictions_channel
    end

    PG-->>NT: LISTEN fires on predictions_channel
    NT->>NT: dedup check (Redis)
    NT->>NT: send Discord webhook<br/>+ Telegram message

    Note over CL,PG: Client fetches predictions via REST API

    CL->>+API: GET /v1/predictions/game/{game_id}<br/>Authorization: Bearer <token>
    API->>API: verify JWT / API key hash
    API->>PG: SELECT * FROM predictions<br/>JOIN models WHERE game_id = X
    PG-->>API: probability + model metadata
    API-->>-CL: JSON { home_win_probability: 0.58,<br/>model_version, as_of_utc, features_hash }
```

---

## 6. Model Architecture Detail

```mermaid
graph TB
    subgraph WinnerModel["Winner Model — IsotonicCalibratedEnsemble"]
        subgraph Input["Input: 70+ Features (numpy array)"]
            F1["Efficiency: net_rtg, off_rtg, def_rtg (×3 windows)"]
            F2["Schedule: rest_days, b2b, streak, win_pct"]
            F3["Matchup: elo_diff, net_rtg_diff, rest_diff"]
            F4["Market: odds_implied_home, line_move"]
            F5["Context: travel_km, tz_change, venue_hca"]
            F6["Roster: starter_availability, depth_diff"]
        end

        subgraph XGB["XGBoost (Primary, 60%)"]
            X1["n_estimators: 200–3000<br/>max_depth: 3–8<br/>learning_rate: 0.005–0.3"]
            X2["subsample · colsample_bytree<br/>reg_alpha · reg_lambda<br/>min_child_weight"]
            X3["predict_proba(X) → p_xgb"]
        end

        subgraph LGB["LightGBM (Secondary, 40%)"]
            L1["num_leaves: 31–127<br/>learning_rate: 0.01–0.2<br/>feature_fraction: 0.5–1.0"]
            L2["bagging_fraction · bagging_freq<br/>min_child_samples: 5–50"]
            L3["predict_proba(X) → p_lgb"]
        end

        ENS["Ensemble<br/>p_raw = 0.60 × p_xgb + 0.40 × p_lgb"]

        subgraph ISO["Isotonic Calibration"]
            IC["IsotonicRegression()<br/>fit on held-out calibration set<br/>monotone non-decreasing"]
            CLIP["clip(p, 1e-6, 1-1e-6)"]
        end

        OUT_W["P(home_win) ∈ (0, 1)<br/>well-calibrated probability"]
    end

    subgraph PropsModel["Props Model — QuantileBundle"]
        subgraph PInput["Input: 40 Player Features"]
            PF1["Recent: stat_last_5, stat_last_10, stat_season_avg"]
            PF2["Usage: minutes, usage_rate, home_away_split"]
            PF3["Defense: opp_def_rtg_vs_position"]
            PF4["Context: pace, rest_days, b2b"]
        end

        subgraph QB["QuantileBundle (5 LGBMs)"]
            Q10["LGBMRegressor α=0.10 → q10"]
            Q25["LGBMRegressor α=0.25 → q25"]
            Q50["LGBMRegressor α=0.50 → q50 (median)"]
            Q75["LGBMRegressor α=0.75 → q75"]
            Q90["LGBMRegressor α=0.90 → q90"]
        end

        IOP["implied_over_probability(line)<br/>interpolate P(stat > line) from CDF<br/>via scipy.interpolate"]

        OUT_P["quantiles dict + P(over line)"]
    end

    F1 & F2 & F3 & F4 & F5 & F6 --> X1 & L1
    X1 --> X2 --> X3
    L1 --> L2 --> L3
    X3 & L3 --> ENS --> IC --> CLIP --> OUT_W

    PF1 & PF2 & PF3 & PF4 --> Q10 & Q25 & Q50 & Q75 & Q90
    Q10 & Q25 & Q50 & Q75 & Q90 --> IOP --> OUT_P

    style WinnerModel fill:#e8f4fd
    style PropsModel fill:#e8fde8
```

---

## 7. Database Schema

```mermaid
erDiagram
    sports {
        int id PK
        varchar code
    }

    teams {
        int id PK
        int sport_id FK
        varchar external_id
        varchar name
        varchar abbrev
        varchar conference
        varchar division
        jsonb meta
    }

    players {
        int id PK
        int sport_id FK
        varchar external_id
        varchar full_name
        varchar position
        date birthdate
        jsonb meta
    }

    venues {
        int id PK
        int sport_id FK
        varchar name
        varchar city
        float lat
        float lon
        bool indoor
    }

    games {
        int id PK
        int sport_id FK
        varchar external_id
        int season
        timestamptz scheduled_utc
        varchar status
        int home_team_id FK
        int away_team_id FK
        int venue_id FK
        int home_score
        int away_score
        jsonb meta
    }

    team_game_stats {
        int game_id FK
        int team_id FK
        jsonb stats
        timestamptz recorded_at
    }

    player_game_stats {
        int game_id FK
        int player_id FK
        int team_id FK
        jsonb stats
        timestamptz recorded_at
    }

    matchup_features {
        int game_id FK
        jsonb features
        timestamptz computed_at
    }

    models {
        int id PK
        int sport_id FK
        varchar kind
        varchar target
        varchar version
        varchar mlflow_run_id
        timestamptz trained_at
        bool active
        jsonb metrics
        varchar feature_spec_hash
    }

    predictions {
        int id PK
        int game_id FK
        int model_id FK
        int player_id FK
        varchar target
        float value
        float probability
        jsonb quantiles
        varchar features_hash
        timestamptz created_at
    }

    game_odds {
        int id PK
        int game_id FK
        varchar book
        float home_ml
        float away_ml
        float spread
        float total
        float implied_home_prob
        timestamptz ingested_at
    }

    api_keys {
        int id PK
        varchar name
        varchar key_prefix
        varchar key_hash
        text[] scopes
        timestamptz created_at
        timestamptz expires_at
        timestamptz revoked_at
    }

    drift_events {
        int id PK
        int sport_id FK
        varchar kind
        varchar target
        varchar drift_type
        varchar metric_name
        float metric_value
        float threshold
        int priority
        timestamptz triggered_at
        timestamptz resolved_at
    }

    sports ||--o{ teams : "has"
    sports ||--o{ players : "has"
    sports ||--o{ games : "has"
    sports ||--o{ models : "tracks"
    teams ||--o{ games : "home/away"
    venues ||--o{ games : "hosts"
    games ||--o{ team_game_stats : "has"
    games ||--o{ player_game_stats : "has"
    games ||--o{ matchup_features : "has"
    games ||--o{ predictions : "has"
    games ||--o{ game_odds : "has"
    players ||--o{ player_game_stats : "plays"
    players ||--o{ predictions : "targets"
    models ||--o{ predictions : "generates"
```

---

## 8. Automated Operations (Celery Schedule)

```mermaid
gantt
    title Daily Automated Pipeline (UTC)
    dateFormat HH:mm
    axisFormat %H:%M

    section Data Ingest
    NBA/MLB Box Scores (yesterday)   :done, 09:00, 1h
    Injury Refresh                   :done, 11:00, 30m
    Odds Ingest (open lines)         :done, 14:00, 30m
    MLB Weather                      :done, 18:00, 20m
    Odds Ingest (closing lines)      :done, 22:00, 30m

    section Features & Scoring
    Rebuild Matchup Features         :active, 12:00, 1h
    Score Upcoming Games             :active, 13:00, 30m
    Re-score on Lineup Changes       :crit, 15:00, 7h

    section Model Ops
    Train Challenger Models (nightly) :done, 02:00, 2h
    Drift Detection                  :done, 03:00, 30m
    Evaluate & Promote               :done, 04:00, 30m

    section Live Monitoring
    Live Game Polling (every 2 min)  :crit, 00:00, 24h
```

---

## 9. Deployment Stack

```mermaid
graph TB
    subgraph Internet["Internet"]
        CLIENT["API Consumers<br/>(web, mobile, scripts)"]
    end

    subgraph Docker["Docker Compose Stack"]
        subgraph Proxy["Ingress"]
            CADDY["Caddy<br/>:443 TLS termination<br/>reverse proxy"]
        end

        subgraph App["Application Services"]
            API["api container<br/>Gunicorn + Uvicorn<br/>2 workers<br/>FastAPI app"]
            WORKER["worker container<br/>Celery worker<br/>concurrency=4<br/>task execution"]
            BEAT["beat container<br/>Celery Beat<br/>cron scheduler"]
            NOTIF["notifier container<br/>PostgreSQL LISTEN<br/>Discord · Telegram"]
        end

        subgraph Data["Data Services"]
            PG[("postgres container<br/>PostgreSQL 15<br/>+ TimescaleDB extension<br/>hypertables: team/player stats")]
            REDIS[("redis container<br/>Celery broker<br/>+ result backend<br/>dedup cache")]
        end

        subgraph Observability["Observability"]
            PROM["/metrics<br/>Prometheus endpoint<br/>request count · latency<br/>model hit rate"]
            SLOG["structlog<br/>structured JSON logs<br/>to stderr"]
        end
    end

    subgraph MLRegistry["Model Registry"]
        MLFLOW[("MLflow<br/>./mlruns/<br/>model artifacts<br/>params · metrics")]
    end

    CLIENT -->|HTTPS| CADDY
    CADDY -->|proxy| API
    API <-->|async SQLAlchemy| PG
    API <-->|cache lookups| REDIS
    API -->|read models| MLFLOW

    BEAT -->|enqueue tasks| REDIS
    REDIS -->|dequeue| WORKER
    WORKER <-->|SQLAlchemy| PG
    WORKER -->|log models| MLFLOW
    WORKER -->|NOTIFY| PG

    PG -->|LISTEN| NOTIF
    NOTIF -->|webhook| DISCORD["Discord"]
    NOTIF -->|bot API| TELEGRAM["Telegram"]

    API --> PROM
    API --> SLOG
    WORKER --> SLOG

    style Docker fill:#f0f8ff
    style Data fill:#e8f4fd
    style App fill:#e8fde8
    style Observability fill:#fdf8e8
    style MLRegistry fill:#f3e8fd
```

---

## 10. Anti-Leakage & Model Integrity

```mermaid
flowchart TD
    subgraph Problem["The Leakage Problem"]
        L1["If we use future data to train,<br/>backtests look great but live performance crashes"]
    end

    subgraph Guards["Guards Built Into The System"]
        G1["as_of_utc cutoff<br/>All features computed as of<br/>game_time - 1 hour<br/>No post-game data ever used"]

        G2["load_*_before() functions<br/>load_team_game_stats_before(team_id, as_of_utc)<br/>SQL: WHERE scheduled_utc < as_of<br/>Hard DB-level guarantee"]

        G3["Walk-forward CV<br/>Fold N: train on seasons 1..N-1<br/>validate on season N<br/>No shuffling. Ever."]

        G4["Assertion checks<br/>walk_forward_splits() asserts:<br/>max(train.scheduled_utc) < min(val.scheduled_utc)<br/>Raises if violated"]

        G5["Feature hash<br/>Each prediction stores SHA-256<br/>of the feature dict used<br/>Enables reproducibility audit"]

        G6["Separate calibration set<br/>Isotonic regression fit on<br/>held-out calibration set<br/>(not training data)"]
    end

    subgraph Result["Result"]
        R1["Walk-forward backtest metrics<br/>match live performance<br/>(within expected variance)"]
    end

    Problem --> Guards
    G1 & G2 & G3 & G4 & G5 & G6 --> Result

    style Problem fill:#ffe8e8
    style Guards fill:#e8f4fd
    style Result fill:#e8fde8
```

---

## What We Built Ourselves

| Component | What We Built | Why It Matters |
|-----------|--------------|----------------|
| **Feature Engineering** | 70+ hand-crafted game and player features: Elo replay, rolling efficiency ratings (net_rtg/pace/TS%), schedule load indicators, H2H history, geographic travel factors, market signals | No off-the-shelf sports ML library does this — all domain knowledge encoded by us |
| **Walk-Forward CV** | Custom chronological splitter with strict season boundary guards and leakage assertions | Standard k-fold would leak future scores into training — we prevented this entirely |
| **IsotonicCalibratedEnsemble** | XGBoost + LightGBM weighted ensemble with isotonic regression calibration on a held-out set | Built and tested the calibration pipeline manually; models output well-calibrated probabilities, not just rankings |
| **QuantileBundle** | 5-quantile LightGBM wrapper with CDF interpolation for P(stat > line) | Custom class that wraps 5 separate regressors and interpolates the over/under probability |
| **Promotion Gate** | Multi-metric challenger vs. champion comparator (logloss + ECE + Brier) | Automated nightly model improvement with guardrails against regressions |
| **Drift Detection** | Rolling 30-game log-loss, ECE, and PSI monitoring with per-metric thresholds | Catches when a deployed model's real-world performance degrades before it causes bad predictions |
| **Ingest Pipeline** | Rate-limited scrapers + official APIs for 4 sportsbooks, 2 leagues, weather | Consensus odds from 3 books; Kalshi prediction market as an independent signal |
| **Anti-leakage architecture** | `as_of_utc` cutoff enforced at every layer (SQL, Python, CV) | Ensures live predictions use only information available at prediction time |
