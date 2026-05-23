# Model Training History

Sport: NBA & MLB | Model type: XGBoost + LightGBM ensemble, isotonic calibration
Promotion gate: challenger must beat champion by ≥ 0.005 log-loss

---

## Model id=1 — First MLB Run
**Date:** 2026-05-22 16:39 UTC
**Sport:** MLB
**Status:** Not active (superseded)

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6953 |
| Accuracy | 52.97% |
| Brier    | 0.2510 |
| ECE      | 0.0144 |
| Samples  | 370    |

**Notes:**
- First ever MLB model trained
- Barely above coin flip accuracy (52.97%)
- Small holdout of 370 samples

---

## Model id=2 — First NBA Run
**Date:** 2026-05-22 16:49 UTC
**Sport:** NBA
**Status:** Not active (superseded)

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6491 |
| Accuracy | 60.71% |
| Brier    | 0.2268 |
| ECE      | 0.1582 |
| Samples  | 56     |

**Notes:**
- First NBA model — decent log-loss but only 56 holdout samples (unreliable)
- High ECE (0.158) means probabilities were poorly calibrated
- Superseded same day by id=3

---

## Model id=3 — NBA Champion (Current)
**Date:** 2026-05-22 21:47 UTC
**Sport:** NBA
**Status:** Active champion ✓

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6677 |
| Accuracy | 66.23% |
| Brier    | 0.2171 |
| ECE      | 0.0389 |

**Notes:**
- Current NBA champion — the target to beat
- Much better calibration than id=2 (ECE 0.039 vs 0.158)
- 50 Optuna trials

---

## Model id=4 — MLB Champion (Current)
**Date:** 2026-05-22 22:22 UTC
**Sport:** MLB
**Status:** Active champion ✓

| Metric   | Value  |
|----------|--------|
| Log-loss | 0.6891 |
| Accuracy | 53.42% |
| Brier    | 0.2480 |
| ECE      | 0.0191 |
| Samples  | 745    |

**Notes:**
- Current MLB champion — the target to beat
- Well-calibrated (ECE 0.019) but low accuracy reflects baseball's inherent randomness
- 50 Optuna trials

---

## Run — Lambda Increase Experiment
**Date:** ~2026-05-22
**Trials:** 50
**Status:** Not promoted (worse)

| Sport | Log-loss | Accuracy |
|-------|----------|----------|
| NBA   | worse    | ~65.5%   |
| MLB   | worse    | ~53.3%   |

**Changes tried:**
- Lambda increased: NBA 0.30 → 0.55, MLB 0.20 → 0.30 (emphasize recent games more)

**Result:** Accuracy regressed. Overconfidence on recent form hurt calibration. Reverted lambda back to 0.30/0.20.

---

## Run 3 — LightGBM CUDA Crash (Failed)
**Date:** ~2026-05-22
**Trials:** 300 (intended)
**Status:** Crashed — all LGB trials failed

**What happened:**
- Set `_LGB_DEVICE = "cuda"` but LightGBM in the Docker image was not compiled with `-DUSE_CUDA=1`
- Every LGB trial threw `LightGBMError: CUDA Tree Learner was not enabled in this build`
- XGB portion completed for MLB (best_n_trees=219) but full run crashed
- No model promoted

**Fix:** Hardcoded `_LGB_DEVICE = "cpu"`. XGBoost still uses GPU via `_XGB_DEVICE = "cuda"`.

---

## Run 4 — Full Feature Rebuild + 300 Trials (Current)
**Date:** 2026-05-23
**Trials:** 300
**Status:** In progress ⏳

| Sport | XGB Best (so far) | Champion   | Delta   |
|-------|-------------------|------------|---------|
| NBA   | **0.63421** (t44) | 0.6677     | -0.033 ✓ |
| MLB   | **0.68175** (t114)| 0.6891     | -0.007 ✓ |

**Changes from champion:**

### Feature additions
| Feature | Sport | Notes |
|---------|-------|-------|
| `margin_last5/10/20` | NBA | Avg point differential — stronger signal than win/loss alone |
| `margin_diff_last5/10/20` | NBA | Home minus away margin cross-feature |
| `home/away_sos_last10` | NBA | Avg Elo of last 10 opponents (strength of schedule) |
| `sos_diff` | NBA | SOS differential |
| `venue_hca` | NBA | Historical home win% at specific arena |
| `roster_star_pts/ts_pct` | NBA | Top-2 player scoring & efficiency |
| `roster_depth_score` | NBA | Avg TS% of players 3-8 |
| `star_usage_conc` | NBA | Top-2 usage share (ball dominance) |
| `travel_km` | NBA | Away team travel distance |
| `road_game_streak` | NBA | Consecutive road games |
| `tz_hours_change` | NBA | Timezone change since last game |
| `starter_availability` | NBA | Fraction of top-8 not out/doubtful |
| SP fallback from box scores | MLB | Pulls actual starter when lineups table empty |

### Training changes
| Parameter | Before | After |
|-----------|--------|-------|
| Optuna trials | 50 | 300 |
| Calibration holdout | 15% | 20% |
| XGB n_estimators search | 100-2000 | 100-5000 |
| XGB max_depth search | 3-8 | 3-10 |
| XGB learning_rate search | 0.005-0.3 | 0.001-0.2 |
| LightGBM device | cpu (broken→cuda→fixed cpu) | cpu |

### Data quality fixes
- **Regular season filter:** NBA `game_type != 'PR'`, MLB `game_type = 'R'` only — playoffs distort team behavior
- **common.py updated:** server had 182-line version missing `load_game_odds`, `load_team_top_player_stats`, `load_game_weather`
- **NBA feature full rebuild:** 6,548 games rebuilt with `--force` to pick up all new features (0 errors)

---

## Targets
| Sport | Must beat | Accuracy goal |
|-------|-----------|---------------|
| NBA   | 0.6627 log-loss | High 60s% |
| MLB   | 0.6841 log-loss | Mid 50s%  |
