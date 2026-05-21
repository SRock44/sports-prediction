"""Canonical feature list for the NBA game-winner model.

This list is the source of truth for:
- Feature matrix column order during training
- Inference-time feature extraction
- feature_spec_hash computation (schema drift detection)

Changes here must be accompanied by a new model training run.
"""

NBA_WINNER_FEATURES = [
    # ── Home team ─────────────────────────────────────────────────────────────
    "home_elo",
    "home_is_home",
    "home_net_rtg_last5",
    "home_net_rtg_last10",
    "home_net_rtg_last20",
    "home_off_rtg_last5",
    "home_off_rtg_last10",
    "home_off_rtg_last20",
    "home_def_rtg_last5",
    "home_def_rtg_last10",
    "home_def_rtg_last20",
    "home_pace_last5",
    "home_pace_last10",
    "home_pace_last20",
    "home_rest_days",
    "home_b2b",
    "home_three_in_four",
    "home_four_in_six",
    "home_home_win_pct",
    "home_away_win_pct",
    "home_overall_win_pct_last10",
    "home_travel_km",
    "home_starter_availability",
    # ── Away team ─────────────────────────────────────────────────────────────
    "away_elo",
    "away_is_home",
    "away_net_rtg_last5",
    "away_net_rtg_last10",
    "away_net_rtg_last20",
    "away_off_rtg_last5",
    "away_off_rtg_last10",
    "away_off_rtg_last20",
    "away_def_rtg_last5",
    "away_def_rtg_last10",
    "away_def_rtg_last20",
    "away_pace_last5",
    "away_pace_last10",
    "away_pace_last20",
    "away_rest_days",
    "away_b2b",
    "away_three_in_four",
    "away_four_in_six",
    "away_home_win_pct",
    "away_away_win_pct",
    "away_overall_win_pct_last10",
    "away_travel_km",
    "away_starter_availability",
    # ── Matchup / cross features ──────────────────────────────────────────────
    "elo_diff",
    "elo_diff_with_hca",
    "elo_home_win_prob",
    "net_rtg_diff_last5",
    "net_rtg_diff_last10",
    "rest_diff",
    "h2h_home_wins",
    "h2h_total",
    "h2h_home_win_pct",
    "away_travel_km",
]
