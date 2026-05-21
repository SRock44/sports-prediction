#!/usr/bin/env bash
# Bootstrap: backfill 5 seasons for both sports, then train initial models.
# Run this once on a fresh install after `docker compose up -d`.
#
# Usage:
#   ./scripts/bootstrap_backfill.sh
#   ./scripts/bootstrap_backfill.sh --dry-run
set -euo pipefail

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
  echo "[dry-run mode — no data will be written]"
fi

PYTHON="python -m src.cli"
SEASONS=5

echo "=== Step 1: NBA backfill ($SEASONS seasons) ==="
$PYTHON backfill --sport nba --seasons $SEASONS $DRY_RUN

echo ""
echo "=== Step 2: MLB backfill ($SEASONS seasons) ==="
$PYTHON backfill --sport mlb --seasons $SEASONS $DRY_RUN

if [[ -n "$DRY_RUN" ]]; then
  echo ""
  echo "Dry run complete. Exiting."
  exit 0
fi

echo ""
echo "=== Step 3: Rebuild features ==="
$PYTHON backfill --sport nba --seasons $SEASONS  # re-runs with features
# Feature rebuild is handled by Celery beat after ingest; trigger manually here
# via the score command which calls feature rebuild internally on missing rows.

echo ""
echo "=== Step 4: Train NBA winner model ==="
$PYTHON train --sport nba --kind winner --trials 50 --promote

echo ""
echo "=== Step 5: Train MLB winner model ==="
$PYTHON train --sport mlb --kind winner --trials 50 --promote

echo ""
echo "=== Step 6: Train NBA props models ==="
$PYTHON train --sport nba --kind props

echo ""
echo "=== Step 7: Train MLB props models ==="
$PYTHON train --sport mlb --kind props

echo ""
echo "=== Step 8: Score upcoming games ==="
$PYTHON score --sport nba --hours 48
$PYTHON score --sport mlb --hours 48

echo ""
echo "=== Bootstrap complete ==="
echo "Start all services: docker compose up -d"
echo "Create your first API key: python -m src.cli keys create --name my-key"
