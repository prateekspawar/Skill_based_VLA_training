#!/usr/bin/env bash
# Red/blue plates, 25 stationary + 25 push: find with preset-view retries, then with all views.
set -uo pipefail
cd "$(dirname "$0")/.."
export SO101_PLATE_COLORS=red,blue CK_PICK=pick9fm CK_PLACE=place9m EPISODES=25
EVAL_EXTRA="--freeze_radius 0.03 --find_views retry" EVAL_OUT=outputs/stage_views_retry/eval \
  RUNS="find_verdict_first_still find_verdict_first_push" scripts/eval_verdict.sh 15000
EVAL_EXTRA="--freeze_radius 0.03 --find_views all" EVAL_OUT=outputs/stage_views_all/eval \
  RUNS="find_verdict_first_still find_verdict_first_push" scripts/eval_verdict.sh 15000
echo "[views] done"
