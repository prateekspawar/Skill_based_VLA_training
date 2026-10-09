#!/usr/bin/env bash
# Resume of run_eval_randcolor.sh after the 9 Oct reboot (new_retry still already done): evaluate on random plate colours (150 seeded episodes each).
#   new models (pick9rc / place9rc / find9rc), retry views: stationary + push
#   new models, fused multi-view find: stationary + push
#   old models (pick9fm / place9m / find9s), retry views: stationary (baseline on the same scenes)
set -uo pipefail
cd "$(dirname "$0")/.."
export SO101_RANDOM_PLATE_COLORS=1 EPISODES=150
CK_PICK=pick9rc CK_PLACE=place9rc CK_FIND=find9rc EVAL_EXTRA="--freeze_radius 0.03 --find_views retry" \
  EVAL_OUT=outputs/stage_rc_eval/new_retry RUNS="find_verdict_first_push" scripts/eval_verdict.sh 15000
CK_PICK=pick9rc CK_PLACE=place9rc CK_FIND=find9rc EVAL_EXTRA="--freeze_radius 0.03 --find_views fused" \
  EVAL_OUT=outputs/stage_rc_eval/new_fused RUNS="find_verdict_first_still find_verdict_first_push" scripts/eval_verdict.sh 15000
CK_PICK=pick9fm CK_PLACE=place9m CK_FIND=find9s EVAL_EXTRA="--freeze_radius 0.03 --find_views retry" \
  EVAL_OUT=outputs/stage_rc_eval/old_retry RUNS="find_verdict_first_still" scripts/eval_verdict.sh 15000
echo "[eval-rc] done"
