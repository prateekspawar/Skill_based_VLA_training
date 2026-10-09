#!/usr/bin/env bash
# Random plate colours, after the time-out fix (find pipeline: simulator episode limit 18 s -> 138 s; a mid-episode
# time-out reset re-randomised plate colours and object layout). Outputs in outputs/stage_rc_eval2.
#   0. time-out check: new models, retry views, OLD 18 s limit, 50 episodes (how often were earlier runs affected?)
#   1. new models (pick9rc / place9rc / find9rc), retry views: stationary + push
#   2. new models, fused multi-view find: stationary + push
#   3. old models (pick9fm / place9m / find9s), retry views: stationary (baseline on the same scenes)
set -uo pipefail
cd "$(dirname "$0")/.."
export SO101_RANDOM_PLATE_COLORS=1
O=outputs/stage_rc_eval2
EPISODES=50 CK_PICK=pick9rc CK_PLACE=place9rc CK_FIND=find9rc EVAL_EXTRA="--freeze_radius 0.03 --find_views retry --episode_length_s 18" \
  EVAL_OUT=$O/timeout_check RUNS="find_verdict_first_still" scripts/eval_verdict.sh 15000
export EPISODES=150
CK_PICK=pick9rc CK_PLACE=place9rc CK_FIND=find9rc EVAL_EXTRA="--freeze_radius 0.03 --find_views retry" \
  EVAL_OUT=$O/new_retry RUNS="find_verdict_first_still find_verdict_first_push" scripts/eval_verdict.sh 15000
CK_PICK=pick9rc CK_PLACE=place9rc CK_FIND=find9rc EVAL_EXTRA="--freeze_radius 0.03 --find_views fused" \
  EVAL_OUT=$O/new_fused RUNS="find_verdict_first_still find_verdict_first_push" scripts/eval_verdict.sh 15000
CK_PICK=pick9fm CK_PLACE=place9m CK_FIND=find9s EVAL_EXTRA="--freeze_radius 0.03 --find_views retry" \
  EVAL_OUT=$O/old_retry RUNS="find_verdict_first_still" scripts/eval_verdict.sh 15000
echo "[eval-rc2] done"
