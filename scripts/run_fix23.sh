#!/usr/bin/env bash
# Tracking follows pushes faster (SAM2 deadband 10 mm, pick target frozen within 3 cm instead of 5 cm) and the
# gripper closes on the way home after find() (pick start state as in training). 150 seeded episodes per run.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -qE "eval-verdict\] .* (done|failed)" logs/cmp.log; do sleep 60; done
SAM2_DEADBAND=0.01 EVAL_EXTRA="--freeze_radius 0.03" CK_PICK=pick9fm CK_PLACE=place9m EVAL_OUT=outputs/stage_fix23/eval \
  EPISODES=150 RUNS="find_verdict_first_push find_verdict_first_still direct_verdict_first_push" scripts/eval_verdict.sh 15000
