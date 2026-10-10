#!/usr/bin/env bash
# Push failures (2026-10-11): pushed episodes fail with a stale target at close (54/109) or with a correct target
# but the gripper closing ~35 mm away (29/109). Same 150 seeded push scenes as outputs/eval/push.log, with push
# timing in the log (push end, tracker caught up, close). Waits for the post-release demo generation (GPU).
#   state     default state tracker (timing baseline)
#   fast      state tracker following after 1 trusted measurement > 1 cm away
#   deadband  displacement deadband 1 cm (best push tracker on white/pink plates, 2026-10-07)
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -qE "^\[postrelease\] [0-9:]+ (done|gen .* failed)$" logs/gen_postrelease.log 2>/dev/null; do sleep 60; done
O=outputs/push_study
EVAL_OUT=$O/state scripts/eval.sh push
SAM2_MOVE_CONFIRM=1 SAM2_MOVE_THR=0.01 EVAL_OUT=$O/fast scripts/eval.sh push
SAM2_MODE=deadband SAM2_DEADBAND=0.01 EVAL_OUT=$O/deadband scripts/eval.sh push
echo "[push-study] done"
