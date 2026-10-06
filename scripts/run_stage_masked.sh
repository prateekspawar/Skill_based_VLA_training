#!/usr/bin/env bash
# Retrain pick (cube yaw folded by its 90 deg symmetry + failure motion masked) and place (failure motion masked),
# then evaluate with the unchanged find model. Usage: scripts/run_stage_masked.sh [steps]
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}
OUT=outputs/stage_masked
mkdir -p $OUT
log() { echo "[masked] $(date +%T) $*"; }
# (previously waited for logs/eval_verdict2.log)
log "smoke test: 40 steps on masked place data"
rm -rf $OUT/smoke
DEPTH_CAMERA=0 train/smolvla_depth.sh datasets/lerobot/s512_place9m $OUT/smoke --policy.use_depth_branch=true \
  --policy.relative_action_dims=9 --steps=40 --batch_size=8 --log_freq=10 --save_checkpoint=false > $OUT/smoke.log 2>&1
losses=$(tr '\r' '\n' < $OUT/smoke.log | grep -oE " loss:[^ ]+" | tr '\n' ' ')
log "smoke losses:$losses"
if [[ -z "$losses" ]] || echo "$losses" | grep -qiE "nan|inf"; then log "smoke test failed"; exit 1; fi
DS_SUFFIX=fm scripts/run_stage_skills.sh $STEPS pick || { log "pick training failed"; exit 1; }
DS_SUFFIX=m scripts/run_stage_skills.sh $STEPS place || { log "place training failed"; exit 1; }
log "evaluating"
CK_PICK=pick9fm CK_PLACE=place9m EVAL_OUT=outputs/stage_masked/eval EVAL_EXTRA=--fold_cube_yaw \
  RUNS="direct_still direct_push verdict_still verdict_first_still verdict_first_push find_verdict_first_still" \
  scripts/eval_verdict.sh $STEPS
log "done"
