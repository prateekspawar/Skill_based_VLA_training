#!/usr/bin/env bash
# Camera-pose sensitivity with the virtual head (so101_lang/robots/virtual_head.py), current models, no retraining:
# 50 seeded stationary episodes each, same scenes as outputs/eval_gate (baseline: camera at the training pose).
#   small  ~2 cm + ~3 deg, virtual head on
#   large  ~4.7 cm + ~6 deg, virtual head on
#   large_novh  same offset, virtual head off (camera pose as plain calibration: the models see the real frame)
# Launch detached: setsid nohup scripts/run_camera_offset.sh > logs/camera_offset.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
O=outputs/camera_offset
log() { echo "[camoff] $(date +%T) $*"; }
run() {  # name, offset, virtual(1/0)
  [[ -f $O/$1/still.log ]] && grep -q "pick success" $O/$1/still.log && { log "$1 already done"; return; }
  log "run $1 (offset $2, virtual head $3)"
  SO101_CAM_OFFSET="$2" SO101_VIRTUAL_HEAD=$3 EPISODES=50 EVAL_OUT=$O/$1 scripts/eval.sh still | sed "s/^/[camoff] $1: /"
}
run small "-0.01,0.015,0.01,1,2,-2" 1
run large "-0.025,0.035,0.02,2,4,-4" 1
run large_novh "-0.025,0.035,0.02,2,4,-4" 0
log done
