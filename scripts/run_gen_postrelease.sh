#!/usr/bin/env bash
# Pick/place demos that keep recording after the release (2026-10-10): the episode ends once the expert has
# retreated with the gripper open and the cube has stayed on the plate 5 frames (gen_demos --post_release_frames),
# so the success label covers a completed release. Same seeds as rc_static_1000 / rc_moving_1000.
# Waits for the combined evaluation (GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/stage_postrelease; R=datasets/raw
mkdir -p $OUT
log() { echo "[postrelease] $(date +%T) $*"; }
until grep -qE "^\[eval\] [0-9:]+ done" logs/eval_combined.log 2>/dev/null; do sleep 60; done
export OMNI_KIT_ACCEPT_EULA=YES
gen() {  # name, args...
  local name=$1; shift
  [[ -f $R/$name.h5 ]] && grep -q "done" $OUT/gen_$name.log 2>/dev/null && return 0
  log "generating $name"
  ~/isaacsim/bin/python scripts/gen_demos.py "$@" --post_release_frames 5 --out $R/$name.h5 > $OUT/gen_$name.log 2>&1 \
    || { log "gen $name failed"; exit 1; }
  tail -1 $OUT/gen_$name.log | sed "s/^/[postrelease] /"
}
gen rc2_static_1000 --num_envs 16 --episodes 1000 --seed 31
gen rc2_moving_1000 --num_envs 16 --episodes 1000 --seed 32 --move_prob_pick 0.5 --move_prob_place 0.5
log "done"
