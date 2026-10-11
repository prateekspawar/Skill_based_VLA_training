#!/usr/bin/env bash
# Late-push recovery demos (2026-10-11): the cube slides 3-6 cm during the pick descent (jaws >= 4 cm above it);
# the expert re-aims (or lifts and re-approaches). Post-release recording on. For the retrain.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/stage_postrelease; mkdir -p $OUT
echo "[latepush] $(date +%T) generating rc2_latepush_500"
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/gen_demos.py --num_envs 16 --episodes 500 --seed 36 \
  --move_prob_pick 1.0 --late_pick_move 1.0 --move_prob_place 0.3 --post_release_frames 5 \
  --out datasets/raw/rc2_latepush_500.h5 > $OUT/gen_rc2_latepush_500.log 2>&1 && echo "[latepush] $(date +%T) done" \
  || echo "[latepush] $(date +%T) failed"
