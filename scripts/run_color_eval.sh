#!/usr/bin/env bash
# Learned colour model vs the HSV rule (2026-10-10): fused find, stationary, 150 seeded episodes, same scenes and
# settings as outputs/stage_rc_eval2/new_fused (8 s skills). Waits for run_place_timing.sh (GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -q "place-timing\] done" logs/place_timing.log 2>/dev/null; do sleep 60; done
SO101_COLOR_MODEL=learned EVAL_OUT=outputs/color_eval/learned scripts/eval.sh still
echo "[color-eval] done"
