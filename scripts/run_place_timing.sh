#!/usr/bin/env bash
# Place "never released" diagnosis (2026-10-10): fused find, stationary, 150 seeded episodes (same scenes as
# outputs/stage_rc_eval2/new_fused). 1) skill time limit 8 -> 16 s; 2) plus opening the gripper after a place
# success verdict. Outputs in outputs/place_timing/.
set -uo pipefail
cd "$(dirname "$0")/.."
O=outputs/place_timing
EVAL_EXTRA="--step_timeout 16" EVAL_OUT=$O/t16_hold scripts/eval.sh still
EVAL_EXTRA="--step_timeout 16 --place_release open" EVAL_OUT=$O/t16_open scripts/eval.sh still
echo "[place-timing] done"
