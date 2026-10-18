#!/usr/bin/env bash
# Random plate colours (docs/NOTES.md section 7, 2026-10-09): new vs old skill models, retry vs fused find,
# 150 seeded episodes each, outputs in outputs/stage_rc_eval2.
# (The original run also had a 50-episode time-out check with --episode_length_s 18.)
set -uo pipefail
cd "$(dirname "$0")/.."
O=outputs/stage_rc_eval2
EVAL_EXTRA="--find_views retry" EVAL_OUT=$O/new_retry scripts/eval.sh still push
EVAL_EXTRA="--find_views fused" EVAL_OUT=$O/new_fused scripts/eval.sh still push
CK_PICK=pick9fm_15000 CK_PLACE=place9m_15000 CK_FIND=find9s_15000 EVAL_EXTRA="--find_views retry" \
  EVAL_OUT=$O/old_retry scripts/eval.sh still
echo "[eval-rc2] done"
