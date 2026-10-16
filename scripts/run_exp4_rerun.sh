#!/usr/bin/env bash
# Rerun the experiment-4 evaluations that crashed (eval_policy indentation bug, 2026-10-16 22:00) once the main
# queue (run_exp_rest.sh) is done. run_exp4_task_experts.sh skips evaluations that already finished.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -qE "^\[chain\] [0-9:]+ done" logs/exp_rest.log 2>/dev/null; do sleep 120; done
scripts/run_exp4_task_experts.sh 7500 > logs/exp4_rerun.log 2>&1
echo "[rerun] $(date +%T) done" >> logs/exp4_rerun.log
