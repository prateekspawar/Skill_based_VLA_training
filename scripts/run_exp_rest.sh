#!/usr/bin/env bash
# Sequential queue after the 2026-10-16 reboot: experiment 4 evaluations (training done), experiment 2 follow-ups,
# experiment 3. Each driver is resumable; each waits for the previous "done" line in its log.
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/run_exp4_task_experts.sh 7500 > logs/exp4.log 2>&1
scripts/run_exp2b.sh 5000 > logs/exp2b.log 2>&1
scripts/run_exp3_compose.sh > logs/exp3.log 2>&1
echo "[chain] $(date +%T) done"
