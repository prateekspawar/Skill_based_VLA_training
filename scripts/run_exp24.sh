#!/usr/bin/env bash
# Queue: experiment 2 (new stack skill) then experiment 4 (task vs skill experts).
set -uo pipefail
cd "$(dirname "$0")/.."
scripts/run_exp2_stack.sh 5000 || echo "[chain] exp2 failed"
scripts/run_exp4_task_experts.sh 7500 || echo "[chain] exp4 failed"
echo "[chain] $(date +%T) done"
