#!/usr/bin/env bash
# Retrain pick on cube targets folded by the cube's 90 deg symmetry (s512_pick9f), after the running evals.
set -uo pipefail
cd "$(dirname "$0")/.."
until grep -qE "eval-verdict\] .* (done|failed)" logs/eval_verdict2.log; do sleep 60; done
DS_SUFFIX=f scripts/run_stage_skills.sh ${1:-15000} pick
