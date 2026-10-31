#!/usr/bin/env bash
# LLM planner + re-planning on the composed tasks of experiment 4 (same 50 seeded scenes per task as outputs/exp3,
# where the oracle sequences ran once without re-planning: pairs 43/50, plate_stack 21/50, tower 8/50).
#   llm_perception  planner LLM (CPU) + re-planning, goals checked by perception (the real-robot setup)
#   oracle_truth    oracle sequences + re-planning, goals checked by simulator truth (re-planning upper bound)
# Skills: pick9rc2 / place9rc2 / find9rc / stack9_place300 (as in experiment 4).
set -uo pipefail
cd "$(dirname "$0")/.."
O=outputs/replan
mkdir -p $O
log() { echo "[replan] $(date +%T) $*"; }
for v in llm_perception oracle_truth; do
  case $v in
    llm_perception) flags="--planner llm --planner_cpu --replan --check perception" ;;
    oracle_truth) flags="--planner oracle --replan --check truth" ;;
  esac
  for c in pairs plate_stack tower; do
    out=$O/${c}_$v
    [[ -f $out/still.log ]] && grep -q "\[eval\] compose $c" $out/still.log && continue
    log "running $c ($v)"
    CK_STACK=stack9_place300_5000 EPISODES=50 EVAL_OUT=$out EVAL_EXTRA="--compose $c $flags" scripts/eval.sh still > /dev/null
    grep -E "\[eval\] (compose|replan|perception goal)" $out/still.log | sed "s/^/[replan] ${c}_$v: /"
  done
done
log "done"
