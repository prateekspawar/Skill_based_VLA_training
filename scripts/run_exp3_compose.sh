#!/usr/bin/env bash
# Experiment 3 (2026-10-16): unseen multi-step tasks composed from the skills (none was ever a training episode).
#   pairs        put cube X on plate A and cube Y on plate B          (pick, place, pick, place)
#   plate_stack  put X on plate A, then stack Y on top of it          (pick, place, pick, stack)
#   tower        stack Y on X, then Z on top of Y (3-cube tower)      (pick, stack, pick, stack)
# Execution: 50 seeded episodes each with the oracle plan (skills: pick9rc2 / place9rc2 / find9rc / stack9_300).
# Planning: the planner LLM (qwen3-vl:8b, Ollama, CPU) plans the same instructions offline (scripts/plan_offline.py);
# end-to-end success = oracle-plan execution success AND plan correct (identical seeded execution).
# The stacking release tolerance is used if experiment 2 found it better than the plate one.
set -uo pipefail
cd "$(dirname "$0")/.."
O=outputs/exp3
mkdir -p $O
log() { echo "[exp3] $(date +%T) $*"; }
until grep -qE "^\[exp2b\] [0-9:]+ done" logs/exp2b.log 2>/dev/null; do sleep 120; done
full() { grep -oE "full pick\+place [0-9]+" $1 2>/dev/null | grep -oE "[0-9]+$" || echo 0; }
EXTRA=""
if (( $(full outputs/exp2/stack_skill_300_gate/still.log) > $(full outputs/exp2/stack_skill_300/still.log) )); then
  EXTRA="--stack_release_gate 0.012 0.05"
fi
log "stack release gate: ${EXTRA:-plate gate}"
for c in pairs plate_stack tower; do
  [[ -f $O/$c/still.log ]] && grep -q "compose $c" $O/$c/still.log && continue
  log "running $c (oracle plans)"
  CK_STACK=stack9_300_5000 EPISODES=50 EVAL_OUT=$O/$c EVAL_EXTRA="--compose $c --planner oracle $EXTRA" scripts/eval.sh still \
    > /dev/null
  grep -E "\[eval\] compose" $O/$c/still.log | sed "s/^/[exp3] /"
done
curl -s localhost:11434/api/tags > /dev/null || { log "ollama not running: start it (CPU) and run scripts/plan_offline.py"; exit 1; }
for c in pairs plate_stack tower; do
  [[ -f $O/plans_$c.json ]] && continue
  log "planning $c offline"
  ~/lerobot-env/bin/python scripts/plan_offline.py $O/$c/still.log --out $O/plans_$c.json > $O/plan_$c.log 2>&1
  tail -1 $O/plan_$c.log | sed "s/^/[exp3] /"
done
log "done"
