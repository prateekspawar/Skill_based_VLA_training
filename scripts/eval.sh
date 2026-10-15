#!/usr/bin/env bash
# Evaluate the skill pipeline: start the policy server, run seeded evaluations, stop the server.
#   scripts/eval.sh [RUN ...]       RUN: still (default) | push        (default runs: still push)
# Environment overrides:
#   CK_PICK / CK_PLACE / CK_FIND   checkpoint names under checkpoints/ (default pick9rc2_15000, place9rc2_15000, find9rc_15000)
#   CK_STACK                       optional stack skill model (evaluate stacking with SO101_TASK_MODE=stack)
#   EPISODES (150)  EVAL_OUT (outputs/eval)  EVAL_EXTRA (extra eval_policy.py flags, e.g. "--find_views retry")
# Outputs: $EVAL_OUT/<run>.log and .mp4, server.log. Launch detached for long runs:
#   setsid nohup scripts/eval.sh > logs/eval.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${EVAL_OUT:-outputs/eval}
mkdir -p $OUT
log() { echo "[eval] $(date +%T) $*"; }
ck() { echo checkpoints/$1/checkpoints/last/pretrained_model; }
rm -f $OUT/server.log  # a stale "listening" line must not start the evaluations early
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --skills pick=$(ck ${CK_PICK:-pick9rc2_15000}) \
  place=$(ck ${CK_PLACE:-place9rc2_15000}) find=$(ck ${CK_FIND:-find9rc_15000}) ${CK_STACK:+stack=$(ck $CK_STACK)} \
  > $OUT/server.log 2>&1 &
server=$!
until grep -q listening $OUT/server.log; do
  kill -0 $server 2>/dev/null || { log "server failed"; tail -5 $OUT/server.log; exit 1; }
  sleep 3
done
for r in ${@:-still push}; do
  case $r in still) flags="" ;; push) flags="--push" ;; *) log "unknown run $r"; continue ;; esac
  log "running $r"
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes ${EPISODES:-150} --seed 0 \
    $flags ${EVAL_EXTRA:-} --video_out $OUT/$r.mp4 > $OUT/$r.log 2>&1
  grep -E "\[eval\] (pick success|pushes|verdict|find\(|find:|simulator time-out|place failures|pick missed|pick retries|release gate)" $OUT/$r.log \
    | sed "s/^/[eval] $r: /"
done
kill $server; wait $server 2>/dev/null
log "done"
