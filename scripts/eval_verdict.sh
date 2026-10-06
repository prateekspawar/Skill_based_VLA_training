#!/usr/bin/env bash
# Evaluate the verdict skill models (shared-backbone swap server, 9-D targets), 50 seeded episodes each:
#   direct_still / direct_push        skills end on simulator truth (comparable with the D2 combined model)
#   verdict_still / verdict_push      skills end on the model's own [done, success]; verdicts scored vs truth
#   find_verdict_still                find(cube), find(plate) with the find model first, then pick / place
# Usage: scripts/eval_verdict.sh [steps]   (waits for the training pipeline to finish)
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}
OUT=outputs/stage_verdict/eval
mkdir -p $OUT
log() { echo "[eval-verdict] $(date +%T) $*"; }
until grep -q "^\[verdict\] .* done" logs/stage_verdict.log; do sleep 60; done
ck() { echo checkpoints/${1}9s_$STEPS/checkpoints/last/pretrained_model; }
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --skills pick=$(ck pick) place=$(ck place) find=$(ck find) \
  --depth fast_foundationstereo > $OUT/server.log 2>&1 &
server=$!
until grep -q listening $OUT/server.log; do
  kill -0 $server 2>/dev/null || { log "server failed"; tail -5 $OUT/server.log; exit 1; }
  sleep 3
done
grep "\[skills\]" $OUT/server.log | sed "s/^/[eval-verdict] /"
run() {
  local name=$1; shift
  log "running $name"
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 50 --step_timeout 8 --seed 0 \
    --target sam2 --target_dim 9 "$@" --video_out $OUT/$name.mp4 > $OUT/$name.log 2>&1
  grep -E "\[eval\] (pick success|pushes|verdict|find\(|skill swap)" $OUT/$name.log | sed "s/^/[eval-verdict] $name: /"
}
run direct_still
run direct_push --push
run verdict_still --verdict
run verdict_push --push --verdict
run find_verdict_still --pipeline find --verdict
kill $server; wait $server 2>/dev/null
log "done"
