#!/usr/bin/env bash
# Evaluate the verdict skill models (shared-backbone swap server, 9-D targets), 50 seeded episodes each:
#   direct_still / direct_push        skills end on simulator truth (comparable with the D2 combined model)
#   verdict_still / verdict_push      skills end on the model's own [done, success]; verdicts scored vs truth
#   find_verdict_still                find(cube), find(plate) with the find model first, then pick / place
# Usage: [RUNS="direct_still ..."] scripts/eval_verdict.sh [steps]   (waits for the training pipeline to finish)
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}
OUT=${EVAL_OUT:-outputs/stage_verdict/eval}
mkdir -p $OUT
log() { echo "[eval-verdict] $(date +%T) $*"; }
until grep -q "^\[verdict\] .* done" logs/stage_verdict.log; do sleep 60; done
[[ -n "${WAIT_FOR:-}" ]] && until grep -qE "${WAIT_FOR#*:}" "${WAIT_FOR%%:*}"; do sleep 60; done
# Checkpoint names per skill (override e.g. CK_PICK=pick9fm): checkpoints/<name>_<steps>.
ck() { local v="CK_${1^^}"; echo checkpoints/${!v:-${1}9s}_$STEPS/checkpoints/last/pretrained_model; }
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
    --target sam2 --target_dim 9 ${EVAL_EXTRA:-} "$@" --video_out $OUT/$name.mp4 > $OUT/$name.log 2>&1
  grep -E "\[eval\] (pick success|pushes|verdict|find\(|skill swap)" $OUT/$name.log | sed "s/^/[eval-verdict] $name: /"
}
for r in ${RUNS:-direct_still direct_push verdict_still verdict_push find_verdict_still}; do
  case $r in
    direct_still) run direct_still ;;
    direct_push) run direct_push --push ;;
    direct_still_fold) run direct_still_fold --fold_cube_yaw ;;
    verdict_still) run verdict_still --verdict ;;
    verdict_push) run verdict_push --push --verdict ;;
    verdict_first_still) run verdict_first_still --verdict --verdict_from first ;;
    verdict_first_push) run verdict_first_push --push --verdict --verdict_from first ;;
    find_verdict_still) run find_verdict_still --pipeline find --verdict ;;
    find_verdict_first_still) run find_verdict_first_still --pipeline find --verdict --verdict_from first ;;
    find_verdict_first_push) run find_verdict_first_push --pipeline find --push --verdict --verdict_from first ;;
  esac
done
kill $server; wait $server 2>/dev/null
log "done"
