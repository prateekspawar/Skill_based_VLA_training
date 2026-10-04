#!/usr/bin/env bash
# Target tracking tests: static perception vs SAM2 tracking (stationary first), then with cube pushes.
# Usage: scripts/run_tracking_test.sh <checkpoint> [episodes]
set -uo pipefail
cd "$(dirname "$0")/.."
CKPT=${1:?checkpoint}
N=${2:-20}
mkdir -p outputs/tracking
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint "$CKPT" --depth none > outputs/tracking/server.log 2>&1 &
server=$!
until grep -q "listening" outputs/tracking/server.log; do sleep 3; done
run() {  # name, extra args
  local name=$1; shift
  echo "[tracking] $(date +%T) $name"
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes "$N" --step_timeout 8 --seed 0 \
    --video_out "outputs/tracking/$name.mp4" "$@" > "outputs/tracking/$name.log" 2>&1
  grep -E "\[eval\] (pick success|pushes|target-point|tracked|tracker)" "outputs/tracking/$name.log" | sed "s/^/[tracking] $name: /"
}
ONLY=${ONLY:-S1 S2}  # subset of variants to run
[[ " $ONLY " == *" S1 "* ]] && run S1_static_still --target perception
[[ " $ONLY " == *" S2 "* ]] && run S2_sam2_still --target sam2
[[ " $ONLY " == *" P1 "* ]] && run P1_static_push --target perception --push
[[ " $ONLY " == *" P2 "* ]] && run P2_sam2_push --target sam2 --push
[[ " $ONLY " == *" P3 "* ]] && run P3_oracle_push --target oracle_tracked --push
kill $server; wait $server 2>/dev/null
echo "[tracking] $(date +%T) done"
