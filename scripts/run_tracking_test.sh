#!/usr/bin/env bash
# Moving-target test: static perception vs Kalman-tracked perception vs oracle tracking, with cube pushes.
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
run A_static_push --target perception --push
run B_tracked_push --target tracked --push
run C_oracle_push --target oracle_tracked --push
run D_tracked_nopush --target tracked
kill $server; wait $server 2>/dev/null
echo "[tracking] $(date +%T) done"
