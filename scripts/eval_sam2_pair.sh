#!/usr/bin/env bash
# SAM2 evaluation of a model (or pick + place models): 50 seeded stationary + 50 sliding-push episodes.
# Usage: scripts/eval_sam2_pair.sh <out_dir> <checkpoint> [place_checkpoint]
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?out dir}; CKPT=${2:?checkpoint}; PLACE=${3:-}
mkdir -p "$OUT"
extra_server=""; [[ -n "$PLACE" ]] && extra_server="--checkpoint_place $PLACE"
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint "$CKPT" $extra_server --depth none > "$OUT/server.log" 2>&1 &
server=$!
until grep -q listening "$OUT/server.log"; do sleep 3; done
for v in still push; do
  extra=""; [[ $v == push ]] && extra="--push"
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 50 --step_timeout 8 --seed 0 \
    --target sam2 $extra --video_out "$OUT/eval_$v.mp4" > "$OUT/eval_$v.log" 2>&1
  grep -E "\[eval\] (pick success|pushes|target-point)" "$OUT/eval_$v.log" | sed "s/^/[eval] $v: /"
done
kill $server; wait $server 2>/dev/null
