#!/usr/bin/env bash
# 512px + DART no-depth baseline: wait for generation -> convert -> train (relative actions) -> open-loop -> eval.
# Usage: scripts/run_s512_baseline.sh <raw.h5> <steps>
set -uo pipefail
cd "$(dirname "$0")/.."
RAW=${1:?raw h5}
STEPS=${2:?steps}
name=s512_rel_none_${STEPS}
data=datasets/lerobot/s512_dart_none
ckpt="checkpoints/$name/checkpoints/last/pretrained_model"
mkdir -p outputs/s512
until grep -q "\[gen\] done" outputs/gen512_1000.log 2>/dev/null; do
  grep -q Traceback outputs/gen512_1000.log 2>/dev/null && { echo "[s512] generation failed"; exit 1; }
  sleep 30
done
echo "[s512] $(date +%T) converting"
~/lerobot-env/bin/python -m so101_lang.data.hdf5_to_lerobot --inputs "$RAW" --root "$data" --depth none --overwrite \
  > outputs/s512/convert.log 2>&1
grep -o "\[convert\] wrote.*" outputs/s512/convert.log | sed "s/^/[s512] /"
echo "[s512] $(date +%T) training $name"
train/smolvla_depth.sh "$data" "checkpoints/$name" --policy.use_depth_branch=false --policy.relative_action_dims=9 \
  --steps="$STEPS" --save_freq=$((STEPS / 2)) > "outputs/s512/train_$name.log" 2>&1
[[ -d "$ckpt" ]] || { echo "[s512] training failed"; exit 1; }
echo "[s512] $(date +%T) open-loop check"
~/lerobot-env/bin/python scripts/openloop_check.py --checkpoint "$ckpt" --dataset "$data" 2>&1 | grep -E "^(pick|place)" | sed "s/^/[s512] /"
echo "[s512] $(date +%T) closed-loop eval"
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint "$ckpt" --depth none > "outputs/s512/server.log" 2>&1 &
server=$!
until grep -q "listening" outputs/s512/server.log; do sleep 3; done
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 30 --step_timeout 8 \
  --video_out "outputs/s512/eval.mp4" > "outputs/s512/eval.log" 2>&1
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 10 --step_timeout 8 --heldout_pairs \
  --video_out "outputs/s512/eval_heldout.mp4" > "outputs/s512/eval_heldout.log" 2>&1
kill $server; wait $server 2>/dev/null
grep -h "\[eval\] pick success" outputs/s512/eval.log outputs/s512/eval_heldout.log | sed "s/^/[s512] /"
echo "[s512] $(date +%T) done"
