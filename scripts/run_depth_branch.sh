#!/usr/bin/env bash
# Variant 4 of the depth comparison: SmolVLA + native depth branch (point-map tokens), same steps/eval as
# scripts/run_depth_ablation.sh. Waits for that driver to finish so the GPU is free.
# Usage: scripts/run_depth_branch.sh <steps>
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:?steps}
EVAL_EPISODES=${EVAL_EPISODES:-30}
HELDOUT_EPISODES=${HELDOUT_EPISODES:-10}
v=pointmap
until grep -q "\[ablation\] .* done" outputs/ablation_driver.log 2>/dev/null; do sleep 30; done
until grep -q "\[convert\] wrote" "outputs/convert_abl500_$v.log" 2>/dev/null; do sleep 30; done
echo "[branch] $(date +%T) training smolvla_depth for $STEPS steps"
rm -rf "checkpoints/abl500_$v"
train/smolvla_depth.sh "datasets/lerobot/abl500_$v" "checkpoints/abl500_$v" --steps="$STEPS" --save_freq="$STEPS" \
  > "outputs/ablation/train_$v.log" 2>&1
ckpt="checkpoints/abl500_$v/checkpoints/last/pretrained_model"
[[ -d "$ckpt" ]] || { echo "[branch] training failed"; exit 1; }
echo "[branch] $(date +%T) evaluating"
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint "$ckpt" --depth fast_foundationstereo \
  > "outputs/ablation/server_$v.log" 2>&1 &
server=$!
until grep -q "listening" "outputs/ablation/server_$v.log"; do sleep 3; done
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes "$EVAL_EPISODES" --step_timeout 8 \
  --video_out "outputs/ablation/eval_$v.mp4" > "outputs/ablation/eval_$v.log" 2>&1
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes "$HELDOUT_EPISODES" --step_timeout 8 \
  --heldout_pairs --video_out "outputs/ablation/eval_${v}_heldout.mp4" > "outputs/ablation/eval_${v}_heldout.log" 2>&1
kill $server; wait $server 2>/dev/null
grep -h "\[eval\] pick success" "outputs/ablation/eval_$v.log" "outputs/ablation/eval_${v}_heldout.log" | sed "s/^/[branch] $v: /"
echo "[branch] $(date +%T) done"
