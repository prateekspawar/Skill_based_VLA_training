#!/usr/bin/env bash
# Depth-input comparison: train SmolVLA on the same episodes with no depth / SGBM depth / Fast-FoundationStereo
# depth, then evaluate each closed-loop through the SO-101 adapter + IK.
# Usage: scripts/run_depth_ablation.sh <steps> [variants...]   (default variants: none sgbm fast_foundationstereo)
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:?steps}
shift
VARIANTS=${*:-none sgbm fast_foundationstereo}
EVAL_EPISODES=${EVAL_EPISODES:-30}
HELDOUT_EPISODES=${HELDOUT_EPISODES:-10}
mkdir -p outputs/ablation

wait_for_dataset() {  # converter prints "[convert] wrote" when the dataset is finalized
  until grep -q "\[convert\] wrote" "outputs/convert_abl500_$1.log" 2>/dev/null; do
    if ! pgrep -f "abl500_$1 " >/dev/null && ! grep -q "\[convert\] wrote" "outputs/convert_abl500_$1.log"; then
      echo "[ablation] conversion for $1 died"; return 1
    fi
    sleep 20
  done
}

for v in $VARIANTS; do
  echo "[ablation] $(date +%T) waiting for dataset $v"
  wait_for_dataset "$v" || continue
  depth_flag=1; [[ "$v" == "none" ]] && depth_flag=0
  echo "[ablation] $(date +%T) training $v for $STEPS steps"
  rm -rf "checkpoints/abl500_$v"
  DEPTH=$depth_flag train/smolvla.sh "datasets/lerobot/abl500_$v" "checkpoints/abl500_$v" \
    --steps="$STEPS" --save_freq="$STEPS" > "outputs/ablation/train_$v.log" 2>&1
  grep -E "step:$STEPS " "outputs/ablation/train_$v.log" | tail -1 | sed "s/^/[ablation] $v final: /"
done

for v in $VARIANTS; do
  ckpt="checkpoints/abl500_$v/checkpoints/last/pretrained_model"
  [[ -d "$ckpt" ]] || { echo "[ablation] no checkpoint for $v"; continue; }
  echo "[ablation] $(date +%T) evaluating $v"
  ~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint "$ckpt" --depth "$v" \
    > "outputs/ablation/server_$v.log" 2>&1 &
  server=$!
  until grep -q "listening" "outputs/ablation/server_$v.log"; do sleep 3; done
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes "$EVAL_EPISODES" \
    --step_timeout 8 --video_out "outputs/ablation/eval_$v.mp4" > "outputs/ablation/eval_$v.log" 2>&1
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes "$HELDOUT_EPISODES" \
    --step_timeout 8 --heldout_pairs --video_out "outputs/ablation/eval_${v}_heldout.mp4" > "outputs/ablation/eval_${v}_heldout.log" 2>&1
  kill $server; wait $server 2>/dev/null
  grep -h "\[eval\] pick success" "outputs/ablation/eval_$v.log" "outputs/ablation/eval_${v}_heldout.log" | sed "s/^/[ablation] $v: /"
done
echo "[ablation] $(date +%T) done"
