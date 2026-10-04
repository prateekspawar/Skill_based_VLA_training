#!/usr/bin/env bash
# No-depth baseline with state-relative actions: train, open-loop check, closed-loop eval.
# Usage: scripts/run_rel_baseline.sh <steps>
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:?steps}
name=rel_none_${STEPS}
ckpt="checkpoints/$name/checkpoints/last/pretrained_model"
mkdir -p outputs/rel
echo "[rel] $(date +%T) training $name"
rm -rf "checkpoints/$name"
train/smolvla_depth.sh datasets/lerobot/abl500_none "checkpoints/$name" --policy.use_depth_branch=false \
  --policy.relative_action_dims=9 --steps="$STEPS" --save_freq=$((STEPS / 2)) > "outputs/rel/train_$name.log" 2>&1
[[ -d "$ckpt" ]] || { echo "[rel] training failed"; exit 1; }
echo "[rel] $(date +%T) open-loop check"
~/lerobot-env/bin/python scripts/openloop_check.py --checkpoint "$ckpt" --dataset datasets/lerobot/abl500_none 2>&1 \
  | grep -E "^(pick|place)" | sed "s/^/[rel] /"
echo "[rel] $(date +%T) closed-loop eval"
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint "$ckpt" --depth none > "outputs/rel/server_$name.log" 2>&1 &
server=$!
until grep -q "listening" "outputs/rel/server_$name.log"; do sleep 3; done
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 30 --step_timeout 8 \
  --video_out "outputs/rel/eval_$name.mp4" > "outputs/rel/eval_$name.log" 2>&1
OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 10 --step_timeout 8 --heldout_pairs \
  --video_out "outputs/rel/eval_${name}_heldout.mp4" > "outputs/rel/eval_${name}_heldout.log" 2>&1
kill $server; wait $server 2>/dev/null
grep -h "\[eval\] pick success" "outputs/rel/eval_$name.log" "outputs/rel/eval_${name}_heldout.log" | sed "s/^/[rel] /"
echo "[rel] $(date +%T) done"
