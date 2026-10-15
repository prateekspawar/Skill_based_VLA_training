#!/usr/bin/env bash
# Experiment 1 (2026-10-15): one monolithic pick+place model vs separate pick / place models, same data
# (s512_mono9rc2 = s512_pick9rc2 + s512_place9rc2 frames, per-segment labels) and same total compute
# (30k steps vs 2 x 15k), same recipe. Evaluated with the defaults (find9rc, grasp check + retry, release gate)
# on the same 150 still + 150 push scenes as outputs/eval_gate.
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-30000}
name=mono9rc2_$STEPS
mkdir -p outputs/exp1
echo "[exp1] $(date +%T) training $name"
[[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || DEPTH_CAMERA=0 train/smolvla_depth.sh \
  datasets/lerobot/s512_mono9rc2 checkpoints/$name --policy.use_depth_branch=true --policy.relative_action_dims=9 \
  --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=5000 > outputs/exp1/train_$name.log 2>&1
[[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { echo "[exp1] training failed"; exit 1; }
echo "[exp1] $(date +%T) evaluating"
CK_PICK=$name CK_PLACE=$name EVAL_OUT=outputs/exp1/eval_mono scripts/eval.sh still push
echo "[exp1] $(date +%T) done"
