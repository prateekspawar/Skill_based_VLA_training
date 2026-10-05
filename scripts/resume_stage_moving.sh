#!/usr/bin/env bash
# Resume the interrupted stage-1 training from its last checkpoint, then run the SAM2 evaluation.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/stage_moving
CKPT=checkpoints/mix_target_20000/checkpoints/last/pretrained_model
echo "[stage] $(date +%T) resuming training from $(readlink checkpoints/mix_target_20000/checkpoints/last)"
~/lerobot-env/bin/python -m so101_lang.train_cli --config_path=$CKPT/train_config.json --resume=true \
  > $OUT/train_resume.log 2>&1
echo "[stage] $(date +%T) evaluating (SAM2 targets, 50 paired episodes)"
scripts/eval_sam2_pair.sh $OUT $CKPT | sed "s/^\[eval\] /[stage] /"
echo "[stage] $(date +%T) done"
