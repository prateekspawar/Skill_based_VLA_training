#!/usr/bin/env bash
# Fine-tune SmolVLA on the simulated SO-101 pick-and-place dataset.
# Usage: train/smolvla.sh <dataset_root> <output_dir> [extra lerobot-train args...]
set -euo pipefail
DATASET_ROOT=${1:?dataset root}
OUTPUT_DIR=${2:?output dir}
shift 2
exec ~/lerobot-env/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=1 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --dataset.repo_id=local/so101_pickplace_sim \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --num_workers=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_so101_pickplace \
  "$@"
