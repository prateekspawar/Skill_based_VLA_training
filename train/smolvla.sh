#!/usr/bin/env bash
# Fine-tune SmolVLA on the stereo + depth SO-101 dataset (robot-agnostic 10-D head-frame EE state/action).
# Camera slots: left -> camera1, right -> camera2, stereo depth (TURBO colormap) -> camera3.
# Usage: train/smolvla.sh <dataset_root> <output_dir> [extra lerobot-train args...]
set -euo pipefail
DATASET_ROOT=${1:?dataset root}
OUTPUT_DIR=${2:?output dir}
shift 2
exec ~/lerobot-env/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=0 \
  --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2", "observation.images.depth": "observation.images.camera3"}' \
  --dataset.repo_id=local/so101_stereo_ee \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --num_workers=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name=smolvla_so101_stereo_depth \
  "$@"
