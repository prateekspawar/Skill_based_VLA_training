#!/usr/bin/env bash
# Fine-tune SmolVLA on the SO-101 stereo dataset (robot-agnostic 10-D head-frame EE state/action).
# Camera slots: left -> camera1, right -> camera2, depth (TURBO colormap) -> camera3.
# Set DEPTH=0 for a dataset without observation.images.depth (camera3 becomes an empty slot).
# Usage: [DEPTH=0|1] train/smolvla.sh <dataset_root> <output_dir> [extra lerobot-train args...]
set -euo pipefail
DATASET_ROOT=${1:?dataset root}
OUTPUT_DIR=${2:?output dir}
shift 2
if [[ "${DEPTH:-1}" == "1" ]]; then
  RENAME='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2", "observation.images.depth": "observation.images.camera3"}'
  EMPTY=0
else
  RENAME='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}'
  EMPTY=1
fi
exec ~/lerobot-env/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.empty_cameras=$EMPTY \
  --rename_map="$RENAME" \
  --dataset.repo_id=local/so101_stereo_ee \
  --dataset.root="$DATASET_ROOT" \
  --dataset.video_backend=pyav \
  --batch_size=32 \
  --num_workers=8 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=100 \
  --seed=1000 \
  --wandb.enable=false \
  --output_dir="$OUTPUT_DIR" \
  --job_name="smolvla_so101_$(basename "$OUTPUT_DIR")" \
  "$@"
