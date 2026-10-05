#!/usr/bin/env bash
# Fine-tune SmolVLA + native depth branch (smolvla_depth) on a dataset with observation.pointmap.
# Cameras: left -> camera1, right -> camera2, camera3 empty; depth enters as point-map tokens instead.
# Set DEPTH_CAMERA=1 to feed observation.images.depth (colormap depth) as camera3 instead of leaving it empty.
# Usage: [DEPTH_CAMERA=1] train/smolvla_depth.sh <dataset_root> <output_dir> [extra lerobot-train args...]
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET_ROOT=${1:?dataset root}
OUTPUT_DIR=${2:?output dir}
shift 2
RENAME='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}'
EMPTY=1
if [[ "${DEPTH_CAMERA:-0}" == "1" ]]; then
  RENAME='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2", "observation.images.depth": "observation.images.camera3"}'
  EMPTY=0
fi
exec ~/lerobot-env/bin/python -m so101_lang.train_cli \
  --policy.path=models/smolvla_depth_base \
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
  --job_name="smolvla_depth_$(basename "$OUTPUT_DIR")" \
  "$@"
