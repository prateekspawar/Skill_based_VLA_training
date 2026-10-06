#!/usr/bin/env bash
# Separate skill models (shared frozen backbone): train pick and place on 19-D state (9-D target pose),
# depth branch on. Usage: scripts/run_stage_skills.sh <steps> [skills...]
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}; shift || true
SKILLS=${*:-pick place}
OUT=outputs/stage_skills
mkdir -p $OUT
log() { echo "[skills] $(date +%T) $*"; }
for sk in $SKILLS; do
  name=${sk}9_$STEPS
  log "training $name"
  rm -rf checkpoints/$name
  DEPTH_CAMERA=0 train/smolvla_depth.sh datasets/lerobot/s512_${sk}9 checkpoints/$name --policy.use_depth_branch=true \
    --policy.relative_action_dims=9 --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=5000 \
    > $OUT/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] && log "trained $name" || { log "training $name failed"; exit 1; }
done
log "done"
