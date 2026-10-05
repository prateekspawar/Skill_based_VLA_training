#!/usr/bin/env bash
# Stage: one model per skill (pick, place) on static + moving episodes, then SAM2 evaluation with both models.
# Usage: scripts/run_stage_per_skill.sh <steps per model>
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:?steps}
OUT=outputs/stage_per_skill
RAWS="datasets/raw/s512_dart_1000.h5 datasets/raw/s512_dart_moving_1000.h5"
mkdir -p $OUT
log() { echo "[skills] $(date +%T) $*"; }

for sk in pick place; do
  ( ~/lerobot-env/bin/python -m so101_lang.data.hdf5_to_lerobot --inputs $RAWS --root datasets/lerobot/s512_${sk}_none \
      --depth none --skill $sk --overwrite > $OUT/convert_$sk.log 2>&1 \
    && cp datasets/lerobot/s512_dart_none/meta/stereo_calibration.json datasets/lerobot/s512_${sk}_none/meta/ \
    && ~/lerobot-env/bin/python -m so101_lang.data.build_targets --src datasets/lerobot/s512_${sk}_none --raw $RAWS \
      --skill $sk --dst datasets/lerobot/s512_${sk}_target > $OUT/targets_$sk.log 2>&1 ) &
done
log "converting per-skill datasets (parallel)"
wait
for sk in pick place; do
  [[ -d datasets/lerobot/s512_${sk}_target ]] || { log "dataset for $sk failed"; exit 1; }
  tail -1 $OUT/targets_$sk.log | sed "s/^/[skills] $sk: /"
done

for sk in pick place; do
  log "training $sk model ($STEPS steps)"
  rm -rf checkpoints/${sk}_target_$STEPS
  train/smolvla_depth.sh datasets/lerobot/s512_${sk}_target checkpoints/${sk}_target_$STEPS --policy.use_depth_branch=false \
    --policy.relative_action_dims=9 --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=$((STEPS / 2)) \
    > $OUT/train_$sk.log 2>&1
  [[ -d checkpoints/${sk}_target_$STEPS/checkpoints/last/pretrained_model ]] || { log "training $sk failed"; exit 1; }
done

log "evaluating pick model + place model (SAM2 targets, 50 paired episodes)"
~/lerobot-env/bin/python -m so101_lang.serve.policy_server \
  --checkpoint checkpoints/pick_target_$STEPS/checkpoints/last/pretrained_model \
  --checkpoint_place checkpoints/place_target_$STEPS/checkpoints/last/pretrained_model --depth none > $OUT/server.log 2>&1 &
server=$!
until grep -q listening $OUT/server.log; do sleep 3; done
for v in still push; do
  extra=""; [[ $v == push ]] && extra="--push"
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 50 --step_timeout 8 --seed 0 \
    --target sam2 $extra --video_out $OUT/eval_$v.mp4 > $OUT/eval_$v.log 2>&1
  grep -E "\[eval\] (pick success|pushes|target-point)" $OUT/eval_$v.log | sed "s/^/[skills] $v: /"
done
kill $server; wait $server 2>/dev/null
log "done"
