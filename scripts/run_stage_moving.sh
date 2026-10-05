#!/usr/bin/env bash
# Stage: full training on static + moving-target episodes, then SAM2 evaluation (stationary + sliding push).
# Usage: scripts/run_stage_moving.sh <steps>
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:?steps}
OUT=outputs/stage_moving
NAME=mix_target_${STEPS}
CKPT=checkpoints/$NAME/checkpoints/last/pretrained_model
mkdir -p $OUT
log() { echo "[stage] $(date +%T) $*"; }

until grep -q "\[gen\] done" outputs/gen_moving_1000.log 2>/dev/null; do
  grep -q Traceback outputs/gen_moving_1000.log 2>/dev/null && { log "generation failed"; exit 1; }
  sleep 30
done
log "$(grep '\[gen\] done' outputs/gen_moving_1000.log)"

log "converting moving episodes"
~/lerobot-env/bin/python -m so101_lang.data.hdf5_to_lerobot --inputs datasets/raw/s512_dart_moving_1000.h5 \
  --root datasets/lerobot/s512_moving_none --depth none --overwrite > $OUT/convert.log 2>&1 || { log "convert failed"; exit 1; }

log "merging static + moving"
rm -rf datasets/lerobot/s512_mix_none
~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='["local/so101_stereo_ee", "local/so101_stereo_ee_moving"]' \
  --operation.roots='["datasets/lerobot/s512_dart_none", "datasets/lerobot/s512_moving_none"]' \
  --new_repo_id=local/so101_stereo_ee --new_root=datasets/lerobot/s512_mix_none > $OUT/merge.log 2>&1 || { log "merge failed"; exit 1; }
cp datasets/lerobot/s512_dart_none/meta/stereo_calibration.json datasets/lerobot/s512_mix_none/meta/

log "building SAM2-style targets"
~/lerobot-env/bin/python -m so101_lang.data.build_targets --src datasets/lerobot/s512_mix_none \
  --raw datasets/raw/s512_dart_1000.h5 datasets/raw/s512_dart_moving_1000.h5 \
  --dst datasets/lerobot/s512_mix_target > $OUT/targets.log 2>&1 || { log "targets failed"; exit 1; }
tail -1 $OUT/targets.log | sed "s/^/[stage] /"

log "training $NAME"
rm -rf checkpoints/$NAME
train/smolvla_depth.sh datasets/lerobot/s512_mix_target checkpoints/$NAME --policy.use_depth_branch=false \
  --policy.relative_action_dims=9 --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=$((STEPS / 2)) \
  > $OUT/train.log 2>&1
[[ -d $CKPT ]] || { log "training failed"; exit 1; }

log "evaluating (SAM2 targets, 50 paired episodes)"
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint $CKPT --depth none > $OUT/server.log 2>&1 &
server=$!
until grep -q listening $OUT/server.log; do sleep 3; done
for v in still push; do
  extra=""; [[ $v == push ]] && extra="--push"
  OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --episodes 50 --step_timeout 8 --seed 0 \
    --target sam2 $extra --video_out $OUT/eval_$v.mp4 > $OUT/eval_$v.log 2>&1
  grep -E "\[eval\] (pick success|pushes|target-point)" $OUT/eval_$v.log | sed "s/^/[stage] $v: /"
done
kill $server; wait $server 2>/dev/null
log "done"
