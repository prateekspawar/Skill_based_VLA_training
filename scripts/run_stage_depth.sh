#!/usr/bin/env bash
# Depth comparison on the stage-1 recipe: D1 = colormap depth as camera3, D2 = native point-map tokens.
# Same 2000 episodes, SAM2-style 13-D state, 20k steps, SAM2 evaluation; baseline = checkpoints/mix_target_20000.
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-20000}
OUT=outputs/stage_depth
mkdir -p $OUT
log() { echo "[depth] $(date +%T) $*"; }

log "converting static + moving episodes with depth video and point map (parallel)"
for part in static:s512_dart_1000 moving:s512_dart_moving_1000; do
  name=${part%%:*}; raw=${part#*:}
  ~/lerobot-env/bin/python -m so101_lang.data.hdf5_to_lerobot --inputs datasets/raw/$raw.h5 \
    --root datasets/lerobot/s512_${name}_depth --depth fast_foundationstereo --pointmap fast_foundationstereo \
    --overwrite > $OUT/convert_$name.log 2>&1 &
done
wait
for name in static moving; do
  grep -q "\[convert\] wrote" $OUT/convert_$name.log || { log "convert $name failed"; exit 1; }
done

log "merging + building SAM2-style targets"
rm -rf datasets/lerobot/s512_mix_depth_none
~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='["local/so101_stereo_ee", "local/so101_stereo_ee_moving"]' \
  --operation.roots='["datasets/lerobot/s512_static_depth", "datasets/lerobot/s512_moving_depth"]' \
  --new_repo_id=local/so101_stereo_ee --new_root=datasets/lerobot/s512_mix_depth_none > $OUT/merge.log 2>&1 || { log "merge failed"; exit 1; }
cp datasets/lerobot/s512_static_depth/meta/stereo_calibration.json datasets/lerobot/s512_mix_depth_none/meta/
~/lerobot-env/bin/python -m so101_lang.data.build_targets --src datasets/lerobot/s512_mix_depth_none \
  --raw datasets/raw/s512_dart_1000.h5 datasets/raw/s512_dart_moving_1000.h5 \
  --dst datasets/lerobot/s512_mix_depth_target > $OUT/targets.log 2>&1 || { log "targets failed"; exit 1; }
tail -1 $OUT/targets.log | sed "s/^/[depth] /"

train() {  # name, extra env/args
  local name=$1; shift
  log "training $name ($STEPS steps)"
  rm -rf checkpoints/$name
  env "$@" train/smolvla_depth.sh datasets/lerobot/s512_mix_depth_target checkpoints/$name \
    --policy.relative_action_dims=9 --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=5000 \
    $EXTRA > $OUT/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { log "training $name failed"; return 1; }
}
EXTRA="--policy.use_depth_branch=false" train d1_depthcam_$STEPS DEPTH_CAMERA=1 || exit 1
EXTRA="--policy.use_depth_branch=true" train d2_pointmap_$STEPS DEPTH_CAMERA=0 || exit 1

for name in d1_depthcam_$STEPS d2_pointmap_$STEPS; do
  log "evaluating $name (SAM2 targets, 50 paired episodes)"
  DEPTH=fast_foundationstereo scripts/eval_sam2_pair.sh $OUT/$name checkpoints/$name/checkpoints/last/pretrained_model \
    | sed "s/^\[eval\] /[depth] $name /"
done
log "done"
