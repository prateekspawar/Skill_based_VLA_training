#!/usr/bin/env bash
# Continue the depth stage after the merge failure: strip point maps, merge, re-attach, targets, train, eval.
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-20000}
OUT=outputs/stage_depth
log() { echo "[depth] $(date +%T) $*"; }
L=datasets/lerobot
PY=~/lerobot-env/bin/python

log "merge workaround: stripping point maps"
for p in static moving; do
  [[ -f $L/s512_${p}_depth_nopm/meta/info.json ]] && continue
  $PY -m so101_lang.data.pointmap_merge strip $L/s512_${p}_depth $L/s512_${p}_depth_nopm >> $OUT/pointmap.log 2>&1 \
    || { log "strip $p failed"; exit 1; }
done
[[ -f $L/s512_mix_depth_nopm/meta/info.json ]] || ~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='["local/so101_stereo_ee", "local/so101_stereo_ee_moving"]' \
  --operation.roots="[\"$L/s512_static_depth_nopm\", \"$L/s512_moving_depth_nopm\"]" \
  --new_repo_id=local/so101_stereo_ee --new_root=$L/s512_mix_depth_nopm > $OUT/merge.log 2>&1 || { log "merge failed"; exit 1; }
log "re-attaching point maps"
$PY -m so101_lang.data.pointmap_merge attach $L/s512_mix_depth_nopm $L/s512_mix_depth_none \
  $L/s512_static_depth $L/s512_moving_depth >> $OUT/pointmap.log 2>&1 || { log "attach failed"; exit 1; }
cp $L/s512_static_depth/meta/stereo_calibration.json $L/s512_mix_depth_none/meta/
$PY -m so101_lang.data.build_targets --src $L/s512_mix_depth_none \
  --raw datasets/raw/s512_dart_1000.h5 datasets/raw/s512_dart_moving_1000.h5 \
  --dst $L/s512_mix_depth_target > $OUT/targets.log 2>&1 || { log "targets failed"; exit 1; }
tail -1 $OUT/targets.log | sed "s/^/[depth] /"

train() {
  local name=$1; shift
  log "training $name ($STEPS steps)"
  rm -rf checkpoints/$name
  env "$@" train/smolvla_depth.sh $L/s512_mix_depth_target checkpoints/$name \
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
