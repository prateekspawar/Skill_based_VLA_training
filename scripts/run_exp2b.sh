#!/usr/bin/env bash
# Experiment 2, follow-ups (2026-10-16):
#   replay:  the experiment-1 monolith fine-tuned on the 300 stack demos + 290 replayed pick/place episodes (every
#            10th of s512_mono9rc2) - the standard fix for the forgetting of plain sequential fine-tuning (0 %).
#   gate:    the stack model (300 demos) with a stacking release tolerance (12 mm sideways, 5 cm above the base
#            cube centre) instead of the plate one (4 cm / 6.5 cm): stack releases were centred but too high.
#   reuse:   the existing place model used for stacking, zero-shot (no stack training).
# Dataset build runs at once (CPU); training and evaluation wait for experiment 4 (GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-5000}
O=outputs/exp2; L=datasets/lerobot; PY=~/lerobot-env/bin/python
log() { echo "[exp2b] $(date +%T) $*"; }
if [[ ! -f $L/s512_replay_stack300/meta/info.json ]]; then
  log "building replay dataset"
  $PY -c "import json; json.dump(list(range(0, 2900, 10)), open('$O/episodes_replay.json', 'w'))"
  $PY -m so101_lang.data.skill_subset --src $L/s512_mono9rc2 --dst $L/s512_mono9rc2_r290 --skill all \
    --episodes $O/episodes_replay.json > $O/replay.log 2>&1 || { log "replay subset failed"; exit 1; }
  for d in s512_mono9rc2_r290 s512_stack9_300; do
    $PY -m so101_lang.data.pointmap_merge strip $L/$d $L/_${d}_nopm >> $O/replay.log 2>&1 || { log "strip $d failed"; exit 1; }
  done
  rm -rf $L/_replay_nopm
  ~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge \
    --operation.repo_ids='["local/r290", "local/stack300"]' \
    --operation.roots="[\"$L/_s512_mono9rc2_r290_nopm\", \"$L/_s512_stack9_300_nopm\"]" \
    --new_repo_id=local/so101_stereo_ee --new_root=$L/_replay_nopm >> $O/replay.log 2>&1 || { log "merge failed"; exit 1; }
  $PY -m so101_lang.data.pointmap_merge attach $L/_replay_nopm $L/s512_replay_stack300 $L/s512_mono9rc2_r290 \
    $L/s512_stack9_300 >> $O/replay.log 2>&1 || { log "attach failed"; exit 1; }
  cp $L/s512_stack9_300/meta/stereo_calibration.json $L/s512_replay_stack300/meta/
  $PY -m so101_lang.data.restore_nan $L/s512_replay_stack300 >> $O/replay.log 2>&1  # merge turns NaN into null
  log "replay dataset: $($PY -c "import json; i=json.load(open('$L/s512_replay_stack300/meta/info.json')); print(i['total_episodes'], 'episodes', i['total_frames'], 'frames')")"
fi
until grep -qE "^\[exp4\] [0-9:]+ done" logs/exp4.log 2>/dev/null; do sleep 120; done  # GPU
name=mono9rc2_ft_replay_$STEPS
if [[ ! -d checkpoints/$name/checkpoints/last/pretrained_model ]]; then
  log "training $name"
  rm -rf checkpoints/$name
  $PY -m so101_lang.train_cli \
    --policy.path=checkpoints/mono9rc2_30000/checkpoints/last/pretrained_model --policy.push_to_hub=false \
    --policy.device=cuda --policy.empty_cameras=1 \
    --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}' \
    --dataset.repo_id=local/so101_stereo_ee --dataset.root=$L/s512_replay_stack300 --dataset.video_backend=pyav \
    --batch_size=32 --num_workers=8 --steps=$STEPS --save_freq=2500 --log_freq=100 --seed=1000 --wandb.enable=false \
    --policy.scheduler_decay_steps=$STEPS --output_dir=checkpoints/$name --job_name=$name > $O/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { log "training $name failed"; exit 1; }
fi
ev() {  # out_dir, env...
  local out=$1; shift
  [[ -f $O/$out/still.log ]] && grep -q "pick success" $O/$out/still.log && return 0
  log "eval $out"
  env "$@" EVAL_OUT=$O/$out scripts/eval.sh still | grep -E "pick success|place failures" | sed "s/^/[exp2b] $out: /"
}
ev stack_mono_replay SO101_TASK_MODE=stack CK_PICK=$name CK_PLACE=$name CK_STACK=$name
ev place_mono_replay CK_PICK=$name CK_PLACE=$name
ev stack_skill_300_gate SO101_TASK_MODE=stack CK_STACK=stack9_300_5000 EVAL_EXTRA="--release_gate 0.012 0.05"
ev stack_reuse_place SO101_TASK_MODE=stack CK_STACK=place9rc2_15000
ev stack_reuse_place_gate SO101_TASK_MODE=stack CK_STACK=place9rc2_15000 EVAL_EXTRA="--release_gate 0.012 0.05"
log "done"
