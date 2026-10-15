#!/usr/bin/env bash
# Experiment 2 (2026-10-15): add a new skill ("stack": cube on cube) to the system.
#   (a) skill-level: a new stack model (frozen backbone + new action expert, like the other skills) trained on
#       100 / 300 stack demos; pick / place / find untouched.
#   (b) monolithic: the experiment-1 pick+place model fine-tuned on the same 300 stack demos (sequential FT).
# Evaluation (150 seeded episodes each): stacking (SO101_TASK_MODE=stack) for both, and pick+place again for the
# fine-tuned monolith (forgetting; before: outputs/exp1/eval_mono/still.log).
# Usage: scripts/run_exp2_stack.sh [steps]   (resumable)
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-5000}
O=outputs/exp2; R=datasets/raw; L=datasets/lerobot; PY=~/lerobot-env/bin/python
mkdir -p $O
log() { echo "[exp2] $(date +%T) $*"; }
export OMNI_KIT_ACCEPT_EULA=YES
if [[ ! -f $R/stack_500.h5 ]] || ! grep -q "done" $O/gen_stack_500.log 2>/dev/null; then
  log "generating stack_500"
  SO101_TASK_MODE=stack ~/isaacsim/bin/python scripts/gen_demos.py --num_envs 16 --episodes 500 --seed 51 \
    --out $R/stack_500.h5 > $O/gen_stack_500.log 2>&1 || { log "generation failed"; exit 1; }
fi
if [[ ! -f $L/s512_stack9_500/meta/info.json ]]; then
  log "converting + targets"
  $PY -m so101_lang.data.hdf5_to_lerobot --inputs $R/stack_500.h5 --root $L/s512_stack_500 --depth none \
    --pointmap fast_foundationstereo --status --overwrite > $O/convert.log 2>&1 || { log "convert failed"; exit 1; }
  $PY -m so101_lang.data.build_targets --src $L/s512_stack_500 --raw $R/stack_500.h5 --rot6d \
    --dst $L/s512_stack_all9 > $O/targets.log 2>&1 || { log "targets failed"; exit 1; }
  $PY -m so101_lang.data.skill_subset --src $L/s512_stack_all9 --dst $L/_stack9_unlabelled --skill stack > $O/subset.log 2>&1
  $PY -m so101_lang.data.relabel_release --src $L/_stack9_unlabelled --dst $L/s512_stack9_500 >> $O/subset.log 2>&1 \
    || { log "subset failed"; exit 1; }
  for n in 100 300; do  # first n episodes
    $PY -c "import json; json.dump(list(range($n)), open('$O/episodes_$n.json', 'w'))"
    $PY -m so101_lang.data.skill_subset --src $L/s512_stack9_500 --dst $L/s512_stack9_$n --skill stack \
      --episodes $O/episodes_$n.json >> $O/subset.log 2>&1 || { log "subset $n failed"; exit 1; }
  done
  tail -4 $O/subset.log | sed "s/^/[exp2] /"
fi
train() {  # name, dataset, [extra args]
  local name=$1 ds=$2; shift 2
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] && return 0
  log "training $name"
  DEPTH_CAMERA=0 train/smolvla_depth.sh $L/$ds checkpoints/$name --policy.use_depth_branch=true \
    --policy.relative_action_dims=9 --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=2500 "$@" \
    > $O/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { log "training $name failed"; exit 1; }
}
train stack9_100_$STEPS s512_stack9_100
train stack9_300_$STEPS s512_stack9_300
# Monolith fine-tuned on the stack demos, starting from the experiment-1 checkpoint.
name=mono9rc2_ft_stack300_$STEPS
if [[ ! -d checkpoints/$name/checkpoints/last/pretrained_model ]]; then
  log "training $name"
  ~/lerobot-env/bin/python -m so101_lang.train_cli \
    --policy.path=checkpoints/mono9rc2_30000/checkpoints/last/pretrained_model --policy.push_to_hub=false \
    --policy.device=cuda --policy.empty_cameras=1 \
    --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}' \
    --dataset.repo_id=local/so101_stereo_ee --dataset.root=$L/s512_stack9_300 --dataset.video_backend=pyav \
    --batch_size=32 --num_workers=8 --steps=$STEPS --save_freq=2500 --log_freq=100 --seed=1000 --wandb.enable=false \
    --policy.scheduler_decay_steps=$STEPS --output_dir=checkpoints/$name --job_name=$name > $O/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { log "training $name failed"; exit 1; }
fi
ev() {  # out_dir, runs, env...
  local out=$1; shift
  [[ -f $O/$out/still.log ]] && grep -q "pick success" $O/$out/still.log && return 0
  log "eval $out"
  env "$@" EVAL_OUT=$O/$out scripts/eval.sh still | grep -E "pick success|place failures" | sed "s/^/[exp2] $out: /"
}
ev stack_skill_100 SO101_TASK_MODE=stack CK_STACK=stack9_100_$STEPS
ev stack_skill_300 SO101_TASK_MODE=stack CK_STACK=stack9_300_$STEPS
ev stack_mono_ft SO101_TASK_MODE=stack CK_PICK=$name CK_PLACE=$name CK_STACK=$name
ev place_mono_ft CK_PICK=$name CK_PLACE=$name
log "done"
