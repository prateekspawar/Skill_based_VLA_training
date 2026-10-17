#!/usr/bin/env bash
# Experiment 3 (2026-10-17): approaches on unseen multi-step tasks, without an LLM planner (oracle skill sequences),
# so only execution is compared. None of these tasks was ever a training episode.
#   pairs        put cube X on plate A and cube Y on plate B          (pick, place, pick, place)
#   plate_stack  put X on plate A, then stack Y on top of it          (pick, place, pick, stack)
#   tower        stack Y on X, then Z on top of Y (3-cube tower)      (pick, stack, pick, stack)
# Approaches, 50 seeded episodes per task (same scenes):
#   skills     pick9rc2 / place9rc2 / find9rc + the better stack model of
#              stack9_300 (from scratch) and stack9_place300 (initialised from the place model, trained first here:
#              experiment 2c, the modular counterpart of the monolith fine-tuned with replay)
#   monolith   mono9rc2_ft_replay_5000 for pick, place and stack (find9rc)
set -uo pipefail
cd "$(dirname "$0")/.."
O=outputs/exp3; O2=outputs/exp2; L=datasets/lerobot; PY=~/lerobot-env/bin/python
mkdir -p $O
log() { echo "[exp3] $(date +%T) $*"; }
until grep -qE "^\[exp2b\] [0-9:]+ done" logs/exp2b.log 2>/dev/null; do sleep 120; done
# Experiment 2c: new stack skill initialised from the place model (5k steps on the same 300 demos).
name=stack9_place300_5000
if [[ ! -d checkpoints/$name/checkpoints/last/pretrained_model ]]; then
  log "training $name (from place9rc2_15000)"
  rm -rf checkpoints/$name
  $PY -m so101_lang.train_cli \
    --policy.path=checkpoints/place9rc2_15000/checkpoints/last/pretrained_model --policy.push_to_hub=false \
    --policy.device=cuda --policy.empty_cameras=1 \
    --rename_map='{"observation.images.left": "observation.images.camera1", "observation.images.right": "observation.images.camera2"}' \
    --dataset.repo_id=local/so101_stereo_ee --dataset.root=$L/s512_stack9_300 --dataset.video_backend=pyav \
    --batch_size=32 --num_workers=8 --steps=5000 --save_freq=2500 --log_freq=100 --seed=1000 --wandb.enable=false \
    --policy.scheduler_decay_steps=5000 --output_dir=checkpoints/$name --job_name=$name > $O2/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { log "training $name failed"; exit 1; }
fi
if ! grep -q "pick success" $O2/stack_skill_place300/still.log 2>/dev/null; then
  log "eval stack_skill_place300"
  SO101_TASK_MODE=stack CK_STACK=$name EVAL_OUT=$O2/stack_skill_place300 scripts/eval.sh still \
    | grep -E "pick success|place failures" | sed "s/^/[exp3] stack_skill_place300: /"
fi
full() { grep -oE "full pick\+place [0-9]+" $1 2>/dev/null | grep -oE "[0-9]+$" || echo 0; }
STACK_CK=stack9_300_5000
(( $(full $O2/stack_skill_place300/still.log) > $(full $O2/stack_skill_300/still.log) )) && STACK_CK=$name
log "skills use $STACK_CK"
mono=mono9rc2_ft_replay_5000
for c in pairs plate_stack tower; do
  for a in skills mono; do
    out=$O/${c}_$a
    [[ -f $out/still.log ]] && grep -q "compose $c" $out/still.log && continue
    log "running $c ($a)"
    if [[ $a == skills ]]; then ck="CK_STACK=$STACK_CK"; else ck="CK_PICK=$mono CK_PLACE=$mono CK_STACK=$mono"; fi
    env $ck EPISODES=50 EVAL_OUT=$out EVAL_EXTRA="--compose $c --planner oracle" scripts/eval.sh still > /dev/null
    grep -E "\[eval\] compose" $out/still.log | sed "s/^/[exp3] ${c}_$a: /"
  done
done
log "done"
