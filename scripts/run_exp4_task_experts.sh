#!/usr/bin/env bash
# Experiment 4 (2026-10-15): task-level experts (CORAL-style: one pick+place model per instruction, trained only on
# that instruction's episodes) vs the skill-level experts. Same recipe; 4 x 7.5k steps = the 30k-step budget of
# experiment 1. Datasets s512_task_{bb,ry,bw,rg} (skill_subset --skill all --episodes outputs/exp4_episodes_*.json).
# Evaluation: stationary, 50 matching episodes per task (--only_tasks; same seeded scene sequence):
#   seen tasks: each task with its own expert, and with the skill experts (defaults);
#   unseen tasks: routed to the expert with the same cube colour, and with the skill experts.
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-7500}
O=outputs/exp4
mkdir -p $O
log() { echo "[exp4] $(date +%T) $*"; }
declare -A TASK=([bb]="blue cube:blue plate" [ry]="red cube:yellow plate" [bw]="blue cube:white plate" [rg]="red cube:green plate")
for k in bb ry bw rg; do
  name=task_${k}_$STEPS
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] && continue
  log "training $name"
  DEPTH_CAMERA=0 train/smolvla_depth.sh datasets/lerobot/s512_task_$k checkpoints/$name --policy.use_depth_branch=true \
    --policy.relative_action_dims=9 --policy.scheduler_decay_steps=$STEPS --steps=$STEPS --save_freq=2500 \
    > $O/train_$name.log 2>&1
  [[ -d checkpoints/$name/checkpoints/last/pretrained_model ]] || { log "training $name failed"; exit 1; }
done
ev() {  # out_dir, tasks, [CK_PICK=... CK_PLACE=...]
  local out=$1 tasks=$2; shift 2
  [[ -f $O/$out/still.log ]] && grep -q "pick success" $O/$out/still.log && return 0
  log "eval $out ($tasks)"
  env "$@" EPISODES=4000 EVAL_OUT=$O/$out SO101_ONLY_TASKS="$tasks" SO101_TASK_EPISODES=50 scripts/eval.sh still \
    | grep "pick success" | sed "s/^/[exp4] $out: /"
}
for k in bb ry bw rg; do
  ev seen_task_$k "${TASK[$k]}" CK_PICK=task_${k}_$STEPS CK_PLACE=task_${k}_$STEPS
  ev seen_skill_$k "${TASK[$k]}"
done
ev unseen_task_bo "blue cube:orange plate" CK_PICK=task_bb_$STEPS CK_PLACE=task_bb_$STEPS
ev unseen_skill_bo "blue cube:orange plate"
ev unseen_task_rp "red cube:purple plate" CK_PICK=task_ry_$STEPS CK_PLACE=task_ry_$STEPS
ev unseen_skill_rp "red cube:purple plate"
log "done"
