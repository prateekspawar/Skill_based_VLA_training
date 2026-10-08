#!/usr/bin/env bash
# Retrain pick / place / find on new demonstrations with per-episode random plate colours (language.PLATE_PALETTE)
# and realistic cube-yaw noise in the pick targets (empirical perception errors). Same data volume and number of
# passes as the current models. Usage: scripts/run_stage_randcolor.sh [steps]   (resumable: finished steps skip)
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}
OUT=outputs/stage_randcolor
R=datasets/raw
L=datasets/lerobot
PY=~/lerobot-env/bin/python
SIM=~/isaacsim/bin/python
mkdir -p $OUT
log() { echo "[randcolor] $(date +%T) $*"; }
export SO101_RANDOM_PLATE_COLORS=1 OMNI_KIT_ACCEPT_EULA=YES

gen() {  # name, args...
  local name=$1; shift
  [[ -f $R/$name.h5 ]] && grep -q "done" $OUT/gen_$name.log 2>/dev/null && return 0
  log "generating $name"
  $SIM "$@" --out $R/$name.h5 > $OUT/gen_$name.log 2>&1 || { log "gen $name failed"; exit 1; }
  tail -1 $OUT/gen_$name.log | sed "s/^/[randcolor] /"
}
gen rc_static_1000 scripts/gen_demos.py --num_envs 16 --episodes 1000 --seed 31
gen rc_moving_1000 scripts/gen_demos.py --num_envs 16 --episodes 1000 --seed 32 --move_prob_pick 0.5 --move_prob_place 0.5
gen rc_fail_400 scripts/gen_fail_demos.py --num_envs 16 --episodes 400 --seed 33
gen rc_find_1000 scripts/gen_find_demos.py --num_envs 16 --episodes 1000 --seed 34
gen rc_find_absent_200 scripts/gen_find_demos.py --num_envs 16 --episodes 200 --absent_prob 1.0 --seed 35

if [[ ! -f $L/s512_rc_all9/meta/info.json ]]; then
  log "converting (point map + status), 3 in parallel"
  for n in rc_static_1000 rc_moving_1000 rc_fail_400; do
    [[ -f $L/s512_$n/meta/info.json ]] && continue
    $PY -m so101_lang.data.hdf5_to_lerobot --inputs $R/$n.h5 --root $L/s512_$n --depth none \
      --pointmap fast_foundationstereo --status --overwrite > $OUT/convert_$n.log 2>&1 &
  done
  wait
  for n in rc_static_1000 rc_moving_1000 rc_fail_400; do
    grep -q "\[convert\] wrote" $OUT/convert_$n.log || { log "convert $n failed"; exit 1; }
    $PY -m so101_lang.data.pointmap_merge strip $L/s512_$n $L/_${n}_nopm >> $OUT/pointmap.log 2>&1 || { log "strip $n failed"; exit 1; }
  done
  log "merging"
  rm -rf $L/s512_rc_all_nopm $L/s512_rc_all
  ~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge \
    --operation.repo_ids='["local/rc_static", "local/rc_moving", "local/rc_fail"]' \
    --operation.roots="[\"$L/_rc_static_1000_nopm\", \"$L/_rc_moving_1000_nopm\", \"$L/_rc_fail_400_nopm\"]" \
    --new_repo_id=local/so101_stereo_ee --new_root=$L/s512_rc_all_nopm > $OUT/merge.log 2>&1 || { log "merge failed"; exit 1; }
  $PY -m so101_lang.data.pointmap_merge attach $L/s512_rc_all_nopm $L/s512_rc_all \
    $L/s512_rc_static_1000 $L/s512_rc_moving_1000 $L/s512_rc_fail_400 >> $OUT/pointmap.log 2>&1 || { log "attach failed"; exit 1; }
  cp $L/s512_rc_static_1000/meta/stereo_calibration.json $L/s512_rc_all/meta/
  log "targets (9-D, cube yaw folded, empirical yaw noise)"
  $PY -m so101_lang.data.build_targets --src $L/s512_rc_all --raw $R/rc_static_1000.h5 $R/rc_moving_1000.h5 $R/rc_fail_400.h5 \
    --rot6d --yaw_noise empirical --dst $L/s512_rc_all9 > $OUT/targets.log 2>&1 || { log "targets failed"; exit 1; }
  tail -1 $OUT/targets.log | sed "s/^/[randcolor] /"
fi
for sk in pick place; do
  [[ -f $L/s512_${sk}9rc/meta/info.json ]] && continue
  $PY -m so101_lang.data.skill_subset --src $L/s512_rc_all9 --dst $L/_${sk}9rc_unmasked --skill $sk > $OUT/subset_$sk.log 2>&1 \
    || { log "subset $sk failed"; exit 1; }
  $PY -m so101_lang.data.mask_failure_motion --src $L/_${sk}9rc_unmasked --dst $L/s512_${sk}9rc >> $OUT/subset_$sk.log 2>&1 \
    || { log "mask $sk failed"; exit 1; }
  tail -2 $OUT/subset_$sk.log | sed "s/^/[randcolor] /"
done
if [[ ! -f $L/s512_find9rc/meta/info.json ]]; then
  log "converting find"
  $PY -m so101_lang.data.hdf5_to_lerobot --inputs $R/rc_find_1000.h5 $R/rc_find_absent_200.h5 --root $L/s512_find9rc \
    --depth none --pointmap fast_foundationstereo --status --pad_state 9 --overwrite > $OUT/convert_find.log 2>&1 \
    || { log "convert find failed"; exit 1; }
fi
log "training pick, place, find ($STEPS steps each)"
DS_SUFFIX=rc scripts/run_stage_skills.sh $STEPS pick place find || { log "training failed"; exit 1; }
log "done"
