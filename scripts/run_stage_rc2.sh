#!/usr/bin/env bash
# Retrain pick + place (2026-10-11) on the post-release demos: rc2_static_1000 + rc2_moving_1000 + rc2_latepush_500
# (success label on a completed release; cube sliding during the descent) + the existing rc_fail_400 failure demos.
# Pick targets get the 3-deg Gaussian yaw noise again (the empirical noise of *9rc left the jaws 14-17 deg off the
# cube faces). find9rc_15000 is kept. Then evaluation: still + push (defaults), and still without the forced release.
# Usage: scripts/run_stage_rc2.sh [steps]   (resumable: finished steps skip; waits for run_gen_latepush.sh)
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}
OUT=outputs/stage_rc2
R=datasets/raw
L=datasets/lerobot
PY=~/lerobot-env/bin/python
mkdir -p $OUT
log() { echo "[rc2] $(date +%T) $*"; }
until grep -qE "^\[latepush\] [0-9:]+ (done|failed)" logs/gen_latepush.log 2>/dev/null; do sleep 60; done
grep -q failed logs/gen_latepush.log && { log "late-push generation failed"; exit 1; }
SETS="rc2_static_1000 rc2_moving_1000 rc2_latepush_500 rc_fail_400"

if [[ ! -f $L/s512_rc2_all9/meta/info.json ]]; then
  log "converting (point map + status), 4 in parallel"
  for n in $SETS; do
    [[ -f $L/s512_$n/meta/info.json ]] && continue
    $PY -m so101_lang.data.hdf5_to_lerobot --inputs $R/$n.h5 --root $L/s512_$n --depth none \
      --pointmap fast_foundationstereo --status --overwrite > $OUT/convert_$n.log 2>&1 &
  done
  wait
  roots=""; ids=""; full=""
  for n in $SETS; do
    grep -q "\[convert\] wrote" $OUT/convert_$n.log || { log "convert $n failed"; exit 1; }
    [[ -d $L/_${n}_nopm ]] || $PY -m so101_lang.data.pointmap_merge strip $L/s512_$n $L/_${n}_nopm >> $OUT/pointmap.log 2>&1 \
      || { log "strip $n failed"; exit 1; }
    roots+="\"$L/_${n}_nopm\", "; ids+="\"local/$n\", "; full+="$L/s512_$n "
  done
  log "merging"
  rm -rf $L/s512_rc2_all_nopm $L/s512_rc2_all
  ~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge --operation.repo_ids="[${ids%, }]" \
    --operation.roots="[${roots%, }]" --new_repo_id=local/so101_stereo_ee --new_root=$L/s512_rc2_all_nopm \
    > $OUT/merge.log 2>&1 || { log "merge failed"; exit 1; }
  $PY -m so101_lang.data.pointmap_merge attach $L/s512_rc2_all_nopm $L/s512_rc2_all $full >> $OUT/pointmap.log 2>&1 \
    || { log "attach failed"; exit 1; }
  cp $L/s512_rc2_static_1000/meta/stereo_calibration.json $L/s512_rc2_all/meta/
  log "targets (9-D, cube yaw folded, 3-deg Gaussian yaw noise)"
  $PY -m so101_lang.data.build_targets --src $L/s512_rc2_all --raw $(for n in $SETS; do echo -n "$R/$n.h5 "; done) \
    --rot6d --dst $L/s512_rc2_all9 > $OUT/targets.log 2>&1 || { log "targets failed"; exit 1; }
  tail -1 $OUT/targets.log | sed "s/^/[rc2] /"
fi
for sk in pick place; do
  [[ -f $L/s512_${sk}9rc2/meta/info.json ]] && continue
  $PY -m so101_lang.data.skill_subset --src $L/s512_rc2_all9 --dst $L/_${sk}9rc2_unmasked --skill $sk > $OUT/subset_$sk.log 2>&1 \
    || { log "subset $sk failed"; exit 1; }
  $PY -m so101_lang.data.mask_failure_motion --src $L/_${sk}9rc2_unmasked --dst $L/s512_${sk}9rc2 >> $OUT/subset_$sk.log 2>&1 \
    || { log "mask $sk failed"; exit 1; }
  tail -2 $OUT/subset_$sk.log | sed "s/^/[rc2] /"
done
log "training pick, place ($STEPS steps each)"
DS_SUFFIX=rc2 scripts/run_stage_skills.sh $STEPS pick place || { log "training failed"; exit 1; }
log "evaluating"
export CK_PICK=pick9rc2_$STEPS CK_PLACE=place9rc2_$STEPS
EVAL_OUT=outputs/eval_rc2 scripts/eval.sh still push
EVAL_EXTRA="--place_release hold" EVAL_OUT=outputs/eval_rc2_hold scripts/eval.sh still
log "done"
