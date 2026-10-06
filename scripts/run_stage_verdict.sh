#!/usr/bin/env bash
# Skill models with a pass/fail verdict (12-D action = 10 + [done, success]):
# wait for the raw failure / find data, convert, merge with the all-success data, per-skill subsets, train
# pick, place and find on one shared frozen backbone. Usage: scripts/run_stage_verdict.sh [steps]
set -uo pipefail
cd "$(dirname "$0")/.."
STEPS=${1:-15000}
OUT=outputs/stage_verdict
L=datasets/lerobot
R=datasets/raw
PY=~/lerobot-env/bin/python
mkdir -p $OUT
log() { echo "[verdict] $(date +%T) $*"; }

log "waiting for raw data"
until grep -q "\[fail\] done" logs/gen_fail_400.log 2>/dev/null; do sleep 60; done
log "converting failure episodes (depth video + point map + status)"
$PY -m so101_lang.data.hdf5_to_lerobot --inputs $R/s512_fail_400.h5 --root $L/s512_fail_depth \
  --depth fast_foundationstereo --pointmap fast_foundationstereo --status --overwrite > $OUT/convert_fail.log 2>&1 \
  || { log "convert fail failed"; exit 1; }
cp $L/s512_mix_depth_target9s/meta/stereo_calibration.json $L/s512_fail_depth/meta/
$PY -m so101_lang.data.build_targets --src $L/s512_fail_depth --raw $R/s512_fail_400.h5 --rot6d \
  --dst $L/s512_fail_target9s > $OUT/targets_fail.log 2>&1 || { log "targets failed"; exit 1; }
log "merging with the all-success data"
rm -rf $L/s512_all9s_nopm $L/s512_all9s $L/_m9s_nopm $L/_f9s_nopm
$PY -m so101_lang.data.pointmap_merge strip $L/s512_mix_depth_target9s $L/_m9s_nopm > $OUT/pointmap.log 2>&1 || { log "strip failed"; exit 1; }
$PY -m so101_lang.data.pointmap_merge strip $L/s512_fail_target9s $L/_f9s_nopm >> $OUT/pointmap.log 2>&1 || { log "strip failed"; exit 1; }
~/lerobot-env/bin/lerobot-edit-dataset --operation.type=merge \
  --operation.repo_ids='["local/so101_stereo_ee", "local/so101_fail"]' \
  --operation.roots="[\"$L/_m9s_nopm\", \"$L/_f9s_nopm\"]" \
  --new_repo_id=local/so101_stereo_ee --new_root=$L/s512_all9s_nopm > $OUT/merge.log 2>&1 || { log "merge failed"; exit 1; }
$PY -m so101_lang.data.pointmap_merge attach $L/s512_all9s_nopm $L/s512_all9s \
  $L/s512_mix_depth_target9s $L/s512_fail_target9s >> $OUT/pointmap.log 2>&1 || { log "attach failed"; exit 1; }
cp $L/s512_mix_depth_target9s/meta/stereo_calibration.json $L/s512_all9s/meta/
for sk in pick place; do
  $PY -m so101_lang.data.skill_subset --src $L/s512_all9s --dst $L/s512_${sk}9s --skill $sk > $OUT/subset_$sk.log 2>&1 \
    || { log "subset $sk failed"; exit 1; }
  tail -1 $OUT/subset_$sk.log | sed "s/^/[verdict] /"
done

until grep -q "\[find\] done.*absent" logs/gen_find_all.log 2>/dev/null; do sleep 60; done
log "converting find episodes (point map + status, state padded to 19-D)"
$PY -m so101_lang.data.hdf5_to_lerobot --inputs $R/find_1000.h5 $R/find_absent_200.h5 --root $L/s512_find9s \
  --depth none --pointmap fast_foundationstereo --status --pad_state 9 --overwrite > $OUT/convert_find.log 2>&1 \
  || { log "convert find failed"; exit 1; }

log "training pick, place, find ($STEPS steps each)"
DS_SUFFIX=s scripts/run_stage_skills.sh $STEPS pick place find
log "done"
