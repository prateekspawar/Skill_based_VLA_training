#!/usr/bin/env bash
# Stage-wise progress of the training pipeline (stage 1: static+moving model, stage 2: per-skill models).
cd "$(dirname "$0")/.."
pct() { awk -v a="$1" -v b="$2" 'BEGIN { if (b > 0) printf "%3.0f%%", 100 * a / b; else printf "  0%%" }'; }
last_num() { [ -f "$1" ] && tr '\r' '\n' < "$1" | grep -oE "$2" | tail -1 | grep -oE '[0-9]+' | head -1; }
train_step() {  # current training step from a lerobot-train log ("12/20000" progress bar)
  [ -f "$1" ] && tr '\r' '\n' < "$1" | grep -aoE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f1
}
eval_done() { [ -f "$1" ] && grep -cE "\[eval\] +[0-9]+ " "$1" || echo 0; }

echo "=== Stage 1: static + moving targets, one model ==="
g=$(last_num outputs/gen_moving_1000.log 'saved [0-9]+/1000'); g=${g:-0}
grep -q "\[gen\] done" outputs/gen_moving_1000.log 2>/dev/null && g=1000
printf "  1a record moving demos   %s  (%s/1000 episodes)\n" "$(pct $g 1000)" "$g"
c=$(last_num outputs/stage_moving/convert.log '\[convert\] [0-9]+'); c=${c:-0}
grep -q "\[convert\] wrote" outputs/stage_moving/convert.log 2>/dev/null && c=1000
printf "  1b convert               %s  (%s/1000 episodes)\n" "$(pct $c 1000)" "$c"
m=0; [ -f datasets/lerobot/s512_mix_target/meta/info.json ] && m=1
printf "  1c merge + targets       %s\n" "$(pct $m 1)"
t=$(train_step outputs/stage_moving/train.log); t=${t:-0}
if [ -f outputs/stage_moving/train_resume.log ]; then  # resumed from the step-10000 checkpoint
  r=$(train_step outputs/stage_moving/train_resume.log); t=$((10000 + ${r:-0}))
fi
printf "  1d train (20k steps)     %s  (step %s/20000)\n" "$(pct $t 20000)" "$t"
es=$(eval_done outputs/stage_moving/eval_still.log); ep=$(eval_done outputs/stage_moving/eval_push.log)
printf "  1e evaluate              %s  (stationary %s/50, push %s/50)\n" "$(pct $((es + ep)) 100)" "$es" "$ep"
# weights ~ hours: record 1.1, convert 1.0, merge/targets 0.05, train 2.8, eval 0.4
awk -v g=$g -v c=$c -v m=$m -v t=$t -v e=$((es + ep)) 'BEGIN {
  done = 1.1*g/1000 + 1.0*c/1000 + 0.05*m + 2.8*t/20000 + 0.4*e/100; total = 5.35
  printf "  Stage 1 overall          %3.0f%%  (~%.1f h left)\n", 100*done/total, total-done }'
grep -E "\[stage\] (still|push):" outputs/stage_moving_driver.log 2>/dev/null | sed 's/^/  result: /'

echo "=== Stage 2: separate pick / place models ==="
if [ ! -d outputs/stage_per_skill ]; then
  echo "  not started (runs only if stage 1 beats the current model)"
else
  for sk in pick place; do
    cv=$(last_num outputs/stage_per_skill/convert_$sk.log '\[convert\] [0-9]+'); cv=${cv:-0}
    grep -q "\[convert\] wrote" outputs/stage_per_skill/convert_$sk.log 2>/dev/null && cv=2000
    printf "  2a convert %-5s          %s  (%s/2000 episodes)\n" "$sk" "$(pct $cv 2000)" "$cv"
  done
  for sk in pick place; do
    ts=$(train_step outputs/stage_per_skill/train_$sk.log); ts=${ts:-0}
    tot=$(tr '\r' '\n' < outputs/stage_per_skill/train_$sk.log 2>/dev/null | grep -oE '[0-9]+/[0-9]+ \[' | tail -1 | cut -d/ -f2 | cut -d' ' -f1)
    printf "  2b train %-5s            %s  (step %s/%s)\n" "$sk" "$(pct ${ts:-0} ${tot:-1})" "${ts:-0}" "${tot:-?}"
  done
  es=$(eval_done outputs/stage_per_skill/eval_still.log); ep=$(eval_done outputs/stage_per_skill/eval_push.log)
  printf "  2c evaluate              %s  (stationary %s/50, push %s/50)\n" "$(pct $((es + ep)) 100)" "$es" "$ep"
  grep -E "\[skills\] (still|push):" outputs/stage_per_skill_driver.log 2>/dev/null | sed 's/^/  result: /'
fi

echo "=== Depth comparison (D1 depth image as camera3, D2 native point-map tokens) ==="
if [ ! -d outputs/stage_depth ]; then
  echo "  not started"
else
  D=outputs/stage_depth
  for part in static moving; do
    cv=$(last_num $D/convert_$part.log '\[convert\] [0-9]+'); cv=${cv:-0}
    grep -q "\[convert\] wrote" $D/convert_$part.log 2>/dev/null && cv=1000
    printf "  convert %-7s           %s  (%s/1000 episodes)\n" "$part" "$(pct $cv 1000)" "$cv"
  done
  mt=0; [ -f datasets/lerobot/s512_mix_depth_target/meta/info.json ] && mt=1
  printf "  merge + targets          %s\n" "$(pct $mt 1)"
  for v in d1_depthcam_20000 d2_pointmap_20000; do
    ts=$(train_step $D/train_$v.log); ts=${ts:-0}
    printf "  train %-18s %s  (step %s/20000)\n" "$v" "$(pct $ts 20000)" "$ts"
  done
  for v in d1_depthcam_20000 d2_pointmap_20000; do
    es=$(eval_done $D/$v/eval_still.log); ep=$(eval_done $D/$v/eval_push.log)
    printf "  eval  %-18s %s  (stationary %s/50, push %s/50)\n" "$v" "$(pct $((es + ep)) 100)" "$es" "$ep"
  done
  grep -E "\[depth\] d[12]_.*(still|push):" outputs/stage_depth_driver.log 2>/dev/null | grep -E "pick success" | sed 's/^/  result: /'
  echo "  baseline (no depth): stationary 64% pick / 44% full, push 50% / 34%"
fi

echo "=== Separate skill models (pick / place / find, shared backbone, 9-D targets) ==="
if [ ! -d outputs/stage_skills ]; then
  echo "  not started"
else
  for name in pick9_15000 place9_15000 find9_8000; do
    lg=outputs/stage_skills/train_$name.log
    ts=$(train_step $lg); ts=${ts:-0}
    tot=${name##*_}
    [ -f "$lg" ] && printf "  train %-14s %s  (step %s/%s)\n" "$name" "$(pct $ts $tot)" "$ts" "$tot" || printf "  train %-14s   0%%  (not started)\n" "$name"
  done
  fg=$(last_num outputs/gen_find_1000.log 'saved [0-9]+/1000'); fg=${fg:-0}
  grep -q "\[find\] done" outputs/gen_find_1000.log 2>/dev/null && fg=1000
  printf "  find demos               %s  (%s/1000 episodes)\n" "$(pct $fg 1000)" "$fg"
  for v in direct_still direct_push find_still find_push; do
    es=$(eval_done outputs/stage_skills/eval_$v.log)
    printf "  eval %-19s %s  (%s/50)\n" "$v" "$(pct $es 50)" "$es"
  done
fi
