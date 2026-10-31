#!/usr/bin/env bash
# Interactive demo: type an instruction, watch the arm do it in the Isaac Sim window.
#   scripts/demo.sh            (needs `ollama serve`; the planner LLM runs on the CPU, ~10 s per plan)
# Each round shows a new random scene (4 cubes, 2 plates) and prints its objects. Type e.g.
#   "put the red cube on the pink plate"   or   "put the blue cube on the orange plate and the green one on the brown plate"
# Empty line = new scene, q = quit. Goals are checked by perception after every step and retried up to 2 times.
# Videos of each round: outputs/demo/demo.mp4, demo_ep<N>.mp4. Logs: outputs/demo/server.log, sim.log.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/demo
mkdir -p $OUT
ck() { echo checkpoints/$1/checkpoints/last/pretrained_model; }
curl -s localhost:11434/api/version > /dev/null || { echo "[demo] start the planner first: ollama serve"; exit 1; }
rm -f $OUT/sim.log $OUT/server.log; touch $OUT/server.log
~/lerobot-env/bin/python -m so101_lang.serve.policy_server --skills pick=$(ck pick9rc2_15000) \
  place=$(ck place9rc2_15000) find=$(ck find9rc_15000) stack=$(ck stack9_place300_5000) > $OUT/server.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
echo "[demo] loading the skill models..."
until grep -q listening $OUT/server.log; do
  kill -0 $server 2>/dev/null || { echo "[demo] policy server failed:"; tail -5 $OUT/server.log; exit 1; }
  sleep 3
done
echo "[demo] starting the simulator (the window opens in ~1 min)..."
SO101_SHOW_HEAD=1 SO101_NO_CAMERA_MOUNT=1 OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim/bin/python scripts/eval_policy.py --interactive --planner_cpu --replan \
  --check perception --episodes 1000 --seed ${SEED:-$RANDOM} --viz kit --video_out $OUT/demo.mp4 "$@" > $OUT/sim.log 2>&1
echo "[demo] finished (simulator log: $OUT/sim.log)"
