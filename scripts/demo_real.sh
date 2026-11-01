#!/usr/bin/env bash
# Real-robot demo window (scripts/demo_real.py): instruction -> LLM plan -> perception checks the goals on the table,
# optionally runs pick + place on the arm ("Execute on the arm").
#   scripts/demo_real.sh [--objects "red cube" "blue cube" "yellow plate"] [--execute]
# Needs arm_host.py / stereo_host.py on the Pi and `ollama serve`. Starts the policy server (perception + skills,
# real head offset and calibration) unless one is already listening on :6000. Log: outputs/real/demo/server.log.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/real/demo
mkdir -p $OUT
curl -s localhost:11434/api/version > /dev/null || { echo "[demo] start the planner first: ollama serve"; exit 1; }
server=
if ! ss -ltn | grep -q ':6000 '; then
  [[ -f outputs/real/stereo_calibration_real.json ]] || ~/lerobot-env/bin/python scripts/run_real.py --write-calib
  echo "[demo] starting the policy server (perception + skills)..."
  SO101_CAM_OFFSET=0.0351,-0.0229,-0.0024,-1.99,-1.44,1.76 ~/lerobot-env/bin/python -m so101_lang.serve.policy_server \
    --calibration outputs/real/stereo_calibration_real.json > $OUT/server.log 2>&1 &
  server=$!
  trap 'kill $server 2>/dev/null' EXIT
  until grep -q listening $OUT/server.log; do
    kill -0 $server 2>/dev/null || { echo "[demo] policy server failed:"; tail -5 $OUT/server.log; exit 1; }
    sleep 3
  done
fi
~/lerobot-env/bin/python scripts/demo_real.py "$@"
