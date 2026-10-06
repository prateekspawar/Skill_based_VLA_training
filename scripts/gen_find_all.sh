#!/usr/bin/env bash
# find() data: 1000 successful + 200 target-absent (failure) episodes.
set -e
cd "$(dirname "$0")/.."
export OMNI_KIT_ACCEPT_EULA=YES
~/isaacsim/bin/python scripts/gen_find_demos.py --num_envs 16 --episodes 1000 --seed 11 --out datasets/raw/find_1000.h5
~/isaacsim/bin/python scripts/gen_find_demos.py --num_envs 16 --episodes 200 --absent_prob 1.0 --seed 12 --out datasets/raw/find_absent_200.h5
