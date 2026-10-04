"""Accuracy of TargetLocator (OWLv2 + stereo depth) against true object positions in recorded frames (~/lerobot-env).

    ~/lerobot-env/bin/python scripts/eval_locator.py --raw datasets/raw/s512_test.h5 --episodes 10
"""

import argparse
import json
import time

import h5py
import numpy as np

from so101_lang import language
from so101_lang.data.add_target_point import env_to_head
from so101_lang.perception.locate import TargetLocator

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--raw", required=True)
ap.add_argument("--episodes", type=int, default=10)
args = ap.parse_args()

with h5py.File(args.raw, "r") as h5:
    calib = json.loads(h5.attrs["calibration"])
    locator = TargetLocator(calib)
    head_in_base = np.array(calib["head_in_base"])
    names = language.OBJECT_REFS + language.TARGET_REFS
    errs = {n: [] for n in names}
    wrong, times = 0, []
    for ep_name in sorted(h5.keys())[: args.episodes]:
        ep = h5[ep_name]
        left, right = ep["left_rgb"][0], ep["right_rgb"][0]
        truth = env_to_head(ep["init_poses"][()][:, :3], head_in_base)
        for i, name in enumerate(names):
            t0 = time.perf_counter()
            p, info = locator.locate(left, right, name)
            times.append(time.perf_counter() - t0)
            e = np.linalg.norm(p - truth[i])
            errs[name].append(e)
            # Wrong object if another object's true position is closer than the named one.
            if np.argmin(np.linalg.norm(truth - p, axis=1)) != i:
                wrong += 1
total = sum(len(v) for v in errs.values())
for n, v in errs.items():
    v = 1000 * np.array(v)
    print(f"{n:12s}: median {np.median(v):5.1f} mm, max {v.max():6.1f} mm")
print(f"wrong object: {wrong}/{total}; mean locate time {1000 * np.mean(times[1:]):.0f} ms")
