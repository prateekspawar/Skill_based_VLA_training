"""SAM2 tracking accuracy on recorded episodes (stationary objects until grasped) vs true positions (~/lerobot-env).

Starts tracking each episode's pick cube and place plate on frame 0, then updates every ``--every`` frames
while the arm moves (occlusions), comparing the tracked points against the recorded initial positions while
the objects are still in place (cube: until the pick skill ends).

    ~/lerobot-env/bin/python scripts/eval_sam2_offline.py --raw datasets/raw/s512_test.h5 --episodes 10
"""

import argparse
import json
import time

import h5py
import numpy as np

from so101_lang import language
from so101_lang.data.add_target_point import env_to_head
from so101_lang.perception.locate import TargetLocator
from so101_lang.perception.sam2_tracker import Sam2PointTracker

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--raw", required=True)
ap.add_argument("--episodes", type=int, default=10)
ap.add_argument("--every", type=int, default=10, help="Frames between updates (10 = every policy query at 20 Hz).")
args = ap.parse_args()

with h5py.File(args.raw, "r") as h5:
    calib = json.loads(h5.attrs["calibration"])
    tracker = Sam2PointTracker(TargetLocator(calib))
    hib = np.array(calib["head_in_base"])
    err = {"pick": [], "place": []}
    first = {"pick": [], "place": []}
    held, total, times = 0, 0, []
    for name in sorted(h5.keys())[: args.episodes]:
        ep = h5[name]
        poses = ep["init_poses"][()]
        cube_i, plate_i = int(ep.attrs["cube_idx"]), int(ep.attrs["plate_idx"])
        truth = {"pick": env_to_head(poses[cube_i : cube_i + 1, :3], hib)[0],
                 "place": env_to_head(poses[language.NUM_CUBES + plate_i : language.NUM_CUBES + plate_i + 1, :3], hib)[0]}
        q = {"pick": language.cube_ref(cube_i), "place": language.plate_ref(plate_i)}
        skill = ep["skill"][()]
        left, right = ep["left_rgb"], ep["right_rgb"]
        pts, info = tracker.start(left[0], right[0], [q["pick"], q["place"]])
        for sk in err:
            first[sk].append(1000 * np.linalg.norm(pts[q[sk]] - truth[sk]))
        pick_end = int(np.argmax(skill == 1)) if (skill == 1).any() else len(skill)
        ep_err = {"pick": [], "place": []}
        for t in range(args.every, len(skill), args.every):
            t0 = time.perf_counter()
            # Freeze the pick target once the gripper is within 5 cm of it (as in closed loop).
            near = np.linalg.norm(ep["state"][t][:3] - pts[q["pick"]]) < 0.05
            pts, info = tracker.update(left[t], right[t], frozen=(q["pick"],) if near else ())
            times.append(time.perf_counter() - t0)
            for sk in err:
                if sk == "pick" and t >= pick_end - 10:  # cube about to be grasped / moving with the gripper
                    continue
                err[sk].append(1000 * np.linalg.norm(pts[q[sk]] - truth[sk]))
                ep_err[sk].append(err[sk][-1])
                total += 1
                held += info[q[sk]]["status"] == "held"
        print(f"  {name} {q['pick']:11s} start {first['pick'][-1]:5.0f} max {max(ep_err['pick'], default=0):5.0f} | "
              f"{q['place']:11s} start {first['place'][-1]:5.0f} max {max(ep_err['place'], default=0):5.0f} mm")
for sk in err:
    e = np.array(err[sk])
    print(f"{sk:5s} target: start median {np.median(first[sk]):5.1f} mm | while tracking median {np.median(e):5.1f} mm, "
          f"p90 {np.percentile(e, 90):5.1f} mm, max {e.max():6.1f} mm [{len(e)} updates]")
print(f"held (occluded) updates: {held}/{total}; mean update time {1000 * np.mean(times):.0f} ms")
