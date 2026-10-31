"""Collect real-camera pixels of known-colour objects for the colour model (``scripts/fit_color_model.py --real``).

    ~/lerobot-env/bin/python scripts/collect_real_colors.py --objects "white cube" "yellow plate" [--out ...]

Needs ``arm_host.py`` / ``stereo_host.py`` on the Pi and the policy server (for OWLv2). The arm visits a few side
poses that keep it out of the workspace view; at each, both cameras' rectified 512 x 512 images (as the models see
them; the left image, with depth from the pair) are searched for every object with the server's ranked detection, and the pixels of the box's inner part
that are brighter than the black mat (V > 60) are kept as that object's colour samples; a ring of mat around the
objects gives "table" samples. One entry per object per frame, so the fit can hold out whole views.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_handeye as ch  # noqa: E402
import run_real as rr  # noqa: E402

# Side poses (degrees, readings): the arm swung away from the workspace at different heights, so the lit
# faces and the arm's shadow differ between views.
VIEWS = [dict(shoulder_pan=p, shoulder_lift=l, elbow_flex=e, wrist_flex=20.0, wrist_roll=0.0)
         for p, l, e in ((60, -10, -20), (75, -10, -20), (60, -30, 10), (45, -35, 20), (70, 0, -40))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", default=["white cube", "yellow plate"])
    ap.add_argument("--out", default=str(rr.ROOT / "outputs/real/color_samples.npz"))
    ap.add_argument("--frames", type=int, default=2, help="pairs per view")
    args = ap.parse_args()
    train = json.loads(rr.TRAIN_CALIB.read_text())
    st = json.loads((rr.ROOT / "so101_lang/real/stereo_calib.json").read_text())
    rect = rr.Rectifier(st, train["intrinsics"])
    arm, cams = ch.Arm(), ch.Cams()
    conn = Client(("localhost", 6000), authkey=b"so101_lang")
    samples, debug = {}, []
    try:
        for k, q in enumerate(VIEWS):
            cur = {j: arm.call({"cmd": "status"})["pos"][j] for j in rr.ARM + ("gripper",)}
            tgt = {**q, "gripper": cur["gripper"]}
            mid = {**dict.fromkeys(rr.ARM, 0.0), "gripper": cur["gripper"]}
            if ch.safe_path(cur, tgt, None):
                legs = [tgt]
            elif ch.safe_path(cur, mid, None) and ch.safe_path(mid, tgt, None):
                legs = [mid, tgt]
            else:
                print(f"view {k}: no table-safe path, skipped", flush=True)
                continue
            for leg in legs:
                arm.move(leg, 15)
            time.sleep(0.8)
            for f in range(args.frames):
                L, R, _ = cams.get()
                img, right = rect(L, R)
                if True:
                    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    boxes = []
                    for obj in args.objects:
                        conn.send({"cmd": "rank_debug", "left": img, "right": right, "query": obj,
                                   "true_point": [0.5, 0.0, 0.0]})  # only the winner's box is used
                        x0, y0, x1, y1 = conn.recv()["winner"]["box"]
                        boxes.append((x0, y0, x1, y1))
                        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, 0.3 * (x1 - x0), 0.3 * (y1 - y0)
                        crop = img[int(cy - hh):int(cy + hh) + 1, int(cx - hw):int(cx + hw) + 1].reshape(-1, 3)
                        v = cv2.cvtColor(crop.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)[:, 2]
                        name, kind = obj.split()
                        samples.setdefault(f"{name}|{kind}_real", []).append(crop[v > 60])
                        cv2.rectangle(vis, (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh)), (0, 255, 0), 2)
                    # mat: pixels in the objects' neighbourhood that are outside every box
                    xs0, ys0 = min(b[0] for b in boxes) - 40, min(b[1] for b in boxes) - 40
                    xs1, ys1 = max(b[2] for b in boxes) + 40, max(b[3] for b in boxes) + 40
                    m = np.zeros(img.shape[:2], bool)
                    m[max(int(ys0), 0):int(ys1), max(int(xs0), 0):int(xs1)] = True
                    for x0, y0, x1, y1 in boxes:
                        m[int(y0) - 5:int(y1) + 5, int(x0) - 5:int(x1) + 5] = False
                    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                    samples.setdefault("table|table_real", []).append(img[m & (hsv[..., 2] < 90)])
                    if f == 0:
                        debug.append(cv2.resize(vis, (256, 256)))
            print(f"view {k}: {[(key, len(v[-1])) for key, v in samples.items()]}", flush=True)
    finally:
        cur = {j: arm.call({"cmd": "status"})["pos"][j] for j in rr.ARM + ("gripper",)}
        robot = rr.Robot(json.loads((rr.ROOT / "so101_lang/real/handeye.json").read_text()),
                         np.array(train["head_in_base"], float))
        home = {**robot.reading_q(rr.HOME), "gripper": cur["gripper"]}
        if ch.safe_path(cur, home, None):
            arm.move(home, 15)
        arm.close()
    np.savez_compressed(args.out, **{f"{k}#{i}": p for k, v in samples.items() for i, p in enumerate(v)})
    cv2.imwrite(str(Path(args.out).with_suffix(".jpg")), np.vstack([np.hstack(debug[i:i + 4]) for i in range(0, len(debug) - 3, 4)]))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
