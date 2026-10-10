"""Fit the learned colour model used by ranked detection (so101_lang/perception/color_model.json).

Rendered pixels with free labels: each pick / place frame's target pose (state dims 10:13) is projected into the
left image, the inner disc of the cube / plate gives that colour name's pixels, a ring just outside the plate rim
gives table pixels. One Gaussian in CIELab per (name, kind) and for the table; 75 % of episodes fit, 25 % test.
Run in ~/lerobot-env:

    ~/lerobot-env/bin/python scripts/fit_color_model.py [--per_name 80] [--out so101_lang/perception/color_model.json]
"""

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pick", default="datasets/lerobot/s512_pick9rc")
parser.add_argument("--place", default="datasets/lerobot/s512_place9rc")
parser.add_argument("--per_name", type=int, default=80, help="Episodes per colour name and kind.")
parser.add_argument("--out", default="so101_lang/perception/color_model.json")
args = parser.parse_args()
OUTLIER_M2 = 16.3  # chi-square(3) 99.9 %


def lab(px):
    return cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)


samples = collections.defaultdict(list)  # "name|kind" -> [pixels per episode]
for root, kind, radius, inner, n_names in ((args.pick, "cube", 0.015, 0.7, 4), (args.place, "plate", 0.045, 0.6, 9)):
    ds = LeRobotDataset("local/color", root=root)
    cal = json.load(open(f"{root}/meta/stereo_calibration.json"))
    k, off = cal["intrinsics"], np.asarray(cal["cameras"]["left"]["position_in_head"])
    ep_from, cnt = ds.meta.episodes["dataset_from_index"], collections.Counter()
    for e in np.random.default_rng(1).permutation(len(ep_from)):
        item = ds[int(ep_from[e])]
        m = re.search(rf"(\w+) {kind}", item["task"])
        if not m or cnt[m.group(1)] >= args.per_name:
            continue
        img = (item["observation.images.left"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        q = item["observation.state"].numpy()[10:13] - off
        u, v = k["cx"] + k["fx"] * (-q[1]) / q[0], k["cy"] + k["fy"] * (-q[2]) / q[0]
        r = k["fx"] * radius / q[0]
        yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
        d2 = (xx - u) ** 2 + (yy - v) ** 2
        obj = img[d2 < (inner * r) ** 2]
        if len(obj) < 15:
            continue
        cnt[m.group(1)] += 1
        samples[f"{m.group(1)}|{kind}"].append(obj)
        if kind == "plate":
            samples["table|table"].append(img[(d2 > (1.4 * r) ** 2) & (d2 < (1.8 * r) ** 2)])
        if len(cnt) >= n_names and all(c >= args.per_name for c in cnt.values()):
            break

components, test = {}, {}
for key, eps in samples.items():
    n = int(len(eps) * 0.75)
    x = np.concatenate([lab(p) for p in eps[:n]])
    components[key] = {"mean": x.mean(0).round(3).tolist(), "cov": (np.cov(x.T) + np.eye(3) * 4.0).round(3).tolist()}
    test[key] = eps[n:]
Path(args.out).write_text(json.dumps({"space": "cv2_lab", "outlier_m2": OUTLIER_M2, "components": components}, indent=1))
print(f"[color] {len(components)} components -> {args.out}")

import os  # noqa: E402

os.environ["SO101_COLOR_MODEL"] = "learned"
from so101_lang.perception import verify  # noqa: E402

verify.COLOR_MODEL_PATH = Path(args.out)
verify._color_model.cache_clear()
gate = verify.COLOR_GATE["learned"]
for key, eps in sorted(test.items()):
    name = key.split("|")[0]
    if name == "table":
        continue
    post = [verify.color_posterior(p) for p in eps]
    own = np.array([p[name] for p in post])
    best = np.mean([max((v, c) for c, v in p.items() if c != "table")[1] == name for p in post])
    print(f"[color] {key:14} held-out {len(eps):3}: own score median {np.median(own):.2f}, "
          f"accepted (> {gate}) {np.mean(own > gate):.0%}, own name highest {best:.0%}")
