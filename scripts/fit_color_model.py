"""Fit the learned colour model used by ranked detection (so101_lang/perception/color_model.json).

Rendered pixels with free labels: each pick / place frame's target pose (state dims 10:13) is projected into the
left image, the inner disc of the cube / plate gives that colour name's pixels, a ring just outside the plate rim
gives table pixels. One Gaussian in CIELab per (name, kind) and for the table; 75 % of episodes fit, 25 % test.
``--real`` adds real-camera samples (``scripts/collect_real_colors.py``: one entry per object per frame) as
components "<name>|<kind>_real"; frames whose median colour is far from the majority of that object's frames (a
wrong detection) are dropped, the last third of the frames is held out.

The test reports the own-colour score ``verify.color_own_fit`` (no other names involved) of held-out objects of
that colour, and the false positives it must not produce: the same score for every other colour's objects and for
the table, per threshold. Run in ~/lerobot-env:

    ~/lerobot-env/bin/python scripts/fit_color_model.py [--per_name 80] [--real outputs/real/color_samples.npz]
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
parser.add_argument("--pick", default="datasets/lerobot/s512_pick9rc2")
parser.add_argument("--place", default="datasets/lerobot/s512_place9rc2")
parser.add_argument("--real", default=None, help="real samples (.npz from scripts/collect_real_colors.py)")
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

real_keys = set()
if args.real:
    z = np.load(args.real)
    frames = collections.defaultdict(list)
    for k in sorted(z.files, key=lambda k: (k.split("#")[0], int(k.split("#")[1]))):
        frames[k.split("#")[0]].append(z[k])
    for key, eps in frames.items():
        med = np.array([np.median(lab(p), 0) for p in eps])
        ok = np.linalg.norm(med - np.median(med, 0), axis=1) < 20
        print(f"[color] real {key}: {ok.sum()} / {len(eps)} frames kept (dropped: median colour > 20 Lab off)")
        samples[key] = [p for p, o in zip(eps, ok) if o]
        real_keys.add(key)

components, test = {}, {}
for key, eps in samples.items():
    n = int(len(eps) * (0.67 if key in real_keys else 0.75))
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
names = sorted({k.split("|")[0] for k in components})
score = {key: {n: np.array([verify.color_own_fit(p, n) for p in eps]) for n in names}
         for key, eps in test.items()}  # held-out sample -> own-colour score for every name
post = {key: {n: np.array([verify.color_posterior(p)[n] for p in eps]) for n in names} for key, eps in test.items()}
gates = (0.05, 0.08, 0.10, 0.15, 0.20, 0.25)
print("[color] true objects: own-colour score of held-out samples of that colour (accepted at each threshold)")
for key in sorted(test):
    name = key.split("|")[0]
    if name == "table":
        continue
    own = score[key][name]
    print(f"  {key:18} n {len(own):3}  median {np.median(own):.2f}  " + "  ".join(f">{g}: {np.mean(own > g):4.0%}" for g in gates))
print("[color] false positives: share of OTHER-colour objects / table samples scoring above the threshold, per name")
for n in names:
    if n == "table":
        continue
    others = np.concatenate([score[k][n] for k in test if k.split("|")[0] not in (n, "table")])
    table = np.concatenate([score[k][n] for k in test if k.startswith("table")])
    worst = max(((np.median(score[k][n]), k) for k in test if k.split("|")[0] not in (n, "table")))
    print(f"  {n:7} others " + " ".join(f"{np.mean(others > g):4.0%}" for g in gates) + "   table " +
          " ".join(f"{np.mean(table > g):4.0%}" for g in gates) + f"   closest other: {worst[1]} median {worst[0]:.2f}")
print("[color] ranking: share of (requested-colour object, other-colour object) pairs where the requested one scores "
      "higher for its name (what decides detection when both are in view)")
for n in names:
    if n == "table":
        continue
    own = np.concatenate([score[k][n] for k in test if k.split("|")[0] == n])
    others = {k: score[k][n] for k in test if k.split("|")[0] not in (n, "table")}
    auc = {k: float(np.mean(own[:, None] > o[None, :])) for k, o in others.items()}
    worst = sorted(auc.items(), key=lambda kv: kv[1])[:2]
    own_p = np.concatenate([post[k][n] for k in test if k.split("|")[0] == n])
    auc_p = {k: float(np.mean(own_p[:, None] > post[k][n][None, :])) for k in others}
    print(f"  {n:7} all others {np.mean(list(auc.values())):.1%}   hardest: " + ", ".join(f"{k} {v:.0%}" for k, v in worst)
          + f"   | relative posterior: all {np.mean(list(auc_p.values())):.1%}, hardest "
          + ", ".join(f"{k} {v:.0%}" for k, v in sorted(auc_p.items(), key=lambda kv: kv[1])[:2]))
