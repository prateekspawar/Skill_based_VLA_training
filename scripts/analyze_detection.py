"""Why does the first detection fail? OWLv2 top-K analysis on recorded first frames (~/lerobot-env).

For every object in the first frame of each episode: is the top box correct (its 3-D point within 3 cm of a
cube's / 5 cm of a plate's true centre), where does the correct box rank among the top K, and is the object
occluded (stereo depth at its projected centre much closer than its true distance -> something is in front)?

    ~/lerobot-env/bin/python scripts/analyze_detection.py --raw datasets/raw/s512_dart_1000.h5 --episodes 100
"""

import argparse
import collections
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from so101_lang import language
from so101_lang.data.add_target_point import env_to_head
from so101_lang.perception.locate import TargetLocator, detector_prompt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--raw", required=True)
ap.add_argument("--episodes", type=int, default=100)
ap.add_argument("--topk", type=int, default=5)
ap.add_argument("--out", default="outputs/detection")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def project(p_head, calib):
    """Head-frame point -> left-image pixel (u, v) and its optical depth."""
    off = np.asarray(calib["cameras"]["left"]["position_in_head"])
    k = calib["intrinsics"]
    q = p_head - off  # head axes: x fwd, y left, z up -> optical: x right = -y, y down = -z, z fwd = x
    z = q[0]
    return k["cx"] + k["fx"] * (-q[1]) / z, k["cy"] + k["fy"] * (-q[2]) / z, z


@torch.no_grad()
def topk_boxes(loc, rgb, query, k):
    inputs = loc.processor(text=[[detector_prompt(query)]], images=rgb, return_tensors="pt").to(loc.device)
    out = loc.model(**inputs)
    size = max(rgb.shape[:2])
    res = loc.processor.post_process_grounded_object_detection(
        out, threshold=0.0, target_sizes=torch.tensor([[size, size]], device=loc.device))[0]
    order = torch.argsort(res["scores"], descending=True)[:k]
    return res["boxes"][order].cpu().numpy(), res["scores"][order].cpu().numpy()


with h5py.File(args.raw, "r") as h5:
    calib = json.loads(h5.attrs["calibration"])
    loc = TargetLocator(calib)
    hib = np.array(calib["head_in_base"])
    names = language.OBJECT_REFS + language.TARGET_REFS
    stats = collections.Counter()
    by_obj = collections.defaultdict(collections.Counter)
    saved = 0
    for ep_name in sorted(h5.keys())[: args.episodes]:
        ep = h5[ep_name]
        left, right = ep["left_rgb"][0], ep["right_rgb"][0]
        depth = loc.depth.depth(left, right)
        truth = env_to_head(ep["init_poses"][()][:, :3], hib)
        for i, q in enumerate(names):
            tol = 0.03 if "cube" in q else 0.05
            u, v, zt = project(truth[i], calib)
            ui, vi = int(np.clip(u, 0, 511)), int(np.clip(v, 0, 511))
            zm = depth[vi, ui]
            in_view = 0 <= u < 512 and 0 <= v < 512
            occluded = in_view and np.isfinite(zm) and zm < zt - 0.04
            boxes, scores = topk_boxes(loc, left, q, args.topk)
            correct_rank = None
            for r, b in enumerate(boxes):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                hw, hh = (b[2] - b[0]) / 4 + 1, (b[3] - b[1]) / 4 + 1
                mask = np.zeros(depth.shape, bool)
                mask[int(max(cy - hh, 0)):int(cy + hh) + 1, int(max(cx - hw, 0)):int(cx + hw) + 1] = True
                try:
                    p, _ = loc.point_from_mask(depth, mask, q)
                except RuntimeError:
                    continue
                if np.linalg.norm(p - truth[i]) < tol:
                    correct_rank = r
                    break
            cat = ("ok" if correct_rank == 0 else
                   "out_of_view" if not in_view else
                   "occluded" if occluded else
                   f"correct_at_rank_{correct_rank}" if correct_rank is not None else "not_in_topk")
            stats[cat] += 1
            by_obj[q][cat] += 1
            if cat != "ok" and saved < 24:
                im = Image.fromarray(left.copy()); d = ImageDraw.Draw(im)
                for r, b in enumerate(boxes[:3]):
                    d.rectangle(b.tolist(), outline=(255, 0, 0) if r == 0 else (255, 200, 0), width=3 if r == 0 else 1)
                d.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(0, 255, 0), width=3)
                d.rectangle([0, 0, 511, 18], fill=(0, 0, 0)); d.text((4, 3), f"{q}: {cat}  score {scores[0]:.2f}", fill=(255, 255, 255))
                im.save(f"{args.out}/{ep_name}_{q.replace(' ', '_')}.png")
                saved += 1

total = sum(stats.values())
print(f"[detect] {total} detections over {args.episodes} episodes")
for k, v in stats.most_common():
    print(f"  {k:22s} {v:4d}  ({100 * v / total:.1f}%)")
print("  per object (non-ok):")
for q in names:
    bad = {k: v for k, v in by_obj[q].items() if k != "ok"}
    print(f"    {q:12s} ok {by_obj[q]['ok']:3d}  {bad}")
