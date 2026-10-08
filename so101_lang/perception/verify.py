"""Verified open-vocabulary detection: pick the best detector box that also passes two general checks (~/lerobot-env).

OWLv2 ranks boxes by similarity to the whole phrase, and colour dominates that similarity: a blue plate can
outscore a partly hidden blue cube for "blue cube". Each candidate box (top-K by the detector score) is checked:

1. Attribute contrast (off by default, see USE_*_SWAPS). In the same detector pass, the box must score higher for the requested name ("a blue cube")
   than for every variant with one attribute swapped: other nouns from a small generic list ("a blue plate",
   "a blue bowl", ...) and other colours from a generic colour list ("a red cube", ...). No list of the objects
   in the scene is needed, only the requested name split into attributes + noun.
2. Size (stereo). The table plane is estimated locally (lower percentile of the base-frame heights around the box;
   "up" comes from the hand-eye calibration). The object's height above it and its horizontal footprint are
   compared with a coarse range for the noun (cube / block 1.5-10 cm tall, plate under 2.5 cm and 6-35 cm wide,
   ...). Nouns without a range skip this check.

The first candidate (in detector order) passing both checks is used. If none does, the best box passing the size
check is used, then the best passing the contrast, then the plain detector best; ``verified`` reports whether
both checks passed. Only images, stereo depth and calibration are used: no simulator information.
"""

from __future__ import annotations

import numpy as np
import torch

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black", "gray", "brown"]
NOUNS = ["cube", "block", "plate", "bowl", "cup", "ball", "box", "bottle"]
# Coarse size ranges [m] per noun: h = height above the supporting surface, w = largest horizontal extent.
SIZE_PRIORS = {
    "cube": {"h": (0.015, 0.10), "w": (0.015, 0.10)},
    "block": {"h": (0.015, 0.10), "w": (0.015, 0.12)},
    "plate": {"h": (0.0, 0.025), "w": (0.04, 0.40)},
    "dish": {"h": (0.0, 0.04), "w": (0.06, 0.35)},
    "bowl": {"h": (0.02, 0.12), "w": (0.06, 0.35)},
    "cup": {"h": (0.05, 0.15), "w": (0.04, 0.12)},
    "ball": {"h": (0.02, 0.25), "w": (0.02, 0.25)},
    "box": {"h": (0.02, 0.30), "w": (0.02, 0.40)},
}
TOP_K = 10
# Near-synonyms are not contrasts ("a blue box" is a fair description of a blue cube).
NOUN_GROUPS = [{"cube", "block", "box"}, {"plate", "dish"}, {"cup", "mug"}]
# The attribute contrast is off by default: OWLv2 is not reliable at fine attribute distinctions (it prefers
# "a pink cube" for a third of rendered red cubes and "a pink / white block" for flat plates), so as a hard check
# it rejects correct boxes. The stereo size check separates cubes from plates cleanly and is used alone.
USE_NOUN_SWAPS, USE_COLOR_SWAPS, USE_SIZE = False, False, True


def split_query(query: str) -> tuple[list[str], str]:
    words = query.lower().split()
    return words[:-1], words[-1]


def contrast_prompts(query: str) -> tuple[str, list[str]]:
    """Uniformly phrased main prompt and its one-attribute swaps."""
    attrs, noun = split_query(query)
    main = " ".join(["a", *attrs, noun])
    group = next((g for g in NOUN_GROUPS if noun in g), {noun})
    swaps = [" ".join(["a", *attrs, n]) for n in NOUNS if n not in group] if USE_NOUN_SWAPS else []
    for i, a in enumerate(attrs):
        if a in COLORS and USE_COLOR_SWAPS:
            swaps += [" ".join(["a", *attrs[:i], c, *attrs[i + 1 :], noun]) for c in COLORS if c != a]
    return main, swaps


def _nms(boxes: np.ndarray, order: np.ndarray, k: int, iou_thr: float = 0.5) -> list[int]:
    keep = []
    for i in order:
        b = boxes[i]
        ok = True
        for j in keep:
            c = boxes[j]
            ix = max(0.0, min(b[2], c[2]) - max(b[0], c[0]))
            iy = max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
            inter = ix * iy
            union = (b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append(int(i))
        if len(keep) >= k:
            break
    return keep


def _pixels_head(calib: dict, depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    k = calib["intrinsics"]
    vs, us = np.nonzero(mask & np.isfinite(depth))
    d = depth[vs, us]
    xc, yc = (us + 0.5 - k["cx"]) / k["fx"] * d, (vs + 0.5 - k["cy"]) / k["fy"] * d
    off = calib["cameras"]["left"]["position_in_head"]
    return np.stack([d + off[0], -xc + off[1], -yc + off[2]], axis=1)


def measure_size(calib: dict, depth: np.ndarray, box, robot_points=None, robot_radius: float = 0.035):
    """(height above the local support plane, horizontal footprint) [m] of the object in ``box``, or None.

    Only the central 70 % of the box is measured and points near the robot keypoints (forward kinematics) are
    dropped, so an arm passing in front of or behind the object does not count as object height."""
    hib = np.asarray(calib["head_in_base"], dtype=np.float64)
    h_img, w_img = depth.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    box_m = np.zeros(depth.shape, dtype=bool)
    box_m[int(max(y0, 0)) : int(min(y1, h_img - 1)) + 1, int(max(x0, 0)) : int(min(x1, w_img - 1)) + 1] = True
    inner = np.zeros(depth.shape, dtype=bool)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    inner[int(max(cy - 0.35 * bh, 0)) : int(min(cy + 0.35 * bh, h_img - 1)) + 1,
          int(max(cx - 0.35 * bw, 0)) : int(min(cx + 0.35 * bw, w_img - 1)) + 1] = True
    outer = np.zeros(depth.shape, dtype=bool)
    ex, ey = 0.6 * bw + 4, 0.6 * bh + 4
    outer[int(max(y0 - ey, 0)) : int(min(y1 + ey, h_img - 1)) + 1, int(max(x0 - ex, 0)) : int(min(x1 + ex, w_img - 1)) + 1] = True
    ring = outer & ~box_m
    pin, pring = _pixels_head(calib, depth, inner), _pixels_head(calib, depth, ring)
    if robot_points is not None and len(robot_points):
        rp = np.asarray(robot_points)
        far = lambda p: np.linalg.norm(p[:, None, :] - rp[None, :, :], axis=2).min(axis=1) > robot_radius
        pin, pring = pin[far(pin)], pring[far(pring)]
    if len(pin) < 20 or len(pring) < 20:
        return None
    to_b = lambda p: p @ hib[:3, :3].T + hib[:3, 3]
    bin_, bring = to_b(pin), to_b(pring)
    table_z = np.percentile(bring[:, 2], 30)  # the surrounding support surface (robust to other objects)
    hgt = bin_[:, 2] - table_z
    height = float(np.percentile(hgt, 80))
    obj = bin_[hgt > max(0.4 * height, 0.003)]
    if len(obj) < 10:
        return height, 0.0
    xy = obj[:, :2] - obj[:, :2].mean(0)
    axes = np.linalg.svd(xy, full_matrices=False)[2]
    proj = xy @ axes.T
    width = float((np.percentile(proj, 95, axis=0) - np.percentile(proj, 5, axis=0)).max())
    return height, width


@torch.no_grad()
def detect_verified(locator, rgb: np.ndarray, query: str, depth: np.ndarray | None, detector_prompt: str,
                    robot_points=None):
    """Best verified box for ``query``: (box, score, info)."""
    main, swaps = contrast_prompts(query)
    texts = [detector_prompt, main, *swaps]
    inputs = locator.processor(text=[texts], images=rgb, return_tensors="pt").to(locator.device)
    out = locator.model(**inputs)
    logits = out.logits[0].float().cpu().numpy()  # (patches, prompts)
    h, w = rgb.shape[:2]
    size = max(h, w)  # OWLv2 pads to a square
    cxcywh = out.pred_boxes[0].float().cpu().numpy() * size
    boxes = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                      cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], 1)
    boxes = np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1])
    order = np.argsort(-logits[:, 0])
    cands = _nms(boxes, order, TOP_K)
    noun = split_query(query)[1]
    prior = SIZE_PRIORS.get(noun)
    checked = []
    for i in cands:
        name_ok = bool(logits[i, 1] > logits[i, 2:].max()) if len(texts) > 2 else True
        geom_ok, size_m = None, None
        if prior is not None and depth is not None and USE_SIZE:
            size_m = measure_size(locator.calib, depth, boxes[i], robot_points)
            if size_m is not None:
                (h0, h1), (w0, w1) = prior["h"], prior["w"]
                geom_ok = bool(h0 <= size_m[0] <= h1 and w0 <= size_m[1] <= w1)
        checked.append((i, name_ok, geom_ok, size_m))
    pick = (next((c for c in checked if c[1] and c[2] is not False), None)
            or next((c for c in checked if c[2]), None)
            or next((c for c in checked if c[1]), None)
            or checked[0])
    i, name_ok, geom_ok, size_m = pick
    score = float(1 / (1 + np.exp(-logits[i, 0])))
    info = {"verified": bool(name_ok and geom_ok is not False), "name_ok": name_ok, "geom_ok": geom_ok,
            "size_m": None if size_m is None else [round(size_m[0], 4), round(size_m[1], 4)],
            "rank": cands.index(i), "n_candidates": len(cands)}
    return boxes[i], score, info
