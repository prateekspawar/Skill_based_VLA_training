"""Ranked open-vocabulary detection: choose the scene object that best matches the request (~/lerobot-env).

OWLv2 scores boxes by similarity to the whole phrase, and colour dominates that similarity: a blue plate can
outscore a partly hidden blue cube for "blue cube", and a pale-red cube may look "pink" to it. Nothing is
rejected by a hard threshold. Every plausible object is ranked against the request and the best match wins:

* candidates: the best boxes for the full request ("a small red cube") and for its noun alone ("a cube"), so
  every object of that kind in view is considered (non-maximum suppression across both);
* detector terms: the box's OWLv2 score for the request and for the noun;
* size likelihood (stereo): height above the locally estimated support plane ("up" from the hand-eye
  calibration) and horizontal footprint, compared with a coarse range for the noun (cube 2-10 cm tall, plate
  under 2.5 cm, ...). It is 1 inside the range and decays smoothly outside. Nouns without a range get 1;
* colour similarity: each of the object's own pixels (above the support plane, not on the robot) is compared
  with the requested colour. Chromatic colours use hue distance weighted by saturation; white / gray / black use
  saturation and value. The pixels are averaged, so a multi-coloured object scores by how much of it has the
  colour. A small floor keeps any candidate from being zeroed out.

The candidate with the highest sum of log-terms is returned, with its terms. "Not found" is decided elsewhere
(find() verdict and retries). Only images, stereo depth, calibration and robot FK are used.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
import torch

COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "white", "black", "gray", "brown"]
# Coarse size ranges [m] per noun: h = height above the supporting surface, w = largest horizontal extent.
SIZE_PRIORS = {
    "cube": {"h": (0.02, 0.10), "w": (0.015, 0.10)},
    "block": {"h": (0.015, 0.10), "w": (0.015, 0.12)},
    "plate": {"h": (0.0, 0.025), "w": (0.04, 0.40)},
    "dish": {"h": (0.0, 0.04), "w": (0.06, 0.35)},
    "bowl": {"h": (0.02, 0.12), "w": (0.06, 0.35)},
    "cup": {"h": (0.05, 0.15), "w": (0.04, 0.12)},
    "ball": {"h": (0.02, 0.25), "w": (0.02, 0.25)},
    "box": {"h": (0.02, 0.30), "w": (0.02, 0.40)},
}
HUES = {"red": 0, "orange": 30, "yellow": 58, "green": 125, "blue": 215, "purple": 275, "pink": 330, "brown": 25}
TOP_K = 10
HUE_SIGMA = 18.0  # [deg]; red vs yellow (~56 deg apart) must not share much credit
SAT_FULL = 0.25  # saturation at which a pixel's hue counts fully
COLOR_FLOOR = 0.05
SIZE_SIGMA = 0.2  # in log-ratio units outside the range (a 1.2 cm flat plate scores ~0.04 as a cube)
USE_SIZE, USE_COLOR = True, True
PRIOR_SIGMA = 0.03  # [m] soft preference for candidates near a given prior point


def split_query(query: str) -> tuple[list[str], str]:
    words = query.lower().split()
    return words[:-1], words[-1]


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


def _pixels_head(calib: dict, depth: np.ndarray, mask: np.ndarray, return_px: bool = False):
    k = calib["intrinsics"]
    vs, us = np.nonzero(mask & np.isfinite(depth))
    if return_px:
        return _pixels_head(calib, depth, mask), (vs, us)
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
    (pin, (vs, us)), pring = _pixels_head(calib, depth, inner, return_px=True), _pixels_head(calib, depth, ring)
    if robot_points is not None and len(robot_points):
        rp = np.asarray(robot_points)
        far = lambda p: np.linalg.norm(p[:, None, :] - rp[None, :, :], axis=2).min(axis=1) > robot_radius
        keep = far(pin)
        pin, vs, us = pin[keep], vs[keep], us[keep]
        pring = pring[far(pring)]
    measure_size.last_pixels = None
    measure_size.last_center = None
    if len(pin) < 20 or len(pring) < 20:
        return None
    to_b = lambda p: p @ hib[:3, :3].T + hib[:3, 3]
    bin_, bring = to_b(pin), to_b(pring)
    table_z = np.percentile(bring[:, 2], 30)  # the surrounding support surface (robust to other objects)
    hgt = bin_[:, 2] - table_z
    height = float(np.percentile(hgt, 80))
    sel = hgt > max(0.4 * height, 0.003)
    obj = bin_[sel]
    measure_size.last_pixels = (vs[sel], us[sel])  # the object's own pixels (above the support, not robot)
    measure_size.last_center = np.median(pin[sel], axis=0) if sel.any() else None  # head frame
    if len(obj) < 10:
        return height, 0.0
    xy = obj[:, :2] - obj[:, :2].mean(0)
    axes = np.linalg.svd(xy, full_matrices=False)[2]
    proj = xy @ axes.T
    width = float((np.percentile(proj, 95, axis=0) - np.percentile(proj, 5, axis=0)).max())
    return height, width


# Learned colour model (default; SO101_COLOR_MODEL=rule for the hand-set HSV rules below). One Gaussian in CIELab
# per (colour name, object kind) plus the table, fitted on rendered pixels of the training data
# (scripts/fit_color_model.py). A pixel's score for a name is its posterior over all names (equal priors), 0 when
# it fits no component (chi-square 99.9 %): the requested colour only has to look more like itself than like the
# other names under this lighting, which washes every colour towards pastel.
COLOR_MODEL_PATH = Path(__file__).with_name("color_model.json")
COLOR_GATE = {"learned": 0.25, "rule": 0.15}  # plausibility threshold on the colour term


def color_mode() -> str:
    return os.environ.get("SO101_COLOR_MODEL", "learned")


@functools.lru_cache(maxsize=1)
def _color_model():
    m = json.loads(COLOR_MODEL_PATH.read_text())
    keys = list(m["components"])
    mus = np.array([m["components"][k]["mean"] for k in keys])
    cinv = np.array([np.linalg.inv(m["components"][k]["cov"]) for k in keys])
    logdet = np.array([np.log(np.linalg.det(m["components"][k]["cov"])) for k in keys])
    names = sorted({k.split("|")[0] for k in keys})
    member = np.array([[k.split("|")[0] == n for k in keys] for n in names])  # (names, components)
    return names, mus, cinv, logdet, member, m["outlier_m2"]


def color_posterior(rgb_pixels: np.ndarray) -> dict[str, float]:
    """Mean posterior over the learned colour names (and "table") of the pixels (N, 3 uint8)."""
    import cv2

    names, mus, cinv, logdet, member, out_m2 = _color_model()
    x = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(float)
    d = x[None] - mus[:, None]  # (components, N, 3)
    m2 = np.einsum("kni,kij,knj->kn", d, cinv, d)
    ll = -0.5 * m2 - 0.5 * logdet[:, None]
    L = np.stack([ll[mem].max(0) for mem in member])  # best component per name
    p = np.exp(L - L.max(0))
    p /= p.sum(0)
    p[:, m2.min(0) > out_m2] = 0.0
    return {n: float(v) for n, v in zip(names, p.mean(1))}


def color_similarity(rgb_pixels: np.ndarray, color: str) -> float:
    """Similarity of the pixels (N, 3 uint8) to a colour name, in [0, 1]: learned posterior, or the HSV rule."""
    if color_mode() == "learned" and color in _color_model()[0]:
        return color_posterior(rgb_pixels)[color]
    return color_similarity_rule(rgb_pixels, color)


def color_similarity_rule(rgb_pixels: np.ndarray, color: str) -> float:
    """Mean similarity of the pixels (N, 3 uint8) to a basic colour name, in [0, 1] (hand-set HSV rules)."""
    import cv2

    hsv = cv2.cvtColor(rgb_pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(float)
    h, s, v = hsv[:, 0] * 2.0, hsv[:, 1] / 255.0, hsv[:, 2] / 255.0
    achroma = np.clip(1 - s / SAT_FULL, 0, 1)
    if color == "white":
        sim = achroma * np.clip((v - 0.5) / 0.4, 0, 1)
    elif color == "gray":
        sim = achroma * np.clip(1 - np.abs(v - 0.5) / 0.35, 0, 1)
    elif color == "black":
        sim = np.clip(1 - v / 0.3, 0, 1)
    else:
        dh = np.abs((h - HUES[color] + 180) % 360 - 180)
        sim = np.exp(-0.5 * (dh / HUE_SIGMA) ** 2) * np.clip(s / SAT_FULL, 0, 1)
        if color == "brown":
            sim = sim * np.clip((0.7 - v) / 0.3, 0, 1)
    return float(sim.mean())


def size_likelihood(size_m, prior) -> float:
    if size_m is None or prior is None:
        return 1.0
    lik = 1.0
    for x, (lo, hi) in ((size_m[0], prior["h"]), (size_m[1], prior["w"])):
        x = max(x, 1e-3)
        d = np.log(lo / x) if lo > 0 and x < lo else np.log(x / hi) if x > hi else 0.0
        lik *= float(np.exp(-0.5 * (d / SIZE_SIGMA) ** 2))
    return lik


@torch.no_grad()
def detect_ranked(locator, rgb: np.ndarray, query: str, depth: np.ndarray | None, detector_prompt: str,
                  robot_points=None, prior_point=None):
    """Best-matching box for ``query``: (box, score, info). ``info["terms"]`` lists the ranking terms."""
    attrs, noun = split_query(query)
    texts = [detector_prompt, f"a {noun}"]
    inputs = locator.processor(text=[texts], images=rgb, return_tensors="pt").to(locator.device)
    out = locator.model(**inputs)
    logits = out.logits[0].float().cpu().numpy()  # (patches, 2)
    h, w = rgb.shape[:2]
    size = max(h, w)  # OWLv2 pads to a square
    cxcywh = out.pred_boxes[0].float().cpu().numpy() * size
    boxes = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                      cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], 1)
    boxes = np.clip(boxes, 0, [w - 1, h - 1, w - 1, h - 1])
    order = np.argsort(-np.maximum(logits[:, 0], logits[:, 1]))
    cands = _nms(boxes, order, 2 * TOP_K)
    colors = [a for a in attrs if a in COLORS]
    prior = SIZE_PRIORS.get(noun)
    logsig = lambda x: -np.log1p(np.exp(-x))
    rows = []
    for i in cands:
        size_m, pixels = None, None
        if depth is not None and (USE_SIZE or USE_COLOR):
            size_m = measure_size(locator.calib, depth, boxes[i], robot_points)
            pixels = measure_size.last_pixels
        s_lik = size_likelihood(size_m, prior) if USE_SIZE else 1.0
        if USE_COLOR and colors:
            if pixels is None or len(pixels[0]) < 10:  # fall back to the central box pixels
                x0, y0, x1, y1 = boxes[i]
                cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
                px = rgb[int(max(cy - hh, 0)) : int(cy + hh) + 1, int(max(cx - hw, 0)) : int(cx + hw) + 1].reshape(-1, 3)
            else:
                px = rgb[pixels[0], pixels[1]]
            c_sim = float(np.mean([color_similarity(px, c) for c in colors])) if len(px) else 0.0
        else:
            c_sim = 1.0
        total = logsig(logits[i, 0]) + logsig(logits[i, 1]) + np.log(max(s_lik, 1e-6)) + np.log(COLOR_FLOOR + c_sim)
        if prior_point is not None:  # an earlier estimate of where the object is (e.g. fused from several views)
            ctr = measure_size.last_center if depth is not None else None
            if ctr is not None:
                total += -0.5 * min((np.linalg.norm(ctr - np.asarray(prior_point)) / PRIOR_SIGMA) ** 2, 50.0)
        rows.append((total, i, s_lik, c_sim, size_m))
    rows.sort(key=lambda r: -r[0])
    detect_ranked.last_rows = [  # every candidate's terms (diagnostics only)
        {"box": [round(float(v)) for v in boxes[r[1]]], "total": round(float(r[0]), 2),
         "det": round(float(1 / (1 + np.exp(-logits[r[1], 0]))), 3), "noun": round(float(1 / (1 + np.exp(-logits[r[1], 1]))), 3),
         "size": round(r[2], 3), "color": round(r[3], 3),
         "size_m": None if r[4] is None else [round(r[4][0], 3), round(r[4][1], 3)]} for r in rows]
    total, i, s_lik, c_sim, size_m = rows[0]
    score = float(1 / (1 + np.exp(-logits[i, 0])))
    info = {"terms": {"det": round(score, 3), "noun": round(float(1 / (1 + np.exp(-logits[i, 1]))), 3),
                      "size": round(s_lik, 3), "color": round(c_sim, 3)},
            "size_m": None if size_m is None else [round(size_m[0], 4), round(size_m[1], 4)],
            "plausible": bool(s_lik > 0.3 and (c_sim > COLOR_GATE[color_mode()] or not colors)),
            "rank_by_detector": cands.index(i), "n_candidates": len(cands),
            "margin": round(float(total - rows[1][0]), 2) if len(rows) > 1 else None, "total": round(float(total), 3)}
    return boxes[i], score, info
