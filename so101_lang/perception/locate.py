"""Locate a named object in 3-D: open-vocabulary detection on the left image + stereo depth (~/lerobot-env).

    locator = TargetLocator(calibration)
    point, info = locator.locate(left_rgb, right_rgb, "red cube")   # (3,) head-frame metres

Detection: OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) scores the query against the image.
3-D: Fast-FoundationStereo depth inside the detected box (or a tracked mask, see sam2_tracker.py) is
back-projected to the head frame and the median is taken; for solid objects the point is pushed half an object depth
further along the viewing ray so it approximates the object's centre rather than its visible surface.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from so101_lang import language
from so101_lang.data.hdf5_to_lerobot import make_depth_backend

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OBJECT_HALF_DEPTH = {"cube": 0.015, "plate": 0.0}  # [m] visible surface -> centre, along the view ray
# Detector prompts per object kind (more descriptive phrasing helps OWLv2 separate cubes from plates).
PROMPTS = {"cube": "a small {color} cube", "plate": "a round flat {color} plate"}


def detector_prompt(query: str) -> str:
    words = query.split()
    color, kind = (words[0] if len(words) > 1 else ""), words[-1]
    return " ".join(PROMPTS[kind].format(color=color).split()) if kind in PROMPTS else f"a photo of a {query}"


class TargetLocator:
    def __init__(self, calibration: dict, depth_kind: str = "fast_foundationstereo", device: str = "cuda"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.calib, self.device = calibration, device
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).eval()
        self.depth = make_depth_backend(depth_kind, calibration)
        # Verified detection (attribute contrast + stereo size check, perception/verify.py); SO101_VERIFY_DET=0 off.
        self.verify = os.environ.get("SO101_VERIFY_DET", "1") != "0"
        self.last_detect_info: dict = {}

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, query: str, depth: np.ndarray | None = None,
               robot_points=None, prior_point=None) -> tuple[np.ndarray, float]:
        """Best box (x0, y0, x1, y1) in pixels and its score for a text query like "red cube".

        With verification on, the best box that also passes the attribute-contrast and size checks
        (``depth``: stereo depth of the same frame, for the size check); details in ``last_detect_info``."""
        if self.verify:
            from .verify import detect_ranked

            box, score, self.last_detect_info = detect_ranked(self, rgb, query, depth, detector_prompt(query), robot_points,
                                                              prior_point)
            return box, score
        self.last_detect_info = {}
        inputs = self.processor(text=[[detector_prompt(query)]], images=rgb, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        h, w = rgb.shape[:2]
        size = max(h, w)  # OWLv2 pads to a square
        res = self.processor.post_process_grounded_object_detection(
            out, threshold=0.0, target_sizes=torch.tensor([[size, size]], device=self.device)
        )[0]
        best = int(torch.argmax(res["scores"]))
        box = res["boxes"][best].cpu().numpy()
        box = np.clip(box, 0, [w - 1, h - 1, w - 1, h - 1])
        return box, float(res["scores"][best])

    def locate(self, left: np.ndarray, right: np.ndarray, query: str, robot_points=None,
               prior_point=None) -> tuple[np.ndarray, dict]:
        depth = self.depth.depth(left, right)
        box, score = self.detect(left, query, depth, robot_points, prior_point)
        det_info = dict(self.last_detect_info)
        x0, y0, x1, y1 = box
        # Central 50% of the box avoids background pixels at the edges.
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
        mask = np.zeros(left.shape[:2], dtype=bool)
        mask[int(max(cy - hh, 0)) : int(cy + hh) + 1, int(max(cx - hw, 0)) : int(cx + hw) + 1] = True
        p, n = self.point_from_mask(depth, mask, query)
        return p, {"box": box.tolist(), "score": score, "n_depth_px": n, "detect": det_info}

    def point_from_mask(
        self,
        depth: np.ndarray,
        mask: np.ndarray,
        query: str,
        exclude_points: np.ndarray | None = None,
        exclude_radius: float = 0.035,
        min_px: int = 1,
    ) -> tuple[np.ndarray, int]:
        """Median head-frame 3-D point of the masked pixels (pushed to the object centre for solid objects).

        ``exclude_points`` (K, 3) are robot keypoints in the head frame; mask pixels whose 3-D point lies within
        ``exclude_radius`` of any of them are dropped (robot self-filter).
        """
        k = self.calib["intrinsics"]
        vs, us = np.nonzero(mask & np.isfinite(depth))
        if len(us) == 0:
            raise RuntimeError(f"no valid depth inside the mask for '{query}'")
        d, u, v = depth[vs, us], us + 0.5, vs + 0.5
        # Back-project (optical: x right, y down, z forward) and move to the head frame (x fwd, y left, z up).
        xc, yc = (u - k["cx"]) / k["fx"] * d, (v - k["cy"]) / k["fy"] * d
        off = self.calib["cameras"]["left"]["position_in_head"]
        pts = np.stack([d + off[0], -xc + off[1], -yc + off[2]], axis=1)
        if exclude_points is not None and len(exclude_points):
            dist = np.linalg.norm(pts[:, None, :] - np.asarray(exclude_points)[None, :, :], axis=2).min(axis=1)
            pts = pts[dist > exclude_radius]
        if len(pts) < min_px:
            raise RuntimeError(f"only {len(pts)} valid pixels left for '{query}'")
        p = np.median(pts, axis=0)
        ray = p - np.asarray(off)
        kind = next((k for k in OBJECT_HALF_DEPTH if k in query), None)
        if kind:
            p = p + ray / np.linalg.norm(ray) * OBJECT_HALF_DEPTH[kind]
        return p.astype(np.float32), int(len(pts))



def _box_points_head(locator, depth, box):
    """Head-frame 3-D points of all valid pixels inside a detection box."""
    k = locator.calib["intrinsics"]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    vs, us = np.mgrid[max(y0, 0) : y1 + 1, max(x0, 0) : x1 + 1]
    d = depth[vs, us]
    ok = np.isfinite(d)
    d, u, v = d[ok], us[ok] + 0.5, vs[ok] + 0.5
    xc, yc = (u - k["cx"]) / k["fx"] * d, (v - k["cy"]) / k["fy"] * d
    off = locator.calib["cameras"]["left"]["position_in_head"]
    return np.stack([d + off[0], -xc + off[1], -yc + off[2]], axis=1)


def estimate_yaw_base(locator, depth, box, query) -> float:
    """Object yaw about the base z axis from the top face of its 3-D points (cubes); 0 for plates / unknown.

    A cube's top face only determines yaw modulo 90 deg, so the estimate lies in [-45, 45) deg.
    """
    import cv2

    if "cube" not in query:
        return 0.0
    hib = np.asarray(locator.calib["head_in_base"])
    pts_h = _box_points_head(locator, depth, box)
    pts_b = pts_h @ hib[:3, :3].T + hib[:3, 3]
    if len(pts_b) < 20:
        return 0.0
    top = pts_b[pts_b[:, 2] > np.percentile(pts_b[:, 2], 95) - 0.006]  # the cube's top face
    if len(top) < 10:
        return 0.0
    (_, _), (_, _), angle = cv2.minAreaRect((top[:, :2] * 1000).astype(np.float32))
    yaw = np.deg2rad(angle)
    return float(language.fold_yaw(yaw, "cube"))


def rot6d_from_yaw(yaw_base: float, calib: dict) -> np.ndarray:
    hib = np.asarray(calib["head_in_base"])
    c, s = np.cos(yaw_base), np.sin(yaw_base)
    r_head = hib[:3, :3].T @ np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return np.concatenate([r_head[:, 0], r_head[:, 1]]).astype(np.float32)


@torch.no_grad()
def verify_name(locator: "TargetLocator", rgb: np.ndarray, query: str) -> tuple[bool, float]:
    """Does the requested name win at the detected box against every other name of the same kind?

    The box that best matches ``query`` is scored against all registry names of that object kind
    ("red cube" vs green / blue / yellow cube). Returns (query wins, logit margin over the best other name).
    Measured on find() end views: present cubes win 98 %, absent cubes 4 %. With the current pale plate
    colours it does not separate plates (~53 %).
    """
    from so101_lang import language

    kind = query.split()[-1]
    refs = language.OBJECT_REFS if kind == "cube" else language.TARGET_REFS if kind == "plate" else [query]
    if query not in refs or len(refs) < 2:
        return True, float("inf")
    inputs = locator.processor(text=[[detector_prompt(r) for r in refs]], images=rgb, return_tensors="pt").to(locator.device)
    logits = locator.model(**inputs).logits[0]  # (boxes, names)
    qi = refs.index(query)
    b = int(torch.argmax(logits[:, qi]))
    others = torch.cat([logits[b, :qi], logits[b, qi + 1 :]])
    margin = float(logits[b, qi] - others.max())
    return margin > 0, margin


def locate9(locator: "TargetLocator", left: np.ndarray, right: np.ndarray, query: str,
            robot_points=None, prior_point=None) -> tuple[np.ndarray, dict]:
    """9-D pose (head-frame position + 6-D rotation) of a named object; the output of the find() skill."""
    point, info = locator.locate(left, right, query, robot_points, prior_point)
    depth = locator.depth.depth(left, right)
    yaw = estimate_yaw_base(locator, depth, info["box"], query)
    info["yaw_base_deg"] = float(np.rad2deg(yaw))
    return np.concatenate([point, rot6d_from_yaw(yaw, locator.calib)]).astype(np.float32), info
