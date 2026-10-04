"""Locate a named object in 3-D: open-vocabulary detection on the left image + stereo depth (~/lerobot-env).

    locator = TargetLocator(calibration)
    point, info = locator.locate(left_rgb, right_rgb, "red cube")   # (3,) head-frame metres

Detection: OWLv2 (google/owlv2-base-patch16-ensemble, Apache-2.0) scores the query against the image.
3-D: Fast-FoundationStereo depth inside the detected box (or a tracked mask, see sam2_tracker.py) is
back-projected to the head frame and the median is taken; for solid objects the point is pushed half an object depth
further along the viewing ray so it approximates the object's centre rather than its visible surface.
"""

from __future__ import annotations

import numpy as np
import torch

from so101_lang.data.hdf5_to_lerobot import make_depth_backend

OWL_MODEL = "google/owlv2-base-patch16-ensemble"
OBJECT_HALF_DEPTH = {"cube": 0.015, "plate": 0.0}  # [m] visible surface -> centre, along the view ray
# Detector prompts per object kind (more descriptive phrasing helps OWLv2 separate cubes from plates).
PROMPTS = {"cube": "a small {color} cube", "plate": "a round flat {color} plate"}


def detector_prompt(query: str) -> str:
    color, kind = query.split()[0], query.split()[-1]
    return PROMPTS[kind].format(color=color) if kind in PROMPTS else f"a photo of a {query}"


class TargetLocator:
    def __init__(self, calibration: dict, depth_kind: str = "fast_foundationstereo", device: str = "cuda"):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.calib, self.device = calibration, device
        self.processor = Owlv2Processor.from_pretrained(OWL_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL).to(device).eval()
        self.depth = make_depth_backend(depth_kind, calibration)

    @torch.no_grad()
    def detect(self, rgb: np.ndarray, query: str) -> tuple[np.ndarray, float]:
        """Best box (x0, y0, x1, y1) in pixels and its score for a text query like "red cube"."""
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

    def locate(self, left: np.ndarray, right: np.ndarray, query: str) -> tuple[np.ndarray, dict]:
        box, score = self.detect(left, query)
        x0, y0, x1, y1 = box
        # Central 50% of the box avoids background pixels at the edges.
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 4, (y1 - y0) / 4
        mask = np.zeros(left.shape[:2], dtype=bool)
        mask[int(max(cy - hh, 0)) : int(cy + hh) + 1, int(max(cx - hw, 0)) : int(cx + hw) + 1] = True
        p, n = self.point_from_mask(self.depth.depth(left, right), mask, query)
        return p, {"box": box.tolist(), "score": score, "n_depth_px": n}

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

