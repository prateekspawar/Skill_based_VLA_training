"""Track named objects through the stereo stream with SAM2 and turn each mask into a 3-D point (~/lerobot-env).

    tracker = Sam2PointTracker(locator)
    points, info = tracker.start(left, right, ["red cube", "pink plate"], robot_points)   # OWLv2 boxes seed SAM2
    points, info = tracker.update(left, right, robot_points)                              # next frame

``start`` detects each query once with OWLv2 and prompts one SAM2 video session with the boxes (one object
id per query). ``update`` streams the next left frame through SAM2, which propagates every mask; the 3-D
point of each object is the stereo-depth median over the central part of its (eroded) mask.

Robustness, all without a motion model:
  * robot self-filter: mask pixels whose 3-D point is within ``self_filter_radius`` of the robot keypoints
    (link origins + TCP, densified along the chain; from simulation or forward kinematics) are dropped,
    so the arm / gripper sweeping over an object cannot drag its point;
  * area check: if the mask grows beyond ``max_area_growth`` x its initial size it merged with something
    else -> the update is rejected;
  * jump check: a point moving more than ``max_jump`` since the last accepted one is held back until the
    next update confirms it within ``confirm_radius`` (real moves are followed one update later, single
    glitches are ignored);
  * occlusion: if SAM2 says the object is not visible or too few pixels remain, the last point is kept.

SAM2 is used to measure *displacement*, not absolute position: the published point starts at the box-based
detection (the same value static perception would give) and only changes once the tracked mask point has
moved more than ``deadband`` from where it was at the start -> published = initial detection + displacement.
Stationary objects therefore get exactly the static-perception target (no jitter, no mask bias); moved
objects are followed.

Model: facebook/sam2.1-hiera-small (Apache-2.0).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .locate import TargetLocator

SAM2_MODEL = "facebook/sam2.1-hiera-small"


class Sam2PointTracker:
    def __init__(
        self,
        locator: TargetLocator,
        model_id: str = SAM2_MODEL,
        device: str = "cuda",
        min_mask_px: int = 20,
        self_filter_radius: float = 0.035,
        max_area_growth: float = 1.8,
        max_jump: float = 0.04,
        confirm_radius: float = 0.015,
        deadband: float = 0.025,
    ):
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.locator, self.device = locator, device
        self.min_mask_px, self.self_filter_radius = min_mask_px, self_filter_radius
        self.max_area_growth, self.max_jump, self.confirm_radius = max_area_growth, max_jump, confirm_radius
        self.deadband = deadband
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device).eval()
        self.session, self.queries = None, []
        self.points, self.area0, self.pending = {}, {}, {}  # points = tracked mask points
        self.base, self.ref = {}, {}  # box-based detection and mask point at the start

    def published(self) -> dict:
        """Target points to use: the initial detection, shifted by the tracked displacement beyond the deadband."""
        out = {}
        for q in self.queries:
            disp = self.points[q] - self.ref[q] if q in self.ref else np.zeros(3, np.float32)
            out[q] = (self.base[q] + disp if np.linalg.norm(disp) > self.deadband else self.base[q]).astype(np.float32)
        return out

    def _measure(self, depth, mask, query, robot_points):
        """3-D point from the central part of the eroded mask, with the robot filtered out."""
        core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
        if core.sum() >= self.min_mask_px:
            mask = core
        ys, xs = np.nonzero(mask)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        hw, hh = (xs.max() - xs.min()) / 4 + 1, (ys.max() - ys.min()) / 4 + 1
        central = np.zeros_like(mask)
        central[int(max(cy - hh, 0)) : int(cy + hh) + 1, int(max(cx - hw, 0)) : int(cx + hw) + 1] = True
        if (mask & central).sum() >= self.min_mask_px:
            mask = mask & central
        return self.locator.point_from_mask(
            depth, mask, query, exclude_points=robot_points, exclude_radius=self.self_filter_radius,
            min_px=self.min_mask_px,
        )

    @torch.no_grad()
    def _step(self, left, right, robot_points, prompt_boxes=None, frozen=()):
        inputs = self.processor(images=left, device=self.device, return_tensors="pt")
        if prompt_boxes is not None:
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session,
                frame_idx=0,
                obj_ids=list(range(1, len(prompt_boxes) + 1)),
                input_boxes=[prompt_boxes],
                original_size=inputs.original_sizes[0],
            )
        out = self.model(inference_session=self.session, frame=inputs.pixel_values[0])
        masks = self.processor.post_process_masks([out.pred_masks], original_sizes=inputs.original_sizes)[0]
        visible = (out.object_score_logits.reshape(-1) > 0).cpu().numpy()
        depth = self.locator.depth.depth(left, right)
        info = {}
        for i, query in enumerate(self.queries):
            mask = masks[i, 0].cpu().numpy().astype(bool)
            area = int(mask.sum())
            if query in frozen:
                info[query] = {"status": "frozen", "mask_px": area}
                continue
            if not visible[i] or area < self.min_mask_px:
                info[query] = {"status": "occluded", "mask_px": area}
                continue
            if prompt_boxes is None and query in self.area0 and area > self.max_area_growth * self.area0[query]:
                info[query] = {"status": "rejected_area", "mask_px": area}
                continue
            try:
                p, n = self._measure(depth, mask, query, robot_points)
            except RuntimeError:
                info[query] = {"status": "occluded", "mask_px": area}  # nothing left after the self-filter
                continue
            if prompt_boxes is not None:  # first frame: initialize
                self.points[query], self.area0[query], self.ref[query] = p, area, p
                info[query] = {"status": "init", "mask_px": area, "px_used": n}
                continue
            if np.linalg.norm(p - self.points[query]) > self.max_jump:
                cand = self.pending.get(query)
                if cand is not None and np.linalg.norm(p - cand) < self.confirm_radius:
                    self.points[query] = p  # confirmed move (e.g. the object was pushed)
                    self.pending.pop(query, None)
                    info[query] = {"status": "moved", "mask_px": area, "px_used": n}
                else:
                    self.pending[query] = p
                    info[query] = {"status": "rejected_jump", "mask_px": area, "px_used": n}
                continue
            self.points[query] = p
            self.pending.pop(query, None)
            info[query] = {"status": "tracked", "mask_px": area, "px_used": n}
        return info

    def start(self, left, right, queries, robot_points=None):
        self.queries = list(queries)
        self.points, self.area0, self.pending, self.base, self.ref = {}, {}, {}, {}, {}
        self.session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
        boxes, scores = [], {}
        for q in self.queries:
            box, score = self.locator.detect(left, q)
            boxes.append([float(v) for v in box])
            scores[q] = score
            # Box-based detection: the published starting point (and fallback if SAM2's first mask is unusable).
            self.base[q], _ = self.locator.locate(left, right, q)
            self.points[q] = self.base[q]
        info = self._step(left, right, robot_points, prompt_boxes=boxes)
        for q in self.queries:
            info[q]["score"] = scores[q]
        return self.published(), info

    def update(self, left, right, robot_points=None, frozen=()):
        """Propagate masks to a new frame. Objects in ``frozen`` keep their point (masks still propagate)."""
        if self.session is None:
            raise RuntimeError("call start() first")
        info = self._step(left, right, robot_points, frozen=tuple(frozen))
        return self.published(), info
