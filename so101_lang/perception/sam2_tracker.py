"""Track named objects through the stereo stream with SAM2 and turn each mask into a 3-D point (~/lerobot-env).

    tracker = Sam2PointTracker(locator)
    points = tracker.start(left, right, ["red cube", "pink plate"])   # OWLv2 boxes seed SAM2
    points = tracker.update(left, right)                                # mask propagation on a new frame

``start`` detects each query once with OWLv2 and prompts one SAM2 video session with the boxes (one object
id per query). ``update`` streams the next left frame through SAM2, which propagates every mask, and the 3-D
point of each object is the stereo-depth median inside its mask. When SAM2 reports an object as not visible
(occluded by the arm, or held in the gripper) its last good point is kept. Scales to any number of objects
in one session and needs no per-object motion model.

Model: facebook/sam2.1-hiera-small (Apache-2.0).
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .locate import TargetLocator

SAM2_MODEL = "facebook/sam2.1-hiera-small"


class Sam2PointTracker:
    def __init__(self, locator: TargetLocator, model_id: str = SAM2_MODEL, device: str = "cuda",
                 min_mask_px: int = 20):
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.locator, self.device, self.min_mask_px = locator, device, min_mask_px
        self.processor = Sam2VideoProcessor.from_pretrained(model_id)
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device).eval()
        self.session, self.queries, self.points = None, [], {}

    @torch.no_grad()
    def _step(self, left: np.ndarray, right: np.ndarray, prompt_boxes: list | None = None) -> dict:
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
            # Use the mask interior: edge pixels mix in sides / background and bias the depth median.
            core = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=2).astype(bool)
            mask = core if core.sum() >= self.min_mask_px else mask
            # Central half of the mask's bounding box (as the box-based locator does): robust to partial
            # overlap with the gripper at the mask edges.
            ys, xs = np.nonzero(mask)
            if len(xs):
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                hw, hh = (xs.max() - xs.min()) / 4 + 1, (ys.max() - ys.min()) / 4 + 1
                central = np.zeros_like(mask)
                central[int(max(cy - hh, 0)) : int(cy + hh) + 1, int(max(cx - hw, 0)) : int(cx + hw) + 1] = True
                if (mask & central).sum() >= self.min_mask_px:
                    mask = mask & central
            n = int(mask.sum())
            if visible[i] and n >= self.min_mask_px:
                try:
                    self.points[query], _ = self.locator.point_from_mask(depth, mask, query)
                    info[query] = {"status": "tracked", "mask_px": n}
                    continue
                except RuntimeError:
                    pass
            info[query] = {"status": "held", "mask_px": n}  # occluded / lost: keep the last good point
        return info

    def start(self, left: np.ndarray, right: np.ndarray, queries: list[str]) -> tuple[dict, dict]:
        self.queries = list(queries)
        self.points = {}
        self.session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
        boxes, scores = [], {}
        for q in self.queries:
            box, score = self.locator.detect(left, q)
            boxes.append([float(v) for v in box])
            scores[q] = score
            # Fallback point from the box in case SAM2's first mask is empty.
            self.points[q], _ = self.locator.locate(left, right, q)
        info = self._step(left, right, prompt_boxes=boxes)
        for q in self.queries:
            info[q]["score"] = scores[q]
        return dict(self.points), info

    def update(self, left: np.ndarray, right: np.ndarray, frozen: tuple[str, ...] = ()) -> tuple[dict, dict]:
        """Propagate masks to a new frame. Objects in ``frozen`` keep their point (masks still propagate)."""
        if self.session is None:
            raise RuntimeError("call start() first")
        keep = {q: self.points[q] for q in frozen if q in self.points}
        info = self._step(left, right)
        for q, p in keep.items():
            self.points[q] = p
            info[q]["status"] = "frozen"
        return dict(self.points), info
