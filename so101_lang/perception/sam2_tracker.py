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
        shared: "Sam2PointTracker | None" = None,
        mode: str = "state",
        move_thr: float = 0.015,
        move_confirm: int = 2,
        settle_tol: float = 0.005,
        settle_n: int = 3,
        area_band: tuple[float, float] = (0.7, 1.4),
        max_robot_frac: float = 0.15,
        follow_area_band: tuple[float, float] = (0.7, 1.4),  # looser (0.4, 2.0) / 0.5 tested: worse (README)
        follow_max_robot_frac: float = 0.15,
    ):
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.locator, self.device = locator, device
        self.min_mask_px, self.self_filter_radius = min_mask_px, self_filter_radius
        self.max_area_growth, self.max_jump, self.confirm_radius = max_area_growth, max_jump, confirm_radius
        self.deadband = deadband
        self.mode = mode  # "state": locked / following state machine; "deadband": the earlier displacement deadband
        self.move_thr, self.move_confirm = move_thr, move_confirm
        self.settle_tol, self.settle_n = settle_tol, settle_n
        self.area_band, self.max_robot_frac = area_band, max_robot_frac
        # Looser trust while following a confirmed move: the gripper often overlaps a cube that is still sliding.
        self.follow_area_band, self.follow_max_robot_frac = follow_area_band, follow_max_robot_frac
        self.state, self.anchor, self.recent, self.departs = {}, {}, {}, {}
        if shared is not None:  # reuse another tracker's weights (one SAM2 model, several sessions)
            self.processor, self.model = shared.processor, shared.model
        else:
            self.processor = Sam2VideoProcessor.from_pretrained(model_id)
            self.model = Sam2VideoModel.from_pretrained(model_id).to(device).eval()
        self.session, self.queries = None, []
        self.points, self.area0, self.pending = {}, {}, {}  # points = tracked mask points
        self.base, self.ref = {}, {}  # box-based detection and mask point at the start

    def published(self) -> dict:
        """Target points to use: the initial detection, shifted by the tracked displacement beyond the deadband."""
        out = {}
        if self.mode == "state":
            for q in self.queries:
                out[q] = self.anchor.get(q, self.base[q]).astype(np.float32)
            return out
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
        p, n = self.locator.point_from_mask(
            depth, mask, query, exclude_points=robot_points, exclude_radius=self.self_filter_radius,
            min_px=self.min_mask_px,
        )
        n_all = int((mask & np.isfinite(depth)).sum())
        self._robot_frac = 1.0 - n / max(n_all, 1)  # share of the object's pixels lying on the robot
        return p, n

    @torch.no_grad()
    def _step(self, left, right, robot_points, prompt_boxes=None, frozen=(), depth=None):
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
        if depth is None:
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
            if (self.mode != "state" and prompt_boxes is None and query in self.area0
                    and area > self.max_area_growth * self.area0[query]):
                info[query] = {"status": "rejected_area", "mask_px": area}
                continue
            try:
                p, n = self._measure(depth, mask, query, robot_points)
            except RuntimeError:
                info[query] = {"status": "occluded", "mask_px": area}  # nothing left after the self-filter
                continue
            if prompt_boxes is not None:  # first frame: initialize
                self.points[query], self.area0[query], self.ref[query] = p, area, p
                self.state[query], self.anchor[query] = "locked", self.base[query]
                self.recent[query], self.departs[query] = [], []
                info[query] = {"status": "init", "mask_px": area, "px_used": n}
                continue
            if self.mode == "state":
                info[query] = self._state_update(query, p, area, n)
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

    def _state_update(self, q, p, area, n):
        """Locked / following state machine on *trusted* measurements (images, depth and robot FK only).

        A measurement is trusted when the mask is about its initial size (not partly hidden, not merged) and
        few of its pixels lie on the robot. Locked: the published point stays at the (clear-view) detection;
        it switches to following only when ``move_confirm`` consecutive trusted measurements are more than
        ``move_thr`` from the reference in a consistent place (a real move by anything). Following: the
        published point = anchor + measured displacement on every trusted update; when ``settle_n`` trusted
        measurements agree within ``settle_tol`` the object has stopped and it locks again there.
        Untrusted measurements never move the published point.
        """
        if q not in self.area0:  # the first mask was unusable (hidden at start): initialise on this measurement
            self.points[q], self.area0[q], self.ref[q] = p, area, p
            self.state[q], self.anchor[q], self.recent[q], self.departs[q] = "locked", self.base[q], [], []
            return {"status": "init", "mask_px": area, "px_used": n}
        following = self.state[q] == "following"
        lo, hi = self.follow_area_band if following else self.area_band
        a0 = self.area0[q]
        if not lo * a0 <= area <= hi * a0:
            return {"status": "untrusted_area", "mask_px": area, "px_used": n}
        if self._robot_frac > (self.follow_max_robot_frac if following else self.max_robot_frac):
            return {"status": "untrusted_robot", "mask_px": area, "px_used": n}
        self.points[q] = p
        disp = p - self.ref[q]
        if self.state[q] == "locked":
            if np.linalg.norm(disp) > self.move_thr:
                d = self.departs[q]
                if d and np.linalg.norm(p - d[-1]) > 2 * self.move_thr:
                    d.clear()  # inconsistent with the previous departure: start counting again
                d.append(p)
                if len(d) >= self.move_confirm:
                    self.state[q] = "following"
                    self.anchor[q] = self.anchor[q] + disp
                    self.ref[q] = p
                    self.recent[q] = [p]
                    d.clear()
                    return {"status": "moved", "mask_px": area, "px_used": n}
                return {"status": "departing", "mask_px": area, "px_used": n}
            self.departs[q].clear()
            return {"status": "locked", "mask_px": area, "px_used": n}
        # following
        self.anchor[q] = self.anchor[q] + disp
        self.ref[q] = p
        r = self.recent[q]
        r.append(p)
        del r[: -self.settle_n]
        if len(r) == self.settle_n and max(np.linalg.norm(a - b) for a in r for b in r) < self.settle_tol:
            self.state[q] = "locked"
            r.clear()
            return {"status": "settled", "mask_px": area, "px_used": n}
        return {"status": "following", "mask_px": area, "px_used": n}

    def start(self, left, right, queries, robot_points=None, prior_points=None, base_points=None):
        """``prior_points``: per query, an earlier position estimate that steers the detection; ``base_points``:
        per query, the published starting point (e.g. fused from several views) instead of this frame's."""
        prior_points, base_points = prior_points or {}, base_points or {}
        self.queries = list(queries)
        self.points, self.area0, self.pending, self.base, self.ref = {}, {}, {}, {}, {}
        self.state, self.anchor, self.recent, self.departs = {}, {}, {}, {}
        self.session = self.processor.init_video_session(inference_device=self.device, dtype=torch.float32)
        boxes, scores = [], {}
        for q in self.queries:
            box, score = self.locator.detect(left, q, self.locator.depth.depth(left, right), robot_points,
                                             prior_points.get(q))
            boxes.append([float(v) for v in box])
            scores[q] = score
            # Box-based detection: the published starting point (and fallback if SAM2's first mask is unusable).
            self.base[q], _ = self.locator.locate(left, right, q, robot_points, prior_points.get(q))
            if base_points.get(q) is not None:
                self.base[q] = np.asarray(base_points[q], dtype=np.float32)
            self.points[q] = self.base[q]
        info = self._step(left, right, robot_points, prompt_boxes=boxes)
        for q in self.queries:
            info[q]["score"] = scores[q]
        return self.published(), info

    def update(self, left, right, robot_points=None, frozen=(), depth=None):
        """Propagate masks to a new frame. Objects in ``frozen`` keep their point (masks still propagate)."""
        if self.session is None:
            raise RuntimeError("call start() first")
        info = self._step(left, right, robot_points, frozen=tuple(frozen), depth=depth)
        return self.published(), info


class Sam2MultiTracker:
    """One SAM2 session per object, so objects can be added at different times (each from its own find() view)
    and all keep tracking through every later frame. One SAM2 model and one depth map per frame are shared.

        mt = Sam2MultiTracker(locator)
        mt.add(left, right, "red cube", robot_points)     # after find(red cube)
        mt.update(left, right, robot_points)              # every few control steps from now on (cube keeps tracking)
        mt.add(left, right, "pink plate", robot_points)   # after find(pink plate)
        points, info = mt.update(left, right, robot_points, frozen=("red cube",))
    """

    def __init__(self, locator: TargetLocator, **kwargs):
        self.locator, self.kwargs = locator, kwargs
        self.trackers: dict[str, Sam2PointTracker] = {}
        self._model_owner: Sam2PointTracker | None = None

    def reset(self):
        """Drop all trackers and their SAM2 video sessions (the per-frame state otherwise accumulates on the GPU:
        composed tasks start many trackers per episode and ran the server out of memory)."""
        for t in self.trackers.values():
            t.session = None
        if self._model_owner is not None:
            self._model_owner.session = None
        self.trackers = {}
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def published(self) -> dict:
        out = {}
        for t in self.trackers.values():
            out.update(t.published())
        return out

    def add(self, left, right, query, robot_points=None, prior_point=None, base_point=None):
        """(Re)start tracking ``query`` from this frame: OWLv2 box -> SAM2 prompt; other objects are untouched."""
        if self._model_owner is None:
            self._model_owner = Sam2PointTracker(self.locator, **self.kwargs)
            t = self._model_owner
        else:
            t = Sam2PointTracker(self.locator, shared=self._model_owner, **self.kwargs)
        _, info = t.start(left, right, [query], robot_points, {query: prior_point}, {query: base_point})
        self.trackers[query] = t
        return self.published(), info

    def remove(self, query):
        self.trackers.pop(query, None)

    def start(self, left, right, queries, robot_points=None):
        """All objects from the same frame (direct pipeline)."""
        self.reset()
        info = {}
        for q in queries:
            info.update(self.add(left, right, q, robot_points)[1])
        return self.published(), info

    def update(self, left, right, robot_points=None, frozen=()):
        if not self.trackers:
            return {}, {}
        depth = self.locator.depth.depth(left, right)
        info = {}
        for q, t in self.trackers.items():
            info.update(t.update(left, right, robot_points, frozen=tuple(f for f in frozen if f == q), depth=depth)[1])
        return self.published(), info
