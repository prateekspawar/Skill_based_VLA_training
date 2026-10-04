"""3-D target tracking: a Kalman filter over noisy, sometimes-wrong perception points.

Tabletop objects sit still until something moves them, so the motion model is a slow random walk in
position (a velocity state would turn detection noise into spurious drift during occlusions); real
moves such as pushes are handled by re-acquisition.

Each perception result (head-frame point + detector score) is
  * skipped if the detector is not confident (occlusion, object in the gripper),
  * accepted if it is statistically consistent with the prediction (Mahalanobis gate),
  * otherwise counted as an outlier; if ``reacquire_after`` consecutive confident outliers agree with each
    other (within ``reacquire_radius``), the object really moved (e.g. it was pushed) and the track is
    re-initialized there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackerCfg:
    meas_sigma: float = 0.01  # [m] perception noise
    walk_sigma: float = 0.01  # [m/sqrt(s)] random-walk process noise (slow creep); jumps use re-acquisition
    gate_chi2: float = 16.0  # ~99.9 % for 3 DoF
    min_score: float = 0.05  # OWLv2 score floor; scores overlap for right/wrong detections, the gate does the rest
    reacquire_after: int = 2  # consecutive agreeing outliers that mean "the object moved"
    reacquire_radius: float = 0.02  # [m]


class TargetTracker:
    def __init__(self, cfg: TrackerCfg | None = None):
        self.cfg = cfg or TrackerCfg()
        self.x = None  # [px, py, pz]
        self.P = None
        self.t = None
        self.outliers: list[np.ndarray] = []

    @property
    def position(self) -> np.ndarray | None:
        return None if self.x is None else self.x.astype(np.float32)

    def _init(self, z: np.ndarray, t: float):
        self.x = np.asarray(z, dtype=float).copy()
        self.P = np.eye(3) * self.cfg.meas_sigma**2
        self.t = t
        self.outliers = []

    def predict(self, t: float):
        if self.x is None:
            return
        dt = max(t - self.t, 0.0)
        self.P = self.P + np.eye(3) * self.cfg.walk_sigma**2 * dt
        self.t = t

    def update(self, z: np.ndarray | None, score: float, t: float) -> str:
        """Fuse one perception result at time ``t`` [s]. Returns 'init', 'accepted', 'reacquired',
        'outlier' or 'skipped'."""
        c = self.cfg
        if z is None or score < c.min_score:
            self.predict(t)
            return "skipped"
        z = np.asarray(z, dtype=float)
        if self.x is None:
            self._init(z, t)
            return "init"
        self.predict(t)
        s = self.P + np.eye(3) * c.meas_sigma**2
        innov = z - self.x
        if innov @ np.linalg.solve(s, innov) <= c.gate_chi2:
            k = self.P @ np.linalg.inv(s)
            self.x = self.x + k @ innov
            self.P = (np.eye(3) - k) @ self.P
            self.outliers = []
            return "accepted"
        # Outlier: either a wrong detection or the object really moved.
        if self.outliers and np.linalg.norm(z - self.outliers[-1]) > c.reacquire_radius:
            self.outliers = []
        self.outliers.append(z)
        if len(self.outliers) >= c.reacquire_after:
            self._init(np.mean(self.outliers, axis=0), t)
            return "reacquired"
        return "outlier"
