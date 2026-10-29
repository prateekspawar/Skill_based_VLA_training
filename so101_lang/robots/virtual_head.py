"""Virtual head frame: run the trained models with the stereo camera at a different pose, without retraining.

The models' state, target, point map and actions live in the head frame of the camera pose used in training. With a
real camera somewhere else, the "virtual head" keeps that training frame for everything the models see: perception
measures in the real camera's frame (with the real calibration) and its outputs are re-expressed in the virtual
frame; the policy's state, targets and actions stay in the virtual frame. Only the images differ from training.

Configuration (environment variables, read by the simulator and the policy server):

    SO101_CAM_OFFSET="dx,dy,dz,roll,pitch,yaw"   pose of the real camera in the virtual (training) head frame:
                                                 metres along head x (forward), y (left), z (up), then degrees
                                                 about head x, y, z (applied as Rz(yaw) Ry(pitch) Rx(roll)).
                                                 Unset = camera at the training pose.
    SO101_VIRTUAL_HEAD=1 (default) / 0           0 = no virtual frame: the models get the real camera's frame
                                                 directly (camera pose as plain calibration), for comparison.

On a real robot the offset comes from the hand-eye calibration: real_in_virtual = inv(H_train) @ H_real, with H the
head-in-base transforms.
"""

from __future__ import annotations

import os

import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def real_in_virtual() -> np.ndarray:
    """4x4 pose of the real camera's head frame in the virtual (training) head frame; identity when unset."""
    t = np.eye(4)
    spec = os.environ.get("SO101_CAM_OFFSET", "").strip()
    if spec:
        dx, dy, dz, roll, pitch, yaw = (float(v) for v in spec.split(","))
        t[:3, :3] = rpy_matrix(*np.deg2rad([roll, pitch, yaw]))
        t[:3, 3] = (dx, dy, dz)
    return t


def enabled() -> bool:
    return os.environ.get("SO101_VIRTUAL_HEAD", "1") != "0"


def policy_from_real() -> np.ndarray:
    """4x4 map from the real camera's head frame to the frame the models use (virtual head, or the real head when
    the virtual frame is off)."""
    return real_in_virtual() if enabled() else np.eye(4)


def apply(t: np.ndarray, p):
    """Transform points (..., 3), or 9-D poses (position + rot6d, ...) with t (4x4)."""
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float32)
    if p.shape[-1] == 9:
        r = t[:3, :3].astype(np.float32)
        pos = p[..., :3] @ r.T + t[:3, 3].astype(np.float32)
        return np.concatenate([pos, p[..., 3:6] @ r.T, p[..., 6:9] @ r.T], axis=-1).astype(np.float32)
    return (p @ t[:3, :3].T.astype(np.float32) + t[:3, 3].astype(np.float32)).astype(np.float32)


def apply_pointmap(t: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """(3, H, W) point map; zeros (unknown) stay zero."""
    valid = np.any(pm != 0, axis=0)
    out = np.einsum("ij,jhw->ihw", t[:3, :3], pm) + t[:3, 3][:, None, None]
    return np.where(valid[None], out, 0.0).astype(np.float32)
