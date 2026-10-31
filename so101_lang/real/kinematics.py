"""Forward kinematics of the SO-101 from the sim URDF (numpy only, usable on the PC and the Pi).

Joint angles in the LeRobot calibration's degrees (0 = middle pose) equal the URDF ``so101_new_calib`` angles with
the same signs (checked joint by joint on the real arm), so ``fk(q_deg)`` is the real arm's link pose in its base
frame. The arm joints (pan .. wrist_roll) move ``gripper_link``; the ``gripper`` joint (URDF degrees, + = open) moves
only ``moving_jaw_so101_v1_link``.

The simulator loads Isaac Lab's SO-101 USD instead, which differs from this URDF in two ways: its root body ``base``
is a different frame from ``base_link`` (``SIM_BASE_FROM_BASE``; the sim's ``head_in_base`` is relative to that
body), and its wrist_roll joint lacks the URDF's 2.8 deg tilt (``fk(..., sim=True)`` reproduces the sim's arm).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parents[2] / "hardware/so101_gripper/sim_meshes/so101_new_calib.urdf"
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
# Isaac USD shoulder_pan joint frame in its ``base`` body (localPos0, localRot0 as wxyz); the shoulder links agree.
_USD_PAN_POS, _USD_PAN_WXYZ = (0.0207909, -0.0230745, 0.0948817), (8.963205e-7, -0.7071055, -0.7071081, 8.9632374e-7)


def rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    """URDF convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis: np.ndarray, a: float) -> np.ndarray:
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


@lru_cache
def chain(tip: str = "gripper_link") -> tuple:
    """(name, 4x4 origin, axis) for each joint from base_link to ``tip``."""
    joints = {}
    for j in ET.parse(URDF).getroot().findall("joint"):  # top level only (transmissions name joints too)
        o = j.find("origin")
        t = np.eye(4)
        t[:3, :3] = rpy_matrix(*map(float, o.get("rpy", "0 0 0").split()))
        t[:3, 3] = list(map(float, o.get("xyz", "0 0 0").split()))
        ax = j.find("axis")
        joints[j.find("child").get("link")] = (j.get("name"), j.find("parent").get("link"), t,
                                                np.array(list(map(float, ax.get("xyz").split()))) if ax is not None else None)
    out, link = [], tip
    while link != "base_link":
        name, parent, t, ax = joints[link]
        out.append((name, t, ax))
        link = parent
    return tuple(reversed(out))


def quat_wxyz_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _sim_base_from_base() -> np.ndarray:
    pan = np.eye(4)
    pan[:3, :3], pan[:3, 3] = quat_wxyz_matrix(*_USD_PAN_WXYZ), _USD_PAN_POS
    return pan @ np.linalg.inv(chain()[0][1])


SIM_BASE_FROM_BASE = _sim_base_from_base()  # URDF base_link in the sim's base body (-90 deg about z + offset)


def fk(q_deg: dict, tip: str = "gripper_link", sim: bool = False) -> np.ndarray:
    """4x4 pose of ``tip`` in base_link for joint angles in degrees (missing joints = 0). ``sim``: the Isaac asset's
    arm (no wrist_roll tilt), in the sim's base body frame."""
    t = SIM_BASE_FROM_BASE.copy() if sim else np.eye(4)
    for name, origin, ax in chain(tip):
        if sim and name == "wrist_roll":
            origin = origin.copy()
            origin[:3, :3] = rpy_matrix(np.pi / 2, 0.0, np.pi)
        t = t @ origin
        if name in ARM or name == "gripper":
            r = np.eye(4)
            r[:3, :3] = axis_angle(ax, np.radians(q_deg.get(name, 0.0)))
            t = t @ r
    return t
