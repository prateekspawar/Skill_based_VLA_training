"""Stereo-head calibration in simulation: the head frame's pose in the robot base frame."""

from __future__ import annotations

import torch

from so101_lang.robots import frames, virtual_head

from .pick_place_env_cfg import HEAD_CENTER, HEAD_QUAT, LEFT_CAM_POS, RIGHT_CAM_POS, STEREO_BASELINE, stereo_intrinsics


def head_in_env(device) -> torch.Tensor:
    """Pose of the head frame the models use: the training pose. With a camera offset (SO101_CAM_OFFSET) and the
    virtual head off (SO101_VIRTUAL_HEAD=0), the rendered camera's own pose instead (so101_lang.robots.virtual_head)."""
    pos = torch.tensor(HEAD_CENTER, dtype=torch.float32, device=device)
    rot = frames.quat_to_matrix(torch.tensor(HEAD_QUAT, dtype=torch.float32, device=device))
    h = frames.make_transform(pos, rot)
    if not virtual_head.enabled():
        h = h @ torch.tensor(virtual_head.real_in_virtual(), dtype=torch.float32, device=device)
    return h


def head_in_base(env) -> torch.Tensor:
    """(N, 4, 4) head pose in each env's robot base frame (the hand-eye calibration on real hardware)."""
    robot = env.scene["robot"]
    base_pos = robot.data.root_pos_w.torch - env.scene.env_origins
    base_in_env = frames.make_transform(base_pos, frames.quat_to_matrix(robot.data.root_quat_w.torch))
    return frames.invert_transform(base_in_env) @ head_in_env(env.device).expand(env.num_envs, 4, 4)


def calibration_dict(head_in_base_0: torch.Tensor) -> dict:
    """JSON-serializable stereo calibration (intrinsics, extrinsics in the head frame, head-in-base)."""
    head_inv = frames.invert_transform(head_in_env("cpu"))
    cams = {}
    for name, pos in (("left", LEFT_CAM_POS), ("right", RIGHT_CAM_POS)):
        p = (head_inv[:3, :3] @ torch.tensor(pos) + head_inv[:3, 3]).tolist()
        cams[name] = {"position_in_head": [round(v, 6) for v in p], "orientation": "parallel to head (x forward, y left, z up)"}
    return {
        "convention": "head frame: origin between optical centres, +X forward, +Y left, +Z up; quaternions xyzw",
        "baseline_m": STEREO_BASELINE,
        "intrinsics": stereo_intrinsics(),
        "cameras": cams,
        "head_in_base": head_in_base_0.cpu().tolist(),
    }
