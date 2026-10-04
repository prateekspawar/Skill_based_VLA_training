"""Small, dependency-free (torch only) rigid-transform helpers shared by both venvs.

Quaternions are ``(x, y, z, w)``. Rotations fed to / produced by policies use the continuous 6-D
representation (the first two columns of the rotation matrix, Zhou et al. 2019).
"""

from __future__ import annotations

import torch


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(*q.shape[:-1], 3, 3)


def matrix_to_quat(m: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> unit quaternion (xyzw) with w >= 0."""
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    qw = 0.5 * torch.sqrt((1 + m00 + m11 + m22).clamp_min(1e-12))
    qx = 0.5 * torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-12))
    qy = 0.5 * torch.sqrt((1 - m00 + m11 - m22).clamp_min(1e-12))
    qz = 0.5 * torch.sqrt((1 - m00 - m11 + m22).clamp_min(1e-12))
    qx = torch.copysign(qx, m[..., 2, 1] - m[..., 1, 2])
    qy = torch.copysign(qy, m[..., 0, 2] - m[..., 2, 0])
    qz = torch.copysign(qz, m[..., 1, 0] - m[..., 0, 1])
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    return q / q.norm(dim=-1, keepdim=True)


def rot6d_from_matrix(m: torch.Tensor) -> torch.Tensor:
    return torch.cat([m[..., :, 0], m[..., :, 1]], dim=-1)


def matrix_from_rot6d(r: torch.Tensor) -> torch.Tensor:
    """Gram-Schmidt the two predicted columns back into a proper rotation matrix."""
    a1, a2 = r[..., :3], r[..., 3:6]
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    a2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = a2 / a2.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def make_transform(pos: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    t = torch.zeros(*pos.shape[:-1], 4, 4, dtype=pos.dtype, device=pos.device)
    t[..., :3, :3] = rot
    t[..., :3, 3] = pos
    t[..., 3, 3] = 1.0
    return t


def invert_transform(t: torch.Tensor) -> torch.Tensor:
    rot_t = t[..., :3, :3].transpose(-1, -2)
    return make_transform(-(rot_t @ t[..., :3, 3:4]).squeeze(-1), rot_t)


def apply_transform(t: torch.Tensor, pos: torch.Tensor, rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Express a pose (pos, rot) given in frame B in frame A, where ``t`` is B's pose in A."""
    return (t[..., :3, :3] @ pos.unsqueeze(-1)).squeeze(-1) + t[..., :3, 3], t[..., :3, :3] @ rot
