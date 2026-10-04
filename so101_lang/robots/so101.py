"""SO-101 embodiment adapter: robot-agnostic end-effector actions <-> SO-101 IK commands.

The VLA works in the **head frame** (the stereo camera mount: origin midway between the two optical
centres, +X forward along the viewing direction, +Y to the left, +Z up). Its state and action are

    [tcp_x, tcp_y, tcp_z, r11, r21, r31, r12, r22, r32, gripper_opening]   (10-D)

* ``tcp`` is the grasp point between the jaw pads, the rotation is the TCP frame as a 6-D vector
  (first two matrix columns), and ``gripper_opening`` is normalized 0 = closed, 1 = fully open.

Everything robot-specific lives here: where the TCP sits on the gripper link, how the gripper opening
maps to the gripper joint, and the head-to-base calibration (hand-eye calibration on a real robot).
The SO-101's own IK (Isaac Lab's damped-least-squares pose IK in simulation) turns the resulting
gripper-link pose into joint angles. Another arm only needs its own adapter + IK.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import frames

STATE_NAMES = ["tcp.x", "tcp.y", "tcp.z", "tcp.r11", "tcp.r21", "tcp.r31", "tcp.r12", "tcp.r22", "tcp.r32", "gripper.open"]


@dataclass
class SO101Adapter:
    # Grasp centre in the gripper-link frame [m] (measured from the collision geometry).
    tcp_offset: tuple[float, float, float] = (0.009, 0.0, -0.090)
    gripper_open_rad: float = 1.745  # gripper joint angle when fully open (0 rad = closed)

    def _offset(self, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.tcp_offset, dtype=like.dtype, device=like.device)

    # ---------------------------------------------------------------- gripper
    def opening_from_joint(self, gripper_joint: torch.Tensor) -> torch.Tensor:
        return (gripper_joint / self.gripper_open_rad).clamp(0.0, 1.0)

    # ---------------------------------------------------------------- base-frame poses
    def tcp_from_link(self, link_pos_b: torch.Tensor, link_quat_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rot = frames.quat_to_matrix(link_quat_b)
        return link_pos_b + (rot @ self._offset(link_pos_b).unsqueeze(-1)).squeeze(-1), rot

    def link_from_tcp(self, tcp_pos_b: torch.Tensor, tcp_rot_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return tcp_pos_b - (tcp_rot_b @ self._offset(tcp_pos_b).unsqueeze(-1)).squeeze(-1), frames.matrix_to_quat(tcp_rot_b)

    # ---------------------------------------------------------------- robot-agnostic <-> SO-101
    def to_agnostic(
        self, tcp_pos_b: torch.Tensor, tcp_rot_b: torch.Tensor, opening: torch.Tensor, head_in_base: torch.Tensor
    ) -> torch.Tensor:
        """Base-frame TCP pose + opening -> 10-D head-frame vector."""
        base_in_head = frames.invert_transform(head_in_base)
        pos_h, rot_h = frames.apply_transform(base_in_head, tcp_pos_b, tcp_rot_b)
        return torch.cat([pos_h, frames.rot6d_from_matrix(rot_h), opening.unsqueeze(-1)], dim=-1)

    def state(
        self, link_pos_b: torch.Tensor, link_quat_b: torch.Tensor, gripper_joint: torch.Tensor, head_in_base: torch.Tensor
    ) -> torch.Tensor:
        tcp_pos_b, tcp_rot_b = self.tcp_from_link(link_pos_b, link_quat_b)
        return self.to_agnostic(tcp_pos_b, tcp_rot_b, self.opening_from_joint(gripper_joint), head_in_base)

    def ik_command(self, action: torch.Tensor, head_in_base: torch.Tensor) -> torch.Tensor:
        """10-D head-frame action -> SO-101 IK action ``[link_pos_b, link_quat_b_xyzw, closedness]``."""
        pos_b, rot_b = frames.apply_transform(head_in_base, action[..., :3], frames.matrix_from_rot6d(action[..., 3:9]))
        link_pos, link_quat = self.link_from_tcp(pos_b, rot_b)
        closedness = 1.0 - action[..., 9:10].clamp(0.0, 1.0)
        return torch.cat([link_pos, link_quat, closedness], dim=-1)
