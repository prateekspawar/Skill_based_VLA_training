"""GPU-batched scripted pick-and-place expert for ``SO101-PickPlace-Lang-IK-v0``.

Reads ground-truth cube / plate poses (robot base frame) from the ``privileged`` observation group and
emits ``[gripper_body_pos(3), gripper_body_quat_xyzw(4), gripper_closedness(1)]`` IK actions.

Grasps are top-down: the gripper approach axis (gripper-link -Z) points straight down and the
wrist roll aligns the jaws with the nearest cube face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import torch

import isaaclab.utils.math as math_utils

from so101_lang import language


class Phase(IntEnum):
    """Expert phases. APPROACH..LIFT execute the ``pick`` skill, CARRY..RETREAT the ``place`` skill."""

    APPROACH = 0
    DESCEND = 1
    CLOSE = 2
    LIFT = 3
    CARRY = 4
    LOWER = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8


def phase_skill(phase: int) -> str:
    return "pick" if phase <= Phase.LIFT else "place"


@dataclass
class ExpertCfg:
    # Grasp centre (between the jaw pads) in the gripper-link frame [m].
    tcp_offset: tuple[float, float, float] = (0.009, 0.0, -0.090)
    hover_height: float = 0.07  # above the object for approach / lift / carry [m]
    grasp_depth: float = 0.0  # TCP height relative to the cube centre at grasp [m]
    place_clearance: float = 0.012  # cube bottom above the plate top when releasing [m]
    max_speed: float = 0.20  # TCP target speed [m/s]
    reach_tol: float = 0.012  # position error that counts as "arrived" [m]
    phase_timeout: float = 4.0  # [s]
    close_time: float = 0.8  # [s]
    release_time: float = 0.6  # [s]
    open_cmd: float = 0.35  # gripper closedness while approaching (0 = fully open)
    randomize: bool = True  # jitter heights / speed per episode for demo diversity
    # DART-style noise injection: the executed command gets a smooth random offset while the recorded label
    # (cmd_pos) stays clean, so demonstrations contain recoveries from being slightly off the nominal path.
    dart_sigma_max: float = 0.0  # [m] per-episode sigma is drawn from U(0, dart_sigma_max); 0 disables
    dart_corr: float = 0.9  # per-step correlation of the Ornstein-Uhlenbeck offset


class PickPlaceExpert:
    def __init__(self, num_envs: int, dt: float, device: str, cfg: ExpertCfg | None = None):
        self.cfg = cfg or ExpertCfg()
        self.n, self.dt, self.device = num_envs, dt, device
        self.tcp_offset = torch.tensor(self.cfg.tcp_offset, device=device).expand(num_envs, 3)
        self.phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.timer = torch.zeros(num_envs, device=device)
        self.cmd_pos = torch.zeros(num_envs, 3, device=device)  # smoothed TCP target
        self.grasp_quat = torch.zeros(num_envs, 4, device=device)
        self.pick_pos = torch.zeros(num_envs, 3, device=device)  # cube centre, frozen once grasped
        self.hover = torch.full((num_envs,), self.cfg.hover_height, device=device)
        self.speed = torch.full((num_envs,), self.cfg.max_speed, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.dart_sigma = torch.zeros(num_envs, device=device)
        self.dart_offset = torch.zeros(num_envs, 3, device=device)

    def reset(self, env_ids: torch.Tensor | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = Phase.APPROACH
        self.timer[ids] = 0.0
        self.initialized[ids] = False
        k = self.phase[ids].numel()
        self.dart_offset[ids] = 0.0
        self.dart_sigma[ids] = torch.rand(k, device=self.device) * self.cfg.dart_sigma_max
        if self.cfg.randomize:
            self.hover[ids] = self.cfg.hover_height + (torch.rand(k, device=self.device) - 0.5) * 0.03
            self.speed[ids] = self.cfg.max_speed * (0.8 + 0.4 * torch.rand(k, device=self.device))

    # ------------------------------------------------------------------ geometry helpers
    @staticmethod
    def _top_down_quat(yaw: torch.Tensor) -> torch.Tensor:
        """Gripper-link orientation with z_g = +Z (approach -Z points down) and y_g at angle ``yaw``."""
        c, s = torch.cos(yaw), torch.sin(yaw)
        zeros, ones = torch.zeros_like(c), torch.ones_like(c)
        x_g = torch.stack([s, -c, zeros], dim=-1)
        y_g = torch.stack([c, s, zeros], dim=-1)
        z_g = torch.stack([zeros, zeros, ones], dim=-1)
        return math_utils.quat_from_matrix(torch.stack([x_g, y_g, z_g], dim=-1))

    def _grasp_yaw(self, cube_pos: torch.Tensor, cube_quat: torch.Tensor) -> torch.Tensor:
        """Jaw yaw aligned with a cube face, choosing the 90-degree branch closest to the natural wrist."""
        reach = torch.atan2(cube_pos[:, 1], cube_pos[:, 0])
        natural = reach + math.pi / 2  # y_g is perpendicular to the reach direction at the home pose
        _, _, cube_yaw = math_utils.euler_xyz_from_quat(cube_quat)
        k = torch.round((natural - cube_yaw) / (math.pi / 2))
        return cube_yaw + k * (math.pi / 2)

    # ------------------------------------------------------------------ main step
    def act(self, privileged: torch.Tensor, task: torch.Tensor, gripper_pos_b: torch.Tensor) -> torch.Tensor:
        """Compute the next IK action.

        Args:
            privileged: ``(N, 7 * (NUM_CUBES + NUM_PLATES))`` object poses in the base frame.
            task: ``(N, 2)`` ``[cube_idx, plate_idx]``.
            gripper_pos_b: ``(N, 7)`` current gripper-link pose in the base frame (pos, quat xyzw).
        """
        cfg, rows = self.cfg, torch.arange(self.n, device=self.device)
        poses = privileged.view(self.n, -1, 7)
        cube = poses[rows, task[:, 0]]
        plate = poses[rows, language.NUM_CUBES + task[:, 1]]

        cur_rot = math_utils.matrix_from_quat(gripper_pos_b[:, 3:7])
        tcp_now = gripper_pos_b[:, :3] + (cur_rot @ self.tcp_offset.unsqueeze(-1)).squeeze(-1)

        new = ~self.initialized
        if new.any():
            self.cmd_pos[new] = tcp_now[new]
            self.grasp_quat[new] = self._top_down_quat(self._grasp_yaw(cube[new, :3], cube[new, 3:7]))
            self.initialized[new] = True

        # Track the cube until the jaws close, then freeze (it moves with the gripper afterwards).
        tracking = (self.phase <= Phase.DESCEND).unsqueeze(-1)
        self.pick_pos = torch.where(tracking, cube[:, :3], self.pick_pos)

        # Goal TCP position and gripper command per phase.
        cube_top = self.pick_pos + torch.tensor([0.0, 0.0, cfg.grasp_depth], device=self.device)
        hover = self.hover.unsqueeze(-1) * torch.tensor([0.0, 0.0, 1.0], device=self.device)
        place = plate[:, :3].clone()
        place[:, 2] += 0.004 + cfg.place_clearance + 0.015  # plate half-height + clearance + cube half
        goal = torch.where(
            (self.phase == Phase.APPROACH).unsqueeze(-1),
            cube_top + hover,
            torch.zeros_like(cube_top),
        )
        goal = torch.where((self.phase == Phase.DESCEND).unsqueeze(-1) | (self.phase == Phase.CLOSE).unsqueeze(-1), cube_top, goal)
        goal = torch.where((self.phase == Phase.LIFT).unsqueeze(-1), cube_top + hover, goal)
        goal = torch.where((self.phase == Phase.CARRY).unsqueeze(-1), place + hover, goal)
        goal = torch.where(
            (self.phase == Phase.LOWER).unsqueeze(-1) | (self.phase == Phase.RELEASE).unsqueeze(-1), place, goal
        )
        goal = torch.where((self.phase >= Phase.RETREAT).unsqueeze(-1), place + hover, goal)

        closed = (self.phase >= Phase.CLOSE) & (self.phase <= Phase.LOWER)
        grip = torch.where(closed, torch.ones(self.n, device=self.device), torch.full((self.n,), cfg.open_cmd, device=self.device))
        grip = torch.where(self.phase >= Phase.RELEASE, torch.zeros_like(grip), grip)

        # Rate-limit the commanded TCP toward the goal.
        delta = goal - self.cmd_pos
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-9)
        step = (self.speed * self.dt).unsqueeze(-1)
        self.cmd_pos = torch.where(dist > step, self.cmd_pos + delta / dist * step, goal)

        # Phase transitions.
        self.timer += self.dt
        arrived = (torch.linalg.norm(goal - tcp_now, dim=-1) < cfg.reach_tol) & (dist.squeeze(-1) <= step.squeeze(-1))
        timed_out = self.timer > cfg.phase_timeout
        moving_phase = (
            (self.phase == Phase.APPROACH)
            | (self.phase == Phase.DESCEND)
            | (self.phase == Phase.LIFT)
            | (self.phase == Phase.CARRY)
            | (self.phase == Phase.LOWER)
            | (self.phase == Phase.RETREAT)
        )
        advance = moving_phase & (arrived | timed_out)
        self.last_timed_out = moving_phase & timed_out & ~arrived
        advance |= (self.phase == Phase.CLOSE) & (self.timer > cfg.close_time)
        advance |= (self.phase == Phase.RELEASE) & (self.timer > cfg.release_time)
        advance &= self.phase < Phase.DONE
        self.phase = torch.where(advance, self.phase + 1, self.phase)
        self.timer = torch.where(advance, torch.zeros_like(self.timer), self.timer)

        # Exposed for recording: the commanded TCP pose (base frame) and gripper closedness.
        self.cmd_grip = grip
        # DART: smooth offset on the *executed* target only; full in free-space phases, reduced near the
        # object, off while closing / releasing so grasps and drops stay clean.
        rho = self.cfg.dart_corr
        noise = torch.randn_like(self.dart_offset) * self.dart_sigma.unsqueeze(-1) * (1 - rho**2) ** 0.5
        self.dart_offset = rho * self.dart_offset + noise
        scale = torch.zeros(self.n, device=self.device)
        free = (self.phase == Phase.APPROACH) | (self.phase == Phase.LIFT) | (self.phase == Phase.CARRY)
        near = (self.phase == Phase.DESCEND) | (self.phase == Phase.LOWER)
        scale = torch.where(free, torch.ones_like(scale), scale)
        scale = torch.where(near, torch.full_like(scale, 0.3), scale)
        executed = self.cmd_pos + self.dart_offset * scale.unsqueeze(-1)

        # TCP target -> gripper-link target.
        goal_rot = math_utils.matrix_from_quat(self.grasp_quat)
        body_pos = executed - (goal_rot @ self.tcp_offset.unsqueeze(-1)).squeeze(-1)
        return torch.cat([body_pos, self.grasp_quat, grip.unsqueeze(-1)], dim=-1)
