"""MDP terms for the language-conditioned SO-101 pick-and-place task.

The per-episode task (which cube goes on which plate) lives on the env as ``env.pp_task``,
a ``(num_envs, 2)`` long tensor ``[cube_idx, plate_idx]`` written by :func:`reset_layout_and_task`.
Scripts turn it into text with :mod:`so101_lang.language`.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from so101_lang import language

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

CUBE_NAMES = [f"cube_{i}" for i in range(language.NUM_CUBES)]
PLATE_NAMES = [f"plate_{i}" for i in range(language.NUM_PLATES)]


def _ensure_task_buffer(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "pp_task"):
        env.pp_task = torch.zeros(env.num_envs, 2, dtype=torch.long, device=env.device)
    return env.pp_task


def _sample_layout(radii: list[float], x_range, y_range, margin: float, tries: int = 2000) -> list[tuple[float, float]]:
    """Rejection-sample non-overlapping (x, y) centres for discs of the given radii."""
    for _ in range(tries):
        placed: list[tuple[float, float]] = []
        for r in radii:
            for _ in range(200):
                x, y = random.uniform(*x_range), random.uniform(*y_range)
                if all(math.dist((x, y), p) > r + radii[k] + margin for k, p in enumerate(placed)):
                    placed.append((x, y))
                    break
            else:
                break
        if len(placed) == len(radii):
            return placed
    raise RuntimeError("Could not sample a collision-free layout; widen the workspace or shrink objects.")


def reset_layout_and_task(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    cube_radius: float,
    plate_radius: float,
    cube_z: float,
    plate_z: float,
    margin: float = 0.01,
    pairs: list[tuple[int, int]] | None = None,
):
    """Scatter cubes and plates on the table without overlap, and pick a (cube, plate) task."""
    task = _ensure_task_buffer(env)
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if isinstance(env_ids, slice) else slice(None)]
    pairs = pairs or [(c, p) for c in range(language.NUM_CUBES) for p in range(language.NUM_PLATES)]
    names = PLATE_NAMES + CUBE_NAMES  # place the large plates first so sampling succeeds more often
    radii = [plate_radius] * len(PLATE_NAMES) + [cube_radius] * len(CUBE_NAMES)

    for env_id in env_ids.tolist():
        layout = _sample_layout(radii, x_range, y_range, margin)
        ids = torch.tensor([env_id], device=env.device)
        origin = env.scene.env_origins[env_id]
        for name, (x, y) in zip(names, layout):
            is_cube = name.startswith("cube")
            yaw = random.uniform(-math.pi / 4, math.pi / 4) if is_cube else 0.0
            pos = torch.tensor([[x, y, cube_z if is_cube else plate_z]], device=env.device) + origin
            zero = torch.zeros(1, device=env.device)
            quat = math_utils.quat_from_euler_xyz(zero, zero, torch.full((1,), yaw, device=env.device))
            asset = env.scene[name]
            asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=ids)
            asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
        task[env_id] = torch.tensor(random.choice(pairs), device=env.device)


##
# Observations
##


def task_onehot(env: ManagerBasedEnv) -> torch.Tensor:
    """One-hot ``[cube (NUM_CUBES), plate (NUM_PLATES)]`` encoding of the episode's task."""
    task = _ensure_task_buffer(env)
    cube = torch.nn.functional.one_hot(task[:, 0], language.NUM_CUBES)
    plate = torch.nn.functional.one_hot(task[:, 1], language.NUM_PLATES)
    return torch.cat([cube, plate], dim=-1).float()


def _positions_w(env: ManagerBasedEnv, names: list[str]) -> torch.Tensor:
    return torch.stack([env.scene[n].data.root_pos_w.torch for n in names], dim=1)  # (N, K, 3)


def object_poses_b(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Cube and plate poses in the robot base frame, flattened as ``[cubes..., plates...] x (pos3, quat4)``."""
    robot = env.scene[robot_cfg.name]
    root_pos = robot.data.root_pos_w.torch
    root_quat = robot.data.root_quat_w.torch
    out = []
    for name in CUBE_NAMES + PLATE_NAMES:
        obj = env.scene[name]
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            root_pos, root_quat, obj.data.root_pos_w.torch, obj.data.root_quat_w.torch
        )
        out.append(torch.cat([pos_b, quat_b], dim=-1))
    return torch.cat(out, dim=-1)


##
# Terminations
##


def target_on_plate(
    env: ManagerBasedRLEnv,
    plate_radius: float,
    plate_top_z: float,
    cube_half: float,
    gripper_open_threshold: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Target cube resting on the target plate, gripper released, cube nearly still."""
    task = _ensure_task_buffer(env)
    rows = torch.arange(env.num_envs, device=env.device)
    cubes = _positions_w(env, CUBE_NAMES)[rows, task[:, 0]] - env.scene.env_origins
    plates = _positions_w(env, PLATE_NAMES)[rows, task[:, 1]] - env.scene.env_origins
    cube_vel = torch.stack([env.scene[n].data.root_lin_vel_w.torch for n in CUBE_NAMES], dim=1)[rows, task[:, 0]]

    xy_ok = torch.linalg.norm(cubes[:, :2] - plates[:, :2], dim=-1) < plate_radius - 0.01
    rest_z = plate_top_z + cube_half
    z_ok = (cubes[:, 2] > rest_z - 0.008) & (cubes[:, 2] < rest_z + 0.012)
    still = torch.linalg.norm(cube_vel, dim=-1) < 0.05
    robot = env.scene[robot_cfg.name]
    released = robot.data.joint_pos.torch[:, robot_cfg.joint_ids[0]] > gripper_open_threshold
    return xy_ok & z_ok & still & released


def any_cube_dropped(env: ManagerBasedRLEnv, minimum_height: float = -0.05) -> torch.Tensor:
    z = _positions_w(env, CUBE_NAMES)[..., 2] - env.scene.env_origins[:, 2:3]
    return (z < minimum_height).any(dim=-1)


def randomize_plate_colors(env: ManagerBasedEnv, env_ids: torch.Tensor, palette: list[str]):
    """Give each plate of the reset envs a colour from ``palette`` (two distinct per env) and record the names in
    ``env.pp_plate_colors`` (list per env), so instructions and perception queries can name them.

    Uses its own RNG (seeded from the env seed), so layouts and tasks stay identical to runs without it."""
    from pxr import Gf, UsdShade

    import omni.usd

    if not hasattr(env, "pp_plate_colors"):
        env.pp_plate_colors = [list(language.PLATE_NAMES) for _ in range(env.num_envs)]
        env._plate_color_rng = random.Random(1_000_003 * (env.cfg.seed or 0) + 17)
        env._plate_shaders = {}
    stage = omni.usd.get_context().get_stage()
    ids = range(env.num_envs) if env_ids is None or isinstance(env_ids, slice) else env_ids.tolist()
    for i in ids:
        names = env._plate_color_rng.sample(palette, language.NUM_PLATES)
        env.pp_plate_colors[i] = names
        for j, c in enumerate(names):
            key = (i, j)
            if key not in env._plate_shaders:
                prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/Plate_{j}/geometry/material/Shader")
                env._plate_shaders[key] = UsdShade.Shader(prim)
            env._plate_shaders[key].GetInput("diffuseColor").Set(Gf.Vec3f(*language._PALETTE[c]))
