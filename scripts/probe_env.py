"""Load the pick-and-place scene, hold the arm still, report frames and save camera images.

    ~/isaacsim/bin/python scripts/probe_env.py [--viz kit]
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/probe")
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--geometry", action="store_true", help="Print gripper/jaw collision boxes in the gripper frame.")
args, env_cfg = sim_app.parse(parser, TASK)

with sim_app.launch(env_cfg, args):
    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.utils.math as math_utils

    from so101_lang import language

    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    root_pos, root_quat = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch

    def body_pose_b(name):
        idx = robot.find_bodies(name)[0][0]
        return math_utils.subtract_frame_transforms(
            root_pos, root_quat, robot.data.body_pos_w.torch[:, idx], robot.data.body_quat_w.torch[:, idx]
        )

    print("bodies:", robot.body_names)
    print("joints:", robot.joint_names)
    print("joint limits:", robot.data.joint_pos_limits.torch[0].cpu().numpy().round(3).tolist())

    # Hold the current gripper pose with the gripper open.
    g_pos, g_quat = body_pose_b("gripper")
    action = torch.cat([g_pos, g_quat, torch.zeros(env.num_envs, 1, device=env.device)], dim=-1)
    for _ in range(args.steps):
        obs, *_ = env.step(action)

    for name in ["gripper", "moving_jaw_so101_v1"]:
        p, q = body_pose_b(name)
        rot = math_utils.matrix_from_quat(q)[0].cpu().numpy()
        print(f"{name}: pos_b={p[0].cpu().numpy().round(4)} quat_b(xyzw)={q[0].cpu().numpy().round(4)}")
        print(f"   axes in base frame (columns x,y,z):\n{rot.round(3)}")
    print("joint_pos:", obs["policy"]["joint_pos"][0].cpu().numpy().round(3))
    task = env.pp_task[0].tolist()
    print("task:", task, "->", language.goal_instruction(*task))
    objs = obs["privileged"][0].view(-1, 7).cpu().numpy().round(3)
    for name, row in zip(
        [f"{c} cube" for c in language.CUBE_NAMES] + [f"{p} plate" for p in language.PLATE_NAMES], objs
    ):
        print(f"  {name:12s} pos_b={row[:3]}")

    # Geometry of the gripper and moving jaw, expressed in the gripper-link frame (open, then closed).
    from pxr import Gf, Usd, UsdGeom

    import isaaclab.sim as sim_utils

    def report_geometry(label):
        stage = sim_utils.get_current_stage()
        cache = UsdGeom.XformCache()
        g_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper")
        g_inv = cache.GetLocalToWorldTransform(g_prim).GetInverse()
        for body in ["gripper", "moving_jaw_so101_v1"]:
            root = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{body}")
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Gprim):
                    continue
                if "collision" not in str(prim.GetPath()).lower():
                    continue
                pts = UsdGeom.Mesh(prim).GetPointsAttr().Get() if prim.IsA(UsdGeom.Mesh) else None
                if not pts:
                    continue
                m = cache.GetLocalToWorldTransform(prim) * g_inv
                arr = np.array([list(m.Transform(Gf.Vec3d(*p))) for p in pts])
                print(f"  [{label}] {str(prim.GetPath()).split('Robot/')[1]}: min={arr.min(0).round(4)} max={arr.max(0).round(4)}")

    if args.geometry:
        # Note: with Fabric enabled these are the USD-authored (joint = 0) poses, not the live ones.
        report_geometry("usd")

    os.makedirs(args.out, exist_ok=True)
    for cam in ["left_cam", "right_cam"]:
        rgb = env.scene[cam].data.output["rgb"][0].cpu().numpy()
        Image.fromarray(rgb[..., :3].astype(np.uint8)).save(f"{args.out}/{cam}.png")
    print("saved images to", os.path.abspath(args.out))
    env.close()
