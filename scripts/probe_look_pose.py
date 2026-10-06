"""Find an arm "look" pose that keeps the stereo head's view of the workspace clear (isaacsim env).

Drives the IK env to a few candidate TCP poses (robot base frame), lets the arm settle, saves the left image
and measures how many workspace pixels are covered by the arm (stereo depth much closer than the table).

    ~/isaacsim/bin/python scripts/probe_look_pose.py
"""

import argparse

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", default="outputs/look_pose")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.scene.left_cam.data_types = ["rgb", "distance_to_image_plane"]

# Candidate TCP targets in the robot base frame (workspace lies along -y_base; +z up).
CANDIDATES = {
    "home": None,
    "up_back": (0.00, -0.06, 0.30),
    "right_high": (0.16, -0.06, 0.22),
    "right_low": (0.18, -0.02, 0.12),
    "back_low": (0.00, 0.06, 0.18),
    "right_back": (0.14, 0.06, 0.16),
}

with sim_app.launch(env_cfg, args):
    import os

    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.utils.math as math_utils

    from so101_lang.envs.head import head_in_base
    from so101_lang.robots.so101 import SO101Adapter

    env = sim_app.make_env(TASK, env_cfg)
    adapter = SO101Adapter()
    robot = env.scene["robot"]
    gi = robot.find_bodies("gripper")[0][0]
    os.makedirs(args.out, exist_ok=True)
    torch.inference_mode().__enter__()

    def link_b():
        return math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi])

    rows = []
    for name, target in CANDIDATES.items():
        env.reset()
        pos, quat = link_b()
        hold = torch.cat([pos, quat, torch.zeros(1, 1, device=env.device)], -1)
        for _ in range(3):
            env.step(hold)
        # Clear-view reference: depth of the empty scene is not available, so use the table plane:
        # pixels whose depth is > 4 cm closer than the farthest-plausible table depth along that ray count as arm.
        if target is not None:
            tcp_goal = torch.tensor([target], device=env.device)
            rot = math_utils.matrix_from_quat(quat)
            link_goal = tcp_goal - (rot @ torch.tensor(adapter.tcp_offset, device=env.device).unsqueeze(-1)).squeeze(-1)
            cur = pos.clone()
            for k in range(60):  # move the link target over 3 s, then settle
                a = min((k + 1) / 40, 1.0)
                cmd = torch.cat([cur + (link_goal - cur) * a, quat, torch.zeros(1, 1, device=env.device)], -1)
                env.step(cmd)
        rgb = env.scene["left_cam"].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
        seg_depth = env.scene["left_cam"].data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        Image.fromarray(rgb).save(f"{args.out}/{name}.png")
        # Robot pixels: yellow-ish hue with high saturation (the SO-101 is yellow); cheap but adequate here.
        r, g, b = [rgb[..., c].astype(int) for c in range(3)]
        yellow = (r > 150) & (g > 140) & (b < 120) & (np.abs(r - g) < 60)
        ws = np.zeros_like(yellow)
        ws[150:512, 0:420] = True  # rough workspace region of the image
        pos, _ = link_b()
        rows.append((name, float(yellow[ws].mean()), pos[0].cpu().numpy().round(3).tolist()))
    for name, cover, p in rows:
        print(f"[look] {name:11s} arm covers {100 * cover:5.1f}% of the workspace region | gripper link at {p}")
    env.close()
