"""Measure stereo-depth accuracy against the simulator's ground-truth depth and save a visual grid.

    ~/isaacsim/bin/python scripts/check_stereo_depth.py --frames 8
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", type=int, default=8, help="Number of frames to evaluate (spread over episodes).")
parser.add_argument("--out", default="outputs/stereo_depth")
parser.add_argument("--dump", default=None, help="Also save left/right/GT frames to this .npz for offline evaluation.")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.scene.left_cam.data_types = ["rgb", "distance_to_image_plane"]

with sim_app.launch(env_cfg, args):
    import json

    import numpy as np
    import torch
    from PIL import Image

    import isaaclab.utils.math as math_utils

    from so101_lang.envs.head import calibration_dict, head_in_base
    from so101_lang.expert.pick_place_sm import PickPlaceExpert
    from so101_lang.perception.stereo_depth import StereoDepth, StereoDepthCfg, depth_to_rgb

    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    gi = robot.find_bodies("gripper")[0][0]
    expert = PickPlaceExpert(env.num_envs, env.step_dt, env.device)
    expert.reset()
    calib = calibration_dict(head_in_base(env)[0])
    stereo = StereoDepth(StereoDepthCfg.from_calibration(calib))

    rows, errs, dumps = [], [], []
    step, sample_every = 0, 23
    torch.inference_mode().__enter__()
    while len(rows) < args.frames:
        pos, quat = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi],
        )
        action = expert.act(obs["privileged"], env.pp_task, torch.cat([pos, quat], -1))
        obs, _, term, trunc, _ = env.step(action)
        if (term | trunc).any():
            expert.reset((term | trunc).nonzero().flatten())
            continue
        step += 1
        if step % sample_every:
            continue
        left = env.scene["left_cam"].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
        right = env.scene["right_cam"].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()
        gt = env.scene["left_cam"].data.output["distance_to_image_plane"][0, ..., 0].cpu().numpy()
        dumps.append((left, right, gt))
        est = stereo.depth(left, right)
        roi = np.isfinite(gt) & (gt < 1.0)  # table + objects + arm; excludes far background
        valid = roi & np.isfinite(est)
        err = np.abs(est - gt)[valid]
        errs.append(
            {
                "coverage": valid.sum() / roi.sum(),
                "mae_mm": 1000 * err.mean(),
                "median_mm": 1000 * np.median(err),
                "within_1cm": (err < 0.01).mean(),
                "within_2cm": (err < 0.02).mean(),
            }
        )
        err_img = np.zeros_like(gt)
        err_img[valid] = np.abs(est - gt)[valid]
        err_rgb = (np.clip(err_img / 0.03, 0, 1)[..., None] * np.array([255, 255, 255])).astype(np.uint8)
        rows.append(np.concatenate([left, depth_to_rgb(gt, 0.25, 0.85), depth_to_rgb(est, 0.25, 0.85), err_rgb], 1))

    os.makedirs(args.out, exist_ok=True)
    Image.fromarray(np.concatenate(rows[:4], 0)).save(f"{args.out}/grid.png")
    if args.dump:
        np.savez_compressed(args.dump, left=np.stack([d[0] for d in dumps]), right=np.stack([d[1] for d in dumps]),
                            gt=np.stack([d[2] for d in dumps]), calibration=json.dumps(calib))
    keys = errs[0].keys()
    mean = {k: float(np.mean([e[k] for e in errs])) for k in keys}
    print("[stereo] columns: left | GT depth | stereo depth | |error| (white = 3 cm)")
    print("[stereo] " + ", ".join(f"{k}={v:.3f}" for k, v in mean.items()))
    env.close()
