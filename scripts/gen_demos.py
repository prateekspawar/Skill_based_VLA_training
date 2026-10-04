"""Generate language-labelled pick-and-place demonstrations with the scripted expert.

Only successful episodes are kept. The first step after a reset (stale camera render) and the
terminal step (its joint target is read after the auto-reset) are dropped. Each episode is one HDF5 group:

    /episode_000042
        attrs: instruction, cube_idx, plate_idx
        front_rgb  (T, H, W, 3) uint8      wrist_rgb (T, H, W, 3) uint8
        state      (T, 6) float32           joint positions [rad] before the step
        action     (T, 6) float32           joint position targets [rad] applied during the step
        init_poses (K, 7) float32           env-local cube + plate poses at reset (pos, quat xyzw)
    root attrs: fps, joint_names, object_names

    ~/isaacsim/bin/python scripts/gen_demos.py --num_envs 16 --episodes 200 --out datasets/raw/demos_000.h5
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=100, help="Number of successful episodes to save.")
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.seed = args.seed

with sim_app.launch(env_cfg, args):
    import random
    import time

    import h5py
    import numpy as np
    import torch

    import isaaclab.utils.math as math_utils

    from so101_lang import language
    from so101_lang.envs.mdp import CUBE_NAMES, PLATE_NAMES
    from so101_lang.expert.pick_place_sm import PickPlaceExpert

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    gripper_idx = robot.find_bodies("gripper")[0][0]
    objects = CUBE_NAMES + PLATE_NAMES

    def gripper_pose_b():
        pos, quat = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gripper_idx],
            robot.data.body_quat_w.torch[:, gripper_idx],
        )
        return torch.cat([pos, quat], dim=-1)

    def object_poses_local():
        poses = [
            torch.cat([env.scene[n].data.root_pos_w.torch - env.scene.env_origins, env.scene[n].data.root_quat_w.torch], -1)
            for n in objects
        ]
        return torch.stack(poses, dim=1).cpu().numpy().astype(np.float32)  # (N, K, 7)

    def rgb(cam):
        return env.scene[cam].data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()

    def new_buffers(i, init_poses):
        cube, plate = env.pp_task[i].tolist()
        return {
            "instruction": language.random_train_instruction(cube, plate),
            "task": (cube, plate),
            "init_poses": init_poses[i],
            "front": [], "wrist": [], "state": [], "action": [],
            "skip": True,
        }

    expert = PickPlaceExpert(env.num_envs, env.step_dt, env.device)
    expert.reset()
    init = object_poses_local()
    buffers = [new_buffers(i, init) for i in range(env.num_envs)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h5 = h5py.File(args.out, "w")
    h5.attrs["fps"] = round(1 / env.step_dt)
    h5.attrs["joint_names"] = robot.joint_names
    h5.attrs["object_names"] = objects
    saved, attempted, t0 = 0, 0, time.time()

    torch.inference_mode().__enter__()
    while saved < args.episodes:
        front, wrist = rgb("front_cam"), rgb("wrist_cam")
        state = obs["policy"]["joint_pos"].cpu().numpy()
        action = expert.act(obs["privileged"], env.pp_task, gripper_pose_b())
        obs, _, terminated, truncated, _ = env.step(action)
        target = robot.data.joint_pos_target.torch.cpu().numpy()
        for i, buf in enumerate(buffers):
            if buf.pop("skip", False):
                continue  # frame 0 after a reset still shows the previous layout (stale render)
            buf["front"].append(front[i])
            buf["wrist"].append(wrist[i])
            buf["state"].append(state[i])
            buf["action"].append(target[i])

        ended = (terminated | truncated).nonzero().flatten()
        if len(ended) == 0:
            continue
        success = env.termination_manager.get_term("success").cpu().numpy()
        init = object_poses_local()  # the ended envs have already been reset to their next layout
        for i in ended.tolist():
            attempted += 1
            buf = buffers[i]
            # The terminal step's target is read after the auto-reset (home pose), so drop that step.
            for key in ("front", "wrist", "state", "action"):
                buf[key] = buf[key][:-1]
            if success[i] and saved < args.episodes and len(buf["action"]) > 10:
                g = h5.create_group(f"episode_{saved:06d}")
                g.attrs["instruction"] = buf["instruction"]
                g.attrs["cube_idx"], g.attrs["plate_idx"] = buf["task"]
                for key, name in (("front", "front_rgb"), ("wrist", "wrist_rgb")):
                    arr = np.stack(buf[key])
                    g.create_dataset(name, data=arr, chunks=(1, *arr.shape[1:]), compression="gzip", compression_opts=4)
                g.create_dataset("state", data=np.stack(buf["state"]).astype(np.float32))
                g.create_dataset("action", data=np.stack(buf["action"]).astype(np.float32))
                g.create_dataset("init_poses", data=buf["init_poses"])
                saved += 1
                if saved % 10 == 0:
                    h5.flush()
                    rate = saved / (time.time() - t0) * 3600
                    print(f"[gen] saved {saved}/{args.episodes} (success {saved}/{attempted}), {rate:.0f} eps/h", flush=True)
            buffers[i] = new_buffers(i, init)
        expert.reset(ended)

    h5.close()
    print(f"[gen] done: {saved} episodes from {attempted} attempts -> {os.path.abspath(args.out)}")
    env.close()
