"""Generate skill-labelled pick-and-place demonstrations with the scripted expert.

Observations come from the stereo head; state and action are robot-agnostic 10-D end-effector vectors
in the head frame (see :mod:`so101_lang.robots.so101`). Every frame is labelled with the skill being
executed (``pick`` until the cube is lifted, then ``place``) and that skill's natural-language step.

Only successful episodes are kept; the first step after a reset (stale camera render) is dropped.

    /episode_000042
        attrs: goal, cube_idx, plate_idx, pick_text, place_text
        left_rgb, right_rgb (T, H, W, 3) uint8
        state     (T, 10) float32   measured TCP pose (head frame, pos + rot6d) + gripper opening
        action    (T, 10) float32   clean expert TCP command (head frame) + gripper opening; with DART the
                                    executed command carries a smooth random offset, the label does not
        joint_pos (T, 6)  float32   SO-101 joint angles [rad] (reference only)
        skill     (T,)    uint8     0 = pick, 1 = place
        init_poses (K, 7) float32   env-local cube + plate poses at reset (pos, quat xyzw)
    root attrs: fps, state_names, skills, calibration (JSON)

    ~/isaacsim/bin/python scripts/gen_demos.py --num_envs 16 --episodes 2000 --out datasets/raw/stereo_ee.h5
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=100, help="Number of successful episodes to save.")
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dart_sigma", type=float, default=0.012, help="Max per-episode DART noise sigma [m]; 0 disables.")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.seed = args.seed

with sim_app.launch(env_cfg, args):
    import json
    import random
    import time

    import h5py
    import numpy as np
    import torch

    import isaaclab.utils.math as math_utils

    from so101_lang import language
    from so101_lang.envs.head import calibration_dict, head_in_base
    from so101_lang.envs.mdp import CUBE_NAMES, PLATE_NAMES
    from so101_lang.expert.pick_place_sm import ExpertCfg, PickPlaceExpert, phase_skill
    from so101_lang.robots import frames
    from so101_lang.robots.so101 import STATE_NAMES, SO101Adapter

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    gripper_idx = robot.find_bodies("gripper")[0][0]
    objects = CUBE_NAMES + PLATE_NAMES
    adapter = SO101Adapter()
    skill_ids = {name: i for i, name in enumerate(language.SKILLS)}

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
            "task": (cube, plate),
            "goal": language.goal_instruction(cube, plate),
            "pick_text": language.random_skill_text("pick", object=language.cube_ref(cube)),
            "place_text": language.random_skill_text("place", target=language.plate_ref(plate)),
            "init_poses": init_poses[i],
            "left": [], "right": [], "state": [], "action": [], "joint_pos": [], "skill": [],
            "skip": True,
        }

    expert = PickPlaceExpert(env.num_envs, env.step_dt, env.device, ExpertCfg(dart_sigma_max=args.dart_sigma))
    expert.reset()
    init = object_poses_local()
    buffers = [new_buffers(i, init) for i in range(env.num_envs)]
    head_b = head_in_base(env)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h5 = h5py.File(args.out, "w")
    h5.attrs["fps"] = round(1 / env.step_dt)
    h5.attrs["state_names"] = STATE_NAMES
    h5.attrs["skills"] = list(language.SKILLS)
    h5.attrs["calibration"] = json.dumps(calibration_dict(head_b[0]))
    saved, attempted, t0 = 0, 0, time.time()

    torch.inference_mode().__enter__()
    while saved < args.episodes:
        left, right = rgb("left_cam"), rgb("right_cam")
        link = gripper_pose_b()
        joint_pos = robot.data.joint_pos.torch
        state = adapter.state(link[:, :3], link[:, 3:7], joint_pos[:, 5], head_b).cpu().numpy()
        phase = expert.phase.clone()
        ik_action = expert.act(obs["privileged"], env.pp_task, link)
        # Record the expert's command in the robot-agnostic form: commanded TCP pose + opening.
        cmd = adapter.to_agnostic(
            expert.cmd_pos, frames.quat_to_matrix(expert.grasp_quat), 1.0 - expert.cmd_grip, head_b
        ).cpu().numpy()
        joint_np = joint_pos.cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(ik_action)
        for i, buf in enumerate(buffers):
            if buf.pop("skip", False):
                continue  # frame 0 after a reset still shows the previous layout (stale render)
            buf["left"].append(left[i])
            buf["right"].append(right[i])
            buf["state"].append(state[i])
            buf["action"].append(cmd[i])
            buf["joint_pos"].append(joint_np[i])
            buf["skill"].append(skill_ids[phase_skill(int(phase[i]))])

        ended = (terminated | truncated).nonzero().flatten()
        if len(ended) == 0:
            continue
        success = env.termination_manager.get_term("success").cpu().numpy()
        init = object_poses_local()  # the ended envs have already been reset to their next layout
        for i in ended.tolist():
            attempted += 1
            buf = buffers[i]
            if success[i] and saved < args.episodes and len(buf["action"]) > 10:
                g = h5.create_group(f"episode_{saved:06d}")
                for key in ("goal", "pick_text", "place_text"):
                    g.attrs[key] = buf[key]
                g.attrs["cube_idx"], g.attrs["plate_idx"] = buf["task"]
                for key, name in (("left", "left_rgb"), ("right", "right_rgb")):
                    arr = np.stack(buf[key])
                    g.create_dataset(name, data=arr, chunks=(1, *arr.shape[1:]), compression="gzip", compression_opts=4)
                for key in ("state", "action", "joint_pos"):
                    g.create_dataset(key, data=np.stack(buf[key]).astype(np.float32))
                g.create_dataset("skill", data=np.asarray(buf["skill"], dtype=np.uint8))
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
