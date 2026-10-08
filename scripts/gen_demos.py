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
        cube_pos, plate_pos (T, 3)  the target cube's / plate's centre per frame, head frame
        attrs moved_pick_mm / moved_place_mm: slide distances (0 if that object did not move)
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
parser.add_argument("--move_prob_pick", type=float, default=0.0,
                    help="Probability that the target cube slides 3-6 cm during the pick approach.")
parser.add_argument("--move_prob_place", type=float, default=0.0,
                    help="Probability that the target plate slides 3-6 cm while the cube is carried.")
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
    from so101_lang.envs.pick_place_env_cfg import PLATE_RADIUS, WORKSPACE_X, WORKSPACE_Y
    from so101_lang.expert.pick_place_sm import Phase
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

    SLIDE_TIME = 0.5  # [s]
    slide_steps = round(SLIDE_TIME / env.step_dt)
    rng = np.random.default_rng(args.seed)

    def plan_slide(i, name, clearance):
        """Free destination 3-6 cm away inside the workspace, collision-free path; returns (start, delta) or None."""
        local = object_poses_local()[i]
        k = objects.index(name)
        pos = local[k, :3]
        # Objects missing from ``clearance`` are ignored (e.g. the cube being carried over a sliding plate).
        others = [(local[j, :2], clearance[objects[j]]) for j in range(len(objects)) if j != k and objects[j] in clearance]
        for _ in range(100):
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.06)
            delta = dist * np.array([np.cos(ang), np.sin(ang)])
            new = pos[:2] + delta
            if not (WORKSPACE_X[0] <= new[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= new[1] <= WORKSPACE_Y[1]):
                continue
            path = [pos[:2] + delta * f / 5 for f in range(6)]
            if all(np.linalg.norm(pt - o) > c for pt in path for o, c in others):
                return pos.copy(), delta
        return None

    def drive_slide(i, name, start, delta, k):
        """Move object ``name`` of env ``i`` to step ``k`` of its slide (continuous motion, pose written per step)."""
        obj = env.scene[name]
        p = np.array([start[0] + delta[0] * k / slide_steps, start[1] + delta[1] * k / slide_steps, start[2]])
        pos = torch.tensor(p, dtype=torch.float32, device=env.device).unsqueeze(0) + env.scene.env_origins[i]
        quat = obj.data.root_quat_w.torch[i : i + 1]
        ids = torch.tensor([i], device=env.device)
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1).float(), env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)

    def plate_colors(i):
        """This episode's plate colour names (per-episode randomisation) or None for the fixed colours."""
        return env.pp_plate_colors[i] if hasattr(env, "pp_plate_colors") else None

    def new_buffers(i, init_poses):
        cube, plate = env.pp_task[i].tolist()
        return {
            "move": {
                "pick": rng.random() < args.move_prob_pick, "place": rng.random() < args.move_prob_place,
                "pick_at": rng.uniform(0.3, 1.0), "place_at": rng.uniform(0.1, 0.6),
                "active": None, "moved_pick_mm": 0.0, "moved_place_mm": 0.0,
            },
            "steps": 0, "carry_steps": 0, "cube_pos": [], "plate_pos": [],
            "task": (cube, plate),
            "goal": language.goal_instruction(cube, plate, plate_colors(i)),
            "plate_colors": list(plate_colors(i) or language.PLATE_NAMES),
            "pick_text": language.random_skill_text("pick", object=language.cube_ref(cube)),
            "place_text": language.random_skill_text("place", target=language.plate_ref(plate, plate_colors(i))),
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

    clearance = {n: 0.045 for n in CUBE_NAMES} | {n: PLATE_RADIUS + 0.03 for n in PLATE_NAMES}
    plate_clearance = {n: PLATE_RADIUS + 0.03 for n in CUBE_NAMES} | {n: 2 * PLATE_RADIUS + 0.01 for n in PLATE_NAMES}

    torch.inference_mode().__enter__()
    while saved < args.episodes:
        # Moving targets: trigger and advance slides (cube during the approach, plate during the carry).
        link = gripper_pose_b()
        tcp_b, _ = adapter.tcp_from_link(link[:, :3], link[:, 3:7])
        priv = obs["privileged"].view(env.num_envs, -1, 7)
        for i, buf in enumerate(buffers):
            mv, cube, plate = buf["move"], *buf["task"]
            ph = int(expert.phase[i])
            if mv["active"] is None:
                cube_b, plate_b = priv[i, cube, :3], priv[i, language.NUM_CUBES + plate, :3]
                if (mv["pick"] and ph == Phase.APPROACH and buf["steps"] * env.step_dt >= mv["pick_at"]
                        and float(torch.linalg.norm(tcp_b[i] - cube_b)) > 0.06):
                    mv["pick"] = False
                    plan = plan_slide(i, CUBE_NAMES[cube], {k: v for k, v in clearance.items()})
                    if plan:
                        mv["active"] = {"name": CUBE_NAMES[cube], "start": plan[0], "delta": plan[1], "k": 0, "skill": "pick"}
                elif (mv["place"] and ph == Phase.CARRY and buf["carry_steps"] * env.step_dt >= mv["place_at"]
                        and float(torch.linalg.norm(tcp_b[i, :2] - plate_b[:2])) > 0.06):
                    mv["place"] = False
                    plan = plan_slide(i, PLATE_NAMES[plate], {k: v for k, v in plate_clearance.items()
                                                              if k != CUBE_NAMES[cube]})
                    if plan:
                        mv["active"] = {"name": PLATE_NAMES[plate], "start": plan[0], "delta": plan[1], "k": 0, "skill": "place"}
            if mv["active"] is not None:
                a = mv["active"]
                a["k"] += 1
                drive_slide(i, a["name"], a["start"], a["delta"], a["k"])
                if a["k"] >= slide_steps:
                    mv[f"moved_{a['skill']}_mm"] = 1000 * float(np.linalg.norm(a["delta"]))
                    mv["active"] = None
            buf["steps"] += 1
            buf["carry_steps"] += int(ph == Phase.CARRY)

        left, right = rgb("left_cam"), rgb("right_cam")
        joint_pos = robot.data.joint_pos.torch
        state = adapter.state(link[:, :3], link[:, 3:7], joint_pos[:, 5], head_b).cpu().numpy()
        phase = expert.phase.clone()
        ik_action = expert.act(obs["privileged"], env.pp_task, link)
        # Record the expert's command in the robot-agnostic form: commanded TCP pose + opening.
        cmd = adapter.to_agnostic(
            expert.cmd_pos, frames.quat_to_matrix(expert.grasp_quat), 1.0 - expert.cmd_grip, head_b
        ).cpu().numpy()
        joint_np = joint_pos.cpu().numpy()
        # Target positions per frame (head frame), from the same observation the images belong to.
        base_in_head = frames.invert_transform(head_b)
        objs_h = (priv[..., :3] @ base_in_head[:, :3, :3].transpose(1, 2) + base_in_head[:, None, :3, 3]).cpu().numpy()
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
            cube, plate = buf["task"]
            buf["cube_pos"].append(objs_h[i, cube])
            buf["plate_pos"].append(objs_h[i, language.NUM_CUBES + plate])

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
                g.attrs["plate_colors"] = buf["plate_colors"]
                for key, name in (("left", "left_rgb"), ("right", "right_rgb")):
                    arr = np.stack(buf[key])
                    g.create_dataset(name, data=arr, chunks=(1, *arr.shape[1:]), compression="gzip", compression_opts=4)
                for key in ("state", "action", "joint_pos"):
                    g.create_dataset(key, data=np.stack(buf[key]).astype(np.float32))
                g.create_dataset("skill", data=np.asarray(buf["skill"], dtype=np.uint8))
                g.create_dataset("init_poses", data=buf["init_poses"])
                g.create_dataset("cube_pos", data=np.stack(buf["cube_pos"]).astype(np.float32))
                g.create_dataset("plate_pos", data=np.stack(buf["plate_pos"]).astype(np.float32))
                g.attrs["moved_pick_mm"] = buf["move"]["moved_pick_mm"]
                g.attrs["moved_place_mm"] = buf["move"]["moved_place_mm"]
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
