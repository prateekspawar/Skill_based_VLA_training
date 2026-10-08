"""Generate find() demonstrations: move the arm until the named object is clearly visible, then hold still.

Each episode: a random layout; the arm starts at home (or, with --random_start_prob, first moves unrecorded to
a random pose above the table, which often hides objects). The target is any cube or plate. Visibility is
measured from the simulator's depth image: 9 points on the target's top surface are projected into the left
camera and count as visible when nothing is in front of them. The expert holds still once >= --visible_frac
of them are visible for 0.3 s (then records 0.5 s of holding and ends); otherwise it moves the TCP toward a
clear "look" pose (with DART noise on the executed command) until the target becomes visible.

With --absent_prob p, a fraction of episodes are failures: the queried object is parked out of the camera's view
(behind the head, pose rewritten every step), the expert moves to the look pose, settles for 0.3 s, declares failure
and holds still for 0.5 s.

Output format matches gen_demos.py (skill index 2 = find), plus per episode:
    attrs: find_text, object_name, fail_mode ("" or "absent")
    datasets: target_pose9 (9,) head-frame pos + rot6d (zeros if absent), visible_frac (T,),
              status (T, 2) (done, success): last 5 hold frames [1, 1] on success, the failure hold [1, 0]

    ~/isaacsim/bin/python scripts/gen_find_demos.py --num_envs 16 --episodes 1000 --out datasets/raw/find_1000.h5
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--random_start_prob", type=float, default=0.5)
parser.add_argument("--visible_frac", type=float, default=0.75)
parser.add_argument("--dart_sigma", type=float, default=0.008)
parser.add_argument("--absent_prob", type=float, default=0.0, help="Fraction of target-absent (failure) episodes.")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.seed = args.seed
env_cfg.scene.left_cam.data_types = ["rgb", "distance_to_image_plane"]
env_cfg.episode_length_s = 60.0  # episodes are ended by this script, not by the env

# TCP targets in the robot base frame.
LOOK_POSE = (0.00, -0.06, 0.30)  # arm raised and pulled back: clears the head camera's view (probe_look_pose.py)
START_BOX = ((-0.10, 0.12), (-0.26, -0.10), (0.06, 0.20))  # random start poses above the workspace
HIDE_POS = (-0.45, 0.0, 0.60)  # env-local spot behind the head camera where an absent object is parked

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
    from so101_lang.robots import frames
    from so101_lang.robots.so101 import STATE_NAMES, SO101Adapter

    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    N, dt, dev = env.num_envs, env.step_dt, env.device
    robot = env.scene["robot"]
    gi = robot.find_bodies("gripper")[0][0]
    adapter = SO101Adapter()
    head_b = head_in_base(env)
    calib = calibration_dict(head_b[0])
    K = calib["intrinsics"]
    cam_off = np.asarray(calib["cameras"]["left"]["position_in_head"])
    base_in_head = frames.invert_transform(head_b)[0].cpu().numpy()
    objects = CUBE_NAMES + PLATE_NAMES
    refs = language.OBJECT_REFS + language.TARGET_REFS
    FIND = list(language.SKILLS).index("find")

    # Sample points on each object's top surface, in the object frame.
    g = np.array([-0.01, 0.0, 0.01])
    CUBE_PTS = np.array([[x, y, 0.015] for x in g for y in g])
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    PLATE_PTS = np.vstack([[0, 0, 0.004], np.stack([0.03 * np.cos(ang), 0.03 * np.sin(ang), np.full(8, 0.004)], 1)])

    def link_pose_b():
        p, q = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi])
        return p, q

    def obj_pose_b(i, k):
        p, q = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch[i : i + 1], robot.data.root_quat_w.torch[i : i + 1],
            env.scene[objects[k]].data.root_pos_w.torch[i : i + 1], env.scene[objects[k]].data.root_quat_w.torch[i : i + 1])
        return p[0].cpu().numpy(), math_utils.matrix_from_quat(q)[0].cpu().numpy()

    def visible_fraction(i, k, depth_img):
        pos_b, rot_b = obj_pose_b(i, k)
        pts_b = (rot_b @ (CUBE_PTS if k < len(CUBE_NAMES) else PLATE_PTS).T).T + pos_b
        pts_h = pts_b @ base_in_head[:3, :3].T + base_in_head[:3, 3]
        q = pts_h - cam_off
        z = q[:, 0]
        u = K["cx"] + K["fx"] * (-q[:, 1]) / z
        v = K["cy"] + K["fy"] * (-q[:, 2]) / z
        ok = (u >= 0) & (u < K["width"]) & (v >= 0) & (v < K["height"]) & (z > 0)
        vis = np.zeros(len(z), dtype=bool)
        ui, vi = u[ok].astype(int), v[ok].astype(int)
        vis[ok] = depth_img[vi, ui] >= z[ok] - 0.008
        return float(vis.mean())

    def target_pose9(i, k):
        pos_b, rot_b = obj_pose_b(i, k)
        if k >= len(CUBE_NAMES):
            rot_b = np.eye(3)  # plates are rotationally symmetric
        pos_h = base_in_head[:3, :3] @ pos_b + base_in_head[:3, 3]
        rot_h = base_in_head[:3, :3] @ rot_b
        return np.concatenate([pos_h, rot_h[:, 0], rot_h[:, 1]]).astype(np.float32)

    def rgb(cam):
        return env.scene[cam].data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()

    def episode_refs(i):
        """Object names for env i: cubes fixed, plates with this episode's colours when randomised."""
        if hasattr(env, "pp_plate_colors"):
            return language.OBJECT_REFS + [language.plate_ref(j, env.pp_plate_colors[i]) for j in range(language.NUM_PLATES)]
        return refs

    def new_episode(i):
        k = int(rng.integers(len(objects)))
        absent = bool(rng.random() < args.absent_prob)
        start = None
        if rng.random() < args.random_start_prob:
            start = np.array([rng.uniform(*START_BOX[0]), rng.uniform(*START_BOX[1]), rng.uniform(*START_BOX[2])])
        names = episode_refs(i)
        return {"k": k, "name": names[k], "text": language.random_skill_text("find", object=names[k]), "start": start,
                "absent": absent, "fail_at": None,
                "phase": "pre" if start is not None else "find", "t": 0, "vis_run": 0, "hold": 0, "at_look": 0,
                "grip": float(rng.uniform(0, 1)), "sigma": float(rng.uniform(0, args.dart_sigma)),
                "dart": np.zeros(3), "cmd": None, "quat": None, "skip": True,
                "left": [], "right": [], "state": [], "action": [], "joint_pos": [], "vis": []}

    eps = [new_episode(i) for i in range(N)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h5 = h5py.File(args.out, "w")
    h5.attrs["fps"] = round(1 / dt)
    h5.attrs["state_names"] = STATE_NAMES
    h5.attrs["skills"] = list(language.SKILLS)
    h5.attrs["calibration"] = json.dumps(calib)
    saved, attempted, t0 = 0, 0, time.time()
    look = np.array(LOOK_POSE)
    speed = 0.25  # [m/s]
    torch.inference_mode().__enter__()

    def hide_absent():
        for i, e in enumerate(eps):
            if e["absent"]:
                obj = env.scene[objects[e["k"]]]
                pos = torch.tensor([HIDE_POS], device=dev) + env.scene.env_origins[i : i + 1]
                ids = torch.tensor([i], device=dev)
                obj.write_root_pose_to_sim_index(root_pose=torch.cat([pos, obj.data.root_quat_w.torch[i : i + 1]], -1).float(), env_ids=ids)
                obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=dev), env_ids=ids)

    while saved < args.episodes:
        hide_absent()
        link_p, link_q = link_pose_b()
        tcp_b, _ = adapter.tcp_from_link(link_p, link_q)
        tcp_np, link_np, quat_np = tcp_b.cpu().numpy(), link_p.cpu().numpy(), link_q.cpu().numpy()
        depth = env.scene["left_cam"].data.output["distance_to_image_plane"][..., 0].cpu().numpy()
        left, right = rgb("left_cam"), rgb("right_cam")
        joint = robot.data.joint_pos.torch
        state = adapter.state(link_p, link_q, joint[:, 5], head_b).cpu().numpy()
        actions = torch.zeros(N, 8, device=dev)
        rec_cmd = np.zeros((N, 10), dtype=np.float32)
        record = np.zeros(N, dtype=bool)
        done = []
        for i, e in enumerate(eps):
            if e["cmd"] is None:
                e["cmd"], e["quat"] = tcp_np[i].copy(), quat_np[i].copy()
            if e["phase"] == "pre":  # unrecorded move to the random start pose
                d = e["start"] - e["cmd"]
                step = speed * dt
                e["cmd"] = e["start"] if np.linalg.norm(d) <= step else e["cmd"] + d / np.linalg.norm(d) * step
                e["t"] += 1
                if np.linalg.norm(tcp_np[i] - e["start"]) < 0.012 or e["t"] > 80:
                    e.update(phase="find", t=0, skip=True)
            else:
                vis = visible_fraction(i, e["k"], depth[i])
                e["vis_now"] = vis
                if e["phase"] == "find":
                    e["vis_run"] = e["vis_run"] + 1 if vis >= args.visible_frac else 0
                    if e["absent"] and e["at_look"] >= 6:  # nothing there after settling at the look pose
                        e["phase"], e["fail_at"] = "hold", len(e["action"])
                    elif e["vis_run"] >= 6:
                        e["phase"] = "hold"
                    else:
                        d = look - e["cmd"]
                        step = speed * dt
                        e["cmd"] = look.copy() if np.linalg.norm(d) <= step else e["cmd"] + d / np.linalg.norm(d) * step
                        # Settled at the look pose: reached it, or (out of reach) commanded there and stopped moving.
                        moved = np.linalg.norm(tcp_np[i] - e["prev_tcp"]) if "prev_tcp" in e else 1.0
                        if np.linalg.norm(tcp_np[i] - look) < 0.012 or (np.allclose(e["cmd"], look) and moved < 0.0015):
                            e["at_look"] += 1
                elif e["phase"] == "hold":
                    e["hold"] += 1
                record[i] = not e.pop("skip", False)
                e["prev_tcp"] = tcp_np[i].copy()
                e["t"] += 1
                if e["phase"] == "hold" and e["hold"] >= 10:
                    done.append((i, True))
                elif e["at_look"] > 20 or e["t"] > 200:  # present but never clearly visible: ambiguous, discarded
                    done.append((i, False))
            # DART offset on the executed command only (not during the pre-roll, at the look pose or in the hold).
            rho = 0.9
            e["dart"] = rho * e["dart"] + rng.normal(0, e["sigma"] * (1 - rho**2) ** 0.5, 3)
            executed = e["cmd"] + (e["dart"] if e["phase"] == "find" and not np.allclose(e["cmd"], look) else 0.0)
            rot = math_utils.matrix_from_quat(torch.as_tensor(e["quat"], device=dev).unsqueeze(0))
            link_goal = torch.as_tensor(executed, dtype=torch.float32, device=dev) - (
                rot[0] @ torch.tensor(adapter.tcp_offset, device=dev))
            actions[i, :3] = link_goal
            actions[i, 3:7] = torch.as_tensor(e["quat"], device=dev)
            actions[i, 7] = 1.0 - e["grip"]
            rec_cmd[i] = adapter.to_agnostic(
                torch.as_tensor(e["cmd"], dtype=torch.float32, device=dev).unsqueeze(0), rot,
                torch.tensor([e["grip"]], device=dev), head_b[:1]).cpu().numpy()[0]
        obs, *_ = env.step(actions)
        joint_np = joint.cpu().numpy()
        for i, e in enumerate(eps):
            if record[i]:
                e["left"].append(left[i]); e["right"].append(right[i]); e["state"].append(state[i])
                e["action"].append(rec_cmd[i]); e["joint_pos"].append(joint_np[i]); e["vis"].append(e["vis_now"])
        for i, ok in done:
            attempted += 1
            e = eps[i]
            if ok and saved < args.episodes and len(e["action"]) >= 8:
                g5 = h5.create_group(f"episode_{saved:06d}")
                g5.attrs["find_text"], g5.attrs["object_name"] = e["text"], e["name"]
                g5.attrs["fail_mode"] = "absent" if e["absent"] else ""
                g5.attrs["goal"], g5.attrs["pick_text"], g5.attrs["place_text"] = e["text"], "", ""
                g5.attrs["cube_idx"], g5.attrs["plate_idx"] = 0, 0
                for key, name in (("left", "left_rgb"), ("right", "right_rgb")):
                    arr = np.stack(e[key])
                    g5.create_dataset(name, data=arr, chunks=(1, *arr.shape[1:]), compression="gzip", compression_opts=4)
                for key in ("state", "action", "joint_pos"):
                    g5.create_dataset(key, data=np.stack(e[key]).astype(np.float32))
                g5.create_dataset("skill", data=np.full(len(e["action"]), FIND, dtype=np.uint8))
                g5.create_dataset("visible_frac", data=np.asarray(e["vis"], dtype=np.float32))
                g5.create_dataset("target_pose9", data=np.zeros(9, np.float32) if e["absent"] else target_pose9(i, e["k"]))
                status = np.zeros((len(e["action"]), 2), dtype=np.float32)
                if e["absent"]:
                    status[e["fail_at"]:] = [1.0, 0.0]
                else:
                    status[-5:] = [1.0, 1.0]
                g5.create_dataset("status", data=status)
                saved += 1
                if saved % 20 == 0:
                    h5.flush()
                    print(f"[find] saved {saved}/{args.episodes} (success {saved}/{attempted}), "
                          f"{saved / (time.time() - t0) * 3600:.0f} eps/h", flush=True)
            # New layout for this env, then a fresh episode.
            env._reset_idx(torch.tensor([i], device=dev))
            eps[i] = new_episode(i)
        if done:
            env.scene.write_data_to_sim()

    h5.close()
    print(f"[find] done: {saved} episodes from {attempted} attempts -> {os.path.abspath(args.out)}")
    env.close()
