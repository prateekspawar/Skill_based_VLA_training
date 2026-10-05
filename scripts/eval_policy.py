"""Closed-loop evaluation of a served VLA on the SO-101 through the robot adapter + IK.

Each episode runs the two skills the planner would emit for its goal, using the canonical step text:
``pick(<cube>)`` until the cube is lifted 3 cm (or the step times out), then ``place(<plate>)`` until the
env's success check fires. Start the server first:

    ~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint <.../pretrained_model>
    ~/isaacsim/bin/python scripts/eval_policy.py --episodes 20 [--viz kit] [--heldout_pairs]
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--exec_steps", type=int, default=10, help="Actions executed from each predicted chunk.")
parser.add_argument("--step_timeout", type=float, default=12.0, help="Seconds allowed per skill.")
parser.add_argument("--port", type=int, default=6000)
parser.add_argument("--heldout_pairs", action="store_true", help="Sample only the held-out (cube, plate) pairs.")
parser.add_argument("--target", choices=["none", "oracle", "perception", "sam2", "oracle_tracked"], default="sam2",
                    help="Skill target point (head frame): oracle = simulator truth at skill start; perception = OWLv2 + "
                         "stereo once at episode start; sam2 = OWLv2 seeds SAM2 at episode start, masks are propagated "
                         "every query and turned into points with stereo depth; oracle_tracked = simulator truth "
                         "refreshed every query (upper bound).")
parser.add_argument("--push", action="store_true",
                    help="Moving-target test: during pick, slide the target cube 3-6 cm over 0.5 s while the gripper is >6 cm away.")
parser.add_argument("--seed", type=int, default=0, help="Seeds layouts and pushes so runs are comparable.")
parser.add_argument("--sam2_every", type=int, default=3, help="Control steps between SAM2 updates during pick.")
parser.add_argument("--video_out", default="outputs/eval/episode0.mp4", help="Left-camera video of one episode.")
parser.add_argument("--video_episode", type=int, default=0, help="Which episode to record to --video_out.")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = 2 * args.step_timeout + 2

with sim_app.launch(env_cfg, args):
    from multiprocessing.connection import Client

    import imageio
    import numpy as np
    import torch

    import isaaclab.utils.math as math_utils

    from so101_lang import language
    from so101_lang.envs.head import head_in_base
    from so101_lang.robots import frames
    from so101_lang.robots.so101 import SO101Adapter

    import random

    from so101_lang.envs.mdp import CUBE_NAMES, PLATE_NAMES
    from so101_lang.envs.pick_place_env_cfg import CUBE_Z, PLATE_RADIUS, WORKSPACE_X, WORKSPACE_Y

    if args.heldout_pairs:
        env_cfg.events.reset_layout_and_task.params["pairs"] = language.heldout_pairs()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    env_cfg.seed = args.seed
    env = sim_app.make_env(TASK, env_cfg)
    robot = env.scene["robot"]
    gi = robot.find_bodies("gripper")[0][0]
    adapter = SO101Adapter()
    conn = Client(("localhost", args.port), authkey=b"so101_lang")
    max_steps = int(args.step_timeout / env.step_dt)

    def state_now(head_b):
        pos, quat = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi],
        )
        return adapter.state(pos, quat, robot.data.joint_pos.torch[:, 5], head_b)

    def tcp_and_cubes_b():
        """TCP and all cube centres in the robot base frame."""
        root_p, root_q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        pos, quat = math_utils.subtract_frame_transforms(
            root_p, root_q, robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi]
        )
        tcp, _ = adapter.tcp_from_link(pos, quat)
        cubes = torch.stack([
            math_utils.subtract_frame_transforms(root_p, root_q, env.scene[f"cube_{i}"].data.root_pos_w.torch)[0][0]
            for i in range(language.NUM_CUBES)
        ])
        return tcp[0], cubes

    def oracle_target(skill, cube, plate, head_b):
        """Target point in the head frame: the cube's centre for pick, the plate's centre for place."""
        name = f"cube_{cube}" if skill == "pick" else f"plate_{plate}"
        p_b = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, env.scene[name].data.root_pos_w.torch
        )[0]
        base_in_head = frames.invert_transform(head_b)
        return (base_in_head[0, :3, :3] @ p_b[0] + base_in_head[0, :3, 3]).cpu().numpy()

    def rgb(cam):
        return env.scene[cam].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()

    def perceive_target(skill, cube, plate, head_b):
        """Ask the server's perception (OWLv2 + stereo depth) where the named object is; log its error."""
        query = language.cube_ref(cube) if skill == "pick" else language.plate_ref(plate)
        conn.send({"cmd": "locate", "left": rgb("left_cam"), "right": rgb("right_cam"), "query": query})
        reply = conn.recv()
        err = 1000 * float(np.linalg.norm(reply["point"] - oracle_target(skill, cube, plate, head_b)))
        perception_err.append((skill, err))
        return reply["point"]

    chain = ["base", "shoulder", "upper_arm", "lower_arm", "wrist", "gripper"]
    chain_ids = [robot.find_bodies(n)[0][0] for n in chain]
    jaw_id = robot.find_bodies("moving_jaw_so101_v1")[0][0]

    def robot_points_head(head_b, step=0.01):
        """Robot keypoints in the head frame: link origins along the chain, the moving jaw and the TCP,
        densified every ``step`` metres (what forward kinematics would give on real hardware)."""
        root_p, root_q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        body_w = robot.data.body_pos_w.torch[0]
        to_b = lambda w: math_utils.subtract_frame_transforms(root_p, root_q, w.unsqueeze(0))[0][0]
        link_pos, link_quat = math_utils.subtract_frame_transforms(
            root_p, root_q, robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi]
        )
        tcp, _ = adapter.tcp_from_link(link_pos, link_quat)
        nodes = [to_b(body_w[i]) for i in chain_ids] + [tcp[0]]
        segments = list(zip(nodes[:-1], nodes[1:])) + [(nodes[-2], to_b(body_w[jaw_id])), (to_b(body_w[jaw_id]), tcp[0])]
        pts = []
        for a, b in segments:
            n = max(int(torch.linalg.norm(b - a) / step), 1)
            pts += [a + (b - a) * k / n for k in range(n + 1)]
        p_b = torch.stack(pts)
        base_in_head = frames.invert_transform(head_b)[0]
        return (p_b @ base_in_head[:3, :3].T + base_in_head[:3, 3]).cpu().numpy()

    def sam2_targets(cmd, cube, plate, head_b, frozen=()):
        """Start or update SAM2 tracking of the episode's pick cube and place plate; returns head-frame points."""
        queries = {"pick": language.cube_ref(cube), "place": language.plate_ref(plate)}
        msg = {"cmd": cmd, "left": rgb("left_cam"), "right": rgb("right_cam"), "robot_points": robot_points_head(head_b)}
        if cmd == "sam2_start":
            msg["queries"] = [queries["pick"], queries["place"]]
        else:
            msg["frozen"] = [queries[sk] for sk in frozen]
        conn.send(msg)
        reply = conn.recv()
        out = {}
        for sk, q in queries.items():
            out[sk] = reply["points"][q]
            if sk not in frozen:
                track_err.append((sk, 1000 * float(np.linalg.norm(out[sk] - oracle_target(sk, cube, plate, head_b)))))
            st = reply["info"][q]["status"]
            track_status[st] = track_status.get(st, 0) + 1
        return out

    PUSH_TIME = 0.5  # [s] the cube slides over this long

    def plan_push(cube, rng):
        """Pick a free spot 3-6 cm away in the workspace; returns the planar slide velocity [m/s] or None."""
        origin = env.scene.env_origins[0]
        pos = (env.scene[f"cube_{cube}"].data.root_pos_w.torch[0] - origin).cpu().numpy()
        others = [(env.scene[n].data.root_pos_w.torch[0] - origin).cpu().numpy() for n in CUBE_NAMES + PLATE_NAMES
                  if n != f"cube_{cube}"]
        clear = [0.045] * (len(CUBE_NAMES) - 1) + [PLATE_RADIUS + 0.025] * len(PLATE_NAMES)
        for _ in range(100):
            ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.03, 0.06)
            new = pos[:2] + dist * np.array([np.cos(ang), np.sin(ang)])
            if not (WORKSPACE_X[0] <= new[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= new[1] <= WORKSPACE_Y[1]):
                continue
            # The whole slide path must stay clear of the other objects.
            path = [pos[:2] + (new - pos[:2]) * k / 5 for k in range(6)]
            if all(np.linalg.norm(pt - o[:2]) > c for pt in path for o, c in zip(others, clear)):
                return (new - pos[:2]) / PUSH_TIME
        return None

    def drive_push(cube, start, vel_xy, k, n):
        """Step ``k`` of ``n`` of a push: move the cube a small step along its slide path (continuous motion as
        the cameras see it; written directly because table friction stops a velocity-driven cube within a step)."""
        obj = env.scene[f"cube_{cube}"]
        p = start.clone().unsqueeze(0)
        p[0, :2] += torch.as_tensor(vel_xy, dtype=torch.float32, device=env.device) * PUSH_TIME * k / n
        q = obj.data.root_quat_w.torch[0:1]
        ids = torch.tensor([0], device=env.device)
        obj.write_root_pose_to_sim_index(root_pose=torch.cat([p, q], -1).float(), env_ids=ids)
        obj.write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)

    results, frames_out, latencies, perception_err = [], [], [], []
    track_err, track_status = [], {}
    torch.inference_mode().__enter__()
    for ep in range(args.episodes):
        env.reset()
        head_b = head_in_base(env)
        # One hold step so the cameras show the new layout (first post-reset render is stale).
        env.step(adapter.ik_command(state_now(head_b), head_b))
        cube, plate = env.pp_task[0].tolist()
        cube_name = f"cube_{cube}"
        rest_z = float(env.scene[cube_name].data.root_pos_w.torch[0, 2])
        steps = [("pick", language.skill_text("pick", object=language.cube_ref(cube))),
                 ("place", language.skill_text("place", target=language.plate_ref(plate)))]
        outcome = {"goal": language.goal_instruction(cube, plate), "pick": False, "place": False, "close": None}
        # Perception looks at the scene once, before the arm moves (like a planner's first observation):
        # later views are occluded by the arm and the held cube. Plates never move; cubes only once grasped.
        perceived = (
            {sk: perceive_target(sk, cube, plate, head_b) for sk in ("pick", "place")}
            if args.target == "perception" else {}
        )
        ep_steps = 0
        if args.target == "sam2":
            perceived = sam2_targets("sam2_start", cube, plate, head_b)
        rng = np.random.default_rng(args.seed * 1000 + ep)
        push_at = rng.uniform(0.3, 1.0)  # [s] into the pick skill
        outcome["push_mm"] = None
        push = {"vel": None, "steps_left": 0, "start": None}
        done = False
        for skill_i, (skill, text) in enumerate(steps):
            conn.send({"cmd": "reset", "seed": args.seed * 100000 + ep * 10 + skill_i, "skill": skill})
            conn.recv()
            # The target is fixed for the whole skill (cube before grasp, plate is static).
            target = oracle_target(skill, cube, plate, head_b) if args.target == "oracle" else None
            if args.target in ("perception", "sam2"):
                target = perceived[skill]
            t = 0
            while t < max_steps and not done:
                # Dynamic targets: refresh before every query while the object can still move (pick: until
                # the gripper starts closing on it; place: the plate, which normally stays put).
                still_tracking = skill == "place" or outcome["close"] is None
                if args.target == "sam2" and t > 0:
                    # Freeze a target once the gripper is close to it: the gripper / held cube then overlaps the
                    # object's mask, and an object about to be grasped or placed on cannot be re-targeted anyway.
                    tcp_h = state_now(head_b)[0, :3].cpu().numpy()
                    frozen = tuple(sk for sk, radius in (("pick", 0.05), ("place", 0.06))
                                   if np.linalg.norm(tcp_h - perceived[sk]) < radius
                                   or (sk == "pick" and outcome["close"] is not None))
                    perceived = sam2_targets("sam2_update", cube, plate, head_b, frozen)
                    target = perceived[skill]
                if args.target == "oracle_tracked" and still_tracking:
                    target = oracle_target(skill, cube, plate, head_b)
                reply_req = {"left": rgb("left_cam"), "right": rgb("right_cam"), "state": state_now(head_b)[0].cpu().numpy(), "task": text,
                             "target": target}
                conn.send(reply_req)
                reply = conn.recv()
                latencies.append(reply["t_total"])
                for a in torch.as_tensor(reply["actions"][: args.exec_steps], device=env.device):
                    if skill == "pick" and outcome["close"] is None and float(a[9]) < 0.5:
                        # First closing command: where is the TCP relative to the target / nearest cube?
                        tcp, cubes = tcp_and_cubes_b()
                        d = torch.linalg.norm(cubes[:, :2] - tcp[:2], dim=1)
                        near = int(torch.argmin(d))
                        outcome["close"] = (1000 * float(d[cube]), 1000 * float(tcp[2] - cubes[cube, 2]), language.CUBE_NAMES[near])
                        if target is not None:
                            outcome["target_err_at_close"] = 1000 * float(np.linalg.norm(target - oracle_target("pick", cube, plate, head_b)))
                    _, _, term, trunc, _ = env.step(adapter.ik_command(a.unsqueeze(0), head_b))
                    t += 1
                    ep_steps += 1
                    if (args.push and skill == "pick" and push["start"] is None and outcome["close"] is None
                            and t * env.step_dt >= push_at):
                        tcp, cubes = tcp_and_cubes_b()
                        if float(torch.linalg.norm(cubes[cube] - tcp)) > 0.06:  # gripper still well away (3-D)
                            push["vel"] = plan_push(cube, rng)
                            push["start"] = env.scene[cube_name].data.root_pos_w.torch[0].clone()
                            push["steps_left"] = round(PUSH_TIME / env.step_dt) if push["vel"] is not None else 0
                    if push["steps_left"] > 0:
                        push["steps_left"] -= 1
                        n = round(PUSH_TIME / env.step_dt)
                        drive_push(cube, push["start"], push["vel"], n - push["steps_left"], n)
                        if push["steps_left"] == 0:  # slide finished: measure the actual displacement
                            moved = env.scene[cube_name].data.root_pos_w.torch[0] - push["start"]
                            outcome["push_mm"] = 1000 * float(torch.linalg.norm(moved[:2]))
                            rest_z = float(env.scene[cube_name].data.root_pos_w.torch[0, 2])
                            if args.target == "sam2":
                                outcome["sam2_err_after_push"] = 1000 * float(np.linalg.norm(
                                    perceived["pick"] - oracle_target("pick", cube, plate, head_b)))
                    if (args.target == "sam2" and skill == "pick" and outcome["close"] is None
                            and t % args.sam2_every == 0):
                        # Feed SAM2 the motion between policy queries so masks follow moving objects.
                        tcp_h = state_now(head_b)[0, :3].cpu().numpy()
                        frozen = ("pick",) if np.linalg.norm(tcp_h - perceived["pick"]) < 0.05 else ()
                        perceived = sam2_targets("sam2_update", cube, plate, head_b, frozen + ("place",))
                    if ep == args.video_episode:
                        frames_out.append(rgb("left_cam"))
                    lifted = float(env.scene[cube_name].data.root_pos_w.torch[0, 2]) > rest_z + 0.03
                    if skill == "pick" and lifted:
                        outcome["pick"] = True
                    if term[0] or trunc[0]:
                        outcome["place"] = bool(env.termination_manager.get_term("success")[0])
                        done = True
                        break
                    if skill == "pick" and outcome["pick"]:
                        break
                if skill == "pick" and outcome["pick"]:
                    break
            if done or (skill == "pick" and not outcome["pick"]):
                break
        results.append(outcome)
        close = outcome["close"]
        close_txt = (f"close: target dxy={close[0]:.0f}mm dz={close[1]:+.0f}mm nearest={close[2]}" if close else "never closed")
        if outcome["push_mm"]:
            close_txt += f" | pushed {outcome['push_mm']:.0f}mm"
        if outcome.get("target_err_at_close") is not None:
            close_txt += f" | target err at close {outcome['target_err_at_close']:.0f}mm"
        print(f"[eval] {ep:3d} '{outcome['goal']}': pick={'ok' if outcome['pick'] else 'FAIL'} "
              f"place={'ok' if outcome['place'] else 'FAIL'} | {close_txt}", flush=True)
        if ep == args.video_episode and frames_out:
            os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
            imageio.mimsave(args.video_out, frames_out, fps=round(1 / env.step_dt))

    n = len(results)
    pick = sum(r["pick"] for r in results)
    full = sum(r["place"] for r in results)
    print(f"[eval] pick success {pick}/{n} = {pick / n:.0%}; full pick+place {full}/{n} = {full / n:.0%}; "
          f"mean server latency {1000 * np.mean(latencies):.0f} ms")
    pushed = [r for r in results if r["push_mm"]]
    if args.push:
        print(f"[eval] pushes: {len(pushed)}/{n} episodes; pick success on pushed episodes "
              f"{sum(r['pick'] for r in pushed)}/{len(pushed)}, full {sum(r['place'] for r in pushed)}/{len(pushed)}")
    after = [r["sam2_err_after_push"] for r in results if r.get("sam2_err_after_push") is not None]
    if after:
        moved = [r["push_mm"] for r in results if r.get("sam2_err_after_push") is not None]
        print(f"[eval] SAM2 target error right after a push: median {np.median(after):.0f} mm, max {np.max(after):.0f} mm "
              f"(cube actually moved median {np.median(moved):.0f} mm) [{len(after)}]")
    errs_close = [r["target_err_at_close"] for r in results if r.get("target_err_at_close") is not None]
    if errs_close:
        print(f"[eval] target-point error when the gripper closes: median {np.median(errs_close):.0f} mm, "
              f"max {np.max(errs_close):.0f} mm [{len(errs_close)}]")
    if track_err:
        for sk in ("pick", "place"):
            e = [x for s_, x in track_err if s_ == sk]
            if e:
                print(f"[eval] tracked target error ({sk}): median {np.median(e):.0f} mm, max {np.max(e):.0f} mm [{len(e)}]")
        print(f"[eval] tracker updates: {track_status}")
    if perception_err:
        for sk in ("pick", "place"):
            e = [x for s_, x in perception_err if s_ == sk]
            if e:
                print(f"[eval] perception error ({sk} target): median {np.median(e):.0f} mm, max {np.max(e):.0f} mm [{len(e)}]")
    conn.close()
    env.close()
