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
parser.add_argument("--pipeline", choices=["direct", "find"], default="find",
                    help="find (default): find(cube) -> start tracking the cube from that view -> find(plate) while the "
                         "cube keeps tracking -> start tracking the plate -> return home -> pick -> place, all on the "
                         "same trackers. direct: detect both objects from the home view, then pick -> place.")
parser.add_argument("--find_retries", type=int, default=2,
                    help="Retries of a find() whose model reports failure (each retry starts from home).")
parser.add_argument("--target_dim", type=int, choices=[3, 9], default=3,
                    help="Target sent to the policy: 3 = position, 9 = position + 6-D rotation (19-D state models).")
parser.add_argument("--sam2_every", type=int, default=3, help="Control steps between SAM2 updates during pick.")
parser.add_argument("--verdict", action="store_true",
                    help="12-D action models: a skill ends when the model reports done ([done, success] > 0.5 for 2 "
                         "consecutive actions), not on simulator truth; a failed verdict or a timeout stops the episode "
                         "(hold still). Each verdict is scored against the simulator.")
parser.add_argument("--verdict_from", choices=["executed", "first"], default="executed",
                    help="executed: read [done, success] from every executed action (later chunk entries are predictions "
                         "of the future); first: only from the first action of each new chunk (the current frame).")
parser.add_argument("--fold_cube_yaw", action="store_true",
                    help="Direct pipeline with 9-D targets: give the cube's true yaw folded to [-45, 45) deg, as perception "
                         "(locate9) reports it, instead of the unfolded yaw used in training.")
parser.add_argument("--video_out", default="outputs/eval/episode0.mp4", help="Left-camera video of one episode.")
parser.add_argument("--video_episode", type=int, default=0, help="Which episode to record to --video_out.")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.scene.num_envs = 1
env_cfg.episode_length_s = 2 * args.step_timeout + 2
SUCCESS_FUNC = env_cfg.terminations.success.func
if args.verdict:
    # Place must run until the model reports, so the success term never terminates; it stays registered so the
    # manager resolves its parameters (joint ids), and on_plate() calls the real check with them.
    import functools

    def _never(env, *a, **kw):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    env_cfg.terminations.success.func = functools.update_wrapper(_never, SUCCESS_FUNC)  # same signature

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
    verdicts = {"pick": [], "place": [], "find": []}  # (verdict: True / False / "timeout", truth: bool)

    def on_plate():
        return bool(SUCCESS_FUNC(env, **env.termination_manager.get_term_cfg("success").params)[0])

    class Verdict:
        """Debounced [done, success] reading from the executed actions."""
        def __init__(self, always=False):
            self.run, self.result, self.always = 0, None, always

        def update(self, a, first=True):
            if not (args.verdict or self.always) or len(a) <= 10 or (args.verdict_from == "first" and not first):
                return False
            done, ok = float(a[10]) > 0.5, float(a[11]) > 0.5
            if args.verdict_from == "first":
                # A pass is accepted at once; a fail needs two consecutive readings (one query apart): a real
                # failure is followed by holding still, so the model keeps reporting it, while a single spurious
                # "fail" mid-approach would otherwise end the skill.
                self.run = self.run + 1 if done and not ok else 0
                if done and ok:
                    self.result = True
                elif self.run >= 2:
                    self.result = False
            else:  # executed: 2 consecutive done actions
                self.run = self.run + 1 if done else 0
                if self.run >= 2:
                    self.result = ok
            return self.result is not None

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

    def plate_b(plate):
        return math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, env.scene[f"plate_{plate}"].data.root_pos_w.torch
        )[0][0]

    def place_snapshot(cube, plate, head_b, sent):
        """Geometry at the moment the place policy first commands the gripper open."""
        tcp, cubes = tcp_and_cubes_b()
        pl = plate_b(plate)
        err = None if sent is None else 1000 * float(np.linalg.norm(np.asarray(sent)[:3] - oracle_target("place", cube, plate, head_b)))
        near = None
        if sent is not None:  # which scene object is the place target actually on?
            cands = {f"cube_{i}": oracle_target("pick", i, plate, head_b) for i in range(language.NUM_CUBES)}
            cands.update({f"plate_{j}": oracle_target("place", cube, j, head_b) for j in range(language.NUM_PLATES)})
            cands["tcp"] = state_now(head_b)[0, :3].cpu().numpy()
            k = min(cands, key=lambda n: np.linalg.norm(np.asarray(sent)[:3] - cands[n]))
            held = " (held)" if k == f"cube_{cube}" else (" (target)" if k == f"plate_{plate}" else "")
            near = f"{k}{held} {1000 * np.linalg.norm(np.asarray(sent)[:3] - cands[k]):.0f}mm"
        return {"tcp_plate_dxy": 1000 * float(torch.linalg.norm(tcp[:2] - pl[:2])),
                "tcp_plate_dz": 1000 * float(tcp[2] - pl[2]),
                "cube_in_hand": float(torch.linalg.norm(cubes[cube] - tcp)) < 0.03, "target_err": err, "near": near}

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
                ep_track[sk].append((cmd, track_err[-1][1], reply["info"][q]["status"]))
            st = reply["info"][q]["status"]
            track_status[st] = track_status.get(st, 0) + 1
        return out

    def truth_pose9(name, head_b):
        """True head-frame 9-D pose of a scene object (plates: zero yaw, they are symmetric)."""
        root_p, root_q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        p_b, q_b = math_utils.subtract_frame_transforms(root_p, root_q, env.scene[name].data.root_pos_w.torch,
                                                         env.scene[name].data.root_quat_w.torch)
        rot_b = math_utils.matrix_from_quat(q_b)[0] if name.startswith("cube") else torch.eye(3, device=env.device)
        bih = frames.invert_transform(head_b)[0]
        pos_h, rot_h = bih[:3, :3] @ p_b[0] + bih[:3, 3], bih[:3, :3] @ rot_b
        return torch.cat([pos_h, rot_h[:, 0], rot_h[:, 1]]).cpu().numpy()

    def fold_cube_yaw(pose9, head_b):
        """Fold the yaw (about base z) of a 9-D head-frame pose into [-45, 45) deg, like locate9 for cubes."""
        hb = head_b[0, :3, :3].cpu().numpy()
        yaw = float(np.arctan2(*(hb @ pose9[3:6])[[1, 0]]))
        yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
        c, s_ = np.cos(yaw), np.sin(yaw)
        rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        r_h = hb.T @ rz
        return np.concatenate([pose9[:3], r_h[:, 0], r_h[:, 1]]).astype(np.float32)

    def yaw_err_deg(est9, true9, head_b, symmetric_90):
        """Yaw difference about the base z axis between two 9-D poses (mod 90 deg for cubes)."""
        hb = head_b[0, :3, :3].cpu().numpy()
        ya = np.arctan2(*(hb @ est9[3:6])[[1, 0]]); yb = np.arctan2(*(hb @ true9[3:6])[[1, 0]])
        q = np.pi / 2 if symmetric_90 else 2 * np.pi
        return abs(np.rad2deg((ya - yb + q / 2) % q - q / 2))

    tracked = []  # objects whose SAM2 tracker is running (find pipeline)

    bg_ctx = {}

    def bg_track(head_b):
        """Advance every running tracker by one frame (objects keep tracking while the arm does something else)."""
        if args.target != "sam2" or not tracked:
            return None
        conn.send({"cmd": "sam2_update", "left": rgb("left_cam"), "right": rgb("right_cam"),
                   "robot_points": robot_points_head(head_b), "frozen": []})
        reply = conn.recv()
        for q, st in reply["info"].items():
            track_status[st["status"]] = track_status.get(st["status"], 0) + 1
        if bg_ctx:
            cq = language.cube_ref(bg_ctx["cube"])
            if cq in reply["points"]:
                ep_track["pick"].append(("bg", 1000 * float(np.linalg.norm(
                    reply["points"][cq] - oracle_target("pick", bg_ctx["cube"], bg_ctx["plate"], head_b))),
                    reply["info"].get(cq, {}).get("status")))
        return reply["points"]

    def run_find(query, name, head_b, ep, seed):
        """Run the find model until it reports (or holds still / 6 s), then get the object's 9-D pose from this view
        and (SAM2 targets) start tracking it here. Returns (pose9, ok, points); ok = the model did not report failure."""
        conn.send({"cmd": "reset", "seed": seed, "skill": "find"})
        swap.append(conn.recv().get("swap_ms", 0.0))
        text = language.skill_text("find", object=query)
        t, cmds = 0, []
        vd = Verdict(always=True)  # find always ends on its own verdict when the model has one
        while t < int(6.0 / env.step_dt) and vd.result is None:
            conn.send({"left": rgb("left_cam"), "right": rgb("right_cam"), "state": state_now(head_b)[0].cpu().numpy(),
                       "task": text, "target": np.zeros(9, np.float32)})
            reply = conn.recv()
            for k, a in enumerate(torch.as_tensor(reply["actions"][: args.exec_steps], device=env.device)):
                env.step(adapter.ik_command(a.unsqueeze(0), head_b))
                cmds.append(a[:3].cpu().numpy())
                t += 1
                if t % args.sam2_every == 0:
                    bg_track(head_b)
                if ep == args.video_episode:
                    frames_out.append(rgb("left_cam"))
                if vd.update(a, k == 0):
                    break
            recent = np.array(cmds[-args.exec_steps:])
            if vd.result is None and len(reply["actions"][0]) <= 10 and t >= 10 \
                    and np.linalg.norm(recent - recent.mean(0), axis=1).max() < 0.004:
                break  # model without verdict dims: holding still means it judges the view clear
        points, name_ok = None, None
        if args.target == "sam2":
            conn.send({"cmd": "find_start", "left": rgb("left_cam"), "right": rgb("right_cam"), "query": query,
                       "robot_points": robot_points_head(head_b)})
            reply = conn.recv()
            pose9, points, name_ok = reply["pose9"], reply["points"], reply.get("name_ok")
            if query not in tracked:
                tracked.append(query)
        else:
            conn.send({"cmd": "locate9", "left": rgb("left_cam"), "right": rgb("right_cam"), "query": query})
            pose9 = conn.recv()["pose9"]
        true9 = truth_pose9(name, head_b)
        err = 1000 * float(np.linalg.norm(pose9[:3] - true9[:3]))
        sk = "pick" if name.startswith("cube") else "place"
        ep_track[sk].append(("find_start", err, "init"))
        find_log.append({"obj": query, "steps": t, "pos_err_mm": err,
                         "yaw_err_deg": yaw_err_deg(pose9, true9, head_b, name.startswith("cube"))})
        verdicts["find"].append((vd.result if vd.result is not None else "timeout", err < 30))
        # Accept when either the find model or the detector's name check says the object is there; retry only
        # when both say it is not (each alone has false "not found" calls).
        ok = vd.result is not False or bool(name_ok)
        find_log[-1].update(model=vd.result, name_ok=name_ok, accepted=ok)
        return pose9, ok, points

    def go_home(home_link, head_b, steps=30, on_step=None):
        """Scripted return of the gripper link to its episode-start (home) position over ~1.5 s."""
        pos, quat = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi])
        grip = 1.0 - adapter.opening_from_joint(robot.data.joint_pos.torch[:, 5]).unsqueeze(-1)
        for k in range(steps + 10):
            a = min((k + 1) / steps, 1.0)
            env.step(torch.cat([pos + (home_link[0] - pos) * a, home_link[1], grip], -1))
            if on_step is not None and (k + 1) % args.sam2_every == 0:
                on_step(head_b)

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
    find_log, swap = [], []
    track_err, track_status = [], {}
    ep_track = {"pick": [], "place": []}
    torch.inference_mode().__enter__()
    for ep in range(args.episodes):
        env.reset()
        ep_track["pick"].clear(); ep_track["place"].clear()
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
        rot6d = {}
        find_failed = False
        if args.pipeline == "find":
            home_link = math_utils.subtract_frame_transforms(
                robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
                robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi])
            tracked.clear()
            bg_ctx.update(cube=cube, plate=plate)
            if args.target == "sam2":
                conn.send({"cmd": "sam2_reset"})
                conn.recv()
            poses = {}
            for i_obj, (sk, query, name) in enumerate((("pick", language.cube_ref(cube), f"cube_{cube}"),
                                                      ("place", language.plate_ref(plate), f"plate_{plate}"))):
                for attempt in range(1 + args.find_retries):
                    pose, ok, _ = run_find(query, name, head_b, ep, args.seed * 100000 + ep * 10 + 7 + i_obj + 2 * attempt)
                    outcome[f"find_{sk}_attempts"] = attempt + 1
                    if ok:
                        break
                    go_home(home_link, head_b, on_step=bg_track)  # retry from home
                poses[sk] = pose
                if not ok:
                    find_failed = True
                    outcome["find_failed"] = query
                    break
            if not find_failed:
                rot6d = {sk: v[3:] for sk, v in poses.items()}
                if args.target != "sam2":
                    perceived = {sk: v[:3] for sk, v in poses.items()}
        elif args.target_dim == 9:
            truth = {"pick": truth_pose9(f"cube_{cube}", head_b), "place": truth_pose9(f"plate_{plate}", head_b)}
            if args.fold_cube_yaw:
                truth["pick"] = fold_cube_yaw(truth["pick"], head_b)
            rot6d = {sk: v[3:] for sk, v in truth.items()}  # direct mode: rotation from the simulator
        if args.target == "sam2" and args.pipeline == "direct":
            perceived = sam2_targets("sam2_start", cube, plate, head_b)  # both objects from the home view
        if args.pipeline == "find" and not find_failed:
            go_home(home_link, head_b, on_step=bg_track)  # both trackers keep running
            if args.target == "sam2":
                perceived = sam2_targets("sam2_update", cube, plate, head_b)
                outcome["pick_target_start_err"] = 1000 * float(np.linalg.norm(perceived["pick"] - oracle_target("pick", cube, plate, head_b)))
        rng = np.random.default_rng(args.seed * 1000 + ep)
        place_init = [e for e in ep_track["place"] if e[0] in ("sam2_start", "find_start")]
        outcome["place_init_err"] = place_init[-1][1] if place_init else None
        push_at = rng.uniform(0.3, 1.0)  # [s] into the pick skill
        outcome["push_mm"] = None
        push = {"vel": None, "steps_left": 0, "start": None}
        done = False
        if not find_failed and rot6d.get("pick") is not None:
            true_c = truth_pose9(f"cube_{cube}", head_b)
            outcome["pick_start"] = {
                "yaw_err": yaw_err_deg(np.concatenate([true_c[:3], rot6d["pick"]]), true_c, head_b, True),
                "opening": float(state_now(head_b)[0, 9])}
        if find_failed:
            steps = []  # find() failed after all retries: stop and hold, report to the orchestrator
        for skill_i, (skill, text) in enumerate(steps):
            conn.send({"cmd": "reset", "seed": args.seed * 100000 + ep * 10 + skill_i, "skill": skill})
            conn.recv()
            # The target is fixed for the whole skill (cube before grasp, plate is static).
            target = oracle_target(skill, cube, plate, head_b) if args.target == "oracle" else None
            if args.target in ("perception", "sam2"):
                target = perceived[skill]
            t, vd = 0, Verdict()
            while t < max_steps and not done and vd.result is None:
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
                    sent = target
                if target is not None and args.target_dim == 9:
                    sent = np.concatenate([np.asarray(target, np.float32)[:3], np.asarray(rot6d[skill], np.float32)])
                reply_req = {"left": rgb("left_cam"), "right": rgb("right_cam"), "state": state_now(head_b)[0].cpu().numpy(),
                             "task": text, "target": sent}
                conn.send(reply_req)
                reply = conn.recv()
                latencies.append(reply["t_total"])
                for k, a in enumerate(torch.as_tensor(reply["actions"][: args.exec_steps], device=env.device)):
                    if skill == "place" and outcome.get("release") is None and float(a[9]) > 0.5:
                        outcome["release"] = place_snapshot(cube, plate, head_b, target)
                    if skill == "pick" and outcome["close"] is None and float(a[9]) < 0.5:
                        # First closing command: where is the TCP relative to the target / nearest cube?
                        tcp, cubes = tcp_and_cubes_b()
                        d = torch.linalg.norm(cubes[:, :2] - tcp[:2], dim=1)
                        near = int(torch.argmin(d))
                        outcome["close"] = (1000 * float(d[cube]), 1000 * float(tcp[2] - cubes[cube, 2]), language.CUBE_NAMES[near])
                        st = state_now(head_b)[0].cpu().numpy()
                        outcome["close_yaw"] = yaw_err_deg(st[:9], truth_pose9(f"cube_{cube}", head_b), head_b, True)
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
                    if (skill == "place" and outcome.get("release") is None and outcome.get("dropped") is None
                            and float(env.scene[cube_name].data.root_pos_w.torch[0, 2]) < rest_z + 0.01):
                        outcome["dropped"] = t  # the cube is back on the table before any release command
                    if skill == "pick" and lifted:
                        outcome["pick"] = True
                    if term[0] or trunc[0]:
                        # The env has already reset: read the success term's value from this step, not the new layout.
                        outcome["place"] = bool(env.termination_manager.get_term("success")[0]) and not args.verdict
                        done = True
                        break
                    if vd.update(a, k == 0):
                        tcp, cubes = tcp_and_cubes_b()
                        outcome[f"{skill}_verdict_at"] = (t, 1000 * float(torch.linalg.norm(tcp - cubes[cube])),
                                                          float(state_now(head_b)[0, 9]))
                        break
                    if skill == "pick" and outcome["pick"] and not args.verdict:
                        break
                if skill == "pick" and outcome["pick"] and not args.verdict:
                    break
            if args.verdict:
                if not done:  # hold the last command 0.5 s (as the robot does after a verdict), then check the truth
                    for _ in range(10):
                        env.step(adapter.ik_command(a.unsqueeze(0), head_b))
                lifted = float(env.scene[cube_name].data.root_pos_w.torch[0, 2]) > rest_z + 0.03
                if skill == "place":
                    c = env.scene[cube_name].data.root_pos_w.torch[0]
                    pl = env.scene[f"plate_{plate}"].data.root_pos_w.torch[0]
                    outcome["place_geom"] = (1000 * float(torch.linalg.norm(c[:2] - pl[:2])), 1000 * float(c[2] - pl[2]),
                                             float(adapter.opening_from_joint(robot.data.joint_pos.torch[:, 5])[0]))
                truth = lifted if skill == "pick" else (on_plate() if not done else False)
                if skill == "place":
                    outcome["place"] = truth
                verdicts[skill].append((vd.result if vd.result is not None else "timeout", truth))
                outcome[f"{skill}_verdict"] = verdicts[skill][-1]
                if vd.result is not True:
                    break  # failed / timed out: stop and hold, the orchestrator decides what next
                continue
            if done or (skill == "pick" and not outcome["pick"]):
                break
        results.append(outcome)
        close = outcome["close"]
        close_txt = (f"close: target dxy={close[0]:.0f}mm dz={close[1]:+.0f}mm nearest={close[2]}" if close else "never closed")
        if outcome["push_mm"]:
            close_txt += f" | pushed {outcome['push_mm']:.0f}mm"
        if outcome.get("target_err_at_close") is not None:
            close_txt += f" | target err at close {outcome['target_err_at_close']:.0f}mm"
        if args.verdict:
            close_txt += " | verdicts " + " ".join(f"{sk}={outcome[f'{sk}_verdict'][0]}/{outcome[f'{sk}_verdict'][1]}"
                                                     for sk in ("pick", "place") if f"{sk}_verdict" in outcome)
        if outcome["pick"] and not outcome["place"]:
            r = outcome.get("release")
            if outcome.get("dropped") is not None:
                cause = "dropped before release"
            elif r is None:
                cause = "never released"
            elif r["target_err"] is not None and r["target_err"] > 30:
                cause = "bad plate target"
            elif r["tcp_plate_dxy"] > 35:
                cause = "released off the plate (target ok)"
            elif not r["cube_in_hand"]:
                cause = "cube not in hand at release"
            else:
                cause = "released over the plate"
            outcome["place_cause"] = cause
            if r:
                close_txt += (f" | release: tcp-plate dxy={r['tcp_plate_dxy']:.0f}mm dz={r['tcp_plate_dz']:.0f}mm "
                              f"target err={r['target_err'] if r['target_err'] is None else round(r['target_err'])}mm "
                              f"in hand={r['cube_in_hand']} near={r['near']}")
            if outcome.get("place_init_err") is not None:
                upd = [e for e in ep_track["place"] if e[0] == "sam2_update"]
                close_txt += (f" | plate target at start {outcome['place_init_err']:.0f}mm, max during tracking "
                              f"{max([e[1] for e in upd], default=0):.0f}mm ({len(upd)} updates)")
            close_txt += f" | PLACE FAIL: {cause}"
        if not outcome["pick"] and outcome.get("pick_target_start_err") is not None:
            fs = [e[1] for e in ep_track["pick"] if e[0] == "find_start"]
            bg = [e for e in ep_track["pick"] if e[0] == "sam2_update"]
            bgm = max([e[1] for e in ep_track["pick"] if e[0] == "bg"], default=0)
            close_txt += (f" | cube target: at find {fs[-1] if fs else float('nan'):.0f}mm, max during plate find/home "
                          f"{bgm:.0f}mm, at pick start "
                          f"{outcome['pick_target_start_err']:.0f}mm")
        if outcome.get("pick_start"):
            close_txt += (f" | pick start: target yaw err {outcome['pick_start']['yaw_err']:.0f}deg, gripper opening "
                          f"{outcome['pick_start']['opening']:.2f}")
        if outcome.get("close_yaw") is not None:
            close_txt += f" | jaw-cube yaw at close {outcome['close_yaw']:.0f}deg"
        if not outcome["pick"] and outcome.get("pick_verdict_at"):
            t_, d_, o_ = outcome["pick_verdict_at"]
            close_txt += f" | pick verdict at step {t_}: tcp-cube {d_:.0f}mm, gripper opening {o_:.2f}"
        if outcome.get("find_failed"):
            close_txt += f" | FIND FAILED: {outcome['find_failed']}"
        retries = [outcome.get(f"find_{sk}_attempts", 1) - 1 for sk in ("pick", "place")]
        if any(retries):
            close_txt += f" | find retries cube={retries[0]} plate={retries[1]}"
        if outcome.get("place_geom"):
            close_txt += " | cube-plate dxy={:.0f}mm dz={:.0f}mm opening={:.2f}".format(*outcome["place_geom"])
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
    if find_log:
        for kind in ("cube", "plate"):
            fl = [f for f in find_log if kind in f["obj"]]
            pe = np.array([f["pos_err_mm"] for f in fl]); ye = np.array([f["yaw_err_deg"] for f in fl])
            print(f"[eval] find({kind}): pose position error median {np.median(pe):.0f} mm, >30 mm in {(pe > 30).sum()}/{len(pe)}; "
                  f"yaw error median {np.median(ye):.1f} deg; mean duration {np.mean([f['steps'] for f in fl]) * env.step_dt:.1f} s")
        print(f"[eval] skill swap time: median {np.median(swap):.1f} ms")
    if find_log and "accepted" in find_log[0]:
        fl = [f for f in find_log if "accepted" in f]
        good = [f["pos_err_mm"] < 30 for f in fl]
        acc = [f["accepted"] for f in fl]
        print(f"[eval] find acceptance: {sum(acc)}/{len(fl)} accepted; accepted but wrong (>30 mm) "
              f"{sum(a and not g for a, g in zip(acc, good))}, rejected but right {sum(g and not a for a, g in zip(acc, good))}; "
              f"model said not-found {sum(f['model'] is False for f in fl)}, name check failed "
              f"{sum(f['name_ok'] is False for f in fl)}")
    if args.pipeline == "find":
        n_retry = sum(r.get("find_pick_attempts", 1) + r.get("find_place_attempts", 1) - 2 for r in results)
        print(f"[eval] find: {sum(bool(r.get('find_failed')) for r in results)}/{n} episodes stopped (find failed after "
              f"{args.find_retries} retries); {n_retry} retries in total")
    def med(xs):
        return f"{np.median(xs):.1f}" if xs else "-"
    for name, sel in (("picked", [r for r in results if r["pick"]]),
                      ("missed (closed, not lifted)", [r for r in results if not r["pick"] and r.get("close")])):
        print(f"[eval] pick {name} n={len(sel)}: median target yaw err {med([r['pick_start']['yaw_err'] for r in sel if r.get('pick_start')])} deg, "
              f"jaw-cube yaw at close {med([r['close_yaw'] for r in sel if r.get('close_yaw') is not None])} deg, "
              f"close offset dxy {med([r['close'][0] for r in sel])} mm, start opening {med([r['pick_start']['opening'] for r in sel if r.get('pick_start')])}")
    causes = [r["place_cause"] for r in results if r.get("place_cause")]
    if causes:
        from collections import Counter
        print(f"[eval] place failures after a successful pick: {len(causes)}/{sum(r['pick'] for r in results)}: "
              + ", ".join(f"{c} {k}" for k, c in Counter(causes).most_common()))
    for sk, vs in verdicts.items():
        if vs:
            tp = sum(v is True and t for v, t in vs); tn = sum(v is False and not t for v, t in vs)
            fp = sum(v is True and not t for v, t in vs); fn = sum(v is False and t for v, t in vs)
            to = [t for v, t in vs if v == "timeout"]
            print(f"[eval] verdict {sk}: accuracy {(tp + tn) / max(tp + tn + fp + fn, 1):.0%} on {tp + tn + fp + fn} reports "
                  f"(pass: {tp} right / {fp} wrong, fail: {tn} right / {fn} wrong); "
                  f"timeouts {len(to)} ({sum(to)} of them actually succeeded)")
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
