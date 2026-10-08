"""Generate failed pick / place demonstrations with a per-frame status label (done, success).

Each episode injects one failure into the scripted expert:
  pick : miss (grasp point 2-3.5 cm beside the cube)       slip (gripper opens while lifting)
  place: off  (release point 9-12 cm from the plate centre) drop (gripper opens during the carry)
The outcome comes from the simulator: pick failed if the cube is not 3 cm above its rest height when the lift ends;
place failed if the cube falls during the carry or is not on the plate after release. After a failure the expert
stops and holds still for 0.5 s; those frames are labelled status [done=1, success=0]. The last 5 frames of a skill
that succeeded (the pick in a place-failure episode) are labelled [1, 1]; all other frames [0, 0]. Episodes in which
the injected error did not cause a failure are discarded.

Timeouts / stalls are not generated: a memoryless policy cannot tell "stuck for 3 s" from "just started", so the
executor decides those (no progress or skill time limit -> failed).

Output = gen_demos.py format + per episode: status (T, 2) float32, attrs failed_skill, fail_mode.

    ~/isaacsim/bin/python scripts/gen_fail_demos.py --num_envs 16 --episodes 400 --out datasets/raw/fail_400.h5
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=100, help="Failure episodes to save (split ~evenly pick / place).")
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dart_sigma", type=float, default=0.012)
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.seed = args.seed

MODES = {"pick": ["miss", "slip"], "place": ["off", "drop"]}
HOLD_STEPS = 10  # 0.5 s of holding still after a failure
SUCCESS_TAIL = 5  # frames labelled [1, 1] at the end of a skill that succeeded

with sim_app.launch(env_cfg, args):
    import json
    import time

    import h5py
    import numpy as np
    import torch

    import isaaclab.utils.math as math_utils

    from so101_lang import language
    from so101_lang.envs.head import calibration_dict, head_in_base
    from so101_lang.envs.mdp import CUBE_NAMES, PLATE_NAMES
    from so101_lang.envs.pick_place_env_cfg import PLATE_RADIUS, WORKSPACE_X, WORKSPACE_Y
    from so101_lang.expert.pick_place_sm import ExpertCfg, Phase, PickPlaceExpert, phase_skill
    from so101_lang.robots import frames
    from so101_lang.robots.so101 import STATE_NAMES, SO101Adapter

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    N, dt, dev = env.num_envs, env.step_dt, env.device
    robot = env.scene["robot"]
    gi = robot.find_bodies("gripper")[0][0]
    objects = CUBE_NAMES + PLATE_NAMES
    adapter = SO101Adapter()
    skill_ids = {name: i for i, name in enumerate(language.SKILLS)}
    expert = PickPlaceExpert(N, dt, dev, ExpertCfg(dart_sigma_max=args.dart_sigma))
    expert.reset()
    head_b = head_in_base(env)

    def link_pose_b():
        p, q = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gi], robot.data.body_quat_w.torch[:, gi])
        return torch.cat([p, q], -1)

    def local_poses():
        return torch.stack([torch.cat([env.scene[n].data.root_pos_w.torch - env.scene.env_origins,
                                       env.scene[n].data.root_quat_w.torch], -1) for n in objects], 1).cpu().numpy()

    def rgb(cam):
        return env.scene[cam].data.output["rgb"][..., :3].to(torch.uint8).cpu().numpy()

    def new_episode(n_pick, n_place):
        skill = "pick" if n_pick <= n_place else "place"  # keep the two skills balanced
        return {"skill": skill, "mode": str(rng.choice(MODES[skill])), "u": rng.uniform(), "skip": True,
                "phase_t": 0, "last_phase": -1, "rest_z": None, "failed_at": None, "hold": 0, "pick_ok_at": None,
                "frames": {k: [] for k in ("left", "right", "state", "action", "joint_pos", "skill", "cube_pos", "plate_pos")}}

    def start_episode(i, e):
        """Called once the first fresh observation after a reset is in: task, texts, expert re-init and injection."""
        cube, plate = env.pp_task[i].tolist()
        init = local_poses()[i]
        pcol = env.pp_plate_colors[i] if hasattr(env, "pp_plate_colors") else None
        e.update({"task": (cube, plate), "init": init.astype(np.float32), "goal": language.goal_instruction(cube, plate, pcol),
                  "plate_colors": list(pcol or language.PLATE_NAMES),
                  "pick_text": language.random_skill_text("pick", object=language.cube_ref(cube)),
                  "place_text": language.random_skill_text("place", target=language.plate_ref(plate, pcol))})
        expert.reset(torch.tensor([i], device=dev))  # re-initialise from the fresh object poses
        if e["mode"] == "miss":
            ang = rng.uniform(0, 2 * np.pi)
            expert.grasp_offset[i] = torch.tensor([np.cos(ang), np.sin(ang), 0.0], device=dev) * rng.uniform(0.02, 0.035)
        if e["mode"] == "off":
            pos = init[len(CUBE_NAMES) + plate, :2]
            for _ in range(100):  # release point 9-12 cm from the plate centre, inside the workspace
                ang, d = rng.uniform(0, 2 * np.pi), rng.uniform(0.09, 0.12)
                q = pos + d * np.array([np.cos(ang), np.sin(ang)])
                if WORKSPACE_X[0] <= q[0] <= WORKSPACE_X[1] and WORKSPACE_Y[0] <= q[1] <= WORKSPACE_Y[1]:
                    break
            off_w = torch.tensor([[*(q - pos), 0.0]], dtype=torch.float32, device=dev)  # env-local = world-aligned
            expert.place_offset[i] = math_utils.quat_apply_inverse(robot.data.root_quat_w.torch[i : i + 1], off_w)[0]

    counts = {"pick": 0, "place": 0}
    eps = [new_episode(i % 2, 1 - i % 2) for i in range(N)]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    h5 = h5py.File(args.out, "w")
    h5.attrs["fps"] = round(1 / dt)
    h5.attrs["state_names"] = STATE_NAMES
    h5.attrs["skills"] = list(language.SKILLS)
    h5.attrs["calibration"] = json.dumps(calibration_dict(head_b[0]))
    saved, attempted, t0 = 0, 0, time.time()
    modes = {}
    torch.inference_mode().__enter__()

    while saved < args.episodes:
        link = link_pose_b()
        joint = robot.data.joint_pos.torch
        state = adapter.state(link[:, :3], link[:, 3:7], joint[:, 5], head_b).cpu().numpy()
        left, right = rgb("left_cam"), rgb("right_cam")
        priv = obs["privileged"].view(N, -1, 7)
        phase = expert.phase.clone()
        for i, e in enumerate(eps):
            if e["skip"] or e["failed_at"] is not None:
                continue
            cube, plate = e["task"]
            ph = int(phase[i])
            e["phase_t"] = e["phase_t"] + 1 if ph == e["last_phase"] else 0
            e["last_phase"] = ph
            t_phase, now = e["phase_t"] * dt, len(e["frames"]["action"])
            cube_p, plate_p = priv[i, cube, :3], priv[i, language.NUM_CUBES + plate, :3]
            if e["rest_z"] is None:
                e["rest_z"] = float(cube_p[2])
            lifted = float(cube_p[2]) > e["rest_z"] + 0.03
            # Injection.
            if e["mode"] == "slip" and ph == Phase.LIFT and t_phase >= 0.1 + 0.15 * e["u"]:
                expert.force_open[i] = True
            if e["mode"] == "drop" and ph == Phase.CARRY and t_phase >= 0.15 + 0.5 * e["u"]:
                expert.force_open[i] = True
            # Ground-truth outcome.
            if ph == Phase.CARRY and e["phase_t"] == 0:  # the lift just ended
                if lifted:
                    e["pick_ok_at"] = now
                elif e["skill"] == "pick":
                    e["failed_at"] = now
            if e["skill"] == "place" and e["pick_ok_at"] is not None:
                if ph in (Phase.CARRY, Phase.LOWER) and not lifted:
                    e["failed_at"] = now  # dropped while carrying
                if ph == Phase.RETREAT and float(torch.linalg.norm(cube_p[:2] - plate_p[:2])) > PLATE_RADIUS:
                    e["failed_at"] = now  # released off the plate
            if e["failed_at"] is not None:
                expert.freeze[i] = True  # stop and hold still

        ik_action = expert.act(obs["privileged"], env.pp_task, link)
        cmd = adapter.to_agnostic(expert.cmd_pos, frames.quat_to_matrix(expert.grasp_quat), 1.0 - expert.cmd_grip,
                                  head_b).cpu().numpy()
        base_in_head = frames.invert_transform(head_b)
        objs_h = (priv[..., :3] @ base_in_head[:, :3, :3].transpose(1, 2) + base_in_head[:, None, :3, 3]).cpu().numpy()
        joint_np = joint.cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(ik_action)
        ended = (terminated | truncated).cpu().numpy()
        finished = []
        for i, e in enumerate(eps):
            if e["skip"]:  # frame 0 after a reset still shows the previous layout
                e["skip"] = False
                start_episode(i, e)
                continue
            fr = e["frames"]
            cube, plate = e["task"]
            sk = e["skill"] if e["failed_at"] is not None else phase_skill(int(phase[i]))
            for k, v in (("left", left[i]), ("right", right[i]), ("state", state[i]), ("action", cmd[i]),
                         ("joint_pos", joint_np[i]), ("skill", skill_ids[sk]), ("cube_pos", objs_h[i, cube]),
                         ("plate_pos", objs_h[i, language.NUM_CUBES + plate])):
                fr[k].append(v)
            if e["failed_at"] is not None:
                e["hold"] += 1
            if e["failed_at"] is not None and e["hold"] >= HOLD_STEPS:
                finished.append((i, True))
            elif ended[i]:
                finished.append((i, False))  # ended by the env (success / time limit): not a failure demo
        for i, ok in finished:
            attempted += 1
            e = eps[i]
            fr = e["frames"]
            if ok and saved < args.episodes:
                T = len(fr["action"])
                status = np.zeros((T, 2), dtype=np.float32)
                status[e["failed_at"]:] = [1.0, 0.0]
                if e["skill"] == "place":
                    a = e["pick_ok_at"]
                    status[max(a - SUCCESS_TAIL, 0):a] = [1.0, 1.0]
                g5 = h5.create_group(f"episode_{saved:06d}")
                for key in ("goal", "pick_text", "place_text"):
                    g5.attrs[key] = e[key]
                g5.attrs["cube_idx"], g5.attrs["plate_idx"] = e["task"]
                g5.attrs["plate_colors"] = e["plate_colors"]
                g5.attrs["failed_skill"], g5.attrs["fail_mode"] = e["skill"], e["mode"]
                g5.attrs["moved_pick_mm"] = g5.attrs["moved_place_mm"] = 0.0
                for key, name in (("left", "left_rgb"), ("right", "right_rgb")):
                    arr = np.stack(fr[key])
                    g5.create_dataset(name, data=arr, chunks=(1, *arr.shape[1:]), compression="gzip", compression_opts=4)
                for key in ("state", "action", "joint_pos", "cube_pos", "plate_pos"):
                    g5.create_dataset(key, data=np.stack(fr[key]).astype(np.float32))
                g5.create_dataset("skill", data=np.asarray(fr["skill"], dtype=np.uint8))
                g5.create_dataset("status", data=status)
                g5.create_dataset("init_poses", data=e["init"])
                saved += 1
                counts[e["skill"]] += 1
                modes[e["mode"]] = modes.get(e["mode"], 0) + 1
                if saved % 10 == 0:
                    h5.flush()
                    rate = saved / (time.time() - t0) * 3600
                    print(f"[fail] saved {saved}/{args.episodes} (kept {saved}/{attempted}) {modes}, {rate:.0f} eps/h", flush=True)
            if not ended[i]:
                env._reset_idx(torch.tensor([i], device=dev))
            expert.reset(torch.tensor([i], device=dev))
            eps[i] = new_episode(counts["pick"], counts["place"])
        if any(not ended[i] for i, _ in finished):
            env.scene.write_data_to_sim()

    h5.close()
    print(f"[fail] done: {saved} episodes from {attempted} attempts {modes} -> {os.path.abspath(args.out)}")
    env.close()
