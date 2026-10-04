"""Run the scripted expert, report its success rate and save a front-camera video of env 0.

    ~/isaacsim/bin/python scripts/run_expert.py --num_envs 16 --episodes 64 [--viz kit]
"""

import argparse
import os

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=32)
parser.add_argument("--out", default="outputs/expert")
parser.add_argument("--tcp_z", type=float, default=None, help="Override the TCP z offset in the gripper frame.")
parser.add_argument("--tcp_x", type=float, default=None, help="Override the TCP x offset in the gripper frame.")
args, env_cfg = sim_app.parse(parser, TASK)

with sim_app.launch(env_cfg, args):
    import collections

    import imageio
    import torch

    import isaaclab.utils.math as math_utils

    from so101_lang import language
    from so101_lang.expert.pick_place_sm import ExpertCfg, Phase, PickPlaceExpert

    env = sim_app.make_env(TASK, env_cfg)
    obs, _ = env.reset()
    robot = env.scene["robot"]
    gripper_idx = robot.find_bodies("gripper")[0][0]

    def gripper_pose_b():
        pos, quat = math_utils.subtract_frame_transforms(
            robot.data.root_pos_w.torch,
            robot.data.root_quat_w.torch,
            robot.data.body_pos_w.torch[:, gripper_idx],
            robot.data.body_quat_w.torch[:, gripper_idx],
        )
        return torch.cat([pos, quat], dim=-1)

    cfg = ExpertCfg()
    tcp = list(cfg.tcp_offset)
    if args.tcp_z is not None:
        tcp[2] = args.tcp_z
    if args.tcp_x is not None:
        tcp[0] = args.tcp_x
    cfg.tcp_offset = tuple(tcp)
    expert = PickPlaceExpert(env.num_envs, env.step_dt, env.device, cfg)
    expert.reset()

    done_eps, successes = 0, 0
    fail_phase = collections.Counter()
    stats = collections.Counter()
    frames, recording = [], True
    os.makedirs(args.out, exist_ok=True)
    torch.inference_mode().__enter__()
    while done_eps < args.episodes:
        phase_before = expert.phase.clone()
        action = expert.act(obs["privileged"], env.pp_task, gripper_pose_b())
        timed_out = expert.last_timed_out.clone()
        task_before = env.pp_task[0].tolist()
        obs, _, terminated, truncated, _ = env.step(action)
        # Diagnostics at phase transitions.
        changed = ((expert.phase != phase_before) & (expert.phase > 0)).nonzero().flatten().tolist()
        if changed:
            gp = gripper_pose_b()
            rot = math_utils.matrix_from_quat(gp[:, 3:7])
            tcp_now = gp[:, :3] + (rot @ expert.tcp_offset.unsqueeze(-1)).squeeze(-1)
            for i in changed:
                ph = Phase(int(phase_before[i])).name
                stats[ph + "_timeout"] += int(timed_out[i])
                stats[ph + "_n"] += 1
                if ph == "DESCEND":
                    goal = expert.pick_pos[i]
                    stats["descend_pos_err_mm"] += float(torch.linalg.norm(tcp_now[i] - goal)) * 1000
                    stats["descend_z_err_mm"] += float(tcp_now[i, 2] - goal[2]) * 1000
                    stats["descend_xy_err_mm"] += float(torch.linalg.norm(tcp_now[i, :2] - goal[:2])) * 1000
                    stats["descend_ori_err_deg"] += float(
                        math_utils.quat_error_magnitude(gp[i : i + 1, 3:7], expert.grasp_quat[i : i + 1])
                    ) * 57.3
                if ph == "LIFT":
                    cube = obs["privileged"][i].view(-1, 7)[env.pp_task[i, 0]]
                    stats["lifted"] += int(cube[2] > expert.pick_pos[i, 2] + 0.03)
        if recording:
            frames.append(env.scene["front_cam"].data.output["rgb"][0, ..., :3].cpu().numpy())
        ended = (terminated | truncated).nonzero().flatten()
        if len(ended):
            success = env.termination_manager.get_term("success")[ended]
            for i, ok in zip(ended.tolist(), success.tolist()):
                done_eps += 1
                successes += int(ok)
                if not ok:
                    fail_phase[Phase(int(phase_before[i])).name] += 1
                if i == 0 and recording:
                    imageio.mimsave(f"{args.out}/env0_episode.mp4", frames, fps=int(1 / env.step_dt))
                    print(f"[expert] saved video ({language.instruction(*task_before)}), success={bool(ok)}")
                    recording = False
            expert.reset(ended)
    print(f"[expert] tcp_offset={cfg.tcp_offset} success {successes}/{done_eps} = {successes / done_eps:.1%}")
    print(f"[expert] failures by phase: {dict(fail_phase)}")
    n = max(stats["DESCEND_n"], 1)
    print(f"[expert] descend: mean pos err {stats['descend_pos_err_mm'] / n:.1f} mm, ori err {stats['descend_ori_err_deg'] / n:.1f} deg, z err {stats['descend_z_err_mm'] / n:+.1f} mm, xy err {stats['descend_xy_err_mm'] / n:.1f} mm")
    print(f"[expert] lifted {stats['lifted']}/{stats['LIFT_n']}")
    print("[expert] timeouts:", {k[:-8]: f"{stats[k]}/{stats[k[:-8] + '_n']}" for k in stats if k.endswith('_timeout')})
    env.close()
