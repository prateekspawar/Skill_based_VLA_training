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
parser.add_argument("--video_out", default="outputs/eval/episode0.mp4", help="Left-camera video of episode 0.")
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
    from so101_lang.robots.so101 import SO101Adapter

    if args.heldout_pairs:
        env_cfg.events.reset_layout_and_task.params["pairs"] = language.heldout_pairs()
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

    def rgb(cam):
        return env.scene[cam].data.output["rgb"][0, ..., :3].to(torch.uint8).cpu().numpy()

    results, frames, latencies = [], [], []
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
        done = False
        for skill, text in steps:
            conn.send({"cmd": "reset"})
            conn.recv()
            t = 0
            while t < max_steps and not done:
                reply_req = {"left": rgb("left_cam"), "right": rgb("right_cam"), "state": state_now(head_b)[0].cpu().numpy(), "task": text}
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
                    _, _, term, trunc, _ = env.step(adapter.ik_command(a.unsqueeze(0), head_b))
                    t += 1
                    if ep == 0:
                        frames.append(rgb("left_cam"))
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
        print(f"[eval] {ep:3d} '{outcome['goal']}': pick={'ok' if outcome['pick'] else 'FAIL'} "
              f"place={'ok' if outcome['place'] else 'FAIL'} | {close_txt}", flush=True)
        if ep == 0 and frames:
            os.makedirs(os.path.dirname(args.video_out), exist_ok=True)
            imageio.mimsave(args.video_out, frames, fps=round(1 / env.step_dt))

    n = len(results)
    pick = sum(r["pick"] for r in results)
    full = sum(r["place"] for r in results)
    print(f"[eval] pick success {pick}/{n} = {pick / n:.0%}; full pick+place {full}/{n} = {full / n:.0%}; "
          f"mean server latency {1000 * np.mean(latencies):.0f} ms")
    conn.close()
    env.close()
