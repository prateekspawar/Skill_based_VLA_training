"""Replay recorded joint-target actions open-loop in the joint-position env and check success.

Verifies that the recorded ``action`` space is what the learned policies will drive.

    ~/isaacsim/bin/python scripts/replay_demos.py --file datasets/raw/demos_000.h5 --episodes 10 [--viz kit]
"""

import argparse

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-Joint-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--debug", action="store_true")
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.scene.num_envs = 1

with sim_app.launch(env_cfg, args):
    import h5py
    import torch

    from so101_lang.envs.mdp import CUBE_NAMES, PLATE_NAMES

    env = sim_app.make_env(TASK, env_cfg)
    ids = torch.tensor([0], device=env.device)
    ok_count = 0
    with h5py.File(args.file, "r") as h5, torch.inference_mode():
        names = sorted(h5.keys())[: args.episodes]
        for name in names:
            ep = h5[name]
            env.reset()
            # Restore the recorded layout and task.
            for obj, pose in zip(CUBE_NAMES + PLATE_NAMES, torch.tensor(ep["init_poses"][()], device=env.device)):
                pose = pose.clone().unsqueeze(0)
                pose[:, :3] += env.scene.env_origins[0]
                env.scene[obj].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                env.scene[obj].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
            env.pp_task[0] = torch.tensor([ep.attrs["cube_idx"], ep.attrs["plate_idx"]], device=env.device)
            # The recording drops the first post-reset step, so replay it as "hold the start pose".
            env.step(env.scene["robot"].data.joint_pos.torch.clone())
            success, reason, max_err = False, "end of actions", 0.0
            states = torch.tensor(ep["state"][()], device=env.device)
            robot = env.scene["robot"]
            actions = torch.tensor(ep["action"][()], device=env.device)
            # Hold the final target for up to 1.5 s so the released cube can settle.
            actions = torch.cat([actions, actions[-1:].expand(30, -1)])
            for t, a in enumerate(actions):
                if t < len(states):
                    max_err = max(max_err, float((robot.data.joint_pos.torch[0] - states[t]).abs().max()))
                if t in (0, 1, 2, 20, 60) and args.debug:
                    print(f"   t={t} sim={robot.data.joint_pos.torch[0].cpu().numpy().round(3)} rec={states[t].cpu().numpy().round(3)} act={a.cpu().numpy().round(3)}")
                _, _, terminated, truncated, _ = env.step(a.unsqueeze(0))
                if terminated[0] or truncated[0]:
                    success = bool(env.termination_manager.get_term("success")[0])
                    reason = [n for n in env.termination_manager.active_terms if env.termination_manager.get_term(n)[0]]
                    break
            print(f"   steps={t + 1}/{len(states)} ended={reason} max joint err vs recording={max_err:.3f} rad")
            if args.debug:
                c, p = env.pp_task[0].tolist()
                cube = env.scene[CUBE_NAMES[c]].data.root_pos_w.torch[0] - env.scene.env_origins[0]
                plate = env.scene[PLATE_NAMES[p]].data.root_pos_w.torch[0] - env.scene.env_origins[0]
                rec = ep["init_poses"][()]
                print(f"   cube now {cube.cpu().numpy().round(3)} (rec start {rec[c, :3].round(3)}), plate now {plate.cpu().numpy().round(3)} (rec {rec[4 + p, :3].round(3)}), gripper {robot.data.joint_pos.torch[0, 5]:.2f}")
            ok_count += int(success)
            print(f"[replay] {name} '{ep.attrs['instruction']}' -> {'success' if success else 'FAIL'}")
    print(f"[replay] {ok_count}/{len(names)} replays succeeded")
    env.close()
