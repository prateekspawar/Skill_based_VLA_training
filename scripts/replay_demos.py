"""Replay recorded head-frame end-effector actions through the SO-101 adapter + IK and check success.

This exercises exactly the path a trained VLA will use: 10-D agnostic action -> SO101Adapter.ik_command
-> Isaac Lab pose IK -> joint targets.

    ~/isaacsim/bin/python scripts/replay_demos.py --file datasets/raw/stereo_ee.h5 --episodes 10 [--viz kit]
"""

import argparse

from so101_lang import sim_app

TASK = "SO101-PickPlace-Lang-IK-v0"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--file", required=True)
parser.add_argument("--episodes", type=int, default=10)
args, env_cfg = sim_app.parse(parser, TASK)
env_cfg.scene.num_envs = 1

with sim_app.launch(env_cfg, args):
    import h5py
    import torch

    from so101_lang.envs.head import head_in_base
    from so101_lang.envs.mdp import CUBE_NAMES, PLATE_NAMES
    from so101_lang.robots.so101 import SO101Adapter

    env = sim_app.make_env(TASK, env_cfg)
    adapter = SO101Adapter()
    ids = torch.tensor([0], device=env.device)
    ok_count = 0
    with h5py.File(args.file, "r") as h5, torch.inference_mode():
        names = sorted(h5.keys())[: args.episodes]
        for name in names:
            ep = h5[name]
            env.reset()
            for obj, pose in zip(CUBE_NAMES + PLATE_NAMES, torch.tensor(ep["init_poses"][()], device=env.device)):
                pose = pose.clone().unsqueeze(0)
                pose[:, :3] += env.scene.env_origins[0]
                env.scene[obj].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
                env.scene[obj].write_root_velocity_to_sim_index(root_velocity=torch.zeros(1, 6, device=env.device), env_ids=ids)
            env.pp_task[0] = torch.tensor([ep.attrs["cube_idx"], ep.attrs["plate_idx"]], device=env.device)
            head_b = head_in_base(env)
            actions = torch.tensor(ep["action"][()], device=env.device)
            # Repeat the first command for the dropped post-reset step, then hold the final command
            # for up to 1.5 s so the released cube can settle.
            actions = torch.cat([actions[:1], actions, actions[-1:].expand(30, -1)])
            success, reason = False, "end of actions"
            for t, a in enumerate(actions):
                _, _, terminated, truncated, _ = env.step(adapter.ik_command(a.unsqueeze(0), head_b))
                if terminated[0] or truncated[0]:
                    success = bool(env.termination_manager.get_term("success")[0])
                    reason = [n for n in env.termination_manager.active_terms if env.termination_manager.get_term(n)[0]]
                    break
            ok_count += int(success)
            print(f"[replay] {name} '{ep.attrs['goal']}' -> {'success' if success else 'FAIL'} ({reason}, {t + 1} steps)")
    print(f"[replay] {ok_count}/{len(names)} replays succeeded")
    env.close()
