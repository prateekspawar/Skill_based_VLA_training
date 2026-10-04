"""Open-loop check: predicted action chunks vs recorded expert actions on dataset frames (~/lerobot-env).

Reports TCP position error (mm) for the first k predicted steps and gripper-opening error, split by skill.

    ~/lerobot-env/bin/python scripts/openloop_check.py --checkpoint <.../pretrained_model> --dataset datasets/lerobot/abl500_none
"""

import argparse

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_lang.serve.policy_server import load_policy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--depth", default="none", help="Depth backend for the policy's extra inputs (unused when the dataset has them).")
parser.add_argument("--episodes", type=int, nargs="+", default=[0, 50, 100, 200, 400])
parser.add_argument("--k", type=int, default=10, help="Predicted steps compared per query.")
args = parser.parse_args()

cfg, policy, pre, post, _ = load_policy(args.checkpoint, "none")
ds = LeRobotDataset("local/so101_stereo_ee", root=args.dataset)
image_keys = [k for k in ds.features if k.startswith("observation.images.")]
extra_keys = [k for k in ds.features if k == "observation.pointmap"]
errs = {"pick": [], "place": []}
grip = {"pick": [], "place": []}
with torch.no_grad():
    for ep in args.episodes:
        meta = ds.meta.episodes[ep]
        start, end = meta["dataset_from_index"], meta["dataset_to_index"]
        for i in range(start, end - args.k, 15):
            x = ds[i]
            obs = {k: x[k].unsqueeze(0).cuda() for k in image_keys + extra_keys}
            obs["observation.state"] = x["observation.state"].unsqueeze(0).cuda()
            obs["task"] = [x["task"]]
            policy.reset()
            chunk = post(policy.predict_action_chunk(pre(obs)))[0].float().cpu().numpy()[: args.k]
            gt = np.stack([ds[j]["action"].numpy() for j in range(i, i + args.k)])
            skill = "pick" if x["task"].startswith(("pick", "grab", "grasp")) else "place"
            errs[skill].append(1000 * np.linalg.norm(chunk[:, :3] - gt[:, :3], axis=1).mean())
            grip[skill].append(np.abs(chunk[:, 9] - gt[:, 9]).mean())
for s in errs:
    if errs[s]:
        print(f"{s:5s}: TCP position error over first {args.k} steps = {np.mean(errs[s]):5.1f} mm (median {np.median(errs[s]):5.1f}), "
              f"gripper opening error = {np.mean(grip[s]):.3f}  [{len(errs[s])} queries]")
