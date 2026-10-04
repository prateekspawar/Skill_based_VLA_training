"""Does the policy use its inputs? Perturb one input at a time and measure how much predictions move (~/lerobot-env).

For dataset frames, predict an action chunk (fixed noise), then re-predict with
  * images blacked out,
  * images taken from a different episode (different layout),
  * the task text's object colour swapped,
  * the state shifted by a small amount,
and report the mean change in predicted TCP position (mm) over the first 10 steps.

    ~/lerobot-env/bin/python scripts/sensitivity_check.py --checkpoint <.../pretrained_model> --dataset <root>
"""

import argparse

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_lang.serve.policy_server import load_policy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--dataset", required=True)
parser.add_argument("--frames", type=int, default=20)
args = parser.parse_args()

cfg, policy, pre, post, _ = load_policy(args.checkpoint, "none")
ds = LeRobotDataset("local/so101_stereo_ee", root=args.dataset)
image_keys = [k for k in ds.features if k.startswith("observation.images.")]
colors = ["red", "green", "blue", "yellow"]
rng = np.random.default_rng(0)
idx = rng.choice(len(ds), args.frames, replace=False)


def predict(x, task):
    obs = {k: x[k].unsqueeze(0).cuda() for k in image_keys}
    obs["observation.state"] = x["observation.state"].unsqueeze(0).cuda()
    obs["task"] = [task]
    policy.reset()
    torch.manual_seed(0)
    noise = torch.randn(1, cfg.chunk_size, cfg.max_action_dim, device="cuda")
    with torch.no_grad():
        return post(policy.predict_action_chunk(pre(obs), noise=noise))[0, :10, :3].float().cpu().numpy()


deltas = {"same input (noise check)": [], "black images": [], "other episode's images": [], "colour swapped in text": [],
          "state +2 cm in x": []}
for i in idx:
    x = ds[int(i)]
    base = predict(x, x["task"])
    deltas["same input (noise check)"].append(np.linalg.norm(predict(x, x["task"]) - base, axis=1).mean())
    xb = dict(x)
    for k in image_keys:
        xb[k] = torch.zeros_like(x[k])
    deltas["black images"].append(np.linalg.norm(predict(xb, x["task"]) - base, axis=1).mean())
    other = ds[int((i + len(ds) // 2) % len(ds))]
    xo = dict(x)
    for k in image_keys:
        xo[k] = other[k]
    deltas["other episode's images"].append(np.linalg.norm(predict(xo, x["task"]) - base, axis=1).mean())
    task = x["task"]
    c = next((c for c in colors if c in task), None)
    if c:
        swapped = task.replace(c, colors[(colors.index(c) + 1) % 4])
        deltas["colour swapped in text"].append(np.linalg.norm(predict(x, swapped) - base, axis=1).mean())
    xs = dict(x)
    xs["observation.state"] = x["observation.state"].clone()
    xs["observation.state"][0] += 0.02
    deltas["state +2 cm in x"].append(np.linalg.norm(predict(xs, x["task"]) - base, axis=1).mean())

for k, v in deltas.items():
    print(f"{k:28s}: predictions move {1000 * np.mean(v):6.1f} mm on average  [{len(v)} frames]")
