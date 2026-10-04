"""End-to-end per-query latency (stereo depth + SmolVLA action chunk) for each depth option (~/lerobot-env).

Uses the exact server code path (load_policy / build_obs). For "none" the third camera receives a
black image, which costs the same model compute as the empty-camera slot of an RGB-only model.

    ~/lerobot-env/bin/python scripts/bench_latency.py --checkpoint checkpoints/smoke_smolvla/checkpoints/last/pretrained_model
"""

import argparse
import json
import subprocess
import time

import numpy as np
import torch

from so101_lang.data.hdf5_to_lerobot import make_depth_backend
from so101_lang.serve.policy_server import build_obs, load_policy, to_image


def gpu_mem_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--frames", default="outputs/stereo_depth/frames.npz")
parser.add_argument("--kinds", nargs="+", default=["none", "sgbm", "fast_foundationstereo", "foundationstereo"])
parser.add_argument("--queries", type=int, default=30)
parser.add_argument("--out", default="outputs/latency.json")
args = parser.parse_args()

data = np.load(args.frames)
calib = json.loads(str(data["calibration"]))
mem0 = gpu_mem_mb()
cfg, policy, pre, post, _ = load_policy(args.checkpoint, "none")
policy_mem = gpu_mem_mb() - mem0
state = np.zeros(10, dtype=np.float32)
results = {}
for kind in args.kinds:
    mem_before = gpu_mem_mb()
    depth = make_depth_backend(kind, calib) if kind != "none" else None
    t_depth, t_policy = [], []
    for q in range(args.queries + 3):  # first 3 are warm-up (engine load, cudnn autotune)
        i = q % len(data["left"])
        msg = {"left": data["left"][i], "right": data["right"][i], "state": state, "task": "pick up the red cube"}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        obs = build_obs(msg, depth)
        if depth is None:
            obs["observation.images.depth"] = to_image(np.zeros_like(msg["left"]), "cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            chunk = post(policy.predict_action_chunk(pre(obs)))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if q >= 3:
            t_depth.append(t1 - t0)
            t_policy.append(t2 - t1)
    depth_mem = gpu_mem_mb() - mem_before
    del depth
    r = {
        "depth_ms": 1000 * float(np.median(t_depth)),
        "policy_ms": 1000 * float(np.median(t_policy)),
        "total_ms": 1000 * float(np.median(np.add(t_depth, t_policy))),
        "depth_gpu_mb": depth_mem,
    }
    results[kind] = r
    print(f"{kind:22s} depth {r['depth_ms']:6.1f} ms + policy {r['policy_ms']:6.1f} ms = {r['total_ms']:6.1f} ms/query  (depth +{depth_mem} MB GPU)", flush=True)

results["_policy_gpu_mb"] = policy_mem
results["_chunk_len"] = int(chunk.shape[1])
json.dump(results, open(args.out, "w"), indent=2)
print(f"policy GPU memory ~{policy_mem} MB; chunk length {chunk.shape[1]} actions -> {args.out}")
