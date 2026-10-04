"""Serve a trained LeRobot policy to the simulator over a local socket (run in ``~/lerobot-env``).

The simulator process (``~/isaacsim``) sends ``{"left", "right" (H,W,3 uint8), "state" (10,), "task" (str)}``
and receives an action chunk ``(T, 10)`` of head-frame end-effector targets. Depth for the third camera is
computed here from the stereo pair with the same backend used to build the training dataset.

    ~/lerobot-env/bin/python -m so101_lang.serve.policy_server --checkpoint checkpoints/smolvla/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from so101_lang.data.hdf5_to_lerobot import make_depth_backend

RENAME = {
    "observation.images.left": "observation.images.camera1",
    "observation.images.right": "observation.images.camera2",
    "observation.images.depth": "observation.images.camera3",
}
AUTHKEY = b"so101_lang"


def to_image(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a pretrained_model directory")
    parser.add_argument("--calibration", default=None, help="stereo_calibration.json (defaults to the training dataset's)")
    parser.add_argument("--depth", choices=["foundationstereo", "sgbm", "none"], default="foundationstereo")
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = str(ckpt)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME}},
    )
    uses_depth = "observation.images.camera3" in cfg.input_features and args.depth != "none"
    depth = None
    if uses_depth:
        calib_path = args.calibration or json.loads((ckpt / "train_config.json").read_text())["dataset"]["root"] + "/meta/stereo_calibration.json"
        depth = make_depth_backend(args.depth, json.loads(Path(calib_path).read_text()))
    print(f"[server] {cfg.type} loaded from {ckpt}; depth={'on' if depth else 'off'}; listening on :{args.port}", flush=True)

    with Listener(("localhost", args.port), authkey=AUTHKEY) as listener:
        while True:
            with listener.accept() as conn:
                print("[server] client connected", flush=True)
                while True:
                    try:
                        msg = conn.recv()
                    except EOFError:
                        break
                    if msg.get("cmd") == "reset":
                        policy.reset()
                        conn.send({"ok": True})
                        continue
                    t0 = time.time()
                    obs = {
                        "observation.images.left": to_image(msg["left"], "cuda"),
                        "observation.images.right": to_image(msg["right"], "cuda"),
                        "observation.state": torch.as_tensor(msg["state"], dtype=torch.float32).unsqueeze(0).cuda(),
                        "task": [msg["task"]],
                    }
                    if depth is not None:
                        obs["observation.images.depth"] = to_image(depth.depth_rgb(msg["left"], msg["right"]), "cuda")
                    t_depth = time.time() - t0
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(pre(obs))  # (1, T, D), normalized
                        chunk = post(chunk)
                    conn.send({"actions": chunk[0].float().cpu().numpy(), "t_depth": t_depth, "t_total": time.time() - t0})
                print("[server] client disconnected", flush=True)


if __name__ == "__main__":
    main()
