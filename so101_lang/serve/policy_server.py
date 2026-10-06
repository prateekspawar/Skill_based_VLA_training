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

import so101_lang.policies.smolvla_depth  # noqa: F401  (registers smolvla_depth for loading)
from so101_lang.data.hdf5_to_lerobot import DEPTH_KINDS, POINTMAP_SIZE, make_depth_backend
from so101_lang.perception.stereo_depth import depth_to_pointmap, depth_to_rgb

RENAME = {
    "observation.images.left": "observation.images.camera1",
    "observation.images.right": "observation.images.camera2",
    "observation.images.depth": "observation.images.camera3",
}
AUTHKEY = b"so101_lang"


def to_image(rgb: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)


def load_policy(checkpoint: str, depth_kind: str, calibration: str | None = None):
    """Load a trained policy, its pre/post-processors and (if it has a depth camera) the depth backend."""
    ckpt = Path(checkpoint)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.pretrained_path = str(ckpt)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to("cuda").eval()
    pre, post = make_pre_post_processors(
        cfg,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": "cuda"}, "rename_observations_processor": {"rename_map": RENAME}},
    )
    depth = None
    # The depth image goes to camera3 only if the model was trained that way (its rename map sends depth there);
    # the base config always declares camera3, so input_features alone cannot tell.
    try:
        rename = json.loads((ckpt / "train_config.json").read_text()).get("rename_map") or {}
    except FileNotFoundError:
        rename = {}
    wants_camera = rename.get("observation.images.depth") == "observation.images.camera3"
    wants_pointmap = getattr(cfg, "use_depth_branch", False)
    if (wants_camera or wants_pointmap) and depth_kind != "none":
        calib_path = calibration or json.loads((ckpt / "train_config.json").read_text())["dataset"]["root"] + "/meta/stereo_calibration.json"
        calib = json.loads(Path(calib_path).read_text())
        depth = make_depth_backend(depth_kind, calib)
        depth.uses = {"camera": wants_camera, "pointmap": wants_pointmap, "calibration": calib}
    return cfg, policy, pre, post, depth


def build_obs(msg: dict, depth) -> dict:
    obs = {
        "observation.images.left": to_image(msg["left"], "cuda"),
        "observation.images.right": to_image(msg["right"], "cuda"),
        # A target point (head frame, from the simulator or from perception) extends the state to 13-D.
        "observation.state": torch.as_tensor(
            np.concatenate([msg["state"], msg["target"]]) if msg.get("target") is not None else msg["state"],
            dtype=torch.float32,
        ).unsqueeze(0).cuda(),
        "task": [msg["task"]],
    }
    if depth is not None:
        z = depth.depth(msg["left"], msg["right"])
        if depth.uses["camera"]:
            obs["observation.images.depth"] = to_image(depth_to_rgb(z, depth.cfg.near, depth.cfg.far), "cuda")
        if depth.uses["pointmap"]:
            pm = depth_to_pointmap(z, depth.uses["calibration"], POINTMAP_SIZE)
            obs["observation.pointmap"] = torch.from_numpy(pm).unsqueeze(0).cuda()
    return obs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Path to a pretrained_model directory")
    parser.add_argument("--skills", nargs="+", default=None, metavar="SKILL=CHECKPOINT",
                        help="Separate skill models on one shared backbone, e.g. pick=... place=... find=...")
    parser.add_argument("--checkpoint_place", default=None,
                        help="Optional separate model for the place skill (then --checkpoint serves pick).")
    parser.add_argument("--calibration", default=None, help="stereo_calibration.json (defaults to the training dataset's)")
    parser.add_argument("--depth", choices=DEPTH_KINDS, default="fast_foundationstereo")
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()

    skills = None
    if args.skills:  # shared-backbone skill models, swapped in place
        from so101_lang.serve.skills import SkillModels

        ckpts = dict(item.split("=", 1) for item in args.skills)
        skills = SkillModels(ckpts, args.depth, args.calibration)
        cfg, depth, ckpt = skills.cfg, skills.depth, Path(next(iter(ckpts.values())))
        policy, pre, post = skills.activate(skills.active)
        models = {}
    else:
        ckpt = Path(args.checkpoint)
        cfg, policy, pre, post, depth = load_policy(args.checkpoint, args.depth, args.calibration)
        models = {"pick": (policy, pre, post)}
        if args.checkpoint_place:
            _, p_policy, p_pre, p_post, _ = load_policy(args.checkpoint_place, args.depth, args.calibration)
            models["place"] = (p_policy, p_pre, p_post)
    active = "pick"
    locator = None  # created on the first "locate"/"track" request
    sam2 = None  # SAM2 point tracker, created on the first "sam2_start" request
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
                        if skills is not None:
                            policy, pre, post = skills.activate(msg.get("skill", skills.active))
                        else:
                            active = msg.get("skill", "pick") if msg.get("skill") in models else "pick"
                            policy, pre, post = models[active]
                        policy.reset()
                        if msg.get("seed") is not None:  # reproducible flow-matching noise for paired comparisons
                            torch.manual_seed(int(msg["seed"]))
                        conn.send({"ok": True, "swap_ms": skills.last_swap_ms if skills else 0.0})
                        continue
                    if msg.get("cmd") in ("locate", "locate9", "sam2_start", "sam2_update"):  # perception -> 3-D points (head frame)
                        if locator is None:
                            from so101_lang.perception.locate import TargetLocator

                            calib_path = args.calibration or json.loads((ckpt / "train_config.json").read_text())["dataset"]["root"] + "/meta/stereo_calibration.json"
                            locator = TargetLocator(json.loads(Path(calib_path).read_text()))
                        t0 = time.time()
                        if msg["cmd"] == "locate":
                            point, info = locator.locate(msg["left"], msg["right"], msg["query"])
                            conn.send({"point": point, **info, "t_locate": time.time() - t0})
                            continue
                        if msg["cmd"] == "locate9":  # find(): 9-D pose of the named object
                            from so101_lang.perception.locate import locate9

                            pose, info = locate9(locator, msg["left"], msg["right"], msg["query"])
                            conn.send({"pose9": pose, **info, "t_locate": time.time() - t0})
                            continue
                        if sam2 is None:
                            from so101_lang.perception.sam2_tracker import Sam2PointTracker

                            sam2 = Sam2PointTracker(locator)
                        robot_points = msg.get("robot_points")  # robot keypoints (head frame) for the self-filter
                        if msg["cmd"] == "sam2_start":
                            points, info = sam2.start(msg["left"], msg["right"], msg["queries"], robot_points)
                        else:
                            points, info = sam2.update(
                                msg["left"], msg["right"], robot_points, frozen=tuple(msg.get("frozen", ()))
                            )
                        conn.send({"points": points, "info": info, "t_locate": time.time() - t0})
                        continue
                    t0 = time.time()
                    obs = build_obs(msg, depth)
                    t_depth = time.time() - t0
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(pre(obs))  # (1, T, D), normalized
                        chunk = post(chunk)
                    conn.send({"actions": chunk[0].float().cpu().numpy(), "t_depth": t_depth, "t_total": time.time() - t0})
                print("[server] client disconnected", flush=True)


if __name__ == "__main__":
    main()
