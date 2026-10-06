"""Convert ``scripts/gen_demos.py`` HDF5 files into a local LeRobotDataset (v3: parquet + mp4).

Run in the ``~/lerobot-env`` venv:

    ~/lerobot-env/bin/python -m so101_lang.data.hdf5_to_lerobot \
        --inputs datasets/raw/stereo_ee.h5 --root datasets/lerobot/so101_stereo_ee

Policy-facing features are robot-agnostic:

* ``observation.images.left`` / ``observation.images.right`` - the stereo head pair
* ``observation.images.depth`` - depth computed from the pair (FoundationStereo by default), metric
  0.25-0.85 m encoded with the TURBO colormap (near = red, far = blue), aligned with the left image
* ``observation.state`` - TCP pose in the head frame (pos + rot6d) + gripper opening (10-D)
* ``action`` - commanded TCP pose in the head frame + commanded gripper opening (10-D)
* ``task`` (per frame) - the natural-language step of the skill being executed

``so101.joint_pos`` and ``skill_index`` are kept for analysis; LeRobot only feeds ``observation.*``
keys to policies, so they never reach the model. The stereo calibration is written to
``meta/stereo_calibration.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_lang.perception.stereo_depth import depth_to_pointmap, depth_to_rgb
from so101_lang.data.status import STATUS_NAMES, default_status

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


FS_ONNX = "models/foundationstereo_s/onnx/deployable_foundation_stereo_s_dynamic.onnx"
FFS_ONNX = "models/fast_foundationstereo/onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx"
DEPTH_KINDS = ["fast_foundationstereo", "foundationstereo", "sgbm", "none"]


def make_depth_backend(kind: str, calibration: dict):
    """fast_foundationstereo: TensorRT FP16 @320 (~9 ms); foundationstereo: TensorRT FP32 @448 (~300 ms); sgbm: OpenCV."""
    from so101_lang.perception.stereo_depth import FoundationStereoDepth, StereoDepth, StereoDepthCfg

    cfg = StereoDepthCfg.from_calibration(calibration)
    if kind == "fast_foundationstereo":
        return FoundationStereoDepth(cfg, FFS_ONNX, infer_size=320, normalize="imagenet", tensorrt=True, fp16=True)
    if kind == "foundationstereo":
        return FoundationStereoDepth(cfg, FS_ONNX, infer_size=448, normalize="imagenet")
    if kind == "sgbm":
        return StereoDepth(cfg)
    return None


POINTMAP_SIZE = 64


def features(height: int, width: int, state_names: list[str], with_depth: bool, with_pointmap: bool = False,
             with_status: bool = False) -> dict:
    image = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}
    act_names = state_names[:10] + (STATUS_NAMES if with_status else [])  # TCP pose + gripper opening (+ verdict)
    act = {"dtype": "float32", "shape": (len(act_names),), "names": act_names}
    depth = {"observation.images.depth": dict(image)} if with_depth else {}
    if with_pointmap:
        depth["observation.pointmap"] = {"dtype": "float32", "shape": (3, POINTMAP_SIZE, POINTMAP_SIZE), "names": None}
    return {
        "observation.images.left": image,
        "observation.images.right": dict(image),
        **depth,
        "observation.state": vec,
        "action": act,
        "so101.joint_pos": {"dtype": "float32", "shape": (len(JOINTS),), "names": [f"{j}.pos" for j in JOINTS]},
        "skill_index": {"dtype": "int64", "shape": (1,), "names": None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="HDF5 files from gen_demos.py")
    parser.add_argument("--root", required=True, help="Output dataset directory")
    parser.add_argument("--repo_id", default="local/so101_stereo_ee")
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--depth", choices=DEPTH_KINDS, default="fast_foundationstereo",
                        help="Backend for the colormap depth camera (observation.images.depth); 'none' omits it.")
    parser.add_argument("--pad_state", type=int, default=0,
                        help="Append this many zeros to observation.state (e.g. 9 for find: no target pose yet).")
    parser.add_argument("--skill", choices=["all", "pick", "place", "find"], default="all",
                        help="Keep only the frames of one skill (each episode becomes that skill's segment).")
    parser.add_argument("--status", action="store_true",
                        help="12-D action: append the skill verdict [done, success] (so101_lang.data.status).")
    parser.add_argument("--pointmap", choices=DEPTH_KINDS, default="none",
                        help="Backend for observation.pointmap, the (3,64,64) head-frame XYZ map for smolvla_depth.")
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    dataset, calibration, count, depth, pm_backend = None, None, 0, None, None
    for path in args.inputs:
        with h5py.File(path, "r") as h5:
            fps = int(h5.attrs["fps"])
            skills = list(h5.attrs["skills"])
            calibration = calibration or json.loads(h5.attrs["calibration"])
            if depth is None and args.depth != "none":
                depth = make_depth_backend(args.depth, calibration)
            if pm_backend is None and args.pointmap != "none":
                pm_backend = depth if args.pointmap == args.depth else make_depth_backend(args.pointmap, calibration)
            for name in sorted(h5.keys()):
                if args.max_episodes is not None and count >= args.max_episodes:
                    break
                ep = h5[name]
                left, right = ep["left_rgb"][()], ep["right_rgb"][()]
                state_names = list(h5.attrs["state_names"]) + [
                    f"target.{c}" for c in ("x", "y", "z", "r11", "r21", "r31", "r12", "r22", "r32")][: args.pad_state]
                if dataset is None:
                    dataset = LeRobotDataset.create(
                        repo_id=args.repo_id,
                        fps=fps,
                        features=features(*left.shape[1:3], state_names, depth is not None, pm_backend is not None,
                                          args.status),
                        root=root,
                        robot_type="so101_follower",
                        use_videos=True,
                        image_writer_threads=8,
                    )
                texts = {sk: str(ep.attrs.get(f"{sk}_text", "")) for sk in skills}
                state, action = ep["state"][()], ep["action"][()]
                if args.pad_state:
                    state = np.concatenate([state, np.zeros((len(state), args.pad_state), state.dtype)], axis=1)
                joints, skill = ep["joint_pos"][()], ep["skill"][()]
                if args.status:
                    status = ep["status"][()] if "status" in ep else default_status(skill)
                    action = np.concatenate([action, status.astype(action.dtype)], axis=1)
                frames_kept = range(len(action))
                if args.skill != "all":
                    frames_kept = np.nonzero(skill == skills.index(args.skill))[0]
                    if len(frames_kept) < 5:
                        continue
                for t in frames_kept:
                    extra = {}
                    z = None
                    if depth is not None:
                        z = depth.depth(left[t], right[t])
                        extra["observation.images.depth"] = depth_to_rgb(z, depth.cfg.near, depth.cfg.far)
                    if pm_backend is not None:
                        if z is None or pm_backend is not depth:
                            z = pm_backend.depth(left[t], right[t])
                        extra["observation.pointmap"] = depth_to_pointmap(z, calibration, POINTMAP_SIZE)
                    dataset.add_frame(
                        {
                            **extra,
                            "observation.images.left": left[t],
                            "observation.images.right": right[t],
                            "observation.state": state[t].astype(np.float32),
                            "action": action[t].astype(np.float32),
                            "so101.joint_pos": joints[t].astype(np.float32),
                            "skill_index": np.array([skill[t]], dtype=np.int64),
                            "task": texts[skills[int(skill[t])]],
                        }
                    )
                dataset.save_episode()
                count += 1
                if count % 10 == 0:
                    print(f"[convert] {count} episodes", flush=True)
    if dataset is None:
        raise SystemExit("no episodes found")
    dataset.finalize()
    (root / "meta" / "stereo_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"[convert] wrote {count} episodes, {dataset.meta.total_frames} frames -> {root.resolve()}")


if __name__ == "__main__":
    main()
