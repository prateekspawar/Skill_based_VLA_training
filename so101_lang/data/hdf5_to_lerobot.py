"""Convert ``scripts/gen_demos.py`` HDF5 files into a local LeRobotDataset (v3: parquet + mp4).

Run in the ``~/lerobot-env`` venv:

    ~/lerobot-env/bin/python -m so101_lang.data.hdf5_to_lerobot \
        --inputs datasets/raw/stereo_ee.h5 --root datasets/lerobot/so101_stereo_ee

Policy-facing features are robot-agnostic:

* ``observation.images.left`` / ``observation.images.right`` - the stereo head pair
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

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def features(height: int, width: int, state_names: list[str]) -> dict:
    image = {"dtype": "video", "shape": (height, width, 3), "names": ["height", "width", "channels"]}
    vec = {"dtype": "float32", "shape": (len(state_names),), "names": state_names}
    return {
        "observation.images.left": image,
        "observation.images.right": dict(image),
        "observation.state": vec,
        "action": dict(vec),
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
    args = parser.parse_args()

    root = Path(args.root)
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f"{root} exists; pass --overwrite to replace it")
        shutil.rmtree(root)

    dataset, calibration, count = None, None, 0
    for path in args.inputs:
        with h5py.File(path, "r") as h5:
            fps = int(h5.attrs["fps"])
            skills = list(h5.attrs["skills"])
            calibration = calibration or json.loads(h5.attrs["calibration"])
            for name in sorted(h5.keys()):
                if args.max_episodes is not None and count >= args.max_episodes:
                    break
                ep = h5[name]
                left, right = ep["left_rgb"][()], ep["right_rgb"][()]
                if dataset is None:
                    dataset = LeRobotDataset.create(
                        repo_id=args.repo_id,
                        fps=fps,
                        features=features(*left.shape[1:3], list(h5.attrs["state_names"])),
                        root=root,
                        robot_type="so101_follower",
                        use_videos=True,
                        image_writer_threads=8,
                    )
                texts = {"pick": str(ep.attrs["pick_text"]), "place": str(ep.attrs["place_text"])}
                state, action = ep["state"][()], ep["action"][()]
                joints, skill = ep["joint_pos"][()], ep["skill"][()]
                for t in range(len(action)):
                    dataset.add_frame(
                        {
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
                if count % 50 == 0:
                    print(f"[convert] {count} episodes", flush=True)
    if dataset is None:
        raise SystemExit("no episodes found")
    dataset.finalize()
    (root / "meta" / "stereo_calibration.json").write_text(json.dumps(calibration, indent=2))
    print(f"[convert] wrote {count} episodes, {dataset.meta.total_frames} frames -> {root.resolve()}")


if __name__ == "__main__":
    main()
