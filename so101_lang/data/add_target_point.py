"""Append the skill's target point (head frame) to observation.state of an existing LeRobot dataset.

Pick frames get the target cube's centre, place frames the target plate's centre, both from the recorded
initial poses in the raw HDF5 (cubes are static until grasped, plates never move). The new dataset links
the original videos instead of copying them.

    ~/lerobot-env/bin/python -m so101_lang.data.add_target_point --src datasets/lerobot/s512_dart_none \
        --raw datasets/raw/s512_dart_1000.h5 --dst datasets/lerobot/s512_dart_target
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so101_lang import language

# SO-101 base seat in the env frame (isaaclab_tasks .../so101/stack_joint_pos_env_cfg.py).
BASE_POS = np.array([0.0, 0.0, -0.03008])
BASE_QUAT_XYZW = np.array([0.0, 0.0, 0.70710678, 0.70710678])
TARGET_NAMES = ["target.x", "target.y", "target.z"]


def quat_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def env_to_head(points_env: np.ndarray, head_in_base: np.ndarray) -> np.ndarray:
    r_b = quat_to_matrix(BASE_QUAT_XYZW)
    p_base = (points_env - BASE_POS) @ r_b  # R^T (p - t)
    r_h, t_h = head_in_base[:3, :3], head_in_base[:3, 3]
    return (p_base - t_h) @ r_h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    os.symlink((src / "videos").resolve(), dst / "videos")
    shutil.copytree(src / "meta", dst / "meta")

    calib = json.loads((src / "meta" / "stereo_calibration.json").read_text())
    head_in_base = np.array(calib["head_in_base"])
    # Per-episode target points: [pick target (cube), place target (plate)] in the head frame.
    targets = []
    with h5py.File(args.raw, "r") as h5:
        for name in sorted(h5.keys()):
            ep = h5[name]
            poses = ep["init_poses"][()]
            cube = poses[int(ep.attrs["cube_idx"]), :3]
            plate = poses[language.NUM_CUBES + int(ep.attrs["plate_idx"]), :3]
            targets.append(env_to_head(np.stack([cube, plate]), head_in_base))
    targets = np.asarray(targets, dtype=np.float32)

    states = []
    for f in sorted((src / "data").rglob("*.parquet")):
        table = pq.read_table(f)
        ep = table.column("episode_index").to_numpy()
        skill = np.stack(table.column("skill_index").to_numpy(zero_copy_only=False)).reshape(-1).astype(int)
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        new_state = np.concatenate([state, targets[ep, skill]], axis=1)
        states.append(new_state)
        col = pa.FixedSizeListArray.from_arrays(pa.array(new_state.reshape(-1), pa.float32()), new_state.shape[1])
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", col)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feat = info["features"]["observation.state"]
    feat["shape"] = [len(feat["names"]) + 3]
    feat["names"] = feat["names"] + TARGET_NAMES
    info_path.write_text(json.dumps(info, indent=4))

    all_states = np.concatenate(states)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    s = stats["observation.state"]
    for key in list(s):
        if key == "count":
            continue
        if key == "mean":
            s[key] = all_states.mean(0).tolist()
        elif key == "std":
            s[key] = all_states.std(0).tolist()
        elif key == "min":
            s[key] = all_states.min(0).tolist()
        elif key == "max":
            s[key] = all_states.max(0).tolist()
        elif key.startswith("q") and key[1:].isdigit():
            s[key] = np.quantile(all_states, int(key[1:]) / 100, axis=0).tolist()
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[target] wrote {len(all_states)} frames with 13-D state -> {dst}")


if __name__ == "__main__":
    main()
