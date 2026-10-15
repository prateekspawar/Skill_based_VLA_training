"""Append the skill's target point to observation.state (10-D -> 13-D), emulating the evaluation's SAM2 targets.

Per episode and skill the target is built from the true object positions exactly the way the closed-loop
pipeline publishes it (scripts/eval_policy.py --target sam2 + perception/sam2_tracker.py):

  * start: the object's position at the first frame + a per-episode detection error (``--det_sigma``),
  * the published point only changes once the object has moved more than ``--deadband`` from its start,
    and then becomes start + displacement (+ a small per-update error ``--track_sigma``),
  * updates happen every ``--pick_every`` / ``--place_every`` frames (SAM2 update rate),
  * pick targets freeze once the TCP is within 5 cm of them, place targets within 6 cm.

True per-frame positions come from the raw recordings (``cube_pos`` / ``plate_pos``); recordings without them
(static datasets) use the initial poses. Videos are linked, not copied.

    ~/lerobot-env/bin/python -m so101_lang.data.build_targets --src datasets/lerobot/s512_mix_none \
        --raw datasets/raw/s512_dart_1000.h5 datasets/raw/s512_dart_moving_1000.h5 --dst datasets/lerobot/s512_mix_target
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
from so101_lang.data.add_target_point import TARGET_NAMES, env_to_head

FREEZE_RADIUS = {0: 0.05, 1: 0.06, 3: 0.06}  # skill index -> [m]
# skill index -> which recorded object is its target: 0 = cube_pos (pick), 1 = plate_pos (place; for stack demos
# plate_pos holds the base cube)
TARGET_SLOT = {0: 0, 1: 1, 3: 1}


def true_positions(ep, head_in_base):
    """(T?, 2, 3) per-frame [cube, plate] head-frame positions, or a single constant row for static recordings."""
    if "cube_pos" in ep:
        return np.stack([ep["cube_pos"][()], ep["plate_pos"][()]], axis=1)
    poses = ep["init_poses"][()]
    c, p = int(ep.attrs["cube_idx"]), language.NUM_CUBES + int(ep.attrs["plate_idx"])
    return env_to_head(np.stack([poses[c, :3], poses[p, :3]]), head_in_base)[None]


def wrap_yaw(yaw: float) -> float:
    """Wrap a yaw to [-pi, pi). No symmetry folding: the true object orientation is kept, so asymmetric objects
    are represented correctly (for symmetric ones the model learns the equivalence)."""
    return float((yaw + np.pi) % (2 * np.pi) - np.pi)


def is_stack(ep) -> bool:
    return "stack_text" in ep.attrs


def true_yaws(ep):
    """Base-frame yaw of the episode's [cube, plate]: cube from its recorded orientation, plate 0 (symmetric); for
    stack demos the second object is the base cube (its recorded orientation)."""
    poses = ep["init_poses"][()]

    def yaw_base(i):
        x, y, z, w = poses[i, 3:7]
        yaw_env = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return wrap_yaw(yaw_env - np.pi / 2)  # the robot base is rotated +90 deg about z in the env frame

    second = yaw_base(int(ep.attrs["plate_idx"])) if is_stack(ep) else 0.0
    return np.array([yaw_base(int(ep.attrs["cube_idx"])), second])


def rot6d_head(yaw_base: float, head_in_base: np.ndarray) -> np.ndarray:
    """6-D rotation (first two columns) of a z-rotation by ``yaw_base`` in the base frame, expressed in the head frame."""
    c, s_ = np.cos(yaw_base), np.sin(yaw_base)
    r_base = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    r_head = head_in_base[:3, :3].T @ r_base
    return np.concatenate([r_head[:, 0], r_head[:, 1]]).astype(np.float32)


def published_targets(pos, tcp, skill, rng, args):
    """Emulate the closed-loop target stream for one episode. pos: (T or 1, 2, 3); tcp: (T, 3); skill: (T,)."""
    T = len(tcp)
    pos = np.broadcast_to(pos, (T, 2, 3)) if len(pos) == 1 else pos
    every = {0: args.pick_every, 1: args.place_every}
    start = pos[0].copy()
    det_err = rng.normal(0, args.det_sigma, size=(2, 3))
    published = start + det_err
    frozen = np.zeros(2, dtype=bool)
    out = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        s = TARGET_SLOT.get(int(skill[t]), 0)
        fr = FREEZE_RADIUS.get(int(skill[t]), 0.05)
        for obj in (0, 1):
            if frozen[obj] or t % every[obj]:
                continue
            disp = pos[t, obj] - start[obj]
            if np.linalg.norm(disp) > args.deadband:
                published[obj] = start[obj] + det_err[obj] + disp + rng.normal(0, args.track_sigma, 3)
            else:
                published[obj] = start[obj] + det_err[obj]
        # Freeze the active skill's target once the gripper is close (as eval does).
        if np.linalg.norm(tcp[t] - published[s]) < fr:
            frozen[s] = True
        out[t] = published[s]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="LeRobot dataset with 10-D state")
    ap.add_argument("--raw", nargs="+", required=True, help="Raw HDF5 files in the same episode order as --src")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--det_sigma", type=float, default=0.005)
    ap.add_argument("--track_sigma", type=float, default=0.003)
    ap.add_argument("--deadband", type=float, default=0.025)
    ap.add_argument("--pick_every", type=int, default=3)
    ap.add_argument("--place_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rot6d", action="store_true",
                    help="Append the target's 6-D rotation (head frame): 9-D target, 19-D state.")
    ap.add_argument("--yaw_sigma_deg", type=float, default=3.0, help="Per-episode yaw detection error (gauss).")
    ap.add_argument("--yaw_noise", choices=["gauss", "empirical"], default="gauss",
                    help="empirical: per-episode yaw error drawn from the errors perception actually makes "
                         "(data/yaw_error_samples.json, measured in find-pipeline evaluations), random sign.")
    ap.add_argument("--skill", choices=["all", "pick", "place"], default="all",
                    help="Must match the --skill used to convert --src (per-skill datasets).")
    args = ap.parse_args()
    yaw_samples = None
    if args.yaw_noise == "empirical":
        yaw_samples = np.asarray(json.loads((Path(__file__).parent / "yaw_error_samples.json").read_text())["abs_deg"], float)
    rng = np.random.default_rng(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    os.symlink((src / "videos").resolve(), dst / "videos")
    shutil.copytree(src / "meta", dst / "meta")
    head_in_base = np.array(json.loads((src / "meta" / "stereo_calibration.json").read_text())["head_in_base"])

    positions, yaws, stacks = [], [], []  # per global episode index
    for path in args.raw:
        with h5py.File(path, "r") as h5:
            for n in sorted(h5.keys()):
                pos = true_positions(h5[n], head_in_base)
                if args.skill != "all":  # per-skill datasets keep only that skill's frames (same rule as the converter)
                    keep = h5[n]["skill"][()] == list(language.SKILLS).index(args.skill)
                    if keep.sum() < 5:
                        continue
                    if len(pos) > 1:
                        pos = pos[keep]
                positions.append(pos)
                yaws.append(true_yaws(h5[n]))
                stacks.append(is_stack(h5[n]))

    tables = [(f, pq.read_table(f)) for f in sorted((src / "data").rglob("*.parquet"))]
    all_states = []
    for f, table in tables:
        ep = table.column("episode_index").to_numpy()
        skill = np.stack(table.column("skill_index").to_numpy(zero_copy_only=False)).reshape(-1).astype(int)
        state = np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        target = np.zeros((len(ep), 9 if args.rot6d else 3), dtype=np.float32)
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            rows = rows[np.argsort(table.column("frame_index").to_numpy()[rows])]
            pos = positions[int(e)]
            if len(pos) > 1:
                pos = pos[: len(rows)]
            target[rows, :3] = published_targets(pos, state[rows, :3], skill[rows], rng, args)
            if args.rot6d:  # the active skill's object orientation (constant within the episode) + detection error
                if args.yaw_noise == "empirical":
                    err = rng.choice(yaw_samples) * rng.choice([-1.0, 1.0])
                else:
                    err = rng.normal(0, args.yaw_sigma_deg)
                st = stacks[int(e)]
                err2 = (rng.normal(0, args.yaw_sigma_deg) if args.yaw_noise == "gauss" else
                        rng.choice(yaw_samples) * rng.choice([-1.0, 1.0])) if st else 0.0
                noisy = yaws[int(e)] + np.deg2rad(np.array([err, err2]))
                # Reported modulo the cube's 90 deg symmetry, as perception (locate9) does.
                second = rot6d_head(language.fold_yaw(noisy[1], "cube"), head_in_base) if st else rot6d_head(0.0, head_in_base)
                r6 = np.stack([rot6d_head(language.fold_yaw(noisy[0], "cube"), head_in_base), second])
                target[rows, 3:] = r6[[TARGET_SLOT.get(int(k), 0) for k in skill[rows]]]
        new_state = np.concatenate([state, target], axis=1)
        all_states.append(new_state)
        col = pa.FixedSizeListArray.from_arrays(pa.array(new_state.reshape(-1), pa.float32()), new_state.shape[1])
        table = table.set_column(table.schema.get_field_index("observation.state"), "observation.state", col)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out)

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feat = info["features"]["observation.state"]
    rot_names = [f"target.{c}" for c in ("r11", "r21", "r31", "r12", "r22", "r32")] if args.rot6d else []
    feat["names"] = list(feat["names"])[:10] + TARGET_NAMES + rot_names
    feat["shape"] = [10 + target.shape[1]]
    info_path.write_text(json.dumps(info, indent=4))
    states = np.concatenate(all_states)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    s = stats["observation.state"]
    for key in list(s):
        if key == "mean":
            s[key] = states.mean(0).tolist()
        elif key == "std":
            s[key] = states.std(0).tolist()
        elif key == "min":
            s[key] = states.min(0).tolist()
        elif key == "max":
            s[key] = states.max(0).tolist()
        elif key.startswith("q") and key[1:].isdigit():
            s[key] = np.quantile(states, int(key[1:]) / 100, axis=0).tolist()
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[targets] {len(positions)} episodes, {len(states)} frames, {states.shape[1]}-D state -> {dst}")


if __name__ == "__main__":
    main()
