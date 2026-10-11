"""Success label for every frame after a completed release (~/lerobot-env).

Post-release demos (gen_demos --post_release_frames) record the release, the retreat and a short hold. The expert's
retreat phase usually runs into its 4 s timeout, so a place segment ends with ~90 nearly identical hover frames of
which only the last 5 carried [done, success] = [1, 1]: a memoryless policy cannot tell them apart. Here every frame
of a successful place segment from the first fully open gripper (state opening >= --open) is labelled [1, 1], so
the model learns "released -> report success". Failure episodes (any [1, 0] frame) are left unchanged.

    ~/lerobot-env/bin/python -m so101_lang.data.relabel_release --src datasets/lerobot/s512_place9rc2 \
        --dst datasets/lerobot/s512_place9rc2r
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from so101_lang.data.skill_subset import vec_stats

MOTION, OPENING = 10, 9  # action status dims start at 10; state dim 9 = gripper opening


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--open", type=float, default=0.9, help="Gripper opening that counts as released.")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name in ("videos", "images"):
            os.symlink(item.resolve(), dst / item.name)
        elif item.name == "meta":
            shutil.copytree(item, dst / "meta")
    actions, n_eps, n_ok, before, after = [], 0, 0, 0, 0
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        state = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        ep, fr = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            rows = rows[np.argsort(fr[rows])]
            n_eps += 1
            status = act[rows, MOTION:]
            if ((status[:, 0] > 0.5) & (status[:, 1] < 0.5)).any() or status[-1, 1] < 0.5:
                continue  # failure episode
            opened = np.nonzero(state[rows, OPENING] >= args.open)[0]
            if not len(opened):
                continue
            before += int((status[:, 1] > 0.5).sum())
            act[rows[opened[0]:], MOTION:] = 1.0
            after += len(rows) - int(opened[0])
            n_ok += 1
        actions.append(act)
        i = t.schema.get_field_index("action")
        offsets = pa.array(np.arange(0, act.shape[1] * len(act) + 1, act.shape[1], dtype=np.int32))
        col = pa.ListArray.from_arrays(offsets, pa.array(act.reshape(-1), pa.float32()))  # NaN stays a float NaN
        t = t.set_column(i, t.schema.field(i), col.cast(t.schema.field(i).type))
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = vec_stats(np.concatenate(actions), stats["action"])
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[relabel] {n_ok}/{n_eps} episodes relabelled: success frames {before} -> {after} -> {dst}")


if __name__ == "__main__":
    main()
