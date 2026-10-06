"""Append the skill verdict [done, success] to the action of an existing all-success dataset (~/lerobot-env).

Rewrites the parquet files only (videos are symlinked, nothing re-encoded): action 10-D -> 12-D with
:func:`so101_lang.data.status.default_status` per episode (last 5 frames of every skill segment = [1, 1]).
info.json feature shapes / names and stats.json action statistics are updated.

    ~/lerobot-env/bin/python -m so101_lang.data.add_status --src datasets/lerobot/s512_mix_depth_target9 \
        --dst datasets/lerobot/s512_mix_depth_target9s
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
from so101_lang.data.status import STATUS_NAMES, default_status


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
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

    actions = []
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        if act.shape[1] != 10:
            raise SystemExit(f"{f}: action is {act.shape[1]}-D, expected 10")
        ep = t.column("episode_index").to_numpy()
        fr = t.column("frame_index").to_numpy()
        skill = np.stack(t.column("skill_index").to_numpy(zero_copy_only=False)).reshape(-1)
        status = np.zeros((len(ep), 2), dtype=np.float32)
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            rows = rows[np.argsort(fr[rows])]
            status[rows] = default_status(skill[rows])
        new = np.concatenate([act, status], 1)
        offsets = pa.array(np.arange(0, 12 * len(new) + 1, 12, dtype=np.int32))
        i = t.schema.get_field_index("action")
        col = pa.ListArray.from_arrays(offsets, pa.array(new.reshape(-1), pa.float32())).cast(t.schema.field(i).type)
        t = t.set_column(i, t.schema.field(i), col)
        actions.append(new)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["features"]["action"]["shape"] = [12]
    info["features"]["action"]["names"] = list(info["features"]["action"]["names"])[:10] + STATUS_NAMES
    info_path.write_text(json.dumps(info, indent=4))
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["action"] = vec_stats(np.concatenate(actions), stats["action"])
    stats_path.write_text(json.dumps(stats, indent=4))
    a = np.concatenate(actions)
    print(f"[status] {len(a)} frames, done={a[:, 10].mean():.3f}, success={a[:, 11].mean():.3f} -> {dst}")


if __name__ == "__main__":
    main()
