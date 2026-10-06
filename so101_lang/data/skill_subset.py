"""Per-skill dataset from a mixed one by filtering frames on ``skill_index`` (no video re-encoding) (~/lerobot-env).

Each episode keeps only its frames of the chosen skill (a contiguous segment: pick before the lift, place after).
Rows keep their timestamps, so they still address the right video frames; global ``index`` and the episode
metadata (length, dataset_from_index / dataset_to_index) are recomputed, and state / action statistics are
recomputed on the kept frames. Videos are symlinked.

    ~/lerobot-env/bin/python -m so101_lang.data.skill_subset --src datasets/lerobot/s512_mix_depth_target9 \
        --dst datasets/lerobot/s512_pick9 --skill pick
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

from so101_lang import language


def vec_stats(x: np.ndarray, like: dict) -> dict:
    out = {}
    for key in like:
        if key == "mean":
            out[key] = x.mean(0).tolist()
        elif key == "std":
            out[key] = x.std(0).tolist()
        elif key == "min":
            out[key] = x.min(0).tolist()
        elif key == "max":
            out[key] = x.max(0).tolist()
        elif key == "count":
            out[key] = [len(x)]
        elif key.startswith("q") and key[1:].isdigit():
            out[key] = np.quantile(x, int(key[1:]) / 100, axis=0).tolist()
        else:
            out[key] = like[key]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--skill", choices=list(language.SKILLS), required=True)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    k = list(language.SKILLS).index(args.skill)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for item in src.iterdir():
        if item.name in ("videos", "images"):
            os.symlink(item.resolve(), dst / item.name)
        elif item.name == "meta":
            shutil.copytree(item, dst / "meta")

    # Filter frames, keeping file layout; global index is reassigned in episode order.
    kept = {}  # file -> filtered table
    per_episode = {}  # episode -> kept frame count
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        skill = np.stack(t.column("skill_index").to_numpy(zero_copy_only=False)).reshape(-1)
        t = t.filter(pa.array(skill == k))
        kept[f] = t
        for e, n in zip(*np.unique(t.column("episode_index").to_numpy(), return_counts=True)):
            per_episode[int(e)] = per_episode.get(int(e), 0) + int(n)
    # Episodes with no frames of this skill (e.g. pick failures for place) are dropped; the rest are renumbered.
    renum = {e: n for n, e in enumerate(sorted(e for e, c in per_episode.items() if c > 0))}
    starts, pos = {}, 0
    for e in sorted(renum):
        starts[e] = pos
        pos += per_episode[e]
    states, actions = [], []
    for f, t in kept.items():
        ep, fr = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        # Row order within an episode follows frame_index; new index = episode start + rank of the frame.
        new_index = np.zeros(len(ep), dtype=np.int64)
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            order = rows[np.argsort(fr[rows])]
            new_index[order] = starts[int(e)] + np.arange(len(order))
        t = t.set_column(t.schema.get_field_index("index"), "index", pa.array(new_index, t.schema.field("index").type))
        t = t.set_column(t.schema.get_field_index("episode_index"), "episode_index",
                         pa.array([renum[int(e)] for e in ep], t.schema.field("episode_index").type))
        if not len(t):
            continue  # empty parquet files break the HF datasets loader
        if len(t):  # a file can hold no frames of this skill (e.g. only pick-failure episodes for place)
            states.append(np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)))
            actions.append(np.stack(t.column("action").to_numpy(zero_copy_only=False)))
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)

    for mf in sorted((dst / "meta" / "episodes").rglob("*.parquet")):
        mt = pq.read_table(mf)
        mt = mt.filter(pa.array([int(e) in renum for e in mt.column("episode_index").to_numpy()]))
        if not len(mt):
            mf.unlink()
            continue
        eps = mt.column("episode_index").to_numpy()
        cols = {
            "episode_index": [renum[int(e)] for e in eps],
            "length": [per_episode.get(int(e), 0) for e in eps],
            "dataset_from_index": [starts.get(int(e), 0) for e in eps],
            "dataset_to_index": [starts.get(int(e), 0) + per_episode.get(int(e), 0) for e in eps],
        }
        for name, vals in cols.items():
            mt = mt.set_column(mt.schema.get_field_index(name), name, pa.array(vals, mt.schema.field(name).type))
        pq.write_table(mt, mf)

    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["total_frames"] = int(pos)
    n_src = info["total_episodes"]
    info["total_episodes"] = len(renum)
    if "splits" in info:
        info["splits"] = {"train": f"0:{len(renum)}"}
    info_path.write_text(json.dumps(info, indent=4))
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["observation.state"] = vec_stats(np.concatenate(states), stats["observation.state"])
    stats["action"] = vec_stats(np.concatenate(actions), stats["action"])
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[subset] {args.skill}: {pos} frames from {len(renum)} episodes "
          f"({n_src - len(renum)} without this skill dropped) -> {dst}")


if __name__ == "__main__":
    main()
