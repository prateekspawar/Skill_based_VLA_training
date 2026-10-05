"""Work around LeRobot's merge failing on the multi-dimensional ``observation.pointmap`` column (~/lerobot-env).

    strip   SRC DST              copy of SRC without the point-map column (videos symlinked)
    attach  MERGED DST SRC...    MERGED + the point map re-attached from the SRC datasets, matched by
                                 (episode, frame); SRC datasets are given in merge order, so merged episode
                                 e maps to source k = the one whose episode range contains e.

    python -m so101_lang.data.pointmap_merge strip datasets/lerobot/a datasets/lerobot/a_nopm
    lerobot-edit-dataset --operation.type=merge ... (a_nopm, b_nopm) -> merged_nopm
    python -m so101_lang.data.pointmap_merge attach merged_nopm merged datasets/lerobot/a datasets/lerobot/b
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

KEY = "observation.pointmap"


def _files(root: Path):
    return sorted((root / "data").rglob("*.parquet"))


def _link_or_copy_tree(src: Path, dst: Path):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.name in ("videos", "images"):
            os.symlink(item.resolve(), target)
        elif item.name != "data":
            shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)


def strip(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    _link_or_copy_tree(src, dst)
    for f in _files(src):
        t = pq.read_table(f)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t.drop([KEY]), out)
    info = json.loads((dst / "meta" / "info.json").read_text())
    info["features"].pop(KEY, None)
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    stats.pop(KEY, None)
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    print(f"[pointmap] stripped {src} -> {dst}")


def _load_pointmaps(root: Path):
    """All point maps of a dataset as one (N, 3, 64, 64) array plus (episode, frame) -> row lookup."""
    eps, frames, maps = [], [], []
    for f in _files(root):
        t = pq.read_table(f, columns=["episode_index", "frame_index", KEY])
        eps.append(t.column("episode_index").to_numpy())
        frames.append(t.column("frame_index").to_numpy())
        col = t.column(KEY).combine_chunks()
        flat = col.flatten().flatten().flatten().to_numpy(zero_copy_only=False)
        maps.append(flat.astype(np.float32).reshape(len(col), 3, 64, 64))
    eps, frames, maps = np.concatenate(eps), np.concatenate(frames), np.concatenate(maps)
    lookup = {(int(e), int(fr)): i for i, (e, fr) in enumerate(zip(eps, frames))}
    return maps, lookup, int(eps.max()) + 1


def _nested_list(arr: np.ndarray, like: pa.DataType) -> pa.Array:
    """(N, A, B, C) float32 -> list<list<list<float>>> matching the original column type."""
    n, a, b, c = arr.shape
    values = pa.array(arr.reshape(-1), type=like.value_type.value_type.value_type)
    inner = pa.ListArray.from_arrays(pa.array(np.arange(0, n * a * b * c + 1, c, dtype=np.int32)), values)
    mid = pa.ListArray.from_arrays(pa.array(np.arange(0, n * a * b + 1, b, dtype=np.int32)), inner)
    return pa.ListArray.from_arrays(pa.array(np.arange(0, n * a + 1, a, dtype=np.int32)), mid)


def attach(merged: Path, dst: Path, sources: list[Path]):
    if dst.exists():
        shutil.rmtree(dst)
    _link_or_copy_tree(merged, dst)
    srcs = [_load_pointmaps(s) for s in sources]
    offsets = np.cumsum([0] + [n for _, _, n in srcs])
    col_type = pq.read_schema(_files(sources[0])[0]).field(KEY).type
    missing = 0
    for f in _files(merged):
        t = pq.read_table(f)
        eps, frames = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        out = np.zeros((len(eps), 3, 64, 64), dtype=np.float32)
        for i, (e, fr) in enumerate(zip(eps, frames)):
            k = int(np.searchsorted(offsets, e, side="right") - 1)
            maps, lookup, _ = srcs[k]
            row = lookup.get((int(e - offsets[k]), int(fr)))
            if row is None:
                missing += 1
                continue
            out[i] = maps[row]
        t = t.append_column(KEY, _nested_list(out, col_type))
        o = dst / f.relative_to(merged)
        o.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, o)
    src_info = json.loads((sources[0] / "meta" / "info.json").read_text())
    info = json.loads((dst / "meta" / "info.json").read_text())
    info["features"][KEY] = src_info["features"][KEY]
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    src_stats = json.loads((sources[0] / "meta" / "stats.json").read_text())
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    stats[KEY] = src_stats[KEY]
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
    if missing:
        raise SystemExit(f"[pointmap] {missing} rows had no source point map")
    print(f"[pointmap] attached point maps from {len(sources)} sources -> {dst}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "strip":
        strip(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "attach":
        attach(Path(sys.argv[2]), Path(sys.argv[3]), [Path(p) for p in sys.argv[4:]])
    else:
        raise SystemExit(__doc__)
