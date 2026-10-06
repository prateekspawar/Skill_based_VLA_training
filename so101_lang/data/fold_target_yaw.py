"""Fold the target yaw in observation.state by the object's symmetry, in place of a full rebuild (~/lerobot-env).

The 9-D target (state[13:19] = rot6d, head frame) of pick frames is a z-rotation in the base frame; its yaw is
folded with :func:`so101_lang.language.fold_yaw` ("cube", 90 deg), matching what perception (locate9) reports.
Place targets (plates, yaw 0) are unchanged. Only parquet files and state statistics are rewritten.

    ~/lerobot-env/bin/python -m so101_lang.data.fold_target_yaw --src datasets/lerobot/s512_pick9s \
        --dst datasets/lerobot/s512_pick9f
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
from so101_lang.data.skill_subset import vec_stats

R6 = slice(13, 19)


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
    hb = np.array(json.loads((src / "meta" / "stereo_calibration.json").read_text())["head_in_base"])[:3, :3]
    pick = list(language.SKILLS).index("pick")
    states, changed, max_err = [], 0, 0.0
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        st = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        skill = np.stack(t.column("skill_index").to_numpy(zero_copy_only=False)).reshape(-1)
        rows = np.nonzero(skill == pick)[0]
        if len(rows):
            r6 = st[rows, R6].astype(np.float64)
            c1_base = r6[:, 0:3] @ hb.T  # first rotation column in the base frame
            max_err = max(max_err, float(np.abs(c1_base[:, 2]).max()))  # must be a pure z-rotation
            yaw = np.arctan2(c1_base[:, 1], c1_base[:, 0])
            folded = np.array([language.fold_yaw(y, "cube") for y in yaw])
            c, s = np.cos(folded), np.sin(folded)
            col1 = np.stack([c, s, np.zeros_like(c)], 1) @ hb  # base -> head (hb.T maps head->base)
            col2 = np.stack([-s, c, np.zeros_like(c)], 1) @ hb
            changed += int((np.abs(folded - yaw) > 1e-6).sum())
            st[rows, R6] = np.concatenate([col1, col2], 1).astype(np.float32)
        states.append(st)
        i = t.schema.get_field_index("observation.state")
        offsets = pa.array(np.arange(0, st.shape[1] * len(st) + 1, st.shape[1], dtype=np.int32))
        col = pa.ListArray.from_arrays(offsets, pa.array(st.reshape(-1), pa.float32())).cast(t.schema.field(i).type)
        t = t.set_column(i, t.schema.field(i), col)
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, out)
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())
    stats["observation.state"] = vec_stats(np.concatenate(states), stats["observation.state"])
    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"[fold] {changed} pick frames re-folded (max |z| of target x-axis {max_err:.1e}) -> {dst}")


if __name__ == "__main__":
    main()
