"""Monolithic (all-skill) dataset with the per-skill label fixes applied per skill segment (~/lerobot-env).

The per-skill datasets get two fixes after splitting: failure motion masked (mask_failure_motion) and the place
success label moved onto every frame after a completed release (relabel_release). Both work per episode, which is
wrong for a mixed dataset: a place failure would also mask the successful pick before it. Here each episode's skill
segments (``skill_index``) are treated separately, giving the same labels as the per-skill datasets, so a single
model trained on all frames sees exactly the data the separate skill models saw (experiment 1: monolithic vs
per-skill models).

    ~/lerobot-env/bin/python -m so101_lang.data.mono_labels --src datasets/lerobot/s512_rc2_all9 \
        --dst datasets/lerobot/s512_mono9rc2
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

MOTION, OPENING = 10, 9  # action status dims start at 10; state dim 9 = gripper opening
PLACE = list(language.SKILLS).index("place")


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
    actions, masked, relabelled = [], 0, 0
    for f in sorted((src / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        act = np.stack(t.column("action").to_numpy(zero_copy_only=False)).astype(np.float32)
        state = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        skill = np.asarray(t.column("skill_index").to_numpy(zero_copy_only=False)).reshape(-1)
        ep, fr = t.column("episode_index").to_numpy(), t.column("frame_index").to_numpy()
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            rows = rows[np.argsort(fr[rows])]
            for s in np.unique(skill[rows]):
                seg = rows[skill[rows] == s]
                status = act[seg, MOTION:]
                failed = np.nonzero((status[:, 0] > 0.5) & (status[:, 1] < 0.5))[0]
                if len(failed):  # failure segment: learn only the verdict before the failure is detected
                    act[seg[: failed[0]], :MOTION] = np.nan
                    masked += int(failed[0])
                elif s == PLACE and status[-1, 1] > 0.5:  # successful place: success from the completed release
                    opened = np.nonzero(state[seg, OPENING] >= args.open)[0]
                    if len(opened):
                        act[seg[opened[0]:], MOTION:] = 1.0
                        relabelled += 1
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
    total = sum(len(a) for a in actions)
    print(f"[mono] {total} frames: motion masked on {masked}, {relabelled} place segments relabelled -> {dst}")


if __name__ == "__main__":
    main()
