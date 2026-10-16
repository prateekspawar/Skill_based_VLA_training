"""Turn null list elements back into NaN after a LeRobot merge (~/lerobot-env).

Masked action dims (mask_failure_motion: NaN = not trained) come out of ``lerobot-edit-dataset --operation.type=merge``
as nulls, which the data loader cannot turn into tensors ("Could not infer dtype of NoneType"). This rewrites the
float list columns in place with NaN for null elements.

    ~/lerobot-env/bin/python -m so101_lang.data.restore_nan datasets/lerobot/<merged dataset>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def main():
    root = Path(sys.argv[1])
    fixed = 0
    for f in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(f)
        changed = False
        for name in ("action", "observation.state"):
            if name not in t.column_names:
                continue
            col = t.column(name)
            if pc.list_flatten(col).null_count == 0:
                continue
            dim = len(col[0].as_py())
            vals = np.array([[np.nan if v is None else v for v in row] for row in col.to_pylist()], dtype=np.float32)
            fixed += int(np.isnan(vals).sum())
            offsets = pa.array(np.arange(0, dim * len(vals) + 1, dim, dtype=np.int32))
            arr = pa.ListArray.from_arrays(offsets, pa.array(vals.reshape(-1), pa.float32()))
            i = t.schema.get_field_index(name)
            t = t.set_column(i, t.schema.field(i), arr.cast(t.schema.field(i).type))
            changed = True
        if changed:
            pq.write_table(t, f)
    print(f"[restore_nan] {fixed} null elements -> NaN in {root}")


if __name__ == "__main__":
    main()
