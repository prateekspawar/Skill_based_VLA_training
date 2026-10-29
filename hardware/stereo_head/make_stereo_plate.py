"""Build stereo_plate.stl, the same part as stereo_plate.scad (keep the parameters in sync), without OpenSCAD.

    uv run --with manifold3d --with numpy python hardware/stereo_head/make_stereo_plate.py
"""

import struct
from pathlib import Path

import numpy as np
from manifold3d import Manifold

BASELINE, PLATE_L, PLATE_H, PLATE_T = 60.0, 90.0, 35.0, 10.0
HOLE_DX, HOLE_DY = 21.0, 12.5  # boards upright, ribbons down
LENS_OFF_X, LENS_OFF_Y = 0.0, 7.9  # lens vs hole-pattern centre (Y: below); photo estimate, measure (see .scad)
BOARD_H, LENS_TO_BOTTOM = 23.86, 7.8
PLATE_CY = BOARD_H / 2 - LENS_TO_BOTTOM  # plate centred on the boards vertically
PIN_D, PIN_H, STANDOFF_D, STANDOFF_H = 1.9, 2.5, 4.5, 3.0
M4_D, M4_CBORE_D, M4_CBORE_H, M4_DY = 4.4, 8.0, 3.0, 8.0
SEG = 48


def cyl(d, h, x, y, z, d2=None):
    return Manifold.cylinder(h, d / 2, (d2 if d2 is not None else d) / 2, SEG).translate((x, y, z))


def box(sx, sy, sz, x, y, z):
    return Manifold.cube((sx, sy, sz)).translate((x, y, z))


plate = box(PLATE_L, PLATE_H, PLATE_T, -PLATE_L / 2, PLATE_CY - PLATE_H / 2, 0)
for sy in (-1, 1):
    plate -= cyl(M4_D, PLATE_T + 2, 0, PLATE_CY + sy * M4_DY, -1)
    plate -= cyl(M4_CBORE_D, M4_CBORE_H + 1, 0, PLATE_CY + sy * M4_DY, PLATE_T - M4_CBORE_H)
for side in (-1, 1):  # lens-centre marks
    x = side * BASELINE / 2
    plate -= box(6, 0.6, 1, x - 3, -0.3, PLATE_T - 0.6)
    plate -= box(0.6, 6, 1, x - 0.3, -3, PLATE_T - 0.6)
for side in (-1, 1):
    cx, cy = side * BASELINE / 2 - LENS_OFF_X, LENS_OFF_Y
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * HOLE_DX / 2, cy + sy * HOLE_DY / 2
            plate += cyl(STANDOFF_D, STANDOFF_H, x, y, PLATE_T)
            plate += cyl(PIN_D, PIN_H - 0.4, x, y, PLATE_T + STANDOFF_H)
            plate += cyl(PIN_D, 0.4, x, y, PLATE_T + STANDOFF_H + PIN_H - 0.4, d2=PIN_D - 0.6)

mesh = plate.to_mesh()
v, f = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
tris = v[f].astype(np.float32)
normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
out = Path(__file__).parents[1] / "prints" / "stereo_plate.stl"
with open(out, "wb") as fh:
    fh.write(b"stereo_plate".ljust(80, b" ") + struct.pack("<I", len(tris)))
    for n, t in zip(normals.astype(np.float32), tris):
        fh.write(n.tobytes() + t.tobytes() + b"\0\0")
lo, hi = v.min(0), v.max(0)
print(f"{out}: {len(tris)} triangles, {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm, "
      f"volume {plate.volume()/1000:.1f} cm3, genus {plate.genus()}")
