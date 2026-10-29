"""Print files (written to hardware/prints/) for the task objects, at the simulation's dimensions (so101_lang/envs/pick_place_env_cfg.py):

    uv run --with manifold3d --with numpy python hardware/objects/make_objects.py

cube.stl            30 mm cube (CUBE_SIZE = 0.03)
plate_decagon.stl   90 mm across corners, 8 mm thick: the simulated cylinder (PLATE_RADIUS = 0.045,
                    PLATE_HEIGHT = 0.008) as Isaac Sim renders it (10 flat sides, corners on the 45 mm radius)
"""

import struct
from pathlib import Path

import numpy as np
from manifold3d import Manifold

CUBE_SIZE = 30.0
PLATE_RADIUS, PLATE_HEIGHT = 45.0, 8.0
OUT = Path(__file__).parents[1] / "prints"


def write_stl(m: Manifold, name: str):
    mesh = m.to_mesh()
    v, f = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
    tris = v[f].astype(np.float32)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    with open(OUT / name, "wb") as fh:
        fh.write(name.encode().ljust(80, b" ") + struct.pack("<I", len(tris)))
        for nn, t in zip(n.astype(np.float32), tris):
            fh.write(nn.tobytes() + t.tobytes() + b"\0\0")
    lo, hi = v.min(0), v.max(0)
    print(f"{name}: {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm, volume {m.volume()/1000:.1f} cm3")


write_stl(Manifold.cube((CUBE_SIZE,) * 3, center=True).translate((0, 0, CUBE_SIZE / 2)), "cube.stl")
write_stl(Manifold.cylinder(PLATE_HEIGHT, PLATE_RADIUS, PLATE_RADIUS, 10), "plate_decagon.stl")
