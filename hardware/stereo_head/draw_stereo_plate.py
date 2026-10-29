"""Dimensioned front view + 3-D render of stereo_plate.stl -> stereo_plate_drawing.png (~/lerobot-env python)."""

import struct
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = Path(__file__).parent
B, L, H = 60, 90, 35
DX, DY, LOX, LOY = 21.0, 12.5, 0.0, 7.9  # hole spacing; lens offset from the hole-pattern centre (LOY: below)
BW, BH, LENS_TO_BOTTOM = 25.0, 23.86, 7.8  # board as mounted upright, ribbon down
PCY = BH / 2 - LENS_TO_BOTTOM  # plate centre height above the lens line

fig = plt.figure(figsize=(14, 6.5))
ax = fig.add_subplot(1, 2, 1)
ax.set_aspect("equal")
ax.add_patch(Rectangle((-L / 2, PCY - H / 2), L, H, fc="#e8c547", ec="k"))
for s in (-1, 1):  # s = side of the drawing (viewer facing the lenses)
    lx = s * B / 2
    cx, cy = lx - LOX, LOY
    ax.add_patch(Rectangle((lx - BW / 2, -LENS_TO_BOTTOM), BW, BH, fc="#2e8b57", ec="k", alpha=0.45, ls="--"))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Circle((cx + sx * DX / 2, cy + sy * DY / 2), 2.25, fc="#c9a227", ec="k"))
            ax.add_patch(Circle((cx + sx * DX / 2, cy + sy * DY / 2), 0.95, fc="k"))
    ax.add_patch(Circle((lx, 0), 3.5, fc="#222", ec="w"))
    ax.plot(lx, 0, "w+", ms=8)
    ribbon_end = PCY - H / 2 - 10  # 16 mm ribbon from the board's bottom edge down past the plate
    ax.add_patch(Rectangle((lx - 8, ribbon_end), 16, -LENS_TO_BOTTOM - ribbon_end, fc="#dfe7f2", ec="steelblue",
                           alpha=0.8))
    ax.text(lx, PCY - H / 2 - 6, "ribbon", ha="center", color="steelblue", fontsize=8)
    # Facing the lenses, the camera's own left is on the viewer's right.
    ax.text(lx + 9, 4, "R" if s < 0 else "L", ha="center", fontsize=11, weight="bold")
for sy in (-1, 1):
    ax.add_patch(Circle((0, PCY + sy * 8), 4, fc="w", ec="k"))
    ax.add_patch(Circle((0, PCY + sy * 8), 2.2, fc="#888", ec="k"))


def dim(x0, x1, y, t):
    ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text((x0 + x1) / 2, y + 0.8, t, ha="center", fontsize=8)


top = PCY + H / 2
dim(-30, 30, top + 3, "60 lens centre to centre")
dim(-L / 2, L / 2, top + 8, "90")
dim(30 - DX / 2, 30 + DX / 2, top - 4.5, "21")
ax.annotate("", xy=(L / 2 + 3, PCY - H / 2), xytext=(L / 2 + 3, PCY + H / 2), arrowprops=dict(arrowstyle="<->", lw=0.8))
ax.text(L / 2 + 4, PCY, "35", fontsize=8)
ax.annotate("", xy=(-L / 2 - 3, LOY - DY / 2), xytext=(-L / 2 - 3, LOY + DY / 2), arrowprops=dict(arrowstyle="<->", lw=0.8))
ax.plot([-L / 2 - 4, -30 - DX / 2], [LOY - DY / 2] * 2, "k:", lw=0.6)
ax.plot([-L / 2 - 4, -30 - DX / 2], [LOY + DY / 2] * 2, "k:", lw=0.6)
ax.text(-L / 2 - 9, LOY, "12.5", fontsize=8)
ax.text(0, PCY - H / 2 - 17, "front view facing the lenses (L = camera's own left, as in training); both ribbons exit "
        "downward;\npins 1.9 mm on 4.5 x 3 mm standoffs; 2x M4 counterbored; plate 10 mm thick; all mm",
        ha="center", fontsize=8)
ax.set_xlim(-L / 2 - 12, L / 2 + 10)
ax.set_ylim(PCY - H / 2 - 20, top + 12)
ax.axis("off")
ax.set_title("Stereo plate with boards (dashed) and lenses")

d = (HERE.parent / "prints" / "stereo_plate.stl").read_bytes()
n = struct.unpack("<I", d[80:84])[0]
t = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"]
t = t.reshape(-1, 3, 3).astype(float)
m = t.reshape(-1, 3)
ext = m.max(0) - m.min(0)
nr = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
nr /= np.linalg.norm(nr, axis=1, keepdims=True) + 1e-12
sh = 0.35 + 0.65 * np.clip(nr @ np.array([0.3, -0.5, 0.8]), 0, 1)
a3 = fig.add_subplot(1, 2, 2, projection="3d")
a3.add_collection3d(Poly3DCollection(t, facecolors=np.c_[0.91 * sh, 0.77 * sh, 0.28 * sh, np.ones_like(sh)],
                                     edgecolor="none"))
a3.auto_scale_xyz(*[(m[:, k].min(), m[:, k].max()) for k in range(3)])
a3.set_box_aspect(tuple(ext))
a3.axis("off")
a3.view_init(40, -65)
a3.set_title("stereo_plate.stl (print orientation, pins up)")
plt.tight_layout()
plt.savefig(HERE / "stereo_plate_drawing.png", dpi=110)
