"""Hand-eye and stereo calibration targets for the real SO-101 setup.

    uv run --with manifold3d --with numpy --with opencv-python-headless --with matplotlib \
        python hardware/calibration/make_calibration.py

Writes
  hardware/prints/charuco_handeye_5x5_18mm.pdf     A4, 90 x 90 mm ChArUco board for the hand-eye calibration
  hardware/prints/charuco_stereo_8x11_22mm.pdf     A4, 176 x 242 mm ChArUco board for the cameras' own calibration
  hardware/prints/handeye_jaw_clamp.stl                 sleeve clamp for the SO-101 fixed jaw, with a flange
  hardware/prints/handeye_board_plate.stl               98 x 98 x 3 mm backing plate for the hand-eye board
and checks that OpenCV detects every corner of both boards in a rendered image.

The clamp is built from the fixed jaw of hardware/prints/Wrist_Roll_Follower_SO101.stl (STL frame: z along the jaw
towards its tip, +x towards the moving jaw). It slides on from the tip and seats where the jaw widens; an M3 pinch
screw across a slot on the inner face locks it. Its cavity at each height is the union of every jaw cross-section
above it (everything it passes on the way down), grown by CLEARANCE, so it can actually be slid on.
"""

from pathlib import Path

import cv2
import manifold3d
import matplotlib
import numpy as np
from manifold3d import CrossSection, JoinType, Manifold, Mesh

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).parent
PRINTS = HERE.parent / "prints"
DICT = cv2.aruco.DICT_4X4_100

# Boards: (squares x, squares y, square mm, marker mm, first marker id). Marker = 3/4 of the square, so every
# marker bit is an exact 1/8 of a square and the board can be drawn as vector cells.
HANDEYE = dict(name="charuco_handeye_5x5_18mm", nx=5, ny=5, square=18.0, marker=13.5, id0=50)
STEREO = dict(name="charuco_stereo_8x11_22mm", nx=8, ny=11, square=22.0, marker=16.5, id0=0)

# Clamp (mm, STL frame of the gripper body)
BAND = (55.0, 72.0)  # sleeve height range on the jaw
CLEARANCE = 0.25
WALL = 3.0
LAYER = 0.5
SLOT = 1.6  # pinch slot width
EAR = 8.0  # ear length beyond the sleeve's inner wall
PLATE = 98.0  # backing plate size (board 90 + 4 mm margin each side)
PLATE_T = 3.0
PLATE_BACK_X = -40.5  # flange face = back of the plate; the gripper body ends at x = -35.2
SCREW_Y = 9.0  # the two M3 plate screws, +-y, at the band's mid height


# ----------------------------------------------------------------------------------------------- boards
def board(spec):
    ids = np.arange(spec["id0"], spec["id0"] + (spec["nx"] * spec["ny"]) // 2)
    return cv2.aruco.CharucoBoard((spec["nx"], spec["ny"]), spec["square"], spec["marker"],
                                  cv2.aruco.getPredefinedDictionary(DICT), ids)


def board_cells(spec, px=20):
    """Binary image of the board with one pixel per marker bit (1/8 square); True = black."""
    nx, ny = spec["nx"] * 8, spec["ny"] * 8
    img = board(spec).generateImage((nx * px, ny * px), marginSize=0, borderBits=1)
    return img[px // 2::px, px // 2::px] < 128


def check_detection(spec):
    cells = board_cells(spec)
    img = np.kron(~cells, np.ones((12, 12), np.uint8)) * 255
    img = cv2.copyMakeBorder(img.astype(np.uint8), 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board(spec)).detectBoard(img)
    n_exp = (spec["nx"] - 1) * (spec["ny"] - 1)
    n = 0 if ids is None else len(ids)
    print(f"{spec['name']}: OpenCV detected {n}/{n_exp} ChArUco corners")
    assert n == n_exp


def board_pdf(spec, plate_outline=None):
    a4w, a4h = 210.0, 297.0
    cells = board_cells(spec)
    cell = spec["square"] / 8
    bw, bh = cells.shape[1] * cell, cells.shape[0] * cell
    x0, y0 = (a4w - bw) / 2, a4h - 15 - bh  # top-left placement, y up in the figure
    fig = plt.figure(figsize=(a4w / 25.4, a4h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, a4w)
    ax.set_ylim(0, a4h)
    ax.axis("off")
    for r in range(cells.shape[0]):  # black runs per row, as vector rectangles
        row, c = cells[r], 0
        while c < len(row):
            if row[c]:
                c1 = c
                while c1 < len(row) and row[c1]:
                    c1 += 1
                ax.add_patch(Rectangle((x0 + c * cell, y0 + bh - (r + 1) * cell), (c1 - c) * cell, cell,
                                       fc="black", ec="none", lw=0, antialiased=False))
                c = c1
            else:
                c += 1
    if plate_outline:  # cutting guide = backing plate size
        m = (plate_outline - bw) / 2
        ax.add_patch(Rectangle((x0 - m, y0 - m), plate_outline, plate_outline, fill=False, ec="0.6", lw=0.4, ls="--"))
    # scale check
    ys = y0 - (10 if plate_outline is None else 14)
    ax.plot([x0, x0 + 50], [ys, ys], "k-", lw=0.8)
    for x in (x0, x0 + 50):
        ax.plot([x, x], [ys - 2, ys + 2], "k-", lw=0.8)
    ax.text(x0 + 52, ys - 1, "50 mm: check with calipers", fontsize=7, va="center")
    ids = board(spec).getIds()
    ax.text(x0, ys - 5,
            f"ChArUco {spec['nx']} x {spec['ny']} squares, square {spec['square']:.3f} mm, marker {spec['marker']:.3f} mm, "
            f"DICT_4X4_100, marker ids {ids.min()}-{ids.max()}\n"
            "Print at 100 % / actual size (no fit-to-page). Measure a square and use the measured size.\n"
            "Glue flat and wrinkle-free on a rigid backing; matte paper.", fontsize=7, va="top")
    out = PRINTS / f"{spec['name']}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"{out}: board {bw:.1f} x {bh:.1f} mm on A4")


# ----------------------------------------------------------------------------------------------- clamp
def load_stl(path):
    import struct

    d = path.read_bytes()
    n = struct.unpack("<I", d[80:84])[0]
    v = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"]
    v = np.round(v.reshape(-1, 3).astype(np.float64), 4)
    u, inv = np.unique(v, axis=0, return_inverse=True)
    return Manifold(Mesh(vert_properties=u.astype(np.float32), tri_verts=inv.reshape(-1, 3).astype(np.uint32)))


def section(m, z):
    return CrossSection(m.slice(z).to_polygons())


def write_stl(m, path):
    import struct

    mesh = m.to_mesh()
    v, f = np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts)
    tris = v[f].astype(np.float32)
    nrm = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    with open(path, "wb") as fh:
        fh.write(path.name.encode().ljust(80, b" ") + struct.pack("<I", len(tris)))
        for a, t in zip(nrm.astype(np.float32), tris):
            fh.write(a.tobytes() + t.tobytes() + b"\0\0")
    lo, hi = v.min(0), v.max(0)
    print(f"{path}: {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm, genus {m.genus()}")


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def cyl_y(d, x, z, y0, y1):
    return Manifold.cylinder(y1 - y0, d / 2, d / 2, 48).rotate((-90, 0, 0)).translate((x, y0, z))


def cyl_x(d, y, z, x0, x1):
    return Manifold.cylinder(x1 - x0, d / 2, d / 2, 48).rotate((0, 90, 0)).translate((x0, y, z))


def make_clamp():
    body = load_stl(PRINTS / "Wrist_Roll_Follower_SO101.stl")
    z_tip = body.bounding_box()[5]
    above = {}
    zs = np.arange(BAND[0], z_tip, LAYER)
    for z in zs:
        above[z] = section(body, z + LAYER / 2)
    cavity_layers, shell_layers = [], []
    acc = CrossSection()
    for z in zs[::-1]:  # from the tip down: union of everything above
        acc = acc + above[z]
        if z < BAND[1]:
            cav = acc.offset(CLEARANCE, JoinType.Round)
            cavity_layers.append(cav.extrude(LAYER).translate((0, 0, z)))
            shell_layers.append(cav.offset(WALL, JoinType.Round).extrude(LAYER).translate((0, 0, z)))
    cavity = Manifold.batch_boolean(cavity_layers, manifold3d.OpType.Add)
    shell = Manifold.batch_boolean(shell_layers, manifold3d.OpType.Add)
    bb = cavity.bounding_box()  # xmin, ymin, zmin, xmax, ymax, zmax
    x_in, x_out = bb[3], bb[0]  # inner (moving-jaw side) and outer cavity faces
    zm = (BAND[0] + BAND[1]) / 2
    # flange for the plate on the outer side
    flange = box(PLATE_BACK_X, x_out + 2.0, -14, 14, BAND[0], BAND[1])
    # pinch ears on the inner side
    ears = box(x_in, x_in + WALL + EAR, -(SLOT / 2 + 5), SLOT / 2 + 5, BAND[0], BAND[1])
    clamp = shell + flange + ears - cavity
    clamp -= box(x_in - 0.5, x_in + WALL + EAR + 1, -SLOT / 2, SLOT / 2, BAND[0] - 1, BAND[1] + 1)  # slot
    xs = x_in + WALL + EAR / 2 + 0.5
    clamp -= cyl_y(3.4, xs, zm, -20, 20)  # M3 clearance
    nut = CrossSection.circle(5.6 / np.sqrt(3), 6).extrude(3.0).rotate((90, 0, 0))  # M3 nut, 5.5 mm AF + play
    clamp -= nut.translate((xs, -(SLOT / 2 + 5) + 2.5, zm))
    for sy in (-1, 1):  # plate screws: M3 self-tapping pilot holes, 9 mm deep
        clamp -= cyl_x(2.6, sy * SCREW_Y, zm, PLATE_BACK_X - 1, PLATE_BACK_X + 9)
    return clamp, zm


def make_plate():
    plate = box(-PLATE / 2, PLATE / 2, -PLATE / 2, PLATE / 2, 0, PLATE_T)
    for sx in (-1, 1):  # countersunk M3 (90 deg, 6 mm head) flush with the board side (z = PLATE_T)
        x = sx * SCREW_Y
        plate -= Manifold.cylinder(PLATE_T + 2, 1.7, 1.7, 48).translate((x, 0, -1))
        plate -= Manifold.cylinder(1.7, 1.7, 3.1, 48).translate((x, 0, PLATE_T - 1.7 + 0.01))
    return plate


if __name__ == "__main__":
    for spec in (HANDEYE, STEREO):
        check_detection(spec)
    board_pdf(HANDEYE, plate_outline=PLATE)
    board_pdf(STEREO)
    clamp, zm = make_clamp()
    write_stl(clamp, PRINTS / "handeye_jaw_clamp.stl")
    write_stl(make_plate(), PRINTS / "handeye_board_plate.stl")
    # assembly check: clamp must not touch the gripper body
    body = load_stl(PRINTS / "Wrist_Roll_Follower_SO101.stl")
    print(f"clamp / gripper overlap: {(clamp ^ body).volume():.3f} mm3 (should be 0)")
