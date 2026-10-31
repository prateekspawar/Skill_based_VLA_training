"""End-to-end check of the real head's calibration: a horizontal textured surface seen by stereo must come out level
in the arm base, at its measured height (z ~ 0 for the surface the robot stands on; the floor under a table at
minus the table height).

    ~/lerobot-env/bin/python scripts/check_table_plane.py [--frames 3]

Sparse on purpose: SIFT matches between the left and right images (fundamental-matrix RANSAC), triangulated with
``real/stereo_calib.json`` (rectified disparity through Q), moved into the arm base with ``real/handeye.json`` (left camera
in base_link). A RANSAC plane fit over the selected points reports the plane's tilt from the base's x-y plane and
its height under the base and at the workspace centre. This tests the lenses, the stereo pair, the
hand-eye result and the frame conventions together. Textured surfaces only (a plain dark mat gives no matches);
whatever stands on the plane (the arm, objects) is rejected as outliers. Writes ``outputs/handeye/table_plane.jpg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
PI_HOST = "10.0.0.4"


def grab(sock):
    sock.send(b"get")
    jl, jr, _ = sock.recv_multipart()
    return [cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in (jl, jr)]


def triangulate(c, L, R):
    """Matched points: (N, 3) in the left camera frame [m] and their left-image pixels."""
    K1, D1, K2, D2 = (np.array(c[s][k], float) for s in ("left", "right") for k in ("K", "D"))
    R1, R2, P1, P2 = (np.array(c[k], float) for k in ("R1", "R2", "P1", "P2"))
    sift = cv2.SIFT_create(6000)
    ka, da = sift.detectAndCompute(cv2.cvtColor(L, cv2.COLOR_BGR2GRAY), None)
    kb, db = sift.detectAndCompute(cv2.cvtColor(R, cv2.COLOR_BGR2GRAY), None)
    good = [a for a, b in cv2.BFMatcher().knnMatch(da, db, k=2) if a.distance < 0.7 * b.distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    _, inl = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.5)
    pa, pb = pa[inl.ravel() > 0], pb[inl.ravel() > 0]
    ua = cv2.undistortPoints(pa.reshape(-1, 1, 2), K1, D1, R=R1, P=P1).reshape(-1, 2)
    ub = cv2.undistortPoints(pb.reshape(-1, 1, 2), K2, D2, R=R2, P=P2).reshape(-1, 2)
    dy = ua[:, 1] - ub[:, 1]
    print(f"  rectified row difference: median {np.median(dy):+.2f} px (0 when the stereo calibration is current)")
    keep = np.abs(dy - np.median(dy)) < 3.0  # matches must agree with the pair's common row offset
    Q = np.array(c["Q"], float)  # rectified pair: [x, y, disparity, 1] -> homogeneous 3-D point
    ua, ub = ua[keep], ub[keep]
    h = np.c_[ua, ua[:, 0] - ub[:, 0], np.ones(len(ua))] @ Q.T
    rect = h[:, :3] / h[:, 3:]
    return rect @ R1, pa[keep]  # rectified-left -> left camera: x_cam = R1^T x_rect


def fit_plane(p, rng, iters=2000, tol=0.004):
    best = None
    for _ in range(iters):
        s = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((p - s[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    q = p[best]
    centre = q.mean(0)
    n = np.linalg.svd(q - centre)[2][-1]  # least squares on the inliers
    return (n if n[2] > 0 else -n), centre, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--max-radius", type=float, default=1.5, help="[m] from the base axis")
    ap.add_argument("--max-z", type=float, default=-0.05,
                    help="[m] keep points below this (excludes the arm); the textured plane may be the floor under "
                         "the robot's table rather than the table itself")
    ap.add_argument("--stereo", default=str(ROOT / "so101_lang/real/stereo_calib.json"))
    ap.add_argument("--handeye", default=str(ROOT / "so101_lang/real/handeye.json"))
    args = ap.parse_args()
    c = json.loads(Path(args.stereo).read_text())
    X = np.array(json.loads(Path(args.handeye).read_text())["left_cam_in_base"])
    sock = zmq.Context().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.connect(f"tcp://{PI_HOST}:5601")
    pts, pix = [], []
    for _ in range(args.frames):
        L, R = grab(sock)
        pc, uv = triangulate(c, L, R)
        pts.append(pc @ X[:3, :3].T + X[:3, 3])
        pix.append(uv)
    pts, pix = np.concatenate(pts), np.concatenate(pix)
    r = np.hypot(pts[:, 0], pts[:, 1])
    sel = (r > 0.12) & (r < args.max_radius) & (pts[:, 2] < args.max_z)
    print(f"{len(pts)} triangulated points, {sel.sum()} within 0.12-{args.max_radius} m of the base axis and below "
          f"z = {args.max_z} m")
    n, centre, inl = fit_plane(pts[sel], np.random.default_rng(0))
    q = pts[sel][inl]
    tilt = np.degrees(np.arccos(n[2]))
    z_at = lambda x, y: centre[2] - (n[0] * (x - centre[0]) + n[1] * (y - centre[1])) / n[2]  # noqa: E731
    resid = (q - centre) @ n
    print(f"plane: {inl.sum()} inliers, residual std {1000 * resid.std():.1f} mm, spans x {q[:, 0].min():.2f}..{q[:, 0].max():.2f} "
          f"y {q[:, 1].min():.2f}..{q[:, 1].max():.2f} m")
    print(f"tilt from the base x-y plane: {tilt:.2f} deg (normal {np.round(n, 3).tolist()}, "
          f"downhill toward azimuth {np.degrees(np.arctan2(n[1], n[0])):.0f} deg)")
    print(f"height: under the base {1000 * z_at(0, 0):+.1f} mm, at the workspace centre (0.25, 0) {1000 * z_at(0.25, 0):+.1f} mm, "
          f"at (0.25, +-0.15) {1000 * z_at(0.25, 0.15):+.1f} / {1000 * z_at(0.25, -0.15):+.1f} mm")
    img = L.copy()
    uv_all = pix[sel]
    for (u, v), z, ok in zip(uv_all, pts[sel][:, 2], inl):
        col = (0, 255, 0) if ok else (0, 0, 255)
        cv2.circle(img, (int(u), int(v)), 3, col, -1)
    out = ROOT / "outputs/handeye/table_plane.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.resize(img, None, fx=0.6, fy=0.6))
    print("wrote", out, "(green = on the plane, red = off it)")


if __name__ == "__main__":
    main()
