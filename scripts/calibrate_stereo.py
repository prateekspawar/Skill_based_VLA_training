"""Stereo calibration of the real head cameras with the ChArUco board (``hardware/prints/charuco_stereo_8x11_22mm.pdf``).

    ~/lerobot-env/bin/python scripts/calibrate_stereo.py capture            # live window, auto-saves good pairs
    ~/lerobot-env/bin/python scripts/calibrate_stereo.py capture --resume <dir> --n 25   # add views to a set
    ~/lerobot-env/bin/python scripts/calibrate_stereo.py solve <dir> [--square 22.0]

``capture`` pulls exposure-matched pairs from ``so101_lang/real/stereo_host.py`` on the Pi and saves a pair when the
board is seen by both cameras, has been still for a moment (no motion blur), and sits in a new place / size / tilt
compared with the pairs already saved. The window shows both views, the detected corners and the coverage so far.
Keys: q = finish, s = save this pair anyway, u = undo the last save.
``solve`` calibrates each camera, then the pair (OpenCV stereoCalibrate with the intrinsics refined jointly), drops
outlier views, rectifies, and writes ``stereo_calib.json`` (K, distortion, R, T, rectification maps' inputs, errors).
Measure a printed square with calipers and pass it as ``--square`` (mm).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

NX, NY, DICT = 8, 11, cv2.aruco.DICT_4X4_100  # as hardware/calibration/make_calibration.py (STEREO)
PI_HOST = "10.0.0.4"


def make_board(square_mm: float):
    ids = np.arange(0, NX * NY // 2)
    return cv2.aruco.CharucoBoard((NX, NY), square_mm / 1000, 0.75 * square_mm / 1000,
                                  cv2.aruco.getPredefinedDictionary(DICT), ids)


def detect(detector, img):
    """ChArUco corners (N, 2) and ids (N,), or empty arrays."""
    corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 4:
        return np.zeros((0, 2), np.float32), np.zeros((0,), int)
    return corners.reshape(-1, 2), ids.ravel()


# --------------------------------------------------------------------------------------------- capture
def signature(c, size):
    """Where the board is, how big and how tilted (normalized), to keep the saved views diverse."""
    w, h = size
    centre = c.mean(0) / [w, h]
    cov = np.cov((c / [w, h]).T)
    ev, evec = np.linalg.eigh(cov)
    scale = np.sqrt(ev.sum())
    elong = np.sqrt(ev[1] / max(ev[0], 1e-9))  # foreshortening from tilt
    angle = np.arctan2(evec[1, 1], evec[0, 1]) % np.pi
    return np.array([centre[0], centre[1], 3 * scale, 0.3 * (elong - 1), 0.15 * np.cos(2 * angle), 0.15 * np.sin(2 * angle)])


def capture(args):
    import zmq

    out = Path(args.resume) if args.resume else Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    detector = cv2.aruco.CharucoDetector(make_board(22.0))
    sock = zmq.Context().socket(zmq.REQ)
    sock.connect(f"tcp://{args.host}:5601")
    saved, sigs, prev, still_since, last_save = [], [], None, None, 0.0
    cover = None
    for fl in sorted(out.glob("pair_*_L.png")):  # --resume: keep the earlier pairs' views and coverage
        L, R = cv2.imread(str(fl)), cv2.imread(str(fl).replace("_L.png", "_R.png"))
        h, w = L.shape[:2]
        cover = np.zeros((h, 2 * w), np.uint8) if cover is None else cover
        (cl, il), (cr, ir) = detect(detector, L), detect(detector, R)
        saved.append(int(fl.stem.split("_")[1]))
        if len(il):
            sigs.append(signature(cl, (w, h)))
        for c, x0 in ((cl, 0), (cr, w)):
            for p in c:
                cv2.circle(cover, (int(p[0]) + x0, int(p[1])), 14, 255, -1)
    args.n += len(saved)
    if saved:
        print(f"resuming {out}: {len(saved)} pairs already there")
    cv2.namedWindow("stereo calibration", cv2.WINDOW_NORMAL)
    print(f"saving to {out}; target {args.n} pairs. Keys: q finish, s save anyway, u undo")
    while True:
        sock.send(b"get")
        jl, jr, meta = sock.recv_multipart()
        L, R = (cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) for b in (jl, jr))
        h, w = L.shape[:2]
        if cover is None:
            cover = np.zeros((h, 2 * w), np.uint8)
        (cl, il), (cr, ir) = detect(detector, L), detect(detector, R)
        common = np.intersect1d(il, ir)
        now = time.time()
        ok = len(il) >= args.min_corners and len(ir) >= args.min_corners and len(common) >= args.min_corners
        moved = np.inf
        if ok and prev is not None:
            ids_p, c_p = prev
            both = np.intersect1d(ids_p, il)
            if len(both) >= 6:
                moved = np.median(np.linalg.norm(cl[np.searchsorted(il, both)] - c_p[np.searchsorted(ids_p, both)], axis=1))
        prev = (il, cl) if ok else None
        still_since = (still_since or now) if moved < args.still_px else None
        sig = signature(cl, (w, h)) if ok else None
        novelty = min((np.linalg.norm(sig - s) for s in sigs), default=np.inf) if ok else 0.0
        status = "board not seen by both cameras"
        if ok:
            status = (f"corners L {len(il)} R {len(ir)} common {len(common)} | "
                      + ("hold still..." if still_since is None or now - still_since < args.still_s
                         else "too similar to a saved view: move / tilt the board" if novelty < args.novelty
                         else "SAVING"))
        key = cv2.waitKey(1) & 0xFF
        auto = ok and still_since is not None and now - still_since >= args.still_s and novelty >= args.novelty
        if (auto and now - last_save > 0.8) or (key == ord("s") and ok):
            k = max(saved, default=-1) + 1
            cv2.imwrite(str(out / f"pair_{k:03d}_L.png"), L)
            cv2.imwrite(str(out / f"pair_{k:03d}_R.png"), R)
            (out / f"pair_{k:03d}_meta.json").write_text(meta.decode())
            saved.append(k)
            sigs.append(sig)
            last_save, still_since = now, None
            for c, x0 in ((cl, 0), (cr, w)):
                for p in c:
                    cv2.circle(cover, (int(p[0]) + x0, int(p[1])), 14, 255, -1)
            print(f"saved pair {k:3d}: {status}", flush=True)
        if key == ord("u") and saved:
            k = saved.pop()
            sigs.pop()
            for side in ("L", "R"):
                (out / f"pair_{k:03d}_{side}.png").unlink()
            (out / f"pair_{k:03d}_meta.json").unlink()
            print(f"removed pair {k}")  # coverage overlay keeps its dots
        # ----- display -----
        view = np.hstack([L, R])
        tint = view.copy()
        tint[cover > 0] = (0.6 * tint[cover > 0] + [0, 90, 0]).clip(0, 255).astype(np.uint8)
        view = tint
        for c, ids, x0 in ((cl, il, 0), (cr, ir, w)):
            for p, i in zip(c, ids):
                col = (0, 255, 0) if i in common else (0, 0, 255)
                cv2.circle(view, (int(p[0]) + x0, int(p[1])), 5, col, -1)
        bar = np.zeros((90, view.shape[1], 3), np.uint8)
        cv2.putText(bar, f"saved {len(saved)}/{args.n}   {status}", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255) if "SAVING" in status else (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bar, "green tint = covered so far: fill both images, incl. corners and edges; tilt the board "
                    "up to ~45 deg; near (0.35 m) and far (0.8 m)", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(view, "LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(view, "RIGHT", (w + 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imshow("stereo calibration", cv2.resize(np.vstack([view, cv2.resize(bar, (view.shape[1], 90))]), None,
                                                    fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
        if key == ord("q") or len(saved) >= args.n:
            break
    cv2.destroyAllWindows()
    print(f"{len(saved)} pairs in {out}")


# ----------------------------------------------------------------------------------------------- solve
def solve(args):
    d = Path(args.dir)
    board = make_board(args.square)
    detector = cv2.aruco.CharucoDetector(board)
    obj_all = board.getChessboardCorners()
    views = []
    for fl in sorted(d.glob("pair_*_L.png")):
        L, R = cv2.imread(str(fl)), cv2.imread(str(fl).replace("_L.png", "_R.png"))
        (cl, il), (cr, ir) = detect(detector, L), detect(detector, R)
        views.append(dict(name=fl.stem[:-2], L=(cl, il), R=(cr, ir)))
    size = (L.shape[1], L.shape[0])
    print(f"{len(views)} pairs, image size {size}, square {args.square} mm")

    def mono(side, keep):
        obj = [obj_all[v[side][1]].astype(np.float32) for v in keep]
        img = [v[side][0].astype(np.float32) for v in keep]
        # Standard 5-coefficient model: the 8-coefficient rational model fit the OV5647s 0.003 px better but its
        # mapping folded back before the image corners (non-monotonic along the diagonal).
        rms, K, D, rv, tv = cv2.calibrateCamera(obj, img, size, None, None)
        per = [np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, D)[0].reshape(-1, 2) - i) ** 2, 1)))
               for o, i, r, t in zip(obj, img, rv, tv)]
        return rms, K, D, np.array(per)

    def stereo(keep, Kl, Dl, Kr, Dr):
        obj, pl, pr = [], [], []
        for v in keep:
            (cl, il), (cr, ir) = v["L"], v["R"]
            common = np.intersect1d(il, ir)
            obj.append(obj_all[common].astype(np.float32))
            pl.append(cl[np.searchsorted(il, common)].astype(np.float32))
            pr.append(cr[np.searchsorted(ir, common)].astype(np.float32))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
        out = cv2.stereoCalibrateExtended(obj, pl, pr, Kl, Dl, Kr, Dr, size, None, None, None, None,
                                          flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)
        rms, Kl, Dl, Kr, Dr, R, T, E, F, _, _, per = out
        return rms, Kl, Dl, Kr, Dr, R, T, per.max(1), (obj, pl, pr)

    keep = [v for v in views if min(len(v["L"][1]), len(v["R"][1]), len(np.intersect1d(v["L"][1], v["R"][1]))) >= 12]
    for it in range(3):  # drop views whose stereo error is far above the rest, re-solve
        rl, Kl, Dl, pel = mono("L", keep)
        rr, Kr, Dr, per_ = mono("R", keep)
        rs, Kl, Dl, Kr, Dr, R, T, pes, pts = stereo(keep, Kl, Dl, Kr, Dr)
        bad = [i for i, e in enumerate(pes) if e > max(3 * np.median(pes), 0.6)]
        print(f"pass {it}: {len(keep)} views | RMS left {rl:.3f} px, right {rr:.3f} px, stereo {rs:.3f} px"
              + (f" | dropping {[keep[i]['name'] for i in bad]}" if bad else ""))
        if not bad:
            break
        keep = [v for i, v in enumerate(keep) if i not in bad]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(Kl, Dl, Kr, Dr, size, R, T, alpha=0)
    # rectification check: matched corners must land on the same row
    dy = []
    for o, a, b in zip(*pts):
        ua = cv2.undistortPoints(a.reshape(-1, 1, 2), Kl, Dl, R=R1, P=P1).reshape(-1, 2)
        ub = cv2.undistortPoints(b.reshape(-1, 1, 2), Kr, Dr, R=R2, P=P2).reshape(-1, 2)
        dy.append(np.abs(ua[:, 1] - ub[:, 1]))
    dy = np.concatenate(dy)
    rot_deg = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    fov = lambda K: np.degrees(2 * np.arctan(size[0] / 2 / K[0, 0]))
    res = {
        "image_size": list(size), "square_mm": args.square, "n_views": len(keep), "views": [v["name"] for v in keep],
        "rms_px": {"left": rl, "right": rr, "stereo": rs},
        "rectified_row_error_px": {"median": float(np.median(dy)), "p95": float(np.percentile(dy, 95))},
        "baseline_mm": float(np.linalg.norm(T) * 1000), "T_mm": (T.ravel() * 1000).tolist(), "R_deg": rot_deg,
        "left": {"K": Kl.tolist(), "D": Dl.ravel().tolist(), "hfov_deg": fov(Kl)},
        "right": {"K": Kr.tolist(), "D": Dr.ravel().tolist(), "hfov_deg": fov(Kr)},
        "R": R.tolist(), "T": T.ravel().tolist(), "R1": R1.tolist(), "R2": R2.tolist(), "P1": P1.tolist(),
        "P2": P2.tolist(), "Q": Q.tolist(), "distortion_model": "standard (k1 k2 p1 p2 k3)",
        "convention": "left = cam1 (stereo_config.json); T, R map left-camera coordinates to the right camera [m]",
    }
    (d / "stereo_calib.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("n_views", "rms_px", "rectified_row_error_px", "baseline_mm", "T_mm",
                                          "R_deg")}, indent=1))
    print(f"fx left {Kl[0, 0]:.1f} right {Kr[0, 0]:.1f} px, horizontal FOV {fov(Kl):.1f} / {fov(Kr):.1f} deg")
    # rectified preview of the first kept pair, with row lines
    L, Rimg = (cv2.imread(str(d / f"{keep[0]['name']}_{s}.png")) for s in "LR")
    m1 = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, size, cv2.CV_32FC1)
    pair = np.hstack([cv2.remap(L, *m1, cv2.INTER_LINEAR), cv2.remap(Rimg, *m2, cv2.INTER_LINEAR)])
    for y in range(40, size[1], 60):
        cv2.line(pair, (0, y), (pair.shape[1], y), (0, 255, 255), 1)
    cv2.imwrite(str(d / "rectified_check.jpg"), cv2.resize(pair, None, fx=0.5, fy=0.5))
    print(f"wrote {d / 'stereo_calib.json'} and {d / 'rectified_check.jpg'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--host", default=PI_HOST)
    c.add_argument("--out", default="outputs/stereo_calib")
    c.add_argument("--n", type=int, default=35, help="pairs to collect (in this session)")
    c.add_argument("--resume", help="add pairs to an earlier capture directory")
    c.add_argument("--min_corners", type=int, default=20)
    c.add_argument("--still_px", type=float, default=1.5, help="max corner motion between frames to count as still")
    c.add_argument("--still_s", type=float, default=0.5)
    c.add_argument("--novelty", type=float, default=0.12)
    s = sub.add_parser("solve")
    s.add_argument("dir")
    s.add_argument("--square", type=float, default=22.0, help="measured square size [mm]")
    a = ap.parse_args()
    capture(a) if a.cmd == "capture" else solve(a)
