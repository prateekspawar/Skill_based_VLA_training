"""Hand-eye calibration of the real stereo head: where the cameras sit relative to the SO-101 base.

    ~/lerobot-env/bin/python scripts/calibrate_handeye.py capture [--n 24]   # drives the arm, saves views
    ~/lerobot-env/bin/python scripts/calibrate_handeye.py solve <dir>        # solve + write handeye.json

Target: the 5 x 5 ChArUco board (18 mm squares, ids 50-61, ``hardware/prints/charuco_handeye_5x5_18mm.pdf``) on the
moving jaw's outer face (``--board-on moving``, default) or the fixed jaw (``fixed``, clamp of hardware/calibration).
The camera is fixed and the board rides on the gripper (eye-to-hand), so two transforms are unknown: the left camera
in the arm base (X) and the board in its jaw link (Y). For every pose, ``left_cam <- board = X^-1 . fk(q) . Y``; the
right camera adds the stereo extrinsics. On the moving jaw the opening is a sixth joint: it tilts the board about the
hinge, so the steep head view can see it face-on. The gripper's zero offset is absorbed by Y; its scale comes from
the LeRobot calibration (% of the calibrated range).

``capture`` needs ``arm_host.py`` running on the Pi (reached through an ssh tunnel) and ``stereo_host.py``.
1. probe: with ``--rough`` (an earlier solve, head unchanged) a wrist roll x opening sweep at a few gripper-down
   poses until the board is seen, which gives Y directly; without it, 8 poses planned from priors (head at the
   training pose ``--train-calib``, board as mounted on the fixed jaw) and a rough solve (Park-Martin + refinement);
3. plan: random joint configurations (roll and opening chosen to face the board at the camera) whose predicted
   view is inside the left image, not hidden by the arm, at a usable distance, with the gripper and board clear of
   the table, base and head (also along the joint path); a diverse subset is visited nearest-first;
4. return to the first pose to check repeatability (clamp rigidity, backlash).
At each pose the arm settles, then the measured joint angles and one synced pair are saved. Window: q = stop.

``solve`` refines X, Y (and optionally the lift/elbow/wrist_flex zero offsets) by minimising the corner reprojection
error in both cameras, for the URDF wrist and the sim asset's wrist (no 2.8 deg tilt), and writes ``handeye.json``
with ``head_in_base`` in the head-frame convention of the sim (``docs/NOTES.md``, frames) for both the URDF base
and the sim's base body, plus the matching ``SO101_CAM_OFFSET`` against a training calibration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from so101_lang.real import kinematics  # noqa: E402

NX, NY, ID0, DICT = 5, 5, 50, cv2.aruco.DICT_4X4_100  # as hardware/calibration/make_calibration.py (HANDEYE)
PI_HOST, TUNNEL_PORT = "10.0.0.4", 15610
STEREO_CALIB = ROOT / "so101_lang/real/stereo_calib.json"
ARM = kinematics.ARM
OFFSET_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex")  # pan offset ~ X, roll offset ~ Y: not identifiable

# Probe: poses planned with the prior head pose (the training head; the real head is mounted to match it) and the
# board's mounting (prior_board_in_gripper), kept well inside the image and facing the cameras.
PROBE_N = 8
# Planning ranges (deg) and safety margins (m, base frame; the table top is base z ~ 0).
PLAN_RANGE = dict(shoulder_pan=(-70, 70), shoulder_lift=(-45, 45), elbow_flex=(-70, 60), wrist_flex=(-85, 85),
                  wrist_roll=(-165, 165))  # the host keeps 10 deg from the calibrated ends (wrist_flex +-86)
MIN_Z, MIN_RADIUS = 0.05, 0.11
HEAD_KEEP_OUT = 0.17  # [m] horizontal radius around the head and its post, once the rough solve has located it
KEEP_OUT_XY = None  # head position (x, y) in the base, set after the rough solve
GRIPPER_OPEN = 90.0  # % for the fixed-jaw board (README: keep open; the clamp sits where the moving jaw closes)
OPENINGS = (35.0, 50.0, 65.0, 80.0, 95.0)  # % tried for the moving-jaw board
BOARD_ON = "moving"  # set from --board-on
JAW_LINK = {"fixed": "gripper_link", "moving": "moving_jaw_so101_v1_link"}
_CAL = json.loads((Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json").read_text())
GRIPPER_DEG_PER_PCT = (_CAL["gripper"]["range_max"] - _CAL["gripper"]["range_min"]) * 360 / 4096 / 100


def jaw(q: dict, sim: bool = False) -> np.ndarray:
    """Pose of the board's jaw link in base_link; ``q`` in arm degrees + gripper % (moving jaw only)."""
    if BOARD_ON == "fixed":
        return kinematics.fk(q, sim=sim)
    return kinematics.fk({**q, "gripper": q["gripper"] * GRIPPER_DEG_PER_PCT}, JAW_LINK["moving"], sim)


def joints(q: dict) -> tuple:
    return ARM + (("gripper",) if BOARD_ON == "moving" and "gripper" in q else ())


def make_board(square_mm: float):
    return cv2.aruco.CharucoBoard((NX, NY), square_mm / 1000, 0.75 * square_mm / 1000,
                                  cv2.aruco.getPredefinedDictionary(DICT), np.arange(ID0, ID0 + NX * NY // 2))


def detect(detector, img):
    """ChArUco corners (N, 2) and ids; retried on a 2x upscale, as the 90 mm board is small in the head's view."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for scale in (1, 2):
        gi = g if scale == 1 else cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        corners, ids, _, _ = detector.detectBoard(gi)
        if ids is not None and len(ids) >= 6:
            return (corners.reshape(-1, 2) + 0.5) / scale - 0.5, ids.ravel()
    return np.zeros((0, 2), np.float32), np.zeros((0,), int)


def load_stereo():
    c = json.loads(STEREO_CALIB.read_text())
    a = {k: np.array(c[k], float) for k in ("R", "T", "R1")}
    for s in ("left", "right"):
        a[f"K_{s}"], a[f"D_{s}"] = np.array(c[s]["K"], float), np.array(c[s]["D"], float).ravel()
    a["size"] = tuple(c["image_size"])
    a["T"] = a["T"].ravel()
    return a


def to_T(rvec, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = Rot.from_rotvec(rvec).as_matrix(), t
    return m


def from_T(m):
    return np.r_[Rot.from_matrix(m[:3, :3]).as_rotvec(), m[:3, 3]]


def inv(m):
    r = np.eye(4)
    r[:3, :3], r[:3, 3] = m[:3, :3].T, -m[:3, :3].T @ m[:3, 3]
    return r


def project(st, cam_T_pts, pts, side):
    """Pixels of board points ``pts`` (N, 3) given the left camera <- board transform."""
    m = cam_T_pts if side == "left" else np.r_[np.c_[st["R"], st["T"]], [[0, 0, 0, 1]]] @ cam_T_pts
    uv, _ = cv2.projectPoints(pts, cv2.Rodrigues(m[:3, :3])[0], m[:3, 3], st[f"K_{side}"], st[f"D_{side}"])
    return uv.reshape(-1, 2), (m[:3, :3] @ pts.T).T + m[:3, 3]


def board_pose_pnp(st, board, corners, ids, side="left"):
    obj = board.getChessboardCorners()[ids]
    ok, rv, tv = cv2.solvePnP(obj, corners, st[f"K_{side}"], st[f"D_{side}"], flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        return None
    rv, tv = cv2.solvePnPRefineLM(obj, corners, st[f"K_{side}"], st[f"D_{side}"], rv, tv)
    m = to_T(rv.ravel(), tv.ravel())
    if side == "right":  # into the left camera
        m = inv(np.r_[np.c_[st["R"], st["T"]], [[0, 0, 0, 1]]]) @ m
    return m


# --------------------------------------------------------------------------------------------------- solve
def fk_all(views, offsets, sim):
    return [jaw({j: v["q"][j] + offsets.get(j, 0.0) for j in joints(v["q"])}, sim) for v in views]


def pnp_side(v):
    """(corners, ids, side) of the camera that sees more of the board."""
    return (*v["L"], "left") if len(v["L"][1]) >= len(v["R"][1]) else (*v["R"], "right")


def park_martin(A, B):
    """X with A_k X = X B_k (Park & Martin 1994): rotation from the rotation logs, then translation by least squares."""
    M = sum(np.outer(Rot.from_matrix(b[:3, :3]).as_rotvec(), Rot.from_matrix(a[:3, :3]).as_rotvec()) for a, b in zip(A, B))
    w, v = np.linalg.eigh(M.T @ M)
    r = v @ np.diag(w ** -0.5) @ v.T @ M.T
    lhs = np.concatenate([a[:3, :3] - np.eye(3) for a in A])
    rhs = np.concatenate([r @ b[:3, 3] - a[:3, 3] for a, b in zip(A, B)])
    return np.r_[np.c_[r, np.linalg.lstsq(lhs, rhs, rcond=None)[0]], [[0, 0, 0, 1]]]


def initial_guess(st, board, views, sim):
    """Eye-to-hand: C_i = X^-1 F_i Y gives (F_i F_j^-1) X = X (C_i C_j^-1); then Y averaged over the views."""
    F = fk_all(views, {}, sim)
    C = [board_pose_pnp(st, board, *pnp_side(v)) for v in views]
    pairs = [(i, j) for i in range(len(views)) for j in range(i + 1, len(views))]
    X = park_martin([F[i] @ inv(F[j]) for i, j in pairs], [C[i] @ inv(C[j]) for i, j in pairs])
    Ys = [inv(f) @ X @ c for f, c in zip(F, C)]
    rot = Rot.from_matrix([y[:3, :3] for y in Ys]).mean().as_matrix()
    Y = np.r_[np.c_[rot, np.mean([y[:3, 3] for y in Ys], 0)], [[0, 0, 0, 1]]]
    return X, Y


def residuals(p, st, board, views, sim, fit_offsets):
    X, Y = to_T(p[:3], p[3:6]), to_T(p[6:9], p[9:12])
    offs = dict(zip(OFFSET_JOINTS, p[12:15])) if fit_offsets else {}
    Xi, obj, out = inv(X), board.getChessboardCorners(), []
    for f, v in zip(fk_all(views, offs, sim), views):
        cTb = Xi @ f @ Y
        for side in ("left", "right"):
            c, ids = v[side[0].upper()]
            if len(ids):
                out.append((project(st, cTb, obj[ids], side)[0] - c).ravel())
    return np.concatenate(out)


def refine(st, board, views, X, Y, sim, fit_offsets, loss="soft_l1"):
    p0 = np.r_[from_T(X), from_T(Y), np.zeros(3) if fit_offsets else []]
    r = least_squares(residuals, p0, args=(st, board, views, sim, fit_offsets), loss=loss, f_scale=1.0,
                      x_scale="jac")
    X, Y = to_T(r.x[:3], r.x[3:6]), to_T(r.x[6:9], r.x[9:12])
    offs = dict(zip(OFFSET_JOINTS, r.x[12:15])) if fit_offsets else {}
    return X, Y, offs, r


def per_view_rms(st, board, views, X, Y, offs, sim):
    obj, Xi, out = board.getChessboardCorners(), inv(X), []
    for f, v in zip(fk_all(views, offs, sim), views):
        e = []
        for side in ("left", "right"):
            c, ids = v[side[0].upper()]
            if len(ids):
                e.append(project(st, Xi @ f @ Y, obj[ids], side)[0] - c)
        out.append(float(np.sqrt(np.mean(np.concatenate(e) ** 2))))
    return np.array(out)


def solve_all(st, board, views, fit_offsets_options=(False, True), verbose=True):
    """Best model over wrist geometry (URDF / sim asset) x zero offsets; drops views > 3 x median."""
    best = None
    for sim in (False, True):
        X0, Y0 = initial_guess(st, board, views, sim)
        for fo in fit_offsets_options:
            if fo and len(views) < 10:
                continue
            X, Y, offs, _ = refine(st, board, views, X0, Y0, sim, fo)
            e = per_view_rms(st, board, views, X, Y, offs, sim)
            keep = e <= max(3 * np.median(e), 1.0)
            vk = [v for v, k in zip(views, keep) if k]
            X, Y, offs, _ = refine(st, board, vk, X, Y, sim, fo)
            ek = per_view_rms(st, board, vk, X, Y, offs, sim)
            rms = float(np.sqrt(np.mean(ek ** 2)))
            if verbose:
                print(f"  wrist {'sim' if sim else 'urdf'} offsets {'fit' if fo else 'none'}: rms {rms:.2f} px, "
                      f"{keep.sum()}/{len(views)} views" + (f", offsets {{{', '.join(f'{k}: {v:+.2f}' for k, v in offs.items())}}} deg" if fo else ""))
            cand = dict(sim=sim, fit_offsets=fo, X=X, Y=Y, offsets=offs, rms=rms, keep=keep, per_view=e)
            # offsets must earn their keep: > 10 % lower error
            if best is None or rms < best["rms"] * (0.9 if fo and not best["fit_offsets"] else 1.0):
                best = cand
    return best


def restereo(st, board, views, min_common=8):
    """Re-measure the left-right pose from the views both cameras see (intrinsics fixed) and recompute the
    rectification. The camera modules shift on the plate; the hand-eye views are a fresh stereo calibration."""
    obj_all, O, PL, PR = board.getChessboardCorners(), [], [], []
    for v in views:
        (cl, il), (cr, ir) = v["L"], v["R"]
        common = np.intersect1d(il, ir)
        if len(common) >= min_common:
            O.append(obj_all[common].astype(np.float32))
            PL.append(cl[np.searchsorted(il, common)].astype(np.float32))
            PR.append(cr[np.searchsorted(ir, common)].astype(np.float32))
    if len(O) < 8:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)
    rms, *_, R, T, _, _ = cv2.stereoCalibrate(O, PL, PR, st["K_left"], st["D_left"], st["K_right"], st["D_right"],
                                             st["size"], flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(st["K_left"], st["D_left"], st["K_right"], st["D_right"], st["size"],
                                                R, T, alpha=0)
    old = (np.linalg.norm(st["T"]) * 1000, np.degrees(np.linalg.norm(cv2.Rodrigues(st["R"])[0])))
    st.update(R=R, T=T.ravel(), R1=R1)
    return dict(n_pairs=len(O), rms_px=float(rms), baseline_mm=float(np.linalg.norm(T) * 1000),
                R_deg=float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))), previous_baseline_mm=old[0],
                previous_R_deg=old[1], R=R.tolist(), T=T.ravel().tolist(), R1=R1.tolist(), R2=R2.tolist(),
                P1=P1.tolist(), P2=P2.tolist(), Q=Q.tolist())


def head_from_left_cam(st) -> np.ndarray:
    """Head frame (origin between the optical centres, x forward, y left, z up; axes of the rectified pair) in the
    left camera's frame."""
    c_right = -st["R"].T @ st["T"]  # right optical centre in the left camera frame
    # columns: head x, y, z in rectified-camera axes (x right, y down, z forward) = z, -x, -y
    r_rect = st["R1"].T @ np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = r_rect, c_right / 2
    return m


def rpy_deg(r):
    """roll, pitch, yaw with R = Rz(yaw) Ry(pitch) Rx(roll) (as so101_lang.robots.virtual_head)."""
    return Rot.from_matrix(r).as_euler("ZYX", degrees=True)[::-1]


def solve(args):
    d = Path(args.dir)
    st, board = load_stereo(), make_board(args.square)
    views = load_views(d, board)
    print(f"{len(views)} views with the board in at least one camera "
          f"({sum(min(len(v['L'][1]), len(v['R'][1])) >= 6 for v in views)} in both)")
    stereo = restereo(st, board, views) if args.restereo else None
    if stereo:
        print(f"stereo re-measured from {stereo['n_pairs']} views: rms {stereo['rms_px']:.2f} px, baseline "
              f"{stereo['previous_baseline_mm']:.2f} -> {stereo['baseline_mm']:.2f} mm, rotation "
              f"{stereo['previous_R_deg']:.2f} -> {stereo['R_deg']:.2f} deg")
    best = solve_all(st, board, views)
    X, Y = best["X"], best["Y"]
    head_L = head_from_left_cam(st)
    head_in_base = X @ head_L
    head_in_simbase = kinematics.SIM_BASE_FROM_BASE @ head_in_base
    out = dict(
        convention=f"X = left camera (OpenCV) in base_link; Y = board in {JAW_LINK[BOARD_ON]}; head frame: origin "
                   "between the optical centres, +x forward, +y left, +z up, rectified-pair axes; [m]",
        board_on=BOARD_ON,
        square_mm=args.square, n_views=int(best["keep"].sum()), views_total=len(views),
        wrist_model="sim asset (no tilt)" if best["sim"] else "URDF (2.8 deg wrist_roll tilt)",
        joint_zero_offsets_deg=best["offsets"], rms_px=best["rms"],
        per_view_rms_px=best["per_view"].round(3).tolist(),
        left_cam_in_base=X.tolist(), board_in_gripper=Y.tolist(),
        head_in_base_link=head_in_base.tolist(), head_in_sim_base=head_in_simbase.tolist(),
        stereo_from_views=stereo,
    )
    p = head_in_simbase[:3, 3]
    print(f"best: wrist {out['wrist_model']}, offsets {best['offsets'] or 'none'}, rms {best['rms']:.2f} px")
    print(f"head in sim base: position {np.round(p, 4).tolist()} m, rpy {np.round(rpy_deg(head_in_simbase[:3, :3]), 2).tolist()} deg")
    rep = repeatability(d, st, board, X, Y, best["offsets"])
    if rep:
        out["repeatability"] = rep
        print(f"repeatability (first pose vs return): {rep['mm']:.2f} mm, {rep['deg']:.3f} deg; not explained by the "
              f"measured joints: {rep['unexplained_mm']:.2f} mm, {rep['unexplained_deg']:.3f} deg")
    if args.train_calib:
        train = np.array(json.loads(Path(args.train_calib).read_text())["head_in_base"])
        real_in_virtual = inv(train) @ head_in_simbase
        off = np.r_[real_in_virtual[:3, 3], rpy_deg(real_in_virtual[:3, :3])]
        out["SO101_CAM_OFFSET"] = ",".join(f"{v:.4f}" if i < 3 else f"{v:.2f}" for i, v in enumerate(off))
        print(f"SO101_CAM_OFFSET={out['SO101_CAM_OFFSET']}  (vs {args.train_calib})")
    (d / "handeye.json").write_text(json.dumps(out, indent=1))
    print("wrote", d / "handeye.json")


def load_views(d: Path, board):
    det = cv2.aruco.CharucoDetector(board)
    views = []
    for m in sorted(d.glob("pose_*_meta.json")):
        meta = json.loads(m.read_text())
        if meta.get("kind") == "return":
            continue
        L = detect(det, cv2.imread(str(m).replace("_meta.json", "_L.png")))
        R = detect(det, cv2.imread(str(m).replace("_meta.json", "_R.png")))
        if max(len(L[1]), len(R[1])) >= 6:  # one camera is enough; detect() returns nothing below 6 corners
            views.append(dict(name=m.stem, q=meta["q"], L=L, R=R))
    if views and BOARD_ON == "moving" and "gripper" not in views[0]["q"]:
        raise SystemExit("these views have no gripper angle: solve them with --board-on fixed")
    return views


def repeatability(d, st, board, X=None, Y=None, offs=None):
    """Board pose at the return visit vs the visit it repeats (the pose with the nearest joint angles). With the
    solution, also the part the measured joint angles do not explain (mount slip; the arm settles a little
    differently depending on where it came from)."""
    metas = sorted(d.glob("pose_*_meta.json"))
    ret = metas[-1]
    rq = json.loads(ret.read_text())
    if rq.get("kind") != "return":
        return None
    first = min(metas[:-1], key=lambda m: max(abs(json.loads(m.read_text())["q"][j] - rq["q"][j]) for j in rq["q"]))
    det = cv2.aruco.CharucoDetector(board)
    poses, qs = [], []
    for m in (first, ret):
        c, ids = detect(det, cv2.imread(str(m).replace("_meta.json", "_L.png")))
        if len(ids) < 6:
            return None
        poses.append(board_pose_pnp(st, board, c, ids))
        qs.append(json.loads(m.read_text())["q"])
    dm = inv(poses[0]) @ poses[1]
    out = dict(mm=float(1000 * np.linalg.norm(poses[1][:3, 3] - poses[0][:3, 3])),
               deg=float(np.degrees(Rot.from_matrix(dm[:3, :3]).magnitude())),
               joint_diff_deg={j: round(qs[1][j] - qs[0][j], 2) for j in qs[0]})
    if X is not None:
        pred = [inv(X) @ jaw({j: q[j] + offs.get(j, 0.0) for j in joints(q)}) @ Y for q in qs]
        u = inv(poses[1]) @ pred[1] @ inv(pred[0]) @ poses[0]
        out.update(unexplained_mm=float(1000 * np.linalg.norm(u[:3, 3])),
                   unexplained_deg=float(np.degrees(Rot.from_matrix(u[:3, :3]).magnitude())))
    return out


# ------------------------------------------------------------------------------------------------- capture
class Arm:
    """arm_host.py on the Pi through an ssh tunnel."""

    def __init__(self):
        import zmq

        self.tunnel = subprocess.Popen(["ssh", "-N", "-o", "ExitOnForwardFailure=yes", "-L",
                                        f"{TUNNEL_PORT}:127.0.0.1:5610", "pi5-eth"])
        time.sleep(1.5)
        self.zmq = zmq

    def call(self, msg):
        s = self.zmq.Context.instance().socket(self.zmq.REQ)
        s.setsockopt(self.zmq.RCVTIMEO, 3000)
        s.setsockopt(self.zmq.LINGER, 0)
        s.connect(f"tcp://127.0.0.1:{TUNNEL_PORT}")
        s.send_string(json.dumps(msg))
        r = json.loads(s.recv())
        s.close()
        return r

    def move(self, q: dict, rate: float, pump=None):
        r = self.call({"cmd": "move", "targets": q, "rate": rate})
        if r.get("error"):
            raise RuntimeError(r["error"])
        while True:
            if pump:
                pump()
            else:
                time.sleep(0.2)
            r = self.call({"cmd": "status"})
            if r["state"] == "STOPPED":
                raise RuntimeError(f"arm stopped: {r['reason']}")
            if r["state"] != "MOVING":
                return r

    def close(self):
        self.tunnel.terminate()


class Cams:
    def __init__(self):
        import zmq

        self.s = zmq.Context.instance().socket(zmq.REQ)
        self.s.setsockopt(zmq.RCVTIMEO, 5000)
        self.s.connect(f"tcp://{PI_HOST}:5601")

    def get(self):
        self.s.send(b"get")
        jl, jr, meta = self.s.recv_multipart()
        dec = lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)  # noqa: E731
        return dec(jl), dec(jr), json.loads(meta)


def body_points(q, Y, sim=False):
    """Base-frame points that must stay clear of the table: wrist, gripper link, TCP and (if known) board corners."""
    pts = [kinematics.fk(q, "wrist_link", sim)[:3, 3], kinematics.fk(q, sim=sim)[:3, 3],
           kinematics.fk(q, "gripper_frame_link", sim)[:3, 3]]
    if Y is not None:
        g = jaw(q, sim) @ Y
        s = NX * 0.018
        pts += [(g @ [x, y, 0, 1])[:3] for x in (0, s) for y in (0, s)]
    return np.array(pts)


def safe(q, Y, min_z=MIN_Z) -> bool:
    p = body_points(q, Y)
    if (p[:, 2] < min_z).any():
        return False
    low = p[:, 2] < 0.18  # near the base, stay out of the base column
    if (np.hypot(p[low, 0], p[low, 1]) < MIN_RADIUS).any():
        return False
    return KEEP_OUT_XY is None or not (np.hypot(*(p[:, :2] - KEEP_OUT_XY).T) < HEAD_KEEP_OUT).any()


def safe_path(qa, qb, Y) -> bool:
    """Table/base/head clearance along the straight joint path. A start pose already a hair under the table margin
    (measured joints sag a little from the planned ones) only has to get no lower."""
    js = [j for j in joints(qb) if j in qa]
    min_z = min(MIN_Z, body_points(qa, Y)[:, 2].min() - 0.003)
    return all(safe({j: qa[j] + s * (qb[j] - qa[j]) for j in js}, Y, min_z if s < 0.5 else MIN_Z)
               for s in np.linspace(0, 1, 12)[1:])


ARM_LINKS = ("base_link", "shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link")
ARM_RADIUS = 0.04  # [m] links as capsules between the joint frames (the wrist/gripper is beside the board)


def occluded(st, cTb, cam_T_base, q) -> bool:
    """Does the arm (base to wrist, as capsules) cover part of the board in the left image?"""
    s = NX * 0.018
    quad = (cTb @ np.array([[0, 0, 0, 1], [s, 0, 0, 1], [s, s, 0, 1], [0, s, 0, 1]], float).T)[:3].T
    uv_q = (st["K_left"] @ (quad / quad[:, 2:]).T).T[:, :2].astype(np.float32)
    z_board = quad[:, 2].min()
    origins = [(cam_T_base @ (kinematics.fk(q, link) if link != "base_link" else np.eye(4)))[:3, 3] for link in ARM_LINKS]
    for a, b in zip(origins[:-1], origins[1:]):
        for t in np.linspace(0, 1, 8):
            p = a + t * (b - a)
            if p[2] <= 0.05 or p[2] >= z_board:
                continue
            uv = st["K_left"] @ (p / p[2])
            r_px = st["K_left"][0, 0] * ARM_RADIUS / p[2]
            if cv2.pointPolygonTest(uv_q, (float(uv[0]), float(uv[1])), True) > -r_px:
                return True
    return False


def predicted_view_ok(st, board, q, X, Y, margin=40, max_angle=65):
    """Board fully inside the left image (the right camera's view is used when it sees it too), printed face toward
    the camera (< ``max_angle``; the steep head view leaves ~55-65 deg at best), 0.2-1.0 m away, not behind the arm."""
    cTb = inv(X) @ jaw(q) @ Y
    obj = board.getChessboardCorners()
    w, h = st["size"]
    for side in ("left",):
        uv, pc = project(st, cTb, obj, side)
        if (pc[:, 2] < 0.2).any() or (pc[:, 2] > 1.0).any():
            return False
        if (uv[:, 0] < margin).any() or (uv[:, 0] > w - margin).any() or (uv[:, 1] < margin).any() or (uv[:, 1] > h - margin).any():
            return False
    n, centre = cTb[:3, 2], cTb[:3, :3] @ obj.mean(0) + cTb[:3, 3]
    # board z points into the board (away from a camera that sees the printed face)
    if n @ centre <= 0 or np.degrees(np.arccos(n @ centre / np.linalg.norm(centre))) >= max_angle:
        return False
    return not occluded(st, cTb, inv(X), q)


def prior_left_cam(train_calib: str, st) -> np.ndarray:
    """Left camera in base_link if the head sat exactly at the training pose."""
    head_sim = np.array(json.loads(Path(train_calib).read_text())["head_in_base"])
    return inv(kinematics.SIM_BASE_FROM_BASE) @ head_sim @ inv(head_from_left_cam(st))


def prior_board_in_gripper() -> np.ndarray:
    """Board in gripper_link as mounted: glued parallel to the fixed jaw's outer face, printed face along -x (away from
    the moving jaw; checked on probe images), centre ~3 cm outside the jaw and ~6.5 cm down it. Only a guide for
    the first poses."""
    y = np.eye(4)
    y[:3, :3] = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float)  # columns: board x = gripper y, y = z, z = x
    y[:3, 3] = np.array([-0.03, 0.0, -0.065]) - y[:3, :3] @ [0.045, 0.045, 0]
    return y


def sweep_poses(st, X, n=3):
    """Arm poses (roll 0) with the gripper pointing down, its link in view, 6+ cm above the table; spread over the
    image. Used to find a board whose pose on the jaw is unknown (the sweep's rolls get it out from behind the arm)."""
    w, h = st["size"]
    cands = []
    for pan in range(-40, 41, 10):
        for lift in range(-20, 41, 10):
            for elbow in range(-20, 61, 10):
                for wf in range(30, 86, 10):
                    q = dict(shoulder_pan=pan, shoulder_lift=lift, elbow_flex=elbow, wrist_flex=wf, wrist_roll=0.0)
                    g = kinematics.fk(q)
                    if -g[2, 2] > -0.7 or not safe(q, None) or body_points(q, None)[:, 2].min() < 0.06:
                        continue  # gripper axis = -z of gripper_link must point down
                    p = inv(X) @ g[:, 3]
                    uv = st["K_left"] @ p[:3] / p[2]
                    if not (0.1 * w < uv[0] < 0.9 * w and 0.1 * h < uv[1] < 0.9 * h):
                        continue
                    cands.append((q, uv[:2]))
    if not cands:
        return []
    chosen = [min(range(len(cands)), key=lambda i: np.linalg.norm(cands[i][1] - [w / 2, h / 2]))]
    while len(chosen) < min(n, len(cands)):
        chosen.append(max(range(len(cands)), key=lambda i: min(np.linalg.norm(cands[i][1] - cands[j][1]) for j in chosen)))
    return [cands[i][0] for i in chosen]


def best_wrist(X, Y, q):
    """Wrist roll (deg) and, for the moving jaw, opening (%) that turn the printed face most directly toward the left
    camera. Roll and gripper are the chain's last joints: jaw = gripper_link(roll 0) . Rz(roll) [. J . Rz(open)]."""
    g0 = inv(X) @ kinematics.fk({**q, "wrist_roll": 0.0})
    tail = [(None, np.eye(4))]
    if BOARD_ON == "moving":
        j_origin = kinematics.chain(JAW_LINK["moving"])[-1][1]
        tail = [(o, j_origin @ rz4(np.radians(o * GRIPPER_DEG_PER_PCT))) for o in OPENINGS]
    best, best_cos = (0.0, None), -2.0
    for r in np.arange(PLAN_RANGE["wrist_roll"][0], PLAN_RANGE["wrist_roll"][1] + 1, 5.0):
        gr = g0 @ rz4(np.radians(r))
        for o, t in tail:
            cTb = gr @ t @ Y
            c = cTb[:3, :3] @ [0.045, 0.045, 0] + cTb[:3, 3]
            cos = cTb[:3, 2] @ c / np.linalg.norm(c)
            if cos > best_cos:
                best, best_cos = (float(r), o), cos
    return best


def rz4(a: float) -> np.ndarray:
    m = np.eye(4)
    m[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return m


def plan(st, board, X, Y, q_now, n, rng, margin=40, max_angle=65, samples=20000):
    lo = np.array([PLAN_RANGE[j][0] for j in ARM])
    hi = np.array([PLAN_RANGE[j][1] for j in ARM])
    cands = []
    for qv in rng.uniform(lo, hi, size=(samples, 5)):
        q = dict(zip(ARM, qv.tolist()))
        if BOARD_ON == "moving":
            q["gripper"] = float(rng.choice(OPENINGS))
        if rng.random() < 0.7:  # most samples: roll (and opening) facing the camera (+ jitter), the rest random
            roll, opening = best_wrist(X, Y, q)
            q["wrist_roll"] = float(np.clip(roll + rng.normal(0, 15), *PLAN_RANGE["wrist_roll"]))
            if opening is not None:
                q["gripper"] = opening
        if safe(q, Y) and predicted_view_ok(st, board, q, X, Y, margin, max_angle):
            cTb = inv(X) @ jaw(q) @ Y
            cands.append((q, Rot.from_matrix(cTb[:3, :3]), cTb[:3, 3]))
    print(f"plan: {len(cands)} feasible configurations of {samples}")
    if not cands:
        return []
    chosen = [int(rng.integers(len(cands)))]
    dist = np.full(len(cands), np.inf)
    while len(chosen) < min(n, len(cands)):  # farthest point: board orientation (30 deg ~ 8 cm) + position
        q0, r0, t0 = cands[chosen[-1]]
        for i, (_, r, t) in enumerate(cands):
            dist[i] = min(dist[i], np.degrees((r0.inv() * r).magnitude()) / 30 + np.linalg.norm(t - t0) / 0.08)
        chosen.append(int(np.argmax(dist)))
    todo, order, cur = [cands[i][0] for i in chosen], [], q_now
    mid = {**dict.fromkeys(ARM, 0.0), **({"gripper": GRIPPER_OPEN} if BOARD_ON == "moving" else {})}

    def cost(q):  # joint-space distance along a table-safe path: straight, else through the middle pose
        if safe_path(cur, q, Y):
            return max(abs(q[j] - cur[j]) for j in ARM), False
        if safe_path(cur, mid, Y) and safe_path(mid, q, Y):
            return max(abs(cur[j]) for j in ARM) + max(abs(q[j]) for j in ARM), True
        return np.inf, False

    while todo:  # nearest first
        costs = [cost(q) for q in todo]
        k = int(np.argmin([c for c, _ in costs]))
        if not np.isfinite(costs[k][0]):
            print(f"plan: {len(todo)} poses not reachable on a table-safe path; skipped")
            break
        if costs[k][1]:
            order.append({**mid, "_via": True})
        cur = todo.pop(k)
        order.append(cur)
    return order


def capture(args):
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    st, board = load_stereo(), make_board(args.square)
    det = cv2.aruco.CharucoDetector(board)
    arm, cams = Arm(), Cams()
    rng = np.random.default_rng(args.seed)
    views, k = [], 0
    win = "hand-eye (q = stop)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1296, 486)
    last = {}

    def show(l, r, cl, cr, text):
        vis = []
        for img, (c, ids) in ((l, cl), (r, cr)):
            im = img.copy()
            if len(ids):
                cv2.aruco.drawDetectedCornersCharuco(im, c.reshape(-1, 1, 2), ids.reshape(-1, 1), (0, 255, 0))
            vis.append(im)
        im = np.hstack(vis)
        cv2.putText(im, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
        cv2.imshow(win, im)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def pump():
        l, r, _ = cams.get()
        cl, cr = detect(det, l), detect(det, r)
        show(l, r, cl, cr, last.get("text", ""))

    known = {"Y": None}  # board in gripper once estimated: its corners join the table check

    def go(q, use_pump=True) -> bool:
        """Move only along a table-safe joint path: straight, else through the middle pose; else stay put."""
        s = arm.call({"cmd": "status"})
        cur = {j: s["pos"][j] for j in ARM + ("gripper",)}
        q = {"gripper": GRIPPER_OPEN, **{j: float(v) for j, v in q.items() if j in ARM + ("gripper",)}}
        mid = {**dict.fromkeys(ARM, 0.0), "gripper": GRIPPER_OPEN}
        if safe_path(cur, q, known["Y"]):
            legs = [q]
        elif safe_path(cur, mid, known["Y"]) and safe_path(mid, q, known["Y"]):
            legs = [mid, q]
        else:
            print("  no table-safe path from", {j: round(v) for j, v in cur.items()}, "to",
                  {j: round(v) for j, v in q.items()}, "- not moving", flush=True)
            return False
        for leg in legs:
            arm.move(leg, args.rate, pump if use_pump else None)
        return True

    def visit(q, kind):
        nonlocal k
        last["text"] = f"{kind} {k}: moving"
        if not go(q):
            return False
        t_end = time.time() + args.settle
        while time.time() < t_end:
            pump()
        s = arm.call({"cmd": "status"})
        l, r, meta = cams.get()
        cl, cr = detect(det, l), detect(det, r)
        name = f"pose_{k:03d}"
        cv2.imwrite(str(out / f"{name}_L.png"), l)
        cv2.imwrite(str(out / f"{name}_R.png"), r)
        qm = {j: s["pos"][j] for j in ARM + ("gripper",)}
        (out / f"{name}_meta.json").write_text(json.dumps(dict(kind=kind, q=qm, board_on=BOARD_ON,
                                                               q_target=q, stereo=meta, n_corners=[len(cl[1]), len(cr[1])])))
        last["text"] = f"{kind} {k}: L {len(cl[1])} R {len(cr[1])} corners"
        show(l, r, cl, cr, last["text"])
        print(f"{name} {kind}: corners L {len(cl[1])} R {len(cr[1])} | q " +
              " ".join(f"{s['pos'][j]:+.1f}" for j in ARM), flush=True)
        k += 1
        if max(len(cl[1]), len(cr[1])) >= 6:
            views.append(dict(name=name, q=qm, L=cl, R=cr))
            return True
        return False

    try:
        mid = dict.fromkeys(ARM, 0.0)
        print("to the middle pose ...")
        if not go(mid):
            raise RuntimeError("the arm is not on a table-safe path to the middle pose; move it there first")
        first_q = None
        global KEEP_OUT_XY
        if args.rough:  # a previous solve with the head where it is now
            J = json.loads(Path(args.rough).read_text())
            X = np.array(J["left_cam_in_base"])
            Y = np.array(J["board_in_gripper"]) if J.get("board_on", "fixed") == BOARD_ON else None
            KEEP_OUT_XY = X[:2, 3]
            print(f"rough X{', Y' if Y is not None else ''} from {args.rough}")
            if Y is None:  # sweep roll x opening at gripper-down poses until the board shows; one view gives Y
                sweep = sweep_poses(st, X)
                print("sweep poses:", [[round(p[j]) for j in ARM[:4]] for p in sweep])
                Ys = []
                for pa in sweep:
                    for opening in (OPENINGS[1], OPENINGS[-1]) if BOARD_ON == "moving" else (GRIPPER_OPEN,):
                        for roll in range(-135, 181, 45):
                            if visit({**pa, "wrist_roll": float(min(roll, PLAN_RANGE["wrist_roll"][1])),
                                      "gripper": opening}, "sweep"):
                                v = views[-1]
                                Ys.append(inv(jaw(v["q"])) @ X @ board_pose_pnp(st, board, *pnp_side(v)))
                                first_q = first_q or v["q"]
                        if len(Ys) >= 2:
                            break
                    if len(Ys) >= 2:
                        break
                if not Ys:
                    raise RuntimeError("the sweep never saw the board")
                Y = Ys[0]
                if len(views) >= 3:
                    _, Y, _, r = refine(st, board, views, X, Y, False, False)
                print(f"board on the jaw from {len(Ys)} sweep views")
        else:
            X_prior, Y_prior = prior_left_cam(args.train_calib, st), prior_board_in_gripper()
            KEEP_OUT_XY = X_prior[:2, 3]
            print("probe: poses planned from the training head pose and the board's mounting")
            for q in plan(st, board, X_prior, Y_prior, dict.fromkeys(ARM, 0.0), PROBE_N, rng, margin=80, max_angle=60,
                          samples=40000):
                if q.get("_via"):
                    continue  # go() routes through the middle pose by itself
                if visit(q, "probe") and first_q is None:
                    first_q = views[-1]["q"]
            if len(views) < 4:
                raise RuntimeError(f"board seen in only {len(views)} probe poses; check the head points at the arm")
            # 2. rough solve, 3. plan + capture
            print("rough solve:")
            rough = solve_all(st, board, views, fit_offsets_options=(False,), verbose=True)
            X, Y = rough["X"], rough["Y"]
            if rough["sim"]:  # plan with the URDF model; only a guide
                X, Y = initial_guess(st, board, views, False)
        KEEP_OUT_XY, known["Y"] = X[:2, 3], Y
        print(f"head (left camera) at {np.round(X[:3, 3], 3).tolist()} m in the base; keeping {HEAD_KEEP_OUT} m clear")
        q_now = {j: arm.call({"cmd": "status"})["pos"][j] for j in ARM + ("gripper",)}
        for q in plan(st, board, X, Y, q_now, args.n, rng):
            if not q.get("_via") and visit(q, "plan") and first_q is None:
                first_q = views[-1]["q"]
        # 4. repeatability
        if first_q is not None:
            visit(first_q, "return")
        print(f"captured {len(views)} usable views in {out}")
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        try:
            if not go(dict.fromkeys(ARM, 0.0), use_pump=False):
                print("left the arm holding where it is")
        except Exception as e:  # noqa: BLE001
            print("could not return to the middle pose:", e)
        arm.close()
        cv2.destroyAllWindows()
    args.dir = str(out)
    if len(views) >= 6:
        solve(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("capture", "solve"):
        p = sub.add_parser(name)
        p.add_argument("--square", type=float, default=18.0, help="measured square size [mm]")
        p.add_argument("--board-on", choices=("moving", "fixed"), default="moving", help="jaw the board is mounted on")
        p.add_argument("--train-calib", default=str(ROOT / "datasets/lerobot/s512_find9rc/meta/stereo_calibration.json"),
                       help="training stereo_calibration.json, for SO101_CAM_OFFSET ('' to skip)")
        if name == "capture":
            p.add_argument("--out", default=str(ROOT / "outputs/handeye"))
            p.add_argument("--n", type=int, default=24, help="planned poses after the probe")
            p.add_argument("--rate", type=float, default=10.0, help="joint speed [deg/s] (host caps at 20)")
            p.add_argument("--settle", type=float, default=1.2, help="wait after each move [s]")
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--rough", help="handeye.json of an earlier run with the head in the same place (its X; "
                                           "the board's Y too if on the same jaw)")
        else:
            p.add_argument("dir")
        p.add_argument("--no-restereo", dest="restereo", action="store_false",
                       help="keep stereo_calib.json's left-right pose instead of re-measuring it from the views")
    args = ap.parse_args()
    global BOARD_ON
    BOARD_ON = args.board_on
    if args.train_calib and not Path(args.train_calib).exists():
        args.train_calib = ""
    capture(args) if args.mode == "capture" else solve(args)


if __name__ == "__main__":
    main()
