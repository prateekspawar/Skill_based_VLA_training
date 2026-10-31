"""End-to-end check of the arm kinematics and the hand-eye result on new poses: the forward-kinematics prediction of
the arm, drawn over live left-camera images.

    ~/lerobot-env/bin/python scripts/check_tip_overlay.py [--n 6]

Needs ``arm_host.py`` and ``stereo_host.py`` on the Pi. Visits ``--n`` gripper-down poses spread over the image
(``calibrate_handeye.sweep_poses``; wrist roll and opening turned to show the jaw's board to the camera), each along a
table-safe joint path, then returns to the middle pose (torque stays on). At each pose the measured joints, with the
zero offsets of ``real/handeye.json`` added, give the link origins, the TCP (``gripper_frame_link``) and the board's
corners. Where the board is detected, the corner error in pixels is reported with and without the offsets; these
poses were not part of the hand-eye fit. Images: ``outputs/handeye/tip_overlay/``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_handeye as ch  # noqa: E402
from so101_lang.real import kinematics  # noqa: E402

LINKS = ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link")
MESHES = ch.ROOT / "hardware/so101_gripper/sim_meshes"
MOVING_JAW_VISUAL = ((0.0, 0.0, 0.0189), (0.0, 0.0, 0.0))


def stl_vertices(path: Path, origin: np.ndarray, every: int = 7) -> np.ndarray:
    """Binary STL vertices (subsampled) in the link frame, through the URDF visual origin."""
    b = path.read_bytes()
    n = int.from_bytes(b[80:84], "little")
    v = np.frombuffer(b[84:84 + 50 * n], np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))["v"].reshape(-1, 3)
    v = v[::every].astype(float)
    return v @ origin[:3, :3].T + origin[:3, 3]


def visual_origin(xyz, rpy) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = kinematics.rpy_matrix(*rpy), xyz
    return m


JAWS = {  # link: mesh vertices in it (URDF visual origins)
    "gripper_link": stl_vertices(MESHES / "wrist_roll_follower_so101_v1.stl",
                                 visual_origin((0, -0.000218214, 0.000949706), (-3.14159, 0, 0))),
    "moving_jaw_so101_v1_link": stl_vertices(MESHES / "moving_jaw_so101_v1.stl",
                                             visual_origin(*MOVING_JAW_VISUAL)),
}


def draw(st, img, X, q, colour):
    """Arm skeleton (base to TCP) and the TCP for joint angles ``q``; returns the TCP pixel."""
    cam = ch.inv(X)
    pts = np.array([np.zeros(3)] + [kinematics.fk(q, link)[:3, 3] for link in LINKS])
    pc = (cam[:3, :3] @ pts.T).T + cam[:3, 3]
    uv, _ = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), st["K_left"], st["D_left"])
    uv = uv.reshape(-1, 2).astype(int)
    for a, b in zip(uv[:-1], uv[1:]):
        cv2.line(img, tuple(a), tuple(b), colour, 2)
    for p in uv:
        cv2.circle(img, tuple(p), 4, colour, -1)
    cv2.drawMarker(img, tuple(uv[-1]), colour, cv2.MARKER_CROSS, 24, 2)
    return uv[-1]


def draw_jaws(st, img, X, q, colour):
    """The jaw meshes (fixed jaw on gripper_link, moving jaw) as dots; ``q`` in degrees, gripper in %."""
    cam = ch.inv(X)
    qd = {**q, "gripper": q["gripper"] * ch.GRIPPER_DEG_PER_PCT}
    for link, v in JAWS.items():
        t = cam @ kinematics.fk(qd, link)
        uv, _ = cv2.projectPoints(v @ t[:3, :3].T + t[:3, 3], np.zeros(3), np.zeros(3), st["K_left"], st["D_left"])
        for u, w in uv.reshape(-1, 2).astype(int):
            if 0 <= u < img.shape[1] and 0 <= w < img.shape[0]:
                img[w, u] = colour


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--handeye", default=str(ch.ROOT / "so101_lang/real/handeye.json"))
    args = ap.parse_args()
    d = json.loads(Path(args.handeye).read_text())
    X, Y = np.array(d["left_cam_in_base"]), np.array(d["board_in_gripper"])
    offs = d["joint_zero_offsets_deg"]
    ch.BOARD_ON = d["board_on"]
    ch.KEEP_OUT_XY = X[:2, 3]
    st = ch.load_stereo()
    board = ch.make_board(d["square_mm"])
    detector = cv2.aruco.CharucoDetector(board)
    obj = board.getChessboardCorners()
    out = ch.ROOT / "outputs/handeye/tip_overlay"
    out.mkdir(parents=True, exist_ok=True)

    poses = []
    for q in ch.sweep_poses(st, X, n=args.n):
        roll, opening = ch.best_wrist(X, Y, q)
        poses.append({**q, "wrist_roll": roll, "gripper": opening if opening is not None else ch.GRIPPER_OPEN})
    print(f"{len(poses)} poses planned", flush=True)

    arm, cams = ch.Arm(), ch.Cams()

    def go(q) -> bool:
        s = arm.call({"cmd": "status"})
        cur = {j: s["pos"][j] for j in ch.ARM + ("gripper",)}
        mid = {**dict.fromkeys(ch.ARM, 0.0), "gripper": q["gripper"]}
        if ch.safe_path(cur, q, Y):
            legs = [q]
        elif ch.safe_path(cur, mid, Y) and ch.safe_path(mid, q, Y):
            legs = [mid, q]
        else:
            print("  no table-safe path - not moving", flush=True)
            return False
        for leg in legs:
            arm.move(leg, args.rate)
        return True

    errs, log = [], []
    try:
        for k, q in enumerate(poses):
            if not go(q):
                continue
            time.sleep(args.settle)
            s = arm.call({"cmd": "status"})
            meas = {j: s["pos"][j] for j in ch.ARM + ("gripper",)}
            L, _, _ = cams.get()
            img = L.copy()
            corrected = {j: meas[j] + offs.get(j, 0.0) for j in meas}
            draw(st, img, X, meas, (0, 0, 255))  # red: readings as they are
            draw(st, img, X, corrected, (255, 0, 255))  # magenta: with the zero offsets
            draw_jaws(st, img, X, corrected, (255, 255, 0))  # cyan: jaw meshes, with the offsets
            log.append({"pose": k, "q": meas})
            corners, ids = ch.detect(detector, L)
            line = f"pose {k}: " + " ".join(f"{j[:5]} {meas[j]:+.0f}" for j in meas)
            if len(ids):
                row = []
                for qq in (meas, corrected):
                    uv, _ = ch.project(st, ch.inv(X) @ ch.jaw(qq) @ Y, obj[ids], "left")
                    row.append(np.sqrt(((uv - corners) ** 2).sum(1).mean()))
                for u, v in corners.astype(int):
                    cv2.circle(img, (u, v), 3, (0, 255, 0), -1)
                uv, _ = ch.project(st, ch.inv(X) @ ch.jaw(corrected) @ Y, obj[ids], "left")
                for u, v in uv.astype(int):
                    cv2.circle(img, (u, v), 6, (255, 0, 255), 1)
                errs.append(row)
                line += f" | board {len(ids)} corners, rms {row[0]:.1f} px raw, {row[1]:.1f} px with offsets"
            else:
                line += " | board not detected"
            print(line, flush=True)
            cv2.imwrite(str(out / f"pose_{k}.jpg"), img)
    finally:
        try:
            go({**dict.fromkeys(ch.ARM, 0.0), "gripper": ch.GRIPPER_OPEN})
        finally:
            arm.close()
    if errs:
        e = np.array(errs)
        print(f"board corner rms over {len(e)} poses: median {np.median(e[:, 0]):.1f} px raw, "
              f"{np.median(e[:, 1]):.1f} px with offsets (hand-eye fit: {d['rms_px']:.1f} px)")
    (out / "poses.json").write_text(json.dumps(log, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
