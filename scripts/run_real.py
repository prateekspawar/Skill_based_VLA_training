"""Run the skill models on the real SO-101: find (perception) -> pick -> place, through the policy server.

    SO101_CAM_OFFSET=<handeye.json SO101_CAM_OFFSET> ~/lerobot-env/bin/python -m so101_lang.serve.policy_server \
        --calibration outputs/real/stereo_calibration_real.json            # written by this script (--write-calib)
    ~/lerobot-env/bin/python scripts/run_real.py --cube "white cube" --plate "yellow plate" [--dry]

Needs ``arm_host.py`` and ``stereo_host.py`` on the Pi. The real robot stands in for the simulator of
``eval_policy.py``:

* images: both cameras rectified (``real/stereo_calib.json``) straight to the training lens (512 x 512, fx 781.87,
  principal point centred), RGB. The real baseline (63 mm) goes to the server's depth through
  ``--calibration``; the real head pose relative to the training head through ``SO101_CAM_OFFSET`` (virtual head).
* state: measured joints + the hand-eye zero offsets -> URDF FK (gripper link) in the sim's base body ->
  TCP (adapter offset) in the training head frame; gripper opening = joint angle / 1.745 rad.
* actions: 10-D head-frame TCP pose -> gripper-link pose in the base -> damped least-squares IK on the 5 arm joints
  (from the last solution) -> joint readings (minus the offsets) -> arm host. Paced in arm time by default
  (``--pace track``: the next action once the arm has reached this one, as the sim arm does within a step or two;
  on wall-clock pacing the 3-5x slower real arm lagged and the model closed 2.5-5 cm short and off-centre). Gripper: opening -> % (0 % = closed; 1.0 = 1.745 rad = 89.5 %).
* safety: TCP target held >= ``--min-tcp-z`` above the table and inside ``--max-reach``; any IK solution that
  leaves the joint ranges or puts gripper points below the table is skipped (arm holds); Ctrl-C freezes the arm.

Sequence (a short version of eval_policy's find pipeline, no find model or extra viewing poses yet): home pose ->
``find_start`` for the cube and the plate from the home view (OWLv2 + stereo + SAM2 trackers) -> pick (targets
refreshed by SAM2 before every query, the pick target frozen within 3 cm of the TCP) -> grasp check -> place
(release gate) -> open -> home. ``--dry``: no arm motion after reaching home; queries the models once per skill and
prints the commanded TCP path and the IK solution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_handeye as ch  # noqa: E402
from so101_lang.real import kinematics  # noqa: E402

ROOT = ch.ROOT
TRAIN_CALIB = ROOT / "datasets/lerobot/s512_find9rc/meta/stereo_calibration.json"
REAL_CALIB_OUT = ROOT / "outputs/real/stereo_calibration_real.json"
TCP_OFFSET = np.array([0.009, 0.0, -0.090])  # SO101Adapter.tcp_offset (gripper link frame)
GRIPPER_OPEN_RAD = 1.745
PCT_PER_RAD = np.degrees(1.0) / ch.GRIPPER_DEG_PER_PCT  # real gripper % per rad of jaw angle
ARM = kinematics.ARM
# Sim episode start (Isaac Lab SO-101 stack config), degrees; gripper closed as pick starts in training.
HOME = dict(shoulder_pan=0.0, shoulder_lift=np.degrees(-0.6), elbow_flex=np.degrees(0.8),
            wrist_flex=np.degrees(0.6), wrist_roll=0.0)
# Clear view for find: arm swung to the side, out of the head's view of the workspace (home hides the plate).
FIND_VIEW = dict(shoulder_pan=60.0, shoulder_lift=-10.0, elbow_flex=-20.0, wrist_flex=20.0, wrist_roll=0.0)
STEP_DT = 0.05
EXEC_STEPS = 10
PICK_TEXT, PLACE_TEXT = "pick up the {}", "place it on the {}"


# ------------------------------------------------------------------------------------------------ geometry
def real_calibration_file(train: dict, st_json: dict) -> dict:
    """The training calibration with the real baseline (rectified pair at the training intrinsics)."""
    b = float(np.linalg.norm(np.array(st_json["T"], float)))
    c = json.loads(json.dumps(train))
    c["baseline_m"] = b
    c["cameras"]["left"]["position_in_head"] = [0.0, b / 2, 0.0]
    c["cameras"]["right"]["position_in_head"] = [0.0, -b / 2, 0.0]
    c["note"] = "real stereo head: rectified to the training intrinsics, real baseline; head_in_base = training head"
    return c


class Rectifier:
    """Raw 1296 x 972 pair -> rectified 512 x 512 RGB pair with the training intrinsics."""

    def __init__(self, st_json: dict, intr: dict):
        K_new = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
        size = (intr["width"], intr["height"])
        self.maps = {}
        for side, R in (("left", "R1"), ("right", "R2")):
            K, D = np.array(st_json[side]["K"], float), np.array(st_json[side]["D"], float).ravel()
            self.maps[side] = cv2.initUndistortRectifyMap(K, D, np.array(st_json[R], float), K_new, size, cv2.CV_32FC1)

    def __call__(self, L, R):
        out = [cv2.remap(img, *self.maps[s], cv2.INTER_LINEAR) for img, s in ((L, "left"), (R, "right"))]
        return [cv2.cvtColor(o, cv2.COLOR_BGR2RGB) for o in out]


class Robot:
    def __init__(self, handeye: dict, head_b: np.ndarray):
        self.offs = handeye["joint_zero_offsets_deg"]
        self.head_b, self.base_in_head = head_b, np.linalg.inv(head_b)
        self.S = kinematics.SIM_BASE_FROM_BASE  # URDF base_link in the sim's base body
        self.S_inv = np.linalg.inv(self.S)
        cal = ch._CAL
        span = {n: (cal[n]["range_max"] - cal[n]["range_min"]) * 360 / 4096 for n in ARM}
        host = json.loads((ROOT / "so101_lang/real/arm_config.json").read_text())
        m = {n: host.get("range_margin_override_deg", {}).get(n, host["range_margin_deg"]) for n in ARM}
        self.qlim = {n: (-span[n] / 2 + m[n], span[n] / 2 - m[n]) for n in ARM}  # the host's clamp

    def true_q(self, reading: dict) -> dict:
        return {j: reading[j] + self.offs.get(j, 0.0) for j in ARM}

    def reading_q(self, q_true: dict) -> dict:
        return {j: q_true[j] - self.offs.get(j, 0.0) for j in ARM}

    def link_b(self, q_true: dict) -> np.ndarray:
        """Gripper link in the sim base body."""
        return self.S @ kinematics.fk(q_true)

    def state(self, reading: dict) -> np.ndarray:
        g = self.link_b(self.true_q(reading))
        tcp_b = g[:3, 3] + g[:3, :3] @ TCP_OFFSET
        pos_h = self.base_in_head[:3, :3] @ tcp_b + self.base_in_head[:3, 3]
        rot_h = self.base_in_head[:3, :3] @ g[:3, :3]
        opening = np.clip(np.radians(reading["gripper"] * ch.GRIPPER_DEG_PER_PCT) / GRIPPER_OPEN_RAD, 0, 1)
        return np.r_[pos_h, rot_h[:, 0], rot_h[:, 1], opening].astype(np.float32)

    def tcp_b(self, reading: dict) -> np.ndarray:
        g = self.link_b(self.true_q(reading))
        return g[:3, 3] + g[:3, :3] @ TCP_OFFSET

    def action_to_link_b(self, a: np.ndarray, min_tcp_z: float, max_reach: float):
        """10-D head-frame action -> desired gripper-link pose (sim base) with the TCP kept in the safe box, and the
        gripper % command."""
        pos_b = self.head_b[:3, :3] @ a[:3] + self.head_b[:3, 3]
        c1 = a[3:6] / np.linalg.norm(a[3:6])
        c2 = a[6:9] - (a[6:9] @ c1) * c1
        c2 /= np.linalg.norm(c2)
        rot_b = self.head_b[:3, :3] @ np.c_[c1, c2, np.cross(c1, c2)]
        tbl = self.S[2, 3]  # table = base_link z 0 (base sits on it) -> sim base z
        pos_b[2] = max(pos_b[2], tbl + min_tcp_z)
        r = np.hypot(*pos_b[:2])
        if r > max_reach:
            pos_b[:2] *= max_reach / r
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = rot_b, pos_b - rot_b @ TCP_OFFSET
        pct = float(np.clip(a[9], 0, 1)) * GRIPPER_OPEN_RAD * PCT_PER_RAD
        return t, pct

    def ik(self, t_des_b: np.ndarray, q0: dict, iters: int = 40, rot_w: float = 0.3, lam: float = 0.02):
        """Damped least squares on the 5 arm joints (degrees, true angles) for the gripper link pose (sim base; the
        position error is taken at the TCP),
        clamped to the host's joint ranges (the real wrist_flex stops short of the sim's: at the far reach a
        vertical gripper needs ~100 deg, the host allows 86; the gripper then tilts and the position still holds)."""
        lo = np.array([self.qlim[j][0] + self.offs.get(j, 0.0) + 0.5 for j in ARM])
        hi = np.array([self.qlim[j][1] + self.offs.get(j, 0.0) - 0.5 for j in ARM])
        q = np.clip(np.array([q0[j] for j in ARM], float), lo, hi)
        target = self.S_inv @ t_des_b

        tcp_target = target[:3, 3] + target[:3, :3] @ TCP_OFFSET  # match the grasp point, not the link origin

        def err(qv):
            f = kinematics.fk(dict(zip(ARM, qv)))
            e_p = tcp_target - (f[:3, 3] + f[:3, :3] @ TCP_OFFSET)
            r = target[:3, :3] @ f[:3, :3].T
            e_r = 0.5 * np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
            return np.r_[e_p, rot_w * e_r]

        for _ in range(iters):
            e = err(q)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 2e-3:
                break
            J = np.zeros((6, 5))
            for i in range(5):
                dq = np.zeros(5)
                dq[i] = 0.05
                J[:, i] = (e - err(q + dq)) / np.radians(0.05)
            free = np.ones(5, bool)
            for _ in range(5):  # active set: a joint at its limit that the step pushes further is held there
                Jf = J * free
                step = Jf.T @ np.linalg.solve(Jf @ Jf.T + lam ** 2 * np.eye(6), e)
                pinned = free & (((q <= lo + 1e-6) & (step < 0)) | ((q >= hi - 1e-6) & (step > 0)))
                if not pinned.any():
                    break
                free &= ~pinned
            q = np.clip(q + np.degrees(np.clip(step, -0.2, 0.2)), lo, hi)
        e = err(q)
        if np.linalg.norm(e[:3]) > 3e-3 and rot_w > 0.05:  # at a joint limit: keep the position, give up orientation
            return self.ik(t_des_b, dict(zip(ARM, q)), iters, rot_w=0.03, lam=lam)
        return dict(zip(ARM, q)), float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]) / rot_w)

    def ok(self, q_true: dict) -> bool:
        r = self.reading_q(q_true)
        if any(not (self.qlim[j][0] <= r[j] <= self.qlim[j][1]) for j in ARM):
            return False
        pts = np.array([kinematics.fk(q_true, link)[:3, 3] for link in ("wrist_link", "gripper_link", "gripper_frame_link")]
                       + [kinematics.fk({**q_true, "gripper": 0.0}, "moving_jaw_so101_v1_link")[:3, 3]])
        return bool((pts[:, 2] > 0.0).all() and (np.hypot(pts[:, 0], pts[:, 1])[pts[:, 2] < 0.12] > 0.10).all())

    def robot_points_head(self, reading: dict, step: float = 0.01) -> np.ndarray:
        """Keypoints along the chain (as eval_policy.robot_points_head) in the training head frame."""
        q = self.true_q(reading)
        qd = {**q, "gripper": reading["gripper"] * ch.GRIPPER_DEG_PER_PCT}
        nodes = [np.zeros(3)] + [kinematics.fk(q, l)[:3, 3] for l in
                                 ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link")]
        g = kinematics.fk(q)
        tcp = g[:3, 3] + g[:3, :3] @ TCP_OFFSET
        jaw = kinematics.fk(qd, "moving_jaw_so101_v1_link")[:3, 3]
        nodes.append(tcp)
        segs = list(zip(nodes[:-1], nodes[1:])) + [(nodes[-2], jaw), (jaw, tcp)]
        pts = []
        for a, b in segs:
            n = max(int(np.linalg.norm(b - a) / step), 1)
            pts += [a + (b - a) * k / n for k in range(n + 1)]
        p_b = (self.S[:3, :3] @ np.array(pts).T).T + self.S[:3, 3]
        return (self.base_in_head[:3, :3] @ p_b.T).T + self.base_in_head[:3, 3]


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", default="white cube")
    ap.add_argument("--plate", default="yellow plate")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--write-calib", action="store_true", help="only write the server's calibration file")
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--rate", type=float, default=20.0, help="[deg/s] host ramp cap for policy moves")
    ap.add_argument("--min-tcp-z", type=float, default=0.012, help="[m] TCP above the table")
    ap.add_argument("--max-reach", type=float, default=0.40)
    ap.add_argument("--timeout", type=float, default=16.0, help="[s] of sim time (executed steps x 50 ms) per skill")
    ap.add_argument("--pace", choices=["track", "clock"], default="track",
                    help="track: each action waits until the arm is within --track-tol of it (the models were trained "
                         "on a sim arm that tracks its commands within a step or two; the real arm is 3-5x slower); "
                         "clock: one action per 50 ms regardless")
    ap.add_argument("--track-tol", type=float, default=0.5, help="[deg]")
    ap.add_argument("--track-timeout", type=float, default=1.0, help="[s] per action")
    ap.add_argument("--integral", type=float, default=0.5,
                    help="gain of the per-joint integral correction (bias += gain x remaining error after each action, "
                         "+-4 deg): the servos stop ~1-3 deg short, which left the TCP ~2 mm short per chunk toward "
                         "the fixed jaw; the model corrects back only slowly, so it summed to 1.5-2.5 cm at the close")
    ap.add_argument("--freeze-radius", type=float, default=0.03)
    ap.add_argument("--release-gate", type=float, nargs=2, default=(0.04, 0.065))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hover", type=float, default=None, metavar="Z",
                    help="[m] diagnostics: TCP floor at Z (no contact with a 3 cm cube from ~0.05), pick only, "
                         "observations saved")
    ap.add_argument("--save-obs", action="store_true", help="save every query (images, state, target, actions)")
    ap.add_argument("--track", action="store_true", help="SAM2 target updates during the skills (default: the targets "
                    "found from the clear view stay fixed)")
    args = ap.parse_args()
    if args.hover is not None:
        args.min_tcp_z, args.save_obs = args.hover, True

    train = json.loads(TRAIN_CALIB.read_text())
    st_json = json.loads((ROOT / "so101_lang/real/stereo_calib.json").read_text())
    REAL_CALIB_OUT.parent.mkdir(parents=True, exist_ok=True)
    REAL_CALIB_OUT.write_text(json.dumps(real_calibration_file(train, st_json), indent=1))
    if args.write_calib:
        print("wrote", REAL_CALIB_OUT)
        return
    handeye = json.loads((ROOT / "so101_lang/real/handeye.json").read_text())
    head_b = np.array(train["head_in_base"], float)
    robot = Robot(handeye, head_b)
    rect = Rectifier(st_json, train["intrinsics"])
    out = ROOT / "outputs/real" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    log = open(out / "log.jsonl", "w")
    arm, cams = ch.Arm(), ch.Cams()
    conn = Client(("localhost", args.port), authkey=b"so101_lang")
    frames = []

    def status():
        return arm.call({"cmd": "status"})

    def reading():
        return {j: status()["pos"][j] for j in ARM + ("gripper",)}

    def images():
        L, R, _ = cams.get()
        l, r = rect(L, R)
        frames.append(l)
        return l, r

    def go_joints(q_reading: dict, gripper: float, rate=10.0):
        cur = reading()
        tgt = {**{j: float(q_reading[j]) for j in ARM}, "gripper": float(gripper)}
        mid = {**dict.fromkeys(ARM, 0.0), "gripper": cur["gripper"]}
        path = [tgt] if ch.safe_path(cur, tgt, None) else ([mid, tgt] if ch.safe_path(cur, mid, None) and ch.safe_path(mid, tgt, None) else None)
        if path is None:
            raise RuntimeError(f"no table-safe path to {tgt}")
        for leg in path:
            arm.move(leg, rate)

    def server(msg):
        conn.send(msg)
        return conn.recv()

    def run_skill(skill, text, perceived, rot6d, queries):
        server({"cmd": "reset", "seed": args.seed, "skill": skill})
        q_seed = robot.true_q(reading())
        t0, n, verdict_run, result, released = time.time(), 0, 0, None, False
        bias = dict.fromkeys(ARM, 0.0)
        while n < args.timeout / STEP_DT and result is None:  # timeout in sim time (executed steps)
            rd = reading()
            tcp_h = robot.state(rd)[:3]
            l, r = images()
            frozen = [queries[sk] for sk, rad in (("pick", args.freeze_radius), ("place", 0.06))
                      if np.linalg.norm(tcp_h - perceived[sk]) < rad or (sk == "pick" and skill == "place")
                      or sk == "place"]  # the plate stays as found from the clear view (the arm hides it from home)
            if args.track:
                rep = server({"cmd": "sam2_update", "left": l, "right": r, "robot_points": robot.robot_points_head(rd),
                              "frozen": frozen})
                for sk, q in queries.items():
                    if q in rep["points"]:
                        perceived[sk] = np.asarray(rep["points"][q])[:3]
            target = np.r_[perceived[skill], rot6d[skill]].astype(np.float32)
            st = robot.state(rd)
            rep = server({"left": l, "right": r, "state": st, "task": text, "target": target})
            acts = np.asarray(rep["actions"])
            if args.save_obs:
                k_obs = len(list(out.glob(f"obs_{skill}_*")))
                np.savez_compressed(out / f"obs_{skill}_{k_obs:03d}.npz", left=l, right=r,
                                    state=st, target=target, actions=acts, t=time.time() - t0, reading=json.dumps(rd))
            log.write(json.dumps({"skill": skill, "t": time.time() - t0, "state": st.tolist(), "target": target.tolist(),
                                  "actions": acts[:EXEC_STEPS].tolist(), "reading": rd}) + "\n")
            if args.dry:
                print(f"[{skill}] TCP {np.round(st[:3], 3)} -> chunk end {np.round(acts[EXEC_STEPS - 1, :3], 3)}; grip "
                      f"{st[9]:.2f} -> {acts[EXEC_STEPS - 1, 9]:.2f}; done/success {acts[0, 10]:.2f}/{acts[0, 11]:.2f}")
                for a in acts[:EXEC_STEPS]:
                    t_des, pct = robot.action_to_link_b(a, args.min_tcp_z, args.max_reach)
                    q_seed, ep, er = robot.ik(t_des, q_seed)
                print("   IK last:", {j: round(v, 1) for j, v in robot.reading_q(q_seed).items()},
                      f"gripper {pct:.0f}% | err {1000 * ep:.1f} mm {np.degrees(er):.1f} deg | safe {robot.ok(q_seed)}")
                return None
            for k, a in enumerate(acts[:EXEC_STEPS]):
                t_step = time.time()
                if k == 0 and a[10] > 0.5:  # debounced verdict, as eval_policy (first action of each chunk)
                    if a[11] > 0.5:
                        result = True
                        break
                    verdict_run += 1
                    if verdict_run >= 2:
                        result = False
                        break
                elif k == 0:
                    verdict_run = 0
                a = a.copy()
                if skill == "place" and a[9] > 0.05 and not released:  # release gate
                    d = robot.tcp_b(reading()) - (head_b[:3, :3] @ perceived["place"] + head_b[:3, 3])
                    if np.hypot(*d[:2]) > args.release_gate[0] or d[2] > args.release_gate[1]:
                        a[9] = 0.0
                    else:
                        released = True
                t_des, pct = robot.action_to_link_b(a, args.min_tcp_z, args.max_reach)
                q_sol, ep, er = robot.ik(t_des, q_seed)
                if not robot.ok(q_sol) or ep > 0.02:
                    print(f"  [{skill}] skipped action (IK err {1000 * ep:.0f} mm, safe {robot.ok(q_sol)})", flush=True)
                    continue
                q_seed = q_sol
                goal = robot.reading_q(q_sol)
                cmd = {j: goal[j] + bias[j] for j in ARM}  # integral correction of the servos' steady-state error
                r_ = arm.call({"cmd": "move", "targets": {**cmd, "gripper": pct}, "rate": args.rate})
                if r_.get("error") or r_["state"] == "STOPPED":
                    raise RuntimeError(f"arm: {r_.get('error') or r_['reason']}")
                n += 1
                if args.pace == "track":  # arm time: next action once the arm has (nearly) reached this one
                    while time.time() - t_step < args.track_timeout:
                        pos = status()["pos"]
                        if max(abs(pos[j] - goal[j]) for j in ARM) < args.track_tol:
                            break
                        time.sleep(0.02)
                    if args.integral > 0:
                        pos = status()["pos"]
                        for j in ARM:
                            bias[j] = float(np.clip(bias[j] + args.integral * (goal[j] - pos[j]), -4.0, 4.0))
                time.sleep(max(0.0, STEP_DT - (time.time() - t_step)))
            print(f"  [{skill}] t {time.time() - t0:4.1f}s TCP {np.round(robot.state(reading())[:3], 3)} target "
                  f"{np.round(perceived[skill], 3)} grip {robot.state(reading())[9]:.2f} done/success "
                  f"{acts[0, 10]:.2f}/{acts[0, 11]:.2f}", flush=True)
        arm.move({}, args.rate)  # wait for the ramp to finish
        return result

    try:
        home_r = robot.reading_q(HOME)
        print("find view:", FIND_VIEW, flush=True)
        go_joints(robot.reading_q(FIND_VIEW), 0.0)
        time.sleep(1.0)
        rd = reading()
        l, r = images()
        cv2.imwrite(str(out / "find_left.jpg"), cv2.cvtColor(l, cv2.COLOR_RGB2BGR))
        queries = {"pick": args.cube, "place": args.plate}
        server({"cmd": "sam2_reset"})
        poses = {}
        for sk, q in queries.items():
            rep = server({"cmd": "find_start" if args.track else "locate9", "left": l, "right": r, "query": q,
                          "robot_points": robot.robot_points_head(rd)})
            poses[sk] = np.asarray(rep["pose9"])
            p_bl = robot.S_inv @ np.r_[head_b[:3, :3] @ poses[sk][:3] + head_b[:3, 3], 1]
            print(f"find '{q}': base_link {np.round(p_bl[:3], 3)} | plausible {rep.get('detect', {}).get('plausible')} "
                  f"| detect {rep.get('detect', {}).get('terms')}", flush=True)
        print("moving home:", {j: round(v) for j, v in home_r.items()}, flush=True)
        go_joints(home_r, 0.0)
        time.sleep(0.5)
        perceived = {sk: p[:3].copy() for sk, p in poses.items()}
        rot6d = {sk: p[3:] for sk, p in poses.items()}
        res = run_skill("pick", PICK_TEXT.format(args.cube), perceived, rot6d, queries)
        print("pick verdict", res, "| gripper opening", round(float(robot.state(reading())[9]), 3), flush=True)
        if args.hover is not None:
            pass
        elif not args.dry:
            if robot.state(reading())[9] < 0.05:
                print("grasp check failed: gripper closed on nothing", flush=True)
            else:
                res = run_skill("place", PLACE_TEXT.format(args.plate), perceived, rot6d, queries)
                print("place verdict", res, flush=True)
            rd = reading()
            arm.move({**{j: rd[j] for j in ARM}, "gripper": 60.0}, 30)
        else:
            run_skill("place", PLACE_TEXT.format(args.plate), perceived, rot6d, queries)
    except KeyboardInterrupt:
        print("interrupted: freezing", flush=True)
        arm.call({"cmd": "freeze"})
    finally:
        try:
            if not args.dry:
                go_joints(home_r, reading()["gripper"])
        finally:
            arm.close()
            log.close()
            if frames:
                import imageio

                imageio.mimsave(out / "left.mp4", frames, fps=5)
            print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
