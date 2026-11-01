"""Demo window for the real SO-101, like the sim demo's (``so101_lang/demo_ui.py``): type an instruction, the LLM
planner turns it into goals, perception finds the objects and checks which goals the table already meets.

    scripts/demo_real.sh            (starts the policy server for perception if it is not running)
    ~/lerobot-env/bin/python scripts/demo_real.py [--objects "white cube" "yellow plate"] [--execute]

Needs ``arm_host.py`` / ``stereo_host.py`` on the Pi, the policy server (OWLv2 + stereo depth; started with
``SO101_CAM_OFFSET`` and ``--calibration outputs/real/stereo_calibration_real.json``) and ``ollama serve``.

Window: the live left camera (rectified, as the models see it) with the perceived objects boxed; the objects on the
table (editable; the planner only knows these); the instruction box; the plan with each goal's status; the step
running now. "Look" moves the arm to the side view (it hides the plates from home), finds every object and returns
home. "Run" plans, looks and checks the goals. With "Execute on the arm" on, each unmet cube-on-plate goal runs
``scripts/run_real.py --cube .. --plate ..`` (pick + place policies), then the table is looked at again; a goal is
tried up to ``--tries`` times (``planner.next_goal``, as the sim demo's re-planning). Stacking is not run on the arm.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from multiprocessing.connection import Client
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_handeye as ch  # noqa: E402
import run_real as rr  # noqa: E402
from so101_lang import planner as planner_mod  # noqa: E402

CUBE_SIZE, PLATE_RADIUS = 0.03, 0.045  # [m] as so101_lang/envs/pick_place_env_cfg.py (sim) and the real cubes
VIEW_PX = 512
COLORS = {"seen": (90, 230, 90), "unseen": (255, 90, 90), "pick": (0, 220, 255), "place": (255, 150, 0)}  # RGB
RUN_REAL_ARGS = ["--track", "--min-tcp-z", "0.018", "--track-tol", "1.0", "--track-timeout", "0.6"]


def perceived_on(c, s, stack: bool):
    """Cube centre ``c`` resting on support ``s`` (base frame), as eval_policy.perceived_on; None if either unseen."""
    if c is None or s is None:
        return None
    dxy, dz = float(np.linalg.norm(c[:2] - s[:2])), float(c[2] - s[2])
    if stack:
        return dxy < 0.02 and CUBE_SIZE - 0.012 < dz < CUBE_SIZE + 0.015
    return dxy < PLATE_RADIUS and -0.025 < dz < 0.032


class RealScene:
    """Arm, cameras and perception for the window (the arm connection is opened only while moving)."""

    def __init__(self, port: int):
        self.port = port
        train = json.loads(rr.TRAIN_CALIB.read_text())
        self.head_b = np.array(train["head_in_base"], float)
        self.robot = rr.Robot(json.loads((rr.ROOT / "so101_lang/real/handeye.json").read_text()), self.head_b)
        self.rect = rr.Rectifier(json.loads((rr.ROOT / "so101_lang/real/stereo_calib.json").read_text()),
                                 train["intrinsics"])
        self.cams = ch.Cams()
        self.cam_lock = threading.Lock()
        self.conn = None

    def images(self):
        with self.cam_lock:
            L, R, _ = self.cams.get()
        return self.rect(L, R)

    def server(self, msg):
        if self.conn is None:
            self.conn = Client(("localhost", self.port), authkey=b"so101_lang")
        self.conn.send(msg)
        return self.conn.recv()

    def release_server(self):
        """The policy server takes one client at a time: free it for run_real.py."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def _go(self, arm, q_reading: dict, gripper: float, rate=10.0) -> bool:
        cur = {j: arm.call({"cmd": "status"})["pos"][j] for j in rr.ARM + ("gripper",)}
        tgt = {**{j: float(q_reading[j]) for j in rr.ARM}, "gripper": float(gripper)}
        mid = {**dict.fromkeys(rr.ARM, 0.0), "gripper": cur["gripper"]}
        if ch.safe_path(cur, tgt, None):
            legs = [tgt]
        elif ch.safe_path(cur, mid, None) and ch.safe_path(mid, tgt, None):
            legs = [mid, tgt]
        else:
            return False
        for leg in legs:
            arm.move(leg, rate)
        return True

    def look(self, names: list[str], say) -> dict:
        """Side view, locate9 of every object, back home. name -> dict(base, box, plausible); base None if unseen."""
        arm = ch.Arm()
        try:
            say("moving to the side view")
            moved = self._go(arm, self.robot.reading_q(rr.FIND_VIEW), 0.0)
            if not moved:
                say("no table-safe path to the side view: looking from here")
            time.sleep(1.0)
            rd = {j: arm.call({"cmd": "status"})["pos"][j] for j in rr.ARM + ("gripper",)}
            l, r = self.images()
            pts = self.robot.robot_points_head(rd)
            seen = {}
            for n in names:
                say(f"finding the {n}")
                rep = self.server({"cmd": "locate9", "left": l, "right": r, "query": n, "robot_points": pts})
                ok = rep.get("detect", {}).get("plausible", True) is not False
                p = np.asarray(rep["pose9"][:3])
                seen[n] = {"base": self.head_b[:3, :3] @ p + self.head_b[:3, 3] if ok else None,
                           "box": rep.get("box"), "plausible": ok, "terms": rep.get("detect", {}).get("terms")}
            if moved:
                say("going home")
                self._go(arm, self.robot.reading_q(rr.HOME), rd["gripper"])
            return seen
        finally:
            arm.close()


class DemoWindow:
    def __init__(self, args):
        self.args = args
        self.scene = RealScene(args.port)
        self.planner = planner_mod.Planner(gpu=args.planner_gpu)
        self.ui_q: queue.Queue = queue.Queue()  # callables run on the Tk thread
        self.frame = None  # latest rectified left image
        self.seen: dict = {}
        self.roles: dict[str, str] = {}
        self.busy = False

        self.root = tk.Tk()
        self.root.title("SO-101 real robot demo")
        self.root.configure(bg="#202124")
        fg, bg = "#e8e8e8", "#202124"
        left = tk.Frame(self.root, bg=bg)
        left.pack(side=tk.LEFT, padx=8, pady=8)
        tk.Label(left, text="Robot view: left head camera + perceived objects", fg=fg, bg=bg).pack(anchor="w")
        self.view = tk.Label(left, bg="black")
        self.view.pack()
        right = tk.Frame(self.root, bg=bg, width=440)
        right.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        self.phase = tk.Label(right, text="", fg="#00dcff", bg=bg, font=("Helvetica", 14), wraplength=430,
                              justify="left")
        self.phase.pack(anchor="w", pady=(0, 6))
        self.result = tk.Label(right, text="", fg=fg, bg=bg, font=("Helvetica", 15, "bold"), wraplength=430,
                               justify="left")
        self.result.pack(anchor="w")
        tk.Label(right, text="Objects on the table (comma separated; the planner knows only these):", fg=fg,
                 bg=bg).pack(anchor="w", pady=(10, 0))
        self.objects = tk.Entry(right, width=52)
        self.objects.insert(0, ", ".join(args.objects))
        self.objects.pack(anchor="w")
        self.seen_label = tk.Label(right, text="", fg="#b0b0b0", bg=bg, wraplength=430, justify="left")
        self.seen_label.pack(anchor="w")
        tk.Label(right, text="Instruction:", fg=fg, bg=bg).pack(anchor="w", pady=(10, 0))
        self.field = tk.Entry(right, width=52, font=("Helvetica", 13))
        self.field.pack(anchor="w")
        self.field.bind("<Return>", lambda _e: self.run())
        row = tk.Frame(right, bg=bg)
        row.pack(anchor="w", pady=6)
        tk.Button(row, text="Run", command=self.run).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(row, text="Look", command=self.look).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(row, text="Quit", command=self.root.destroy).pack(side=tk.LEFT)
        self.execute = tk.BooleanVar(value=args.execute)
        tk.Checkbutton(right, text="Execute on the arm (pick + place policies)", variable=self.execute, fg=fg, bg=bg,
                       selectcolor="#404040", activebackground=bg, activeforeground=fg).pack(anchor="w")
        self.plan_label = tk.Label(right, text="", fg=fg, bg=bg, font=("Helvetica", 12), wraplength=430,
                                   justify="left")
        self.plan_label.pack(anchor="w", pady=(10, 0))
        self.log = tk.Text(right, width=60, height=12, bg="#111", fg="#9a9a9a", font=("Courier", 9))
        self.log.pack(anchor="w", pady=(10, 0))
        self.set_phase("waiting for an instruction")
        threading.Thread(target=self._camera_loop, daemon=True).start()
        threading.Thread(target=lambda: self.planner.plan("put the white cube on the yellow plate", ["white cube"],
                                                          ["yellow plate"]), daemon=True).start()  # loads the LLM
        self.root.after(100, self._tick)
        if args.instruction:  # scripted start (tests): as if typed
            self.field.insert(0, args.instruction)
            self.root.after(2000, self.run)

    # ----- threads -> Tk -----
    def call(self, fn, *a):
        self.ui_q.put(lambda: fn(*a))

    def say(self, text: str):
        print(f"[demo] {text}", flush=True)
        self.call(self._log, text)
        self.call(self.set_phase, text)

    def _log(self, text):
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)

    def set_phase(self, text):
        self.phase.configure(text=f"Now: {text}")

    def _camera_loop(self):
        while True:
            try:
                self.frame = self.scene.images()[0]
            except Exception as e:  # camera host restarting: keep trying
                print(f"[demo] camera: {e}", flush=True)
                time.sleep(1.0)
            time.sleep(0.15)

    def _tick(self):
        while not self.ui_q.empty():
            self.ui_q.get_nowait()()
        if self.frame is not None:
            img = self.frame.copy()
            for name, s in self.seen.items():
                if s.get("box") is None:
                    continue
                col = COLORS[self.roles.get(name, "seen" if s["plausible"] else "unseen")]
                x0, y0, x1, y1 = (int(v) for v in s["box"])
                cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
                role = self.roles.get(name)
                tag = (f"{role}: " if role else "") + name + ("" if s["plausible"] else " (rejected)")
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                org = (min(max(x0, 2), VIEW_PX - tw - 2), max(y0 - 6, th + 4))
                cv2.rectangle(img, (org[0] - 2, org[1] - th - 3), (org[0] + tw + 2, org[1] + 4), (0, 0, 0), -1)
                cv2.putText(img, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
            self._photo = ImageTk.PhotoImage(Image.fromarray(img))
            self.view.configure(image=self._photo)
        self.root.after(120, self._tick)

    # ----- actions -----
    def object_names(self) -> list[str]:
        return [n.strip().lower() for n in self.objects.get().split(",") if n.strip()]

    def _start(self, fn):
        if self.busy:
            return
        self.busy = True

        def run():
            try:
                fn()
            except Exception as e:
                self.say(f"error: {e}")
            finally:
                self.busy = False
                self.call(self.set_phase, "waiting for an instruction")
        threading.Thread(target=run, daemon=True).start()

    def _look(self, names):
        seen = self.scene.look(names, self.say)
        self.seen = seen
        txt = "; ".join(f"{n}: {'seen at ' + ', '.join(f'{100 * v:.0f}' for v in s['base'][:2]) + ' cm' if s['base'] is not None else 'NOT seen'}"
                        for n, s in seen.items())
        self.call(self.seen_label.configure, {"text": "Perceived (base x, y): " + txt})
        if self.frame is not None:
            cv2.imwrite(str(self.out / f"look_{time.strftime('%H%M%S')}.jpg"), cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
        return seen

    def look(self):
        self._start(lambda: self._look(self.object_names()))

    def run(self):
        instr = self.field.get().strip()
        if instr:
            self._start(lambda: self._run(instr))

    def _show_plan(self, instr, goals, status, seconds):
        text = f"Instruction: {instr}\n"
        if goals:
            text += f"Plan (LLM, {seconds:.1f} s):\n" + "\n".join(
                f"  {i + 1}. {c} on {o}   [{s}]" for i, ((c, o), s) in enumerate(zip(goals, status)))
        print(f"[demo] {text.strip()}", flush=True)
        self.call(self.plan_label.configure, {"text": text})

    def _run(self, instr):
        names = self.object_names()
        cubes = [n for n in names if n.endswith("cube")]
        plates = [n for n in names if n.endswith("plate")]
        self.call(self.result.configure, {"text": ""})
        self.roles = {}
        self.say("planning with the LLM (5-30 s)")
        plan = self.planner.plan(instr, cubes, plates)
        self.call(self._log, f"plan: {plan.calls if plan.ok else plan.error} ({plan.seconds:.1f} s)")
        if not plan.ok:
            self._show_plan(instr, [], [], plan.seconds)
            self.call(self.result.configure, {"text": f"Not doing it: {plan.error}", "fg": "#ff6060"})
            return
        goals = [(c, o) for c, o in plan.goals if o in cubes + plates]
        if not goals:  # e.g. "pick up the X" (held in the gripper): no goal on the table to check
            self._show_plan(instr, [], [], plan.seconds)
            self.call(self.result.configure, {"text": f"Plan {plan.calls}: no goal on the table to check", "fg": "#e8e8e8"})
            return
        status = ["pending"] * len(goals)
        self._show_plan(instr, goals, status, plan.seconds)
        tries = [0] * len(goals)
        while True:
            seen = self._look(sorted({n for g in goals for n in g}))
            met = []
            for i, (c, o) in enumerate(goals):
                m = perceived_on(seen.get(c, {}).get("base"), seen.get(o, {}).get("base"), o.endswith("cube"))
                status[i] = "met" if m else ("object not seen" if m is None else "not met yet")
                met.append(bool(m))
            self._show_plan(instr, goals, status, plan.seconds)
            i, stop = planner_mod.next_goal(goals, met, tries, self.args.tries)
            if i is None:
                ok = stop is None
                self.call(self.result.configure, {"text": f"{'DONE' if ok else 'STOPPED'}: {sum(met)}/{len(goals)} goals met"
                                                  + ("" if ok else f" ({stop})"), "fg": "#50d050" if ok else "#ff6060"})
                return
            if not self.execute.get():
                todo = [f"{c} on {o}" for (c, o), st in zip(goals, status) if st == "not met yet"]
                unseen = sorted({n for g, st in zip(goals, status) if st == "object not seen" for n in g
                                 if seen.get(n, {}).get("base") is None})
                self.call(self.result.configure, {"text": f"Plan checked, {sum(met)}/{len(goals)} goals met"
                                                  + (f"; would run: {'; '.join(todo)}" if todo else "")
                                                  + (f"; not seen: {', '.join(unseen)}" if unseen else ""),
                                                  "fg": "#e8e8e8"})
                return
            c, o = goals[i]
            if o.endswith("cube"):
                self.call(self.result.configure, {"text": f"Stacking ({c} on {o}) is not run on the arm",
                                                  "fg": "#ff6060"})
                return
            tries[i] += 1
            status[i] = f"running (attempt {tries[i]}/{self.args.tries})"
            self.roles = {c: "pick", o: "place"}
            self._show_plan(instr, goals, status, plan.seconds)
            self.say(f"pick the {c}, place it on the {o}")
            self.scene.release_server()
            proc = subprocess.Popen([sys.executable, "-u", str(Path(__file__).parent / "run_real.py"), "--cube", c,
                                     "--plate", o, *RUN_REAL_ARGS], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, cwd=rr.ROOT)
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith(("find ", "  [", "pick verdict", "place verdict", "grasp check", "moving home",
                                    "wrote", "Traceback", "RuntimeError", "interrupted")):
                    self.call(self._log, line)
                    if line.startswith("  ["):
                        self.call(self.set_phase, f"{line.split(']')[0].strip()[1:]}: {c} -> {o}")
            proc.wait()
            self.roles = {}

    def mainloop(self):
        self.out = rr.ROOT / "outputs/real/demo"
        self.out.mkdir(parents=True, exist_ok=True)
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", nargs="+", default=["white cube", "yellow plate"])
    ap.add_argument("--execute", action="store_true", help="start with 'Execute on the arm' on")
    ap.add_argument("--tries", type=int, default=2, help="attempts per goal when executing")
    ap.add_argument("--planner-gpu", action="store_true", help="run the LLM on the GPU (default CPU, as the sim demo)")
    ap.add_argument("--port", type=int, default=6000)
    ap.add_argument("--instruction", default=None, help="run this instruction once at start")
    DemoWindow(ap.parse_args()).mainloop()


if __name__ == "__main__":
    main()
