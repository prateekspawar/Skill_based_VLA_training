"""Demo window inside the Isaac Sim app (``scripts/eval_policy.py --interactive``, launched by ``scripts/demo.sh``).

One Kit window next to the viewport: what the robot sees (left head camera, with the perceived targets drawn on
it), the instruction box, the LLM plan with each goal's status, and the step running now. The viewport camera stays
free (right mouse + WASD to fly, Alt + left mouse to orbit). Instructions come from the window or the terminal.
``~/isaacsim`` venv only (omni.ui exists only inside the running app).
"""

from __future__ import annotations

import queue
import sys
import threading

import cv2
import numpy as np

VIEW_PX = 380  # robot-view image size in the window (the camera image scaled)
COLORS = {"pick": (0, 220, 255), "place": (255, 150, 0), "seen": (230, 230, 230)}  # RGB


class DemoUI:
    def __init__(self, intrinsics: dict, baseline: float, every: int = 2):
        import omni.ui as ui

        self.ui, self.K, self.baseline, self.every = ui, intrinsics, baseline, every
        self.lines: queue.Queue[str] = queue.Queue()
        self.targets: dict[str, tuple[np.ndarray, str, float]] = {}  # label -> (head-frame point, role, radius m)
        self.instruction, self.plan_s, self.goals, self.status = "", None, [], []
        self.phase, self.result, self.n_steps = "waiting for an instruction", "", 0
        self.docked = False
        threading.Thread(target=self._read_stdin, daemon=True).start()

        # Docked right of the viewport on the first wait (floating until then); can be dragged anywhere.
        self.window = ui.Window("SO-101 demo", width=VIEW_PX + 40, height=760)
        with self.window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=5, height=0):
                    ui.Label("Robot view: left head camera + perceived targets", height=18)
                    self.provider = ui.ByteImageProvider()
                    ui.ImageWithProvider(self.provider, width=VIEW_PX, height=VIEW_PX)
                    self.phase_label = ui.Label("", word_wrap=True, height=40, style={"color": 0xFF00DCFF, "font_size": 17})
                    self.result_label = ui.Label("", word_wrap=True, height=26, style={"font_size": 19})
                    self.plan_label = ui.Label("", word_wrap=True, alignment=ui.Alignment.LEFT_TOP, height=110)
                    self.scene_label = ui.Label("", word_wrap=True, height=34, style={"color": 0xFFB0B0B0})
                    self.field = ui.StringField(height=28)
                    self.field.model.add_end_edit_fn(lambda _m: self._submit())  # Enter submits
                    with ui.HStack(height=28, spacing=6):
                        ui.Button("Run", clicked_fn=self._submit)
                        ui.Button("New scene", clicked_fn=lambda: self.lines.put(""))
                        ui.Button("Quit", clicked_fn=lambda: self.lines.put("q"))
                    ui.Label("Viewport: right mouse + WASD to fly, Alt + left mouse to orbit, F to frame the selection.",
                             word_wrap=True, height=34, style={"color": 0xFF909090})
        self._refresh_text()

    # ----- instructions -----
    def _read_stdin(self):
        for line in sys.stdin:
            self.lines.put(line.strip())
        self.lines.put("q")  # end of input

    def _submit(self):
        text = self.field.model.get_value_as_string().strip()
        if text:
            self.field.model.set_value("")  # cleared first: Run after Enter (or focus loss) must not resubmit
            self.lines.put(text)

    def _dock(self):
        """Dock right of the viewport so the panel does not cover the scene."""
        vp = self.ui.Workspace.get_window("Viewport")
        if vp is not None and not self.docked:
            self.window.dock_in(vp, self.ui.DockPosition.RIGHT, 0.34)
            self.docked = True

    def wait_instruction(self, render, scene_text: str, left_rgb=None) -> str:
        """Keep the app rendering (free viewport camera) until an instruction arrives from the window or terminal."""
        self.targets.clear()
        self.instruction, self.goals, self.status, self.result = "", [], [], ""
        self.phase = "waiting for an instruction"
        self.scene_label.text = scene_text
        self._refresh_text()
        if left_rgb is not None:
            self.draw(left_rgb)
        for _ in range(3):
            render()
        self._dock()
        while True:
            try:
                text = self.lines.get_nowait()
                break
            except queue.Empty:
                render()
        return text

    def run_rendering(self, fn, render, phase: str):
        """fn() in a thread while the app keeps rendering (the planner LLM takes 5-30 s on the CPU)."""
        self.set_phase(phase)
        out = {}

        def run():
            try:
                out["value"] = fn()
            except Exception as e:  # re-raised in the caller's thread
                out["error"] = e
        th = threading.Thread(target=run, daemon=True)
        th.start()
        while th.is_alive():
            render()
        if "error" in out:
            raise out["error"]
        return out["value"]

    # ----- state shown in the window -----
    def set_plan(self, instruction: str, goals: list[str] | None, plan_s: float | None, error: str | None = None):
        self.instruction, self.plan_s, self.goals = instruction, plan_s, goals or []
        self.status = ["pending"] * len(self.goals)
        self.result = "" if goals else f"Not doing it: {error}"
        self.phase = "planned, starting" if goals else "waiting for an instruction"
        self._refresh_text()

    def set_phase(self, text: str):
        self.phase = text
        self._refresh_text()

    def set_goal_status(self, met=None, running: int | None = None, attempt: int | None = None, tries: int = 3):
        if met is not None:
            for i, m in enumerate(met):
                if m:
                    self.status[i] = "met"
                elif self.status[i] in ("met", "running") or self.status[i].startswith("attempt"):
                    self.status[i] = "not met yet"
        if running is not None:
            self.status[running] = f"running (attempt {attempt}/{tries})"
        self._refresh_text()

    def set_result(self, ok: bool, done: int, n: int, stop: str | None):
        self.result = f"{'DONE' if ok else 'FAILED'}: {done}/{n} goals met" + (f" ({stop})" if stop and not ok else "")
        self.phase = "waiting for an instruction"
        self._refresh_text()

    def set_targets(self, targets: dict[str, tuple[np.ndarray, str, float]], replace: bool = False):
        if replace:
            self.targets.clear()
        self.targets.update(targets)

    def _refresh_text(self):
        plan = f"Instruction: {self.instruction or '-'}\n"
        if self.goals:
            plan += f"Plan (LLM{f', {self.plan_s:.1f} s' if self.plan_s is not None else ''}):\n"
            plan += "\n".join(f"  {i + 1}. {g}   [{s}]" for i, (g, s) in enumerate(zip(self.goals, self.status)))
        self.plan_label.text = plan
        self.phase_label.text = f"Now: {self.phase}"
        self.result_label.text = self.result
        self.result_label.style = {"font_size": 20, "color": 0xFF50D050 if self.result.startswith("DONE") else 0xFF5050FF}

    # ----- robot view -----
    def project(self, p_head: np.ndarray) -> tuple[float, float, float] | None:
        """Head-frame point (x forward, y left, z up) -> left-camera pixel (u, v) and depth."""
        x, y, z = float(p_head[0]), float(p_head[1]) - self.baseline / 2, float(p_head[2])
        if x <= 0.05:
            return None
        return self.K["cx"] - self.K["fx"] * y / x, self.K["cy"] - self.K["fy"] * z / x, x

    def on_step(self, get_left_rgb):
        """Called after every simulator step: redraw the robot view every few steps."""
        self.n_steps += 1
        if self.n_steps % self.every == 0:
            self.draw(get_left_rgb())

    def draw(self, left_rgb: np.ndarray):
        img = cv2.resize(left_rgb, (VIEW_PX, VIEW_PX), interpolation=cv2.INTER_AREA)
        s = VIEW_PX / self.K["width"]
        for label, (p, role, radius) in self.targets.items():
            uvd = self.project(p)
            if uvd is None:
                continue
            u, v, d = uvd
            c = (int(u * s), int(v * s))
            r = max(int(self.K["fx"] * radius / d * s), 6)
            col = COLORS.get(role, COLORS["seen"])
            cv2.circle(img, c, r, col, 2, cv2.LINE_AA)
            cv2.drawMarker(img, c, col, cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
            tag = f"{role}: {label}" if role in ("pick", "place") else label
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            org = (min(max(c[0] - tw // 2, 2), VIEW_PX - tw - 2), max(c[1] - r - 8, th + 4))
            cv2.rectangle(img, (org[0] - 2, org[1] - th - 3), (org[0] + tw + 2, org[1] + 4), (0, 0, 0), -1)
            cv2.putText(img, tag, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        cv2.rectangle(img, (0, VIEW_PX - 26), (VIEW_PX, VIEW_PX), (0, 0, 0), -1)
        cv2.putText(img, self.phase[:56], (6, VIEW_PX - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                    cv2.LINE_AA)
        rgba = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=2)
        self.provider.set_data_array(np.ascontiguousarray(rgba), [VIEW_PX, VIEW_PX])
