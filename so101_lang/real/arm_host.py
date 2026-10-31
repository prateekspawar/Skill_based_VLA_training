"""SO-101 arm host on the Pi: owns the servo bus, holds the arm, moves it only on request (ZeroMQ REP 127.0.0.1:5610).

    setsid nohup ~/lerobot-venv/bin/python ~/so101_real/arm_host.py > ~/so101_real/arm_host.log 2>&1 < /dev/null &

Every start applies ``arm_config.json``: per-joint P/I/D gains, gripper protections, torque/speed/acceleration caps.
If torque is already on (servos holding from an earlier host), the held pose is adopted without releasing it; the
EEPROM gain registers are then written through an unlock/relock. Moves ramp the setpoints at <= ``ramp_max_deg_s``,
targets are clamped ``range_margin_deg`` inside the calibrated ranges, and a fault seen on two consecutive readings
(following error, current, temperature, supply; single corrupted bus reads do occur) FREEZES the arm at its
present pose until ``reset``. Torque goes off only on ``release``. Uses LeRobot's calibration file
(``~/.cache/huggingface/lerobot/calibration/robots/so_follower/<calibration_id>.json``): degrees, gripper 0-100.
"""

import json
import pathlib
import time

import zmq
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

CFG = json.loads(pathlib.Path(__file__).with_name("arm_config.json").read_text())
SAFE = CFG["safety"]
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
CAL_DIR = pathlib.Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
MA_PER_COUNT = 6.5  # STS3215 Present_Current unit

cal = json.loads((CAL_DIR / f"{CFG['calibration_id']}.json").read_text())
bus = FeetechMotorsBus(port=CFG["port"], motors={
    n: Motor(cal[n]["id"], "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
    for n in NAMES}, calibration={n: MotorCalibration(**cal[n]) for n in NAMES})
span = {n: (cal[n]["range_max"] - cal[n]["range_min"]) * 360 / 4096 for n in NAMES}
margin = CFG["range_margin_deg"]
limit = {n: (5.0, 95.0) if n == "gripper" else (-span[n] / 2 + margin, span[n] / 2 - margin) for n in NAMES}


def write_eeprom(motor: str, values: dict, unlock: bool):
    """EEPROM registers (gains, protections) need the lock open; with torque on, relock afterwards."""
    if unlock:
        bus.write("Lock", motor, 0)
    for reg, v in values.items():
        bus.write(reg, motor, v)
    if unlock:
        bus.write("Lock", motor, 1)


def apply_config(torque_on: bool) -> dict:
    gp = CFG["gripper_protection"]
    for n in NAMES:
        if not torque_on:
            bus.write("Operating_Mode", n, OperatingMode.POSITION.value)
        g = CFG["gains"][n]
        regs = {"P_Coefficient": g["p"], "I_Coefficient": g["i"], "D_Coefficient": g["d"]}
        if n == "gripper":
            regs |= {"Max_Torque_Limit": gp["max_torque_limit"], "Protection_Current": gp["protection_current"],
                     "Overload_Torque": gp["overload_torque"]}
        write_eeprom(n, regs, unlock=torque_on)
        # RAM registers: safe to write with torque on
        bus.write("Torque_Limit", n, CFG["torque_cap"])
        bus.write("Goal_Velocity", n, CFG["motor_speed"])
        bus.write("Acceleration", n, CFG["motor_accel"])
    gains = {n: [bus.read(r, n) for r in ("P_Coefficient", "I_Coefficient", "D_Coefficient")] for n in NAMES}
    wanted = {n: [CFG["gains"][n][k] for k in "pid"] for n in NAMES}
    if gains != wanted:
        raise RuntimeError(f"gain read-back {gains} != config {wanted}")
    return gains


def freeze() -> dict:
    raw = bus.sync_read("Present_Position", normalize=False)
    bus.sync_write("Goal_Position", raw, normalize=False)
    return bus.sync_read("Present_Position")


bus.connect(handshake=True)
torque_on = all(bus.sync_read("Torque_Enable", normalize=False).values())
if torque_on:
    gains = apply_config(torque_on=True)
    pos = freeze()
    start_note = "adopted the held pose (torque already on)"
else:
    bus.disable_torque()
    gains = apply_config(torque_on=False)
    pos = freeze()
    bus.enable_torque()
    start_note = "torque was off: configured and enabled"
print(start_note, {n: round(pos[n], 1) for n in NAMES}, "| P/I/D", gains, flush=True)

hz, rate = CFG["ramp_hz"], float(CFG["ramp_default_deg_s"])
setp, target = dict(pos), dict(pos)
state, reason, worst, tele = "HOLD", "", {n: 0.0 for n in NAMES}, {}
sock = zmq.Context().socket(zmq.REP)
sock.bind("tcp://127.0.0.1:5610")
k, running, strikes = 0, True, 0
while running:
    t = time.time()
    try:  # ----- commands -----
        while True:
            msg = json.loads(sock.recv(zmq.NOBLOCK))
            c, reply = msg.get("cmd"), {}
            if c == "move":
                bad = [j for j in msg["targets"] if j not in NAMES]
                if state == "STOPPED":
                    reply = {"error": f"STOPPED ({reason}); send reset first"}
                elif bad:
                    reply = {"error": f"unknown joints {bad}"}
                else:
                    rate = min(float(msg.get("rate", CFG["ramp_default_deg_s"])), CFG["ramp_max_deg_s"])
                    req = {**target, **{j: float(v) for j, v in msg["targets"].items()}}
                    target = {n: min(max(req[n], limit[n][0]), limit[n][1]) for n in NAMES}
                    worst, state = {n: 0.0 for n in NAMES}, "MOVING"
                    reply = {"clamped": [n for n in NAMES if target[n] != req[n]]}
            elif c in ("freeze", "reset"):
                pos = freeze()
                setp, target, state = dict(pos), dict(pos), "HOLD"
                if c == "reset":
                    reason = ""
            elif c == "release":
                bus.disable_torque()
                state, running = "RELEASED", False
            elif c != "status":
                reply = {"error": f"unknown cmd {c}"}
            sock.send_string(json.dumps({**reply, "state": state, "reason": reason, "pos": tele.get("pos"),
                                         "setpoint": setp, "target": target, "max_following": worst,
                                         "current_mA": tele.get("cur"), "temp_C": tele.get("temp"),
                                         "supply_V": tele.get("volt"), "limits": limit, "gains": gains}))
    except zmq.Again:
        pass
    if not running:
        break
    # ----- control: ramp the setpoint toward the target -----
    if state == "MOVING":
        step = rate / hz
        setp = {n: setp[n] + max(-step, min(step, target[n] - setp[n])) for n in NAMES}
        bus.sync_write("Goal_Position", setp)
    # ----- telemetry + safety, ~10 Hz -----
    if k % 5 == 0:
        try:
            p = bus.sync_read("Present_Position")
            cur = bus.sync_read("Present_Current", normalize=False)
            temp = bus.sync_read("Present_Temperature", normalize=False)
            volt = bus.sync_read("Present_Voltage", normalize=False)["shoulder_pan"] / 10
            tele = {"pos": p, "cur": {n: abs(cur[n]) * MA_PER_COUNT for n in NAMES}, "temp": temp, "volt": volt}
            if state in ("MOVING", "HOLD"):
                for n in NAMES:
                    worst[n] = max(worst[n], abs(p[n] - setp[n]))
                fault = ([f"{n}: following error {abs(p[n] - setp[n]):.1f} deg" for n in NAMES
                          if abs(p[n] - setp[n]) > SAFE["max_following_deg"]]
                         + [f"{n}: current {tele['cur'][n]:.0f} mA" for n in NAMES if tele["cur"][n] > SAFE["max_current_ma"]]
                         + [f"{n}: {temp[n]} C" for n in NAMES if temp[n] > SAFE["max_temp_c"]]
                         + ([f"supply {volt:.1f} V"] if volt < SAFE["min_supply_v"] else []))
                strikes = strikes + 1 if fault else 0
                if strikes == 1:
                    print("fault on one reading (ignored unless repeated):", "; ".join(fault), flush=True)
                if strikes >= 2:
                    pos = freeze()
                    setp, target = dict(pos), dict(pos)
                    state, reason = "STOPPED", "; ".join(fault)
                    print("STOPPED:", reason, flush=True)
                elif state == "MOVING" and all(abs(setp[n] - target[n]) < 1e-6 for n in NAMES):
                    state = "HOLD"
        except Exception as e:  # a read glitch: keep holding, report it
            print("telemetry error:", e, flush=True)
    k += 1
    time.sleep(max(0.0, 1 / hz - (time.time() - t)))
bus.port_handler.closePort()
print("released, host exit", flush=True)
