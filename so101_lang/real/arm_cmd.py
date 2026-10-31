"""Client for arm_host.py (run on the Pi).

    arm_cmd.py status | freeze | reset | release | move <joint>=<deg> ... [rate=10] [wait]
"""

import json
import sys
import time

import zmq


def call(msg: dict) -> dict:
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 3000)
    s.setsockopt(zmq.LINGER, 0)
    s.connect("tcp://127.0.0.1:5610")
    s.send_string(json.dumps(msg))
    return json.loads(s.recv())


def show(r: dict):
    if r.get("error"):
        print("ERROR:", r["error"])
    print(f"state {r['state']}" + (f" ({r['reason']})" if r["reason"] else "") + f" | supply {r['supply_V']} V")
    for n in r["target"]:
        p, cur, tc = (r["pos"] or {}).get(n), (r["current_mA"] or {}).get(n), (r["temp_C"] or {}).get(n)
        unit = "%" if n == "gripper" else "°"
        print(f"  {n:14s} pos {p:+7.1f}{unit}  target {r['target'][n]:+7.1f}{unit}  max following "
              f"{r['max_following'][n]:4.1f}  {cur:4.0f} mA  {tc} C  P/I/D {r['gains'][n]}")
    if r.get("clamped"):
        print("  clamped to the safe range:", r["clamped"])


args = sys.argv[1:]
cmd = args[0] if args else "status"
if cmd == "move":
    kv = dict(a.split("=") for a in args[1:] if "=" in a)
    rate = float(kv.pop("rate", 10))
    r = call({"cmd": "move", "targets": {j: float(v) for j, v in kv.items()}, "rate": rate})
    if "wait" in args and not r.get("error"):
        t_end = time.time() + 60
        while time.time() < t_end:
            time.sleep(0.3)
            r = call({"cmd": "status"})
            if r["state"] != "MOVING":
                time.sleep(0.6)  # settle
                r = call({"cmd": "status"})
                break
else:
    r = call({"cmd": cmd})
show(r)
