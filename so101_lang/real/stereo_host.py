"""Stereo camera host on the Pi: both CSI cameras free-run, the newest exposure-matched pair goes out as JPEGs.

    setsid nohup python3 ~/so101_real/stereo_host.py > ~/so101_real/stereo_host.log 2>&1 < /dev/null &

Settings in ``stereo_config.json`` (which CSI port is the left camera, mode, JPEG quality). With ``sync`` on, the
libcamera software sync locks the two sensors' frame timing (camera ``left`` leads); pairs are then matched by
sensor timestamp (< half a frame apart, waiting briefly for the lagging camera). ZeroMQ on all interfaces:
REP :5601, ``get`` -> [left.jpg, right.jpg, meta.json]; ``push N`` -> N pairs, one per new left frame, on PUSH
:5602; ``stop`` exits. meta: exposure skew, frame age at send, encode time, frame counters, sensor timestamps.
Uses the system python3 (picamera2, simplejpeg, zmq from apt).
"""

import json
import os
import pathlib
import threading
import time

import simplejpeg
import zmq
from libcamera import controls
from picamera2 import Picamera2

CFG = json.loads(pathlib.Path(__file__).with_name("stereo_config.json").read_text())
W, H, FPS, Q = CFG["width"], CFG["height"], float(CFG["fps"]), CFG["jpeg_quality"]


class Cam:
    def __init__(self, idx: int, sync_role):
        self.cam = Picamera2(idx)
        ctrl = {"FrameDurationLimits": (int(1e6 / FPS),) * 2}
        if sync_role is not None:
            ctrl["SyncMode"] = sync_role
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (W, H), "format": "RGB888"}, buffer_count=4, controls=ctrl))
        self.cond, self.frame, self.ts, self.n = threading.Condition(), None, 0, 0
        self.cam.start()
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            req = self.cam.capture_request()
            arr, ts = req.make_array("main"), req.get_metadata()["SensorTimestamp"]
            req.release()
            with self.cond:
                self.frame, self.ts, self.n = arr, ts, self.n + 1
                self.cond.notify_all()

    def latest(self):
        with self.cond:
            return self.frame, self.ts, self.n


roles = (controls.rpi.SyncModeEnum.Server, controls.rpi.SyncModeEnum.Client) if CFG["sync"] else (None, None)
left, right = Cam(CFG["left_camera"], roles[0]), Cam(CFG["right_camera"], roles[1])
while left.latest()[0] is None or right.latest()[0] is None:
    time.sleep(0.01)
print("sensor mode:", left.cam.camera_configuration()["sensor"], "| left = camera", CFG["left_camera"], flush=True)


def matched(timeout=0.05):
    """Newest frames whose exposures are within half a period (waits for the lagging camera's next frame)."""
    t_end, half = time.perf_counter() + timeout, 0.5e9 / FPS
    while True:
        a, b = left.latest(), right.latest()
        if abs(a[1] - b[1]) < half or time.perf_counter() > t_end:
            return a, b
        lag = left if a[1] < b[1] else right
        with lag.cond:
            lag.cond.wait(timeout=0.005)


def pair():
    (l, tl, nl), (r, tr, nr) = matched()
    t = time.perf_counter()
    jl = simplejpeg.encode_jpeg(l, quality=Q, colorspace="BGR")  # picamera2 "RGB888" is BGR byte order
    jr = simplejpeg.encode_jpeg(r, quality=Q, colorspace="BGR")
    enc = time.perf_counter() - t
    now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    meta = {"skew_ms": (tl - tr) / 1e6, "age_ms": (now - min(tl, tr)) / 1e6, "enc_ms": enc * 1e3, "n": [nl, nr],
            "ts_ns": [tl, tr]}
    return [jl, jr, json.dumps(meta).encode()]


ctx = zmq.Context()
rep, push = ctx.socket(zmq.REP), ctx.socket(zmq.PUSH)
rep.bind("tcp://*:5601")
push.bind("tcp://*:5602")
push.setsockopt(zmq.SNDHWM, 4)
while True:
    msg = rep.recv().decode()
    if msg == "get":
        rep.send_multipart(pair())
    elif msg.startswith("push"):
        n, last = int(msg.split()[1]), None
        rep.send(b"ok")
        for _ in range(n):
            with left.cond:  # one pair per new left frame
                left.cond.wait_for(lambda: left.n != last)
                last = left.n
            push.send_multipart(pair())
    elif msg == "stop":
        rep.send(b"bye")
        time.sleep(0.2)
        os._exit(0)  # capture threads block in capture_request; skip a clean camera stop
    else:
        rep.send(b"unknown")
