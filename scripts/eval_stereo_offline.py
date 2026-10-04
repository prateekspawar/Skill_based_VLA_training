"""Score stereo-depth backends on frames dumped by check_stereo_depth.py --dump (run in ~/lerobot-env).

    ~/lerobot-env/bin/python scripts/eval_stereo_offline.py --frames outputs/stereo_depth/frames.npz
"""

import argparse
import json
import time

import numpy as np
from PIL import Image

from so101_lang.perception.stereo_depth import FoundationStereoDepth, StereoDepth, StereoDepthCfg, depth_to_rgb

ONNX = "models/foundationstereo_s/onnx/deployable_foundation_stereo_s_dynamic.onnx"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", default="outputs/stereo_depth/frames.npz")
parser.add_argument("--out", default="outputs/stereo_depth/compare.png")
args = parser.parse_args()

data = np.load(args.frames)
cfg = StereoDepthCfg.from_calibration(json.loads(str(data["calibration"])))
# Built one at a time (each FoundationStereo session holds its own GPU memory).
backends = {


    "sgbm": lambda: StereoDepth(cfg),
    "fs448_cuda_fp32": lambda: FoundationStereoDepth(cfg, ONNX, 448, "imagenet", tensorrt=False),
    "fs448_trt_fp32": lambda: FoundationStereoDepth(cfg, ONNX, 448, "imagenet", tensorrt=True, fp16=False),
}
rows = {name: [] for name in backends}
for name, make in backends.items():
    be = make()
    errs, t = [], 0.0
    for i, (l, r, gt) in enumerate(zip(data["left"], data["right"], data["gt"])):
        t0 = time.time()
        est = be.depth(l, r)
        t += time.time() - t0 if i else 0.0  # skip warm-up
        roi = np.isfinite(gt) & (gt < 1.0)
        valid = roi & np.isfinite(est)
        e = np.abs(est - gt)[valid]
        errs.append([valid.sum() / roi.sum(), 1000 * e.mean(), 1000 * np.median(e), (e < 0.01).mean(), (e < 0.02).mean()])
        if i < 3:
            rows[name].append(depth_to_rgb(est, cfg.near, cfg.far))
    del be
    m = np.mean(errs, 0)
    print(f"{name:15s} coverage={m[0]:.3f} mae={m[1]:6.1f}mm median={m[2]:5.1f}mm <1cm={m[3]:.2f} <2cm={m[4]:.2f}  {t / (len(errs) - 1) * 1000:.0f} ms/frame")

cols = [np.concatenate([data["left"][i] for i in range(3)], 0), np.concatenate([depth_to_rgb(data["gt"][i], cfg.near, cfg.far) for i in range(3)], 0)]
cols += [np.concatenate(rows[n], 0) for n in backends]
Image.fromarray(np.concatenate(cols, 1)).save(args.out)
print("columns: left | GT |", " | ".join(backends), "->", args.out)
