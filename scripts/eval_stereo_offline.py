"""Score stereo-depth backends on frames dumped by check_stereo_depth.py --dump (run in ~/lerobot-env).

Reports accuracy against simulator ground truth, steady-state latency and peak GPU memory per backend.

    ~/lerobot-env/bin/python scripts/eval_stereo_offline.py --backends sgbm fs_trt ffs_trt_448
"""

import argparse
import json
import subprocess
import time

import numpy as np
from PIL import Image

from so101_lang.perception.stereo_depth import FoundationStereoDepth, StereoDepth, StereoDepthCfg, depth_to_rgb

FS = "models/foundationstereo_s/onnx/deployable_foundation_stereo_s_dynamic.onnx"
FFS = "models/fast_foundationstereo/onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx"


def backend_factory(name: str, cfg: StereoDepthCfg):
    """Names: sgbm | fs_cuda | fs_trt | ffs_{cuda,trt,trt16}_{size}[_raw]  (default normalization: imagenet)."""
    if name == "sgbm":
        return StereoDepth(cfg)
    parts = name.split("_")
    model, runtime = parts[0], parts[1]
    size = int(parts[2]) if len(parts) > 2 else 448
    normalize = "none" if name.endswith("_raw") else "imagenet"
    path = FS if model == "fs" else FFS
    return FoundationStereoDepth(
        cfg, path, size, normalize, tensorrt=runtime.startswith("trt"), fp16=runtime == "trt16"
    )


def gpu_mem_mb() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--frames", default="outputs/stereo_depth/frames.npz")
parser.add_argument("--backends", nargs="+", default=["sgbm", "fs_trt", "ffs_trt_448"])
parser.add_argument("--out", default="outputs/stereo_depth/compare.png")
parser.add_argument("--repeats", type=int, default=3, help="Timing passes over the frames after warm-up.")
args = parser.parse_args()

data = np.load(args.frames)
cfg = StereoDepthCfg.from_calibration(json.loads(str(data["calibration"])))
rows, results = {}, {}
for name in args.backends:
    base_mem = gpu_mem_mb()
    try:
        be = backend_factory(name, cfg)
        be.depth(data["left"][0], data["right"][0])  # warm-up (TensorRT engine build on first use)
    except Exception as e:  # unsupported size etc.
        print(f"{name:22s} FAILED: {str(e).splitlines()[0][:100]}")
        continue
    errs, rows[name] = [], []
    for i, (l, r, gt) in enumerate(zip(data["left"], data["right"], data["gt"])):
        est = be.depth(l, r)
        roi = np.isfinite(gt) & (gt < 1.0)
        valid = roi & np.isfinite(est)
        e = np.abs(est - gt)[valid]
        errs.append([valid.sum() / roi.sum(), 1000 * e.mean(), 1000 * np.median(e), (e < 0.01).mean(), (e < 0.02).mean()])
        if i < 3:
            rows[name].append(depth_to_rgb(est, cfg.near, cfg.far))
    t0 = time.perf_counter()
    for _ in range(args.repeats):
        for l, r in zip(data["left"], data["right"]):
            be.depth(l, r)
    ms = (time.perf_counter() - t0) / (args.repeats * len(data["left"])) * 1000
    mem = gpu_mem_mb() - base_mem
    del be
    m = np.mean(errs, 0)
    results[name] = dict(coverage=m[0], mae_mm=m[1], median_mm=m[2], within_1cm=m[3], within_2cm=m[4], ms=ms, gpu_mb=mem)
    print(f"{name:22s} median={m[2]:5.1f}mm mae={m[1]:6.1f}mm <1cm={m[3]:.2f} <2cm={m[4]:.2f} cov={m[0]:.3f}  {ms:6.1f} ms/frame  +{mem} MB GPU", flush=True)

json.dump(results, open(args.out.replace(".png", ".json"), "w"), indent=2)
cols = [np.concatenate([data["left"][i] for i in range(3)], 0), np.concatenate([depth_to_rgb(data["gt"][i], cfg.near, cfg.far) for i in range(3)], 0)]
cols += [np.concatenate(rows[n], 0) for n in rows]
Image.fromarray(np.concatenate(cols, 1)).save(args.out)
print("columns: left | GT |", " | ".join(rows), "->", args.out)
