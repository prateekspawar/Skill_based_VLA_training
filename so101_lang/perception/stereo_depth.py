"""Depth from the calibrated stereo head (numpy + OpenCV core only, usable in both venvs).

The head's cameras are parallel with identical intrinsics, so the pair is already rectified and
``depth = fx * baseline / disparity``. The same code runs on simulated and real images; on real
hardware the pair must first be rectified with the calibrated distortion/extrinsics.

Pipeline: SGBM disparity (left) + right-matcher left/right consistency check -> speckle removal ->
hole filling with the nearer-to-background neighbour along each row -> median filter -> depth [m].

For the VLA, :func:`depth_to_rgb` encodes metric depth over a fixed range as a 3-channel image so it
can be fed through a stock RGB camera slot without changing the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class StereoDepthCfg:
    fx: float
    baseline: float  # [m]
    num_disparities: int = 64  # multiple of 16; covers depth >= fx * baseline / num_disparities
    block_size: int = 5
    uniqueness_ratio: int = 10
    lr_max_diff: float = 1.0  # [px] left/right consistency tolerance
    near: float = 0.25  # depth range encoded by depth_to_rgb [m]
    far: float = 0.85

    @classmethod
    def from_calibration(cls, calib: dict, **overrides) -> "StereoDepthCfg":
        return cls(fx=float(calib["intrinsics"]["fx"]), baseline=float(calib["baseline_m"]), **overrides)


class StereoDepth:
    def __init__(self, cfg: StereoDepthCfg):
        self.cfg = cfg
        p1, p2 = 8 * 3 * cfg.block_size**2, 32 * 3 * cfg.block_size**2
        common = dict(
            blockSize=cfg.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=-1,
            uniquenessRatio=cfg.uniqueness_ratio,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        self.left_matcher = cv2.StereoSGBM_create(minDisparity=0, numDisparities=cfg.num_disparities, **common)
        # Matching right->left gives negative disparities; used only for the consistency check.
        self.right_matcher = cv2.StereoSGBM_create(
            minDisparity=-cfg.num_disparities, numDisparities=cfg.num_disparities, **common
        )

    def disparity(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Float32 disparity [px] for the left image; 0 where it could not be estimated or filled."""
        gl = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        gr = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        d_l = self.left_matcher.compute(gl, gr).astype(np.float32) / 16.0
        d_r = -self.right_matcher.compute(gr, gl).astype(np.float32) / 16.0

        h, w = d_l.shape
        valid = d_l > 0.5
        # Left/right consistency: pixel x in left maps to x - d in right, whose disparity must agree.
        xs = np.arange(w)[None, :].repeat(h, 0)
        xr = np.clip(np.round(xs - d_l).astype(int), 0, w - 1)
        d_r_at = np.take_along_axis(d_r, xr, axis=1)
        valid &= np.abs(d_l - d_r_at) <= self.cfg.lr_max_diff
        d = np.where(valid, d_l, 0.0).astype(np.float32)
        return self._fill_holes(d)

    @staticmethod
    def _fill_holes(d: np.ndarray) -> np.ndarray:
        """Fill invalid pixels with the smaller (farther) of the nearest valid disparities on the row."""
        h, w = d.shape
        valid = d > 0
        idx = np.where(valid, np.arange(w)[None, :], 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        left_fill = np.take_along_axis(d, idx, axis=1)
        left_fill[~np.maximum.accumulate(valid, axis=1)] = 0
        idx_r = np.where(valid, np.arange(w)[None, :], w - 1)
        idx_r = np.flip(np.minimum.accumulate(np.flip(idx_r, 1), axis=1), 1)
        right_fill = np.take_along_axis(d, idx_r, axis=1)
        right_fill[~np.flip(np.maximum.accumulate(np.flip(valid, 1), axis=1), 1)] = 0
        both = (left_fill > 0) & (right_fill > 0)
        fill = np.where(both, np.minimum(left_fill, right_fill), np.maximum(left_fill, right_fill))
        out = np.where(valid, d, fill).astype(np.float32)
        return cv2.medianBlur(out, 5)

    def depth(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        """Float32 depth along the optical axis [m]; NaN where unknown."""
        d = self.disparity(left_rgb, right_rgb)
        with np.errstate(divide="ignore"):
            z = self.cfg.fx * self.cfg.baseline / d
        z[d <= 0] = np.nan
        return z.astype(np.float32)

    def depth_rgb(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        return depth_to_rgb(self.depth(left_rgb, right_rgb), self.cfg.near, self.cfg.far)


def depth_to_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """Fixed-range metric depth -> uint8 RGB (TURBO colormap, near = red, far = blue, unknown = black).

    A fixed range keeps the colour <-> distance mapping identical across episodes and robots, and the
    colormap spreads depth over all three channels, which survives video compression better than grey.
    """
    norm = np.clip((depth - near) / (far - near), 0.0, 1.0)
    u8 = (255 * (1.0 - norm)).astype(np.uint8)  # near -> high value -> red in TURBO
    rgb = cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~np.isfinite(depth)] = 0
    return rgb


class FoundationStereoDepth:
    """Learned zero-shot stereo (NVIDIA FoundationStereo-small, ONNX) behind the same interface as StereoDepth.

    The dynamic ONNX export only accepts some resolutions (448x448 and 512x512 verified), and batch size 1,
    so images are resized to ``infer_size``, matched, and the disparity is scaled back to the input size.
    Model: https://huggingface.co/nvidia/c-foundationstereo-s (NVIDIA Open Model License).
    """

    def __init__(
        self,
        cfg: StereoDepthCfg,
        onnx_path: str,
        infer_size: int = 448,
        normalize: str = "none",
        tensorrt: bool = True,
        fp16: bool = False,  # FP16 engines lose accuracy on this model (64 mm vs 2.6 mm median error)
    ):
        import os

        import onnxruntime as ort

        try:  # load the CUDA/cuDNN libraries bundled with PyTorch, if present
            import torch  # noqa: F401

            ort.preload_dlls(directory="")
        except Exception:
            pass
        providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
        if tensorrt and _load_tensorrt_libs():
            shape = f"left_image:1x3x{infer_size}x{infer_size},right_image:1x3x{infer_size}x{infer_size}"
            cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(onnx_path))), "trt_cache")
            os.makedirs(cache, exist_ok=True)
            trt_opts = {
                "trt_fp16_enable": fp16,
                "trt_layer_norm_fp32_fallback": True,  # keep LayerNorm in FP32 so FP16 stays accurate
                "trt_engine_cache_enable": True,  # first build takes minutes; later sessions load it in seconds
                "trt_engine_cache_path": cache,
                "trt_profile_min_shapes": shape,
                "trt_profile_opt_shapes": shape,
                "trt_profile_max_shapes": shape,
                "trt_max_workspace_size": 4 << 30,
            }
            providers.insert(0, ("TensorrtExecutionProvider", trt_opts))
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(onnx_path, so, providers=providers)
        self.cfg, self.size, self.normalize = cfg, infer_size, normalize

    def _prep(self, rgb: np.ndarray) -> np.ndarray:
        x = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self.normalize == "imagenet":
            x = (x / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        elif self.normalize == "unit":
            x = x / 255.0
        return x.transpose(2, 0, 1)[None]

    def disparity(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        h, w = left_rgb.shape[:2]
        d = self.session.run(None, {"left_image": self._prep(left_rgb), "right_image": self._prep(right_rgb)})[0][0, 0]
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) * (w / self.size)

    def depth(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        d = self.disparity(left_rgb, right_rgb)
        with np.errstate(divide="ignore"):
            z = self.cfg.fx * self.cfg.baseline / d
        z[d <= 0.1] = np.nan
        return z.astype(np.float32)

    def depth_rgb(self, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        return depth_to_rgb(self.depth(left_rgb, right_rgb), self.cfg.near, self.cfg.far)


def _load_tensorrt_libs() -> bool:
    """Make the pip-installed TensorRT libraries visible to ONNX Runtime's TensorRT provider."""
    import ctypes
    import os

    try:
        import tensorrt_libs
    except ImportError:
        return False
    libdir = os.path.dirname(tensorrt_libs.__file__)
    try:
        for lib in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
            ctypes.CDLL(os.path.join(libdir, lib), mode=ctypes.RTLD_GLOBAL)
    except OSError:
        return False
    return True
