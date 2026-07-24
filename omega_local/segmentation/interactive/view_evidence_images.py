"""Image conversion helpers for interactive view evidence."""

from __future__ import annotations

import numpy as np
from PIL import Image


def resize_rgb(rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if rgb.shape[1] == size[0] and rgb.shape[0] == size[1]:
        return rgb.astype(np.uint8, copy=False)
    image = Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize(size, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def resize_depth(depth: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if depth.shape[1] == size[0] and depth.shape[0] == size[1]:
        return depth.astype(np.float32, copy=False)
    image = Image.fromarray(depth.astype(np.float32), mode="F").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def decode_normal_rgb(rgb: np.ndarray) -> np.ndarray:
    normal = np.asarray(rgb, dtype=np.float32) / 255.0 * 2.0 - 1.0
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return (normal / np.maximum(norm, 1e-6)).astype(np.float32, copy=False)


def normal_edge_rgb(normal: np.ndarray) -> np.ndarray:
    dx = np.zeros(normal.shape[:2], dtype=np.float32)
    dy = np.zeros(normal.shape[:2], dtype=np.float32)
    dx[:, 1:] = np.linalg.norm(normal[:, 1:] - normal[:, :-1], axis=-1)
    dy[1:, :] = np.linalg.norm(normal[1:, :] - normal[:-1, :], axis=-1)
    mag = np.sqrt(dx * dx + dy * dy)
    return scalar_gray_rgb(mag)


def depth_rgb(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    rgb = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return rgb
    lo, hi = np.percentile(depth[valid], [2.0, 98.0])
    t = np.clip((depth - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    out = color_ramp_rgb(t, _DEPTH_RAMP)
    rgb[valid] = np.clip(out[valid], 0, 255).astype(np.uint8)
    return rgb


def depth_edge_rgb(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    dx = np.zeros(depth.shape, dtype=np.float32)
    dy = np.zeros(depth.shape, dtype=np.float32)
    dx[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
    dy[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
    edges = np.sqrt(dx * dx + dy * dy)
    edges[~valid] = 0.0
    return scalar_gray_rgb(edges)


def scalar_gray_rgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros(values.shape + (3,), dtype=np.uint8)
    scale = float(np.percentile(values[finite], 98.0))
    scale = max(scale, 1e-6)
    gray = np.clip(values / scale, 0.0, 1.0)
    return np.repeat(np.rint(gray * 255.0).astype(np.uint8)[..., None], 3, axis=-1)


def color_ramp_rgb(values: np.ndarray, ramp: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    x = np.clip(values, 0.0, 1.0).reshape(-1)
    stops = ramp[:, 0]
    colors = ramp[:, 1:]
    channels = [np.interp(x, stops, colors[:, channel]) for channel in range(3)]
    return np.stack(channels, axis=-1).reshape(values.shape + (3,)).astype(np.float32, copy=False)


# Near depth is warm and far depth is cool/dark. The extra color stops make
# small depth changes easier to see than the old yellow-blue blend.
_DEPTH_RAMP = np.array(
    [
        [0.00, 255, 244, 188],
        [0.10, 255, 187,  88],
        [0.22, 231,  87,  54],
        [0.34, 184,  56, 138],
        [0.48,  92,  82, 190],
        [0.62,  43, 134, 211],
        [0.76,  46, 189, 181],
        [0.90,  40, 103, 139],
        [1.00,  24,  34,  74],
    ],
    dtype=np.float32,
)
