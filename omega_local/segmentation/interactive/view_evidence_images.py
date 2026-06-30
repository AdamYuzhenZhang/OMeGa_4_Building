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
    near = np.array([246, 214, 105], dtype=np.float32)
    mid = np.array([95, 187, 198], dtype=np.float32)
    far = np.array([38, 74, 139], dtype=np.float32)
    low = t <= 0.5
    a0 = np.clip(t / 0.5, 0, 1)[..., None]
    a1 = np.clip((t - 0.5) / 0.5, 0, 1)[..., None]
    out = np.empty_like(rgb, dtype=np.float32)
    out[low] = near * (1 - a0[low]) + mid * a0[low]
    out[~low] = mid * (1 - a1[~low]) + far * a1[~low]
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
