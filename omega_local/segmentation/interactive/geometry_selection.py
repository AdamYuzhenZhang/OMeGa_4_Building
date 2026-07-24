"""Geometry-aware pixel selection helpers for the interactive editor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GrowSelection:
    mode: str
    strength: str
    seed: tuple[int, int]
    mask: np.ndarray
    parameters: dict[str, float]

    def to_json(self) -> dict[str, Any]:
        area = int(np.count_nonzero(self.mask))
        ys, xs = np.nonzero(self.mask)
        bbox = [0, 0, 0, 0]
        if area:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        return {
            "mode": self.mode,
            "strength": self.strength,
            "seed": {"x": int(self.seed[0]), "y": int(self.seed[1])},
            "areaPixels": area,
            "coverage": float(area / max(self.mask.size, 1)),
            "bbox": bbox,
            "parameters": self.parameters,
        }


_NORMAL_PRESETS: dict[str, dict[str, float]] = {
    "tight": {"angleDeg": 8.0, "edgeStopDeg": 18.0, "smoothRadius": 0.55},
    "normal": {"angleDeg": 14.0, "edgeStopDeg": 30.0, "smoothRadius": 0.85},
    "loose": {"angleDeg": 22.0, "edgeStopDeg": 44.0, "smoothRadius": 1.15},
}


def grow_normal_selection(
    path: Path,
    seed_xy: tuple[float, float],
    strength: str,
    *,
    normal_angle_deg: float | None = None,
) -> GrowSelection:
    normal, valid = _load_normal(path)
    height, width = normal.shape[:2]
    seed_x, seed_y = _nearest_valid_seed(valid, seed_xy, width, height)
    preset = _preset(_NORMAL_PRESETS, strength)
    angle_deg = _float_param(normal_angle_deg, float(preset["angleDeg"]), 2.0, 75.0)
    edge_stop_deg = min(80.0, max(angle_deg + 8.0, angle_deg * 2.0))
    smooth = _smooth_normals(normal, valid, float(preset["smoothRadius"]))
    seed_normal = _seed_normal(smooth, valid, seed_x, seed_y)

    cosine = np.sum(smooth * seed_normal.reshape(1, 1, 3), axis=-1)
    angle_threshold = np.deg2rad(angle_deg)
    edge_threshold = np.deg2rad(edge_stop_deg)
    normal_edge = _normal_edge_angle(smooth, valid)
    candidate = valid & (cosine >= np.cos(angle_threshold)) & (normal_edge <= edge_threshold)
    candidate[seed_y, seed_x] = True
    mask = _connected_seed_component(candidate, seed_x, seed_y)
    return GrowSelection(
        mode="normal-grow",
        strength=_strength_name(strength),
        seed=(seed_x, seed_y),
        mask=mask,
        parameters={
            "normalAngleDeg": float(angle_deg),
            "normalEdgeStopDeg": float(edge_stop_deg),
            "smoothRadius": float(preset["smoothRadius"]),
        },
    )


def _load_normal(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"StableNormal evidence does not exist: {path}")
    with np.load(path) as data:
        if "normal" not in data:
            raise ValueError(f"Normal evidence npz is missing 'normal': {path}")
        normal = np.asarray(data["normal"], dtype=np.float32)
        valid = np.asarray(data["valid_mask"], dtype=bool) if "valid_mask" in data else np.ones(normal.shape[:2], dtype=bool)
    if normal.ndim != 3 or normal.shape[2] != 3:
        raise ValueError(f"Normal evidence must have shape HxWx3, got {normal.shape}: {path}")
    normal = _renormalize(normal)
    valid &= np.isfinite(normal).all(axis=-1)
    return normal, valid


def _load_depth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Depth Anything evidence does not exist: {path}")
    with np.load(path) as data:
        if "depth_m" not in data:
            raise ValueError(f"Depth evidence npz is missing 'depth_m': {path}")
        depth = np.asarray(data["depth_m"], dtype=np.float32)
        valid = np.asarray(data["valid"], dtype=bool) if "valid" in data else np.ones(depth.shape, dtype=bool)
    if depth.ndim != 2:
        raise ValueError(f"Depth evidence must have shape HxW, got {depth.shape}: {path}")
    valid &= np.isfinite(depth) & (depth > 0)
    return depth, valid


def _preset(presets: dict[str, dict[str, float]], strength: str) -> dict[str, float]:
    return presets[_strength_name(strength)]


def _strength_name(value: str) -> str:
    name = str(value or "normal").strip().lower()
    return name if name in {"tight", "normal", "loose"} else "normal"


def _float_param(value: float | None, default: float, min_value: float, max_value: float) -> float:
    if value is None:
        return float(default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(np.clip(out, min_value, max_value))


def _nearest_valid_seed(
    valid: np.ndarray,
    seed_xy: tuple[float, float],
    width: int,
    height: int,
    *,
    max_radius: int = 16,
) -> tuple[int, int]:
    x = int(round(float(seed_xy[0])))
    y = int(round(float(seed_xy[1])))
    x = int(np.clip(x, 0, max(width - 1, 0)))
    y = int(np.clip(y, 0, max(height - 1, 0)))
    if valid[y, x]:
        return x, y
    for radius in range(1, max_radius + 1):
        y0 = max(0, y - radius)
        y1 = min(height, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(width, x + radius + 1)
        ys, xs = np.nonzero(valid[y0:y1, x0:x1])
        if xs.size == 0:
            continue
        xs = xs + x0
        ys = ys + y0
        best = int(np.argmin((xs - x) * (xs - x) + (ys - y) * (ys - y)))
        return int(xs[best]), int(ys[best])
    raise ValueError("Clicked seed is not near valid generated evidence.")


def _seed_normal(normal: np.ndarray, valid: np.ndarray, x: int, y: int, radius: int = 2) -> np.ndarray:
    y0 = max(0, y - radius)
    y1 = min(normal.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(normal.shape[1], x + radius + 1)
    local = normal[y0:y1, x0:x1]
    local_valid = valid[y0:y1, x0:x1]
    if not np.any(local_valid):
        return normal[y, x]
    seed = np.mean(local[local_valid], axis=0)
    norm = float(np.linalg.norm(seed))
    if not np.isfinite(norm) or norm < 1e-6:
        return normal[y, x]
    return (seed / norm).astype(np.float32, copy=False)


def _smooth_normals(normal: np.ndarray, valid: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return normal
    weight = _blur_float(valid.astype(np.float32), radius)
    channels = []
    for channel in range(3):
        blurred = _blur_float(normal[..., channel] * valid.astype(np.float32), radius)
        channels.append(blurred / np.maximum(weight, 1e-6))
    return _renormalize(np.stack(channels, axis=-1))


def _blur_float(values: np.ndarray, radius: float) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    if radius <= 0:
        return values
    try:
        import cv2  # type: ignore[import-not-found]

        kernel = max(3, int(round(float(radius) * 4.0)) | 1)
        return cv2.GaussianBlur(values, (kernel, kernel), sigmaX=float(radius), sigmaY=float(radius))
    except Exception:
        return _box_blur_float(values, radius)


def _box_blur_float(values: np.ndarray, radius: float) -> np.ndarray:
    radius_i = max(1, int(round(float(radius) * 2.0)))
    kernel = radius_i * 2 + 1
    padded = np.pad(values.astype(np.float32, copy=False), ((radius_i, radius_i), (radius_i, radius_i)), mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    summed = integral[kernel:, kernel:] - integral[:-kernel, kernel:] - integral[kernel:, :-kernel] + integral[:-kernel, :-kernel]
    return (summed / float(kernel * kernel)).astype(np.float32, copy=False)


def _renormalize(normal: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return (normal / np.maximum(norm, 1e-6)).astype(np.float32, copy=False)


def _normal_edge_angle(normal: np.ndarray, valid: np.ndarray) -> np.ndarray:
    height, width = valid.shape
    edge = np.zeros((height, width), dtype=np.float32)
    if width > 1:
        dot_x = np.sum(normal[:, 1:] * normal[:, :-1], axis=-1)
        angle_x = np.arccos(np.clip(dot_x, -1.0, 1.0)).astype(np.float32, copy=False)
        angle_x[~(valid[:, 1:] & valid[:, :-1])] = np.inf
        edge[:, 1:] = np.maximum(edge[:, 1:], angle_x)
        edge[:, :-1] = np.maximum(edge[:, :-1], angle_x)
    if height > 1:
        dot_y = np.sum(normal[1:, :] * normal[:-1, :], axis=-1)
        angle_y = np.arccos(np.clip(dot_y, -1.0, 1.0)).astype(np.float32, copy=False)
        angle_y[~(valid[1:, :] & valid[:-1, :])] = np.inf
        edge[1:, :] = np.maximum(edge[1:, :], angle_y)
        edge[:-1, :] = np.maximum(edge[:-1, :], angle_y)
    edge[~valid] = np.inf
    return edge


def _resize_rgb_to(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB image must have shape HxWx3, got {rgb.shape}")
    rgb = rgb[..., :3].astype(np.uint8, copy=False)
    if rgb.shape[1] == width and rgb.shape[0] == height:
        return rgb
    try:
        import cv2  # type: ignore[import-not-found]

        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)
    except Exception:
        from PIL import Image

        return np.asarray(Image.fromarray(rgb, mode="RGB").resize((width, height), Image.Resampling.LANCZOS))


def _connected_seed_component(candidate: np.ndarray, seed_x: int, seed_y: int) -> np.ndarray:
    if not candidate[seed_y, seed_x]:
        return np.zeros(candidate.shape, dtype=bool)
    try:
        import cv2  # type: ignore[import-not-found]

        _count, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
        component = int(labels[seed_y, seed_x])
        return labels == component
    except Exception:
        return _connected_seed_component_fallback(candidate, seed_x, seed_y)


def _connected_seed_component_fallback(candidate: np.ndarray, seed_x: int, seed_y: int) -> np.ndarray:
    height, width = candidate.shape
    out = np.zeros(candidate.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque([(seed_x, seed_y)])
    out[seed_y, seed_x] = True
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if out[ny, nx] or not candidate[ny, nx]:
                continue
            out[ny, nx] = True
            queue.append((nx, ny))
    return out
