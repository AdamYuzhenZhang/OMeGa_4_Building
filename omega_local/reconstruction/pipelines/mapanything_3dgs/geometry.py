"""Point-cloud IO and calibrated front-surface projection."""

from __future__ import annotations

from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement


FRONT_BAND_ABSOLUTE = 0.05
FRONT_BAND_RELATIVE = 0.01


def read_rgb_points(path: Any) -> tuple[np.ndarray, np.ndarray]:
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    names = set(vertices.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"Point-cloud PLY has no XYZ fields: {path}")
    points = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float32, copy=False)
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        raise ValueError(f"Point-cloud PLY has no finite points: {path}")
    points = points[finite]
    if {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack(
            [vertices["red"], vertices["green"], vertices["blue"]]
        )[finite]
        if np.issubdtype(colors.dtype, np.floating):
            maximum = float(np.max(colors, initial=0.0))
            if maximum <= 1.0:
                colors = colors * 255.0
        colors = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    else:
        colors = np.full((points.shape[0], 3), 180, dtype=np.uint8)
    return points, colors


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(qvec, dtype=np.float64).tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def project_visible_points(
    points: np.ndarray,
    frame: Any,
    *,
    front_band_absolute: float = FRONT_BAND_ABSOLUTE,
    front_band_relative: float = FRONT_BAND_RELATIVE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = points @ frame.rotation.T + frame.translation
    depth = camera[:, 2]
    finite = np.all(np.isfinite(camera), axis=1) & (depth > 1.0e-8)
    point_ids = np.flatnonzero(finite)
    if point_ids.size == 0:
        return _empty_projection()
    z = depth[point_ids]
    u = camera[point_ids, 0] * frame.fx / z + frame.cx
    v = camera[point_ids, 1] * frame.fy / z + frame.cy
    finite_uv = np.isfinite(u) & np.isfinite(v)
    point_ids = point_ids[finite_uv]
    z = z[finite_uv]
    xy = np.rint(np.column_stack((u[finite_uv], v[finite_uv]))).astype(
        np.int64
    )
    inside = (
        (xy[:, 0] >= 0)
        & (xy[:, 0] < frame.width)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < frame.height)
    )
    point_ids = point_ids[inside]
    z = z[inside]
    xy = xy[inside]
    if point_ids.size == 0:
        return _empty_projection()

    linear = xy[:, 1] * frame.width + xy[:, 0]
    order = np.lexsort((z, linear))
    linear_sorted = linear[order]
    z_sorted = z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(linear_sorted)) + 1]
    counts = np.diff(np.r_[starts, linear_sorted.size])
    nearest = z_sorted[starts]
    tolerance = np.maximum(
        float(front_band_absolute),
        nearest * float(front_band_relative),
    )
    visible = z_sorted <= np.repeat(nearest + tolerance, counts)
    rows = order[visible]
    return (
        point_ids[rows].astype(np.int64, copy=False),
        xy[rows].astype(np.int32, copy=False),
        z[rows].astype(np.float32, copy=False),
    )


def projected_label_map(
    xy: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint16)
    positive = labels > 0
    if not np.any(positive):
        return output
    xy = xy[positive]
    depth = depth[positive]
    labels = labels[positive]
    linear = xy[:, 1].astype(np.int64) * width + xy[:, 0]
    order = np.lexsort((depth, linear))
    sorted_linear = linear[order]
    first = np.r_[0, np.flatnonzero(np.diff(sorted_linear)) + 1]
    selected = order[first]
    output[xy[selected, 1], xy[selected, 0]] = labels[selected]
    return output


def write_rgb_ply(
    points: np.ndarray,
    colors: np.ndarray,
    destination: Any,
) -> None:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.zeros(points.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
        byte_order="<",
    ).write(destination)


def _empty_projection() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.float32),
    )
