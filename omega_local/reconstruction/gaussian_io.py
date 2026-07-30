"""Shared paths and render-safe PLY transforms for 3D Gaussian models."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData, PlyElement

from omega_local.segmentation.color_palette import categorical_rgb


def trained_point_cloud_path(model_dir: Path, iterations: int) -> Path:
    return (
        model_dir
        / "point_cloud"
        / f"iteration_{int(iterations)}"
        / "point_cloud.ply"
    )


def write_segmented_gaussian_ply(
    source: Path,
    destination: Path,
    labels: np.ndarray,
) -> None:
    ply = PlyData.read(source, mmap="r")
    source_vertex = ply["vertex"].data
    names = set(source_vertex.dtype.names or ())
    required = {"f_dc_0", "f_dc_1", "f_dc_2"}
    if not required.issubset(names):
        raise ValueError(
            f"Gaussian PLY is missing spherical-harmonic color fields: {source}"
        )
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != source_vertex.shape[0]:
        raise ValueError(
            "Global Gaussian labels do not align with the source PLY: "
            f"{labels.shape[0]} labels for {source_vertex.shape[0]} vertices."
        )

    field_names = _render_fields(
        source_vertex.dtype.names or (),
        include_higher_order_sh=False,
    )
    vertex = _copy_fields(source_vertex, field_names)
    _apply_semantic_colors(vertex, labels)
    _write_vertex_ply(ply, vertex, destination)


def ensure_viewer_gaussian_ply(source: Path, destination: Path) -> None:
    if (
        destination.is_file()
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        return
    ply = PlyData.read(source, mmap="r")
    source_vertex = ply["vertex"].data
    field_names = _render_fields(
        source_vertex.dtype.names or (),
        include_higher_order_sh=True,
    )
    vertex = _copy_fields(source_vertex, field_names)
    _write_vertex_ply(ply, vertex, destination)


def write_viewer_gaussian_subset(
    source: Path,
    destination: Path,
    keep: np.ndarray,
) -> int:
    """Write a render-only subset while preserving trained GS attributes."""
    ply = PlyData.read(source, mmap="r")
    source_vertex = ply["vertex"].data
    keep = np.asarray(keep, dtype=bool).reshape(-1)
    if keep.shape[0] != source_vertex.shape[0]:
        raise ValueError(
            "Gaussian subset mask does not align with the source PLY: "
            f"{keep.shape[0]} values for {source_vertex.shape[0]} vertices."
        )
    field_names = _render_fields(
        source_vertex.dtype.names or (),
        include_higher_order_sh=True,
    )
    vertex = _copy_fields(source_vertex[keep], field_names)
    _write_vertex_ply(ply, vertex, destination)
    return int(vertex.shape[0])


def write_viewer_gaussian_partitions(
    source: Path,
    output_dir: Path,
    labels: np.ndarray,
    *,
    allowed_ids: set[int] | None = None,
    include_unassigned: bool = False,
    semantic_colors: bool = False,
) -> list[dict[str, Any]]:
    """Write disjoint per-label viewer PLYs while reading the source once."""
    ply = PlyData.read(source, mmap="r")
    source_vertex = ply["vertex"].data
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    if labels.shape[0] != source_vertex.shape[0]:
        raise ValueError(
            "Gaussian partition labels do not align with the source PLY: "
            f"{labels.shape[0]} labels for {source_vertex.shape[0]} vertices."
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    field_names = _render_fields(
        source_vertex.dtype.names or (),
        include_higher_order_sh=not semantic_colors,
    )
    instance_ids = sorted(
        int(value)
        for value in np.unique(labels)
        if value > 0 or (include_unassigned and value == 0)
    )
    if allowed_ids is not None:
        instance_ids = [value for value in instance_ids if value in allowed_ids]

    rows = []
    for instance_id in instance_ids:
        vertex = _copy_fields(
            source_vertex[labels == instance_id],
            field_names,
        )
        if semantic_colors:
            _apply_semantic_colors(
                vertex,
                np.full(vertex.shape[0], instance_id, dtype=np.int32),
            )
        output = output_dir / f"object_{instance_id:06d}.ply"
        _write_vertex_ply(ply, vertex, output)
        rows.append(
            {
                "instanceId": instance_id,
                "pointCount": int(vertex.shape[0]),
                "path": str(output),
            }
        )
    return rows


def _apply_semantic_colors(
    vertex: np.ndarray,
    labels: np.ndarray,
) -> None:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    rgb = categorical_rgb(
        labels,
        background=(180, 180, 180),
    ).astype(np.float32) / np.float32(255.0)
    active = labels > 0
    sh_c0 = np.float32(0.28209479177387814)
    dc = (rgb - np.float32(0.5)) / sh_c0
    for channel in range(3):
        vertex[f"f_dc_{channel}"] = dc[:, channel]
    if "opacity" not in (vertex.dtype.names or ()):
        return
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)
    alpha = np.float32(1.0) / (
        np.float32(1.0) + np.exp(-np.clip(opacity, -20.0, 20.0))
    )
    alpha[active] = np.maximum(alpha[active], np.float32(0.04))
    alpha = np.clip(alpha, np.float32(1e-6), np.float32(1.0 - 1e-6))
    vertex["opacity"] = np.log(alpha / (np.float32(1.0) - alpha))
    vertex["opacity"][~active] = np.float32(-12.0)


def _render_fields(
    names: tuple[str, ...],
    *,
    include_higher_order_sh: bool,
) -> list[str]:
    exact = {
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
    }
    prefixes = ["scale_", "rot_"]
    if include_higher_order_sh:
        prefixes.append("f_rest_")
    selected = [
        name
        for name in names
        if name in exact or any(name.startswith(prefix) for prefix in prefixes)
    ]
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = sorted(required.difference(selected))
    if missing:
        raise ValueError(
            "Gaussian PLY is missing required render fields: "
            + ", ".join(missing)
        )
    return selected


def _copy_fields(source: np.ndarray, names: list[str]) -> np.ndarray:
    dtype = np.dtype(
        [(name, source.dtype.fields[name][0]) for name in names]
    )
    output = np.empty(source.shape[0], dtype=dtype)
    for name in names:
        output[name] = source[name]
    return output


def _write_vertex_ply(
    source_ply: PlyData,
    vertex: np.ndarray,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = PlyData(
        [PlyElement.describe(vertex, "vertex")],
        text=source_ply.text,
        byte_order=source_ply.byte_order,
        comments=list(source_ply.comments),
        obj_info=list(source_ply.obj_info),
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    output.write(temporary)
    temporary.replace(destination)
