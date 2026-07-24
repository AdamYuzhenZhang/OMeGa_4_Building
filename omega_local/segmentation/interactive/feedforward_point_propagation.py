"""Persistent-region transfer through an aligned feed-forward point cloud."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .colmap_track_propagation import _rasterize_exclusive_points
from .omega_mesh_point_cloud import (
    OmegaMeshHybridPointCloudConfig,
    OmegaMeshPointCloudConfig,
    ensure_omega_mesh_hybrid_point_cloud,
    ensure_omega_mesh_point_cloud,
)
from .regions import persistent_region_color_map
from .sam2_video_propagation import LabeledPropagationSource, PropagationFrame


@dataclass(frozen=True)
class FeedForwardPointConfig:
    points_path: Path
    source_name: str = "OMeGa Initializer Points"
    init_mesh_path: Path | None = None
    mesh_point_config: OmegaMeshPointCloudConfig | OmegaMeshHybridPointCloudConfig | None = None
    segmented_cache_path: Path | None = None
    segmented_ply_path: Path | None = None
    segmented_summary_path: Path | None = None
    visibility_depth_band_m: float = 0.05
    visibility_depth_band_relative: float = 0.01
    raster_radius: int = 1


class FeedForwardPointPropagationSession:
    """Labels an arbitrary aligned cloud by visible complete-frame evidence."""

    def __init__(self, config: FeedForwardPointConfig) -> None:
        self.config = config
        self._run_lock = threading.Lock()

    def propagate_labeled_sources(
        self,
        *,
        frames: list[PropagationFrame],
        sources: list[LabeledPropagationSource],
        start_local_index: int,
        region_rows: list[dict[str, Any]],
        progress_callback: Callable[[int, int, int], None] | None = None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None = None,
    ) -> list[dict[str, Any]]:
        del start_local_index
        if not frames:
            raise ValueError("Feed-forward point propagation needs at least one frame.")
        if not sources:
            raise ValueError("Feed-forward point propagation needs at least one complete frame.")

        with self._run_lock:
            if self.config.mesh_point_config is not None:
                _ensure_mesh_point_cloud(self.config.mesh_point_config)
            _clear_segmented_outputs(self.config)
            points, source_colors = load_feedforward_point_cloud(self.config.points_path)
            point_labels, vote_summary = label_points_from_anchors(
                points,
                sources,
                frames,
                depth_band_m=float(self.config.visibility_depth_band_m),
                depth_band_relative=float(self.config.visibility_depth_band_relative),
            )
            if not np.any(point_labels > 0):
                raise ValueError(
                    f"No {self.config.source_name} received a persistent-region label. Check that the "
                    "cloud and staged cameras use the same registered world coordinate system."
                )
            segmented_summary = save_segmented_feedforward_points(
                points,
                source_colors,
                point_labels,
                region_rows,
                self.config,
            )

            rows: list[dict[str, Any]] = []
            total = len(frames)
            for index, frame in enumerate(frames):
                labels, point_count = rasterize_projected_point_labels(
                    points,
                    point_labels,
                    frame,
                    depth_band_m=float(self.config.visibility_depth_band_m),
                    depth_band_relative=float(self.config.visibility_depth_band_relative),
                    radius=int(self.config.raster_radius),
                )
                if save_callback is not None:
                    save_callback(frame, labels)
                rows.append(
                    {
                        "frameId": int(frame.frame_id),
                        "labeledPointCount": int(point_count),
                        "rasterPixelCount": int(np.count_nonzero(labels)),
                        "coverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
                    }
                )
                if progress_callback is not None:
                    progress_callback(index + 1, total, int(frame.frame_id))

            rows.append(
                {
                    "pointVoteSummary": vote_summary,
                    "segmentedPointCloud": segmented_summary,
                }
            )
            return rows


def feedforward_point_availability(config: FeedForwardPointConfig) -> tuple[bool, str]:
    if config.mesh_point_config is not None:
        source = config.mesh_point_config.mesh_path.expanduser().resolve()
        if not source.is_file():
            return False, f"{config.source_name} mesh is missing: {source}"
    else:
        source = config.points_path.expanduser().resolve()
        if not source.is_file():
            return False, f"{config.source_name} point cloud is missing: {source}"
    try:
        import open3d  # noqa: F401
    except ImportError:
        return False, "Open3D is not installed in the editor environment."
    return True, "Ready"


def _ensure_mesh_point_cloud(
    config: OmegaMeshPointCloudConfig | OmegaMeshHybridPointCloudConfig,
) -> dict[str, Any]:
    if isinstance(config, OmegaMeshHybridPointCloudConfig):
        return ensure_omega_mesh_hybrid_point_cloud(config)
    return ensure_omega_mesh_point_cloud(config)


def load_feedforward_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Open3D is required to read the projected point cloud.") from exc

    source = path.expanduser().resolve()
    cloud = o3d.io.read_point_cloud(str(source))
    points = np.asarray(cloud.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Could not read a non-empty point cloud from {source}")
    colors_float = np.asarray(cloud.colors, dtype=np.float64)
    colors = (
        np.clip(np.rint(colors_float * 255.0), 0, 255).astype(np.uint8)
        if colors_float.shape == points.shape
        else np.full(points.shape, 255, dtype=np.uint8)
    )
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    colors = colors[finite]
    if points.shape[0] == 0:
        raise ValueError(f"Feed-forward point cloud contains no finite points: {source}")
    return points.astype(np.float32, copy=False), colors


def project_visible_points(
    points: np.ndarray,
    frame: PropagationFrame,
    *,
    depth_band_m: float,
    depth_band_relative: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points and retain the front depth band at each integer pixel."""

    fx, fy, cx, cy, pose = _camera_values(frame)
    world = np.asarray(points, dtype=np.float64)
    world_to_camera = np.linalg.inv(pose)
    camera = world @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
    depth = camera[:, 2]
    finite = np.all(np.isfinite(camera), axis=1) & (depth > 1.0e-8)
    point_ids = np.flatnonzero(finite)
    if point_ids.size == 0:
        return _empty_projection()

    valid_depth = depth[point_ids]
    u_float = camera[point_ids, 0] * fx / valid_depth + cx
    v_float = camera[point_ids, 1] * fy / valid_depth + cy
    finite_uv = np.isfinite(u_float) & np.isfinite(v_float)
    point_ids = point_ids[finite_uv]
    valid_depth = valid_depth[finite_uv]
    pixel_xy = np.rint(np.column_stack((u_float[finite_uv], v_float[finite_uv]))).astype(np.int64)
    inside = (
        (pixel_xy[:, 0] >= 0)
        & (pixel_xy[:, 0] < int(frame.width))
        & (pixel_xy[:, 1] >= 0)
        & (pixel_xy[:, 1] < int(frame.height))
    )
    point_ids = point_ids[inside]
    pixel_xy = pixel_xy[inside]
    valid_depth = valid_depth[inside]
    if point_ids.size == 0:
        return _empty_projection()

    linear = pixel_xy[:, 1] * int(frame.width) + pixel_xy[:, 0]
    order = np.lexsort((valid_depth, linear))
    linear_sorted = linear[order]
    depth_sorted = valid_depth[order]
    starts = np.r_[0, np.flatnonzero(np.diff(linear_sorted)) + 1]
    counts = np.diff(np.r_[starts, linear_sorted.size])
    nearest = depth_sorted[starts]
    tolerance = np.maximum(float(depth_band_m), np.maximum(nearest, 0.0) * float(depth_band_relative))
    visible_sorted = depth_sorted <= np.repeat(nearest + tolerance, counts)
    visible_rows = order[visible_sorted]
    return (
        point_ids[visible_rows].astype(np.int64, copy=False),
        pixel_xy[visible_rows].astype(np.int32, copy=False),
        valid_depth[visible_rows].astype(np.float32, copy=False),
    )


def label_points_from_anchors(
    points: np.ndarray,
    sources: list[LabeledPropagationSource],
    frames: list[PropagationFrame],
    *,
    depth_band_m: float,
    depth_band_relative: float,
) -> tuple[np.ndarray, dict[str, int]]:
    frame_by_id = {int(frame.frame_id): frame for frame in frames}
    observed_points: list[np.ndarray] = []
    observed_labels: list[np.ndarray] = []
    visible_observations = 0
    for source in sources:
        frame = frame_by_id[int(source.frame_id)]
        labels = np.asarray(source.labels, dtype=np.uint16)
        expected = (int(frame.height), int(frame.width))
        if labels.shape != expected:
            raise ValueError(
                f"Persistent region map shape {labels.shape} does not match frame {frame.frame_id} shape {expected}."
            )
        point_ids, pixel_xy, _depth = project_visible_points(
            points,
            frame,
            depth_band_m=depth_band_m,
            depth_band_relative=depth_band_relative,
        )
        visible_observations += int(point_ids.size)
        values = labels[pixel_xy[:, 1], pixel_xy[:, 0]] if point_ids.size else np.empty(0, dtype=np.uint16)
        positive = values > 0
        if np.any(positive):
            observed_points.append(point_ids[positive])
            observed_labels.append(values[positive])

    point_labels = np.zeros(int(np.asarray(points).shape[0]), dtype=np.uint16)
    if not observed_points:
        return point_labels, {
            "visibleAnchorObservations": visible_observations,
            "positiveAnchorObservations": 0,
            "votedPointCount": 0,
            "labeledPointCount": 0,
            "tiedPointCount": 0,
        }

    point_ids = np.concatenate(observed_points).astype(np.uint64, copy=False)
    labels = np.concatenate(observed_labels).astype(np.uint64, copy=False)
    packed = point_ids * np.uint64(1 << 16) + labels
    pair_keys, pair_counts = np.unique(packed, return_counts=True)
    pair_points = (pair_keys >> np.uint64(16)).astype(np.int64)
    pair_labels = (pair_keys & np.uint64(0xFFFF)).astype(np.uint16)
    starts = np.r_[0, np.flatnonzero(np.diff(pair_points)) + 1]
    maxima = np.maximum.reduceat(pair_counts, starts)
    repeated_maxima = np.repeat(maxima, np.diff(np.r_[starts, pair_counts.size]))
    best = pair_counts == repeated_maxima
    best_points = pair_points[best]
    best_labels = pair_labels[best]
    winner_points, winner_starts, winner_counts = np.unique(
        best_points, return_index=True, return_counts=True
    )
    unambiguous = winner_counts == 1
    point_labels[winner_points[unambiguous]] = best_labels[winner_starts[unambiguous]]
    return point_labels, {
        "visibleAnchorObservations": visible_observations,
        "positiveAnchorObservations": int(point_ids.size),
        "votedPointCount": int(starts.size),
        "labeledPointCount": int(np.count_nonzero(point_labels)),
        "tiedPointCount": int(np.count_nonzero(~unambiguous)),
    }


def rasterize_projected_point_labels(
    points: np.ndarray,
    point_labels: np.ndarray,
    frame: PropagationFrame,
    *,
    depth_band_m: float,
    depth_band_relative: float,
    radius: int,
) -> tuple[np.ndarray, int]:
    if radius < 0 or radius > 8:
        raise ValueError("Sparse point raster radius must be between 0 and 8 pixels.")
    point_ids, pixel_xy, _depth = project_visible_points(
        points,
        frame,
        depth_band_m=depth_band_m,
        depth_band_relative=depth_band_relative,
    )
    labels = np.asarray(point_labels, dtype=np.uint16)[point_ids]
    positive = labels > 0
    output = _rasterize_exclusive_points(
        pixel_xy[positive],
        labels[positive],
        width=int(frame.width),
        height=int(frame.height),
        radius=radius,
    )
    return output, int(np.count_nonzero(positive))


def save_segmented_feedforward_points(
    points: np.ndarray,
    source_colors: np.ndarray,
    point_labels: np.ndarray,
    region_rows: list[dict[str, Any]],
    config: FeedForwardPointConfig,
) -> dict[str, Any] | None:
    output_paths = (
        config.segmented_cache_path,
        config.segmented_ply_path,
        config.segmented_summary_path,
    )
    if all(path is None for path in output_paths):
        return None
    if any(path is None for path in output_paths):
        raise ValueError("Segmented projected-point export requires NPZ, PLY, and summary paths.")
    cache_path, ply_path, summary_path = (
        path.expanduser().resolve() for path in output_paths if path is not None
    )
    labels = np.asarray(point_labels, dtype=np.uint16)
    positive = labels > 0
    source_indices = np.flatnonzero(positive).astype(np.int64)
    positions = np.asarray(points, dtype=np.float32)[positive]
    segmented_labels = labels[positive]
    if positions.shape[0] == 0:
        raise ValueError(f"{config.source_name} voting produced no labeled points.")

    colors_by_region = persistent_region_color_map(region_rows)
    missing = sorted(set(int(value) for value in np.unique(segmented_labels)) - set(colors_by_region))
    if missing:
        raise ValueError(f"Segmented {config.source_name} reference unknown persistent regions: {missing}")
    colors = np.asarray([colors_by_region[int(label)] for label in segmented_labels], dtype=np.uint8)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        positions=positions,
        colors=colors,
        source_colors=np.asarray(source_colors, dtype=np.uint8)[positive],
        source_indices=source_indices,
        point_ids=source_indices,
        labels=segmented_labels,
    )
    temporary_cache.replace(cache_path)

    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Open3D is required to export segmented projected points.") from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64, copy=False))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_ply = ply_path.with_name(ply_path.stem + ".tmp.ply")
    if not o3d.io.write_point_cloud(str(temporary_ply), cloud, write_ascii=False, compressed=True):
        temporary_ply.unlink(missing_ok=True)
        raise RuntimeError(f"Open3D could not write segmented {config.source_name}: {temporary_ply}")
    temporary_ply.replace(ply_path)

    region_by_id = {int(row["id"]): row for row in region_rows if int(row.get("id", 0)) > 0}
    unique, counts = np.unique(segmented_labels, return_counts=True)
    source_path = config.points_path.expanduser().resolve()
    source_stat = source_path.stat()
    summary = {
        "schemaVersion": 1,
        "sourcePath": str(source_path),
        "sourceName": str(config.source_name),
        "source": {
            "path": str(source_path),
            "size": int(source_stat.st_size),
            "mtimeNs": int(source_stat.st_mtime_ns),
        },
        "initMeshPath": (
            str(config.init_mesh_path.expanduser().resolve())
            if config.init_mesh_path is not None and config.init_mesh_path.exists()
            else None
        ),
        "geometrySourcePath": (
            str(config.mesh_point_config.mesh_path.expanduser().resolve())
            if config.mesh_point_config is not None
            else None
        ),
        "inputPointCount": int(np.asarray(points).shape[0]),
        "labeledPointCount": int(positions.shape[0]),
        "unlabeledPointCount": int(np.asarray(points).shape[0] - positions.shape[0]),
        "cachePath": str(cache_path),
        "plyPath": str(ply_path),
        "colorSpace": "persistent_region",
        "regions": [
            {
                "regionId": int(region_id),
                "name": str(region_by_id[int(region_id)].get("name", f"region_{int(region_id):03d}")),
                "color": str(region_by_id[int(region_id)].get("color", "")),
                "pointCount": int(count),
            }
            for region_id, count in zip(unique.tolist(), counts.tolist(), strict=True)
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(summary_path)
    return summary


def _camera_values(frame: PropagationFrame) -> tuple[float, float, float, float, np.ndarray]:
    values = (frame.fx, frame.fy, frame.cx, frame.cy)
    if any(value is None or not np.isfinite(float(value)) for value in values):
        raise ValueError(f"Frame {frame.frame_id} has no valid calibrated pinhole intrinsics.")
    if frame.pose_world_from_camera is None:
        raise ValueError(f"Frame {frame.frame_id} has no world-from-camera pose.")
    pose = np.asarray(frame.pose_world_from_camera, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"Frame {frame.frame_id} has an invalid world-from-camera pose.")
    return float(values[0]), float(values[1]), float(values[2]), float(values[3]), pose


def _empty_projection() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.float32),
    )


def _clear_segmented_outputs(config: FeedForwardPointConfig) -> None:
    for path in (
        config.segmented_cache_path,
        config.segmented_ply_path,
        config.segmented_summary_path,
    ):
        if path is not None:
            path.expanduser().resolve().unlink(missing_ok=True)
