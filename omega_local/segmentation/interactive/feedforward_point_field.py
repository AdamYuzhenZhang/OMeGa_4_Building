"""Projected point-field adapter for segmented feed-forward clouds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .colmap_point_field import ProjectedPointField
from .feedforward_point_propagation import load_feedforward_point_cloud, project_visible_points
from .sam2_video_propagation import PropagationFrame


@dataclass(frozen=True)
class FeedForwardPointFieldConfig:
    points_path: Path
    segmented_points_path: Path
    source_name: str = "OMeGa Initializer Points"
    normal_neighbors: int = 16
    visibility_depth_band_m: float = 0.05
    visibility_depth_band_relative: float = 0.01


class FeedForwardProjectedPointSource:
    """Projects region-labeled feed-forward points into calibrated views."""

    def __init__(self, config: FeedForwardPointFieldConfig) -> None:
        self.config = config
        self.positions, _colors = load_feedforward_point_cloud(config.points_path)
        self.labels = np.zeros(self.positions.shape[0], dtype=np.uint16)
        path = config.segmented_points_path.expanduser().resolve()
        try:
            with np.load(path) as payload:
                labels = np.asarray(payload["labels"], dtype=np.uint16)
                source_indices = np.asarray(payload["source_indices"], dtype=np.int64)
        except (OSError, KeyError, ValueError) as exc:
            raise ValueError(f"Could not read segmented {config.source_name} labels: {path}") from exc
        if (
            labels.ndim != 1
            or source_indices.shape != labels.shape
            or labels.size == 0
            or np.any(source_indices < 0)
            or np.any(source_indices >= self.positions.shape[0])
            or np.unique(source_indices).size != source_indices.size
        ):
            raise ValueError(f"Segmented {config.source_name} arrays are invalid: {path}")
        self.labels[source_indices] = labels
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("The feed-forward point field requires SciPy.") from exc
        self._tree = cKDTree(self.positions.astype(np.float64, copy=False))
        self._normal_values = np.zeros(self.positions.shape, dtype=np.float32)
        self._normal_ready = np.zeros(self.positions.shape[0], dtype=bool)

    def project(self, frame: PropagationFrame) -> ProjectedPointField:
        rows, pixel_xy, _depth = project_visible_points(
            self.positions,
            frame,
            depth_band_m=float(self.config.visibility_depth_band_m),
            depth_band_relative=float(self.config.visibility_depth_band_relative),
        )
        positive = self.labels[rows] > 0
        rows = rows[positive]
        return ProjectedPointField(
            point_ids=rows,
            pixel_xy=pixel_xy[positive],
            positions=self.positions[rows],
            normals=self._normals(rows),
            labels=self.labels[rows],
        )

    def _normals(self, point_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(point_ids, dtype=np.int64)
        missing = np.unique(ids[~self._normal_ready[ids]])
        if missing.size:
            neighbors = min(max(int(self.config.normal_neighbors), 3), int(self.positions.shape[0]))
            _distance, indices = self._tree.query(self.positions[missing], k=neighbors, workers=-1)
            neighborhoods = self.positions[np.asarray(indices, dtype=np.int64)].astype(np.float64)
            centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", centered, centered) / max(neighbors - 1, 1)
            _values, vectors = np.linalg.eigh(covariance)
            normals = vectors[:, :, 0]
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
            self._normal_values[missing] = normals.astype(np.float32)
            self._normal_ready[missing] = True
        return self._normal_values[ids]


def feedforward_point_field_availability(config: FeedForwardPointFieldConfig) -> tuple[bool, str]:
    source_path = config.points_path.expanduser().resolve()
    if not source_path.is_file():
        return False, f"{config.source_name} point cloud is missing: {source_path}"
    path = config.segmented_points_path.expanduser().resolve()
    if not path.is_file():
        return False, f"Run {config.source_name} first; segmented point labels are missing: {path}"
    try:
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError as exc:
        return False, f"Projected point-field dependency is missing: {exc}"
    return True, "Ready"
