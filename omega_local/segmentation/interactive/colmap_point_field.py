"""COLMAP observations exposed as a projected 3D point field.

The dense decoder consumes this small interface instead of COLMAP directly. A
future MASt3R adapter can therefore provide denser projected points without
changing the 2D/3D inference code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .colmap_track_propagation import (
    _match_colmap_images,
    _read_pixel_transforms,
    _read_reconstruction,
    _scaled_observations,
    _valid_point_mask,
)
from .sam2_video_propagation import PropagationFrame


@dataclass(frozen=True)
class ColmapPointFieldConfig:
    model_path: Path
    segmented_points_path: Path
    pixel_transform_summary: Path | None = None
    min_track_length: int = 2
    max_reprojection_error: float = 4.0
    normal_neighbors: int = 16


@dataclass(frozen=True)
class ProjectedPointField:
    point_ids: np.ndarray
    pixel_xy: np.ndarray
    positions: np.ndarray
    normals: np.ndarray
    labels: np.ndarray


class ColmapProjectedPointSource:
    """Loads exact visible SfM observations and their voted persistent IDs."""

    def __init__(self, config: ColmapPointFieldConfig, frames: list[PropagationFrame]) -> None:
        self.config = config
        self.reconstruction = _read_reconstruction(config.model_path)
        self.image_by_frame_id = _match_colmap_images(self.reconstruction, frames)
        self.pixel_transforms = _read_pixel_transforms(config.pixel_transform_summary)
        self.valid_points = _valid_point_mask(
            self.reconstruction,
            min_track_length=int(config.min_track_length),
            max_reprojection_error=float(config.max_reprojection_error),
        )
        self.point_labels = _load_segmented_point_labels(
            config.segmented_points_path,
            max_point_id=int(self.valid_points.shape[0] - 1),
        )

        valid_ids = np.flatnonzero(self.valid_points).astype(np.int64)
        if valid_ids.size == 0:
            raise ValueError("The COLMAP point field has no points after track/error filtering.")
        self.valid_ids = valid_ids
        self.valid_positions = np.asarray(
            [_point_xyz(self.reconstruction.points3D[int(point_id)]) for point_id in valid_ids],
            dtype=np.float64,
        )
        if not np.isfinite(self.valid_positions).all():
            raise ValueError("The filtered COLMAP point field contains non-finite positions.")

        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("The COLMAP point field requires SciPy.") from exc
        self._tree = cKDTree(self.valid_positions)
        self._normal_values = np.zeros((self.valid_points.shape[0], 3), dtype=np.float32)
        self._normal_ready = np.zeros(self.valid_points.shape[0], dtype=bool)

    def project(self, frame: PropagationFrame) -> ProjectedPointField:
        image = self.image_by_frame_id[int(frame.frame_id)]
        point_ids, pixel_xy = _scaled_observations(
            image,
            frame,
            self.valid_points,
            pixel_transforms=self.pixel_transforms,
        )
        if point_ids.size == 0:
            return ProjectedPointField(
                point_ids=np.empty(0, dtype=np.int64),
                pixel_xy=np.empty((0, 2), dtype=np.int32),
                positions=np.empty((0, 3), dtype=np.float32),
                normals=np.empty((0, 3), dtype=np.float32),
                labels=np.empty(0, dtype=np.uint16),
            )
        positions = np.asarray(
            [_point_xyz(self.reconstruction.points3D[int(point_id)]) for point_id in point_ids],
            dtype=np.float32,
        )
        return ProjectedPointField(
            point_ids=point_ids.astype(np.int64, copy=False),
            pixel_xy=pixel_xy.astype(np.int32, copy=False),
            positions=positions,
            normals=self._normals(point_ids),
            labels=self.point_labels[point_ids].astype(np.uint16, copy=False),
        )

    def _normals(self, point_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(point_ids, dtype=np.int64)
        missing = np.unique(ids[~self._normal_ready[ids]])
        if missing.size:
            if self.valid_ids.size < 3:
                self._normal_values[missing] = 0.0
                self._normal_ready[missing] = True
                return self._normal_values[ids]
            query_positions = np.asarray(
                [_point_xyz(self.reconstruction.points3D[int(point_id)]) for point_id in missing],
                dtype=np.float64,
            )
            neighbor_count = min(max(int(self.config.normal_neighbors), 3), int(self.valid_ids.size))
            _distance, neighbor_indices = self._tree.query(query_positions, k=neighbor_count, workers=-1)
            neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
            if neighbor_indices.ndim == 1:
                neighbor_indices = neighbor_indices[:, None]
            neighborhoods = self.valid_positions[neighbor_indices]
            centered = neighborhoods - np.mean(neighborhoods, axis=1, keepdims=True)
            covariance = np.einsum("nki,nkj->nij", centered, centered) / max(neighbor_count - 1, 1)
            _values, vectors = np.linalg.eigh(covariance)
            normals = vectors[:, :, 0]
            normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(normal_norm, 1.0e-12)
            self._normal_values[missing] = normals.astype(np.float32)
            self._normal_ready[missing] = True
        return self._normal_values[ids]


def colmap_point_field_availability(config: ColmapPointFieldConfig) -> tuple[bool, str]:
    model_path = config.model_path.expanduser().resolve()
    if not model_path.is_dir():
        return False, f"COLMAP model is missing: {model_path}"
    segmented_path = config.segmented_points_path.expanduser().resolve()
    if not segmented_path.is_file():
        return False, f"Run COLMAP Tracks first; segmented point labels are missing: {segmented_path}"
    transform_path = config.pixel_transform_summary
    if transform_path is not None and not transform_path.expanduser().resolve().is_file():
        return False, f"COLMAP pixel transform summary is missing: {transform_path}"
    try:
        import pycolmap  # noqa: F401
        from scipy.spatial import cKDTree  # noqa: F401
    except ImportError as exc:
        return False, f"COLMAP point-field dependency is missing: {exc}"
    return True, "Ready"


def _load_segmented_point_labels(path: Path, *, max_point_id: int) -> np.ndarray:
    path = path.expanduser().resolve()
    try:
        with np.load(path) as payload:
            point_ids = np.asarray(payload["point_ids"], dtype=np.int64)
            labels = np.asarray(payload["labels"], dtype=np.uint16)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Could not read segmented COLMAP point labels: {path}") from exc
    if point_ids.ndim != 1 or labels.shape != point_ids.shape:
        raise ValueError(f"Segmented COLMAP point arrays have incompatible shapes: {path}")
    if point_ids.size == 0 or np.any(point_ids < 0) or np.any(point_ids > int(max_point_id)):
        raise ValueError(f"Segmented COLMAP point IDs are empty or outside the reconstruction range: {path}")
    if np.unique(point_ids).size != point_ids.size:
        raise ValueError(f"Segmented COLMAP point IDs are not unique: {path}")
    point_labels = np.zeros(int(max_point_id) + 1, dtype=np.uint16)
    point_labels[point_ids] = labels
    return point_labels


def _point_xyz(point: Any) -> np.ndarray:
    value = getattr(point, "xyz")
    value = value() if callable(value) else value
    return np.asarray(value, dtype=np.float64).reshape(3)
