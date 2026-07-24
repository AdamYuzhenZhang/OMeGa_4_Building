"""Local point-cloud evidence sources for the interactive editor."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .paths import EditorPaths
from .omega_mesh_point_cloud import (
    OmegaMeshHybridPointCloudConfig,
    OmegaMeshPointCloudConfig,
    ensure_omega_mesh_hybrid_point_cloud,
    ensure_omega_mesh_point_cloud,
    omega_mesh_hybrid_point_cloud_status,
    omega_mesh_point_cloud_status,
)


_CACHE_VERSION = 2
_CLEANED_CACHE_VERSION = 1
_CROP_PADDING_FRACTION = 0.20
_MIN_CROP_PADDING = 0.25
_SOR_NEIGHBORS = 20
_SOR_STD_RATIO = 2.0
_SOR_METHOD = "open3d_statistical_outlier_removal"


def _bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.min(points, axis=0), np.max(points, axis=0)


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtimeNs": int(stat.st_mtime_ns),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        positions = np.asarray(data["positions"], dtype=np.float32)
        colors = np.asarray(data["colors"], dtype=np.uint8)
        source_indices = np.asarray(data["source_indices"], dtype=np.int64)
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or positions.shape[0] == 0
        or colors.shape != positions.shape
        or source_indices.shape != (positions.shape[0],)
    ):
        raise ValueError(f"Invalid staged point-cloud cache: {path}")
    return positions, colors, source_indices


class PointCloudSourceManager:
    """Stages aligned evidence clouds without changing source point order."""

    def __init__(self, paths: EditorPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        source = self.paths.colmap_sparse_source
        summary = _read_json(self.paths.colmap_sparse_summary)
        cache_current = False
        if source.exists() and self.paths.colmap_sparse_cache.exists() and summary is not None:
            cache_current = (
                int(summary.get("version", -1)) == _CACHE_VERSION
                and summary.get("source") == _source_signature(source)
            )
        status = {
            "available": bool(source.exists()),
            "ready": bool(cache_current),
            "sourcePath": str(source),
            "cachePath": str(self.paths.colmap_sparse_cache),
            "pointCount": int(summary.get("outputPointCount", 0)) if cache_current and summary else 0,
            "summary": summary if cache_current else None,
        }
        status["cleaned"] = self._cleaned_status(raw_ready=cache_current)
        status["segmented"] = self._segmented_status()
        return status

    def feedforward_status(self) -> dict[str, Any]:
        source = self.paths.feedforward_source
        summary = _read_json(self.paths.feedforward_summary)
        cache_current = False
        if source.exists() and self.paths.feedforward_cache.exists() and summary is not None:
            cache_current = (
                int(summary.get("version", -1)) == _CACHE_VERSION
                and summary.get("source") == _source_signature(source)
            )
        return {
            "available": bool(source.exists()),
            "ready": bool(cache_current),
            "sourcePath": str(source),
            "initMeshPath": str(self.paths.feedforward_init_mesh),
            "cachePath": str(self.paths.feedforward_cache),
            "pointCount": int(summary.get("outputPointCount", 0)) if cache_current and summary else 0,
            "summary": summary if cache_current else None,
            "segmented": self._segmented_feedforward_status(),
        }

    def omega_final_status(self) -> dict[str, Any]:
        status = omega_mesh_point_cloud_status(self._omega_final_config())
        status["hybrid"] = omega_mesh_hybrid_point_cloud_status(self._omega_final_hybrid_config())
        status["segmented"] = self._segmented_omega_final_status()
        return status

    def colmap_payload(
        self,
        reference_points: np.ndarray,
        *,
        max_points: int,
        seed: int,
        mode: str = "raw",
    ) -> dict[str, Any]:
        mode = str(mode).strip().lower()
        if mode == "cleaned":
            cache, summary = self._ensure_cleaned_colmap_cache(reference_points)
            source_name = "colmap_sparse_cleaned"
        elif mode == "segmented":
            cache, summary = self._segmented_colmap_cache()
            source_name = "colmap_sparse_segmented"
        elif mode == "raw":
            cache, summary = self._ensure_colmap_cache(reference_points)
            source_name = "colmap_sparse"
        else:
            raise ValueError(f"Unknown COLMAP point-cloud mode: {mode}")
        payload = self._payload_from_cache(
            cache,
            summary,
            source_name=source_name,
            mode=mode,
            max_points=max_points,
            seed=seed,
        )
        payload["cleaned"] = mode == "cleaned"
        return payload

    def feedforward_payload(
        self,
        reference_points: np.ndarray,
        *,
        max_points: int,
        seed: int,
        mode: str = "raw",
    ) -> dict[str, Any]:
        mode = str(mode).strip().lower()
        if mode == "segmented":
            cache, summary = self._segmented_feedforward_cache()
            source_name = "feedforward_init_segmented"
        elif mode == "raw":
            cache, summary = self._ensure_feedforward_cache(reference_points)
            source_name = "feedforward_init"
        else:
            raise ValueError(f"Unknown feed-forward point-cloud mode: {mode}")
        return self._payload_from_cache(
            cache,
            summary,
            source_name=source_name,
            mode=mode,
            max_points=max_points,
            seed=seed,
        )

    def omega_final_payload(
        self,
        *,
        max_points: int,
        seed: int,
        mode: str = "raw",
    ) -> dict[str, Any]:
        mode = str(mode).strip().lower()
        if mode == "segmented":
            cache, summary = self._segmented_omega_final_cache()
            source_name = "omega_final_clean_hybrid_segmented"
        elif mode == "hybrid":
            with self._lock:
                summary = ensure_omega_mesh_hybrid_point_cloud(self._omega_final_hybrid_config())
            cache = self.paths.omega_final_hybrid_cache
            source_name = "omega_final_clean_hybrid"
        elif mode == "raw":
            with self._lock:
                summary = ensure_omega_mesh_point_cloud(self._omega_final_config())
            cache = self.paths.omega_final_cache
            source_name = "omega_final_vertices"
        else:
            raise ValueError(f"Unknown final OMeGa point-cloud mode: {mode}")
        return self._payload_from_cache(
            cache,
            summary,
            source_name=source_name,
            mode=mode,
            max_points=max_points,
            seed=seed,
        )

    def _payload_from_cache(
        self,
        cache: Path,
        summary: dict[str, Any],
        *,
        source_name: str,
        mode: str,
        max_points: int,
        seed: int,
    ) -> dict[str, Any]:
        points, colors, _ = _load_cache(cache)
        count = int(points.shape[0])
        if count > max_points:
            rng = np.random.default_rng(int(seed))
            indices = np.sort(rng.choice(count, size=int(max_points), replace=False))
            served = points[indices]
            served_colors = colors[indices]
        else:
            served = points
            served_colors = colors
        color_bins = served_colors.astype(np.uint32) >> 3
        color_keys = (color_bins[:, 0] << 10) | (color_bins[:, 1] << 5) | color_bins[:, 2]
        order = np.argsort(color_keys, kind="stable")
        served = served[order]
        served_colors = served_colors[order]
        bounds_min, bounds_max = _bounds(points)
        return {
            "source": source_name,
            "mode": mode,
            "segmented": mode == "segmented",
            "pointCount": count,
            "servedPointCount": int(served.shape[0]),
            "bounds": {
                "min": bounds_min.astype(float).tolist(),
                "max": bounds_max.astype(float).tolist(),
                "center": ((bounds_min + bounds_max) * 0.5).astype(float).tolist(),
                "radius": float(np.linalg.norm(bounds_max - bounds_min) * 0.5),
            },
            "positions": served.reshape(-1).astype(float).tolist(),
            "colors": served_colors.reshape(-1).astype(int).tolist(),
            "cachePath": str(cache),
            "summary": summary,
        }

    def _cleaned_status(self, *, raw_ready: bool) -> dict[str, Any]:
        summary = _read_json(self.paths.colmap_sparse_cleaned_summary)
        cache_current = False
        if (
            raw_ready
            and self.paths.colmap_sparse_cache.exists()
            and self.paths.colmap_sparse_cleaned_cache.exists()
            and self.paths.colmap_sparse_cleaned_ply.exists()
            and summary is not None
        ):
            expected = self._cleaned_expected()
            cache_current = all(summary.get(key) == value for key, value in expected.items())
        return {
            "available": bool(self.paths.colmap_sparse_source.exists()),
            "ready": bool(cache_current),
            "method": _SOR_METHOD,
            "parameters": {
                "nbNeighbors": _SOR_NEIGHBORS,
                "stdRatio": _SOR_STD_RATIO,
            },
            "cachePath": str(self.paths.colmap_sparse_cleaned_cache),
            "plyPath": str(self.paths.colmap_sparse_cleaned_ply),
            "summaryPath": str(self.paths.colmap_sparse_cleaned_summary),
            "pointCount": int(summary.get("outputPointCount", 0)) if cache_current and summary else 0,
            "summary": summary if cache_current else None,
        }

    def _segmented_status(self) -> dict[str, Any]:
        summary = _read_json(self.paths.colmap_sparse_segmented_summary)
        ready = bool(
            self.paths.colmap_sparse_segmented_cache.is_file()
            and self.paths.colmap_sparse_segmented_ply.is_file()
            and summary is not None
            and int(summary.get("schemaVersion", -1)) == 1
        )
        return {
            "available": ready,
            "ready": ready,
            "method": "colmap_track_persistent_region_votes",
            "cachePath": str(self.paths.colmap_sparse_segmented_cache),
            "plyPath": str(self.paths.colmap_sparse_segmented_ply),
            "summaryPath": str(self.paths.colmap_sparse_segmented_summary),
            "pointCount": int(summary.get("labeledPointCount", 0)) if ready and summary else 0,
            "summary": summary if ready else None,
        }

    def _segmented_feedforward_status(self) -> dict[str, Any]:
        summary = _read_json(self.paths.feedforward_segmented_summary)
        ready = bool(
            self.paths.feedforward_source.is_file()
            and self.paths.feedforward_segmented_cache.is_file()
            and self.paths.feedforward_segmented_ply.is_file()
            and summary is not None
            and int(summary.get("schemaVersion", -1)) == 1
            and summary.get("source") == _source_signature(self.paths.feedforward_source)
        )
        return {
            "available": ready,
            "ready": ready,
            "method": "projected_feedforward_persistent_region_votes",
            "cachePath": str(self.paths.feedforward_segmented_cache),
            "plyPath": str(self.paths.feedforward_segmented_ply),
            "summaryPath": str(self.paths.feedforward_segmented_summary),
            "pointCount": int(summary.get("labeledPointCount", 0)) if ready and summary else 0,
            "summary": summary if ready else None,
        }

    def _segmented_omega_final_status(self) -> dict[str, Any]:
        summary = _read_json(self.paths.omega_final_segmented_summary)
        source_current = bool(
            omega_mesh_hybrid_point_cloud_status(self._omega_final_hybrid_config())["ready"]
        )
        ready = bool(
            source_current
            and self.paths.omega_final_hybrid_points.is_file()
            and self.paths.omega_final_segmented_cache.is_file()
            and self.paths.omega_final_segmented_ply.is_file()
            and summary is not None
            and int(summary.get("schemaVersion", -1)) == 1
            and summary.get("source") == _source_signature(self.paths.omega_final_hybrid_points)
        )
        return {
            "available": ready,
            "ready": ready,
            "method": "projected_omega_final_persistent_region_votes",
            "cachePath": str(self.paths.omega_final_segmented_cache),
            "plyPath": str(self.paths.omega_final_segmented_ply),
            "summaryPath": str(self.paths.omega_final_segmented_summary),
            "pointCount": int(summary.get("labeledPointCount", 0)) if ready and summary else 0,
            "summary": summary if ready else None,
        }

    def _segmented_colmap_cache(self) -> tuple[Path, dict[str, Any]]:
        status = self._segmented_status()
        if not status["ready"]:
            raise FileNotFoundError(
                "Segmented COLMAP points are unavailable. Run COLMAP Tracks after assigning "
                "persistent regions to complete frames."
            )
        try:
            _load_cache(self.paths.colmap_sparse_segmented_cache)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Segmented COLMAP point cache is invalid: {self.paths.colmap_sparse_segmented_cache}"
            ) from exc
        return self.paths.colmap_sparse_segmented_cache, dict(status["summary"])

    def _segmented_feedforward_cache(self) -> tuple[Path, dict[str, Any]]:
        status = self._segmented_feedforward_status()
        if not status["ready"]:
            raise FileNotFoundError(
                "Segmented initializer points are unavailable. Run OMeGa Initializer Points after assigning "
                "persistent regions to complete frames."
            )
        try:
            _load_cache(self.paths.feedforward_segmented_cache)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Segmented initializer point cache is invalid: {self.paths.feedforward_segmented_cache}"
            ) from exc
        return self.paths.feedforward_segmented_cache, dict(status["summary"])

    def _segmented_omega_final_cache(self) -> tuple[Path, dict[str, Any]]:
        status = self._segmented_omega_final_status()
        if not status["ready"]:
            raise FileNotFoundError(
                "Segmented final OMeGa points are unavailable. Run OMeGa Final Points after "
                "assigning persistent regions to complete frames."
            )
        try:
            _load_cache(self.paths.omega_final_segmented_cache)
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Segmented final OMeGa point cache is invalid: {self.paths.omega_final_segmented_cache}"
            ) from exc
        return self.paths.omega_final_segmented_cache, dict(status["summary"])

    def _omega_final_config(self) -> OmegaMeshPointCloudConfig:
        return OmegaMeshPointCloudConfig(
            mesh_path=self.paths.omega_final_mesh_source,
            points_path=self.paths.omega_final_points,
            cache_path=self.paths.omega_final_cache,
            summary_path=self.paths.omega_final_summary,
        )

    def _omega_final_hybrid_config(self) -> OmegaMeshHybridPointCloudConfig:
        return OmegaMeshHybridPointCloudConfig(
            mesh_path=self.paths.omega_final_mesh_source,
            points_path=self.paths.omega_final_hybrid_points,
            cache_path=self.paths.omega_final_hybrid_cache,
            summary_path=self.paths.omega_final_hybrid_summary,
            surface_cache_path=self.paths.omega_final_surface_cache,
            surface_summary_path=self.paths.omega_final_surface_summary,
        )

    def _cleaned_expected(self) -> dict[str, Any]:
        return {
            "version": _CLEANED_CACHE_VERSION,
            "method": _SOR_METHOD,
            "parameters": {
                "nbNeighbors": _SOR_NEIGHBORS,
                "stdRatio": _SOR_STD_RATIO,
            },
            "inputCache": _source_signature(self.paths.colmap_sparse_cache),
        }

    def _ensure_colmap_cache(self, reference_points: np.ndarray) -> tuple[Path, dict[str, Any]]:
        return self._ensure_source_cache(
            source=self.paths.colmap_sparse_source,
            cache_path=self.paths.colmap_sparse_cache,
            summary_path=self.paths.colmap_sparse_summary,
            reference_points=reference_points,
            source_label="COLMAP sparse",
            preserve_exact=False,
        )

    def _ensure_feedforward_cache(self, reference_points: np.ndarray) -> tuple[Path, dict[str, Any]]:
        return self._ensure_source_cache(
            source=self.paths.feedforward_source,
            cache_path=self.paths.feedforward_cache,
            summary_path=self.paths.feedforward_summary,
            reference_points=reference_points,
            source_label="feed-forward initializer",
            preserve_exact=True,
        )

    def _ensure_source_cache(
        self,
        *,
        source: Path,
        cache_path: Path,
        summary_path: Path,
        reference_points: np.ndarray,
        source_label: str,
        preserve_exact: bool,
    ) -> tuple[Path, dict[str, Any]]:
        if not source.exists():
            raise FileNotFoundError(f"Aligned {source_label} point cloud does not exist: {source}")

        reference = np.asarray(reference_points, dtype=np.float64)
        if reference.ndim != 2 or reference.shape[1] != 3 or reference.shape[0] == 0:
            raise ValueError("A non-empty Nx3 reconstruction point cloud is required for COLMAP staging.")
        reference = reference[np.all(np.isfinite(reference), axis=1)]
        if reference.shape[0] == 0:
            raise ValueError("The reconstruction point cloud contains no finite points.")

        reference_min, reference_max = _bounds(reference)
        extent = np.maximum(reference_max - reference_min, 1e-6)
        padding = np.maximum(extent * _CROP_PADDING_FRACTION, _MIN_CROP_PADDING)
        crop_min = reference_min - padding
        crop_max = reference_max + padding
        expected = {
            "version": _CACHE_VERSION,
            "source": _source_signature(source),
        }
        if preserve_exact:
            expected["preserveExactPointSet"] = True
        else:
            expected.update(
                {
                    "referenceBounds": {
                        "min": reference_min.astype(float).tolist(),
                        "max": reference_max.astype(float).tolist(),
                    },
                    "cropBounds": {
                        "min": crop_min.astype(float).tolist(),
                        "max": crop_max.astype(float).tolist(),
                    },
                }
            )

        with self._lock:
            summary = _read_json(summary_path)
            if (
                cache_path.exists()
                and summary is not None
                and all(summary.get(key) == value for key, value in expected.items())
            ):
                try:
                    _load_cache(cache_path)
                    return cache_path, summary
                except (KeyError, OSError, ValueError):
                    pass

            summary = self._stage_source(
                source,
                crop_min,
                crop_max,
                expected,
                cache_path=cache_path,
                summary_path=summary_path,
                source_label=source_label,
                preserve_exact=preserve_exact,
            )
            return cache_path, summary

    def _ensure_cleaned_colmap_cache(
        self,
        reference_points: np.ndarray,
    ) -> tuple[Path, dict[str, Any]]:
        raw_cache, _ = self._ensure_colmap_cache(reference_points)
        expected = self._cleaned_expected()
        with self._lock:
            summary = _read_json(self.paths.colmap_sparse_cleaned_summary)
            if (
                self.paths.colmap_sparse_cleaned_cache.exists()
                and self.paths.colmap_sparse_cleaned_ply.exists()
                and summary is not None
                and all(summary.get(key) == value for key, value in expected.items())
            ):
                try:
                    _load_cache(self.paths.colmap_sparse_cleaned_cache)
                    return self.paths.colmap_sparse_cleaned_cache, summary
                except (KeyError, OSError, ValueError):
                    pass

            summary = self._stage_cleaned_colmap(raw_cache, expected)
            return self.paths.colmap_sparse_cleaned_cache, summary

    def _stage_source(
        self,
        source: Path,
        crop_min: np.ndarray,
        crop_max: np.ndarray,
        expected: dict[str, Any],
        *,
        cache_path: Path,
        summary_path: Path,
        source_label: str,
        preserve_exact: bool,
    ) -> dict[str, Any]:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(f"Open3D is required to stage the {source_label} point cloud.") from exc

        cloud = o3d.io.read_point_cloud(str(source))
        points = np.asarray(cloud.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"Could not read a non-empty point cloud from {source}")

        source_count = int(points.shape[0])
        source_indices = np.arange(source_count, dtype=np.int64)
        colors_float = np.asarray(cloud.colors, dtype=np.float64)
        source_has_colors = colors_float.shape == points.shape
        if source_has_colors:
            colors = np.clip(np.rint(colors_float * 255.0), 0, 255).astype(np.uint8)
        else:
            colors = np.full(points.shape, 255, dtype=np.uint8)

        finite = np.all(np.isfinite(points), axis=1)
        points = points[finite]
        colors = colors[finite]
        source_indices = source_indices[finite]
        finite_count = int(points.shape[0])

        if preserve_exact:
            first_indices = np.arange(points.shape[0], dtype=np.int64)
        else:
            _, first_indices = np.unique(points, axis=0, return_index=True)
            first_indices.sort()
            points = points[first_indices]
            colors = colors[first_indices]
            source_indices = source_indices[first_indices]
        unique_count = int(points.shape[0])

        inside = (
            np.ones(points.shape[0], dtype=bool)
            if preserve_exact
            else np.all((points >= crop_min) & (points <= crop_max), axis=1)
        )
        points = points[inside]
        colors = colors[inside]
        source_indices = source_indices[inside]
        if points.shape[0] == 0:
            raise ValueError(f"{source_label} staging removed every point; check coordinate-system alignment.")

        positions = points.astype(np.float32, copy=False)
        output_min, output_max = _bounds(positions)
        self.paths.point_clouds_dir.mkdir(parents=True, exist_ok=True)
        temporary_cache = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_cache,
            positions=positions,
            colors=colors,
            source_indices=source_indices,
        )
        temporary_cache.replace(cache_path)

        summary = {
            **expected,
            "sourcePointCount": source_count,
            "finitePointCount": finite_count,
            "uniquePointCount": unique_count,
            "outputPointCount": int(positions.shape[0]),
            "removedNonFinite": source_count - finite_count,
            "removedDuplicates": finite_count - unique_count,
            "removedOutsideWorkingBounds": unique_count - int(positions.shape[0]),
            "outputBounds": {
                "min": output_min.astype(float).tolist(),
                "max": output_max.astype(float).tolist(),
            },
            "colorsPreserved": bool(source_has_colors),
            "sourceOrderIndexPreserved": True,
        }
        _write_json_atomic(summary_path, summary)
        return summary

    def _stage_cleaned_colmap(
        self,
        raw_cache: Path,
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Open3D is required to clean the COLMAP sparse point cloud.") from exc

        positions, colors, source_indices = _load_cache(raw_cache)
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64, copy=False))
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
        _, inlier_rows = cloud.remove_statistical_outlier(
            nb_neighbors=_SOR_NEIGHBORS,
            std_ratio=_SOR_STD_RATIO,
        )
        inlier_rows = np.asarray(inlier_rows, dtype=np.int64)
        inlier_rows.sort()
        if inlier_rows.size == 0:
            raise ValueError("Statistical outlier removal rejected every COLMAP point.")

        cleaned_positions = positions[inlier_rows]
        cleaned_colors = colors[inlier_rows]
        cleaned_source_indices = source_indices[inlier_rows]
        output_min, output_max = _bounds(cleaned_positions)

        self.paths.point_clouds_dir.mkdir(parents=True, exist_ok=True)
        temporary_cache = self.paths.colmap_sparse_cleaned_cache.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary_cache,
            positions=cleaned_positions,
            colors=cleaned_colors,
            source_indices=cleaned_source_indices,
        )
        temporary_cache.replace(self.paths.colmap_sparse_cleaned_cache)

        cleaned_cloud = o3d.geometry.PointCloud()
        cleaned_cloud.points = o3d.utility.Vector3dVector(cleaned_positions.astype(np.float64, copy=False))
        cleaned_cloud.colors = o3d.utility.Vector3dVector(cleaned_colors.astype(np.float64) / 255.0)
        temporary_ply = self.paths.colmap_sparse_cleaned_ply.with_name(
            self.paths.colmap_sparse_cleaned_ply.stem + ".tmp.ply"
        )
        if not o3d.io.write_point_cloud(
            str(temporary_ply),
            cleaned_cloud,
            write_ascii=False,
            compressed=True,
            print_progress=False,
        ):
            temporary_ply.unlink(missing_ok=True)
            raise RuntimeError(f"Open3D could not write cleaned COLMAP points: {temporary_ply}")
        temporary_ply.replace(self.paths.colmap_sparse_cleaned_ply)

        input_count = int(positions.shape[0])
        output_count = int(cleaned_positions.shape[0])
        summary = {
            **expected,
            "inputPointCount": input_count,
            "outputPointCount": output_count,
            "removedPointCount": input_count - output_count,
            "removedFraction": float((input_count - output_count) / input_count),
            "outputBounds": {
                "min": output_min.astype(float).tolist(),
                "max": output_max.astype(float).tolist(),
            },
            "colorsPreserved": True,
            "sourceOrderIndexPreserved": True,
            "plyPath": str(self.paths.colmap_sparse_cleaned_ply),
        }
        _write_json_atomic(self.paths.colmap_sparse_cleaned_summary, summary)
        return summary
