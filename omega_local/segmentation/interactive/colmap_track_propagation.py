"""Sparse persistent-region transfer through exact COLMAP feature tracks."""

from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .sam2_video_propagation import LabeledPropagationSource, PropagationFrame
from .regions import persistent_region_color_map


@dataclass(frozen=True)
class ColmapTrackConfig:
    model_path: Path
    pixel_transform_summary: Path | None = None
    aligned_points_path: Path | None = None
    segmented_cache_path: Path | None = None
    segmented_ply_path: Path | None = None
    segmented_summary_path: Path | None = None
    min_track_length: int = 2
    max_reprojection_error: float = 4.0
    raster_radius: int = 1


class ColmapTrackPropagationSession:
    """Transfers labels only where an SfM point was observed in both views."""

    def __init__(self, config: ColmapTrackConfig) -> None:
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
            raise ValueError("COLMAP track propagation needs at least one frame.")
        if not sources:
            raise ValueError("COLMAP track propagation needs at least one complete frame.")

        with self._run_lock:
            _clear_segmented_colmap_outputs(self.config)
            reconstruction = _read_reconstruction(self.config.model_path)
            image_by_frame_id = _match_colmap_images(reconstruction, frames)
            pixel_transforms = _read_pixel_transforms(self.config.pixel_transform_summary)
            valid_points = _valid_point_mask(
                reconstruction,
                min_track_length=int(self.config.min_track_length),
                max_reprojection_error=float(self.config.max_reprojection_error),
            )
            point_labels, vote_summary = _label_points_from_anchors(
                sources,
                frames,
                image_by_frame_id,
                valid_points,
                pixel_transforms,
            )
            if int(np.count_nonzero(point_labels)) == 0:
                raise ValueError(
                    "No COLMAP tracks received a persistent-region label from the complete frames. "
                    "Check frame-name alignment and whether edited regions overlap sparse features."
                )
            segmented_summary = _save_segmented_colmap_points(
                reconstruction,
                point_labels,
                region_rows,
                self.config,
            )

            rows: list[dict[str, Any]] = []
            total = len(frames)
            for index, frame in enumerate(frames):
                image = image_by_frame_id[int(frame.frame_id)]
                labels, point_count = _rasterize_frame_observations(
                    image,
                    frame,
                    valid_points,
                    point_labels,
                    pixel_transforms,
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
                    "trackVoteSummary": vote_summary,
                    "segmentedPointCloud": segmented_summary,
                }
            )
            return rows


def colmap_track_availability(config: ColmapTrackConfig) -> tuple[bool, str]:
    path = config.model_path.expanduser().resolve()
    if not path.is_dir():
        return False, f"COLMAP track model is missing: {path}"
    required_groups = [
        ("cameras", path / "cameras.bin", path / "cameras.txt"),
        ("images", path / "images.bin", path / "images.txt"),
        ("points3D", path / "points3D.bin", path / "points3D.txt"),
    ]
    for label, binary, text in required_groups:
        if not binary.is_file() and not text.is_file():
            return False, f"COLMAP {label} file is missing under: {path}"
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        return False, "pycolmap is not installed in the editor environment."
    transform_path = config.pixel_transform_summary
    if transform_path is not None and not transform_path.expanduser().resolve().is_file():
        return False, f"COLMAP-to-editor pixel transform summary is missing: {transform_path}"
    if config.aligned_points_path is not None and not config.aligned_points_path.expanduser().resolve().is_file():
        return False, f"Aligned COLMAP point cloud is missing: {config.aligned_points_path}"
    return True, "Ready"


def _save_segmented_colmap_points(
    reconstruction,
    point_labels: np.ndarray,
    region_rows: list[dict[str, Any]],
    config: ColmapTrackConfig,
) -> dict[str, Any] | None:
    output_paths = (
        config.aligned_points_path,
        config.segmented_cache_path,
        config.segmented_ply_path,
        config.segmented_summary_path,
    )
    if all(path is None for path in output_paths):
        return None
    if any(path is None for path in output_paths):
        raise ValueError("Segmented COLMAP export requires source, NPZ, PLY, and summary paths.")

    source_path, cache_path, ply_path, summary_path = (
        path.expanduser().resolve() for path in output_paths if path is not None
    )
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Open3D is required to export segmented COLMAP points.") from exc

    cloud = o3d.io.read_point_cloud(str(source_path))
    positions = np.asarray(cloud.points, dtype=np.float64)
    source_colors_float = np.asarray(cloud.colors, dtype=np.float64)
    point_ids = np.asarray(sorted(int(point_id) for point_id in reconstruction.points3D.keys()), dtype=np.int64)
    if positions.shape != (point_ids.size, 3):
        raise ValueError(
            "Aligned COLMAP PLY does not preserve the reconstruction point count: "
            f"{positions.shape[0]} PLY points versus {point_ids.size} COLMAP points."
        )

    if source_colors_float.shape != positions.shape:
        raise ValueError("Aligned COLMAP PLY needs source RGB to verify its point-ID ordering.")
    source_colors = np.clip(np.rint(source_colors_float * 255.0), 0, 255).astype(np.uint8)
    reconstruction_colors = np.asarray(
        [reconstruction.points3D[int(point_id)].color for point_id in point_ids],
        dtype=np.uint8,
    )
    agreement = float(np.mean(np.all(source_colors == reconstruction_colors, axis=1)))
    if agreement < 0.99:
        raise ValueError(
            "Aligned COLMAP PLY order cannot be matched safely to point IDs "
            f"(RGB agreement {agreement:.3f})."
        )

    labels = np.asarray(point_labels, dtype=np.uint16)[point_ids]
    positive = labels > 0
    if not np.any(positive):
        raise ValueError("COLMAP track voting produced no points for segmented export.")
    segmented_positions = positions[positive].astype(np.float32, copy=False)
    segmented_labels = labels[positive]
    source_indices = np.flatnonzero(positive).astype(np.int64, copy=False)

    colors_by_region = persistent_region_color_map(region_rows)
    missing = sorted(set(int(value) for value in np.unique(segmented_labels)) - set(colors_by_region))
    if missing:
        raise ValueError(f"Segmented COLMAP points reference unknown persistent regions: {missing}")
    segmented_colors = np.asarray(
        [colors_by_region[int(region_id)] for region_id in segmented_labels],
        dtype=np.uint8,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        positions=segmented_positions,
        colors=segmented_colors,
        source_indices=source_indices,
        labels=segmented_labels,
        point_ids=point_ids[positive],
    )
    temporary_cache.replace(cache_path)

    segmented_cloud = o3d.geometry.PointCloud()
    segmented_cloud.points = o3d.utility.Vector3dVector(segmented_positions.astype(np.float64, copy=False))
    segmented_cloud.colors = o3d.utility.Vector3dVector(segmented_colors.astype(np.float64) / 255.0)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_ply = ply_path.with_name(ply_path.stem + ".tmp.ply")
    if not o3d.io.write_point_cloud(
        str(temporary_ply),
        segmented_cloud,
        write_ascii=False,
        compressed=True,
        print_progress=False,
    ):
        temporary_ply.unlink(missing_ok=True)
        raise RuntimeError(f"Open3D could not write segmented COLMAP points: {temporary_ply}")
    temporary_ply.replace(ply_path)

    region_by_id = {int(row["id"]): row for row in region_rows if int(row.get("id", 0)) > 0}
    unique, counts = np.unique(segmented_labels, return_counts=True)
    summary = {
        "schemaVersion": 1,
        "sourcePath": str(source_path),
        "inputPointCount": int(positions.shape[0]),
        "labeledPointCount": int(segmented_positions.shape[0]),
        "unlabeledPointCount": int(positions.shape[0] - segmented_positions.shape[0]),
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


def _clear_segmented_colmap_outputs(config: ColmapTrackConfig) -> None:
    for path in (
        config.segmented_cache_path,
        config.segmented_ply_path,
        config.segmented_summary_path,
    ):
        if path is not None:
            path.expanduser().resolve().unlink(missing_ok=True)


def _read_reconstruction(model_path: Path):
    available, reason = colmap_track_availability(ColmapTrackConfig(model_path=model_path))
    if not available:
        raise FileNotFoundError(reason)
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path.expanduser().resolve()))
    if not reconstruction.images or not reconstruction.points3D:
        raise ValueError(f"COLMAP model is empty: {model_path}")
    if sum(int(image.num_points3D) for image in reconstruction.images.values()) <= 0:
        raise ValueError(
            f"COLMAP model contains no feature-track observations: {model_path}. "
            "Use the original SfM model, not OMeGa's trackless seed point cloud."
        )
    return reconstruction


def _match_colmap_images(reconstruction, frames: list[PropagationFrame]) -> dict[int, Any]:
    by_basename: dict[str, list[Any]] = {}
    for image in reconstruction.images.values():
        by_basename.setdefault(Path(str(image.name)).name.casefold(), []).append(image)

    matched: dict[int, Any] = {}
    missing: list[str] = []
    for frame in frames:
        image_name = str(frame.image_name or frame.image_path.name)
        candidates = by_basename.get(Path(image_name).name.casefold(), [])
        if len(candidates) != 1:
            missing.append(f"{image_name} ({len(candidates)} matches)")
            continue
        matched[int(frame.frame_id)] = candidates[0]
    if missing:
        sample = ", ".join(missing[:8])
        raise ValueError(f"Could not uniquely align editor frames to COLMAP images: {sample}")
    return matched


@dataclass(frozen=True)
class _PinholePixelTransform:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    input_camera_model: str
    input_camera_params: tuple[float, ...]
    input_distortion: tuple[float, ...]
    output_camera_params: tuple[float, float, float, float]
    rotation: str
    pre_rotation_width: int
    pre_rotation_height: int


def _read_pixel_transforms(path: Path | None) -> dict[str, _PinholePixelTransform]:
    if path is None:
        return {}
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    transforms: dict[str, _PinholePixelTransform] = {}
    for row in payload.get("perFrame", []):
        if not isinstance(row, dict):
            continue
        image_name = Path(str(row.get("outputImagePath") or row.get("sourceImagePath") or "")).name.casefold()
        pixel_rotation = row.get("pixelRotation", {})
        output_params = tuple(float(value) for value in row.get("outputCameraParams", []))
        if not image_name or len(output_params) != 4 or not isinstance(pixel_rotation, dict):
            continue
        transforms[image_name] = _PinholePixelTransform(
            source_width=int(row["sourceWidth"]),
            source_height=int(row["sourceHeight"]),
            output_width=int(row["outputWidth"]),
            output_height=int(row["outputHeight"]),
            input_camera_model=str(row["inputCameraModel"]),
            input_camera_params=tuple(float(value) for value in row["inputCameraParams"]),
            input_distortion=tuple(float(value) for value in row["inputDistortionCoeffsOpenCv"]),
            output_camera_params=(output_params[0], output_params[1], output_params[2], output_params[3]),
            rotation=str(pixel_rotation.get("rotation", "none")).strip().lower(),
            pre_rotation_width=int(pixel_rotation["preRotationWidth"]),
            pre_rotation_height=int(pixel_rotation["preRotationHeight"]),
        )
    if not transforms:
        raise ValueError(f"Pinhole export summary contains no usable per-frame pixel transforms: {path}")
    return transforms


def _map_observation_pixels(
    image,
    frame: PropagationFrame,
    coordinates: np.ndarray,
    transform: _PinholePixelTransform | None,
) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    if transform is None:
        scale = np.asarray(
            [float(frame.width) / float(image.camera.width), float(frame.height) / float(image.camera.height)],
            dtype=np.float64,
        )
        return coordinates * scale[None, :]

    if (int(image.camera.width), int(image.camera.height)) != (
        transform.source_width,
        transform.source_height,
    ):
        raise ValueError(
            f"COLMAP camera grid {image.camera.width}x{image.camera.height} does not match the recorded "
            f"pinhole source grid {transform.source_width}x{transform.source_height} for {frame.image_name}."
        )

    import cv2

    input_k = _input_camera_matrix(transform.input_camera_model, transform.input_camera_params)
    pre_rotation_k = _pre_rotation_camera_matrix(transform)
    distortion = np.asarray(transform.input_distortion, dtype=np.float64)
    undistorted = cv2.undistortPoints(
        coordinates.reshape(-1, 1, 2),
        input_k,
        distortion,
        P=pre_rotation_k,
    ).reshape(-1, 2)
    rotated = _rotate_pixels(
        undistorted,
        rotation=transform.rotation,
        width=transform.pre_rotation_width,
        height=transform.pre_rotation_height,
    )
    scale = np.asarray(
        [float(frame.width) / float(transform.output_width), float(frame.height) / float(transform.output_height)],
        dtype=np.float64,
    )
    return rotated * scale[None, :]


def _input_camera_matrix(model: str, params: tuple[float, ...]) -> np.ndarray:
    model_key = str(model).strip().upper()
    if model_key in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"} and len(params) >= 3:
        fx = fy = float(params[0])
        cx, cy = float(params[1]), float(params[2])
    elif len(params) >= 4:
        fx, fy, cx, cy = (float(params[index]) for index in range(4))
    else:
        raise ValueError(f"Unsupported COLMAP camera parameters for {model}: {params}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _pre_rotation_camera_matrix(transform: _PinholePixelTransform) -> np.ndarray:
    out_fx, out_fy, out_cx, out_cy = transform.output_camera_params
    rotation = transform.rotation
    if rotation == "none":
        fx, fy, cx, cy = out_fx, out_fy, out_cx, out_cy
    elif rotation == "180":
        fx, fy = out_fx, out_fy
        cx = float(transform.pre_rotation_width - 1) - out_cx
        cy = float(transform.pre_rotation_height - 1) - out_cy
    elif rotation == "ccw90":
        fx, fy = out_fy, out_fx
        cx = float(transform.pre_rotation_width - 1) - out_cy
        cy = out_cx
    elif rotation == "cw90":
        fx, fy = out_fy, out_fx
        cx = out_cy
        cy = float(transform.pre_rotation_height - 1) - out_cx
    else:
        raise ValueError(f"Unsupported pinhole pixel rotation: {rotation}")
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rotate_pixels(coordinates: np.ndarray, *, rotation: str, width: int, height: int) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    if rotation == "none":
        return coordinates.copy()
    if rotation == "180":
        return np.column_stack((float(width - 1) - x, float(height - 1) - y))
    if rotation == "ccw90":
        return np.column_stack((y, float(width - 1) - x))
    if rotation == "cw90":
        return np.column_stack((float(height - 1) - y, x))
    raise ValueError(f"Unsupported pinhole pixel rotation: {rotation}")


def _valid_point_mask(
    reconstruction,
    *,
    min_track_length: int,
    max_reprojection_error: float,
) -> np.ndarray:
    if min_track_length < 2:
        raise ValueError("COLMAP points need a minimum track length of at least 2.")
    if not np.isfinite(max_reprojection_error) or max_reprojection_error <= 0:
        raise ValueError("COLMAP maximum reprojection error must be positive and finite.")

    max_point_id = max(int(point_id) for point_id in reconstruction.points3D.keys())
    valid = np.zeros(max_point_id + 1, dtype=bool)
    for point_id, point in reconstruction.points3D.items():
        error = float(point.error)
        valid[int(point_id)] = bool(
            np.isfinite(error)
            and error <= max_reprojection_error
            and int(point.track.length()) >= min_track_length
        )
    return valid


def _label_points_from_anchors(
    sources: list[LabeledPropagationSource],
    frames: list[PropagationFrame],
    image_by_frame_id: dict[int, Any],
    valid_points: np.ndarray,
    pixel_transforms: dict[str, "_PinholePixelTransform"] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    point_labels, _, summary = _label_points_with_confidence_from_anchors(
        sources,
        frames,
        image_by_frame_id,
        valid_points,
        pixel_transforms,
    )
    return point_labels, summary


def _label_points_with_confidence_from_anchors(
    sources: list[LabeledPropagationSource],
    frames: list[PropagationFrame],
    image_by_frame_id: dict[int, Any],
    valid_points: np.ndarray,
    pixel_transforms: dict[str, "_PinholePixelTransform"] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Label tracks by anchor consensus and retain a soft vote confidence."""
    frame_by_id = {int(frame.frame_id): frame for frame in frames}
    votes: dict[int, Counter[int]] = {}
    positive_observations = 0
    for source in sources:
        frame = frame_by_id[int(source.frame_id)]
        image = image_by_frame_id[int(source.frame_id)]
        source_labels = np.asarray(source.labels, dtype=np.uint16)
        expected = (int(frame.height), int(frame.width))
        if source_labels.shape != expected:
            raise ValueError(
                f"Persistent region map shape {source_labels.shape} does not match frame {frame.frame_id} shape {expected}."
            )
        point_ids, pixel_xy = _scaled_observations(
            image,
            frame,
            valid_points,
            pixel_transforms=pixel_transforms,
        )
        if point_ids.size == 0:
            continue
        observed_labels = source_labels[pixel_xy[:, 1], pixel_xy[:, 0]]
        positive = observed_labels > 0
        positive_observations += int(np.count_nonzero(positive))
        for point_id, region_id in zip(point_ids[positive].tolist(), observed_labels[positive].tolist()):
            votes.setdefault(int(point_id), Counter())[int(region_id)] += 1

    point_labels = np.zeros(valid_points.shape[0], dtype=np.uint16)
    point_confidence = np.zeros(valid_points.shape[0], dtype=np.float32)
    tied_points = 0
    for point_id, counts in votes.items():
        total = int(sum(counts.values()))
        best_count = max(counts.values())
        winners = [region_id for region_id, count in counts.items() if count == best_count]
        if len(winners) == 1:
            point_labels[point_id] = np.uint16(winners[0])
            second_count = max((count for count in counts.values() if count < best_count), default=0)
            agreement = float(best_count / max(total, 1))
            margin = float((best_count - second_count) / max(total, 1))
            support = 0.75 if total == 1 else 1.0
            point_confidence[point_id] = np.float32(
                agreement * (0.5 + 0.5 * margin) * support
            )
        else:
            tied_points += 1
    return point_labels, point_confidence, {
        "positiveAnchorObservations": int(positive_observations),
        "votedPointCount": int(len(votes)),
        "labeledPointCount": int(np.count_nonzero(point_labels)),
        "tiedPointCount": int(tied_points),
    }


def _scaled_observations(
    image,
    frame: PropagationFrame,
    valid_points: np.ndarray,
    *,
    pixel_transforms: dict[str, "_PinholePixelTransform"] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    point_ids: list[int] = []
    coordinates: list[tuple[float, float]] = []
    max_point_id = valid_points.shape[0] - 1
    for point in image.points2D:
        if not point.has_point3D():
            continue
        point_id = int(point.point3D_id)
        if point_id < 0 or point_id > max_point_id or not valid_points[point_id]:
            continue
        xy = getattr(point, "xy", None)
        if xy is not None:
            xy = xy() if callable(xy) else xy
            point_x, point_y = np.asarray(xy, dtype=np.float64).reshape(2).tolist()
        else:
            raw_x = point.x() if callable(point.x) else point.x
            raw_y = point.y() if callable(point.y) else point.y
            point_x, point_y = float(raw_x), float(raw_y)
        point_ids.append(point_id)
        coordinates.append((point_x, point_y))
    if not point_ids:
        return np.empty(0, dtype=np.int64), np.empty((0, 2), dtype=np.int32)
    point_ids_array = np.asarray(point_ids, dtype=np.int64)
    coordinates_array = np.asarray(coordinates, dtype=np.float64)
    transform = (pixel_transforms or {}).get(Path(str(frame.image_name)).name.casefold())
    mapped = _map_observation_pixels(image, frame, coordinates_array, transform)
    pixels = np.rint(mapped).astype(np.int32)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < int(frame.width))
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < int(frame.height))
    )
    return point_ids_array[inside], pixels[inside]


def _rasterize_frame_observations(
    image,
    frame: PropagationFrame,
    valid_points: np.ndarray,
    point_labels: np.ndarray,
    pixel_transforms: dict[str, "_PinholePixelTransform"] | None = None,
    *,
    radius: int,
) -> tuple[np.ndarray, int]:
    if radius < 0 or radius > 8:
        raise ValueError("Sparse point raster radius must be between 0 and 8 pixels.")
    point_ids, pixel_xy = _scaled_observations(
        image,
        frame,
        valid_points,
        pixel_transforms=pixel_transforms,
    )
    if point_ids.size == 0:
        return np.zeros((int(frame.height), int(frame.width)), dtype=np.uint16), 0
    labels = point_labels[point_ids]
    positive = labels > 0
    centers = pixel_xy[positive]
    center_labels = labels[positive]
    output = _rasterize_exclusive_points(
        centers,
        center_labels,
        width=int(frame.width),
        height=int(frame.height),
        radius=radius,
    )
    return output, int(centers.shape[0])


def _rasterize_exclusive_points(
    pixel_xy: np.ndarray,
    labels: np.ndarray,
    *,
    width: int,
    height: int,
    radius: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint16)
    if pixel_xy.size == 0:
        return output

    offsets = [
        (dx, dy)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= radius * radius
    ]
    flat_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for dx, dy in offsets:
        x = pixel_xy[:, 0] + dx
        y = pixel_xy[:, 1] + dy
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        if not np.any(inside):
            continue
        flat_parts.append(y[inside].astype(np.int64) * width + x[inside].astype(np.int64))
        label_parts.append(np.asarray(labels[inside], dtype=np.uint16))
    if not flat_parts:
        return output

    flat = np.concatenate(flat_parts)
    values = np.concatenate(label_parts)
    order = np.argsort(flat, kind="stable")
    flat = flat[order]
    values = values[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(flat)) + 1))
    minimum = np.minimum.reduceat(values, starts)
    maximum = np.maximum.reduceat(values, starts)
    unambiguous = minimum == maximum
    output.reshape(-1)[flat[starts[unambiguous]]] = minimum[unambiguous]
    return output
