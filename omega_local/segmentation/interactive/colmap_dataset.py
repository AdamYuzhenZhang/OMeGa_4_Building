"""Normalize a native COLMAP capture into the interactive editor contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_SCHEMA_VERSION = 3
_DISPLAY_MAX_REPROJECTION_ERROR = 4.0
_DISPLAY_MIN_TRACK_LENGTH = 2
_DISPLAY_DISTANCE_QUANTILE = 0.99
_FRAME_NUMBER = re.compile(r"^(?P<stream>.+)_f(?P<frame>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class PreparedColmapDataset:
    root_dir: Path
    baseline_name: str
    baseline_dir: Path
    dataset_dir: Path
    colmap_model: Path
    point_cloud: Path
    frame_count: int


def prepare_colmap_editor_dataset(
    root_dir: Path,
    *,
    baseline_name: str,
    square_images_only: bool,
) -> PreparedColmapDataset:
    """Stage poses and points while referencing source RGB files in place."""

    root_dir = root_dir.expanduser().resolve()
    image_dir = root_dir / "images"
    colmap_model = root_dir / "sparse"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"COLMAP image directory does not exist: {image_dir}")
    if not colmap_model.is_dir():
        raise FileNotFoundError(f"COLMAP sparse model does not exist: {colmap_model}")

    baseline_dir = root_dir / "segmentation" / "baselines" / baseline_name
    dataset_dir = baseline_dir / "dataset"
    scan_dir = dataset_dir / "scans" / root_dir.name
    manifest_path = dataset_dir / "frame_manifest.jsonl"
    summary_path = dataset_dir / "dataset_summary.json"
    points_path = scan_dir / "points.pts"
    point_cloud = scan_dir / "colmap_sparse_points.ply"

    source_signature = _source_signature(
        root_dir,
        colmap_model,
        square_images_only=square_images_only,
    )
    current = _read_json(summary_path)
    if (
        current is not None
        and current.get("sourceSignature") == source_signature
        and manifest_path.is_file()
        and points_path.is_file()
        and point_cloud.is_file()
    ):
        return PreparedColmapDataset(
            root_dir=root_dir,
            baseline_name=baseline_name,
            baseline_dir=baseline_dir,
            dataset_dir=dataset_dir,
            colmap_model=colmap_model,
            point_cloud=point_cloud,
            frame_count=int(current.get("frameCount", 0)),
        )

    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("Preparing a native COLMAP editor dataset requires pycolmap.") from exc

    reconstruction = pycolmap.Reconstruction(str(colmap_model))
    if not reconstruction.images or not reconstruction.points3D:
        raise ValueError(f"COLMAP reconstruction is empty: {colmap_model}")

    image_rows = _selected_images(
        reconstruction,
        image_dir=image_dir,
        square_images_only=square_images_only,
    )
    if not image_rows:
        qualifier = " square" if square_images_only else ""
        raise ValueError(f"COLMAP reconstruction contains no usable{qualifier} images.")

    pose_dir = dataset_dir / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    scan_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for frame_id, image in enumerate(image_rows):
        camera = image.camera
        calibration = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        world_from_camera = _world_from_camera(image)
        pose_path = pose_dir / f"{frame_id:06d}.txt"
        np.savetxt(pose_path, world_from_camera, fmt="%.10f")
        source_image = image_dir / str(image.name)
        manifest_rows.append(
            {
                "sai3dFrameId": int(frame_id),
                "sourceFrameId": int(image.image_id),
                "colmapImageId": int(image.image_id),
                "scanID": root_dir.name,
                "frameID": int(frame_id),
                "imageName": str(image.name),
                "sourceImagePath": str(source_image),
                "colorPath": "",
                "posePath": pose_path.relative_to(dataset_dir).as_posix(),
                "width": int(camera.width),
                "height": int(camera.height),
                "fx": float(calibration[0, 0]),
                "fy": float(calibration[1, 1]),
                "cx": float(calibration[0, 2]),
                "cy": float(calibration[1, 2]),
                "cameraModel": str(camera.model_name),
            }
        )

    point_ids = np.asarray(
        sorted(int(point_id) for point_id in reconstruction.points3D),
        dtype=np.int64,
    )
    positions = np.asarray(
        [reconstruction.points3D[int(point_id)].xyz for point_id in point_ids],
        dtype=np.float64,
    )
    colors = np.asarray(
        [reconstruction.points3D[int(point_id)].color for point_id in point_ids],
        dtype=np.uint8,
    )
    errors = np.asarray(
        [reconstruction.points3D[int(point_id)].error for point_id in point_ids],
        dtype=np.float64,
    )
    track_lengths = np.asarray(
        [len(reconstruction.points3D[int(point_id)].track.elements) for point_id in point_ids],
        dtype=np.int32,
    )
    finite = np.all(np.isfinite(positions), axis=1)
    if not np.all(finite):
        raise ValueError(
            "The COLMAP model contains non-finite points. Point-ID order must remain exact, "
            "so repair the source reconstruction before staging it."
        )

    display_mask, display_filter = _display_point_mask(
        positions,
        errors,
        track_lengths,
    )
    display_positions = positions[display_mask]

    _write_jsonl_atomic(manifest_path, manifest_rows)
    _write_points_atomic(points_path, display_positions)
    _write_binary_ply_atomic(point_cloud, positions, colors)
    _write_json_atomic(
        summary_path,
        {
            "schemaVersion": _SCHEMA_VERSION,
            "stage": "interactive_native_colmap_dataset",
            "sourceRoot": str(root_dir),
            "sourceSignature": source_signature,
            "frameCount": len(manifest_rows),
            "pointCount": int(positions.shape[0]),
            "displayPointCount": int(display_positions.shape[0]),
            "displayPointFilter": display_filter,
            "squareImagesOnly": bool(square_images_only),
            "imageOrdering": "camera_stream_then_numeric_frame",
            "poseConvention": "world_from_opencv_camera",
            "sourceImagesReferencedInPlace": True,
            "frameManifest": str(manifest_path),
            "points": str(points_path),
            "coloredPointCloud": str(point_cloud),
        },
    )
    return PreparedColmapDataset(
        root_dir=root_dir,
        baseline_name=baseline_name,
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        colmap_model=colmap_model,
        point_cloud=point_cloud,
        frame_count=len(manifest_rows),
    )


def _selected_images(
    reconstruction: Any,
    *,
    image_dir: Path,
    square_images_only: bool,
) -> list[Any]:
    rows = []
    for image in reconstruction.images.values():
        source = image_dir / str(image.name)
        if not source.is_file():
            continue
        camera = image.camera
        if square_images_only and int(camera.width) != int(camera.height):
            continue
        rows.append(image)
    rows.sort(key=lambda image: _image_sort_key(str(image.name)))
    return rows


def _image_sort_key(name: str) -> tuple[str, int, str]:
    stem = Path(name).stem
    match = _FRAME_NUMBER.match(stem)
    if match is None:
        return stem.casefold(), -1, name.casefold()
    return (
        match.group("stream").casefold(),
        int(match.group("frame")),
        name.casefold(),
    )


def _world_from_camera(image: Any) -> np.ndarray:
    camera_from_world = np.eye(4, dtype=np.float64)
    camera_from_world[:3, :] = np.asarray(
        image.cam_from_world().matrix(),
        dtype=np.float64,
    ).reshape(3, 4)
    return np.linalg.inv(camera_from_world)


def _display_point_mask(
    positions: np.ndarray,
    errors: np.ndarray,
    track_lengths: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    quality = (
        np.all(np.isfinite(positions), axis=1)
        & np.isfinite(errors)
        & (errors <= _DISPLAY_MAX_REPROJECTION_ERROR)
        & (track_lengths >= _DISPLAY_MIN_TRACK_LENGTH)
    )
    if not np.any(quality):
        raise ValueError("COLMAP has no display points after track/error filtering.")
    center = np.median(positions[quality], axis=0)
    distance = np.linalg.norm(positions - center[None, :], axis=1)
    distance_limit = float(np.quantile(distance[quality], _DISPLAY_DISTANCE_QUANTILE))
    selected = quality & (distance <= distance_limit)
    return selected, {
        "method": "track_error_then_robust_distance_quantile",
        "maxReprojectionError": _DISPLAY_MAX_REPROJECTION_ERROR,
        "minTrackLength": _DISPLAY_MIN_TRACK_LENGTH,
        "distanceCenter": center.astype(float).tolist(),
        "distanceQuantile": _DISPLAY_DISTANCE_QUANTILE,
        "distanceLimit": distance_limit,
        "qualityPointCount": int(np.count_nonzero(quality)),
        "selectedPointCount": int(np.count_nonzero(selected)),
    }


def _source_signature(
    root_dir: Path,
    colmap_model: Path,
    *,
    square_images_only: bool,
) -> dict[str, Any]:
    files = []
    for name in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
        path = colmap_model / name
        if path.is_file():
            stat = path.stat()
            files.append(
                {
                    "name": name,
                    "size": int(stat.st_size),
                    "mtimeNs": int(stat.st_mtime_ns),
                }
            )
    return {
        "schemaVersion": _SCHEMA_VERSION,
        "root": str(root_dir),
        "modelFiles": files,
        "squareImagesOnly": bool(square_images_only),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _write_points_atomic(path: Path, positions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    np.savetxt(temporary, positions, fmt="%.8f")
    temporary.replace(path)


def _write_binary_ply_atomic(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.ply")
    vertices = np.empty(
        positions.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = positions[:, 0]
    vertices["y"] = positions[:, 1]
    vertices["z"] = positions[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {positions.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with temporary.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)
    temporary.replace(path)
