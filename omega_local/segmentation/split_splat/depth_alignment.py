"""Align monocular depth to the global Split&Splat reconstruction."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .adapter import _read_colmap_images_text
from .contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    replace_symlink,
)


ProgressCallback = Callable[[str], None]


def stage_split_depth(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    workspace: Path,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stage the depth maps used by released point visibility projection.

    Murre depth is expected to share the camera coordinate scale and is left
    untouched. Depth Anything is monocular, so its inverse depth is calibrated
    with the median/deviation affine fit shipped in ``make_depth_scale.py``.
    Projected global Gaussian means replace unavailable COLMAP feature tracks.
    """
    target = workspace / "data" / paths.run_id / "depth"
    if config.depth_source != "editor_depth":
        return {
            "mode": "native_metric_depth",
            "depthDir": str(target),
            "calibrated": False,
        }

    cache_dir = paths.global_gs_dir / "split_depth_editor_depth"
    summary_path = cache_dir / "summary.json"
    source_dir = paths.dataset_dir / "depth"
    cache_key = _cache_key(paths, source_dir)
    summary = _read_json(summary_path)
    if summary.get("cacheKey") != cache_key:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        summary = _build_alignment_cache(
            paths,
            source_dir=source_dir,
            images_path=paths.dataset_dir / "sparse" / "0" / "images.txt",
            cameras_path=paths.dataset_dir / "sparse" / "0" / "cameras.txt",
            output_dir=cache_dir,
            cache_key=cache_key,
            progress=progress,
        )
    elif progress is not None:
        progress(
            "Reusing calibrated Depth Anything visibility maps "
            f"({summary.get('frameCount', 0)} frames)."
        )

    replace_symlink(target, cache_dir)
    return summary


def _build_alignment_cache(
    paths: SplitSplatRunPaths,
    *,
    source_dir: Path,
    images_path: Path,
    cameras_path: Path,
    output_dir: Path,
    cache_key: dict[str, Any],
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(paths.global_points_cache) as archive:
        points = np.asarray(archive["points"], dtype=np.float32)
    points = points[np.all(np.isfinite(points), axis=1)]
    images = sorted(
        _read_colmap_images_text(images_path),
        key=lambda row: int(row["image_id"]),
    )
    cameras = _read_pinhole_cameras(cameras_path)
    frame_summaries: list[dict[str, Any]] = []

    if progress is not None:
        progress(
            "Calibrating Depth Anything visibility to "
            f"{points.shape[0]:,} global Gaussian means."
        )

    for index, image in enumerate(images):
        stem = Path(str(image["name"])).stem
        source_npy = source_dir / f"{stem}_pred.npy"
        source_png = source_dir / f"{stem}.png"
        if not source_npy.is_file():
            raise FileNotFoundError(f"Split depth map is missing: {source_npy}")
        raw_depth = np.asarray(np.load(source_npy), dtype=np.float32)
        camera = cameras[int(image["camera_id"])]
        expected_shape = (camera["height"], camera["width"])
        if raw_depth.shape != expected_shape:
            raise ValueError(
                f"Depth/camera shape mismatch for {stem}: "
                f"{raw_depth.shape} != {expected_shape}"
            )

        reference_depth = _project_z_buffer(points, image, camera)
        scale, offset, support = _fit_inverse_depth_affine(
            raw_depth,
            reference_depth,
        )
        aligned_depth = _apply_inverse_depth_affine(raw_depth, scale, offset)
        np.save(output_dir / f"{stem}_pred.npy", aligned_depth)
        if source_png.is_file():
            replace_symlink(output_dir / source_png.name, source_png)

        valid = aligned_depth > 0
        frame_summaries.append(
            {
                "imageName": str(image["name"]),
                "supportPixels": int(support),
                "scale": float(scale),
                "offset": float(offset),
                "validPixels": int(np.count_nonzero(valid)),
                "medianDepth": (
                    float(np.median(aligned_depth[valid]))
                    if np.any(valid)
                    else None
                ),
            }
        )
        completed = index + 1
        if progress is not None and (
            completed == 1
            or completed % 10 == 0
            or completed == len(images)
        ):
            progress(f"Calibrated split visibility depth {completed}/{len(images)}.")

    summary = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "mode": "global_gaussian_inverse_depth_affine",
        "calibrated": True,
        "method": (
            "Split&Splat median/MAD inverse-depth alignment using projected "
            "global Gaussian means"
        ),
        "frameCount": len(frame_summaries),
        "pointCount": int(points.shape[0]),
        "depthDir": str(output_dir),
        "cacheKey": cache_key,
        "frames": frame_summaries,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    return summary


def _project_z_buffer(
    points: np.ndarray,
    image: dict[str, Any],
    camera: dict[str, Any],
) -> np.ndarray:
    rotation = _qvec_to_rotation(np.asarray(image["qvec"], dtype=np.float64))
    translation = np.asarray(image["tvec"], dtype=np.float64)
    points_camera = points @ rotation.T + translation
    z = points_camera[:, 2]
    safe_z = np.maximum(z, np.float32(1e-8))
    u = np.rint(
        camera["fx"] * points_camera[:, 0] / safe_z + camera["cx"]
    ).astype(np.int64)
    v = np.rint(
        camera["fy"] * points_camera[:, 1] / safe_z + camera["cy"]
    ).astype(np.int64)
    valid = (
        (z > 0)
        & (u >= 0)
        & (u < camera["width"])
        & (v >= 0)
        & (v < camera["height"])
    )
    pixel_ids = v[valid] * camera["width"] + u[valid]
    z_buffer = np.full(
        camera["height"] * camera["width"],
        np.inf,
        dtype=np.float32,
    )
    np.minimum.at(z_buffer, pixel_ids, z[valid].astype(np.float32))
    return z_buffer.reshape(camera["height"], camera["width"])


def _fit_inverse_depth_affine(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
) -> tuple[float, float, int]:
    valid = (
        np.isfinite(predicted_depth)
        & (predicted_depth > 0)
        & np.isfinite(reference_depth)
        & (reference_depth > 0)
    )
    support = int(np.count_nonzero(valid))
    if support < 32:
        raise ValueError(
            "Too few projected global Gaussian samples to align depth: "
            f"{support}"
        )
    predicted_inverse = 1.0 / predicted_depth[valid].astype(np.float64)
    reference_inverse = 1.0 / reference_depth[valid].astype(np.float64)
    predicted_center = float(np.median(predicted_inverse))
    reference_center = float(np.median(reference_inverse))
    predicted_spread = float(
        np.mean(np.abs(predicted_inverse - predicted_center))
    )
    reference_spread = float(
        np.mean(np.abs(reference_inverse - reference_center))
    )
    if predicted_spread <= 1e-12 or reference_spread <= 1e-12:
        raise ValueError("Depth alignment has degenerate inverse-depth spread.")
    scale = reference_spread / predicted_spread
    offset = reference_center - predicted_center * scale
    return float(scale), float(offset), support


def _apply_inverse_depth_affine(
    predicted_depth: np.ndarray,
    scale: float,
    offset: float,
) -> np.ndarray:
    valid = np.isfinite(predicted_depth) & (predicted_depth > 0)
    aligned_inverse = np.zeros(predicted_depth.shape, dtype=np.float32)
    aligned_inverse[valid] = (
        scale / predicted_depth[valid].astype(np.float32) + offset
    )
    output = np.zeros(predicted_depth.shape, dtype=np.float32)
    positive = aligned_inverse > 1e-8
    output[positive] = 1.0 / aligned_inverse[positive]
    return output


def _read_pinhole_cameras(path: Path) -> dict[int, dict[str, Any]]:
    cameras: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) != 8 or tokens[1] != "PINHOLE":
            raise ValueError(f"Expected a PINHOLE camera row: {line}")
        camera_id = int(tokens[0])
        cameras[camera_id] = {
            "width": int(tokens[2]),
            "height": int(tokens[3]),
            "fx": float(tokens[4]),
            "fy": float(tokens[5]),
            "cx": float(tokens[6]),
            "cy": float(tokens[7]),
        }
    if not cameras:
        raise ValueError(f"No PINHOLE cameras found: {path}")
    return cameras


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _cache_key(paths: SplitSplatRunPaths, source_dir: Path) -> dict[str, Any]:
    point_stat = paths.global_points_cache.stat()
    depth_files = sorted(source_dir.glob("*_pred.npy"))
    return {
        "globalPointsSize": int(point_stat.st_size),
        "globalPointsMtimeNs": int(point_stat.st_mtime_ns),
        "depthFileCount": len(depth_files),
        "depthLatestMtimeNs": max(
            (int(path.stat().st_mtime_ns) for path in depth_files),
            default=0,
        ),
    }
def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
