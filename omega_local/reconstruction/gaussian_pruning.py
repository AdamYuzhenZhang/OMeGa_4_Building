"""Conservative semantic support pruning for trained object Gaussians."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter
from plyfile import PlyData, PlyElement

from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
)


DEFAULT_OPACITY_THRESHOLD = 0.005
DEFAULT_MIN_VISIBLE_VIEWS = 3
DEFAULT_MASK_DILATION_PIXELS = 1


@dataclass(frozen=True)
class _Frame:
    image_stem: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    mask_path: Path
    camera_center: np.ndarray


def prune_gaussian_floaters(
    point_cloud: Path,
    *,
    frame_map: Path,
    mask_dir: Path,
    summary_path: Path,
    colmap_images: Path | None = None,
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
    min_visible_views: int = DEFAULT_MIN_VISIBLE_VIEWS,
    mask_dilation_pixels: int = DEFAULT_MASK_DILATION_PIXELS,
) -> dict[str, Any]:
    """Remove low-opacity and repeatedly unsupported object Gaussians in place.

    The semantic rule is deliberately conservative: a Gaussian supported by any
    positive mask view is retained. An unsupported Gaussian is removed only
    when its center is front-visible in several positive-mask views.
    """
    point_cloud = point_cloud.expanduser().resolve()
    frame_map = frame_map.expanduser().resolve()
    mask_dir = mask_dir.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    if summary_path.is_file():
        cached = read_json(summary_path)
        if _matches_pruned_output(point_cloud, cached):
            return {**cached, "cacheHit": True}

    opacity_threshold = float(opacity_threshold)
    min_visible_views = int(min_visible_views)
    mask_dilation_pixels = int(mask_dilation_pixels)
    if not 0.0 <= opacity_threshold < 1.0:
        raise ValueError("Opacity threshold must be in [0, 1).")
    if min_visible_views < 1:
        raise ValueError("Minimum visible views must be positive.")
    if mask_dilation_pixels < 0:
        raise ValueError("Mask dilation pixels cannot be negative.")

    ply = PlyData.read(point_cloud)
    vertices = ply["vertex"].data
    names = set(vertices.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    if not required.issubset(names):
        raise ValueError(
            f"Gaussian PLY is missing fields {sorted(required - names)}: "
            f"{point_cloud}"
        )
    points = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float32, copy=False)
    finite = np.all(np.isfinite(points), axis=1)
    opacity = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))
    frames = _load_mask_frames(
        frame_map,
        mask_dir,
        colmap_images=colmap_images,
    )
    oversized, world_scale_threshold = _world_scale_outliers(
        vertices, names, frames
    )

    visible_count = np.zeros(points.shape[0], dtype=np.uint16)
    support_count = np.zeros(points.shape[0], dtype=np.uint16)
    for frame in frames:
        point_ids, xy, _ = _project_front_surface(points, frame)
        if point_ids.size == 0:
            continue
        mask = _load_mask(
            frame.mask_path,
            expected_shape=(frame.height, frame.width),
            dilation_pixels=mask_dilation_pixels,
        )
        visible_count[point_ids] += 1
        supported = mask[xy[:, 1], xy[:, 0]]
        support_count[point_ids[supported]] += 1

    required_visible = min(min_visible_views, len(frames))
    low_opacity = opacity < opacity_threshold
    unsupported = (
        (visible_count >= required_visible)
        & (support_count == 0)
    )
    observed = visible_count > 0
    support_fraction = np.zeros(points.shape[0], dtype=np.float32)
    support_fraction[observed] = (
        support_count[observed] / visible_count[observed]
    )
    remove = (~finite) | low_opacity | oversized | unsupported
    keep = ~remove
    if not np.any(keep):
        raise RuntimeError(
            f"Floater pruning would remove every Gaussian in {point_cloud}."
        )

    _write_filtered_ply(ply, vertices[keep], point_cloud)
    output_stat = point_cloud.stat()
    summary = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "method": "Conservative Clean-GS-style semantic support pruning",
        "pointCloud": str(point_cloud),
        "inputGaussianCount": int(vertices.shape[0]),
        "keptGaussianCount": int(keep.sum()),
        "removedGaussianCount": int(remove.sum()),
        "removedFraction": float(remove.mean()),
        "outputFile": {
            "sizeBytes": int(output_stat.st_size),
            "modifiedTimeNs": int(output_stat.st_mtime_ns),
        },
        "removalReasons": {
            "nonfinite": int((~finite).sum()),
            "lowOpacity": int(low_opacity.sum()),
            "oversizedWorldScale": int(oversized.sum()),
            "multiViewUnsupported": int(unsupported.sum()),
            "lowOpacityAndUnsupported": int(
                np.logical_and(low_opacity, unsupported).sum()
            ),
        },
        "support": {
            "positiveMaskViewCount": len(frames),
            "minimumFrontVisibleViews": required_visible,
            "supportedInAtLeastOneMaskCount": int((support_count > 0).sum()),
            "neverFrontVisibleCount": int((visible_count == 0).sum()),
            "visibleViewPercentiles": np.percentile(
                visible_count, [0, 50, 95, 100]
            ).astype(float).tolist(),
            "supportViewPercentiles": np.percentile(
                support_count, [0, 50, 95, 100]
            ).astype(float).tolist(),
            "supportFractionPercentiles": (
                np.percentile(support_fraction[observed], [0, 5, 50, 95, 100])
                .astype(float)
                .tolist()
                if np.any(observed)
                else []
            ),
            "outsideMaskInMajorityOfVisibleViewsCount": int(
                np.logical_and(observed, support_fraction < 0.5).sum()
            ),
        },
        "settings": {
            "opacityThreshold": opacity_threshold,
            "minimumFrontVisibleViews": min_visible_views,
            "maskDilationPixels": mask_dilation_pixels,
            "worldScaleThreshold": world_scale_threshold,
            "worldScaleRule": "max(scale) > 0.1 * camera extent",
            "spatialOutlierPruning": False,
            "colorValidation": False,
            "supportedGaussianPolicy": (
                "retain when the center is inside any positive-view mask"
            ),
        },
        "note": (
            "Spatial percentile pruning is intentionally disabled because it "
            "can erase valid thin and sparsely observed architectural details."
        ),
    }
    atomic_write_json(summary_path, summary)
    return summary


def prune_composed_gaussian_floaters(
    point_cloud: Path,
    labels: np.ndarray,
    *,
    frame_map: Path,
    colmap_images: Path,
    summary_path: Path,
    opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prune attributes reintroduced by composition while preserving labels."""
    point_cloud = point_cloud.expanduser().resolve()
    labels = np.asarray(labels)
    ply = PlyData.read(point_cloud)
    vertices = ply["vertex"].data
    if labels.shape != (vertices.shape[0],):
        raise ValueError(
            f"Composition labels {labels.shape} do not match "
            f"{vertices.shape[0]} Gaussians."
        )
    names = set(vertices.dtype.names or ())
    required = {"x", "y", "z", "opacity"}
    if not required.issubset(names):
        raise ValueError(
            f"Gaussian PLY is missing fields {sorted(required - names)}: "
            f"{point_cloud}"
        )
    points = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float32, copy=False)
    finite = np.all(np.isfinite(points), axis=1)
    opacity = _sigmoid(np.asarray(vertices["opacity"], dtype=np.float64))
    camera_centers = _load_camera_centers(frame_map, colmap_images)
    oversized, world_scale_threshold = _world_scale_outliers_from_centers(
        vertices,
        names,
        camera_centers,
    )
    low_opacity = opacity < float(opacity_threshold)
    remove = (~finite) | low_opacity | oversized
    keep = ~remove
    if not np.any(keep):
        raise RuntimeError(
            "Post-composition pruning would remove every Gaussian in "
            f"{point_cloud}."
        )
    _write_filtered_ply(ply, vertices[keep], point_cloud)
    filtered_labels = labels[keep]
    summary = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "method": "Post-composition 3DGS attribute pruning",
        "pointCloud": str(point_cloud),
        "inputGaussianCount": int(vertices.shape[0]),
        "keptGaussianCount": int(keep.sum()),
        "removedGaussianCount": int(remove.sum()),
        "removedFraction": float(remove.mean()),
        "removalReasons": {
            "nonfinite": int((~finite).sum()),
            "lowOpacity": int(low_opacity.sum()),
            "oversizedWorldScale": int(oversized.sum()),
        },
        "settings": {
            "opacityThreshold": float(opacity_threshold),
            "worldScaleThreshold": world_scale_threshold,
            "worldScaleRule": "max(scale) > 0.1 * camera extent",
            "semanticSupportPruning": False,
            "reason": (
                "Semantic support was already applied per region before "
                "composition; this pass removes attributes reintroduced by "
                "opacity reset and merge optimization."
            ),
        },
    }
    atomic_write_json(summary_path, summary)
    return filtered_labels, summary


def _matches_pruned_output(
    point_cloud: Path,
    summary: dict[str, Any],
) -> bool:
    output = summary.get("outputFile")
    if not point_cloud.is_file() or not isinstance(output, dict):
        return False
    stat = point_cloud.stat()
    return (
        output.get("sizeBytes") == stat.st_size
        and output.get("modifiedTimeNs") == stat.st_mtime_ns
    )


def _load_mask_frames(
    frame_map: Path,
    mask_dir: Path,
    *,
    colmap_images: Path | None,
) -> list[_Frame]:
    masks = {path.stem: path for path in mask_dir.glob("*.png")}
    poses = (
        _read_colmap_poses(colmap_images)
        if colmap_images is not None
        else {}
    )
    frames: list[_Frame] = []
    for row in read_jsonl(frame_map):
        image_name = (
            row.get("imageName")
            or row.get("splitSplatImageName")
            or row.get("editorImageName")
        )
        if not image_name:
            raise KeyError(f"Frame row has no image name: {row}")
        stem = Path(str(image_name)).stem
        mask_path = masks.get(stem)
        if mask_path is None:
            continue
        qvec = row.get("qvec")
        tvec = row.get("tvec")
        if qvec is None or tvec is None:
            pose = poses.get(stem)
            if pose is None:
                raise KeyError(
                    f"No COLMAP pose found for mask frame {stem}."
                )
            qvec, tvec = pose
        rotation = _qvec_to_rotation(qvec)
        translation = np.asarray(tvec, dtype=np.float32)
        frames.append(
            _Frame(
                image_stem=stem,
                width=int(row["width"]),
                height=int(row["height"]),
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                rotation=rotation,
                translation=translation,
                mask_path=mask_path,
                camera_center=-(rotation.T @ translation),
            )
        )
    if not frames:
        raise FileNotFoundError(
            f"No masks in {mask_dir} match frames from {frame_map}."
        )
    return frames


def _read_colmap_poses(
    images_txt: Path,
) -> dict[str, tuple[list[float], list[float]]]:
    images_txt = images_txt.expanduser().resolve()
    if not images_txt.is_file():
        raise FileNotFoundError(f"COLMAP image poses are missing: {images_txt}")
    poses: dict[str, tuple[list[float], list[float]]] = {}
    for raw in images_txt.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) < 10 or not Path(fields[9]).suffix:
            continue
        try:
            int(fields[0])
            qvec = [float(value) for value in fields[1:5]]
            tvec = [float(value) for value in fields[5:8]]
            int(fields[8])
        except ValueError:
            continue
        poses[Path(fields[9]).stem] = (qvec, tvec)
    if not poses:
        raise ValueError(f"No image poses could be parsed from {images_txt}.")
    return poses


def _load_mask(
    path: Path,
    *,
    expected_shape: tuple[int, int],
    dilation_pixels: int,
) -> np.ndarray:
    source = Image.open(path)
    image = (
        source.getchannel("A")
        if "A" in source.getbands()
        else source.convert("L")
    )
    if image.size != (expected_shape[1], expected_shape[0]):
        raise ValueError(
            f"Mask {path} has size {image.size}; expected "
            f"{(expected_shape[1], expected_shape[0])}."
        )
    if dilation_pixels:
        image = image.filter(
            ImageFilter.MaxFilter(2 * dilation_pixels + 1)
        )
    return np.asarray(image, dtype=np.uint8) > 0


def _project_front_surface(
    points: np.ndarray,
    frame: _Frame,
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
    valid = np.isfinite(u) & np.isfinite(v)
    point_ids = point_ids[valid]
    z = z[valid]
    xy = np.rint(np.column_stack((u[valid], v[valid]))).astype(np.int64)
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
    starts = np.r_[0, np.flatnonzero(np.diff(linear_sorted)) + 1]
    counts = np.diff(np.r_[starts, linear_sorted.size])
    nearest = z[order][starts]
    tolerance = np.maximum(0.05, nearest * 0.01)
    front = z[order] <= np.repeat(nearest + tolerance, counts)
    rows = order[front]
    return (
        point_ids[rows].astype(np.int64, copy=False),
        xy[rows].astype(np.int32, copy=False),
        z[rows].astype(np.float32, copy=False),
    )


def _world_scale_outliers(
    vertices: np.ndarray,
    names: set[str],
    frames: list[_Frame],
) -> tuple[np.ndarray, float | None]:
    centers = np.stack([frame.camera_center for frame in frames])
    return _world_scale_outliers_from_centers(vertices, names, centers)


def _world_scale_outliers_from_centers(
    vertices: np.ndarray,
    names: set[str],
    centers: np.ndarray,
) -> tuple[np.ndarray, float | None]:
    required = {"scale_0", "scale_1", "scale_2"}
    empty = np.zeros(vertices.shape[0], dtype=bool)
    if not required.issubset(names):
        return empty, None
    extent = float(
        np.max(np.linalg.norm(centers - centers.mean(axis=0), axis=1)) * 1.1
    )
    if not np.isfinite(extent) or extent <= 1.0e-8:
        return empty, None
    threshold = 0.1 * extent
    max_scale = np.exp(
        np.column_stack(
            [vertices[f"scale_{axis}"] for axis in range(3)]
        ).astype(np.float64)
    ).max(axis=1)
    return max_scale > threshold, threshold


def _load_camera_centers(
    frame_map: Path,
    colmap_images: Path,
) -> np.ndarray:
    poses = _read_colmap_poses(colmap_images)
    centers = []
    for row in read_jsonl(frame_map):
        image_name = (
            row.get("imageName")
            or row.get("splitSplatImageName")
            or row.get("editorImageName")
        )
        if not image_name:
            continue
        qvec = row.get("qvec")
        tvec = row.get("tvec")
        if qvec is None or tvec is None:
            pose = poses.get(Path(str(image_name)).stem)
            if pose is None:
                continue
            qvec, tvec = pose
        rotation = _qvec_to_rotation(qvec)
        translation = np.asarray(tvec, dtype=np.float32)
        centers.append(-(rotation.T @ translation))
    if not centers:
        raise ValueError(
            "No camera centers could be resolved for composition."
        )
    return np.stack(centers)


def _qvec_to_rotation(values: Any) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(values, dtype=np.float64).tolist()
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def _write_filtered_ply(
    source: PlyData,
    vertices: np.ndarray,
    destination: Path,
) -> None:
    temporary = destination.with_name(f".{destination.name}.pruning.tmp")
    try:
        PlyData(
            [PlyElement.describe(vertices, "vertex")],
            text=source.text,
            byte_order=source.byte_order,
            comments=source.comments,
            obj_info=source.obj_info,
        ).write(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _empty_projection(
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.float32),
    )
