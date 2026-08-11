"""Stage one persistent-mask dataset for both joint static 3DGS methods."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.split_splat.adapter import _colmap_header
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    fingerprint_files,
    now_utc,
    read_complete_stage,
    read_json,
    read_jsonl,
    replace_symlink,
    write_jsonl,
)

from .contract import StaticSemantic3DGSConfig, StaticSemantic3DGSPaths


ProgressCallback = Callable[[str], None]
_UNKNOWN_OBJECT_ID = 255


def prepare_static_semantic_dataset(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Create the common contract and one method-native dataset view."""
    shared = _prepare_shared_dataset(
        config,
        paths,
        editor_paths,
        progress=progress,
    )
    if paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
    if config.method == "segment_then_splat":
        native = _prepare_segment_then_splat(paths, progress=progress)
    elif config.method == "gaussian_grouping":
        native = _prepare_gaussian_grouping(config, paths, progress=progress)
    else:  # Config validation makes this unreachable.
        raise AssertionError(config.method)

    summary = {
        "schemaVersion": 1,
        "stage": "prepare",
        "timestampUtc": now_utc(),
        "method": config.display_name,
        "identityModel": (
            "hard_initializer_ids"
            if config.method == "segment_then_splat"
            else "learned_identity_features"
        ),
        "sourceRunDir": str(paths.source_run_dir),
        "sharedDatasetDir": str(paths.shared_dir),
        "datasetDir": str(paths.dataset_dir),
        "shared": shared,
        "native": native,
    }
    atomic_write_json(paths.stage_summary("prepare"), summary)
    return summary


def _prepare_shared_dataset(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    source_prepare_path = paths.source_run_dir / "01_input" / "stage.json"
    source_split_path = (
        paths.source_run_dir / "02_split" / "clean_masks" / "stage.json"
    )
    source_prepare = _complete_summary(source_prepare_path, "MapAnything prepare")
    source_split = _complete_summary(source_split_path, "MapAnything split")
    source_frame_map = paths.source_run_dir / "01_input" / "frame_map.jsonl"
    source_images = paths.source_run_dir / "01_input" / "dataset" / "images"
    source_masks = (
        paths.source_run_dir / "02_split" / "clean_masks" / "label_maps"
    )
    source_points = (
        paths.source_run_dir
        / "02_split"
        / "point_labels"
        / "segmented_points.npz"
    )
    source_weights = (
        paths.source_run_dir
        / "02_split"
        / "clean_masks"
        / "training_view_weights.json"
    )
    rows = read_jsonl(source_frame_map)
    if not rows:
        raise ValueError(f"MapAnything frame map is empty: {source_frame_map}")

    persistent_ids = sorted(
        int(value)
        for value in source_split.get("instanceIds", [])
        if int(value) > 0
    )
    if not persistent_ids:
        raise ValueError("MapAnything split contains no persistent regions.")
    if len(persistent_ids) > 254:
        raise ValueError(
            "Segment then Splat reserves object ID 255 for unknown points; "
            f"at most 254 persistent regions are supported, found {len(persistent_ids)}."
        )
    persistent_to_class = {
        persistent_id: index + 1
        for index, persistent_id in enumerate(persistent_ids)
    }
    persistent_to_hard = {
        persistent_id: index
        for index, persistent_id in enumerate(persistent_ids)
    }

    mask_paths = [
        source_masks / f"{int(row['frameId']):06d}.npy"
        for row in rows
    ]
    inputs = [
        source_prepare_path,
        source_split_path,
        source_frame_map,
        source_points,
        *mask_paths,
    ]
    if source_weights.is_file():
        inputs.append(source_weights)
    fingerprint = fingerprint_files(
        inputs,
        extra={
            "mapanythingRunId": config.mapanything_run_id,
            "persistentIds": persistent_ids,
        },
    )
    cached = read_complete_stage(paths.shared_summary)
    if cached is not None and cached.get("fingerprint") == fingerprint:
        return {**cached, "cacheHit": True}

    if paths.shared_dir.exists():
        shutil.rmtree(paths.shared_dir)
    paths.shared_images.mkdir(parents=True)
    paths.shared_masks.mkdir(parents=True)
    paths.shared_mask_pngs.mkdir(parents=True)
    paths.shared_sparse.mkdir(parents=True)

    camera_lines = _colmap_header("Camera", len(rows))
    image_lines = _colmap_header("Image", len(rows))
    staged_rows: list[dict[str, Any]] = []
    observed_persistent_ids: set[int] = set()
    for index, row in enumerate(rows):
        frame_id = int(row["frameId"])
        image_name = f"{frame_id:06d}.jpg"
        source_image = source_images / str(row["imageName"])
        source_mask = source_masks / f"{frame_id:06d}.npy"
        if not source_image.is_file():
            raise FileNotFoundError(f"MapAnything RGB is missing: {source_image}")
        if not source_mask.is_file():
            raise FileNotFoundError(f"MapAnything mask is missing: {source_mask}")
        replace_symlink(paths.shared_images / image_name, source_image)

        labels = np.asarray(
            np.load(source_mask, allow_pickle=False),
            dtype=np.int32,
        )
        expected_shape = (int(row["height"]), int(row["width"]))
        if labels.shape != expected_shape:
            raise ValueError(
                f"Frame {frame_id} mask shape {labels.shape} does not match "
                f"camera shape {expected_shape}."
            )
        unknown = sorted(
            int(value)
            for value in np.unique(labels)
            if int(value) > 0 and int(value) not in persistent_to_class
        )
        if unknown:
            raise ValueError(
                f"Frame {frame_id} uses unknown persistent IDs: {unknown}"
            )
        observed_persistent_ids.update(
            int(value) for value in np.unique(labels) if int(value) > 0
        )
        class_labels = _remap_labels(labels, persistent_to_class)
        np.save(paths.shared_masks / f"{frame_id:06d}.npy", class_labels)
        Image.fromarray(class_labels, mode="L").save(
            paths.shared_mask_pngs / f"{frame_id:06d}.png"
        )

        camera_id = int(row["colmapImageId"])
        camera_lines.append(
            f"{camera_id} PINHOLE {int(row['width'])} {int(row['height'])} "
            f"{float(row['fx']):.17g} {float(row['fy']):.17g} "
            f"{float(row['cx']):.17g} {float(row['cy']):.17g}"
        )
        image_lines.extend(
            [
                f"{camera_id} "
                f"{' '.join(f'{float(value):.17g}' for value in row['qvec'])} "
                f"{' '.join(f'{float(value):.17g}' for value in row['tvec'])} "
                f"{camera_id} {image_name}",
                "",
            ]
        )
        staged_rows.append(
            {
                **row,
                "semanticImageName": image_name,
                "semanticMaskName": f"{frame_id:06d}.png",
            }
        )
        _report(progress, "Prepared shared semantic frames", index + 1, len(rows))

    (paths.shared_sparse / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )
    (paths.shared_sparse / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths.shared_frame_map, staged_rows)

    with np.load(source_points, allow_pickle=False) as cache:
        points = np.asarray(cache["points"], dtype=np.float32)
        colors = np.asarray(cache["source_colors"], dtype=np.uint8)
        point_regions = np.asarray(cache["labels"], dtype=np.int32)
    if not (points.shape[0] == colors.shape[0] == point_regions.shape[0]):
        raise ValueError("MapAnything point-cache arrays have inconsistent sizes.")
    point_unknown = sorted(
        int(value)
        for value in np.unique(point_regions)
        if int(value) > 0 and int(value) not in persistent_to_class
    )
    if point_unknown:
        raise ValueError(
            f"Initializer points use unknown persistent IDs: {point_unknown}"
        )
    _write_standard_point_ply(paths.shared_initializer, points, colors)
    np.save(paths.shared_initializer_labels, point_regions)

    region_metadata = _region_metadata(editor_paths)
    region_rows = [
        {
            "persistentRegionId": persistent_id,
            "classId": persistent_to_class[persistent_id],
            "hardObjectId": persistent_to_hard[persistent_id],
            "name": region_metadata.get(persistent_id, {}).get(
                "name",
                f"Region {persistent_id}",
            ),
            "color": region_metadata.get(persistent_id, {}).get("color"),
        }
        for persistent_id in persistent_ids
    ]
    atomic_write_json(
        paths.shared_region_map,
        {
            "schemaVersion": 1,
            "backgroundClassId": 0,
            "unknownHardObjectId": _UNKNOWN_OBJECT_ID,
            "regions": region_rows,
        },
    )
    if source_weights.is_file():
        replace_symlink(paths.shared_training_weights, source_weights)
    else:
        atomic_write_json(paths.shared_training_weights, {"frames": {}})

    summary = {
        "schemaVersion": 1,
        "stage": "shared_prepare",
        "status": "complete",
        "timestampUtc": now_utc(),
        "sourceRunDir": str(paths.source_run_dir),
        "frameCount": len(rows),
        "regionCount": len(region_rows),
        "persistentRegionIds": persistent_ids,
        "observedPersistentRegionIds": sorted(observed_persistent_ids),
        "pointCount": int(points.shape[0]),
        "labeledPointCount": int(np.count_nonzero(point_regions)),
        "images": str(paths.shared_images),
        "masks": str(paths.shared_masks),
        "initializer": str(paths.shared_initializer),
        "initializerRegionIds": str(paths.shared_initializer_labels),
        "regionMap": str(paths.shared_region_map),
        "trainingViewWeights": str(paths.shared_training_weights),
        "fingerprint": fingerprint,
    }
    atomic_write_json(paths.shared_summary, summary)
    return summary


def _prepare_segment_then_splat(
    paths: StaticSemantic3DGSPaths,
    *,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    replace_symlink(paths.dataset_dir / "images", paths.shared_images)
    sparse = paths.dataset_dir / "sparse" / "0"
    sparse.mkdir(parents=True)
    replace_symlink(sparse / "cameras.txt", paths.shared_sparse / "cameras.txt")
    replace_symlink(sparse / "images.txt", paths.shared_sparse / "images.txt")

    region_map = read_json(paths.shared_region_map)
    regions = list(region_map["regions"])
    persistent_to_hard = {
        int(row["persistentRegionId"]): int(row["hardObjectId"])
        for row in regions
    }
    points, colors = _read_standard_point_ply(paths.shared_initializer)
    persistent_labels = np.asarray(
        np.load(paths.shared_initializer_labels, allow_pickle=False),
        dtype=np.int32,
    )
    hard_labels = np.full(
        persistent_labels.shape,
        _UNKNOWN_OBJECT_ID,
        dtype=np.uint8,
    )
    for persistent_id, hard_id in persistent_to_hard.items():
        hard_labels[persistent_labels == persistent_id] = hard_id
    _write_segment_then_splat_ply(sparse / "points3D.ply", points, colors, hard_labels)

    default_dir = paths.dataset_dir / "multiview_masks_default_merged"
    middle_dir = paths.dataset_dir / "multiview_masks_middle_merged"
    small_dir = paths.dataset_dir / "multiview_masks_small_merged"
    middle_dir.mkdir(parents=True)
    small_dir.mkdir(parents=True)
    frame_rows = read_jsonl(paths.shared_frame_map)
    for object_index, region in enumerate(regions):
        class_id = int(region["classId"])
        object_dir = default_dir / f"{object_index:03d}"
        object_dir.mkdir(parents=True)
        for row in frame_rows:
            frame_id = int(row["frameId"])
            labels = np.asarray(
                np.load(
                    paths.shared_masks / f"{frame_id:06d}.npy",
                    allow_pickle=False,
                ),
                dtype=np.uint8,
            )
            mask = np.where(labels == class_id, 255, 0).astype(np.uint8)
            Image.fromarray(mask, mode="L").save(
                object_dir / f"{frame_id:06d}.jpg",
                quality=100,
                subsampling=0,
            )
        if progress is not None:
            progress(
                "Prepared Segment then Splat object masks "
                f"{object_index + 1}/{len(regions)}."
            )
    image_names = [str(row["semanticImageName"]) for row in frame_rows]
    (paths.dataset_dir / "train.txt").write_text(
        "\n".join(image_names) + "\n",
        encoding="utf-8",
    )
    (paths.dataset_dir / "test.txt").write_text("", encoding="utf-8")
    replace_symlink(paths.dataset_dir / "region_id_map.json", paths.shared_region_map)
    return {
        "format": "released_segment_then_splat_single_granularity",
        "frameCount": len(frame_rows),
        "objectCount": len(regions),
        "hardLabeledPointCount": int(np.count_nonzero(hard_labels != 255)),
        "unknownPointCount": int(np.count_nonzero(hard_labels == 255)),
        "activeGranularity": "default",
        "inactiveGranularities": ["middle", "small"],
    }


def _prepare_gaussian_grouping(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
    *,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    replace_symlink(paths.dataset_dir / "images", paths.shared_images)
    sparse = paths.dataset_dir / "sparse" / "0"
    sparse.mkdir(parents=True)
    for name in ("cameras.txt", "images.txt", "points3D.ply"):
        replace_symlink(sparse / name, paths.shared_sparse / name)
    object_masks = paths.dataset_dir / "object_mask"
    object_masks.mkdir(parents=True)
    frame_rows = read_jsonl(paths.shared_frame_map)
    for index, row in enumerate(frame_rows):
        frame_id = int(row["frameId"])
        replace_symlink(
            object_masks / f"{frame_id:06d}.png",
            paths.shared_mask_pngs / f"{frame_id:06d}.png",
        )
        _report(progress, "Prepared Gaussian Grouping masks", index + 1, len(frame_rows))
    replace_symlink(paths.dataset_dir / "region_id_map.json", paths.shared_region_map)

    region_count = len(read_json(paths.shared_region_map)["regions"])
    config_payload = {
        "densify_until_iter": int(config.densify_until_iter),
        "num_classes": region_count + 1,
        "reg3d_interval": int(config.reg3d_interval),
        "reg3d_k": int(config.reg3d_k),
        "reg3d_lambda_val": float(config.reg3d_lambda),
        "reg3d_max_points": int(config.reg3d_max_points),
        "reg3d_sample_size": int(config.reg3d_sample_size),
    }
    atomic_write_json(paths.dataset_dir / "train_config.json", config_payload)
    return {
        "format": "released_gaussian_grouping_indexed_masks",
        "frameCount": len(frame_rows),
        "classCount": region_count + 1,
        "foregroundRegionCount": region_count,
        "config": config_payload,
        "initializerIdentity": "random_16d_as_released",
    }


def _complete_summary(path: Path, label: str) -> dict[str, Any]:
    summary = read_complete_stage(path)
    if summary is None:
        raise RuntimeError(f"{label} is not complete: {path}")
    return summary


def _region_metadata(editor_paths: EditorPaths) -> dict[int, dict[str, Any]]:
    if not editor_paths.regions_summary.is_file():
        return {}
    payload = read_json(editor_paths.regions_summary)
    return {
        int(row["id"]): dict(row)
        for row in payload.get("regions", [])
        if int(row.get("id", 0)) > 0
    }


def _remap_labels(
    labels: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    result = np.zeros(labels.shape, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        result[labels == source_id] = target_id
    return result


def _write_standard_point_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["nx"] = 0
    vertices["ny"] = 0
    vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _read_standard_point_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices = PlyData.read(path)["vertex"]
    points = np.column_stack(
        [np.asarray(vertices[name]) for name in ("x", "y", "z")]
    ).astype(np.float32)
    colors = np.column_stack(
        [np.asarray(vertices[name]) for name in ("red", "green", "blue")]
    ).astype(np.uint8)
    return points, colors


def _write_segment_then_splat_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    hard_labels: np.ndarray,
) -> None:
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("obj_id_default", "u1"),
            ("obj_id_middle", "u1"),
            ("obj_id_small", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["nx"] = 0
    vertices["ny"] = 0
    vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["obj_id_default"] = hard_labels
    vertices["obj_id_middle"] = _UNKNOWN_OBJECT_ID
    vertices["obj_id_small"] = _UNKNOWN_OBJECT_ID
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _report(
    callback: ProgressCallback | None,
    label: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None and (
        completed == 1 or completed % 20 == 0 or completed == total
    ):
        callback(f"{label} {completed}/{total}.")
