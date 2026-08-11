"""Adapt a completed MapAnything split to the released ObjectGS contract."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml
from PIL import Image
from plyfile import PlyData, PlyElement

from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.split_splat.adapter import _colmap_header
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    fingerprint_files,
    now_utc,
    read_json,
    read_jsonl,
    replace_symlink,
    write_jsonl,
)

from .contract import ObjectGSConfig, ObjectGSPaths


ProgressCallback = Callable[[str], None]


def prepare_objectgs_dataset(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Write images, indexed masks, cameras, and labeled MapAnything points."""
    source_prepare = _complete_summary(
        paths.source_run_dir / "01_input" / "stage.json",
        "MapAnything prepare",
    )
    source_split = _complete_summary(
        paths.source_run_dir / "02_split" / "clean_masks" / "stage.json",
        "MapAnything split",
    )
    source_frame_map = paths.source_run_dir / "01_input" / "frame_map.jsonl"
    source_images = paths.source_run_dir / "01_input" / "dataset" / "images"
    source_masks = (
        paths.source_run_dir
        / "02_split"
        / "clean_masks"
        / "label_maps"
    )
    source_points = (
        paths.source_run_dir
        / "02_split"
        / "point_labels"
        / "segmented_points.npz"
    )
    source_provenance = (
        paths.source_run_dir
        / "02_split"
        / "point_labels"
        / "provenance.npy"
    )
    rows = read_jsonl(source_frame_map)
    if not rows:
        raise ValueError(f"MapAnything frame map is empty: {source_frame_map}")

    region_ids = sorted(
        int(value)
        for value in source_split.get("instanceIds", [])
        if int(value) > 0
    )
    if not region_ids:
        raise ValueError("MapAnything split contains no persistent regions.")
    if len(region_ids) > 255:
        raise ValueError(
            "ObjectGS stores labels as uint8 and supports at most 255 "
            f"foreground regions; found {len(region_ids)}."
        )
    persistent_to_objectgs = {
        region_id: index + 1
        for index, region_id in enumerate(region_ids)
    }
    objectgs_to_persistent = {
        value: key for key, value in persistent_to_objectgs.items()
    }
    region_names = _region_names(editor_paths.regions_summary)

    if paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
    image_dir = paths.dataset_dir / "images"
    mask_dir = paths.dataset_dir / "object_mask"
    sparse_dir = paths.dataset_dir / "sparse" / "0"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    camera_lines = _colmap_header("Camera", len(rows))
    image_lines = _colmap_header("Image", len(rows))
    staged_rows: list[dict[str, Any]] = []
    input_files = [
        source_frame_map,
        Path(str(source_prepare["pointCloud"])),
        source_points,
        source_provenance,
    ]
    for index, row in enumerate(rows):
        frame_id = int(row["frameId"])
        staged_name = f"{frame_id:06d}.jpg"
        source_image = source_images / str(row["imageName"])
        source_label_map = source_masks / f"{frame_id:06d}.npy"
        if not source_image.is_file():
            raise FileNotFoundError(f"MapAnything RGB is missing: {source_image}")
        if not source_label_map.is_file():
            raise FileNotFoundError(
                f"MapAnything clean label map is missing: {source_label_map}"
            )
        labels = np.asarray(
            np.load(source_label_map, allow_pickle=False),
            dtype=np.int32,
        )
        remapped = _remap_labels(labels, persistent_to_objectgs)
        expected_shape = (int(row["height"]), int(row["width"]))
        if remapped.shape != expected_shape:
            raise ValueError(
                f"Frame {frame_id} label shape {remapped.shape} does not "
                f"match camera shape {expected_shape}."
            )
        replace_symlink(image_dir / staged_name, source_image)
        Image.fromarray(remapped, mode="L").save(
            mask_dir / f"{frame_id:06d}.png"
        )
        input_files.extend([source_image, source_label_map])

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
                f"{camera_id} {staged_name}",
                "",
            ]
        )
        staged_rows.append(
            {
                **row,
                "objectgsImageName": staged_name,
                "objectgsMaskName": f"{frame_id:06d}.png",
            }
        )
        _report_count(progress, "Prepared ObjectGS frames", index + 1, len(rows))

    (sparse_dir / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths.frame_map, staged_rows)

    point_summary = _write_labeled_initializer(
        source_points,
        source_provenance,
        paths.labeled_initializer,
        persistent_to_objectgs,
        stride=config.initializer_stride,
        geometry_completed_as_unknown=(
            config.geometry_completed_as_unknown
        ),
    )
    region_rows = [
        {
            "persistentRegionId": region_id,
            "objectgsLabelId": persistent_to_objectgs[region_id],
            "name": region_names.get(region_id, f"Region {region_id}"),
        }
        for region_id in region_ids
    ]
    atomic_write_json(
        paths.region_map,
        {
            "schemaVersion": 1,
            "backgroundLabelId": 0,
            "regions": region_rows,
            "persistentToObjectGS": {
                str(key): value for key, value in persistent_to_objectgs.items()
            },
            "objectgsToPersistent": {
                str(key): value for key, value in objectgs_to_persistent.items()
            },
        },
    )
    _write_training_config(config, paths)

    summary = {
        "schemaVersion": 1,
        "stage": "prepare",
        "timestampUtc": now_utc(),
        "method": "MapAnything persistent-region split adapted to ObjectGS",
        "sourceRunDir": str(paths.source_run_dir),
        "datasetDir": str(paths.dataset_dir),
        "frameMap": str(paths.frame_map),
        "regionMap": str(paths.region_map),
        "frameCount": len(staged_rows),
        "regionCount": len(region_rows),
        "labeledInitializer": str(paths.labeled_initializer),
        "pointSummary": point_summary,
        "trainingConfig": str(paths.train_config),
        "semanticContract": {
            "zero": "unknown/background; ignored by semantic cross-entropy",
            "foreground": "fixed ObjectGS label inherited during anchor growth",
            "persistentRegionMapping": str(paths.region_map),
        },
        "fingerprint": fingerprint_files(
            input_files,
            extra={
                "mapanythingRunId": config.mapanything_run_id,
                "objectgsRunId": config.run_id,
                "initializerStride": config.initializer_stride,
                "geometryCompletedAsUnknown": (
                    config.geometry_completed_as_unknown
                ),
            },
        ),
    }
    atomic_write_json(paths.stage_summary("prepare"), summary)
    return summary


def _write_labeled_initializer(
    cache_path: Path,
    provenance_path: Path,
    output_path: Path,
    label_mapping: dict[int, int],
    *,
    stride: int,
    geometry_completed_as_unknown: bool,
) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as cache:
        source_point_count = int(cache["points"].shape[0])
        points = np.asarray(cache["points"], dtype=np.float32)[::stride]
        labels = np.asarray(cache["labels"], dtype=np.int32)[::stride]
        colors = np.asarray(cache["source_colors"], dtype=np.uint8)[::stride]
    provenance = np.asarray(
        np.load(provenance_path, allow_pickle=False),
        dtype=np.uint8,
    )[::stride]
    if not (points.shape[0] == labels.shape[0] == provenance.shape[0]):
        raise ValueError("MapAnything point cache arrays have inconsistent sizes.")

    remapped = np.zeros(labels.shape, dtype=np.uint8)
    for source_id, target_id in label_mapping.items():
        remapped[labels == source_id] = target_id
    unknown_region_count = int(
        np.count_nonzero((labels > 0) & (remapped == 0))
    )
    if unknown_region_count:
        raise ValueError(
            f"{unknown_region_count} initializer points use unmapped region IDs."
        )
    geometry_unknown_count = 0
    if geometry_completed_as_unknown:
        geometry_mask = provenance == 3
        geometry_unknown_count = int(np.count_nonzero(geometry_mask))
        remapped[geometry_mask] = 0
    if not np.any(remapped == 0):
        remapped[int(np.argmin(labels))] = 0
        geometry_unknown_count += 1

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
            ("label", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["nx"] = 0
    vertices["ny"] = 0
    vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["label"] = remapped
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
    ).write(output_path)
    return {
        "sourcePointCount": source_point_count,
        "initializerPointCount": int(points.shape[0]),
        "initializerStride": int(stride),
        "labeledPointCount": int(np.count_nonzero(remapped)),
        "unknownPointCount": int(np.count_nonzero(remapped == 0)),
        "geometryCompletedAsUnknownCount": geometry_unknown_count,
        "objectgsLabelCount": int(np.unique(remapped[remapped > 0]).size),
    }


def _write_training_config(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
) -> None:
    template_path = (
        config.objectgs_root / "config" / "objectgs" / "3d" / "3dovs" / "config.yaml"
    )
    if not template_path.is_file():
        raise FileNotFoundError(
            f"Released ObjectGS 3D configuration is missing: {template_path}"
        )
    payload = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    model = payload["model_params"]
    optimization = payload["optim_params"]
    model.update(
        {
            "source_path": str(paths.dataset_dir),
            "dataset_name": "",
            "exp_name": str(paths.model_parent),
            "eval": False,
            "resolution": -1,
            "ratio": 1,
            "data_format": "colmap",
            "images": "images",
            "add_mask": False,
            "add_depth": False,
        }
    )
    model["model_config"]["kwargs"]["voxel_size"] = float(config.voxel_size)
    optimization["iterations"] = int(config.iterations)
    optimization["offset_lr_max_steps"] = int(config.iterations)
    optimization["mlp_opacity_lr_max_steps"] = int(config.iterations)
    optimization["mlp_color_lr_max_steps"] = int(config.iterations)
    optimization["position_lr_max_steps"] = int(config.iterations)
    optimization["update_until"] = int(config.iterations)
    optimization["lambda_object_loss"] = float(
        config.semantic_loss_weight
    )
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.train_config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def _remap_labels(
    labels: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.uint8)
    for source_id, target_id in mapping.items():
        output[labels == source_id] = target_id
    unknown = (labels > 0) & (output == 0)
    if np.any(unknown):
        values = sorted(int(value) for value in np.unique(labels[unknown]))
        raise ValueError(f"Label map contains unmapped persistent IDs: {values}")
    return output


def _region_names(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    return {
        int(row["id"]): str(row.get("name") or f"Region {row['id']}")
        for row in payload.get("regions", [])
        if int(row.get("id", 0)) > 0
    }


def _complete_summary(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} summary does not exist: {path}")
    payload = read_json(path)
    if payload.get("status") != "complete":
        raise RuntimeError(f"{label} is not complete: {path}")
    return payload


def _report_count(
    callback: ProgressCallback | None,
    label: str,
    completed: int,
    total: int,
) -> None:
    if callback is not None and (
        completed == 1 or completed % 10 == 0 or completed == total
    ):
        callback(f"{label} {completed}/{total}.")
