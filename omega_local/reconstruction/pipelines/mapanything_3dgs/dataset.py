"""Stage calibrated editor frames without depending on Split&Splat run state."""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.split_splat.adapter import (
    _colmap_header,
    _colmap_name_for_row,
    _editor_image_path,
    _match_colmap_image,
    _read_colmap_images_text,
    _source_sparse_model,
    _world_from_camera,
)
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    fingerprint_files,
    now_utc,
    read_jsonl,
    replace_symlink,
    write_jsonl,
)

from .contract import MapAnything3DGSConfig, MapAnything3DGSPaths


ProgressCallback = Callable[[str], None]


def prepare_input_dataset(
    config: MapAnything3DGSConfig,
    editor_paths: EditorPaths,
    paths: MapAnything3DGSPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Stage RGB and COLMAP cameras on the editor pixel grid."""
    source_sparse = _source_sparse_model(config.model_dir)
    source_images = _read_colmap_images_text(source_sparse / "images.txt")
    rows = read_jsonl(editor_paths.frame_manifest)
    if not rows:
        raise ValueError(
            f"Editor frame manifest is empty: {editor_paths.frame_manifest}"
        )
    if not paths.point_cloud.is_file():
        raise FileNotFoundError(
            f"MapAnything point initializer does not exist: {paths.point_cloud}"
        )

    if paths.input_dir.exists():
        shutil.rmtree(paths.input_dir)
    image_dir = paths.dataset_dir / "images"
    sparse_dir = paths.dataset_dir / "sparse" / "0"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    camera_lines = _colmap_header("Camera", len(rows))
    image_lines = _colmap_header("Image", len(rows))
    frame_rows: list[dict[str, Any]] = []
    input_files = [
        editor_paths.frame_manifest,
        source_sparse / "images.txt",
        paths.point_cloud,
    ]
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    for index, row in enumerate(rows):
        frame_id = int(row["sai3dFrameId"])
        source_image = _editor_image_path(editor_paths, row)
        source_meta = _match_colmap_image(
            source_images,
            _colmap_name_for_row(row),
        )
        staged_name = f"{Path(str(source_meta['name'])).stem}.JPEG"
        replace_symlink(image_dir / staged_name, source_image)
        input_files.append(source_image)

        width = int(row["width"])
        height = int(row["height"])
        fx = float(row["fx"])
        fy = float(row["fy"])
        cx = float(row["cx"])
        cy = float(row["cy"])
        camera_id = int(source_meta["image_id"])
        camera_lines.append(
            f"{camera_id} PINHOLE {width} {height} "
            f"{fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}"
        )
        image_lines.extend(
            [
                f"{camera_id} "
                f"{' '.join(f'{value:.17g}' for value in source_meta['qvec'])} "
                f"{' '.join(f'{value:.17g}' for value in source_meta['tvec'])} "
                f"{camera_id} {staged_name}",
                "",
            ]
        )

        pose = np.loadtxt(
            editor_paths.dataset_dir / str(row["posePath"]),
            dtype=np.float64,
        ).reshape(4, 4)
        colmap_pose = _world_from_camera(
            np.asarray(source_meta["qvec"], dtype=np.float64),
            np.asarray(source_meta["tvec"], dtype=np.float64),
        )
        rotation_errors.append(
            _rotation_error_degrees(pose[:3, :3], colmap_pose[:3, :3])
        )
        translation_errors.append(
            float(np.linalg.norm(pose[:3, 3] - colmap_pose[:3, 3]))
        )
        frame_rows.append(
            {
                "frameId": frame_id,
                "sourceFrameId": int(row.get("sourceFrameId", frame_id)),
                "imageName": staged_name,
                "sourceImagePath": str(source_image),
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "colmapImageId": camera_id,
                "qvec": [float(value) for value in source_meta["qvec"]],
                "tvec": [float(value) for value in source_meta["tvec"]],
            }
        )
        _report_count(progress, "Prepared MapAnything input frames", index + 1, len(rows))

    (sparse_dir / "cameras.txt").write_text(
        "\n".join(camera_lines) + "\n",
        encoding="utf-8",
    )
    (sparse_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    replace_symlink(sparse_dir / "points3D.ply", paths.point_cloud)
    write_jsonl(paths.frame_map, frame_rows)

    maximum_rotation = max(rotation_errors, default=0.0)
    maximum_translation = max(translation_errors, default=0.0)
    camera_validation = {
        "passed": maximum_rotation < 1.0e-3 and maximum_translation < 1.0e-5,
        "rotationErrorDegreesMean": float(np.mean(rotation_errors)),
        "rotationErrorDegreesMax": maximum_rotation,
        "translationErrorMean": float(np.mean(translation_errors)),
        "translationErrorMax": maximum_translation,
    }
    if not camera_validation["passed"]:
        raise ValueError(
            "Editor and OMeGa COLMAP poses disagree; refusing a misaligned "
            f"MapAnything run: {camera_validation}"
        )

    summary = {
        "schemaVersion": 1,
        "stage": "prepare",
        "timestampUtc": now_utc(),
        "method": "Calibrated posed-image dataset with MapAnything initializer",
        "frameCount": len(frame_rows),
        "datasetDir": str(paths.dataset_dir),
        "frameMap": str(paths.frame_map),
        "pointCloud": str(paths.point_cloud),
        "pointCloudRole": "exact_omega_mapanything_initializer",
        "cameraValidation": camera_validation,
        "fingerprint": fingerprint_files(
            input_files,
            extra={
                "runId": config.run_id,
                "propagationMethod": config.propagation_method,
            },
        ),
    }
    atomic_write_json(paths.stage_summary("prepare"), summary)
    return summary


def _rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(
        np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    )
    return float(math.degrees(math.acos(cosine)))


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
