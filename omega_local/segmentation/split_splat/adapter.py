"""Non-destructive editor-to-Split&Splat dataset adapter."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.paths import EditorPaths

from .contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    fingerprint_files,
    now_utc,
    read_jsonl,
    replace_symlink,
    write_jsonl,
)


def prepare_dataset(
    config: SplitSplatRunConfig,
    editor_paths: EditorPaths,
    run_paths: SplitSplatRunPaths,
    *,
    overwrite: bool,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Stage the exact editor raster grid as an upstream COLMAP dataset."""
    source_sparse = _source_sparse_model(config.model_dir)
    source_images = _read_colmap_images_text(source_sparse / "images.txt")
    rows = read_jsonl(editor_paths.frame_manifest)
    if not rows:
        raise ValueError(f"Editor frame manifest is empty: {editor_paths.frame_manifest}")

    if run_paths.input_dir.exists() and overwrite:
        shutil.rmtree(run_paths.input_dir)
    if run_paths.stage_summary("prepare").is_file() and not overwrite:
        return json.loads(run_paths.stage_summary("prepare").read_text(encoding="utf-8"))

    dataset_dir = run_paths.dataset_dir
    image_dir = dataset_dir / "images"
    depth_dir = dataset_dir / "depth"
    sparse_dir = dataset_dir / "sparse" / "0"
    for directory in (image_dir, depth_dir, sparse_dir):
        directory.mkdir(parents=True, exist_ok=True)

    depth_source_dir, depth_source_label = _resolve_depth_source(config, editor_paths)
    frame_rows: list[dict[str, Any]] = []
    camera_lines = _colmap_header("Camera", len(rows))
    image_lines = _colmap_header("Image", len(rows))
    depth_params: dict[str, dict[str, float]] = {}
    rotation_errors: list[float] = []
    translation_errors: list[float] = []
    input_files = [editor_paths.frame_manifest, source_sparse / "images.txt"]

    if progress is not None:
        progress(f"Preparing {len(rows)} aligned RGB, depth, and camera frames.")
    for frame_index, row in enumerate(rows):
        frame_id = int(row["sai3dFrameId"])
        source_image = _editor_image_path(editor_paths, row)
        input_files.append(source_image)
        colmap_name = _colmap_name_for_row(row)
        source_meta = _match_colmap_image(source_images, colmap_name)
        staged_name = f"{Path(source_meta['name']).stem}.JPEG"
        staged_image = image_dir / staged_name
        replace_symlink(staged_image, source_image)

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
        image_lines.append(
            f"{source_meta['image_id']} "
            f"{' '.join(f'{value:.17g}' for value in source_meta['qvec'])} "
            f"{' '.join(f'{value:.17g}' for value in source_meta['tvec'])} "
            f"{camera_id} {staged_name}"
        )
        image_lines.append("")

        depth_source = _find_depth_file(depth_source_dir, row, source_meta)
        input_files.append(depth_source)
        metric_depth = _load_metric_depth(depth_source)
        if metric_depth.shape != (height, width):
            raise ValueError(
                f"Depth shape {metric_depth.shape} does not match frame {frame_id} "
                f"shape {(height, width)}: {depth_source}"
            )
        valid = np.isfinite(metric_depth) & (metric_depth > 0.0)
        if not np.any(valid):
            raise ValueError(f"Depth map has no valid positive samples: {depth_source}")
        metric_depth = np.where(valid, metric_depth, 0.0).astype(np.float32)
        stem = Path(staged_name).stem
        np.save(depth_dir / f"{stem}_pred.npy", metric_depth)
        _write_inverse_depth_png(depth_dir / f"{stem}.png", metric_depth)
        depth_params[stem] = {"scale": 1.0, "offset": 0.0}

        pose_path = editor_paths.dataset_dir / str(row["posePath"])
        pose_world_from_camera = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
        colmap_world_from_camera = _world_from_camera(
            np.asarray(source_meta["qvec"], dtype=np.float64),
            np.asarray(source_meta["tvec"], dtype=np.float64),
        )
        rotation_errors.append(_rotation_error_degrees(
            pose_world_from_camera[:3, :3],
            colmap_world_from_camera[:3, :3],
        ))
        translation_errors.append(float(np.linalg.norm(
            pose_world_from_camera[:3, 3] - colmap_world_from_camera[:3, 3]
        )))
        frame_rows.append(
            {
                "frameId": frame_id,
                "sourceFrameId": int(row.get("sourceFrameId", frame_id)),
                "editorImageName": str(row.get("imageName", "")),
                "splitSplatImageName": staged_name,
                "sourceImagePath": str(source_image),
                "depthSourcePath": str(depth_source),
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "colmapImageId": int(source_meta["image_id"]),
            }
        )
        completed = frame_index + 1
        if progress is not None and (
            completed == 1
            or completed % 10 == 0
            or completed == len(rows)
        ):
            progress(f"Prepared input frames {completed}/{len(rows)}.")

    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    replace_symlink(sparse_dir / "points3D.txt", source_sparse / "points3D.txt")
    source_ply = source_sparse / "points3D.ply"
    if source_ply.is_file():
        replace_symlink(sparse_dir / "points3D.ply", source_ply)
    atomic_write_json(sparse_dir / "depth_params.json", depth_params)
    write_jsonl(run_paths.frame_map, frame_rows)

    camera_validation = {
        "schemaVersion": 1,
        "frameCount": len(frame_rows),
        "pixelGrid": {
            "width": int(frame_rows[0]["width"]),
            "height": int(frame_rows[0]["height"]),
        },
        "poseAgreement": {
            "rotationErrorDegreesMean": float(np.mean(rotation_errors)),
            "rotationErrorDegreesMax": float(np.max(rotation_errors)),
            "translationErrorMean": float(np.mean(translation_errors)),
            "translationErrorMax": float(np.max(translation_errors)),
        },
        "passed": bool(
            max(rotation_errors, default=0.0) < 1e-3
            and max(translation_errors, default=0.0) < 1e-5
        ),
        "note": (
            "The staged COLMAP camera must agree with the editor camera. "
            "A failed check means Split&Splat artifacts will not align in the web viewer."
        ),
    }
    atomic_write_json(run_paths.camera_validation, camera_validation)
    if progress is not None:
        progress(
            "Validated editor/COLMAP camera agreement "
            f"(max rotation {camera_validation['poseAgreement']['rotationErrorDegreesMax']:.3g} deg, "
            f"max translation {camera_validation['poseAgreement']['translationErrorMax']:.3g})."
        )
    if not camera_validation["passed"]:
        raise ValueError(
            "Editor and OMeGa COLMAP poses disagree; refusing to stage a visually "
            f"misaligned Split&Splat run. See {run_paths.camera_validation}"
        )

    manifest = {
        "schemaVersion": 1,
        "stage": "prepare",
        "method": "Split&Splat paper baseline input adapter",
        "timestampUtc": now_utc(),
        "sceneName": run_paths.shared_id,
        "frameCount": len(frame_rows),
        "modelDir": str(config.model_dir),
        "editorBaselineName": config.editor_baseline_name,
        "editorFrameManifest": str(editor_paths.frame_manifest),
        "sourceSparseModel": str(source_sparse),
        "depthSource": depth_source_label,
        "depthSourceDir": str(depth_source_dir),
        "depthRepresentation": "metric_depth",
        "datasetDir": str(dataset_dir),
        "frameMap": str(run_paths.frame_map),
        "cameraValidation": str(run_paths.camera_validation),
        "fingerprint": fingerprint_files(
            input_files,
            extra={
                "sharedId": run_paths.shared_id,
                "depthSource": depth_source_label,
                "frameCount": len(frame_rows),
            },
        ),
    }
    atomic_write_json(run_paths.input_manifest, manifest)
    atomic_write_json(run_paths.stage_summary("prepare"), manifest)
    return manifest


def _source_sparse_model(model_dir: Path) -> Path:
    path = model_dir.parent / "dataset" / "sparse" / "0"
    required = (path / "images.txt", path / "points3D.txt")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "Split&Splat requires the OMeGa-aligned text COLMAP model. Missing: "
            + ", ".join(missing)
        )
    return path.resolve()


def _resolve_depth_source(
    config: SplitSplatRunConfig,
    editor_paths: EditorPaths,
) -> tuple[Path, str]:
    if config.depth_dir is not None:
        path = config.depth_dir
    elif config.depth_source == "editor_depth":
        path = editor_paths.interactive_dir / "view_evidence" / "depth_npz"
    else:
        raise FileNotFoundError(
            "The faithful DSLR baseline requires precomputed Murre metric depth. "
            "Pass --depth-dir /path/to/murre/depth. Use --depth-source editor_depth "
            "only for the explicit Depth Anything V2 ablation."
        )
    if not path.is_dir():
        raise FileNotFoundError(f"Depth source directory does not exist: {path}")
    return path.resolve(), config.depth_source


def _editor_image_path(editor_paths: EditorPaths, row: dict[str, Any]) -> Path:
    staged = editor_paths.dataset_dir / str(row["colorPath"])
    if staged.is_file():
        return staged.resolve()
    source = Path(str(row.get("sourceImagePath", ""))).expanduser()
    if source.is_file():
        return source.resolve()
    raise FileNotFoundError(f"No editor image exists for frame {row.get('sai3dFrameId')}")


def _find_depth_file(
    depth_dir: Path,
    row: dict[str, Any],
    source_meta: dict[str, Any],
) -> Path:
    names = [
        f"{int(row['sai3dFrameId']):06d}",
        f"{int(row.get('sourceFrameId', row['sai3dFrameId'])):06d}",
        Path(str(row.get("imageName", ""))).stem,
        Path(str(source_meta["name"])).stem,
    ]
    suffixes = (".npz", ".npy")
    for name in dict.fromkeys(name for name in names if name):
        for suffix in suffixes:
            candidate = depth_dir / f"{name}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    raise FileNotFoundError(
        f"No metric depth found for frame {row['sai3dFrameId']} in {depth_dir}. "
        f"Tried stems: {', '.join(names)}"
    )


def _load_metric_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        values = np.load(path)
    else:
        with np.load(path) as archive:
            key = next(
                (candidate for candidate in ("depth_m", "depth", "predicted_depth") if candidate in archive),
                None,
            )
            if key is None:
                raise KeyError(f"Depth NPZ has no metric depth array: {path}")
            values = archive[key]
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D metric depth map, got {values.shape}: {path}")
    return values


def _write_inverse_depth_png(path: Path, metric_depth: np.ndarray) -> None:
    valid = np.isfinite(metric_depth) & (metric_depth > 0.0)
    inverse = np.zeros(metric_depth.shape, dtype=np.float32)
    inverse[valid] = 1.0 / metric_depth[valid]
    encoded = np.clip(np.rint(inverse * float(2**16)), 0, 65535).astype(np.uint16)
    Image.fromarray(encoded, mode="I;16").save(path)


def _colmap_header(kind: str, count: int) -> list[str]:
    if kind == "Camera":
        return [
            "# Camera list with one line of data per camera:",
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            f"# Number of cameras: {count}",
        ]
    return [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
        f"# Number of images: {count}, mean observations per image: 0",
    ]


def _read_colmap_images_text(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data_lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    for line in data_lines:
        tokens = line.split()
        if len(tokens) != 10 or Path(tokens[9]).suffix.lower() not in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }:
            continue
        rows.append(
            {
                "image_id": int(tokens[0]),
                "qvec": [float(value) for value in tokens[1:5]],
                "tvec": [float(value) for value in tokens[5:8]],
                "camera_id": int(tokens[8]),
                "name": tokens[9],
            }
        )
    if not rows:
        raise ValueError(f"No images found in COLMAP text model: {path}")
    return rows


def _colmap_name_for_row(row: dict[str, Any]) -> str:
    source = Path(str(row.get("sourceImagePath", "")))
    scan_id = str(row.get("scanID") or source.parent.name or "scan_000")
    image_name = str(row.get("imageName") or source.name)
    return f"{scan_id}_{Path(image_name).name}"


def _match_colmap_image(
    source_images: list[dict[str, Any]],
    expected_name: str,
) -> dict[str, Any]:
    expected_stem = Path(expected_name).stem
    matches = [
        row
        for row in source_images
        if Path(str(row["name"])).stem == expected_stem
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"No COLMAP image matches {expected_name}")
    raise ValueError(f"Multiple COLMAP images match {expected_name}")


def _world_from_camera(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec.tolist()
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = -rotation.T @ tvec
    return pose


def _rotation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))
