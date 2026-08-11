"""Prepare a 1024px paper dataset aligned with the interactive editor."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.interactive.paths import EditorPaths, read_jsonl
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    write_jsonl,
)

from .contract import OfficialSegmentThenSplatConfig, OfficialSegmentThenSplatPaths


Progress = Callable[[str], None]


def prepare_paper_dataset(
    config: OfficialSegmentThenSplatConfig,
    paths: OfficialSegmentThenSplatPaths,
    editor: EditorPaths,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(editor.frame_manifest)
    if not rows:
        raise ValueError(f"Frame manifest is empty: {editor.frame_manifest}")
    long_sides = {max(int(row["width"]), int(row["height"])) for row in rows}
    if long_sides != {int(config.image_long_side)}:
        raise ValueError(
            "Prepared editor RGB resolution does not match --image-long-side: "
            f"found {sorted(long_sides)}, expected {config.image_long_side}."
        )
    source_sparse = config.model_dir.parent / "dataset" / "sparse" / "0"
    source_images_txt = source_sparse / "images.txt"
    if not source_images_txt.is_file():
        raise FileNotFoundError(
            "The aligned OMeGa COLMAP images.txt is required: "
            f"{source_images_txt}"
        )

    paths.images_dir.mkdir(parents=True, exist_ok=True)
    paths.sparse_dir.mkdir(parents=True, exist_ok=True)
    frame_map: list[dict[str, Any]] = []
    image_names: list[str] = []
    source_pose_rows = _image_pose_rows(source_images_txt)

    for index, row in enumerate(rows):
        frame_id = int(row["sourceFrameId"])
        output_name = f"scan_000_{frame_id:06d}.jpg"
        source_image = (editor.dataset_dir / str(row["colorPath"])).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(f"Prepared 1024px RGB is missing: {source_image}")
        output_image = paths.images_dir / output_name
        _replace_symlink(output_image, source_image)
        pose = source_pose_rows.get(output_name)
        if pose is None:
            raise KeyError(f"Aligned COLMAP pose is missing for {output_name}")
        frame_map.append(
            {
                "frameId": frame_id,
                "imageName": output_name,
                "trackImageName": str(row.get("sourceImagePath") or row["imageName"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "fx": float(row["fx"]),
                "fy": float(row["fy"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
                "sourceImage": str(source_image),
            }
        )
        image_names.append(output_name)
        if progress is not None and (
            index == 0 or (index + 1) % 25 == 0 or index + 1 == len(rows)
        ):
            progress(f"Prepared paper RGB frames {index + 1}/{len(rows)}.")

    _write_cameras(paths.sparse_dir / "cameras.txt", frame_map)
    _write_images(
        paths.sparse_dir / "images.txt",
        frame_map,
        source_pose_rows,
    )
    (paths.sparse_dir / "points3D.txt").write_text(
        "# 3D point list replaced by the object initializer.\n",
        encoding="utf-8",
    )
    (paths.scene_dir / "train.txt").write_text(
        "".join(f"{name}\n" for name in image_names),
        encoding="utf-8",
    )
    (paths.scene_dir / "test.txt").write_text("", encoding="utf-8")
    write_jsonl(paths.frame_map, frame_map)
    summary = {
        "schemaVersion": 1,
        "stage": "prepare",
        "timestampUtc": now_utc(),
        "paperInput": "released AutoSeg-SAM2 at the editor's 1024px raster",
        "frameCount": len(frame_map),
        "imageLongSide": int(config.image_long_side),
        "sceneDir": str(paths.scene_dir),
        "frameMap": str(paths.frame_map),
        "alignedCameraSource": str(source_sparse),
        "trackModel": str(editor.colmap_track_model),
        "alignedColmapPoints": str(editor.colmap_sparse_source),
        "pixelTransformSummary": str(editor.colmap_pixel_transform_summary),
    }
    atomic_write_json(paths.stage_summary("prepare"), summary)
    return summary


def _replace_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        path.unlink()
    path.symlink_to(target)


def _image_pose_rows(path: Path) -> dict[str, list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    poses: dict[str, list[str]] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 10:
            continue
        poses[Path(tokens[9]).name] = tokens
        if index < len(lines):
            index += 1
    return poses


def _write_cameras(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Camera list with one line of data per camera:",
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"{index} PINHOLE {row['width']} {row['height']} "
            f"{row['fx']:.12g} {row['fy']:.12g} "
            f"{row['cx']:.12g} {row['cy']:.12g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_images(
    path: Path,
    rows: list[dict[str, Any]],
    poses: dict[str, list[str]],
) -> None:
    lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for index, row in enumerate(rows, start=1):
        source = poses[str(row["imageName"])]
        lines.append(" ".join([str(index), *source[1:8], str(index), str(row["imageName"])]))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

