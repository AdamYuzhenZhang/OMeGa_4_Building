"""Paper object-specific initialization without the upstream Nerfstudio dependency."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial.distance import cdist

from omega_local.segmentation.interactive.colmap_track_propagation import (
    _match_colmap_images,
    _read_pixel_transforms,
    _read_reconstruction,
    _scaled_observations,
)
from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.interactive.sam2_video_propagation import PropagationFrame
from omega_local.segmentation.split_splat.artifacts import (
    register_canonical_mask_layer,
    write_label_artifacts,
)
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_jsonl,
)

from .contract import LEVELS, OfficialSegmentThenSplatConfig, OfficialSegmentThenSplatPaths
from .execution import run_command


Progress = Callable[[str], None]
_LEVEL_DIRS = {
    "large": "default",
    "middle": "middle",
    "small": "small",
}
_PLACEHOLDER_COUNTS = {"large": 1000, "middle": 100, "small": 10}


def initialize_paper_objects(
    config: OfficialSegmentThenSplatConfig,
    paths: OfficialSegmentThenSplatPaths,
    editor: EditorPaths,
    *,
    progress: Progress | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    preprocess = config.source_root / "helpers" / "preprocess_mask.py"
    command = [
        str(config.python),
        "-u",
        str(preprocess),
        "--mask_root",
        str(paths.autoseg_output),
        "--out_root",
        str(paths.scene_dir),
        "--image_path",
        str(paths.images_dir),
    ]
    if dry_run:
        return {
            "schemaVersion": 1,
            "stage": "initialize",
            "status": "planned",
            "preprocessCommand": command,
            "sceneDir": str(paths.scene_dir),
        }
    for level_dir in _LEVEL_DIRS.values():
        for suffix in ("", "_merged"):
            target = paths.scene_dir / f"multiview_masks_{level_dir}{suffix}"
            if target.exists():
                shutil.rmtree(target)
    run_command(
        command,
        cwd=config.source_root,
        log_path=paths.initialization_stage / "logs" / "preprocess_masks.log",
        progress=progress,
    )

    frame_rows = read_jsonl(paths.frame_map)
    frames = [
        PropagationFrame(
            frame_id=int(row["frameId"]),
            image_name=str(row["trackImageName"]),
            image_path=Path(str(row["sourceImage"])),
            width=int(row["width"]),
            height=int(row["height"]),
            fx=float(row["fx"]),
            fy=float(row["fy"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
        )
        for row in frame_rows
    ]
    reconstruction = _read_reconstruction(editor.colmap_track_model)
    images_by_frame = _match_colmap_images(reconstruction, frames)
    pixel_transforms = _read_pixel_transforms(editor.colmap_pixel_transform_summary)
    point_ids = np.asarray(
        sorted(int(value) for value in reconstruction.points3D.keys()),
        dtype=np.int64,
    )
    positions, colors = _aligned_colmap_points(
        editor.colmap_sparse_source,
        reconstruction,
        point_ids,
    )
    max_point_id = int(point_ids[-1])
    valid_points = np.ones(max_point_id + 1, dtype=bool)
    point_row = np.full(max_point_id + 1, -1, dtype=np.int64)
    point_row[point_ids] = np.arange(point_ids.size, dtype=np.int64)

    level_masks = {
        level: _load_object_masks(
            paths.scene_dir / f"multiview_masks_{directory}",
            [str(row["imageName"]) for row in frame_rows],
        )
        for level, directory in _LEVEL_DIRS.items()
    }
    oversized = {level: len(value) for level, value in level_masks.items() if len(value) > 255}
    if oversized:
        raise ValueError(
            "The released uint8 object-ID representation supports at most 255 "
            f"objects per level; got {oversized}."
        )
    labels_by_level = {
        level: np.full(point_ids.size, 255, dtype=np.uint8)
        for level in LEVELS
    }

    for frame_index, frame in enumerate(frames):
        image = images_by_frame[int(frame.frame_id)]
        observed_ids, pixels = _scaled_observations(
            image,
            frame,
            valid_points,
            pixel_transforms=pixel_transforms,
        )
        if observed_ids.size:
            rows = point_row[observed_ids]
            valid = rows >= 0
            rows = rows[valid]
            pixels = pixels[valid]
            for level in LEVELS:
                frame_labels = _exclusive_frame_labels(
                    level_masks[level],
                    frame_index,
                )
                values = frame_labels[pixels[:, 1], pixels[:, 0]]
                positive = values > 0
                labels_by_level[level][rows[positive]] = (
                    values[positive] - 1
                ).astype(np.uint8)
        if progress is not None and (
            frame_index == 0
            or (frame_index + 1) % 20 == 0
            or frame_index + 1 == len(frames)
        ):
            progress(
                f"Assigned COLMAP point identities {frame_index + 1}/{len(frames)}."
            )

    positions, colors, labels_by_level = _add_released_placeholders(
        positions,
        colors,
        labels_by_level,
        level_masks,
    )
    merge_rows: dict[str, Any] = {}
    merged_masks: dict[str, list[list[np.ndarray]]] = {}
    for level in LEVELS:
        labels, masks, pairs = _merge_similar_objects(
            positions,
            colors,
            labels_by_level[level],
            level_masks[level],
        )
        labels_by_level[level] = labels
        merged_masks[level] = masks
        merge_rows[level] = {
            "inputObjectCount": len(level_masks[level]),
            "outputObjectCount": len(masks),
            "mergedPairs": [list(pair) for pair in pairs],
            "labeledPointCount": int(np.count_nonzero(labels != 255)),
        }

    _write_merged_masks(paths, frame_rows, merged_masks)
    initializer = paths.sparse_dir / "points3D.ply"
    _write_semantic_ply(initializer, positions, colors, labels_by_level)
    viewer_rows = _register_layers_and_points(
        paths,
        editor,
        frame_rows,
        merged_masks,
        positions,
        colors,
        labels_by_level,
        progress=progress,
    )
    summary = {
        "schemaVersion": 1,
        "stage": "initialize",
        "timestampUtc": now_utc(),
        "method": "Released Segment then Splat object-specific initialization",
        "pointSource": str(editor.colmap_sparse_source),
        "trackSource": str(editor.colmap_track_model),
        "initializerPly": str(initializer),
        "pointCount": int(positions.shape[0]),
        "levels": merge_rows,
        "viewer": viewer_rows,
        "adapterNotes": [
            "The released overlap-filtering script is called unchanged.",
            "COLMAP parsing uses pycolmap instead of the helper's Nerfstudio dependency.",
            "Aligned COLMAP positions preserve the editor/OMeGa world frame.",
            "Geometry+color object merging uses the released xyz + 20*rgb < 0.5 score.",
            "Merged object IDs are reindexed coherently in masks and points.",
        ],
    }
    atomic_write_json(paths.stage_summary("initialize"), summary)
    return summary


def _aligned_colmap_points(
    path: Path,
    reconstruction: Any,
    point_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Aligned COLMAP PLY is missing: {path}")
    vertices = PlyData.read(path)["vertex"].data
    positions = np.column_stack(
        [np.asarray(vertices[name], dtype=np.float32) for name in ("x", "y", "z")]
    )
    colors = np.column_stack(
        [np.asarray(vertices[name], dtype=np.uint8) for name in ("red", "green", "blue")]
    )
    if positions.shape[0] != point_ids.size:
        raise ValueError(
            f"Aligned COLMAP PLY has {positions.shape[0]} points; "
            f"the track model has {point_ids.size}."
        )
    expected_colors = np.asarray(
        [reconstruction.points3D[int(point_id)].color for point_id in point_ids],
        dtype=np.uint8,
    )
    agreement = float(np.mean(np.all(colors == expected_colors, axis=1)))
    if agreement < 0.99:
        raise ValueError(
            "Aligned COLMAP PLY does not preserve point-ID order "
            f"(RGB agreement {agreement:.3f})."
        )
    return positions, colors


def _load_object_masks(
    root: Path,
    image_names: list[str],
) -> list[list[np.ndarray]]:
    if not root.is_dir():
        return []
    rows: list[list[np.ndarray]] = []
    for object_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        masks = []
        for name in image_names:
            path = object_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Paper mask is missing: {path}")
            masks.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
        rows.append(masks)
    return rows


def _exclusive_frame_labels(
    object_masks: list[list[np.ndarray]],
    frame_index: int,
) -> np.ndarray:
    if not object_masks:
        raise ValueError("Segment then Splat produced an empty mask level.")
    height, width = object_masks[0][frame_index].shape
    labels = np.zeros((height, width), dtype=np.uint16)
    for object_id, masks in enumerate(object_masks):
        labels[masks[frame_index] > 127] = object_id + 1
    return labels


def _add_released_placeholders(
    positions: np.ndarray,
    colors: np.ndarray,
    labels: dict[str, np.ndarray],
    masks: dict[str, list[list[np.ndarray]]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    label_parts = {level: [values] for level, values in labels.items()}
    position_parts = [positions]
    color_parts = [colors]
    for level in LEVELS:
        present = set(int(value) for value in np.unique(labels[level]) if int(value) != 255)
        for object_id in range(len(masks[level])):
            if object_id in present:
                continue
            count = _PLACEHOLDER_COUNTS[level]
            position_parts.append(np.zeros((count, 3), dtype=np.float32))
            color_parts.append(np.zeros((count, 3), dtype=np.uint8))
            for candidate in LEVELS:
                value = (
                    object_id
                    if candidate == level or level == "small"
                    else 255
                )
                label_parts[candidate].append(
                    np.full(count, value, dtype=np.uint8)
                )
    position_parts.append(np.zeros((1000, 3), dtype=np.float32))
    color_parts.append(np.zeros((1000, 3), dtype=np.uint8))
    for level in LEVELS:
        label_parts[level].append(np.full(1000, 255, dtype=np.uint8))
    return (
        np.concatenate(position_parts),
        np.concatenate(color_parts),
        {level: np.concatenate(parts) for level, parts in label_parts.items()},
    )


def _merge_similar_objects(
    positions: np.ndarray,
    colors: np.ndarray,
    labels: np.ndarray,
    masks: list[list[np.ndarray]],
) -> tuple[np.ndarray, list[list[np.ndarray]], list[tuple[int, int]]]:
    object_count = len(masks)
    means_xyz = np.zeros((object_count, 3), dtype=np.float64)
    means_rgb = np.zeros((object_count, 3), dtype=np.float64)
    for object_id in range(object_count):
        selected = labels == object_id
        if np.any(selected):
            means_xyz[object_id] = np.mean(positions[selected], axis=0)
            means_rgb[object_id] = np.mean(colors[selected], axis=0)
    xyz = cdist(means_xyz, means_xyz)
    rgb = cdist(means_rgb, means_rgb) / 255.0
    candidate = np.where(
        (xyz + 20.0 * rgb < 0.5) & (xyz > 0.0) & (rgb > 0.0)
    )
    pairs = sorted(
        {tuple(sorted((int(a), int(b)))) for a, b in zip(*candidate) if a != b}
    )
    parent = list(range(object_count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)
    groups: dict[int, list[int]] = {}
    for object_id in range(object_count):
        groups.setdefault(find(object_id), []).append(object_id)
    old_to_new: dict[int, int] = {}
    merged_masks: list[list[np.ndarray]] = []
    for new_id, members in enumerate(groups.values()):
        for old_id in members:
            old_to_new[old_id] = new_id
        output = [mask.copy() for mask in masks[members[0]]]
        for old_id in members[1:]:
            output = [
                cv2.bitwise_or(left, right)
                for left, right in zip(output, masks[old_id])
            ]
        merged_masks.append(output)
    merged_labels = np.full(labels.shape, 255, dtype=np.uint8)
    for old_id, new_id in old_to_new.items():
        merged_labels[labels == old_id] = np.uint8(new_id)
    return merged_labels, merged_masks, pairs


def _write_merged_masks(
    paths: OfficialSegmentThenSplatPaths,
    frame_rows: list[dict[str, Any]],
    masks: dict[str, list[list[np.ndarray]]],
) -> None:
    for level, directory in _LEVEL_DIRS.items():
        root = paths.scene_dir / f"multiview_masks_{directory}_merged"
        root.mkdir(parents=True, exist_ok=True)
        for object_id, frame_masks in enumerate(masks[level]):
            object_dir = root / f"{object_id:03d}"
            object_dir.mkdir(parents=True, exist_ok=True)
            for row, mask in zip(frame_rows, frame_masks):
                cv2.imwrite(str(object_dir / str(row["imageName"])), mask)


def _write_semantic_ply(
    path: Path,
    positions: np.ndarray,
    colors: np.ndarray,
    labels: dict[str, np.ndarray],
) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("obj_id_default", "u1"),
        ("obj_id_middle", "u1"),
        ("obj_id_small", "u1"),
    ]
    vertices = np.empty(positions.shape[0], dtype=dtype)
    vertices["x"], vertices["y"], vertices["z"] = positions.T
    vertices["nx"] = vertices["ny"] = vertices["nz"] = 0
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    vertices["obj_id_default"] = labels["large"]
    vertices["obj_id_middle"] = labels["middle"]
    vertices["obj_id_small"] = labels["small"]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _register_layers_and_points(
    paths: OfficialSegmentThenSplatPaths,
    editor: EditorPaths,
    frame_rows: list[dict[str, Any]],
    masks: dict[str, list[list[np.ndarray]]],
    positions: np.ndarray,
    colors: np.ndarray,
    labels: dict[str, np.ndarray],
    *,
    progress: Progress | None,
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for level in LEVELS:
        label_dir = paths.label_maps_root / level
        overlay_dir = paths.overlays_root / level
        frame_summaries = []
        for index, row in enumerate(frame_rows):
            frame_labels = _exclusive_frame_labels(masks[level], index)
            frame_id = int(row["frameId"])
            write_label_artifacts(
                label_dir,
                overlay_dir,
                frame_id,
                frame_labels,
            )
            frame_summaries.append(
                {
                    "frameId": frame_id,
                    "coverage": float(np.count_nonzero(frame_labels) / frame_labels.size),
                    "visibleInstanceCount": int(
                        np.count_nonzero(np.unique(frame_labels) > 0)
                    ),
                }
            )
        method_id = f"segment_then_splat_paper_{paths.run_dir.name}_{level}"
        register_canonical_mask_layer(
            editor_paths=editor,
            method_id=method_id,
            display_name=f"Segment then Splat Paper {level.title()} Masks",
            description=(
                "Read-only released AutoSeg-SAM2 tracks after overlap filtering "
                "and geometry/color object association."
            ),
            label_namespace=f"segment_then_splat_{level}_object",
            label_maps=label_dir,
            overlays=overlay_dir,
            frames=frame_summaries,
            canonical_run_dir=paths.run_dir,
            canonical_summary=paths.stage_summary("initialize"),
            stage="paper_object_initialization",
            engine_name="Segment then Splat",
            proposal_kind="paper_multiview_object",
        )
        cache = paths.initialization_stage / f"points_{level}.npz"
        np.savez_compressed(
            cache,
            points=positions.astype(np.float32),
            colors=colors.astype(np.uint8),
            labels=np.where(labels[level] == 255, 0, labels[level].astype(np.int32) + 1),
        )
        point_run_id = f"{method_id}_initializer"
        point_run_dir = editor.segmentation3d_dir / "runs" / point_run_id
        point_run_dir.mkdir(parents=True, exist_ok=True)
        positive_count = int(np.count_nonzero(labels[level] != 255))
        atomic_write_json(
            point_run_dir / "experiment.json",
            {
                "schemaVersion": 1,
                "runId": point_run_id,
                "methodId": "segment_then_splat_paper",
                "experimentFamily": "segment_then_splat_paper",
                "artifactRole": "object_initialization",
                "baseRunId": paths.run_dir.name,
                "inputId": "official_autoseg_sam2",
                "displayName": f"Paper {level.title()} COLMAP Objects",
                "labelSpace": f"segment_then_splat_{level}_object",
                "pointsCachePath": str(cache),
                "geometrySource": "aligned_colmap_sparse",
                "ready": True,
                "timestampUtc": now_utc(),
                "canonicalRunDir": str(paths.run_dir),
                "pointSummary": {
                    "pointCount": int(positions.shape[0]),
                    "labeledPointCount": positive_count,
                    "labelCount": len(masks[level]),
                },
            },
        )
        rows[level] = {
            "methodId": method_id,
            "objectCount": len(masks[level]),
            "labelMaps": str(label_dir),
            "overlays": str(overlay_dir),
            "pointsCache": str(cache),
        }
        if progress is not None:
            progress(f"Registered paper {level} masks and initialized points.")
    return rows

