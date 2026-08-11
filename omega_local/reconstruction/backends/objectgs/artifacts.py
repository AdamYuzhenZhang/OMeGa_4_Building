"""Canonical ObjectGS exports and editor registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from omega_local.reconstruction.pipelines.mapanything_3dgs.artifacts import (
    register_mask_layer,
)
from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.interactive.regions import (
    persistent_region_color_map,
)
from omega_local.segmentation.split_splat.artifacts import (
    write_label_artifacts,
)
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
    replace_symlink,
)

from .contract import ObjectGSConfig, ObjectGSPaths


ProgressCallback = Callable[[str], None]


def export_objectgs(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    pointer = read_json(paths.model_pointer)
    model_dir = Path(str(pointer.get("modelDir") or ""))
    anchor_ply = Path(str(pointer.get("anchorPly") or ""))
    if not model_dir.is_dir() or not anchor_ply.is_file():
        raise FileNotFoundError(
            f"Completed ObjectGS model is unavailable: {model_dir}"
        )
    region_map = read_json(paths.region_map)
    objectgs_to_persistent = {
        int(key): int(value)
        for key, value in region_map["objectgsToPersistent"].items()
    }
    frame_rows = read_jsonl(paths.frame_map)
    render_root = model_dir / "train" / f"ours_{config.iterations}"
    source_semantics = render_root / "semantic"
    source_rgb = render_root / "renders"
    if not source_semantics.is_dir() or not source_rgb.is_dir():
        raise FileNotFoundError(
            "ObjectGS did not produce its train RGB and semantic renders: "
            f"{render_root}"
        )

    paths.outputs_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(paths.rgb_renders, source_rgb)
    mask_rows = []
    for index, row in enumerate(frame_rows):
        frame_id = int(row["frameId"])
        source = source_semantics / f"{index:05d}.png"
        if not source.is_file():
            raise FileNotFoundError(
                f"ObjectGS semantic render is missing: {source}"
            )
        objectgs_labels = np.asarray(Image.open(source), dtype=np.int32)
        labels = _restore_persistent_ids(
            objectgs_labels,
            objectgs_to_persistent,
        )
        write_label_artifacts(
            paths.predicted_label_maps,
            paths.predicted_overlays,
            frame_id,
            labels,
        )
        mask_rows.append(
            {
                "frameId": frame_id,
                "coverage": float(
                    np.count_nonzero(labels) / max(labels.size, 1)
                ),
                "labelCount": int(np.unique(labels[labels > 0]).size),
            }
        )
        _report_count(
            progress,
            "Exported ObjectGS semantic renders",
            index + 1,
            len(frame_rows),
        )

    anchor_summary = _export_anchor_points(
        anchor_ply,
        paths.anchor_cache,
        paths.colored_anchors,
        objectgs_to_persistent,
        editor_paths,
    )
    register_mask_layer(
        editor_paths,
        method_id=(
            f"{config.mapanything_run_id}_{config.run_id}_semantic"
        ),
        display_name="ObjectGS Joint Predicted IDs",
        description=(
            "Scene-level ObjectGS semantic rendering after joint RGB and "
            "persistent-region optimization."
        ),
        label_maps=paths.predicted_label_maps,
        overlays=paths.predicted_overlays,
        frames=mask_rows,
        canonical_run_dir=paths.run_dir,
        canonical_summary=paths.stage_summary("export"),
        layer_group="joint_reconstruction",
    )
    experiment_path = _register_anchor_experiment(
        config,
        paths,
        editor_paths,
        anchor_summary,
    )
    summary = {
        "schemaVersion": 1,
        "stage": "export",
        "timestampUtc": now_utc(),
        "method": "Canonical ObjectGS joint-scene outputs",
        "modelDir": str(model_dir),
        "neuralAnchorModel": str(anchor_ply),
        "modelRepresentation": (
            "ObjectGS Scaffold-GS neural anchors; not a conventional explicit "
            "3DGS PLY"
        ),
        "rgbRenders": str(paths.rgb_renders),
        "predictedLabelMaps": str(paths.predicted_label_maps),
        "predictedOverlays": str(paths.predicted_overlays),
        "frames": mask_rows,
        "meanMaskCoverage": float(
            np.mean([row["coverage"] for row in mask_rows])
        ),
        "anchors": anchor_summary,
        "editorExperiment": str(experiment_path),
    }
    atomic_write_json(paths.stage_summary("export"), summary)
    return summary


def _export_anchor_points(
    source: Path,
    cache_path: Path,
    ply_path: Path,
    mapping: dict[int, int],
    editor_paths: EditorPaths,
) -> dict[str, Any]:
    vertices = PlyData.read(source)["vertex"]
    points = np.column_stack(
        [
            np.asarray(vertices[name], dtype=np.float32)
            for name in ("x", "y", "z")
        ]
    )
    objectgs_labels = np.rint(
        np.asarray(vertices["label"], dtype=np.float32)
    ).astype(np.int32)
    labels = _restore_persistent_ids(objectgs_labels, mapping).reshape(-1)
    colors = _persistent_colors(labels, editor_paths)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, points=points, labels=labels)
    output = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("label", "u2"),
        ],
    )
    output["x"], output["y"], output["z"] = points.T
    output["red"], output["green"], output["blue"] = colors.T
    output["label"] = labels.astype(np.uint16)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(output, "vertex")], text=False).write(ply_path)
    return {
        "pointCount": int(points.shape[0]),
        "labeledPointCount": int(np.count_nonzero(labels)),
        "labelCount": int(np.unique(labels[labels > 0]).size),
        "pointsCachePath": str(cache_path),
        "coloredPointCloud": str(ply_path),
    }


def _register_anchor_experiment(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
    editor_paths: EditorPaths,
    summary: dict[str, Any],
) -> Path:
    run_id = f"{config.mapanything_run_id}_{config.run_id}_anchors"
    experiment = editor_paths.segmentation3d_dir / "runs" / run_id
    experiment.mkdir(parents=True, exist_ok=True)
    output = experiment / "experiment.json"
    atomic_write_json(
        output,
        {
            "schemaVersion": 1,
            "runId": run_id,
            "methodId": "objectgs_joint",
            "experimentFamily": "mapanything_objectgs",
            "artifactRole": "joint_reconstruction_labels",
            "baseRunId": config.mapanything_run_id,
            "inputId": config.mapanything_run_id,
            "displayName": (
                f"ObjectGS Joint Anchors: {config.mapanything_run_id}"
            ),
            "labelSpace": "persistent_region",
            "geometrySource": "omega_mapanything_initializer",
            "ready": True,
            "timestampUtc": now_utc(),
            "canonicalRunDir": str(paths.run_dir),
            "pointsCachePath": str(paths.anchor_cache),
            "rgbRendersPath": str(paths.rgb_renders),
            "frameMapPath": str(paths.frame_map),
            "predictedLabelMapsPath": str(paths.predicted_label_maps),
            "maskLayerKey": (
                "propagation_"
                f"{config.mapanything_run_id}_{config.run_id}_semantic"
            ),
            "objectgsRunId": config.run_id,
            "pointSummary": {
                "globalPointCount": int(summary["pointCount"]),
                "labeledGlobalPointCount": int(
                    summary["labeledPointCount"]
                ),
                "labelCount": int(summary["labelCount"]),
            },
        },
    )
    return output


def _restore_persistent_ids(
    labels: np.ndarray,
    mapping: dict[int, int],
) -> np.ndarray:
    output = np.zeros(labels.shape, dtype=np.uint16)
    for source_id, target_id in mapping.items():
        output[labels == source_id] = target_id
    unknown = (labels > 0) & (output == 0)
    if np.any(unknown):
        values = sorted(int(value) for value in np.unique(labels[unknown]))
        raise ValueError(
            f"ObjectGS output contains unmapped label IDs: {values}"
        )
    return output


def _persistent_colors(
    labels: np.ndarray,
    editor_paths: EditorPaths,
) -> np.ndarray:
    store = (
        read_json(editor_paths.regions_summary)
        if editor_paths.regions_summary.is_file()
        else {}
    )
    color_map = persistent_region_color_map(store.get("regions", []))
    output = np.full((labels.shape[0], 3), 184, dtype=np.uint8)
    output[labels == 0] = 52
    for label in np.unique(labels):
        if int(label) <= 0:
            continue
        color = color_map.get(int(label))
        if color is None:
            hue = (int(label) * 137) % 255
            color = (hue, 255 - hue // 2, 96 + hue // 3)
        output[labels == label] = np.asarray(color, dtype=np.uint8)
    return output


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
