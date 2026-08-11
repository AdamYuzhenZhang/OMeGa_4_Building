"""Editor registrations for MapAnything Split and region 3DGS outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from omega_local.reconstruction.gaussian_diagnostics import (
    summarize_gaussian_ply,
)

from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.reconstruction.gaussian_io import ensure_viewer_gaussian_ply
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    replace_symlink,
    write_jsonl,
)

from .contract import MapAnything3DGSConfig, MapAnything3DGSPaths


def register_split_artifacts(
    config: MapAnything3DGSConfig,
    paths: MapAnything3DGSPaths,
    editor_paths: EditorPaths,
    summary: dict[str, Any],
) -> None:
    register_mask_layer(
        editor_paths,
        method_id=f"{config.run_id}_split",
        display_name="MapAnything Identity-Cleaned Masks",
        description=str(summary["layerDescription"]),
        label_maps=paths.clean_label_maps,
        overlays=paths.clean_overlays,
        frames=summary["frames"],
        canonical_run_dir=paths.run_dir,
        canonical_summary=paths.stage_summary("split"),
        layer_group="refined_masks",
    )
    register_mask_layer(
        editor_paths,
        method_id=f"{config.run_id}_support",
        display_name="MapAnything Projected Region Support",
        description=(
            "Sparse z-buffered projection of persistent-region ownership on "
            "the MapAnything initializer. This is evidence, not a dense mask."
        ),
        label_maps=paths.projected_support_label_maps,
        overlays=paths.projected_support_overlays,
        frames=summary["frames"],
        canonical_run_dir=paths.run_dir,
        canonical_summary=paths.stage_summary("split"),
        layer_group="geometry_support",
    )
    run_id = f"{config.run_id}_split"
    experiment = editor_paths.segmentation3d_dir / "runs" / run_id
    experiment.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        experiment / "experiment.json",
        {
            "schemaVersion": 1,
            "runId": run_id,
            "methodId": "mapanything_region_3dgs",
            "experimentFamily": "mapanything_region_3dgs",
            "artifactRole": "split_labels",
            "baseRunId": config.run_id,
            "inputId": config.propagation_method,
            "displayName": f"MapAnything Split: {config.run_id}",
            "labelSpace": "persistent_region",
            "pointsCachePath": str(paths.point_cache),
            "geometrySource": "omega_mapanything_initializer",
            "manualFrameWeight": config.manual_frame_weight,
            "ready": True,
            "timestampUtc": now_utc(),
            "canonicalRunDir": str(paths.run_dir),
            "pointSummary": summary["points"],
        },
    )


def register_splat_artifacts(
    config: MapAnything3DGSConfig,
    paths: MapAnything3DGSPaths,
    editor_paths: EditorPaths,
    summary: dict[str, Any],
) -> dict[str, Any]:
    ensure_viewer_gaussian_ply(paths.composed_scene, paths.viewer_scene)
    diagnostics = {
        "schemaVersion": 1,
        "composedScene": summarize_gaussian_ply(paths.composed_scene),
        "regions": [],
    }
    viewer_regions: dict[int, Path] = {}
    for row in summary["regions"]:
        region_id = int(row["regionId"])
        source = Path(str(row["pointCloud"]))
        viewer = paths.viewer_regions / f"region_{region_id}.ply"
        ensure_viewer_gaussian_ply(source, viewer)
        viewer_regions[region_id] = viewer
        diagnostics["regions"].append(
            {
                "regionId": region_id,
                **summarize_gaussian_ply(source),
            }
        )
    atomic_write_json(paths.appearance_diagnostics, diagnostics)

    run_id = f"{config.run_id}_splat"
    experiment = editor_paths.segmentation3d_dir / "runs" / run_id
    experiment.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Any]] = {
        "appearance": {
            "displayName": "Composed RGB 3DGS",
            "path": str(paths.viewer_scene),
            "format": "3dgs_ply",
            "pointCount": int(summary["gaussianCount"]),
            "artifactGroup": "composed_scene",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        },
        "region_ids": {
            "displayName": "Composed Persistent Region IDs",
            "path": str(paths.composed_region_ids),
            "format": "3dgs_ply",
            "pointCount": int(summary["gaussianCount"]),
            "labelCount": int(summary["regionCount"]),
            "labeledPointCount": int(summary["gaussianCount"]),
            "artifactGroup": "composed_scene",
            "colorSpace": "persistent_region",
            "radiometricAppearance": False,
        },
    }
    for row in summary["regions"]:
        region_id = int(row["regionId"])
        artifacts[f"rgb_region_{region_id}"] = {
            "displayName": f"Region {region_id} RGB",
            "path": str(viewer_regions[region_id]),
            "format": "3dgs_ply",
            "pointCount": int(row["gaussianCount"]),
            "instanceId": region_id,
            "artifactGroup": "rgb_objects",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    payload = {
        "schemaVersion": 1,
        "runId": run_id,
        "methodId": "mapanything_region_3dgs",
        "experimentFamily": "mapanything_region_3dgs",
        "artifactRole": "reconstruction",
        "baseRunId": config.run_id,
        "reconstructionVariant": "mapanything_point_initialized_3dgs",
        "inputId": config.propagation_method,
        "displayName": f"MapAnything Region 3DGS: {config.run_id}",
        "labelSpace": "persistent_region",
        "geometrySource": "omega_mapanything_initializer",
        "manualFrameWeight": config.manual_frame_weight,
        "ready": True,
        "timestampUtc": now_utc(),
        "canonicalRunDir": str(paths.run_dir),
        "pointSummary": {
            "globalPointCount": int(summary["gaussianCount"]),
            "labeledGlobalPointCount": int(summary["gaussianCount"]),
        },
        "gaussianArtifacts": artifacts,
        "appearanceDiagnostics": str(paths.appearance_diagnostics),
    }
    atomic_write_json(experiment / "experiment.json", payload)
    return {
        "experimentPath": str(experiment / "experiment.json"),
        "runId": run_id,
        "artifactCount": len(artifacts),
    }


def register_mask_layer(
    editor_paths: EditorPaths,
    *,
    method_id: str,
    display_name: str,
    description: str,
    label_maps: Path,
    overlays: Path,
    frames: list[dict[str, Any]],
    canonical_run_dir: Path,
    canonical_summary: Path,
    layer_group: str,
) -> None:
    layer_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / method_id
    )
    layer_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(layer_dir / "label_maps", label_maps)
    replace_symlink(layer_dir / "overlays", overlays)
    metadata_dir = layer_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for row in frames:
        frame_id = int(row["frameId"])
        npy_path = label_maps / f"{frame_id:06d}.npy"
        labels = (
            np.asarray(np.load(npy_path, allow_pickle=False), dtype=np.uint16)
            if npy_path.is_file()
            else np.asarray(
                Image.open(label_maps / f"{frame_id:06d}.png"),
                dtype=np.uint16,
            )
        )
        label_rows = [
            {
                "labelId": int(label_id),
                "pixelCount": int(np.count_nonzero(labels == label_id)),
                "layer": f"propagation_{method_id}",
                "labelSpace": "persistent_region",
                "proposalKind": layer_group,
            }
            for label_id in np.unique(labels)
            if int(label_id) > 0
        ]
        metadata = {
            "frameId": frame_id,
            "labelSpace": "persistent_region",
            "labelCount": len(label_rows),
            "coverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
            "labels": label_rows,
        }
        atomic_write_json(metadata_dir / f"{frame_id:06d}.json", metadata)
        frame_rows.append(metadata)
    write_jsonl(layer_dir / "frames.jsonl", frame_rows)
    atomic_write_json(
        layer_dir / "summary.json",
        {
            "schemaVersion": 1,
            "timestampUtc": now_utc(),
            "methodId": method_id,
            "displayName": display_name,
            "engineName": "MapAnything Region 3DGS",
            "description": description,
            "labelSpace": "persistent_region",
            "layerGroup": layer_group,
            "stage": "mapanything_split",
            "readOnly": True,
            "frameCount": len(frame_rows),
            "completedFrameCount": len(frame_rows),
            "canonicalRunDir": str(canonical_run_dir),
            "canonicalSummary": str(canonical_summary),
        },
    )
    atomic_write_json(
        layer_dir / "progress.json",
        {
            "status": "complete",
            "running": False,
            "failed": False,
            "message": f"{display_name} ready",
            "updatedUtc": now_utc(),
        },
    )
