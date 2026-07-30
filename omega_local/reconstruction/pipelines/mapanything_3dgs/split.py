"""Persistent-region ownership and identity-only mask cleanup on MapAnything."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.interactive.regions import (
    persistent_region_color_map,
)
from omega_local.segmentation.split_splat.anchored_ownership import (
    PROVENANCE_UNKNOWN,
    assign_point_labels,
    complete_point_labels,
)
from omega_local.segmentation.split_splat.anchored_split import (
    _relabel_components,
)
from omega_local.segmentation.split_splat.artifacts import (
    write_label_artifacts,
)
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
)

from .artifacts import register_split_artifacts
from .contract import MapAnything3DGSConfig, MapAnything3DGSPaths
from .geometry import (
    FRONT_BAND_ABSOLUTE,
    FRONT_BAND_RELATIVE,
    projected_label_map,
    project_visible_points,
    qvec_to_rotation,
    read_rgb_points,
    write_rgb_ply,
)


ProgressCallback = Callable[[str], None]
_MIN_COMPONENT_SUPPORT = 4
_MIN_COMPONENT_SUPPORT_FRACTION = 0.0005
_MIN_COMPONENT_MAJORITY = 0.55


@dataclass(frozen=True)
class _Frame:
    frame_id: int
    image_stem: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    labels: np.ndarray
    is_manual: bool


def run_split(
    config: MapAnything3DGSConfig,
    editor_paths: EditorPaths,
    paths: MapAnything3DGSPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Assign sparse point ownership and correct propagated component IDs."""
    regions = _load_regions(editor_paths)
    region_ids = sorted(
        int(row["id"])
        for row in regions.get("regions", [])
        if int(row.get("id", 0)) > 0
    )
    if not region_ids:
        raise ValueError("No persistent regions exist in the editor.")
    complete_frames = {
        int(frame_id)
        for frame_id, row in regions.get("frameStates", {}).items()
        if str(frame_id).isdigit() and bool(row.get("complete"))
    }
    if not complete_frames:
        raise ValueError(
            "MapAnything Split needs at least one completed manual frame."
        )

    propagation_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / config.propagation_method
    )
    propagation_summary_path = propagation_dir / "summary.json"
    if not propagation_summary_path.is_file():
        raise FileNotFoundError(
            f"Propagation summary is missing: {propagation_summary_path}"
        )
    propagation_summary = read_json(propagation_summary_path)
    if str(propagation_summary.get("labelSpace") or "") != "persistent_region":
        raise ValueError(
            f"{config.propagation_method} does not use persistent-region IDs."
        )

    frames = _load_frames(
        paths,
        editor_paths,
        propagation_dir / "label_maps",
        complete_frames,
    )
    observed_ids = {
        int(value)
        for frame in frames
        for value in np.unique(frame.labels)
        if int(value) > 0
    }
    unknown = sorted(observed_ids.difference(region_ids))
    if unknown:
        raise ValueError(
            "Masks reference persistent region IDs missing from regions.json: "
            + ", ".join(str(value) for value in unknown)
        )
    active_ids = sorted(observed_ids.intersection(region_ids))
    if not active_ids:
        raise ValueError("No persistent regions appear in the input masks.")

    _reset_outputs(paths)
    points, source_colors = read_rgb_points(paths.point_cloud)
    labels, confidence, provenance, vote_summary = _vote_points(
        points,
        frames,
        active_ids,
        manual_frame_weight=config.manual_frame_weight,
        progress=progress,
    )
    np.save(paths.point_labels, labels)
    np.save(paths.point_confidence, confidence)
    np.save(paths.point_provenance, provenance)
    point_summary = _write_point_artifacts(
        paths,
        points,
        source_colors,
        labels,
        regions,
    )
    mask_summary = _write_clean_masks(
        paths,
        frames,
        points,
        labels,
        active_ids,
        progress=progress,
    )
    training_weights = _write_training_weights(
        paths,
        frames,
        config.manual_frame_weight,
    )

    summary = {
        "schemaVersion": 1,
        "stage": "split",
        "timestampUtc": now_utc(),
        "method": (
            "Manual-weighted persistent-region voting on MapAnything with "
            "identity-only connected-component mask cleanup"
        ),
        "displayName": f"MapAnything Split: {config.propagation_method}",
        "runId": config.run_id,
        "labelNamespace": "persistent_region",
        "geometryCarrier": "mapanything",
        "pointCloud": str(paths.point_cloud),
        "sourceMethodId": config.propagation_method,
        "sourcePropagationSummary": str(propagation_summary_path),
        "manualFrameIds": sorted(complete_frames),
        "manualFrameWeight": config.manual_frame_weight,
        "instanceIds": active_ids,
        "instanceCount": len(active_ids),
        "points": point_summary,
        "pointVoting": vote_summary,
        "frames": mask_summary["frames"],
        "maskCleanup": mask_summary["cleanup"],
        "labelMapsDir": str(paths.clean_label_maps),
        "overlaysDir": str(paths.clean_overlays),
        "binaryInstanceMasksDir": str(paths.split_instances_dir),
        "trainingViewWeightsPath": str(paths.training_view_weights),
        "trainingViewWeightPolicy": training_weights["policy"],
        "layerDescription": (
            "Manual masks pass through exactly. Propagated foreground shapes "
            "are preserved; only a whole connected component ID may change "
            "when projected MapAnything ownership has decisive support."
        ),
    }
    atomic_write_json(paths.stage_summary("split"), summary)
    register_split_artifacts(config, paths, editor_paths, summary)
    return summary


def _load_regions(editor_paths: EditorPaths) -> dict[str, Any]:
    if not editor_paths.regions_summary.is_file():
        raise FileNotFoundError(
            f"Persistent-region summary is missing: {editor_paths.regions_summary}"
        )
    return json.loads(
        editor_paths.regions_summary.read_text(encoding="utf-8")
    )


def _load_frames(
    paths: MapAnything3DGSPaths,
    editor_paths: EditorPaths,
    propagation_maps: Path,
    complete_frames: set[int],
) -> list[_Frame]:
    frames = []
    for row in read_jsonl(paths.frame_map):
        frame_id = int(row["frameId"])
        manual = frame_id in complete_frames
        source = (
            editor_paths.region_maps_dir / f"{frame_id:06d}.npy"
            if manual
            else propagation_maps / f"{frame_id:06d}.npy"
        )
        labels = _load_label_map(source)
        shape = (int(row["height"]), int(row["width"]))
        if labels.shape != shape:
            raise ValueError(
                f"Frame {frame_id} labels have shape {labels.shape}; expected {shape}."
            )
        frames.append(
            _Frame(
                frame_id=frame_id,
                image_stem=Path(str(row["imageName"])).stem,
                width=shape[1],
                height=shape[0],
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                rotation=qvec_to_rotation(
                    np.asarray(row["qvec"], dtype=np.float64)
                ).astype(np.float32),
                translation=np.asarray(row["tvec"], dtype=np.float32),
                labels=labels,
                is_manual=manual,
            )
        )
    return frames


def _load_label_map(path: Path) -> np.ndarray:
    if path.is_file():
        values = np.load(path, allow_pickle=False)
    else:
        png = path.with_suffix(".png")
        if not png.is_file():
            raise FileNotFoundError(f"Label map is missing: {path}")
        values = np.asarray(Image.open(png))
    values = np.asarray(values)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer label map: {path}")
    if np.any(values < 0) or int(values.max(initial=0)) > 65535:
        raise ValueError(f"Label values are outside uint16: {path}")
    return values.astype(np.uint16, copy=False)


def _vote_points(
    points: np.ndarray,
    frames: list[_Frame],
    region_ids: list[int],
    *,
    manual_frame_weight: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    columns = {region_id: index for index, region_id in enumerate(region_ids)}
    shape = (points.shape[0], len(region_ids))
    manual_votes = np.zeros(shape, dtype=np.uint16)
    propagated_votes = np.zeros(shape, dtype=np.uint16)
    frame_rows = []
    for index, frame in enumerate(frames):
        point_ids, xy, _ = project_visible_points(points, frame)
        values = (
            frame.labels[xy[:, 1], xy[:, 0]]
            if point_ids.size
            else np.empty(0, dtype=np.uint16)
        )
        positive = values > 0
        observed_points = point_ids[positive]
        observed_labels = values[positive]
        observed_columns = np.fromiter(
            (columns[int(value)] for value in observed_labels),
            dtype=np.int64,
            count=observed_labels.size,
        )
        target = manual_votes if frame.is_manual else propagated_votes
        if observed_points.size:
            flat = observed_points * len(region_ids) + observed_columns
            np.add.at(target.reshape(-1), flat, 1)
        frame_rows.append(
            {
                "frameId": frame.frame_id,
                "manual": frame.is_manual,
                "visiblePointCount": int(point_ids.size),
                "positiveObservationCount": int(observed_points.size),
            }
        )
        _report_count(progress, "Accumulated MapAnything votes", index + 1, len(frames))

    labels, confidence, provenance, assignment = assign_point_labels(
        manual_votes,
        propagated_votes,
        np.asarray(region_ids, dtype=np.uint16),
        manual_frame_weight=manual_frame_weight,
    )
    scores = (
        manual_votes.astype(np.uint32) * int(manual_frame_weight)
        + propagated_votes.astype(np.uint32)
    )
    best = scores.max(axis=1)
    tied = (
        np.count_nonzero(scores == best[:, None], axis=1) > 1
    ) & (best > 0)
    labels[tied] = 0
    confidence[tied] = 0.0
    provenance[tied] = PROVENANCE_UNKNOWN
    labels, confidence, provenance, completion = complete_point_labels(
        points,
        labels,
        confidence,
        provenance,
        np.asarray(region_ids, dtype=np.uint16),
    )
    return labels, confidence, provenance, {
        "equation": "manual_weight * manual_votes + propagated_votes",
        "manualFrameWeight": int(manual_frame_weight),
        "backgroundPolicy": "zero_is_unknown_and_casts_no_negative_vote",
        "unknownPointPolicy": (
            "unseen or tied points inherit confidence-weighted local 3D "
            "consensus so initializer geometry is not dropped"
        ),
        "frontBandAbsolute": FRONT_BAND_ABSOLUTE,
        "frontBandRelative": FRONT_BAND_RELATIVE,
        "pointCount": int(points.shape[0]),
        "labeledPointCount": int(np.count_nonzero(labels)),
        "manualPointCount": int(np.count_nonzero(provenance == 2)),
        "propagatedPointCount": int(np.count_nonzero(provenance == 1)),
        "unassignedPointCount": int(np.count_nonzero(labels == 0)),
        "tiedPointCount": int(np.count_nonzero(tied)),
        "directAssignment": assignment,
        "completion": completion,
        "frames": frame_rows,
    }


def _write_point_artifacts(
    paths: MapAnything3DGSPaths,
    points: np.ndarray,
    source_colors: np.ndarray,
    labels: np.ndarray,
    region_summary: dict[str, Any],
) -> dict[str, Any]:
    color_map = persistent_region_color_map(
        list(region_summary.get("regions", []))
    )
    semantic = np.full((points.shape[0], 3), 150, dtype=np.uint8)
    for region_id, color in color_map.items():
        semantic[labels == region_id] = color
    np.savez_compressed(
        paths.point_cache,
        points=points.astype(np.float32),
        labels=labels.astype(np.int32),
        colors=semantic,
        source_colors=source_colors,
    )
    write_rgb_ply(points, semantic, paths.segmented_points)

    region_rows = []
    for region_id in sorted(int(value) for value in np.unique(labels) if value > 0):
        selected = labels == region_id
        directory = paths.split_instances_dir / str(region_id)
        directory.mkdir(parents=True, exist_ok=True)
        point_path = directory / f"region_{region_id}.ply"
        write_rgb_ply(points[selected], source_colors[selected], point_path)
        region_rows.append(
            {
                "regionId": region_id,
                "pointCount": int(np.count_nonzero(selected)),
                "pointCloud": str(point_path),
            }
        )
    return {
        "globalPointCount": int(points.shape[0]),
        "labeledPointCount": int(np.count_nonzero(labels)),
        "unlabeledPointCount": int(np.count_nonzero(labels == 0)),
        "labeledFraction": float(np.count_nonzero(labels) / points.shape[0]),
        "pointLabels": str(paths.point_labels),
        "pointConfidence": str(paths.point_confidence),
        "pointProvenance": str(paths.point_provenance),
        "pointsCachePath": str(paths.point_cache),
        "segmentedPointsPath": str(paths.segmented_points),
        "regions": region_rows,
    }


def _write_clean_masks(
    paths: MapAnything3DGSPaths,
    frames: list[_Frame],
    points: np.ndarray,
    point_labels: np.ndarray,
    region_ids: list[int],
    *,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    frame_rows = []
    transitions: dict[tuple[int, int], int] = {}
    for index, frame in enumerate(frames):
        point_ids, xy, depth = project_visible_points(points, frame)
        support = projected_label_map(
            xy,
            depth,
            point_labels[point_ids],
            frame.width,
            frame.height,
        )
        write_label_artifacts(
            paths.projected_support_label_maps,
            paths.projected_support_overlays,
            frame.frame_id,
            support,
        )
        if frame.is_manual:
            output = frame.labels.copy()
            cleanup = {
                "componentCount": 0,
                "reassignedComponentCount": 0,
                "changedPixelCount": 0,
                "reassignments": [],
            }
        else:
            output, cleanup = _relabel_components(
                frame.labels,
                support,
                minimum_support=_MIN_COMPONENT_SUPPORT,
                minimum_support_fraction=_MIN_COMPONENT_SUPPORT_FRACTION,
                minimum_majority=_MIN_COMPONENT_MAJORITY,
            )
        if np.count_nonzero(output) != np.count_nonzero(frame.labels):
            raise RuntimeError(
                f"Identity cleanup changed foreground shape in frame {frame.frame_id}."
            )
        write_label_artifacts(
            paths.clean_label_maps,
            paths.clean_overlays,
            frame.frame_id,
            output,
        )
        for region_id in region_ids:
            mask = output == region_id
            if not np.any(mask):
                continue
            directory = paths.split_instances_dir / str(region_id) / "masks"
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                directory / f"{frame.image_stem}.png"
            )
        for row in cleanup.get("reassignments", []):
            key = (
                int(row["sourceRegionId"]),
                int(row["targetRegionId"]),
            )
            transitions[key] = transitions.get(key, 0) + int(
                row["pixelCount"]
            )
        frame_rows.append(
            {
                "frameId": frame.frame_id,
                "isManualAnchor": frame.is_manual,
                "labelMapPath": str(
                    paths.clean_label_maps / f"{frame.frame_id:06d}.png"
                ),
                "coverage": float(
                    np.count_nonzero(output) / max(output.size, 1)
                ),
                "supportCoverage": float(
                    np.count_nonzero(support) / max(support.size, 1)
                ),
                **cleanup,
            }
        )
        _report_count(progress, "Cleaned MapAnything masks", index + 1, len(frames))
    return {
        "frames": frame_rows,
        "cleanup": {
            "policy": "connected_component_identity_only",
            "manualFramePolicy": "exact_passthrough",
            "foregroundShapePolicy": "preserved",
            "minimumProjectedSupport": _MIN_COMPONENT_SUPPORT,
            "minimumProjectedSupportFraction": _MIN_COMPONENT_SUPPORT_FRACTION,
            "minimumProjectedMajority": _MIN_COMPONENT_MAJORITY,
            "transitionPixelTotals": [
                {
                    "sourceRegionId": source,
                    "targetRegionId": target,
                    "pixelCount": count,
                }
                for (source, target), count in sorted(transitions.items())
            ],
        },
    }

def _write_training_weights(
    paths: MapAnything3DGSPaths,
    frames: list[_Frame],
    manual_weight: int,
) -> dict[str, Any]:
    payload = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "policy": (
            "completed manual frames receive the configured mask-loss weight; "
            "propagated frames receive weight 1"
        ),
        "manualFrameWeight": int(manual_weight),
        "frames": [
            {
                "frameId": frame.frame_id,
                "imageStem": frame.image_stem,
                "isManualAnchor": frame.is_manual,
                "weight": int(manual_weight) if frame.is_manual else 1,
            }
            for frame in frames
        ],
    }
    atomic_write_json(paths.training_view_weights, payload)
    return payload

def _reset_outputs(paths: MapAnything3DGSPaths) -> None:
    if paths.split_dir.exists():
        shutil.rmtree(paths.split_dir)
    for directory in (
        paths.point_labels_dir,
        paths.projected_support_label_maps,
        paths.projected_support_overlays,
        paths.clean_label_maps,
        paths.clean_overlays,
        paths.split_instances_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

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
