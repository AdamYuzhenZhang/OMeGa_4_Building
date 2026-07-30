"""Persistent-ID-preserving 3D voting for Split&Splat experiments."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.paths import EditorPaths

from .adapter import _read_colmap_images_text
from .artifacts import (
    finalize_prepared_split,
    register_split_support_layer,
    write_label_artifacts,
)
from .anchored_ownership import (
    PROVENANCE_GEOMETRY,
    PROVENANCE_MANUAL,
    assign_point_labels as _assign_point_labels,
    complete_point_labels,
)
from .contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
)


ProgressCallback = Callable[[str], None]

_FRONT_BAND_ABSOLUTE = 0.05
_FRONT_BAND_RELATIVE = 0.02
_MIN_COMPONENT_SUPPORT = 4
_MIN_COMPONENT_SUPPORT_FRACTION = 0.0005
_MIN_COMPONENT_MAJORITY = 0.55


@dataclass(frozen=True)
class _Frame:
    frame_id: int
    stem: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray
    translation: np.ndarray
    labels: np.ndarray
    is_manual_anchor: bool
    view_weight: int


def run_anchored_split(
    config: SplitSplatRunConfig,
    editor_paths: EditorPaths,
    run_paths: SplitSplatRunPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Partition the complete global 3DGS and relabel masks in place."""
    proposal_summary = read_json(run_paths.stage_summary("proposals"))
    if proposal_summary.get("sourceLabelSpace") != "persistent_region":
        raise ValueError(
            "anchored_3d requires proposal maps in the persistent_region label space."
        )

    region_summary = _load_region_summary(editor_paths)
    complete_frame_ids = _complete_frame_ids(region_summary, editor_paths)
    if not complete_frame_ids:
        raise ValueError(
            "anchored_3d needs at least one completed manual region frame."
        )
    region_ids = sorted(
        {
            int(row["id"])
            for row in region_summary.get("regions", [])
            if int(row.get("id", 0)) > 0
        }
    )
    if not region_ids:
        raise ValueError("The editor has no persistent regions to anchor.")

    frames = _load_frames(
        config,
        editor_paths,
        run_paths,
        complete_frame_ids=complete_frame_ids,
    )
    observed_ids = sorted(
        {
            int(value)
            for frame in frames
            for value in np.unique(frame.labels)
            if int(value) > 0
        }
    )
    unknown_ids = sorted(set(observed_ids).difference(region_ids))
    if unknown_ids:
        raise ValueError(
            "Propagation maps reference persistent IDs missing from regions.json: "
            + ", ".join(str(value) for value in unknown_ids)
        )
    active_ids = sorted(set(region_ids).intersection(observed_ids))
    if not active_ids:
        raise ValueError("The anchored masks contain no known persistent regions.")

    _reset_output_dirs(run_paths)
    points = _read_global_points(run_paths.global_point_cloud)
    if progress is not None:
        progress(
            f"Voting {len(active_ids)} persistent regions onto "
            f"{points.shape[0]:,} global Gaussian means."
        )
    point_labels, confidence, provenance, vote_summary = _vote_point_labels(
        points,
        frames,
        active_ids,
        manual_frame_weight=config.manual_frame_weight,
        progress=progress,
    )
    missing_regions = vote_summary["geometryCompletion"][
        "regionIdsWithoutDirectSeeds"
    ]
    if missing_regions:
        raise RuntimeError(
            "The global 3DGS has no direct front-surface evidence for persistent "
            "regions: "
            + ", ".join(str(value) for value in missing_regions)
        )
    np.save(run_paths.point_label_confidence, confidence)
    np.save(run_paths.point_label_provenance, provenance)

    point_instance_ids = _write_instance_point_clouds(
        run_paths,
        points,
        point_labels,
        active_ids,
    )
    (
        mask_instance_ids,
        relabel_summaries,
        support_summaries,
    ) = _write_shape_preserving_masks(
        run_paths,
        frames,
        points,
        point_labels,
        provenance,
        active_ids,
        progress=progress,
    )
    training_weights = _write_training_weights(
        run_paths,
        frames,
        manual_frame_weight=config.manual_frame_weight,
    )
    source_display_name = str(
        proposal_summary.get("sourceDisplayName")
        or proposal_summary.get("sourceMethodId")
        or config.propagation_method
    )

    summary = finalize_prepared_split(
        run_paths,
        editor_paths,
        point_instance_ids=point_instance_ids,
        mask_instance_ids=mask_instance_ids,
        method=(
            "OMeGa complete global-Gaussian ownership with shape-preserving "
            "persistent-region correction"
        ),
        display_name=f"Anchored 3D Split: {source_display_name}",
        label_namespace="persistent_region",
        global_point_labels=point_labels,
        extra_summary={
            "splitMethod": "anchored_3d",
            "sourcePersistentRegionIds": active_ids,
            "manualFrameIds": sorted(complete_frame_ids),
            "manualFrameWeight": int(config.manual_frame_weight),
            "pointVoting": vote_summary,
            "relabeling": {
                "policy": "connected_component_complete_3d_majority",
                "shapePolicy": "foreground_union_preserved",
                "manualFramePolicy": "exact_passthrough",
                "evidencePolicy": (
                    "the completed global 3DGS ownership field may relabel "
                    "propagated connected components"
                ),
                "minimumProjectedSupport": _MIN_COMPONENT_SUPPORT,
                "minimumProjectedSupportFraction": (
                    _MIN_COMPONENT_SUPPORT_FRACTION
                ),
                "minimumProjectedMajority": _MIN_COMPONENT_MAJORITY,
                "transitionTotals": _aggregate_reassignments(
                    relabel_summaries
                ),
                "frames": relabel_summaries,
            },
            "projectedSupport": {
                "policy": "complete_global_3dgs_ownership",
                "labelMapsDir": str(run_paths.projected_support_label_maps),
                "overlaysDir": str(run_paths.projected_support_overlays),
                "frames": support_summaries,
            },
            "trainingViewWeightsPath": str(run_paths.training_view_weights),
            "trainingViewWeightPolicy": training_weights["policy"],
            "exclusivePreviewConflictPolicy": "not_applicable_exclusive_masks",
            "layerDescription": (
                "Persistent-region masks whose foreground shapes come directly "
                "from completed manual frames or the selected propagation. The "
                "shared global 3DGS may relabel connected components, but no SAM2 "
                "refinement changes their boundaries."
            ),
            "geometryVisibility": {
                "policy": "global_3dgs_front_surface_z_buffer",
                "externalDepthGate": False,
                "frontBandAbsolute": _FRONT_BAND_ABSOLUTE,
                "frontBandRelative": _FRONT_BAND_RELATIVE,
            },
        },
        progress=progress,
    )
    register_split_support_layer(
        run_paths,
        editor_paths,
        label_maps_dir=run_paths.projected_support_label_maps,
        overlays_dir=run_paths.projected_support_overlays,
        frame_summaries=support_summaries,
        label_namespace="persistent_region",
    )
    return summary


def _load_region_summary(editor_paths: EditorPaths) -> dict[str, Any]:
    if not editor_paths.regions_summary.is_file():
        raise FileNotFoundError(
            f"Persistent-region summary does not exist: {editor_paths.regions_summary}"
        )
    return json.loads(editor_paths.regions_summary.read_text(encoding="utf-8"))


def _complete_frame_ids(
    summary: dict[str, Any],
    editor_paths: EditorPaths,
) -> set[int]:
    states = summary.get("frameStates", {})
    return {
        int(frame_id)
        for frame_id, row in states.items()
        if str(frame_id).isdigit()
        and bool(row.get("complete"))
        and (editor_paths.region_maps_dir / f"{int(frame_id):06d}.npy").is_file()
    }


def _load_frames(
    config: SplitSplatRunConfig,
    editor_paths: EditorPaths,
    run_paths: SplitSplatRunPaths,
    *,
    complete_frame_ids: set[int],
) -> list[_Frame]:
    images = {
        int(row["image_id"]): row
        for row in _read_colmap_images_text(
            run_paths.dataset_dir / "sparse" / "0" / "images.txt"
        )
    }
    frames: list[_Frame] = []
    for row in read_jsonl(run_paths.frame_map):
        frame_id = int(row["frameId"])
        image = images[int(row["colmapImageId"])]
        stem = Path(str(row["splitSplatImageName"])).stem
        is_manual = frame_id in complete_frame_ids
        label_path = (
            editor_paths.region_maps_dir / f"{frame_id:06d}.npy"
            if is_manual
            else run_paths.proposal_label_maps / f"{frame_id:06d}.npy"
        )
        labels = _load_label_map(label_path)
        expected = (int(row["height"]), int(row["width"]))
        if labels.shape != expected:
            raise ValueError(
                f"Anchored frame {frame_id} expects {expected}; "
                f"labels={labels.shape}."
            )
        frames.append(
            _Frame(
                frame_id=frame_id,
                stem=stem,
                width=expected[1],
                height=expected[0],
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                rotation=_qvec_to_rotation(
                    np.asarray(image["qvec"], dtype=np.float64)
                ).astype(np.float32),
                translation=np.asarray(
                    image["tvec"], dtype=np.float32
                ),
                labels=labels,
                is_manual_anchor=is_manual,
                view_weight=(
                    int(config.manual_frame_weight)
                    if is_manual
                    else 1
                ),
            )
        )
    return frames


def _load_label_map(path: Path) -> np.ndarray:
    if not path.is_file():
        png = path.with_suffix(".png")
        if not png.is_file():
            raise FileNotFoundError(f"Anchored label map does not exist: {path}")
        values = np.asarray(Image.open(png))
    else:
        values = np.asarray(np.load(path, allow_pickle=False))
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"Expected a 2D integer label map: {path}")
    if np.any(values < 0) or int(values.max(initial=0)) > np.iinfo(np.uint16).max:
        raise ValueError(f"Label map contains values outside uint16: {path}")
    return values.astype(np.uint16, copy=False)


def _read_global_points(path: Path) -> np.ndarray:
    from plyfile import PlyData

    vertex = PlyData.read(path, mmap="r")["vertex"].data
    points = np.column_stack(
        [vertex["x"], vertex["y"], vertex["z"]]
    ).astype(np.float32, copy=False)
    if points.shape[0] == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"Global Gaussian PLY has invalid means: {path}")
    return points


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(qvec, dtype=np.float64).tolist()
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _project_visible(
    points: np.ndarray,
    frame: _Frame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (
        points @ frame.rotation.T
        + frame.translation
    )
    depth = camera[:, 2]
    finite = np.all(np.isfinite(camera), axis=1) & (depth > 1.0e-8)
    point_ids = np.flatnonzero(finite)
    if point_ids.size == 0:
        return _empty_projection()

    z = depth[point_ids]
    u_float = camera[point_ids, 0] * frame.fx / z + frame.cx
    v_float = camera[point_ids, 1] * frame.fy / z + frame.cy
    valid_uv = np.isfinite(u_float) & np.isfinite(v_float)
    point_ids = point_ids[valid_uv]
    z = z[valid_uv]
    xy = np.rint(
        np.column_stack((u_float[valid_uv], v_float[valid_uv]))
    ).astype(np.int64)
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
    z_sorted = z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(linear_sorted)) + 1]
    counts = np.diff(np.r_[starts, linear_sorted.size])
    nearest = z_sorted[starts]
    z_tolerance = np.maximum(
        _FRONT_BAND_ABSOLUTE,
        nearest * _FRONT_BAND_RELATIVE,
    )
    front_band = z_sorted <= np.repeat(nearest + z_tolerance, counts)
    visible_rows = order[front_band]
    return (
        point_ids[visible_rows].astype(np.int64, copy=False),
        xy[visible_rows].astype(np.int32, copy=False),
        z[visible_rows].astype(np.float32, copy=False),
    )


def _vote_point_labels(
    points: np.ndarray,
    frames: list[_Frame],
    region_ids: list[int],
    *,
    manual_frame_weight: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    id_to_column = {region_id: index for index, region_id in enumerate(region_ids)}
    manual_votes = np.zeros(
        (points.shape[0], len(region_ids)),
        dtype=np.uint16,
    )
    propagated_votes = np.zeros_like(manual_votes)
    frame_rows = []
    for index, frame in enumerate(frames):
        point_ids, xy, _ = _project_visible(points, frame)
        values = (
            frame.labels[xy[:, 1], xy[:, 0]]
            if point_ids.size
            else np.empty(0, dtype=np.uint16)
        )
        positive = values > 0
        observed_points = point_ids[positive]
        observed_labels = values[positive]
        columns = np.fromiter(
            (id_to_column[int(value)] for value in observed_labels),
            dtype=np.int64,
            count=observed_labels.size,
        )
        target = manual_votes if frame.is_manual_anchor else propagated_votes
        if observed_points.size:
            flat = (
                observed_points * len(region_ids)
                + columns
            )
            np.add.at(target.reshape(-1), flat, 1)
        frame_rows.append(
            {
                "frameId": frame.frame_id,
                "isManualAnchor": frame.is_manual_anchor,
                "viewWeight": frame.view_weight,
                "visiblePointCount": int(point_ids.size),
                "positiveObservationCount": int(observed_points.size),
            }
        )
        _progress_count(progress, "Accumulated anchored 3D votes", index + 1, len(frames))

    labels, confidence, provenance, assignment = _assign_point_labels(
        manual_votes,
        propagated_votes,
        np.asarray(region_ids, dtype=np.uint16),
        manual_frame_weight=manual_frame_weight,
    )
    labels, confidence, provenance, completion = complete_point_labels(
        points,
        labels,
        confidence,
        provenance,
        np.asarray(region_ids, dtype=np.uint16),
    )
    return labels, confidence, provenance, {
        "equation": "manual_weight * manual_votes + propagated_votes",
        "manualPolicy": (
            "Manual-vote winners are authoritative; weighted propagated "
            "evidence breaks manual ties."
        ),
        "backgroundPolicy": "zero_is_unknown_and_casts_no_negative_vote",
        "assignmentPolicy": (
            "every directly observed Gaussian is assigned; confidence records "
            "ambiguity but never rejects ownership"
        ),
        "completionPolicy": (
            "Gaussians with no direct observation inherit a local weighted "
            "consensus from directly labeled Gaussian neighbors"
        ),
        "manualFrameWeight": int(manual_frame_weight),
        "frontBandAbsolute": _FRONT_BAND_ABSOLUTE,
        "frontBandRelative": _FRONT_BAND_RELATIVE,
        "externalDepthGate": False,
        "pointCount": int(points.shape[0]),
        "labeledPointCount": int(np.count_nonzero(labels)),
        "manualPointCount": int(
            np.count_nonzero(provenance == PROVENANCE_MANUAL)
        ),
        "propagatedPointCount": int(np.count_nonzero(provenance == 1)),
        "geometryCompletedPointCount": int(
            np.count_nonzero(provenance == PROVENANCE_GEOMETRY)
        ),
        "frames": frame_rows,
        "directAssignment": assignment,
        "geometryCompletion": completion,
    }


def _reset_output_dirs(run_paths: SplitSplatRunPaths) -> None:
    for path in (
        run_paths.split_raw_output,
        run_paths.projected_support_dir,
        run_paths.consistent_masks_dir,
    ):
        if path.exists():
            shutil.rmtree(path)
    run_paths.split_raw_output.mkdir(parents=True, exist_ok=True)
    run_paths.point_labels_dir.mkdir(parents=True, exist_ok=True)
    run_paths.projected_support_label_maps.mkdir(parents=True, exist_ok=True)
    run_paths.projected_support_overlays.mkdir(parents=True, exist_ok=True)


def _write_instance_point_clouds(
    run_paths: SplitSplatRunPaths,
    points: np.ndarray,
    labels: np.ndarray,
    region_ids: list[int],
) -> list[int]:
    from plyfile import PlyData, PlyElement

    dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4")])
    written = []
    for region_id in region_ids:
        selected = points[labels == region_id]
        if selected.shape[0] == 0:
            continue
        directory = run_paths.split_raw_output / str(region_id)
        directory.mkdir(parents=True, exist_ok=True)
        vertex = np.empty(selected.shape[0], dtype=dtype)
        vertex["x"], vertex["y"], vertex["z"] = selected.T
        PlyData(
            [PlyElement.describe(vertex, "vertex")],
            text=False,
            byte_order="<",
        ).write(directory / f"label_{region_id}.ply")
        written.append(region_id)
    return written


def _write_shape_preserving_masks(
    run_paths: SplitSplatRunPaths,
    frames: list[_Frame],
    points: np.ndarray,
    point_labels: np.ndarray,
    point_provenance: np.ndarray,
    region_ids: list[int],
    *,
    progress: ProgressCallback | None,
) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    mask_ids: set[int] = set()
    frame_rows = []
    support_rows = []
    for index, frame in enumerate(frames):
        point_ids, xy, depth = _project_visible(points, frame)
        values = point_labels[point_ids]
        support = _projected_label_map(
            xy,
            depth,
            values,
            width=frame.width,
            height=frame.height,
        )
        write_label_artifacts(
            run_paths.projected_support_label_maps,
            run_paths.projected_support_overlays,
            frame.frame_id,
            support,
        )
        if frame.is_manual_anchor:
            output = frame.labels.copy()
            stats = {
                "componentCount": 0,
                "insufficientSupportComponentCount": 0,
                "supportedComponentCount": 0,
                "reassignedComponentCount": 0,
                "changedPixelCount": 0,
                "reassignments": [],
            }
        else:
            output, stats = _relabel_components(
                frame.labels,
                support,
                minimum_support=_MIN_COMPONENT_SUPPORT,
                minimum_support_fraction=_MIN_COMPONENT_SUPPORT_FRACTION,
                minimum_majority=_MIN_COMPONENT_MAJORITY,
            )
        if np.count_nonzero(output) != np.count_nonzero(frame.labels):
            raise RuntimeError(
                f"Anchored relabeling changed foreground coverage in frame {frame.frame_id}."
            )
        visible_ids = [
            int(value)
            for value in np.unique(output)
            if int(value) > 0
        ]
        mask_ids.update(visible_ids)
        for region_id in visible_ids:
            directory = run_paths.split_raw_output / str(region_id)
            directory.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                (output == region_id).astype(np.uint8) * 255
            ).save(directory / f"{frame.stem}.png")
        frame_rows.append(
            {
                "frameId": frame.frame_id,
                "isManualAnchor": frame.is_manual_anchor,
                "sourceCoverage": float(
                    np.count_nonzero(frame.labels) / max(frame.labels.size, 1)
                ),
                "outputCoverage": float(
                    np.count_nonzero(output) / max(output.size, 1)
                ),
                **stats,
            }
        )
        support_rows.append(
            {
                "frameId": frame.frame_id,
                "supportPointCount": int(np.count_nonzero(values)),
                "directSupportPointCount": int(
                    np.count_nonzero(
                        point_provenance[point_ids] != PROVENANCE_GEOMETRY
                    )
                ),
                "geometryCompletedSupportPointCount": int(
                    np.count_nonzero(
                        point_provenance[point_ids] == PROVENANCE_GEOMETRY
                    )
                ),
                "supportPixelCount": int(np.count_nonzero(support)),
                "supportCoverage": float(
                    np.count_nonzero(support) / max(support.size, 1)
                ),
            }
        )
        _progress_count(progress, "Relabeled anchored masks", index + 1, len(frames))
    return sorted(mask_ids), frame_rows, support_rows


def _projected_label_map(
    xy: np.ndarray,
    depth: np.ndarray,
    labels: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    output = np.zeros((height, width), dtype=np.uint16)
    positive = labels > 0
    if not np.any(positive):
        return output
    xy = xy[positive]
    depth = depth[positive]
    labels = labels[positive]
    linear = xy[:, 1].astype(np.int64) * width + xy[:, 0]
    order = np.lexsort((depth, linear))
    linear_sorted = linear[order]
    first = np.r_[0, np.flatnonzero(np.diff(linear_sorted)) + 1]
    selected = order[first]
    output[xy[selected, 1], xy[selected, 0]] = labels[selected]
    return output


def _relabel_components(
    source: np.ndarray,
    support: np.ndarray,
    *,
    minimum_support: int,
    minimum_support_fraction: float,
    minimum_majority: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Relabel source components without changing their foreground pixels."""
    source = np.asarray(source, dtype=np.uint16)
    support = np.asarray(support, dtype=np.uint16)
    if source.shape != support.shape:
        raise ValueError("Source and projected support maps must have the same shape.")
    output = source.copy()
    component_count = 0
    insufficient_support_count = 0
    supported_count = 0
    reassigned_count = 0
    changed_pixels = 0
    transitions: dict[tuple[int, int], dict[str, int]] = {}
    for source_id in np.unique(source):
        if int(source_id) <= 0:
            continue
        count, components = cv2.connectedComponents(
            (source == source_id).astype(np.uint8),
            connectivity=8,
        )
        for component_id in range(1, count):
            component_count += 1
            component = components == component_id
            component_pixels = int(np.count_nonzero(component))
            observed = support[component]
            observed = observed[observed > 0]
            required_support = max(
                int(minimum_support),
                int(np.ceil(component_pixels * minimum_support_fraction)),
            )
            if observed.size < required_support:
                insufficient_support_count += 1
                continue
            ids, counts = np.unique(observed, return_counts=True)
            best_index = int(np.argmax(counts))
            majority = float(counts[best_index] / observed.size)
            if majority < minimum_majority:
                continue
            supported_count += 1
            target_id = int(ids[best_index])
            if target_id == int(source_id):
                continue
            output[component] = target_id
            reassigned_count += 1
            changed_pixels += component_pixels
            transition = transitions.setdefault(
                (int(source_id), target_id),
                {
                    "componentCount": 0,
                    "pixelCount": 0,
                    "supportPixelCount": 0,
                    "targetSupportPixelCount": 0,
                },
            )
            transition["componentCount"] += 1
            transition["pixelCount"] += component_pixels
            transition["supportPixelCount"] += int(observed.size)
            transition["targetSupportPixelCount"] += int(counts[best_index])
    return output, {
        "componentCount": int(component_count),
        "insufficientSupportComponentCount": int(
            insufficient_support_count
        ),
        "supportedComponentCount": int(supported_count),
        "reassignedComponentCount": int(reassigned_count),
        "changedPixelCount": int(changed_pixels),
        "reassignments": [
            {
                "sourceRegionId": source_id,
                "targetRegionId": target_id,
                **counts,
            }
            for (source_id, target_id), counts in sorted(transitions.items())
        ],
    }


def _aggregate_reassignments(
    frame_rows: list[dict[str, Any]],
) -> list[dict[str, int]]:
    totals: dict[tuple[int, int], dict[str, int]] = {}
    for row in frame_rows:
        for transition in row.get("reassignments", []):
            key = (
                int(transition["sourceRegionId"]),
                int(transition["targetRegionId"]),
            )
            total = totals.setdefault(
                key,
                {
                    "frameCount": 0,
                    "componentCount": 0,
                    "pixelCount": 0,
                    "supportPixelCount": 0,
                    "targetSupportPixelCount": 0,
                },
            )
            total["frameCount"] += 1
            total["componentCount"] += int(transition["componentCount"])
            total["pixelCount"] += int(transition["pixelCount"])
            total["supportPixelCount"] += int(
                transition["supportPixelCount"]
            )
            total["targetSupportPixelCount"] += int(
                transition["targetSupportPixelCount"]
            )
    return [
        {
            "sourceRegionId": source_id,
            "targetRegionId": target_id,
            **counts,
        }
        for (source_id, target_id), counts in sorted(totals.items())
    ]


def _write_training_weights(
    run_paths: SplitSplatRunPaths,
    frames: list[_Frame],
    *,
    manual_frame_weight: int,
) -> dict[str, Any]:
    payload = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "policy": (
            "completed manual frames receive the configured weight; propagated "
            "frames receive weight 1"
        ),
        "manualFrameWeight": int(manual_frame_weight),
        "frames": [
            {
                "frameId": frame.frame_id,
                "imageStem": frame.stem,
                "isManualAnchor": frame.is_manual_anchor,
                "maskSource": (
                    "manual_region"
                    if frame.is_manual_anchor
                    else "propagation"
                ),
                "weight": frame.view_weight,
            }
            for frame in frames
        ],
    }
    atomic_write_json(run_paths.training_view_weights, payload)
    return payload


def _empty_projection() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.empty(0, dtype=np.int64),
        np.empty((0, 2), dtype=np.int32),
        np.empty(0, dtype=np.float32),
    )


def _progress_count(
    callback: ProgressCallback | None,
    label: str,
    completed: int,
    total: int,
) -> None:
    if callback is None:
        return
    if completed == 1 or completed % 10 == 0 or completed == total:
        callback(f"{label} {completed}/{total}.")
