"""Canonicalize upstream Split&Splat outputs and register editor views."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from omega_local.reconstruction.gaussian_io import (
    ensure_viewer_gaussian_ply,
    write_segmented_gaussian_ply,
)
from omega_local.segmentation.color_palette import categorical_rgb
from omega_local.segmentation.interactive.paths import EditorPaths

from .contract import (
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
    replace_symlink,
    write_jsonl,
)


def finalize_official_proposals(
    run_paths: SplitSplatRunPaths,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    work_output = (
        run_paths.proposals_dir
        / "work"
        / "output"
        / f"{run_paths.run_id}_autoseg_mask"
    )
    if not work_output.is_dir():
        raise FileNotFoundError(f"Upstream automatic masks do not exist: {work_output}")
    if run_paths.proposal_binary_masks.exists():
        shutil.rmtree(run_paths.proposal_binary_masks)
    shutil.move(str(work_output), str(run_paths.proposal_binary_masks))
    run_paths.proposal_label_maps.mkdir(parents=True, exist_ok=True)
    run_paths.proposal_overlays.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(run_paths.frame_map)
    diagnostics = []
    if progress is not None:
        progress(
            f"Canonicalizing official proposal masks for {len(rows)} frames."
        )
    for frame_index, row in enumerate(rows):
        frame_id = int(row["frameId"])
        split_stem = Path(str(row["splitSplatImageName"])).stem
        mask_dir = run_paths.proposal_binary_masks / split_stem
        label_map, frame_stats = _exclusive_from_binary_masks(
            sorted(mask_dir.glob("*.png"), key=_numeric_path_key),
            width=int(row["width"]),
            height=int(row["height"]),
            conflict_policy="small_masks_overwrite",
        )
        write_label_artifacts(
            run_paths.proposal_label_maps,
            run_paths.proposal_overlays,
            frame_id,
            label_map,
        )
        diagnostics.append({"frameId": frame_id, **frame_stats})
        completed = frame_index + 1
        if progress is not None and (
            completed == 1
            or completed % 10 == 0
            or completed == len(rows)
        ):
            progress(
                f"Canonicalized official proposal frames {completed}/{len(rows)}."
            )

    summary = {
        "schemaVersion": 1,
        "stage": "proposals",
        "timestampUtc": now_utc(),
        "method": "Released Split&Splat four-grid SAM2 automatic masks",
        "proposalSource": "official_auto",
        "sourceMethodId": "split_splat_four_grid_sam2",
        "sourceDisplayName": "Official Four-Grid SAM2",
        "sourceLabelSpace": "frame_local_proposal",
        "settings": {
            "pointsPerSide": [1, 4, 8, 16],
            "source": "third_party/Split_and_Splat/sam2/auto_seg.py",
        },
        "frameCount": len(rows),
        "binaryMasksDir": str(run_paths.proposal_binary_masks),
        "labelMapsDir": str(run_paths.proposal_label_maps),
        "overlaysDir": str(run_paths.proposal_overlays),
        "meanProposalCount": float(np.mean([row["proposalCount"] for row in diagnostics])),
        "meanCoverage": float(np.mean([row["coverage"] for row in diagnostics])),
        "frames": diagnostics,
    }
    atomic_write_json(run_paths.stage_summary("proposals"), summary)
    if progress is not None:
        progress(
            f"Official proposals ready: mean {summary['meanProposalCount']:.1f} "
            f"masks/frame, mean coverage {summary['meanCoverage']:.3f}."
        )
    return summary


def finalize_split(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    work_output = (
        run_paths.split_work_dir
        / "output"
        / f"{run_paths.run_id}_masks"
    )
    if not work_output.is_dir():
        raise FileNotFoundError(f"Upstream Split output does not exist: {work_output}")
    if run_paths.split_raw_output.exists():
        shutil.rmtree(run_paths.split_raw_output)
    shutil.move(str(work_output), str(run_paths.split_raw_output))

    instance_dirs = sorted(
        (path for path in run_paths.split_raw_output.iterdir() if path.is_dir()),
        key=_numeric_path_key,
    )
    instance_ids = [
        int(path.name)
        for path in instance_dirs
        if path.name.isdigit() and int(path.name) > 0
    ]
    proposal_summary = read_json(run_paths.stage_summary("proposals"))
    source_display_name = str(
        proposal_summary.get("sourceDisplayName") or "Unknown Proposals"
    )
    display_name = (
        "Split&Splat Official"
        if proposal_summary.get("proposalSource") == "official_auto"
        else f"Split&Splat: {source_display_name} Input"
    )
    return finalize_prepared_split(
        run_paths,
        editor_paths,
        point_instance_ids=instance_ids,
        mask_instance_ids=instance_ids,
        method="Released Split&Splat depth-visible point-label propagation",
        display_name=display_name,
        label_namespace="split_splat_instance",
        progress=progress,
    )


def finalize_prepared_split(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    *,
    point_instance_ids: list[int],
    mask_instance_ids: list[int],
    method: str,
    display_name: str,
    label_namespace: str,
    global_point_labels: np.ndarray | None = None,
    extra_summary: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Register a prepared Split-compatible set of masks and labeled points."""
    point_instance_ids = sorted(
        {int(value) for value in point_instance_ids if int(value) > 0}
    )
    mask_instance_ids = sorted(
        {int(value) for value in mask_instance_ids if int(value) > 0}
    )
    frame_summaries = _canonicalize_split_masks(
        run_paths,
        editor_paths,
        mask_instance_ids,
        progress=progress,
    )
    rows = read_jsonl(run_paths.frame_map)

    if progress is not None:
        progress("Collecting globally labeled Gaussian means.")
    point_summary = _canonicalize_points(
        run_paths,
        point_instance_ids,
        global_point_labels=global_point_labels,
    )
    proposal_summary = read_json(run_paths.stage_summary("proposals"))
    source_display_name = str(
        proposal_summary.get("sourceDisplayName") or "Unknown Proposals"
    )
    summary = {
        "schemaVersion": 1,
        "stage": "split",
        "timestampUtc": now_utc(),
        "method": str(method),
        "displayName": display_name,
        "proposalSource": proposal_summary.get("proposalSource"),
        "sourceMethodId": proposal_summary.get("sourceMethodId"),
        "sourceDisplayName": source_display_name,
        "labelNamespace": str(label_namespace),
        "frameCount": len(rows),
        "instanceCount": len(point_instance_ids),
        "instanceIds": point_instance_ids,
        "maskInstanceCount": len(mask_instance_ids),
        "maskInstanceIds": mask_instance_ids,
        "binaryInstanceMasksDir": str(run_paths.split_raw_output),
        "labelMapsDir": str(run_paths.consistent_label_maps),
        "overlaysDir": str(run_paths.consistent_overlays),
        "exclusivePreviewConflictPolicy": "smaller_instance_masks_overwrite",
        "frames": frame_summaries,
        "points": point_summary,
        **(extra_summary or {}),
    }
    atomic_write_json(run_paths.split_summary, summary)
    atomic_write_json(run_paths.stage_summary("split"), summary)
    register_editor_artifacts(run_paths, editor_paths, summary)
    if progress is not None:
        progress(
            f"Registered {summary['instanceCount']} instances and "
            f"{point_summary['labeledPointCount']} labeled Gaussian means."
        )
    return summary


def refresh_split_mask_artifacts(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Rebuild the exclusive editor preview without rerunning upstream Split."""
    summary = read_json(run_paths.stage_summary("split"))
    instance_ids = [
        int(value)
        for value in summary.get("maskInstanceIds", summary["instanceIds"])
    ]
    frames = _canonicalize_split_masks(
        run_paths,
        editor_paths,
        instance_ids,
        progress=progress,
    )
    refreshed = {
        **summary,
        "frames": frames,
        "exclusivePreviewConflictPolicy": "smaller_instance_masks_overwrite",
    }
    atomic_write_json(run_paths.split_summary, refreshed)
    atomic_write_json(run_paths.stage_summary("split"), refreshed)
    _register_mask_layer(run_paths, editor_paths, refreshed)
    return refreshed


def _canonicalize_split_masks(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    instance_ids: list[int],
    *,
    progress: Callable[[str], None] | None,
) -> list[dict[str, Any]]:
    run_paths.consistent_label_maps.mkdir(parents=True, exist_ok=True)
    run_paths.consistent_overlays.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(run_paths.frame_map)
    frame_summaries = []
    if progress is not None:
        progress(
            f"Canonicalizing {len(instance_ids)} Split instances across "
            f"{len(rows)} frames."
        )
    for frame_index, row in enumerate(rows):
        frame_id = int(row["frameId"])
        source_name = f"{Path(str(row['splitSplatImageName'])).stem}.png"
        masks: list[tuple[int, np.ndarray]] = []
        for instance_id in instance_ids:
            path = run_paths.split_raw_output / str(instance_id) / source_name
            if not path.is_file():
                continue
            mask = np.asarray(Image.open(path).convert("L")) > 0
            masks.append((instance_id, mask))
        labels, conflict_pixels = _exclusive_global_labels(
            masks,
            width=int(row["width"]),
            height=int(row["height"]),
        )
        label_path = run_paths.consistent_label_maps / f"{frame_id:06d}.png"
        _write_label_png(label_path, labels)
        _write_overlay(
            run_paths.consistent_overlays / f"{frame_id:06d}.png",
            _editor_image_path(editor_paths, frame_id),
            labels,
        )
        frame_summaries.append(
            {
                "frameId": frame_id,
                "labelMapPath": str(label_path),
                "visibleInstanceCount": int(
                    np.count_nonzero(np.unique(labels) > 0)
                ),
                "coverage": float(
                    np.count_nonzero(labels) / max(labels.size, 1)
                ),
                "conflictPixels": int(conflict_pixels),
            }
        )
        completed = frame_index + 1
        if progress is not None and (
            completed == 1
            or completed % 10 == 0
            or completed == len(rows)
        ):
            progress(f"Canonicalized Split masks {completed}/{len(rows)}.")
    return frame_summaries


def register_editor_artifacts(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    split_summary: dict[str, Any],
) -> None:
    _register_mask_layer(run_paths, editor_paths, split_summary)
    _register_point_cloud_runs(run_paths, editor_paths, split_summary)


def register_splat_refined_mask_layer(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    splat_summary: dict[str, Any],
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Expose the per-view masks consumed after Splat-stage SAM2 refinement."""
    frame_map = read_jsonl(run_paths.frame_map)
    cached = _complete_splat_mask_preview(run_paths, len(frame_map))
    if cached is None:
        cached = _canonicalize_splat_refined_masks(
            run_paths,
            frame_map,
            splat_summary,
            progress=progress,
        )
    _register_canonical_mask_layer(
        editor_paths=editor_paths,
        method_id=f"{run_paths.run_id}_splat_masks",
        display_name=f"{_run_display_name(run_paths.run_id)} Splat-Refined Masks",
        description=(
            "Read-only per-frame masks after the released Splat-stage Gaussian "
            "reprojection and SAM2 refinement. These masks supervise refined "
            "object training and later composition; composition does not update "
            "them again."
        ),
        label_namespace=str(cached["labelNamespace"]),
        label_maps=run_paths.splat_mask_label_maps,
        overlays=run_paths.splat_mask_overlays,
        frames=cached["frames"],
        canonical_run_dir=run_paths.run_dir,
        canonical_summary=run_paths.splat_mask_preview_summary,
        stage="splat_refined_masks",
    )
    return cached


def _complete_splat_mask_preview(
    run_paths: SplitSplatRunPaths,
    expected_frames: int,
) -> dict[str, Any] | None:
    if not run_paths.splat_mask_preview_summary.is_file():
        return None
    try:
        summary = read_json(run_paths.splat_mask_preview_summary)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        summary.get("status") != "complete"
        or len(summary.get("frames", [])) != expected_frames
        or len(list(run_paths.splat_mask_label_maps.glob("*.png"))) != expected_frames
    ):
        return None
    return summary


def _canonicalize_splat_refined_masks(
    run_paths: SplitSplatRunPaths,
    frame_map: list[dict[str, Any]],
    splat_summary: dict[str, Any],
    *,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    prepared = read_json(run_paths.stage_summary("splat_prepare"))
    split_summary = read_json(run_paths.stage_summary("split"))
    instances = [
        (int(row["instanceId"]), Path(str(row["datasetDir"])) / "masks")
        for row in prepared["instances"]
    ]
    if run_paths.splat_mask_preview.exists():
        shutil.rmtree(run_paths.splat_mask_preview)
    run_paths.splat_mask_label_maps.mkdir(parents=True)
    run_paths.splat_mask_overlays.mkdir(parents=True)

    frames = []
    total_changed_pixels = 0
    for index, row in enumerate(frame_map):
        frame_id = int(row["frameId"])
        source_name = f"{Path(str(row['splitSplatImageName'])).stem}.png"
        masks = []
        for instance_id, mask_dir in instances:
            path = mask_dir / source_name
            if path.is_file():
                masks.append((instance_id, _read_binary_mask(path)))
        labels, conflict_pixels = _exclusive_global_labels(
            masks,
            width=int(row["width"]),
            height=int(row["height"]),
        )
        split_path = run_paths.consistent_label_maps / f"{frame_id:06d}.png"
        split_labels = np.asarray(Image.open(split_path), dtype=np.uint16)
        if split_labels.shape != labels.shape:
            raise ValueError(
                f"Split/Splat preview shape mismatch for frame {frame_id}: "
                f"{split_labels.shape} != {labels.shape}"
            )
        changed_pixels = int(np.count_nonzero(labels != split_labels))
        total_changed_pixels += changed_pixels
        write_label_artifacts(
            run_paths.splat_mask_label_maps,
            run_paths.splat_mask_overlays,
            frame_id,
            labels,
        )
        frames.append(
            {
                "frameId": frame_id,
                "labelMapPath": str(
                    run_paths.splat_mask_label_maps / f"{frame_id:06d}.png"
                ),
                "visibleInstanceCount": int(
                    np.count_nonzero(np.unique(labels) > 0)
                ),
                "coverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
                "conflictPixels": int(conflict_pixels),
                "changedPixelCountFromSplit": changed_pixels,
                "changedFractionFromSplit": float(
                    changed_pixels / max(labels.size, 1)
                ),
            }
        )
        completed = index + 1
        if progress is not None and (
            completed == 1
            or completed % 10 == 0
            or completed == len(frame_map)
        ):
            progress(
                f"Canonicalized Splat-refined masks {completed}/{len(frame_map)}."
            )

    label_namespace = str(
        split_summary.get("labelNamespace") or "split_splat_instance"
    )
    summary = {
        "schemaVersion": 1,
        "status": "complete",
        "stage": "splat_refined_masks",
        "timestampUtc": now_utc(),
        "method": str(splat_summary.get("method") or ""),
        "sourceStageSummary": str(run_paths.stage_summary("splat_masks")),
        "labelNamespace": label_namespace,
        "instanceCount": len(instances),
        "refinedOrAddedMaskCount": int(
            sum(
                int(row.get("refinedOrAddedMaskCount", 0))
                for row in splat_summary.get("instances", [])
            )
        ),
        "totalChangedPixelsFromSplit": total_changed_pixels,
        "exclusivePreviewConflictPolicy": "smaller_instance_masks_overwrite",
        "frames": frames,
        "note": (
            "This is the last persisted per-view mask set in the released Splat "
            "pipeline. Refined training and composition consume these masks but "
            "do not generate another per-view mask revision."
        ),
    }
    atomic_write_json(run_paths.splat_mask_preview_summary, summary)
    return summary


def _read_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if "A" in image.getbands():
            return np.asarray(image.getchannel("A")) > 0
        return np.asarray(image.convert("L")) > 0


def _run_display_name(run_id: str) -> str:
    names = {
        "split_splat_official": "Official Baseline",
        "split_splat_sam2_video": "SAM2 Video Adaptation",
        "split_splat_anchored_sam2_video": "Anchored 3D Adaptation",
    }
    return names.get(
        run_id,
        " ".join(part.capitalize() for part in run_id.split("_") if part),
    )


def register_proposal_layer(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    proposal_summary: dict[str, Any],
) -> None:
    """Expose a read-only editor preview without changing the Split input."""
    method_id = f"{run_paths.run_id}_proposals"
    layer_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / method_id
    )
    layer_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(layer_dir / "label_maps", run_paths.proposal_label_maps)
    replace_symlink(layer_dir / "overlays", run_paths.proposal_overlays)

    metadata_dir = layer_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_by_frame = {
        int(row["frameId"]): row
        for row in proposal_summary.get("frames", [])
        if isinstance(row, dict) and "frameId" in row
    }
    frame_rows = []
    for row in read_jsonl(run_paths.frame_map):
        frame_id = int(row["frameId"])
        npy_path = run_paths.proposal_label_maps / f"{frame_id:06d}.npy"
        png_path = run_paths.proposal_label_maps / f"{frame_id:06d}.png"
        if npy_path.is_file():
            labels = np.asarray(np.load(npy_path, allow_pickle=False), dtype=np.uint16)
        elif png_path.is_file():
            labels = np.asarray(Image.open(png_path), dtype=np.uint16)
            np.save(npy_path, labels)
        else:
            raise FileNotFoundError(
                f"Split&Splat proposal preview is missing frame {frame_id}: {npy_path}"
            )

        diagnostics = diagnostics_by_frame.get(frame_id, {})
        label_rows = [
            {
                "labelId": int(label_id),
                "pixelCount": int(np.count_nonzero(labels == label_id)),
                "layer": f"propagation_{method_id}",
                "labelSpace": "frame_local_proposal",
                "proposalKind": "split_splat_input_proposal",
            }
            for label_id in np.unique(labels)
            if int(label_id) > 0
        ]
        metadata = {
            "frameId": frame_id,
            "labelSpace": "frame_local_proposal",
            "labelCount": len(label_rows),
            "coverage": float(
                diagnostics.get(
                    "coverage",
                    np.count_nonzero(labels) / max(labels.size, 1),
                )
            ),
            "overlapCoverage": float(diagnostics.get("overlapCoverage", 0.0)),
            "exclusivePreviewPolicy": str(
                diagnostics.get("exclusivePreviewPolicy") or "source_label_map"
            ),
            "labels": label_rows,
        }
        atomic_write_json(metadata_dir / f"{frame_id:06d}.json", metadata)
        frame_rows.append(metadata)
    write_jsonl(layer_dir / "frames.jsonl", frame_rows)

    source_name = str(
        proposal_summary.get("sourceDisplayName") or "Unknown Proposals"
    )
    display_name = (
        "Split Input: Official SAM2"
        if proposal_summary.get("proposalSource") == "official_auto"
        else f"Split Input: {source_name}"
    )
    layer_summary = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "methodId": method_id,
        "displayName": display_name,
        "engineName": "Split&Splat",
        "description": (
            "Read-only exclusive preview of the 2D proposals supplied to the "
            "Split&Splat Split stage."
        ),
        "labelSpace": "frame_local_proposal",
        "layerGroup": "frame_proposals",
        "stage": "proposal_preview",
        "readOnly": True,
        "frameCount": len(frame_rows),
        "completedFrameCount": len(frame_rows),
        "canonicalRunDir": str(run_paths.run_dir),
        "canonicalSummary": str(run_paths.stage_summary("proposals")),
        "binaryMasksDir": str(run_paths.proposal_binary_masks),
        "previewOnly": True,
        "overlapNote": (
            "The editor label map is exclusive. The Split stage continues to use "
            "the original per-instance binary masks, which may overlap."
        ),
    }
    atomic_write_json(layer_dir / "summary.json", layer_summary)
    atomic_write_json(
        layer_dir / "progress.json",
        {
            "status": "complete",
            "running": False,
            "failed": False,
            "message": "Split&Splat input proposal preview ready",
            "updatedUtc": now_utc(),
        },
    )


def register_split_support_layer(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    *,
    label_maps_dir: Path,
    overlays_dir: Path,
    frame_summaries: list[dict[str, Any]],
    label_namespace: str,
) -> None:
    """Expose sparse projected 3D labels as a read-only diagnostic layer."""
    method_id = f"{run_paths.run_id}_support"
    layer_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / method_id
    )
    layer_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(layer_dir / "label_maps", label_maps_dir)
    replace_symlink(layer_dir / "overlays", overlays_dir)
    metadata_dir = layer_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    diagnostics_by_frame = {
        int(row["frameId"]): row
        for row in frame_summaries
    }
    for frame in read_jsonl(run_paths.frame_map):
        frame_id = int(frame["frameId"])
        labels = np.asarray(
            np.load(label_maps_dir / f"{frame_id:06d}.npy", allow_pickle=False),
            dtype=np.uint16,
        )
        diagnostics = diagnostics_by_frame.get(frame_id, {})
        label_rows = [
            {
                "labelId": int(label_id),
                "pixelCount": int(np.count_nonzero(labels == label_id)),
                "layer": f"propagation_{method_id}",
                "labelSpace": label_namespace,
                "proposalKind": "projected_3d_label_support",
            }
            for label_id in np.unique(labels)
            if int(label_id) > 0
        ]
        metadata = {
            "frameId": frame_id,
            "labelSpace": label_namespace,
            "labelCount": len(label_rows),
            "coverage": float(
                diagnostics.get(
                    "supportCoverage",
                    np.count_nonzero(labels) / max(labels.size, 1),
                )
            ),
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
            "displayName": "Complete 3D Region Ownership",
            "engineName": "OMeGa Anchored Split",
            "description": (
                "Z-buffered projection of the complete persistent-region "
                "ownership field on the shared global 3DGS. Direct manual and "
                "propagated votes define labels; local 3D completion covers "
                "Gaussians with no direct observation."
            ),
            "labelSpace": label_namespace,
            "layerGroup": "geometry_support",
            "stage": "anchored_3d_support",
            "readOnly": True,
            "frameCount": len(frame_rows),
            "completedFrameCount": len(frame_rows),
            "canonicalRunDir": str(run_paths.run_dir),
            "canonicalSummary": str(run_paths.split_summary),
        },
    )
    atomic_write_json(
        layer_dir / "progress.json",
        {
            "status": "complete",
            "running": False,
            "failed": False,
            "message": "Complete anchored 3D ownership ready",
            "updatedUtc": now_utc(),
        },
    )


def register_global_point_cloud(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
) -> dict[str, Any]:
    global_cache, point_count = _ensure_global_points_cache(run_paths)
    ensure_viewer_gaussian_ply(
        run_paths.global_point_cloud,
        run_paths.global_viewer_gaussians,
    )
    run_id = f"{run_paths.shared_id}_global"
    run_dir = editor_paths.segmentation3d_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "globalPointCount": point_count,
        "labeledPointCount": 0,
        "globalCachePath": str(global_cache),
    }
    atomic_write_json(
        run_dir / "experiment.json",
        {
            "schemaVersion": 1,
            "runId": run_id,
            "methodId": "split_splat",
            "experimentFamily": "split_splat",
            "artifactRole": "shared_global",
            "baseRunId": "",
            "inputId": "official_split",
            "displayName": "Split&Splat Shared Global GS",
            "labelSpace": "split_splat_instance",
            "pointsCachePath": str(global_cache),
            "geometrySource": "split_splat_global_gs",
            "ready": True,
            "timestampUtc": now_utc(),
            "canonicalRunDir": str(run_paths.shared_dir),
            "sharedGeometryDir": str(run_paths.shared_dir),
            "pointSummary": summary,
            "gaussianArtifacts": {
                "appearance": {
                    "displayName": "RGB",
                    "path": str(run_paths.global_viewer_gaussians),
                    "format": "3dgs_ply",
                    "pointCount": point_count,
                }
            },
        },
    )
    return summary


def _register_mask_layer(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    split_summary: dict[str, Any],
) -> None:
    label_namespace = str(
        split_summary.get("labelNamespace") or "split_splat_instance"
    )
    display_name = str(
        split_summary.get("displayName")
        or f"Split&Splat: {run_paths.run_id}"
    )
    _register_canonical_mask_layer(
        editor_paths=editor_paths,
        method_id=run_paths.run_id,
        display_name=display_name,
        description=str(
            split_summary.get("layerDescription")
            or (
                "Exclusive editor preview of the Split stage's overlapping "
                "binary instance masks; smaller masks win display conflicts."
            )
        ),
        label_namespace=label_namespace,
        label_maps=run_paths.consistent_label_maps,
        overlays=run_paths.consistent_overlays,
        frames=split_summary["frames"],
        canonical_run_dir=run_paths.run_dir,
        canonical_summary=run_paths.split_summary,
        stage="split_refined_masks",
    )


def _register_canonical_mask_layer(
    *,
    editor_paths: EditorPaths,
    method_id: str,
    display_name: str,
    description: str,
    label_namespace: str,
    label_maps: Path,
    overlays: Path,
    frames: list[dict[str, Any]],
    canonical_run_dir: Path,
    canonical_summary: Path,
    stage: str,
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
    proposal_kind = (
        "anchored_persistent_region"
        if label_namespace == "persistent_region"
        else "split_splat_instance"
    )
    metadata_dir = layer_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for row in frames:
        frame_id = int(row["frameId"])
        label_path = label_maps / f"{frame_id:06d}.png"
        labels = np.asarray(Image.open(label_path), dtype=np.uint16)
        np.save(label_maps / f"{frame_id:06d}.npy", labels)
        label_rows = [
            {
                "labelId": int(label_id),
                "pixelCount": int(np.count_nonzero(labels == label_id)),
                "layer": f"propagation_{method_id}",
                "labelSpace": label_namespace,
                "proposalKind": proposal_kind,
            }
            for label_id in np.unique(labels)
            if int(label_id) > 0
        ]
        metadata = {
            "frameId": frame_id,
            "labelSpace": label_namespace,
            "labelCount": int(np.count_nonzero(np.unique(labels) > 0)),
            "coverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
            "conflictPixels": int(row.get("conflictPixels", 0)),
            "labels": label_rows,
        }
        for key in ("changedPixelCountFromSplit", "changedFractionFromSplit"):
            if key in row:
                metadata[key] = row[key]
        atomic_write_json(metadata_dir / f"{frame_id:06d}.json", metadata)
        frame_rows.append(metadata)
    write_jsonl(layer_dir / "frames.jsonl", frame_rows)
    layer_summary = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "methodId": method_id,
        "displayName": display_name,
        "engineName": "Split&Splat",
        "description": description,
        "labelSpace": label_namespace,
        "layerGroup": "refined_masks",
        "stage": stage,
        "readOnly": True,
        "frameCount": len(frame_rows),
        "completedFrameCount": len(frame_rows),
        "canonicalRunDir": str(canonical_run_dir),
        "canonicalSummary": str(canonical_summary),
    }
    atomic_write_json(layer_dir / "summary.json", layer_summary)
    atomic_write_json(
        layer_dir / "progress.json",
        {
            "status": "complete",
            "running": False,
            "failed": False,
            "message": f"Split&Splat mask layer ready: {display_name}",
            "updatedUtc": now_utc(),
        },
    )


def _register_point_cloud_runs(
    run_paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    split_summary: dict[str, Any],
) -> None:
    register_global_point_cloud(run_paths, editor_paths)
    root = editor_paths.segmentation3d_dir / "runs"
    entries = [
        (
            f"{run_paths.run_id}_labeled",
            f"{split_summary.get('displayName', run_paths.run_id)} Instances",
            run_paths.point_labels_dir / "labeled_points.npz",
            run_paths.run_dir,
        ),
    ]
    label_namespace = str(
        split_summary.get("labelNamespace") or "split_splat_instance"
    )
    for run_id, display_name, cache_path, canonical_dir in entries:
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": 1,
            "runId": run_id,
            "methodId": "split_splat",
            "experimentFamily": "split_splat",
            "artifactRole": "split_labels",
            "baseRunId": run_paths.run_id,
            "inputId": str(
                split_summary.get("sourceMethodId")
                or split_summary.get("proposalSource")
                or run_paths.run_id
            ),
            "displayName": display_name,
            "labelSpace": label_namespace,
            "pointsCachePath": str(cache_path),
            "geometrySource": "split_splat_global_gs",
            "ready": True,
            "timestampUtc": now_utc(),
            "canonicalRunDir": str(canonical_dir),
            "sharedGeometryDir": str(run_paths.shared_dir),
            "pointSummary": split_summary["points"],
            "gaussianArtifacts": {
                "instance_ids": {
                    "displayName": "Instance ID Colors",
                    "path": str(run_paths.segmented_gaussians),
                    "format": "3dgs_ply",
                    "pointCount": int(
                        split_summary["points"]["globalPointCount"]
                    ),
                    "labelCount": int(split_summary["instanceCount"]),
                    "labeledPointCount": int(
                        split_summary["points"]["labeledGlobalPointCount"]
                    ),
                    "colorSpace": label_namespace,
                    "unlabeledColor": "neutral_gray",
                },
            },
        }
        atomic_write_json(run_dir / "experiment.json", payload)


def _canonicalize_points(
    run_paths: SplitSplatRunPaths,
    instance_ids: list[int],
    *,
    global_point_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    global_cache, global_point_count = _ensure_global_points_cache(run_paths)

    point_blocks = []
    label_blocks = []
    for instance_id in instance_ids:
        path = (
            run_paths.split_raw_output
            / str(instance_id)
            / f"label_{instance_id}.ply"
        )
        if not path.is_file():
            continue
        points = _read_ply_points(path)
        if points.size == 0:
            continue
        point_blocks.append(points.astype(np.float32))
        label_blocks.append(np.full(points.shape[0], instance_id, dtype=np.int32))
    labeled_points = (
        np.concatenate(point_blocks, axis=0)
        if point_blocks
        else np.empty((0, 3), dtype=np.float32)
    )
    labels = (
        np.concatenate(label_blocks, axis=0)
        if label_blocks
        else np.empty((0,), dtype=np.int32)
    )
    labeled_cache = run_paths.point_labels_dir / "labeled_points.npz"
    np.savez_compressed(labeled_cache, points=labeled_points, labels=labels)
    if global_point_labels is None:
        global_labels, match_summary = _match_global_gaussian_labels(
            run_paths.global_point_cloud,
            labeled_points,
            labels,
        )
    else:
        global_labels = np.asarray(global_point_labels, dtype=np.int32)
        if global_labels.shape != (global_point_count,):
            raise ValueError(
                "Exact global point labels must align with the shared Gaussian "
                f"rows: labels={global_labels.shape}, points={global_point_count}."
            )
        match_summary = {
            "policy": "exact_row_aligned_ownership",
            "queryPointCount": int(global_point_count),
            "matchedPointCount": int(np.count_nonzero(global_labels)),
            "unmatchedPointCount": int(np.count_nonzero(global_labels == 0)),
        }
    np.save(run_paths.point_labels, global_labels)
    write_segmented_gaussian_ply(
        run_paths.global_point_cloud,
        run_paths.segmented_gaussians,
        global_labels,
    )
    return {
        "globalPointCount": global_point_count,
        "labeledPointCount": int(labeled_points.shape[0]),
        "labeledGlobalPointCount": int(np.count_nonzero(global_labels)),
        "globalCachePath": str(global_cache),
        "labeledCachePath": str(labeled_cache),
        "globalLabelsPath": str(run_paths.point_labels),
        "segmentedGaussiansPath": str(run_paths.segmented_gaussians),
        "labelMatch": match_summary,
    }


def _ensure_global_points_cache(
    run_paths: SplitSplatRunPaths,
) -> tuple[Path, int]:
    global_cache = run_paths.global_points_cache
    if global_cache.is_file():
        with np.load(global_cache) as cached:
            point_count = int(np.asarray(cached["points"]).shape[0])
        return global_cache, point_count
    global_points = _read_ply_points(run_paths.global_point_cloud)
    global_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        global_cache,
        points=global_points.astype(np.float32),
        labels=np.zeros(global_points.shape[0], dtype=np.int32),
    )
    return global_cache, int(global_points.shape[0])


def _read_ply_points(path: Path) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required to register Split&Splat point artifacts.") from exc
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Could not read XYZ points from {path}")
    return points


def _match_global_gaussian_labels(
    global_ply: Path,
    labeled_points: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    from scipy.spatial import cKDTree

    global_points = _read_ply_vertex_xyz(global_ply)
    global_labels = np.zeros(global_points.shape[0], dtype=np.int32)
    if labeled_points.shape[0] == 0:
        return global_labels, {
            "inputPointCount": 0,
            "matchedPointCount": 0,
            "unmatchedPointCount": 0,
            "conflictingGlobalPointCount": 0,
            "distanceTolerance": 0.0,
            "maximumMatchedDistance": 0.0,
        }
    if labeled_points.shape[0] != labels.shape[0]:
        raise ValueError("Split&Splat labeled points and labels have different lengths.")

    scene_extent = float(np.linalg.norm(np.ptp(global_points, axis=0)))
    tolerance = max(scene_extent * 1e-6, 1e-7)
    distances, indices = cKDTree(global_points).query(
        np.asarray(labeled_points, dtype=np.float64),
        k=1,
        workers=-1,
    )
    matched = np.isfinite(distances) & (distances <= tolerance)
    conflicts: set[int] = set()
    for source_index in np.flatnonzero(matched):
        global_index = int(indices[source_index])
        label = int(labels[source_index])
        previous = int(global_labels[global_index])
        if previous == 0 or previous == label:
            global_labels[global_index] = label
        else:
            conflicts.add(global_index)
    if conflicts:
        global_labels[np.fromiter(conflicts, dtype=np.int64)] = 0
    maximum_distance = (
        float(np.max(distances[matched]))
        if np.any(matched)
        else 0.0
    )
    return global_labels, {
        "inputPointCount": int(labeled_points.shape[0]),
        "matchedPointCount": int(np.count_nonzero(matched)),
        "unmatchedPointCount": int(np.count_nonzero(~matched)),
        "conflictingGlobalPointCount": len(conflicts),
        "distanceTolerance": tolerance,
        "maximumMatchedDistance": maximum_distance,
    }


def _read_ply_vertex_xyz(path: Path) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:
        raise RuntimeError(
            "plyfile is required to preserve Split&Splat Gaussian attributes."
        ) from exc
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    required = {"x", "y", "z"}
    names = set(vertex.dtype.names or ())
    if not required.issubset(names):
        raise ValueError(f"Gaussian PLY is missing XYZ fields: {path}")
    return np.column_stack(
        [vertex["x"], vertex["y"], vertex["z"]]
    ).astype(np.float64, copy=False)



def _exclusive_from_binary_masks(
    paths: list[Path],
    *,
    width: int,
    height: int,
    conflict_policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    masks = []
    for path in paths:
        values = np.asarray(Image.open(path).convert("L")) > 0
        if values.shape != (height, width):
            raise ValueError(f"Mask shape {values.shape} does not match {(height, width)}: {path}")
        masks.append((path, values))
    coverage_count = np.zeros((height, width), dtype=np.uint16)
    labels = np.zeros((height, width), dtype=np.uint16)
    ordered = sorted(masks, key=lambda item: int(np.count_nonzero(item[1])), reverse=True)
    for proposal_id, (_, mask) in enumerate(ordered, start=1):
        coverage_count[mask] += 1
        labels[mask] = proposal_id
    return labels, {
        "proposalCount": len(masks),
        "coverage": float(np.count_nonzero(coverage_count) / max(coverage_count.size, 1)),
        "overlapCoverage": float(np.count_nonzero(coverage_count > 1) / max(coverage_count.size, 1)),
        "exclusivePreviewPolicy": conflict_policy,
    }


def _exclusive_global_labels(
    masks: list[tuple[int, np.ndarray]],
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    labels = np.zeros((height, width), dtype=np.uint16)
    coverage = np.zeros((height, width), dtype=np.uint16)
    ordered = sorted(
        masks,
        key=lambda item: (
            int(np.count_nonzero(item[1])),
            int(item[0]),
        ),
        reverse=True,
    )
    for instance_id, mask in ordered:
        coverage[mask] += 1
        # The editor requires one ID per pixel, while upstream Split&Splat
        # intentionally stores overlapping binary instance masks. Resolve only
        # the preview: smaller masks overwrite larger masks so thin instances
        # remain inspectable. Raw upstream masks are left untouched.
        labels[mask] = np.uint16(instance_id)
    return labels, int(np.count_nonzero(coverage > 1))


def _editor_image_path(editor_paths: EditorPaths, frame_id: int) -> Path:
    for row in read_jsonl(editor_paths.frame_manifest):
        if int(row["sai3dFrameId"]) != frame_id:
            continue
        staged = editor_paths.dataset_dir / str(row["colorPath"])
        if staged.is_file():
            return staged
        source = Path(str(row.get("sourceImagePath", "")))
        if source.is_file():
            return source
    raise FileNotFoundError(f"No editor RGB image found for frame {frame_id}")


def _write_overlay(path: Path, image_path: Path, labels: np.ndarray) -> None:
    del image_path
    colors = _label_colors(labels)
    active = labels > 0
    output = np.zeros((*labels.shape, 4), dtype=np.uint8)
    output[..., :3] = colors
    output[active, 3] = 118
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGBA").save(path)


def write_label_artifacts(
    label_maps_dir: Path,
    overlays_dir: Path,
    frame_id: int,
    labels: np.ndarray,
) -> None:
    label_maps_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    np.save(label_maps_dir / f"{frame_id:06d}.npy", labels.astype(np.uint16))
    _write_label_png(label_maps_dir / f"{frame_id:06d}.png", labels)
    _write_overlay(
        overlays_dir / f"{frame_id:06d}.png",
        Path(),
        labels,
    )


def _label_colors(labels: np.ndarray) -> np.ndarray:
    return categorical_rgb(labels)


def _write_label_png(path: Path, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels.astype(np.uint16)).save(path)


def _numeric_path_key(path: Path) -> tuple[int, str]:
    return (int(path.stem) if path.stem.isdigit() else 2**31 - 1, path.name)
