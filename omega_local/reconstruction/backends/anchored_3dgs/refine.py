"""Mask-refinement policies for anchored reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.split_splat.contract import (
    now_utc,
    read_json,
    replace_symlink,
)
from omega_local.segmentation.split_splat.splat import refine_instance_masks

from .contract import AnchoredSplatConfig, AnchoredSplatPaths


ProgressCallback = Callable[[str], None]


def refine_masks(
    config: AnchoredSplatConfig,
    runtime_config: Any,
    paths: AnchoredSplatPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if config.mask_refinement == "none":
        return {
            "schemaVersion": 1,
            "stage": "refine_masks",
            "timestampUtc": now_utc(),
            "method": "No mask refinement",
            "policy": "none",
            "maskMutation": False,
            "sam2Executed": False,
            "message": (
                "Anchored input masks are used unchanged. This is the default "
                "because SAM2 may reinterpret user-defined architectural regions."
            ),
        }
    if config.mask_refinement != "paper_sam2":
        raise ValueError(
            f"Unsupported anchored mask-refinement policy: {config.mask_refinement}"
        )
    if dry_run:
        return {
            "schemaVersion": 1,
            "stage": "refine_masks",
            "timestampUtc": now_utc(),
            "method": "Released Split&Splat SAM2 mask refinement",
            "policy": "paper_sam2",
            "status": "planned",
            "manualAnchorProtection": True,
        }

    anchors = _snapshot_manual_anchor_masks(paths)
    try:
        result = refine_instance_masks(
            runtime_config,
            paths,
            progress=progress,
            command_progress=command_progress,
        )
    finally:
        restored = _restore_manual_anchor_masks(anchors)
    return {
        **result,
        "stage": "refine_masks",
        "policy": "paper_sam2",
        "manualAnchorProtection": True,
        "restoredManualAnchorMaskCount": restored,
        "sourceMutation": False,
        "note": (
            "This optional diagnostic follows the released Split&Splat refiner "
            "inside the anchored run only; manual masks are restored afterward."
        ),
    }


def link_unrefined_models(
    config: AnchoredSplatConfig,
    paths: AnchoredSplatPaths,
) -> dict[str, Any]:
    if config.mask_refinement != "none":
        raise ValueError("Unrefined model linking is only valid for policy 'none'.")
    prepared = read_json(paths.stage_summary("prepare"))
    paths.splat_refined_models.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for item in prepared["instances"]:
        region_id = int(item["regionId"])
        source = paths.splat_initial_models / str(region_id)
        destination = paths.splat_refined_models / str(region_id)
        replace_symlink(destination, source)
        rows.append(
            {
                "instanceId": region_id,
                "regionId": region_id,
                "modelDir": str(destination),
                "sourceModelDir": str(source),
                "retrained": False,
            }
        )
    return {
        "schemaVersion": 1,
        "stage": "refined",
        "timestampUtc": now_utc(),
        "method": "Reuse anchored initial models without mask refinement",
        "policy": "none",
        "instanceCount": len(rows),
        "instances": rows,
        "retrained": False,
    }


def _snapshot_manual_anchor_masks(
    paths: AnchoredSplatPaths,
) -> list[tuple[Path, bytes | None]]:
    weights = json.loads(
        paths.training_view_weights.read_text(encoding="utf-8")
    )
    stems = {
        str(row.get("imageStem") or "").strip()
        for row in weights.get("frames", [])
        if bool(row.get("isManualAnchor"))
    }
    snapshots: list[tuple[Path, bytes | None]] = []
    prepared = read_json(paths.stage_summary("prepare"))
    for item in prepared["instances"]:
        mask_dir = Path(item["datasetDir"]) / "masks"
        for stem in stems:
            path = mask_dir / f"{stem}.png"
            snapshots.append(
                (path, path.read_bytes() if path.is_file() else None)
            )
    return snapshots


def _restore_manual_anchor_masks(
    snapshots: list[tuple[Path, bytes | None]],
) -> int:
    for path, payload in snapshots:
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return len(snapshots)
