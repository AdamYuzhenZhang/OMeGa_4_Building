"""Prepare non-upstream proposal sources for the released Split stage."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.paths import EditorPaths

from .artifacts import write_label_artifacts
from .contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
)


def import_propagation_proposals(
    config: SplitSplatRunConfig,
    editor_paths: EditorPaths,
    run_paths: SplitSplatRunPaths,
    *,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Convert an exclusive editor propagation layer to upstream binary masks."""
    source_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / config.propagation_method
    )
    source_summary_path = source_dir / "summary.json"
    source_label_maps = source_dir / "label_maps"
    if not source_summary_path.is_file():
        raise FileNotFoundError(
            f"Propagation summary does not exist: {source_summary_path}"
        )
    if not source_label_maps.is_dir():
        raise FileNotFoundError(
            f"Propagation label maps do not exist: {source_label_maps}"
        )

    source_summary = read_json(source_summary_path)
    source_label_space = str(source_summary.get("labelSpace") or "")
    if source_label_space != "persistent_region":
        raise ValueError(
            "Split&Splat propagation input must use persistent_region labels; "
            f"{config.propagation_method} declares {source_label_space!r}."
        )

    for path in (
        run_paths.proposal_binary_masks,
        run_paths.proposal_label_maps,
        run_paths.proposal_overlays,
        run_paths.proposals_dir / "source_mappings",
    ):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(run_paths.frame_map)
    diagnostics: list[dict[str, Any]] = []
    all_source_ids: set[int] = set()
    if progress is not None:
        progress(
            f"Importing {len(rows)} frames from propagation layer "
            f"{config.propagation_method}."
        )
    for frame_index, row in enumerate(rows):
        frame_id = int(row["frameId"])
        labels = _load_source_labels(source_label_maps, frame_id)
        expected_shape = (int(row["height"]), int(row["width"]))
        if labels.shape != expected_shape:
            raise ValueError(
                f"Propagation frame {frame_id} has shape {labels.shape}; "
                f"Split&Splat expects {expected_shape}."
            )
        if np.any(labels < 0) or int(labels.max(initial=0)) > np.iinfo(np.uint16).max:
            raise ValueError(
                f"Propagation frame {frame_id} contains labels outside uint16."
            )
        labels = labels.astype(np.uint16, copy=False)
        source_ids = [
            int(label_id)
            for label_id in np.unique(labels)
            if int(label_id) > 0
        ]
        all_source_ids.update(source_ids)

        split_stem = Path(str(row["splitSplatImageName"])).stem
        binary_dir = run_paths.proposal_binary_masks / split_stem
        binary_dir.mkdir(parents=True, exist_ok=True)
        source_rows = []
        for source_id in source_ids:
            mask = labels == source_id
            filename = f"{source_id:06d}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(
                binary_dir / filename
            )
            source_rows.append(
                {
                    "binaryMask": filename,
                    "sourceLabelId": source_id,
                    "pixelCount": int(np.count_nonzero(mask)),
                }
            )

        write_label_artifacts(
            run_paths.proposal_label_maps,
            run_paths.proposal_overlays,
            frame_id,
            labels,
        )
        mapping_path = (
            run_paths.proposals_dir
            / "source_mappings"
            / f"{frame_id:06d}.json"
        )
        atomic_write_json(
            mapping_path,
            {
                "frameId": frame_id,
                "sourceMethodId": config.propagation_method,
                "sourceLabelSpace": source_label_space,
                "labels": source_rows,
            },
        )
        diagnostics.append(
            {
                "frameId": frame_id,
                "proposalCount": len(source_ids),
                "coverage": float(
                    np.count_nonzero(labels) / max(labels.size, 1)
                ),
                "overlapCoverage": 0.0,
                "sourceLabelIds": source_ids,
                "sourceMapping": str(mapping_path),
            }
        )
        completed = frame_index + 1
        if progress is not None and (
            completed == 1
            or completed % 10 == 0
            or completed == len(rows)
        ):
            progress(
                f"Converted propagated proposal frames {completed}/{len(rows)}."
            )

    if int(source_summary.get("frameCount", len(rows))) != len(rows):
        raise ValueError(
            f"Propagation source reports {source_summary.get('frameCount')} frames, "
            f"but Split&Splat expects {len(rows)}."
        )

    display_name = str(
        source_summary.get("displayName")
        or {
            "sam2_video": "SAM2 Video",
            "sam2_bounded": "SAM2 Bounded",
            "xmem": "XMem++",
            "cutie": "Cutie",
        }.get(config.propagation_method)
        or config.propagation_method.replace("_", " ").title()
    )
    summary = {
        "schemaVersion": 1,
        "stage": "proposals",
        "timestampUtc": now_utc(),
        "method": "Exclusive editor propagation labels converted to Split&Splat binary masks",
        "proposalSource": "propagation",
        "sourceMethodId": config.propagation_method,
        "sourceDisplayName": display_name,
        "sourceLabelSpace": source_label_space,
        "sourceRunDir": str(source_dir),
        "sourceSummary": str(source_summary_path),
        "sourceFingerprint": str(
            source_summary.get("inputFingerprint")
            or source_summary.get("fingerprint")
            or ""
        ),
        "sourceLabelIds": sorted(all_source_ids),
        "frameCount": len(rows),
        "binaryMasksDir": str(run_paths.proposal_binary_masks),
        "labelMapsDir": str(run_paths.proposal_label_maps),
        "overlaysDir": str(run_paths.proposal_overlays),
        "sourceMappingsDir": str(run_paths.proposals_dir / "source_mappings"),
        "meanProposalCount": float(
            np.mean([row["proposalCount"] for row in diagnostics])
        ),
        "meanCoverage": float(
            np.mean([row["coverage"] for row in diagnostics])
        ),
        "frames": diagnostics,
    }
    atomic_write_json(run_paths.stage_summary("proposals"), summary)
    if progress is not None:
        progress(
            f"Imported {len(all_source_ids)} persistent region IDs "
            f"with mean coverage {summary['meanCoverage']:.3f}."
        )
    return summary


def _load_source_labels(label_maps_dir: Path, frame_id: int) -> np.ndarray:
    npy_path = label_maps_dir / f"{frame_id:06d}.npy"
    if npy_path.is_file():
        labels = np.asarray(np.load(npy_path, allow_pickle=False))
    else:
        png_path = label_maps_dir / f"{frame_id:06d}.png"
        if not png_path.is_file():
            raise FileNotFoundError(
                f"Propagation label map is missing for frame {frame_id}: {npy_path}"
            )
        labels = np.asarray(Image.open(png_path))
    if labels.ndim != 2:
        raise ValueError(
            f"Propagation frame {frame_id} must be a 2D label map, got {labels.shape}."
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(
            f"Propagation frame {frame_id} must use integer labels, got {labels.dtype}."
        )
    return labels
