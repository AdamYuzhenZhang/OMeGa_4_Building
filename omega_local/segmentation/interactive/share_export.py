"""Export persistent-region propagation as a portable mask dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import read_jsonl, resolve_paths, slug_token
from .regions import persistent_region_color_map
from .share_readme import render_share_readme


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SegmentationShareConfig:
    model_dir: Path
    baseline_name: str
    method_id: str
    output_dir: Path
    overwrite: bool = False


def export_segmentation_share(config: SegmentationShareConfig) -> dict[str, Any]:
    """Write a validated, portable indexed-mask package."""

    paths = resolve_paths(config.model_dir, config.baseline_name)
    method_id = slug_token(config.method_id)
    propagation_dir = paths.interactive_dir / "proposals" / "propagation" / method_id
    output_dir = config.output_dir.expanduser().resolve()
    staging_dir = output_dir.with_name(f".{output_dir.name}.building")

    frame_rows = read_jsonl(paths.frame_manifest)
    propagation_config = _read_json(propagation_dir / "config.json")
    propagation_input = _read_json(propagation_dir / "input.json")
    propagation_summary = _read_json(propagation_dir / "summary.json")
    region_store = _read_json(paths.regions_summary)
    keyframe_summary = _read_json(paths.keyframes_summary)

    _validate_run_contract(
        method_id=method_id,
        frame_rows=frame_rows,
        propagation_config=propagation_config,
        propagation_input=propagation_input,
        propagation_summary=propagation_summary,
    )

    if output_dir.exists() and not config.overwrite:
        raise FileExistsError(f"Share package already exists: {output_dir}. Pass --overwrite to replace it.")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    staging_dir.mkdir(parents=True)
    masks_dir = staging_dir / "masks"
    masks_dir.mkdir()

    try:
        package = _build_package(
            method_id=method_id,
            baseline_name=paths.baseline_name,
            model_dir=paths.model_dir,
            propagation_dir=propagation_dir,
            masks_dir=masks_dir,
            frame_rows=frame_rows,
            propagation_config=propagation_config,
            propagation_input=propagation_input,
            propagation_summary=propagation_summary,
            region_store=region_store,
            keyframe_summary=keyframe_summary,
        )
        _write_json(staging_dir / "manifest.json", package["manifest"])
        _write_json(staging_dir / "regions.json", package["regions"])
        _write_json(staging_dir / "keyframes.json", package["keyframes"])
        _write_jsonl(staging_dir / "frames.jsonl", package["frames"])
        (staging_dir / "README.md").write_text(package["readme"], encoding="utf-8")
        _write_checksums(staging_dir)

        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "outputDir": str(output_dir),
        "methodId": method_id,
        "frameCount": len(package["frames"]),
        "regionCount": len(package["regions"]["regions"]),
        "manualAnchorCount": len(package["keyframes"]["manualAnchors"]),
        "maskResolution": package["manifest"]["maskEncoding"]["resolution"],
        "coverage": package["manifest"]["coverage"],
    }


def _build_package(
    *,
    method_id: str,
    baseline_name: str,
    model_dir: Path,
    propagation_dir: Path,
    masks_dir: Path,
    frame_rows: list[dict[str, Any]],
    propagation_config: dict[str, Any],
    propagation_input: dict[str, Any],
    propagation_summary: dict[str, Any],
    region_store: dict[str, Any],
    keyframe_summary: dict[str, Any],
) -> dict[str, Any]:
    frame_by_id = _rows_by_integer_key(frame_rows, "sai3dFrameId", "frame manifest")
    summary_by_id = _rows_by_integer_key(
        list(propagation_summary.get("frames") or []),
        "frameId",
        "propagation summary",
    )
    if set(summary_by_id) != set(frame_by_id):
        raise ValueError("Propagation summary and frame manifest do not describe the same frame IDs.")
    source_region_ids = {int(value) for value in propagation_input.get("sourceRegionIds") or []}
    if not source_region_ids:
        raise ValueError("Propagation input does not contain sourceRegionIds.")

    region_rows = [
        dict(row)
        for row in region_store.get("regions") or []
        if isinstance(row, dict) and int(row.get("id", 0)) in source_region_ids
    ]
    region_by_id = {int(row["id"]): row for row in region_rows}
    missing_regions = source_region_ids - set(region_by_id)
    if missing_regions:
        raise ValueError(f"Persistent-region metadata is missing propagated IDs: {sorted(missing_regions)}")

    colors = persistent_region_color_map(region_rows)
    source_frames_by_region = {
        int(region_id): [int(frame_id) for frame_id in frame_ids]
        for region_id, frame_ids in (propagation_input.get("sourceFrameIdsByRegion") or {}).items()
    }
    exported_regions = []
    for region_id in sorted(source_region_ids):
        row = region_by_id[region_id]
        rgb = tuple(int(value) for value in colors[region_id])
        exported_regions.append(
            {
                "id": region_id,
                "name": str(row.get("name") or f"region_{region_id:03d}"),
                "colorRgb": list(rgb),
                "colorHex": "#" + "".join(f"{value:02x}" for value in rgb),
                "manualFrameIds": source_frames_by_region.get(region_id, []),
            }
        )

    anchor_ids = [int(value) for value in propagation_input.get("anchorFrameIds") or []]
    anchor_set = set(anchor_ids)
    suggested_keyframes = [int(value) for value in keyframe_summary.get("keyframeIds") or []]
    suggested_set = set(suggested_keyframes)
    propagation_maps = propagation_dir / "label_maps"
    anchor_path_by_id = {
        int(row["frameId"]): Path(str(row["path"])).expanduser().resolve()
        for row in propagation_input.get("anchorMaps") or []
    }

    exported_frames: list[dict[str, Any]] = []
    frame_coverages: list[float] = []
    mask_shape: tuple[int, int] | None = None
    source_shapes: set[tuple[int, int]] = set()
    for frame_id in sorted(frame_by_id):
        frame = frame_by_id[frame_id]
        source_npy = propagation_maps / f"{frame_id:06d}.npy"
        source_png = propagation_maps / f"{frame_id:06d}.png"
        labels = _load_and_validate_mask(
            npy_path=source_npy,
            png_path=source_png,
            frame_id=frame_id,
            known_region_ids=source_region_ids,
        )
        if frame_id in anchor_set:
            anchor_path = anchor_path_by_id.get(frame_id)
            if anchor_path is None or not anchor_path.is_file():
                raise FileNotFoundError(f"Manual anchor map is missing for frame {frame_id}: {anchor_path}")
            anchor_labels = np.load(anchor_path, allow_pickle=False)
            if not np.array_equal(labels, anchor_labels):
                raise ValueError(
                    f"Propagated frame {frame_id} does not exactly preserve its manual anchor mask."
                )

        current_shape = (int(labels.shape[0]), int(labels.shape[1]))
        if mask_shape is None:
            mask_shape = current_shape
        elif current_shape != mask_shape:
            raise ValueError(f"Mask shape changed at frame {frame_id}: {current_shape} != {mask_shape}")

        manifest_shape = (int(frame.get("height", 0)), int(frame.get("width", 0)))
        if manifest_shape != current_shape:
            raise ValueError(
                f"Mask shape {current_shape} does not match frame manifest {manifest_shape} "
                f"for frame {frame_id}."
            )

        source_image = Path(str(frame["sourceImagePath"])).expanduser().resolve()
        if not source_image.is_file():
            raise FileNotFoundError(f"Source image is missing for frame {frame_id}: {source_image}")
        with Image.open(source_image) as image:
            source_shape = (int(image.height), int(image.width))
        source_shapes.add(source_shape)

        destination = masks_dir / f"{frame_id:06d}.png"
        shutil.copy2(source_png, destination)

        positive, counts = np.unique(labels[labels > 0], return_counts=True)
        region_areas = {
            str(int(region_id)): int(area)
            for region_id, area in zip(positive.tolist(), counts.tolist(), strict=True)
        }
        coverage = float(np.count_nonzero(labels) / max(labels.size, 1))
        expected_coverage = float(summary_by_id[frame_id].get("coverage", coverage))
        if not np.isclose(coverage, expected_coverage, atol=1.0 / max(labels.size, 1)):
            raise ValueError(
                f"Coverage mismatch for frame {frame_id}: mask={coverage}, summary={expected_coverage}"
            )
        frame_coverages.append(coverage)
        exported_frames.append(
            {
                "frameId": frame_id,
                "sourceFrameId": int(frame.get("sourceFrameId", frame_id)),
                "imageName": str(frame.get("imageName") or source_image.name),
                "sourceImage": _portable_source_image_path(source_image, model_dir),
                "sourceResolution": {"width": source_shape[1], "height": source_shape[0]},
                "maskPath": f"masks/{frame_id:06d}.png",
                "maskResolution": {"width": current_shape[1], "height": current_shape[0]},
                "isManualAnchor": frame_id in anchor_set,
                "isSuggestedKeyframe": frame_id in suggested_set,
                "coverage": coverage,
                "regionIds": [int(value) for value in positive.tolist()],
                "regionAreas": region_areas,
            }
        )

    if mask_shape is None:
        raise ValueError("No masks were exported.")

    exported_frame_by_id = {int(row["frameId"]): row for row in exported_frames}
    anchor_rows = [
        {
            "frameId": frame_id,
            "imageName": str(frame_by_id[frame_id].get("imageName") or f"{frame_id:06d}.jpg"),
            "maskPath": f"masks/{frame_id:06d}.png",
            "regionIds": list(exported_frame_by_id[frame_id]["regionIds"]),
            "regionAreas": dict(exported_frame_by_id[frame_id]["regionAreas"]),
        }
        for frame_id in anchor_ids
    ]
    coverage_summary = {
        "minimum": float(np.min(frame_coverages)),
        "mean": float(np.mean(frame_coverages)),
        "maximum": float(np.max(frame_coverages)),
    }
    capture_name = model_dir.parents[1].name
    source_shape_rows = [
        {"width": width, "height": height}
        for height, width in sorted(source_shapes)
    ]
    manifest = {
        "schemaVersion": SCHEMA_VERSION,
        "packageType": "omega_interactive_segmentation_masks",
        "createdUtc": _now(),
        "dataset": capture_name,
        "modelRun": model_dir.name,
        "baseline": baseline_name,
        "method": {
            "id": method_id,
            "name": str(propagation_config.get("engineName") or method_id),
            "description": str(propagation_summary.get("method") or ""),
            "labelSpace": "persistent_region",
            "inputFingerprint": str(propagation_input.get("fingerprint") or ""),
            "completedUtc": str(propagation_summary.get("timestampUtc") or ""),
            "primaryFrameId": int(propagation_input.get("primaryFrameId", 0)),
            "primaryPolicy": str(propagation_input.get("primaryPolicy") or ""),
            "settings": _portable_method_settings(propagation_config),
        },
        "frameCount": len(exported_frames),
        "regionCount": len(exported_regions),
        "manualAnchorCount": len(anchor_rows),
        "suggestedKeyframeCount": len(suggested_keyframes),
        "maskEncoding": {
            "format": "PNG",
            "pixelType": "uint16",
            "channelCount": 1,
            "backgroundId": 0,
            "resolution": {"width": mask_shape[1], "height": mask_shape[0]},
            "orientation": "matches the source image files; no display rotation is applied",
        },
        "sourceImages": {
            "included": False,
            "pathBase": "capture_root",
            "dataset": capture_name,
            "resolutions": source_shape_rows,
        },
        "coverage": coverage_summary,
        "files": {
            "frames": "frames.jsonl",
            "regions": "regions.json",
            "keyframes": "keyframes.json",
            "masks": "masks/{frameId:06d}.png",
            "checksums": "checksums.sha256",
        },
    }
    regions = {
        "schemaVersion": SCHEMA_VERSION,
        "labelSpace": "persistent_region",
        "background": {"id": 0, "name": "unassigned", "colorRgb": [0, 0, 0]},
        "regions": exported_regions,
    }
    keyframes = {
        "schemaVersion": SCHEMA_VERSION,
        "manualAnchorFrameIds": anchor_ids,
        "manualAnchors": anchor_rows,
        "suggestedKeyframeIds": suggested_keyframes,
        "note": (
            "Manual anchors are user-confirmed masks that conditioned this propagation. "
            "Suggested keyframes are system recommendations and need not be manually completed."
        ),
    }
    return {
        "manifest": manifest,
        "regions": regions,
        "keyframes": keyframes,
        "frames": exported_frames,
        "readme": render_share_readme(manifest, regions, keyframes),
    }


def _validate_run_contract(
    *,
    method_id: str,
    frame_rows: list[dict[str, Any]],
    propagation_config: dict[str, Any],
    propagation_input: dict[str, Any],
    propagation_summary: dict[str, Any],
) -> None:
    if not frame_rows:
        raise ValueError("Frame manifest is empty.")
    for label, payload in (
        ("config", propagation_config),
        ("summary", propagation_summary),
    ):
        actual = slug_token(str(payload.get("methodId") or ""))
        if actual != method_id:
            raise ValueError(f"Propagation {label} method is '{actual}', expected '{method_id}'.")
    if str(propagation_summary.get("labelSpace")) != "persistent_region":
        raise ValueError("Only persistent-region propagation can be exported.")
    expected_count = len(frame_rows)
    for label, value in (
        ("input", propagation_input.get("frameCount")),
        ("summary", propagation_summary.get("frameCount")),
    ):
        if int(value or -1) != expected_count:
            raise ValueError(f"Propagation {label} frame count is {value}, expected {expected_count}.")
    fingerprints = {
        str(propagation_config.get("inputFingerprint") or ""),
        str(propagation_input.get("fingerprint") or ""),
        str(propagation_summary.get("inputFingerprint") or ""),
    }
    if "" in fingerprints or len(fingerprints) != 1:
        raise ValueError("Propagation config, input, and summary fingerprints do not agree.")


def _load_and_validate_mask(
    *,
    npy_path: Path,
    png_path: Path,
    frame_id: int,
    known_region_ids: set[int],
) -> np.ndarray:
    if not npy_path.is_file() or not png_path.is_file():
        raise FileNotFoundError(f"Propagation mask pair is incomplete for frame {frame_id}.")
    labels = np.load(npy_path, allow_pickle=False)
    if labels.ndim != 2 or labels.dtype != np.uint16:
        raise ValueError(f"Frame {frame_id} must be a 2D uint16 label map, got {labels.shape} {labels.dtype}.")
    with Image.open(png_path) as image:
        png_labels = np.asarray(image)
    if not np.array_equal(labels, png_labels):
        raise ValueError(f"PNG and NPY propagation masks disagree for frame {frame_id}.")
    positive_ids = {int(value) for value in np.unique(labels).tolist() if int(value) > 0}
    unknown = positive_ids - known_region_ids
    if unknown:
        raise ValueError(f"Frame {frame_id} contains unknown persistent region IDs: {sorted(unknown)}")
    return labels


def _rows_by_integer_key(
    rows: list[dict[str, Any]],
    key: str,
    label: str,
) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or key not in row:
            raise ValueError(f"{label} contains a row without '{key}'.")
        value = int(row[key])
        if value in result:
            raise ValueError(f"{label} contains duplicate {key}={value}.")
        result[value] = row
    return result


def _portable_source_image_path(source_image: Path, model_dir: Path) -> str:
    capture_root = model_dir.parents[1]
    try:
        return source_image.relative_to(capture_root).as_posix()
    except ValueError:
        return source_image.name


def _portable_method_settings(config: dict[str, Any]) -> dict[str, Any]:
    backend = config.get("backend") or {}
    settings = {
        str(key): value
        for key, value in backend.items()
        if key not in {"root", "device", "checkpoint"}
        and isinstance(value, (str, int, float, bool, list, dict, type(None)))
    }
    checkpoint = Path(str(backend.get("checkpoint") or "")).name
    if checkpoint:
        settings["checkpoint"] = checkpoint
    if isinstance(config.get("outputResolution"), dict):
        settings["outputResolution"] = dict(config["outputResolution"])
    return settings


def _write_checksums(root: Path) -> None:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(f"{digest.hexdigest()}  {path.relative_to(root).as_posix()}")
    (root / "checksums.sha256").write_text("\n".join(rows) + "\n", encoding="ascii")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="sai3d_area_samples_1024_dense")
    parser.add_argument("--method", default="sam2_video")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export_segmentation_share(
        SegmentationShareConfig(
            model_dir=args.model_dir,
            baseline_name=args.baseline_name,
            method_id=args.method,
            output_dir=args.output_dir,
            overwrite=bool(args.overwrite),
        )
    )
    print(json.dumps(result, indent=2))
    return 0


__all__ = [
    "SegmentationShareConfig",
    "export_segmentation_share",
    "main",
]
