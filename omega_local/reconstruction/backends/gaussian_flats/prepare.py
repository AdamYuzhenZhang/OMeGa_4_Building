"""Prepare a COLMAP scene and persistent-region plane masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.paths import EditorPaths, read_jsonl
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    replace_symlink,
    write_jsonl,
)

from .contract import GaussianFlatsConfig, GaussianFlatsPaths


ProgressCallback = Callable[[str], None]
UPSTREAM_MIN_MASK_PIXELS = 128 * 128
UPSTREAM_MIN_MASK_VIEWS = 11


def prepare_inputs(
    config: GaussianFlatsConfig,
    paths: GaussianFlatsPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_scene = config.model_dir.parent / "dataset"
    source_images = source_scene / "images"
    source_sparse = source_scene / "sparse"
    if not source_images.is_dir() or not source_sparse.is_dir():
        raise FileNotFoundError(
            f"OMeGa COLMAP dataset is incomplete under {source_scene}."
        )

    propagation_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / config.propagation_method
    )
    label_maps_dir = propagation_dir / "label_maps"
    propagation_config = _read_json(propagation_dir / "config.json")
    if propagation_config.get("labelSpace") != "persistent_region":
        raise ValueError(
            "Gaussian Flats requires persistent-region propagation labels, got "
            f"{propagation_config.get('labelSpace')!r}."
        )
    if not label_maps_dir.is_dir():
        raise FileNotFoundError(f"Propagation label maps are missing: {label_maps_dir}")

    region = _resolve_region(editor_paths.regions_summary, config.planar_region)
    frames = read_jsonl(editor_paths.frame_manifest)
    image_names = _colmap_image_names(source_scene / "sparse" / "0" / "images.txt")
    if len(frames) != len(image_names):
        raise ValueError(
            f"Frame/COLMAP count mismatch: {len(frames)} vs {len(image_names)}."
        )

    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.plane_masks_dir.mkdir(parents=True, exist_ok=True)
    replace_symlink(paths.scene_dir / "images", source_images)
    replace_symlink(paths.scene_dir / "sparse", source_sparse)
    replace_symlink(paths.semantic_labels, label_maps_dir)
    replace_symlink(
        paths.input_dir / "persistent_regions.json",
        editor_paths.regions_summary,
    )
    replace_symlink(
        paths.input_dir / "propagation_config.json",
        propagation_dir / "config.json",
    )

    plane_dir = paths.plane_masks_dir / "0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    for stale in plane_dir.glob("*.png"):
        stale.unlink()

    frame_rows: list[dict[str, Any]] = []
    nonempty = 0
    accepted = 0
    total_pixels = 0
    region_id = int(region["id"])
    for index, (frame, image_name) in enumerate(zip(frames, image_names)):
        frame_id = int(frame["sourceFrameId"])
        label_path = label_maps_dir / f"{frame_id:06d}.npy"
        if not label_path.is_file():
            raise FileNotFoundError(f"Propagation label map is missing: {label_path}")
        labels = np.load(label_path)
        expected_shape = (int(frame["height"]), int(frame["width"]))
        if labels.shape != expected_shape:
            raise ValueError(
                f"Label shape mismatch for frame {frame_id}: "
                f"{labels.shape} vs {expected_shape}."
            )
        mask = labels == region_id
        area = int(mask.sum())
        total_pixels += area
        nonempty += int(area > 0)
        accepted += int(area > UPSTREAM_MIN_MASK_PIXELS)
        mask_path = plane_dir / f"{Path(image_name).stem}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        frame_rows.append(
            {
                "index": index,
                "frameId": frame_id,
                "imageName": image_name,
                "sourceLabelMap": str(label_path),
                "planeMask": str(mask_path),
                "areaPixels": area,
                "acceptedByUpstream": area > UPSTREAM_MIN_MASK_PIXELS,
            }
        )
        if progress is not None and (index + 1 == len(frames) or (index + 1) % 25 == 0):
            progress(f"prepare masks {index + 1}/{len(frames)}")

    if accepted < UPSTREAM_MIN_MASK_VIEWS:
        raise RuntimeError(
            f"Plane {region['name']!r} has only {accepted} masks above the released "
            f"{UPSTREAM_MIN_MASK_PIXELS}-pixel cutoff; at least "
            f"{UPSTREAM_MIN_MASK_VIEWS} are required."
        )

    write_jsonl(paths.frame_map, frame_rows)
    plane_manifest = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "method": "3D Gaussian Flats released plane-mask contract",
        "semanticSource": str(label_maps_dir),
        "semanticLabelSpace": "persistent_region",
        "planeIndex": 0,
        "regionId": region_id,
        "regionName": str(region["name"]),
        "maskDirectory": str(plane_dir),
        "frameCount": len(frame_rows),
        "nonemptyFrameCount": nonempty,
        "upstreamAcceptedFrameCount": accepted,
        "upstreamAreaThresholdPixelsExclusive": UPSTREAM_MIN_MASK_PIXELS,
        "upstreamMinimumViews": UPSTREAM_MIN_MASK_VIEWS,
        "totalPlanePixels": total_pixels,
        "policy": (
            "All propagated semantic labels are retained by symlink; only the "
            "selected Door persistent region is exposed as a planar constraint."
        ),
    }
    atomic_write_json(paths.plane_manifest, plane_manifest)
    return {
        "schemaVersion": 1,
        "stage": "prepare",
        "timestampUtc": now_utc(),
        "scene": str(paths.scene_dir),
        "planeMasks": str(paths.plane_masks_dir),
        "semanticLabels": str(paths.semantic_labels),
        "persistentRegions": str(paths.input_dir / "persistent_regions.json"),
        "propagationConfig": str(paths.input_dir / "propagation_config.json"),
        "frameMap": str(paths.frame_map),
        "plane": plane_manifest,
    }


def _resolve_region(regions_path: Path, query: str) -> dict[str, Any]:
    payload = _read_json(regions_path)
    regions = payload.get("regions", [])
    query_text = str(query).strip()
    by_id = query_text.isdigit()
    matches = [
        region
        for region in regions
        if (by_id and int(region.get("id", -1)) == int(query_text))
        or (not by_id and str(region.get("name", "")).casefold() == query_text.casefold())
    ]
    if len(matches) != 1:
        available = ", ".join(
            f"{region.get('id')}:{region.get('name')}" for region in regions
        )
        raise ValueError(
            f"Planar region {query!r} did not resolve uniquely. Available: {available}"
        )
    return matches[0]


def _colmap_image_names(images_txt: Path) -> list[str]:
    if not images_txt.is_file():
        raise FileNotFoundError(f"COLMAP images.txt is missing: {images_txt}")
    names: list[tuple[int, str]] = []
    for raw in images_txt.read_text(encoding="utf-8").splitlines():
        row = raw.strip()
        if not row or row.startswith("#"):
            continue
        fields = row.split()
        # POINTS2D rows are numeric and may be empty. Image rows always have a
        # filename in column ten, so this remains correct for either case.
        if len(fields) >= 10 and Path(fields[9]).suffix:
            names.append((int(fields[0]), fields[9]))
    if not names:
        raise ValueError(f"No camera image rows found in {images_txt}.")
    return [name for _, name in sorted(names)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
