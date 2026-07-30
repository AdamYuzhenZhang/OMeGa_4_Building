from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.share_export import (
    SegmentationShareConfig,
    export_segmentation_share,
)


def test_exports_portable_indexed_mask_package(tmp_path: Path) -> None:
    model_dir = tmp_path / "capture" / "omega_stable_mesh" / "model_run"
    baseline_dir = model_dir / "segmentation" / "baselines" / "demo"
    dataset_dir = baseline_dir / "dataset"
    interactive_dir = baseline_dir / "interactive"
    propagation_dir = interactive_dir / "proposals" / "propagation" / "sam2_video"
    label_maps_dir = propagation_dir / "label_maps"
    region_maps_dir = interactive_dir / "regions" / "keyframe_region_maps"
    source_image = tmp_path / "capture" / "cameras" / "rgb" / "scan_000" / "000000.jpg"

    (dataset_dir / "scans" / "scene").mkdir(parents=True)
    (dataset_dir / "scans" / "scene" / "points.pts").write_text("0 0 0\n", encoding="ascii")
    label_maps_dir.mkdir(parents=True)
    region_maps_dir.mkdir(parents=True)
    source_image.parent.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(20, 30, 40)).save(source_image)

    labels = np.asarray(
        [
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 1],
        ],
        dtype=np.uint16,
    )
    np.save(label_maps_dir / "000000.npy", labels)
    Image.fromarray(labels).save(label_maps_dir / "000000.png")
    np.save(region_maps_dir / "000000.npy", labels)

    _write_jsonl(
        dataset_dir / "frame_manifest.jsonl",
        [
            {
                "sai3dFrameId": 0,
                "sourceFrameId": 0,
                "imageName": "000000.jpg",
                "sourceImagePath": str(source_image),
                "width": 4,
                "height": 3,
            }
        ],
    )
    fingerprint = "fixture-fingerprint"
    _write_json(
        propagation_dir / "config.json",
        {
            "methodId": "sam2_video",
            "engineName": "SAM2 Video",
            "inputFingerprint": fingerprint,
            "backend": {
                "checkpoint": "/models/sam2.pt",
                "config": "configs/sam2.yaml",
            },
        },
    )
    _write_json(
        propagation_dir / "input.json",
        {
            "frameCount": 1,
            "fingerprint": fingerprint,
            "primaryFrameId": 0,
            "primaryPolicy": "sequence_first_frame",
            "anchorFrameIds": [0],
            "anchorMaps": [{"frameId": 0, "path": str(region_maps_dir / "000000.npy")}],
            "sourceRegionIds": [1],
            "sourceFrameIdsByRegion": {"1": [0]},
        },
    )
    _write_json(
        propagation_dir / "summary.json",
        {
            "methodId": "sam2_video",
            "method": "fixture propagation",
            "labelSpace": "persistent_region",
            "frameCount": 1,
            "inputFingerprint": fingerprint,
            "timestampUtc": "2026-01-01T00:00:00+00:00",
            "frames": [{"frameId": 0, "coverage": float(np.count_nonzero(labels) / labels.size)}],
        },
    )
    _write_json(
        interactive_dir / "regions" / "regions.json",
        {
            "regions": [
                {
                    "id": 1,
                    "name": "Door",
                    "color": "hsl(105.0 72% 58%)",
                }
            ]
        },
    )
    _write_json(
        interactive_dir / "keyframes" / "keyframes.json",
        {"keyframeIds": [0]},
    )

    output_dir = tmp_path / "share" / "sam2_video"
    result = export_segmentation_share(
        SegmentationShareConfig(
            model_dir=model_dir,
            baseline_name="demo",
            method_id="sam2_video",
            output_dir=output_dir,
        )
    )

    assert result["frameCount"] == 1
    assert result["regionCount"] == 1
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "capture"
    assert manifest["baseline"] == "demo"
    assert manifest["maskEncoding"]["pixelType"] == "uint16"
    assert "/home/" not in (output_dir / "frames.jsonl").read_text(encoding="utf-8")
    exported = np.asarray(Image.open(output_dir / "masks" / "000000.png"))
    assert np.array_equal(exported, labels)
    assert (output_dir / "checksums.sha256").is_file()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
