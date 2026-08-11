"""Tests for the complete Segment then Splat paper bridge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omega_local.reconstruction.pipelines.segment_then_splat_official.contract import (
    OfficialSegmentThenSplatConfig,
)
from omega_local.reconstruction.pipelines.segment_then_splat_official.initialize import (
    _merge_similar_objects,
)
from omega_local.segmentation.interactive.static_semantic_runs import (
    discover_static_semantic_3dgs_runs,
)


def test_paper_config_keeps_released_defaults(tmp_path: Path) -> None:
    config = OfficialSegmentThenSplatConfig(
        model_dir=tmp_path,
        editor_baseline_name="baseline",
        source_root=tmp_path,
        python=tmp_path / "python",
        sam1_checkpoint=tmp_path / "sam1.pth",
        sam2_checkpoint=tmp_path / "sam2.pt",
        native_extensions_dir=tmp_path / "extensions",
    ).normalized()
    assert config.iterations == 40_000
    assert config.densify_until_iter == 20_000
    assert config.detect_stride == 10
    assert config.image_long_side == 1024


def test_geometry_color_merge_reindexes_masks_and_points() -> None:
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    colors = np.asarray(
        [[100, 100, 100], [101, 100, 100], [240, 20, 20]],
        dtype=np.uint8,
    )
    labels = np.asarray([0, 1, 2], dtype=np.uint8)
    masks = [
        [np.asarray([[255, 0, 0]], dtype=np.uint8)],
        [np.asarray([[0, 255, 0]], dtype=np.uint8)],
        [np.asarray([[0, 0, 255]], dtype=np.uint8)],
    ]
    merged_labels, merged_masks, pairs = _merge_similar_objects(
        positions,
        colors,
        labels,
        masks,
    )
    assert pairs == [(0, 1)]
    assert len(merged_masks) == 2
    assert merged_labels.tolist() == [0, 0, 1]
    assert merged_masks[0][0].tolist() == [[255, 255, 0]]


def test_paper_exports_are_discovered_as_multilevel_static_runs(
    tmp_path: Path,
) -> None:
    run_dir = (
        tmp_path
        / "experiments"
        / "segment_then_splat_paper"
        / "runs"
        / "official_auto"
    )
    outputs = run_dir / "05_outputs"
    objects = outputs / "objects" / "large"
    objects.mkdir(parents=True)
    scene = outputs / "scene.ply"
    part = objects / "region_000000.ply"
    scene.write_bytes(b"ply")
    part.write_bytes(b"ply")
    manifest = {
        "timestampUtc": "2026-08-01T00:00:00+00:00",
        "scenePly": str(scene),
        "gaussianCount": 12,
        "levels": {
            "large": {
                "objectCount": 0,
                "assignedGaussianCount": 0,
                "objects": [
                    {
                        "regionId": 0,
                        "name": "Background / Unassigned",
                        "gaussianCount": 12,
                        "ply": str(part),
                    }
                ],
            }
        },
    }
    (outputs / "stage.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (outputs / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    rows = discover_static_semantic_3dgs_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["methodId"] == "segment_then_splat_paper"
    assert rows[0]["maskMethodId"] == (
        "segment_then_splat_paper_official_auto_large"
    )
    assert rows[0]["gaussianCount"] == 12
