from __future__ import annotations

import json
from pathlib import Path

from omega_local.segmentation.interactive.static_semantic_runs import (
    discover_static_semantic_3dgs_runs,
    live_static_semantic_3dgs_experiment,
)


def test_completed_static_run_is_discovered_for_multipart_viewer(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    source = (
        interactive
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
        / "map_run"
    )
    outputs = (
        source
        / "04_static_semantic_3dgs"
        / "runs"
        / "segment_then_splat"
        / "joint"
        / "03_outputs"
    )
    objects = outputs / "objects"
    objects.mkdir(parents=True)
    scene = outputs / "scene.ply"
    region = objects / "region_000007.ply"
    scene.write_bytes(b"ply\nscene")
    region.write_bytes(b"ply\nregion")
    (source / "run.json").write_text(
        json.dumps(
            {
                "runId": "map_run",
                "propagationMethod": "sam2_video",
                "manualFrameWeight": 8,
            }
        ),
        encoding="utf-8",
    )
    (outputs / "stage.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (outputs / "manifest.json").write_text(
        json.dumps(
            {
                "timestampUtc": "2026-08-01T00:00:00+00:00",
                "gaussianCount": 12,
                "foregroundGaussianCount": 12,
                "backgroundGaussianCount": 0,
                "scenePly": str(scene),
            }
        ),
        encoding="utf-8",
    )
    (outputs / "regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "persistentRegionId": 7,
                        "name": "Door",
                        "gaussianCount": 12,
                        "ply": str(region),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = discover_static_semantic_3dgs_runs(interactive)

    assert len(rows) == 1
    row = rows[0]
    assert row["experimentFamily"] == "static_semantic_3dgs"
    assert row["gaussianCount"] == 12
    assert row["regionCount"] == 1
    assert [item["variantId"] for item in row["gaussianArtifacts"]] == [
        "scene",
        "rgb_region_000007",
    ]
    payload = live_static_semantic_3dgs_experiment(
        interactive,
        row["runId"],
    )
    assert payload is not None
    assert payload["gaussianArtifacts"]["scene"]["path"] == str(scene)


def test_incomplete_partition_is_not_exposed(tmp_path: Path) -> None:
    interactive = tmp_path / "interactive"
    source = (
        interactive
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
        / "map_run"
    )
    outputs = (
        source
        / "04_static_semantic_3dgs"
        / "runs"
        / "gaussian_grouping"
        / "joint"
        / "03_outputs"
    )
    outputs.mkdir(parents=True)
    scene = outputs / "scene.ply"
    scene.write_bytes(b"ply\nscene")
    (outputs / "stage.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (outputs / "manifest.json").write_text(
        json.dumps({"gaussianCount": 12, "scenePly": str(scene)}),
        encoding="utf-8",
    )
    (outputs / "regions.json").write_text(
        json.dumps({"regions": []}),
        encoding="utf-8",
    )

    assert discover_static_semantic_3dgs_runs(interactive) == []
