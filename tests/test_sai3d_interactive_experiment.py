import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omega_local.segmentation.interactive.sai3d_experiment import (
    SAI3DExperimentBackend,
    SAI3DExperimentConfig,
    _label_map_status,
)
from omega_local.segmentation.interactive.segmentation3d_contract import Segmentation3DRunRequest
from omega_local.segmentation.interactive.segmentation3d_manager import Segmentation3DManager
from omega_local.segmentation.sai3d_baseline import _repeat_weighted_views


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_repeat_weighted_views_repeats_matching_label_and_visibility_columns() -> None:
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    seen = np.array([[True, False, True], [False, True, True]])

    weighted_labels, weighted_seen = _repeat_weighted_views(
        labels,
        seen,
        np.array([1, 3, 2], dtype=np.int32),
    )

    expected_columns = np.array([0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(weighted_labels, labels[:, expected_columns])
    np.testing.assert_array_equal(weighted_seen, seen[:, expected_columns])


def test_weighted_input_substitutes_completed_manual_frames_with_symlinks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    interactive_dir = tmp_path / "interactive"
    frame_manifest = dataset_dir / "frame_manifest.jsonl"
    region_maps_dir = interactive_dir / "regions" / "keyframe_region_maps"
    regions_summary = interactive_dir / "regions" / "regions.json"
    propagation_masks = interactive_dir / "proposals" / "propagation" / "sam2_video" / "label_maps"
    propagation_masks.mkdir(parents=True)
    region_maps_dir.mkdir(parents=True)
    posed_images_dir = dataset_dir / "posed_images"
    posed_images_dir.mkdir(parents=True)

    rows = []
    for frame_id in range(2):
        (posed_images_dir / f"{frame_id}.jpg").touch()
        rows.append(
            {
                "sai3dFrameId": frame_id,
                "imageName": f"{frame_id:06d}.jpg",
                "colorPath": f"posed_images/{frame_id}.jpg",
                "posePath": f"posed_images/{frame_id}.txt",
                "width": 4,
                "height": 3,
                "fx": 2.0,
                "fy": 2.0,
                "cx": 1.5,
                "cy": 1.0,
            }
        )
        np.save(propagation_masks / f"{frame_id:06d}.npy", np.full((3, 4), frame_id + 1, np.uint16))
    _write_jsonl(frame_manifest, rows)
    np.save(region_maps_dir / "000001.npy", np.full((3, 4), 9, np.uint16))
    regions_summary.parent.mkdir(parents=True, exist_ok=True)
    regions_summary.write_text(
        json.dumps({"frameStates": {"1": {"complete": True}}}),
        encoding="utf-8",
    )

    paths = SimpleNamespace(
        dataset_dir=dataset_dir,
        frame_manifest=frame_manifest,
        interactive_dir=interactive_dir,
        region_maps_dir=region_maps_dir,
        regions_summary=regions_summary,
        segmentation3d_dir=interactive_dir / "3d_segmentation",
    )
    backend = SAI3DExperimentBackend(
        SAI3DExperimentConfig(paths=paths, sai3d_root=tmp_path / "SAI3D")
    )
    request = Segmentation3DRunRequest(
        method_id="sai3d",
        input_id="propagated_weighted",
        source_id="sam2_video",
        manual_frame_weight=4,
        point_budget=400_000,
        superpoint_target=60_000,
        run_id="sai3d_propagated_weighted_sam2_video_w4",
        run_dir=tmp_path / "run",
    )

    staged = backend._stage_weighted_input(request)
    staged_rows = [json.loads(line) for line in (staged / "frames.jsonl").read_text().splitlines()]

    assert (staged / "label_maps" / "000000.npy").resolve() == (propagation_masks / "000000.npy").resolve()
    assert (staged / "label_maps" / "000001.npy").resolve() == (region_maps_dir / "000001.npy").resolve()
    assert [row["viewWeight"] for row in staged_rows] == [1, 4]
    assert [row["isManualAnchor"] for row in staged_rows] == [False, True]


def test_saved_runs_keep_distinct_visualization_metadata_and_newest_first(tmp_path: Path) -> None:
    root = tmp_path / "3d_segmentation"
    runs = [
        {
            "runId": "sai3d_sam2_auto",
            "methodId": "sai3d",
            "inputId": "sam2_auto",
            "sourceId": "",
            "manualFrameWeight": 1,
            "pointBudget": 200_000,
            "superpointTarget": 4_000,
            "timestampUtc": "2026-07-20T12:00:00+00:00",
        },
        {
            "runId": "sai3d_propagated_weighted_sam2_video_w4",
            "methodId": "sai3d",
            "inputId": "propagated_weighted",
            "sourceId": "sam2_video",
            "manualFrameWeight": 4,
            "pointBudget": 400_000,
            "superpointTarget": 8_000,
            "geometrySource": "omega_final_clean_hybrid",
            "timestampUtc": "2026-07-21T12:00:00+00:00",
        },
    ]
    for payload in runs:
        path = root / "runs" / payload["runId"] / "experiment.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    manager = Segmentation3DManager(root, SimpleNamespace(), max_points=100, seed=71)
    saved = manager._saved_runs()

    assert [row["runId"] for row in saved] == [
        "sai3d_propagated_weighted_sam2_video_w4",
        "sai3d_sam2_auto",
    ]
    assert saved[0]["sourceId"] == "sam2_video"
    assert saved[0]["manualFrameWeight"] == 4
    assert saved[0]["pointBudget"] == 400_000
    assert saved[0]["superpointTarget"] == 8_000
    assert saved[0]["geometrySource"] == "omega_final_clean_hybrid"
    assert saved[1]["geometrySource"] == "omega_final_clean_hybrid"


def test_label_map_status_requires_the_exact_manifest_frame_set(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    label_dir = run_dir / "label_maps"
    label_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    np.save(label_dir / "000000.npy", np.zeros((2, 2), dtype=np.uint16))
    np.save(label_dir / "000099.npy", np.zeros((2, 2), dtype=np.uint16))

    assert _label_map_status(run_dir, [0, 1]) == (1, False)

    np.save(label_dir / "000001.npy", np.zeros((2, 2), dtype=np.uint16))
    assert _label_map_status(run_dir, [0, 1]) == (2, True)


def test_sai3d_rejects_propagation_older_than_manual_anchors(tmp_path: Path) -> None:
    interactive_dir = tmp_path / "interactive"
    run_dir = interactive_dir / "proposals" / "propagation" / "video"
    label_dir = run_dir / "label_maps"
    label_dir.mkdir(parents=True)
    np.save(label_dir / "000000.npy", np.zeros((2, 2), dtype=np.uint16))
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    registry_path = interactive_dir / "proposals" / "propagation" / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "methods": [
                    {
                        "methodId": "video",
                        "displayName": "Video",
                        "supportsFullRun": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    regions_summary = interactive_dir / "regions" / "regions.json"
    regions_summary.parent.mkdir(parents=True)
    regions_summary.write_text("{}", encoding="utf-8")
    os.utime(run_dir / "summary.json", ns=(10, 10))
    os.utime(regions_summary, ns=(20, 20))
    backend = SAI3DExperimentBackend(
        SAI3DExperimentConfig(
            paths=SimpleNamespace(
                interactive_dir=interactive_dir,
                regions_summary=regions_summary,
            ),
            sai3d_root=tmp_path / "SAI3D",
        )
    )

    sources = backend._propagation_sources([0])

    assert len(sources) == 1
    assert sources[0]["ready"] is False
    assert "anchors changed" in sources[0]["message"]


def test_sai3d_rejects_dense_recovery_older_than_its_source(tmp_path: Path) -> None:
    interactive_dir = tmp_path / "interactive"
    propagation_root = interactive_dir / "proposals" / "propagation"
    for method_id in ["points", "dense"]:
        run_dir = propagation_root / method_id
        (run_dir / "label_maps").mkdir(parents=True)
        np.save(run_dir / "label_maps" / "000000.npy", np.zeros((2, 2), dtype=np.uint16))
        (run_dir / "summary.json").write_text("{}", encoding="utf-8")
    (propagation_root / "registry.json").write_text(
        json.dumps(
            {
                "methods": [
                    {"methodId": "points", "displayName": "Points", "supportsFullRun": True},
                    {
                        "methodId": "dense",
                        "displayName": "Dense",
                        "supportsFullRun": True,
                        "sourceMethodId": "points",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    regions_summary = interactive_dir / "regions" / "regions.json"
    regions_summary.parent.mkdir(parents=True)
    regions_summary.write_text("{}", encoding="utf-8")
    os.utime(regions_summary, ns=(5, 5))
    os.utime(propagation_root / "dense" / "summary.json", ns=(10, 10))
    os.utime(propagation_root / "points" / "summary.json", ns=(20, 20))
    backend = SAI3DExperimentBackend(
        SAI3DExperimentConfig(
            paths=SimpleNamespace(
                interactive_dir=interactive_dir,
                regions_summary=regions_summary,
            ),
            sai3d_root=tmp_path / "SAI3D",
        )
    )

    sources = {row["sourceId"]: row for row in backend._propagation_sources([0])}

    assert sources["points"]["ready"] is True
    assert sources["dense"]["ready"] is False
    assert "Source propagation changed" in sources["dense"]["message"]
