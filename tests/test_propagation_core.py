from pathlib import Path
from types import SimpleNamespace
import json
import os

import numpy as np

from omega_local.segmentation.interactive.memory_vos_propagation import _subprocess_device_settings
from omega_local.segmentation.interactive.propagation_backends import (
    PropagationBackend,
    PropagationBackendInfo,
    PropagationBackendRegistry,
    PropagationRunInput,
)
from omega_local.segmentation.interactive.proposals import (
    ProposalFrame,
    ProposalManager,
    ProposalRunPaths,
    _count_complete_frames,
)
from omega_local.segmentation.interactive.sam2_bounded_propagation import (
    _Candidate,
    _bounded_intervals,
    _fuse_interval_candidates,
)
from omega_local.segmentation.interactive.sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
    _label_map_from_logits,
    _prepare_video_sources,
)


def _frame(frame_id: int, *, width: int = 4, height: int = 3) -> PropagationFrame:
    return PropagationFrame(
        frame_id=frame_id,
        image_name=f"{frame_id:06d}.jpg",
        image_path=Path(f"{frame_id:06d}.jpg"),
        width=width,
        height=height,
    )


def test_sam2_logits_form_one_exclusive_label_map() -> None:
    logits = np.asarray(
        [
            [[-1.0, 0.4], [0.1, 0.5]],
            [[-2.0, 0.2], [0.8, 0.5]],
        ],
        dtype=np.float32,
    )

    labels = _label_map_from_logits([7, 9], logits)

    np.testing.assert_array_equal(labels, np.asarray([[0, 7], [9, 7]], dtype=np.uint16))


def test_video_sources_are_validated_and_resized_once() -> None:
    frames = [_frame(0, width=4, height=3)]
    labels = np.zeros((3, 4), dtype=np.uint16)
    labels[:, 2:] = 5
    sources = [LabeledPropagationSource(local_index=0, frame_id=0, labels=labels)]

    prepared, labels_by_local, region_ids = _prepare_video_sources(frames, sources, (2, 2))

    assert len(prepared) == 1
    assert prepared[0].labels.shape == (2, 2)
    np.testing.assert_array_equal(prepared[0].labels, labels_by_local[0])
    assert region_ids == [5]


def test_bounded_intervals_cover_before_between_and_after_anchors() -> None:
    assert _bounded_intervals([2, 5], 8) == [
        (0, 2, False, True),
        (2, 5, True, True),
        (5, 7, True, False),
    ]
    assert _bounded_intervals([0], 1) == [(0, 0, True, True)]


def test_bounded_fusion_keeps_agreement_and_abstains_on_weak_conflict() -> None:
    forward = _Candidate(
        labels=np.asarray([[2, 2, 3, 3]], dtype=np.uint16),
        scores=np.asarray([[1.0, 1.0, 0.1, 1.0]], dtype=np.float32),
    )
    backward = _Candidate(
        labels=np.asarray([[2, 0, 4, 4]], dtype=np.uint16),
        scores=np.asarray([[1.0, 0.0, 0.1, 0.1]], dtype=np.float32),
    )

    labels, stats = _fuse_interval_candidates(
        local_index=5,
        left_index=0,
        right_index=10,
        forward=forward,
        backward=backward,
        shape=(1, 4),
    )

    np.testing.assert_array_equal(labels, np.asarray([[2, 2, 0, 3]], dtype=np.uint16))
    assert stats["agreementPixels"] == 1
    assert stats["conflictPixels"] == 2
    assert stats["uncertainPixels"] == 1


def test_memory_vos_cuda_index_is_isolated_for_child_process() -> None:
    assert _subprocess_device_settings("auto") == ("auto", {})
    assert _subprocess_device_settings("cpu") == ("cpu", {"CUDA_VISIBLE_DEVICES": ""})
    assert _subprocess_device_settings("cuda:2") == ("cuda", {"CUDA_VISIBLE_DEVICES": "2"})


def test_shared_output_contract_preserves_exact_anchor_labels() -> None:
    frame = _frame(0)
    anchor = np.zeros((3, 4), dtype=np.uint16)
    anchor[1, 1] = 6
    run_input = PropagationRunInput(
        trigger_frame_id=0,
        frames=(frame,),
        sources=(LabeledPropagationSource(local_index=0, frame_id=0, labels=anchor),),
        region_rows=({"id": 6, "name": "region"},),
        source_frame_ids_by_region={6: (0,)},
        source_area_pixels={6: 1},
        known_region_ids=frozenset({6}),
        anchor_labels_by_frame={0: anchor},
        anchor_map_paths={0: Path("000000.npy")},
        primary_index=0,
        fingerprint="test",
    )

    predicted = np.full((3, 4), 6, dtype=np.uint16)
    validated = run_input.validate_output(frame, predicted)

    np.testing.assert_array_equal(validated, anchor)
    assert not np.shares_memory(validated, anchor)


def test_propagation_readiness_uses_data_files_not_preview_overlays(tmp_path: Path) -> None:
    paths = ProposalRunPaths(
        run_dir=tmp_path,
        label_map_dir=tmp_path / "label_maps",
        overlay_dir=tmp_path / "overlays",
        metadata_dir=tmp_path / "metadata",
        summary=tmp_path / "summary.json",
        progress=tmp_path / "progress.json",
        frame_index=tmp_path / "frames.jsonl",
        config=tmp_path / "config.json",
        input_manifest=tmp_path / "input.json",
    )
    frame = ProposalFrame(3, 4, 3, "000003.jpg", tmp_path / "000003.jpg")
    paths.overlay_dir.mkdir()
    (paths.overlay_dir / "000003.png").touch()

    assert _count_complete_frames(paths, [frame]) == 0

    paths.label_map_dir.mkdir()
    paths.metadata_dir.mkdir()
    np.save(paths.label_map_dir / "000003.npy", np.zeros((3, 4), dtype=np.uint16))
    (paths.metadata_dir / "000003.json").write_text("{}", encoding="utf-8")

    assert _count_complete_frames(paths, [frame]) == 1


def test_propagation_registry_rejects_missing_source_dependency() -> None:
    backend = PropagationBackend(
        info=PropagationBackendInfo(
            method_id="dense",
            display_name="Dense",
            engine_name="test",
            description="test",
            result_description="test",
            runtime_config={},
            stage="dense_recovery",
            source_method_id="missing",
        ),
        session=object(),
        availability_check=lambda: (True, "Ready"),
    )

    try:
        PropagationBackendRegistry([backend])
    except ValueError as exc:
        assert "unregistered source" in str(exc)
    else:
        raise AssertionError("Registry accepted a missing source dependency.")


def test_full_propagation_becomes_stale_after_anchor_update(tmp_path: Path) -> None:
    interactive_dir = tmp_path / "interactive"
    regions_summary = interactive_dir / "regions" / "regions.json"
    regions_summary.parent.mkdir(parents=True)
    regions_summary.write_text("{}", encoding="utf-8")
    manager = ProposalManager(
        SimpleNamespace(
            interactive_dir=interactive_dir,
            regions_summary=regions_summary,
        ),
        SimpleNamespace(),
    )
    frame = ProposalFrame(0, 4, 3, "000000.jpg", tmp_path / "000000.jpg")
    paths = manager.propagation_paths("video")
    paths.label_map_dir.mkdir(parents=True)
    paths.metadata_dir.mkdir()
    np.save(paths.label_map_dir / "000000.npy", np.zeros((3, 4), dtype=np.uint16))
    (paths.metadata_dir / "000000.json").write_text("{}", encoding="utf-8")
    paths.summary.write_text(json.dumps({"timestampUtc": "2026-01-01T00:00:00Z"}), encoding="utf-8")
    os.utime(paths.summary, ns=(10, 10))
    os.utime(regions_summary, ns=(20, 20))

    status = manager.layer_status(
        [frame],
        {
            "methods": [
                {
                    "methodId": "video",
                    "displayName": "Video",
                    "supportsFullRun": True,
                    "supportsRegionPair": False,
                    "available": True,
                }
            ]
        },
    )
    layer = next(row for row in status["layers"] if row["methodId"] == "video")

    assert layer["ready"] is False
    assert layer["stale"] is True
    assert "manual anchors" in layer["message"]


def test_interrupted_propagation_is_reconciled_on_startup(tmp_path: Path) -> None:
    manager = ProposalManager(
        SimpleNamespace(interactive_dir=tmp_path, regions_summary=tmp_path / "regions.json"),
        SimpleNamespace(),
    )
    progress_path = manager.propagation_paths("video").progress
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({"status": "running", "message": "old worker"}),
        encoding="utf-8",
    )

    manager.reconcile_interrupted_propagation_runs(
        {"methods": [{"methodId": "video"}]}
    )

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "interrupted"
    assert "interrupted" in progress["message"]
