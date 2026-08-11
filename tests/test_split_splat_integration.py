from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from omega_local.segmentation.interactive.gaussian_viewer import viewer_settings
from omega_local.segmentation.interactive.segmentation3d_manager import Segmentation3DManager
from omega_local.segmentation.interactive.propagation_backends import (
    _discover_split_splat_backends,
)
from omega_local.segmentation.split_splat.adapter import (
    _match_colmap_image,
    _read_colmap_images_text,
)
from omega_local.segmentation.split_splat.anchored_split import (
    _assign_point_labels,
    _relabel_components,
)
from omega_local.segmentation.split_splat.anchored_ownership import (
    PROVENANCE_GEOMETRY,
    complete_point_labels,
)
from omega_local.reconstruction.gaussian_io import (
    ensure_viewer_gaussian_ply,
    write_segmented_gaussian_ply,
    write_viewer_gaussian_partitions,
    write_viewer_gaussian_subset,
)
from omega_local.segmentation.split_splat.artifacts import (
    _exclusive_global_labels,
    finalize_official_proposals,
    _match_global_gaussian_labels,
    register_proposal_layer,
    register_splat_refined_mask_layer,
)
from omega_local.segmentation.split_splat.contract import (
    SPLAT_STAGE_ORDER,
    STAGE_DEPENDENCIES,
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    default_shared_id,
    read_complete_stage,
    replace_symlink,
    write_jsonl,
)
from omega_local.segmentation.split_splat.depth_alignment import (
    _apply_inverse_depth_affine,
    _fit_inverse_depth_affine,
)
from omega_local.segmentation.split_splat.launch import (
    _paper_mask_refinement_source,
    _rank_safe_convex_hull,
)
from omega_local.segmentation.split_splat.manager import (
    SplitSplatExperimentManager,
)
from omega_local.segmentation.split_splat.proposal_sources import (
    import_propagation_proposals,
)
from omega_local.segmentation.split_splat.splat import (
    _CompositionNode,
    _composition_cache_input,
    _composition_cache_status,
    _composition_file_identity,
    _composition_model_paths,
    _composition_source_key,
    _composition_storage_key,
    _cleanup_composition_training_artifacts,
    _compose_mask_directories,
    _directional_collision_score,
    _prepare_mask_refinement_workspace,
    _select_collision_pairs,
    prepare_splat,
)
from omega_local.segmentation.split_splat.storage import (
    cleanup_completed_composition,
    cleanup_completed_models,
)
from omega_local.segmentation.split_splat.train_launch import (
    _patched_source,
    _reset_opacity_before_training,
)
from omega_local.segmentation.split_splat.upstream import run_command
from omega_local.segmentation.split_splat.upstream import (
    composition_train_command,
    instance_train_command,
)
from omega_local.stage_graph import downstream_stages


def test_depth_alignment_recovers_inverse_depth_affine() -> None:
    reference = np.linspace(2.0, 9.0, 120, dtype=np.float32).reshape(10, 12)
    scale = 1.8
    offset = 0.025
    predicted = scale / (1.0 / reference - offset)

    fitted_scale, fitted_offset, support = _fit_inverse_depth_affine(
        predicted,
        reference,
    )
    aligned = _apply_inverse_depth_affine(
        predicted,
        fitted_scale,
        fitted_offset,
    )

    assert support == reference.size
    assert np.isclose(fitted_scale, scale, rtol=1e-5)
    assert np.isclose(fitted_offset, offset, rtol=1e-5)
    assert np.allclose(aligned, reference, rtol=1e-5, atol=1e-5)


def test_split_splat_stage_graph_invalidates_only_true_dependents() -> None:
    expected = (
        "split",
        "splat_prepare",
        "splat_initial",
        "splat_masks",
        "splat_refined",
        "splat_compose",
    )
    actual = downstream_stages(
        "global_gs",
        order=tuple(STAGE_DEPENDENCIES),
        dependencies=STAGE_DEPENDENCIES,
    )

    assert actual == expected
    assert "proposals" not in actual


def test_split_preview_keeps_overlap_with_smaller_instance_priority() -> None:
    large = np.zeros((5, 6), dtype=bool)
    large[1:5, 1:6] = True
    small = np.zeros((5, 6), dtype=bool)
    small[2:4, 2:4] = True

    labels, conflict_pixels = _exclusive_global_labels(
        [(7, large), (3, small)],
        width=6,
        height=5,
    )

    assert conflict_pixels == 4
    assert np.all(labels[small] == 3)
    assert np.all(labels[large & ~small] == 7)
    assert np.count_nonzero(labels) == np.count_nonzero(large | small)


def test_colmap_image_parser_accepts_models_without_observation_lines(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text(
        "\n".join(
            [
                "# Image list",
                "1 1 0 0 0 0 0 0 1 scan_000_000000.jpg",
                "",
                "2 1 0 0 0 1 0 0 2 scan_000_000001.jpg",
                "",
            ]
        ),
        encoding="utf-8",
    )

    rows = _read_colmap_images_text(path)

    assert [row["image_id"] for row in rows] == [1, 2]
    assert [row["name"] for row in rows] == [
        "scan_000_000000.jpg",
        "scan_000_000001.jpg",
    ]


def test_colmap_matching_never_falls_back_to_frame_order() -> None:
    rows = [
        {"image_id": 1, "name": "frame_000.JPEG"},
        {"image_id": 2, "name": "frame_001.JPEG"},
    ]

    assert _match_colmap_image(rows, "frame_001.jpg")["image_id"] == 2
    with pytest.raises(KeyError, match="No COLMAP image matches"):
        _match_colmap_image(rows, "missing.jpg")


def test_stage_cache_requires_explicit_completion(tmp_path: Path) -> None:
    summary = tmp_path / "stage.json"
    atomic_write_json(summary, {"stage": "split"})
    assert read_complete_stage(summary) is None

    atomic_write_json(summary, {"stage": "split", "status": "complete"})
    assert read_complete_stage(summary) == {
        "stage": "split",
        "status": "complete",
    }


def test_shared_symlink_replaces_existing_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "staged"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale", encoding="utf-8")

    replace_symlink(destination, source)

    assert destination.is_symlink()
    assert destination.resolve() == source.resolve()


def test_official_proposal_finalizer_preserves_binary_masks_and_builds_preview(
    tmp_path: Path,
) -> None:
    paths = SplitSplatRunPaths(
        tmp_path / "experiments",
        "split_splat_official",
        "dataset_test",
        30_000,
    )
    source = (
        paths.proposals_dir
        / "work"
        / "output"
        / "split_splat_official_autoseg_mask"
        / "scan_000_000000"
    )
    source.mkdir(parents=True)
    first = np.zeros((6, 8), dtype=np.uint8)
    first[1:5, 1:7] = 255
    second = np.zeros((6, 8), dtype=np.uint8)
    second[2:4, 3:5] = 180
    Image.fromarray(first).save(source / "0.png")
    Image.fromarray(second).save(source / "1.png")
    write_jsonl(
        paths.frame_map,
        [
            {
                "frameId": 0,
                "splitSplatImageName": "scan_000_000000.JPEG",
                "width": 8,
                "height": 6,
            }
        ],
    )

    summary = finalize_official_proposals(paths)
    preview = np.asarray(Image.open(paths.proposal_label_maps / "000000.png"))

    assert summary["frameCount"] == 1
    assert summary["frames"][0]["proposalCount"] == 2
    assert (
        paths.proposal_binary_masks / "scan_000_000000" / "0.png"
    ).is_file()
    assert int(preview[2, 3]) == 2
    assert int(preview[1, 1]) == 1


def test_mask_runs_share_geometry_but_not_proposal_outputs(tmp_path: Path) -> None:
    config_a = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=tmp_path / "python",
        run_id="official",
        depth_source="murre",
        depth_dir=tmp_path / "depth",
    ).normalized()
    config_b = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=tmp_path / "python",
        run_id="user_masks",
        proposal_source="propagation",
        propagation_method="sam2_video",
        depth_source="murre",
        depth_dir=tmp_path / "depth",
    ).normalized()
    shared_id = default_shared_id(config_a)
    assert shared_id == default_shared_id(config_b)

    official = SplitSplatRunPaths(tmp_path, config_a.run_id, shared_id, 30_000)
    user_masks = SplitSplatRunPaths(tmp_path, config_b.run_id, shared_id, 30_000)

    assert official.dataset_dir == user_masks.dataset_dir
    assert official.global_gs_model == user_masks.global_gs_model
    assert official.proposals_dir != user_masks.proposals_dir
    assert official.consistent_masks_dir != user_masks.consistent_masks_dir


def test_split_splat_config_rejects_invalid_numeric_settings(tmp_path: Path) -> None:
    common = {
        "model_dir": tmp_path / "model",
        "editor_baseline_name": "editor",
        "split_splat_root": tmp_path / "Split_and_Splat",
        "python": tmp_path / "python",
    }
    with pytest.raises(ValueError, match="Global iterations must be positive"):
        SplitSplatRunConfig(**common, iterations=0).normalized()
    with pytest.raises(ValueError, match=r"finite values in \[0, 1\]"):
        SplitSplatRunConfig(
            **common,
            composition_mask_weights=(0.05, float("nan")),
        ).normalized()
    with pytest.raises(ValueError, match="Unknown depth source"):
        SplitSplatRunConfig(**common, depth_source="unknown").normalized()


def test_anchored_split_reuses_geometry_and_requires_persistent_proposals(
    tmp_path: Path,
) -> None:
    released = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=tmp_path / "python",
        run_id="released",
        proposal_source="propagation",
        propagation_method="sam2_video",
    ).normalized()
    anchored = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=tmp_path / "python",
        run_id="anchored",
        proposal_source="propagation",
        propagation_method="sam2_video",
        split_method="anchored_3d",
        manual_frame_weight=8,
    ).normalized()

    assert anchored.split_method == "anchored_3d"
    assert anchored.manual_frame_weight == 8
    assert default_shared_id(anchored) == default_shared_id(released)
    with pytest.raises(ValueError, match="requires persistent-region"):
        SplitSplatRunConfig(
            model_dir=tmp_path / "model",
            editor_baseline_name="editor",
            split_splat_root=tmp_path / "Split_and_Splat",
            python=tmp_path / "python",
            proposal_source="official_auto",
            split_method="anchored_3d",
        ).normalized()


def test_anchored_votes_prioritize_manual_evidence_without_rejection() -> None:
    manual = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 0],
        ],
        dtype=np.uint16,
    )
    propagated = np.array(
        [
            [0, 20],
            [0, 3],
            [2, 2],
            [0, 3],
        ],
        dtype=np.uint16,
    )

    labels, _confidence, provenance, summary = _assign_point_labels(
        manual,
        propagated,
        np.array([2, 7], dtype=np.uint16),
        manual_frame_weight=8,
    )

    assert labels.tolist() == [2, 7, 2, 7]
    assert provenance.tolist() == [2, 2, 1, 1]
    assert summary["manualTiePointCount"] == 1
    assert summary["directlyObservedPointCount"] == 4
    assert summary["unobservedPointCount"] == 0


def test_anchored_geometry_completion_assigns_every_unobserved_point() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [9.9, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([2, 7, 0, 0], dtype=np.uint16)
    confidence = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    provenance = np.array([2, 1, 0, 0], dtype=np.uint8)

    output, output_confidence, output_provenance, summary = (
        complete_point_labels(
            points,
            labels,
            confidence,
            provenance,
            np.array([2, 7], dtype=np.uint16),
        )
    )

    assert output.tolist() == [2, 7, 2, 7]
    assert np.all(output_confidence > 0)
    assert output_provenance.tolist() == [
        2,
        1,
        PROVENANCE_GEOMETRY,
        PROVENANCE_GEOMETRY,
    ]
    assert summary["completedPointCount"] == 2
    assert summary["regionIdsWithoutDirectSeeds"] == []


def test_anchored_relabeling_changes_identity_without_changing_shape() -> None:
    source = np.zeros((8, 10), dtype=np.uint16)
    source[1:4, 1:4] = 3
    source[4:7, 6:9] = 3
    support = np.zeros_like(source)
    support[1:4, 1:4] = 3
    support[4:7, 6:9] = 9

    output, summary = _relabel_components(
        source,
        support,
        minimum_support=4,
        minimum_support_fraction=0.001,
        minimum_majority=0.6,
    )

    assert np.array_equal(output > 0, source > 0)
    assert np.all(output[1:4, 1:4] == 3)
    assert np.all(output[4:7, 6:9] == 9)
    assert summary["reassignedComponentCount"] == 1
    assert summary["changedPixelCount"] == 9
    assert summary["reassignments"] == [
        {
            "sourceRegionId": 3,
            "targetRegionId": 9,
            "componentCount": 1,
            "pixelCount": 9,
            "supportPixelCount": 9,
            "targetSupportPixelCount": 9,
        }
    ]


def test_anchored_relabeling_rejects_sparse_support_on_large_component() -> None:
    source = np.full((100, 100), 3, dtype=np.uint16)
    support = np.zeros_like(source)
    support.flat[:8] = 9

    output, summary = _relabel_components(
        source,
        support,
        minimum_support=4,
        minimum_support_fraction=0.001,
        minimum_majority=0.6,
    )

    assert np.array_equal(output, source)
    assert summary["insufficientSupportComponentCount"] == 1
    assert summary["reassignedComponentCount"] == 0


def test_atomic_writers_do_not_extend_long_target_names(tmp_path: Path) -> None:
    json_path = tmp_path / ("x" * 250 + ".json")
    jsonl_path = tmp_path / ("y" * 249 + ".jsonl")

    atomic_write_json(json_path, {"complete": True})
    write_jsonl(jsonl_path, [{"frame": 1}])

    assert json.loads(json_path.read_text()) == {"complete": True}
    assert json.loads(jsonl_path.read_text()) == {"frame": 1}


def test_run_identifiers_reject_unsafe_component_lengths(
    tmp_path: Path,
) -> None:
    config = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=Path(sys.executable),
        run_id="x" * 97,
    )

    with pytest.raises(ValueError, match="Run ID is too long"):
        config.normalized()


def test_propagation_proposals_preserve_source_ids_and_binary_contract(
    tmp_path: Path,
) -> None:
    paths = SplitSplatRunPaths(
        tmp_path / "experiments",
        "split_splat_sam2_video",
        "dataset_test",
        30_000,
    )
    write_jsonl(
        paths.frame_map,
        [
            {
                "frameId": 0,
                "splitSplatImageName": "scan_000_000000.JPEG",
                "width": 8,
                "height": 6,
            }
        ],
    )
    interactive_dir = tmp_path / "interactive"
    source_dir = (
        interactive_dir / "proposals" / "propagation" / "sam2_video"
    )
    (source_dir / "label_maps").mkdir(parents=True)
    labels = np.zeros((6, 8), dtype=np.uint16)
    labels[1:4, 1:3] = 7
    labels[2:5, 5:7] = 19
    np.save(source_dir / "label_maps" / "000000.npy", labels)
    (source_dir / "summary.json").write_text(
        json.dumps(
            {
                "frameCount": 1,
                "labelSpace": "persistent_region",
                "methodId": "sam2_video",
                "displayName": "SAM2 Video",
                "inputFingerprint": "test-input",
            }
        ),
        encoding="utf-8",
    )
    config = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=tmp_path / "python",
        run_id="split_splat_sam2_video",
        proposal_source="propagation",
        propagation_method="sam2_video",
    ).normalized()

    summary = import_propagation_proposals(
        config,
        SimpleNamespace(interactive_dir=interactive_dir),
        paths,
    )

    mask_dir = (
        paths.proposal_binary_masks / "scan_000_000000"
    )
    assert summary["proposalSource"] == "propagation"
    assert summary["sourceLabelIds"] == [7, 19]
    assert sorted(path.name for path in mask_dir.glob("*.png")) == [
        "000007.png",
        "000019.png",
    ]
    roundtrip = np.load(paths.proposal_label_maps / "000000.npy")
    assert np.array_equal(roundtrip, labels)


def test_editor_discovers_each_split_splat_run_as_its_own_layer(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "interactive" / "tmp"
    summary_path = (
        tmp_path
        / "interactive"
        / "proposals"
        / "propagation"
        / "split_splat_user_masks"
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(
        json.dumps(
            {
                "methodId": "split_splat_user_masks",
                "displayName": "Split&Splat User Masks",
                "labelSpace": "split_splat_instance",
                "layerGroup": "refined_masks",
                "canonicalRunDir": str(tmp_path / "runs" / "user_masks"),
            }
        ),
        encoding="utf-8",
    )

    backends = _discover_split_splat_backends(work_root)

    assert len(backends) == 1
    assert backends[0].info.method_id == "split_splat_user_masks"
    assert backends[0].info.layer_group == "refined_masks"
    assert backends[0].info.read_only is True
    assert backends[0].status()["available"] is True


def test_splat_refined_masks_use_rgba_alpha_and_register_editor_layer(
    tmp_path: Path,
) -> None:
    paths = SplitSplatRunPaths(
        tmp_path / "experiments",
        "split_splat_official",
        "dataset_test",
        30_000,
    )
    write_jsonl(
        paths.frame_map,
        [
            {
                "frameId": 0,
                "splitSplatImageName": "scan_000_000000.JPEG",
                "width": 8,
                "height": 6,
            }
        ],
    )
    paths.consistent_label_maps.mkdir(parents=True)
    split_labels = np.zeros((6, 8), dtype=np.uint16)
    split_labels[1:3, 1:3] = 1
    Image.fromarray(split_labels).save(
        paths.consistent_label_maps / "000000.png"
    )
    paths.stage_summary("split").write_text(
        json.dumps(
            {
                "instanceIds": [1, 2],
                "labelNamespace": "split_splat_instance",
            }
        ),
        encoding="utf-8",
    )

    instances = []
    for instance_id in (1, 2):
        dataset_dir = paths.splat_instances_dir / str(instance_id)
        (dataset_dir / "masks").mkdir(parents=True)
        instances.append(
            {
                "instanceId": instance_id,
                "datasetDir": str(dataset_dir),
            }
        )
    paths.stage_summary("splat_prepare").write_text(
        json.dumps({"instances": instances}),
        encoding="utf-8",
    )

    grayscale = np.zeros((6, 8), dtype=np.uint8)
    grayscale[1:3, 1:3] = 255
    Image.fromarray(grayscale).save(
        paths.splat_instances_dir
        / "1"
        / "masks"
        / "scan_000_000000.png"
    )
    rgba = np.full((6, 8, 4), 255, dtype=np.uint8)
    rgba[..., 3] = 0
    rgba[3:5, 5:7, 3] = 255
    Image.fromarray(rgba, mode="RGBA").save(
        paths.splat_instances_dir
        / "2"
        / "masks"
        / "scan_000_000000.png"
    )
    editor_paths = SimpleNamespace(interactive_dir=tmp_path / "interactive")
    summary = register_splat_refined_mask_layer(
        paths,
        editor_paths,
        {
            "method": "Released refinement",
            "instances": [
                {"instanceId": 1, "refinedOrAddedMaskCount": 0},
                {"instanceId": 2, "refinedOrAddedMaskCount": 1},
            ],
        },
    )

    labels = np.load(paths.splat_mask_label_maps / "000000.npy")
    layer_summary = json.loads(
        (
            editor_paths.interactive_dir
            / "proposals"
            / "propagation"
            / "split_splat_official_splat_masks"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )

    assert labels[0, 0] == 0
    assert np.all(labels[1:3, 1:3] == 1)
    assert np.all(labels[3:5, 5:7] == 2)
    assert summary["refinedOrAddedMaskCount"] == 1
    assert summary["totalChangedPixelsFromSplit"] == 4
    assert layer_summary["stage"] == "splat_refined_masks"
    assert layer_summary["labelSpace"] == "split_splat_instance"


def test_editor_discovers_only_canonical_persistent_region_split_layers(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "interactive" / "tmp"
    propagation_root = (
        tmp_path / "interactive" / "proposals" / "propagation"
    )
    for method_id, canonical in (
        ("sam2_video", False),
        ("split_splat_anchored", True),
    ):
        method_dir = propagation_root / method_id
        method_dir.mkdir(parents=True)
        payload = {
            "methodId": method_id,
            "labelSpace": "persistent_region",
            "layerGroup": "refined_masks",
        }
        if canonical:
            payload["canonicalRunDir"] = str(tmp_path / "runs" / method_id)
        (method_dir / "summary.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    backends = _discover_split_splat_backends(work_root)

    assert [backend.info.method_id for backend in backends] == [
        "split_splat_anchored"
    ]


def test_editor_registers_split_splat_input_proposals_as_read_only_layer(
    tmp_path: Path,
) -> None:
    paths = SplitSplatRunPaths(
        tmp_path / "experiments",
        "split_splat_official",
        "dataset_test",
        30_000,
    )
    labels = np.zeros((6, 8), dtype=np.uint16)
    labels[1:5, 2:6] = 3
    paths.proposal_label_maps.mkdir(parents=True)
    paths.proposal_overlays.mkdir(parents=True)
    np.save(paths.proposal_label_maps / "000000.npy", labels)
    Image.fromarray(labels).save(paths.proposal_label_maps / "000000.png")
    Image.fromarray(np.zeros((6, 8, 4), dtype=np.uint8)).save(
        paths.proposal_overlays / "000000.png"
    )
    write_jsonl(
        paths.frame_map,
        [
            {
                "frameId": 0,
                "splitSplatImageName": "scan_000_000000.JPEG",
                "width": 8,
                "height": 6,
            }
        ],
    )
    editor_paths = SimpleNamespace(interactive_dir=tmp_path / "interactive")
    proposal_summary = {
        "proposalSource": "official_auto",
        "sourceDisplayName": "Official Four-Grid SAM2",
        "frames": [
            {
                "frameId": 0,
                "proposalCount": 1,
                "coverage": 16 / 48,
                "overlapCoverage": 0.0,
                "exclusivePreviewPolicy": "small_masks_overwrite",
            }
        ],
    }

    register_proposal_layer(paths, editor_paths, proposal_summary)
    backends = _discover_split_splat_backends(
        editor_paths.interactive_dir / "tmp"
    )
    layer_dir = (
        editor_paths.interactive_dir
        / "proposals"
        / "propagation"
        / "split_splat_official_proposals"
    )
    metadata = json.loads(
        (layer_dir / "metadata" / "000000.json").read_text(encoding="utf-8")
    )

    assert len(backends) == 1
    assert backends[0].info.label_space == "frame_local_proposal"
    assert backends[0].info.layer_group == "frame_proposals"
    assert backends[0].info.read_only is True
    assert metadata["labels"][0]["labelId"] == 3
    assert (layer_dir / "label_maps").is_symlink()


class _Registry:
    def status(self):
        return {"defaultMethodId": "unused", "methods": []}

    def get(self, method_id: str):
        raise KeyError(method_id)


def test_segmentation3d_manager_serves_registered_npz_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "split_splat_official_labeled"
    run_dir.mkdir(parents=True)
    cache_path = run_dir / "labeled_points.npz"
    np.savez_compressed(
        cache_path,
        points=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
        colors=np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8),
        labels=np.array([7, 9], dtype=np.int32),
    )
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                "runId": "split_splat_official_labeled",
                "methodId": "split_splat",
                "inputId": "official_split",
                "pointsCachePath": str(cache_path),
            }
        ),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(tmp_path, _Registry(), max_points=10, seed=4)

    payload = manager.result_points("split_splat_official_labeled")

    assert payload["pointCount"] == 2
    assert payload["labelCount"] == 2
    assert payload["positions"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert payload["colors"] == [10, 20, 30, 40, 50, 60]


def test_split_labels_preserve_gaussian_geometry_and_replace_color(
    tmp_path: Path,
) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("f_rest_0", "f4"),
        ("desc_0", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    vertices = np.zeros(3, dtype=dtype)
    vertices["x"] = [0, 1, 2]
    vertices["opacity"] = [0.1, 0.2, 0.3]
    vertices["scale_0"] = [1.1, 1.2, 1.3]
    source = tmp_path / "source.ply"
    appearance = tmp_path / "appearance.ply"
    target = tmp_path / "segments.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(source)

    labels, match = _match_global_gaussian_labels(
        source,
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([7], dtype=np.int32),
    )
    ensure_viewer_gaussian_ply(source, appearance)
    write_segmented_gaussian_ply(source, target, labels)
    partitions = write_viewer_gaussian_partitions(
        source,
        tmp_path / "parts",
        labels,
        include_unassigned=True,
    )
    appearance_result = PlyData.read(appearance)["vertex"].data
    result = PlyData.read(target)["vertex"].data

    assert labels.tolist() == [0, 7, 0]
    assert [row["instanceId"] for row in partitions] == [0, 7]
    assert [row["pointCount"] for row in partitions] == [2, 1]
    assert match["matchedPointCount"] == 1
    assert np.isclose(result["opacity"][1], vertices["opacity"][1])
    assert np.all(result["opacity"][[0, 2]] == -12.0)
    assert np.allclose(result["scale_0"], vertices["scale_0"])
    assert "f_rest_0" in appearance_result.dtype.names
    assert "desc_0" not in appearance_result.dtype.names
    assert "f_rest_0" not in result.dtype.names
    assert "desc_0" not in result.dtype.names
    assert not np.isclose(result["f_dc_0"][1], result["f_dc_0"][0])


def test_segmentation3d_manager_exposes_registered_gaussian_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "split_splat"
    run_dir.mkdir(parents=True)
    ply_path = tmp_path / "experiments" / "scene.ply"
    ply_path.parent.mkdir()
    ply_path.write_bytes(b"ply\n")
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                "runId": "split_splat",
                "methodId": "split_splat",
                "experimentFamily": "split_splat",
                "artifactRole": "split_labels",
                "baseRunId": "split_splat_official",
                "gaussianArtifacts": {
                    "segments": {
                        "displayName": "Segments",
                        "path": str(ply_path),
                        "format": "3dgs_ply",
                        "pointCount": 10,
                        "labelCount": 3,
                        "artifactGroup": "individual_objects",
                        "instanceId": 7,
                    }
                },
                "pointSummary": {"labeledGlobalPointCount": 7},
            }
        ),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(tmp_path, _Registry(), max_points=10, seed=4)

    status = manager.status()

    assert manager.gaussian_artifact("split_splat", "segments") == ply_path
    run = status["runs"][0]
    assert run["experimentFamily"] == "split_splat"
    assert run["artifactRole"] == "split_labels"
    assert run["baseRunId"] == "split_splat_official"
    artifact = run["gaussianArtifacts"][0]
    assert artifact["variantId"] == "segments"
    assert artifact["labeledPointCount"] == 7
    assert artifact["colorSpace"] == "split_splat_instance"
    assert artifact["artifactGroup"] == "individual_objects"
    assert artifact["instanceId"] == 7



def test_segmentation3d_manager_builds_multipart_gaussian_scene(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "multipart"
    run_dir.mkdir(parents=True)
    artifacts = {}
    for variant_id, group, instance_id, radiometric in (
        ("appearance", "composed_scene", 0, True),
        ("appearance_inexact", "composed_scene", 0, True),
        ("rgb_object_000002", "rgb_objects", 2, True),
        ("rgb_object_000001", "rgb_objects", 1, True),
    ):
        path = tmp_path / "gaussians" / f"{variant_id}.ply"
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(f"ply {variant_id}\n".encode("ascii"))
        artifacts[variant_id] = {
            "displayName": variant_id,
            "path": str(path),
            "pointCount": (
                31 if variant_id == "appearance_inexact" else instance_id * 10 or 30
            ),
            "instanceId": instance_id,
            "artifactGroup": group,
            "colorSpace": "rgb_sh",
            "radiometricAppearance": radiometric,
        }
    (run_dir / "experiment.json").write_text(
        json.dumps({"runId": "multipart", "gaussianArtifacts": artifacts}),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(tmp_path, _Registry(), max_points=10, seed=4)

    scene = manager.gaussian_scene("multipart", "appearance")
    single = manager.gaussian_scene("multipart", "rgb_object_000001")
    inexact = manager.gaussian_scene("multipart", "appearance_inexact")
    status_artifact = manager.status()["runs"][0]["gaussianArtifacts"][0]

    assert scene["multipart"] is True
    assert scene["partCount"] == 2
    assert scene["partitionComplete"] is True
    assert scene["unpartitionedPointCount"] == 0
    assert scene["sourceColorMode"] == "rgb"
    assert scene["defaultColorMode"] == "rgb"
    assert [row["modeId"] for row in scene["colorModes"]] == ["rgb", "regions"]
    assert all(part["sourceColorMode"] == "rgb" for part in scene["parts"])
    assert all(len(part["regionColor"]) == 3 for part in scene["parts"])
    assert inexact["defaultColorMode"] == "rgb"
    assert inexact["partitionComplete"] is False
    assert [row["modeId"] for row in inexact["colorModes"]] == ["rgb"]
    assert inexact["parts"][0]["regionColor"] == []
    assert [part["instanceId"] for part in scene["parts"]] == [1, 2]
    assert all(part["contentVersion"] for part in scene["parts"])
    assert single["multipart"] is False
    assert inexact["multipart"] is False
    assert single["parts"][0]["variantId"] == "rgb_object_000001"
    assert status_artifact["contentVersion"]
    assert status_artifact["radiometricAppearance"] is True


def test_segmentation3d_manager_uses_canonical_persistent_region_metadata(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    manager_root = interactive / "3d_segmentation"
    run_dir = manager_root / "runs" / "persistent"
    run_dir.mkdir(parents=True)
    regions_dir = interactive / "regions"
    regions_dir.mkdir()
    (regions_dir / "regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {"id": 1, "name": "Door", "color": "#123456"},
                    {"id": 2, "name": "Arch", "color": "#abcdef"},
                ]
            }
        ),
        encoding="utf-8",
    )
    gaussian_dir = interactive / "gaussians"
    gaussian_dir.mkdir()
    artifacts = {}
    for variant_id, group, instance_id, point_count in (
        ("appearance", "composed_scene", 0, 30),
        ("rgb_region_1", "rgb_objects", 1, 10),
        ("rgb_region_2", "rgb_objects", 2, 20),
    ):
        path = gaussian_dir / f"{variant_id}.ply"
        path.write_bytes(b"ply\n")
        artifacts[variant_id] = {
            "path": str(path),
            "pointCount": point_count,
            "instanceId": instance_id,
            "artifactGroup": group,
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                "runId": "persistent",
                "labelSpace": "persistent_region",
                "gaussianArtifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(
        manager_root,
        _Registry(),
        max_points=10,
        seed=4,
    )

    scene = manager.gaussian_scene("persistent", "appearance")

    assert [part["regionName"] for part in scene["parts"]] == ["Door", "Arch"]
    assert [part["regionColor"] for part in scene["parts"]] == [
        [0x12, 0x34, 0x56],
        [0xAB, 0xCD, 0xEF],
    ]


def test_segmentation3d_manager_exposes_anchored_reconstruction_metadata(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "anchored_splat_no_refine_splat"
    run_dir.mkdir(parents=True)
    ply_path = tmp_path / "reconstruction" / "composed.ply"
    ply_path.parent.mkdir()
    ply_path.write_bytes(b"ply\n")
    (run_dir / "experiment.json").write_text(
        json.dumps(
            {
                "runId": "anchored_splat_no_refine_splat",
                "methodId": "anchored_3dgs",
                "experimentFamily": "split_splat",
                "artifactRole": "reconstruction",
                "baseRunId": "split_splat_anchored_sam2_video",
                "reconstructionVariant": "anchored_3dgs",
                "inputId": "split_splat_anchored_sam2_video",
                "labelSpace": "persistent_region",
                "maskRefinement": "none",
                "canonicalRunDir": str(tmp_path / "reconstruction"),
                "gaussianArtifacts": {
                    "instance_ids": {
                        "path": str(ply_path),
                        "artifactGroup": "composed_scene",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(tmp_path, _Registry(), max_points=10, seed=4)

    run = manager.status()["runs"][0]

    assert run["methodId"] == "anchored_3dgs"
    assert run["experimentFamily"] == "split_splat"
    assert run["artifactRole"] == "reconstruction"
    assert run["baseRunId"] == "split_splat_anchored_sam2_video"
    assert run["reconstructionVariant"] == "anchored_3dgs"
    assert run["inputId"] == "split_splat_anchored_sam2_video"
    assert run["labelSpace"] == "persistent_region"
    assert run["maskRefinement"] == "none"
    assert run["canonicalRunDir"] == str(tmp_path / "reconstruction")


def test_gaussian_viewer_preserves_trained_color_accumulation() -> None:
    settings = viewer_settings()

    assert settings["tonemapping"] == "none"
    assert settings["highPrecisionRendering"] is True
    assert settings["postEffectSettings"]["grading"]["enabled"] is False


def test_gaussian_embed_refreshes_manifest_but_caches_versioned_parts() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "omega_local"
        / "segmentation"
        / "interactive"
        / "static"
        / "js"
        / "57_splat_embed.js"
    ).read_text(encoding="utf-8")
    part_fetch = source.split("async function fetchGaussian", 1)[1].split(
        "function installViewerShell", 1
    )[0]

    assert 'fetchRequired(sceneUrl, "Scene", "json", "no-store")' in source
    assert 'cache: "force-cache"' in part_fetch

def test_gaussian_color_switch_reuses_modifier_and_batches_updates() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "omega_local"
        / "segmentation"
        / "interactive"
        / "static"
        / "js"
        / "57_splat_embed.js"
    ).read_text(encoding="utf-8")
    switch = source.split("async function switchColorMode", 1)[1].split(
        'window.addEventListener("message"', 1
    )[0]

    assert "const REGION_COLOR_MODIFIER" in source
    assert "setPartColorMode" in switch
    assert 'setParameter("omegaRegionMix"' in source
    assert "setWorkBufferModifier" not in switch
    assert "COLOR_SWITCH_POINT_BUDGET" in switch
    assert "requestAnimationFrame" in switch


def test_split_launcher_handles_collinear_prompt_points() -> None:
    from scipy.spatial import ConvexHull

    collinear = np.array(
        [[249.0, 0.0], [250.0, 0.0], [270.0, 0.0], [310.0, 0.0]],
        dtype=np.float64,
    )
    flat_hull = _rank_safe_convex_hull(ConvexHull, collinear)
    regular_hull = _rank_safe_convex_hull(
        ConvexHull,
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.2, 0.2]]),
    )

    assert flat_hull.vertices.tolist() == [0, 1, 2, 3]
    assert set(regular_hull.vertices.tolist()) == {0, 1, 2}


def test_mask_refinement_launcher_applies_paper_controls() -> None:
    source = (
        "if(iter>len(cameras)/2): break\n"
        "if(iou_score_new<0.05): continue\n"
    )

    patched = _paper_mask_refinement_source(source)

    assert "if False: break" in patched
    assert "if(mask is None and iou_score_new<0.95): continue" in patched
    assert "len(cameras)/2" not in patched


def test_upstream_runner_streams_newline_and_carriage_return_progress(
    tmp_path: Path,
) -> None:
    config = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path,
        python=Path(sys.executable),
    ).normalized()
    events: list[str] = []

    result = run_command(
        [
            sys.executable,
            "-c",
            (
                "import sys; print('start'); "
                "sys.stdout.write('step 1/2\\r'); sys.stdout.flush(); "
                "print('step 2/2')"
            ),
        ],
        cwd=tmp_path,
        config=config,
        log_path=tmp_path / "stream.log",
        progress=events.append,
    )

    assert result["returnCode"] == 0
    assert events == ["start", "step 1/2", "step 2/2"]
    assert "step 2/2" in (tmp_path / "stream.log").read_text(encoding="utf-8")


def test_splat_stages_are_per_run_and_follow_split() -> None:
    assert SPLAT_STAGE_ORDER == (
        "splat_prepare",
        "splat_initial",
        "splat_masks",
        "splat_refined",
        "splat_compose",
    )
    assert STAGE_DEPENDENCIES["splat_prepare"] == ("prepare", "split")
    assert STAGE_DEPENDENCIES["splat_compose"] == ("splat_refined",)


def test_splat_workspace_exposes_released_relative_paths(tmp_path: Path) -> None:
    split_splat_root = tmp_path / "Split_and_Splat"
    checkpoint = split_splat_root / "checkpoints" / "sam2.1_hiera_large.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    config = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=split_splat_root,
        python=Path(sys.executable),
        run_id="sam2_video",
    ).normalized()
    paths = SplitSplatRunPaths(
        tmp_path / "experiments",
        config.run_id,
        "shared",
        config.iterations,
    )

    _prepare_mask_refinement_workspace(config, paths)

    assert (
        paths.splat_workspace / "checkpoints" / "sam2.1_hiera_large.pt"
    ).resolve() == checkpoint.resolve()
    assert (
        paths.splat_workspace / "output" / config.run_id / "raw"
    ).resolve() == paths.splat_initial_models.resolve()
    assert paths.splat_scene_dir.is_dir()


def test_released_splat_commands_keep_paper_iteration_settings(
    tmp_path: Path,
) -> None:
    config = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=tmp_path / "python",
    ).normalized()

    initial = instance_train_command(
        config,
        dataset_dir=tmp_path / "instance",
        model_dir=tmp_path / "raw",
        initial_pass=True,
    )
    composition = composition_train_command(
        config,
        dataset_dir=tmp_path / "pair",
        model_dir=tmp_path / "combined",
        mask_weight=0.15,
    )

    assert initial[initial.index("--iterations") + 1] == "1000"
    assert "--init_rec" in initial
    assert "--is_instance" in initial
    assert "--save_iterations" in initial
    assert composition[composition.index("--mask-loss-weight") + 1] == "0.15"
    assert "--paper-reset-opacity" in composition
    assert "--composition" in composition
    assert "--is_instance" in composition
    assert composition[composition.index("--iterations") + 1] == "1000"
    assert composition[composition.index("--densify_from_iter") + 1] == "999999"
    assert "--foreground-balanced" not in composition

    anchored = composition_train_command(
        config,
        dataset_dir=tmp_path / "anchored_pair",
        model_dir=tmp_path / "anchored_combined",
        mask_weight=0.15,
        foreground_balanced=True,
    )
    assert "--foreground-balanced" in anchored


def test_composition_launcher_changes_only_explicit_paper_controls() -> None:
    source = (
        "gaussians.training_setup(opt)\n"
        "loss = rgb + Ll1_mask * 0.25\n"
    )

    patched = _patched_source(
        source,
        mask_loss_weight=0.15,
        paper_reset_opacity=True,
    )

    assert "Ll1_mask * 0.14999999999999999" in patched
    assert "if dataset.composition:" in patched
    assert "_reset_opacity_before_training(gaussians)" in patched
    assert patched.index("_reset_opacity_before_training") < patched.index(
        "gaussians.training_setup(opt)"
    )


def test_anchored_composition_launcher_is_foreground_balanced() -> None:
    source = (
        "def training():\n"
        "    scene = Scene(dataset, gaussians)\n"
        "    gaussians.training_setup(opt)\n"
        "        Ll1_mask = l1_loss(mask, GT_mask)\n"
        "        Ll1 = l1_loss(image, gt_image)\n"
        "        loss = Ll1 + Ll1_mask * 0.25\n"
    )

    patched = _patched_source(
        source,
        mask_loss_weight=0.15,
        paper_reset_opacity=True,
        foreground_balanced=True,
    )

    assert "_sanitize_identity_masks(scene)" in patched
    assert "_balanced_mask_l1(mask, GT_mask)" in patched
    assert "_foreground_l1(image, gt_image, GT_mask)" in patched
    assert "Ll1_mask * 0.14999999999999999" in patched


def test_composition_opacity_reset_precedes_optimizer_state() -> None:
    import torch

    class Gaussians:
        def __init__(self) -> None:
            self._opacity = torch.nn.Parameter(torch.tensor([[2.0], [-1.0]]))

        @property
        def get_opacity(self):
            return torch.sigmoid(self._opacity)

        @staticmethod
        def inverse_opacity_activation(value):
            return torch.log(value / (1.0 - value))

    gaussians = Gaussians()
    original = gaussians._opacity

    _reset_opacity_before_training(gaussians)
    optimizer = torch.optim.Adam([gaussians._opacity], lr=1e-3)

    assert gaussians._opacity is not original
    assert gaussians._opacity.is_leaf
    assert torch.all(gaussians.get_opacity <= 0.0100001)
    assert optimizer.state.get(gaussians._opacity) is None


def test_composition_storage_keys_bound_large_instance_groups() -> None:
    short = (1, 7, 18)
    large = tuple(range(1, 150))

    assert _composition_storage_key(short) == "1_7_18"
    first = _composition_storage_key(large)
    second = _composition_storage_key(large)
    changed = _composition_storage_key((*large[:-1], 151))

    assert first == second
    assert first != changed
    assert len(first) < 64


def test_composition_model_paths_use_only_bounded_storage_keys(
    tmp_path: Path,
) -> None:
    instance_ids = tuple(range(1, 180))
    storage_key = _composition_storage_key(instance_ids)

    model_dir, output_ply = _composition_model_paths(
        tmp_path,
        storage_key=storage_key,
        iterations=1_000,
    )

    assert len(storage_key.encode()) <= 96
    assert model_dir == tmp_path / "models" / storage_key
    assert output_ply == (
        model_dir / "point_cloud" / "iteration_1000" / "point_cloud.ply"
    )


def test_composition_cache_rejects_stale_gaussian_inputs(tmp_path: Path) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    def write_ply(path: Path, point_count: int) -> None:
        vertex = np.zeros(point_count, dtype=dtype)
        vertex["x"] = np.arange(point_count)
        path.parent.mkdir(parents=True, exist_ok=True)
        PlyData([PlyElement.describe(vertex, "vertex")]).write(path)

    def write_mask(directory: Path, value: int) -> None:
        directory.mkdir(parents=True)
        Image.fromarray(np.full((4, 5), value, dtype=np.uint8)).save(
            directory / "frame.png"
        )

    first_ply = tmp_path / "first.ply"
    second_ply = tmp_path / "second.ply"
    first_masks = tmp_path / "first_masks"
    second_masks = tmp_path / "second_masks"
    write_ply(first_ply, 2)
    write_ply(second_ply, 1)
    write_mask(first_masks, 255)
    write_mask(second_masks, 127)
    first = _CompositionNode(
        "1",
        (1,),
        first_ply,
        first_masks,
        np.ones(2, dtype=np.int32),
        _composition_source_key(first_ply, first_masks, instance_ids=(1,)),
    )
    second = _CompositionNode(
        "2",
        (2,),
        second_ply,
        second_masks,
        np.full(1, 2, dtype=np.int32),
        _composition_source_key(second_ply, second_masks, instance_ids=(2,)),
    )
    expected_input = _composition_cache_input(
        first,
        second,
        iterations=1_000,
        mask_weight=0.05,
    )
    output = tmp_path / "model" / "point_cloud.ply"
    manifest = tmp_path / "model" / "composition_cache.json"
    write_ply(output, 3)

    valid, reason = _composition_cache_status(
        manifest,
        output,
        expected_input,
        expected_point_count=3,
    )
    assert valid is False
    assert reason == "manifest_missing"

    atomic_write_json(
        manifest,
        {
            "schemaVersion": 1,
            "input": expected_input,
            "expectedPointCount": 3,
            "output": _composition_file_identity(output),
        },
    )
    assert _composition_cache_status(
        manifest,
        output,
        expected_input,
        expected_point_count=3,
    ) == (True, "valid")

    write_ply(output, 2)
    assert _composition_cache_status(
        manifest,
        output,
        expected_input,
        expected_point_count=3,
    ) == (False, "output_count_mismatch")


def test_composition_cleanup_keeps_only_resumable_checkpoint(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    model = tmp_path / "model"
    dataset_points = dataset / "sparse" / "0" / "points3D.ply"
    input_copy = model / "input.ply"
    midpoint = model / "point_cloud" / "iteration_500" / "point_cloud.ply"
    final = model / "point_cloud" / "iteration_1000" / "point_cloud.ply"
    for path in (dataset_points, input_copy, midpoint, final):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")

    _cleanup_composition_training_artifacts(
        dataset,
        model,
        iterations=1_000,
    )

    assert not dataset_points.exists()
    assert not input_copy.exists()
    assert not midpoint.parent.exists()
    assert final.is_file()


def test_completed_model_cleanup_keeps_final_models_and_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "models"
    model = root / "7"
    final = model / "point_cloud" / "iteration_1000" / "point_cloud.ply"
    midpoint = model / "point_cloud" / "iteration_500" / "point_cloud.ply"
    metadata = model / "cameras.json"
    for path in (
        final,
        midpoint,
        metadata,
        model / "input.ply",
        model / "events.out.tfevents.test",
        model / "chkpnt700.pth",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"generated")

    report = cleanup_completed_models(root, final_iteration=1_000)

    assert final.is_file()
    assert metadata.is_file()
    assert not midpoint.parent.exists()
    assert not (model / "input.ply").exists()
    assert not (model / "events.out.tfevents.test").exists()
    assert not (model / "chkpnt700.pth").exists()
    assert report["removedFiles"] == 4
    assert report["retainedIteration"] == 1_000


def test_completed_composition_cleanup_requires_final_outputs(
    tmp_path: Path,
) -> None:
    composition = tmp_path / "05_composition"
    outputs_dir = tmp_path / "06_outputs"
    (composition / "round_00").mkdir(parents=True)
    (composition / "round_00" / "cache.ply").write_bytes(b"cache")
    outputs_dir.mkdir()
    paths = SimpleNamespace(
        splat_composition=composition,
        splat_outputs=outputs_dir,
    )
    output_paths = {
        "fullGaussians": outputs_dir / "composed_scene.ply",
        "instanceIdGaussians": outputs_dir / "composed_scene_instance_ids.ply",
        "instanceLabels": outputs_dir / "composed_scene_instance_labels.npy",
        "pointsCache": outputs_dir / "composed_scene_points.npz",
    }
    summary = {
        "status": "complete",
        "outputs": {key: str(path) for key, path in output_paths.items()},
    }

    skipped = cleanup_completed_composition(paths, summary)
    assert composition.is_dir()
    assert skipped["skipped"] == "final_outputs_incomplete"

    for path in output_paths.values():
        path.write_bytes(b"final")
    (outputs_dir / "individual_objects").mkdir()
    (outputs_dir / "split_rgb_objects").mkdir()
    removed = cleanup_completed_composition(paths, summary)

    assert not composition.exists()
    assert all(path.is_file() for path in output_paths.values())
    assert removed["categories"]["completedCompositionWorkspace"][
        "removedFiles"
    ] == 1


def test_manager_cleans_completed_and_cached_training_stages(
    tmp_path: Path,
) -> None:
    paths = SplitSplatRunPaths(
        tmp_path / "experiments",
        "run",
        "shared",
        30_000,
    )
    manager = object.__new__(SplitSplatExperimentManager)
    manager.paths = paths
    manager.config = SimpleNamespace(instance_iterations=1_000)
    manager._initialize_run = lambda: None
    manager._require_prerequisites = lambda stage: None
    manager._write_progress = lambda stage, status, message: None
    manager._announce = lambda message: None

    model = paths.splat_initial_models / "1"
    final = model / "point_cloud" / "iteration_1000" / "point_cloud.ply"

    def train(**_kwargs) -> dict[str, object]:
        for path in (
            final,
            model / "point_cloud" / "iteration_500" / "point_cloud.ply",
            model / "input.ply",
            model / "events.out.tfevents.first",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"generated")
        return {"stage": "splat_initial"}

    manager._run_splat_initial = train
    first = manager.run_stage("splat_initial")

    assert first["status"] == "complete"
    assert first["storageCleanup"]["removedFiles"] == 3
    assert final.is_file()
    assert not (model / "input.ply").exists()

    cached_input = model / "input.ply"
    cached_event = model / "events.out.tfevents.cached"
    cached_input.write_bytes(b"stale")
    cached_event.write_bytes(b"stale")
    second = manager.run_stage("splat_initial")

    assert second["cacheHit"] is True
    assert not cached_input.exists()
    assert not cached_event.exists()
    recorded = read_complete_stage(paths.stage_summary("splat_initial"))
    assert recorded is not None
    assert recorded["storageCleanup"]["removedFiles"] == 2


def test_composition_uses_source_collision_pairing_and_mask_order(
    tmp_path: Path,
) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
    ]

    def write_ply(name: str, xyz: list[tuple[float, float, float]]) -> Path:
        vertex = np.zeros(len(xyz), dtype=dtype)
        vertex["x"] = [point[0] for point in xyz]
        vertex["y"] = [point[1] for point in xyz]
        vertex["z"] = [point[2] for point in xyz]
        path = tmp_path / f"{name}.ply"
        PlyData([PlyElement.describe(vertex, "vertex")]).write(path)
        return path

    first_masks = tmp_path / "first"
    second_masks = tmp_path / "second"
    output_masks = tmp_path / "combined"
    first_masks.mkdir()
    second_masks.mkdir()
    output_masks.mkdir()
    large = np.zeros((8, 8, 4), dtype=np.uint8)
    large[1:7, 1:7] = [255, 0, 0, 255]
    small = np.zeros((8, 8, 4), dtype=np.uint8)
    small[3:5, 3:5] = [0, 255, 0, 255]
    Image.fromarray(large, mode="RGBA").save(first_masks / "frame.png")
    Image.fromarray(small, mode="RGBA").save(second_masks / "frame.png")

    first = _CompositionNode(
        "1",
        (1,),
        write_ply("first", [(0, 0, 0), (1, 1, 1)]),
        first_masks,
        np.ones(2, dtype=np.int32),
    )
    second = _CompositionNode(
        "2",
        (2,),
        write_ply("second", [(0.5, 0.5, 0.5)]),
        second_masks,
        np.full(1, 2, dtype=np.int32),
    )

    pairs = _select_collision_pairs([first, second])
    _compose_mask_directories(first_masks, second_masks, output_masks)
    combined = np.asarray(Image.open(output_masks / "frame.png").convert("RGBA"))

    assert len(pairs) == 1
    assert pairs[0][2] == 1.0
    assert combined[3, 3].tolist() == [0, 255, 0, 255]


def test_directional_collision_matches_paper_equation(tmp_path: Path) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]

    def write_ply(name: str, xyz: list[tuple[float, float, float]]) -> Path:
        vertex = np.zeros(len(xyz), dtype=dtype)
        vertex["x"] = [point[0] for point in xyz]
        vertex["y"] = [point[1] for point in xyz]
        vertex["z"] = [point[2] for point in xyz]
        path = tmp_path / f"{name}.ply"
        PlyData([PlyElement.describe(vertex, "vertex")]).write(path)
        return path

    source = write_ply("source", [(0.5, 0.5, 0.5), (2.0, 2.0, 2.0)])
    container = write_ply("container", [(0, 0, 0), (1, 1, 1)])

    assert _directional_collision_score(source, container) == 0.5


def test_viewer_gaussian_subset_preserves_only_render_fields(
    tmp_path: Path,
) -> None:
    from plyfile import PlyData, PlyElement

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("f_rest_0", "f4"),
        ("desc_0", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    vertices = np.zeros(3, dtype=dtype)
    vertices["x"] = [1, 2, 3]
    source = tmp_path / "scene.ply"
    output = tmp_path / "object.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(source)

    count = write_viewer_gaussian_subset(
        source,
        output,
        np.array([False, True, False]),
    )
    result = PlyData.read(output)["vertex"].data

    assert count == 1
    assert result["x"].tolist() == [2]
    assert "f_rest_0" in result.dtype.names
    assert "desc_0" not in result.dtype.names


def test_splat_prepare_reuses_shared_support_and_isolates_instance_masks(
    tmp_path: Path,
) -> None:
    from plyfile import PlyData, PlyElement

    paths = SplitSplatRunPaths(tmp_path, "official", "shared", 30_000)
    paths.consistent_masks_dir.mkdir(parents=True)
    paths.split_raw_output.mkdir(parents=True)
    paths.dataset_dir.mkdir(parents=True)
    image_dir = paths.dataset_dir / "images"
    sparse_dir = paths.dataset_dir / "sparse" / "0"
    depth_dir = paths.global_gs_dir / "split_depth_editor_depth"
    image_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    depth_dir.mkdir(parents=True)

    frame_rows = []
    image_lines = ["# images"]
    for frame_id in range(2):
        stem = f"frame_{frame_id:06d}"
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(
            image_dir / f"{stem}.JPEG"
        )
        np.save(depth_dir / f"{stem}_pred.npy", np.full((6, 8), 2.0, dtype=np.float32))
        image_lines.extend(
            [
                f"{frame_id + 1} 1 0 0 0 0 0 0 {frame_id + 1} {stem}.JPEG",
                "",
            ]
        )
        frame_rows.append(
            {
                "frameId": frame_id,
                "splitSplatImageName": f"{stem}.JPEG",
                "width": 8,
                "height": 6,
                "fx": 10.0,
                "fy": 10.0,
                "cx": 4.0,
                "cy": 3.0,
            }
        )
    (sparse_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n",
        encoding="utf-8",
    )
    (sparse_dir / "cameras.txt").write_text(
        "\n".join(
            [
                "# cameras",
                "1 PINHOLE 8 6 10 10 4 3",
                "2 PINHOLE 8 6 10 10 4 3",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_jsonl(paths.frame_map, frame_rows)
    (paths.stage_summary("split")).write_text(
        json.dumps(
            {
                "instanceIds": [7],
                "trainingViewWeightsPath": str(paths.training_view_weights),
            }
        ),
        encoding="utf-8",
    )
    paths.training_view_weights.write_text(
        json.dumps({"frames": [{"frameId": 0, "weight": 8}]}),
        encoding="utf-8",
    )

    instance_dir = paths.split_raw_output / "7"
    instance_dir.mkdir()
    for frame_id in range(2):
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[1:4, 2:6] = 255
        Image.fromarray(mask).save(instance_dir / f"frame_{frame_id:06d}.png")
    points = np.zeros(3, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    points["x"] = [0.0, 0.5, 1.0]
    PlyData([PlyElement.describe(points, "vertex")]).write(
        instance_dir / "label_7.ply"
    )
    global_dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
    ]
    global_points = np.zeros(3, dtype=global_dtype)
    global_points["x"] = points["x"]
    global_points["f_dc_0"] = [0.1, 0.2, 0.3]
    global_points["f_dc_1"] = [0.2, 0.3, 0.4]
    global_points["f_dc_2"] = [0.3, 0.4, 0.5]
    paths.global_point_cloud.parent.mkdir(parents=True)
    PlyData([PlyElement.describe(global_points, "vertex")]).write(
        paths.global_point_cloud
    )
    config = SplitSplatRunConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        split_splat_root=tmp_path / "Split_and_Splat",
        python=Path(sys.executable),
        run_id="official",
        depth_source="editor_depth",
    ).normalized()

    summary = prepare_splat(config, paths)

    dataset = paths.splat_instances_dir / "7"
    assert summary["instanceIds"] == [7]
    assert (dataset / "images").is_symlink()
    assert (dataset / "training_view_weights.json").is_symlink()
    assert (dataset / "masks" / "frame_000000.png").is_file()
    initialization = dataset / "sparse" / "0" / "points3D.ply"
    assert initialization.is_file()
    initialized = PlyData.read(initialization)["vertex"].data
    assert {"red", "green", "blue"}.issubset(initialized.dtype.names)
    assert np.count_nonzero(initialized["red"]) == 3
    assert (
        paths.shared_splat_support_dir
        / "depth_metric_mm"
        / "frame_000000.png"
    ).is_file()
    assert (paths.splat_scene_dir / "masks").is_symlink()
    assert summary["settings"]["trainingViewWeights"] == {
        "available": True,
        "source": str(paths.training_view_weights.resolve()),
        "appliedByReleasedTrainer": False,
    }


def test_split_splat_frontend_modules_have_single_ownership_and_order() -> None:
    static_dir = (
        Path(__file__).parents[1]
        / "omega_local"
        / "segmentation"
        / "interactive"
        / "static"
    )
    controller = (static_dir / "js" / "54_segmentation3d.js").read_text()
    split_splat = (static_dir / "js" / "59_split_splat_results.js").read_text()
    mapanything = (static_dir / "js" / "60_mapanything_results.js").read_text()
    index = (static_dir / "index.html").read_text()

    assert "function splitSplatExperimentGroups()" not in controller
    assert split_splat.count("function splitSplatExperimentGroups()") == 1
    assert "function mapAnythingPipelineStatusControl(" not in controller
    assert mapanything.count("function mapAnythingPipelineStatusControl(") == 1
    assert controller.count("let mapAnythingPipelinePollTimer") == 0
    assert mapanything.count("let mapAnythingPipelinePollTimer") == 1

    script_names = (
        "54_segmentation3d.js",
        "59_split_splat_results.js",
        "60_mapanything_results.js",
        "56_gaussian_viewer.js",
    )
    positions = [index.index(name) for name in script_names]
    assert positions == sorted(positions)
