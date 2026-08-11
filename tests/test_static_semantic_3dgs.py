from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from plyfile import PlyData

from omega_local.reconstruction.backends.static_semantic_3dgs.contract import (
    StaticSemantic3DGSConfig,
    StaticSemantic3DGSPaths,
)
from omega_local.reconstruction.backends.static_semantic_3dgs.export import (
    _gaussian_grouping_region_ids,
    _segment_then_splat_region_ids,
    _standard_gaussian_vertices,
)
from omega_local.reconstruction.backends.static_semantic_3dgs.prepare import (
    _remap_labels,
    _write_segment_then_splat_ply,
)
from omega_local.reconstruction.backends.static_semantic_3dgs.runner import (
    _gaussian_grouping_command,
    _segment_then_splat_command,
)


def _config(tmp_path: Path, method: str, **changes) -> StaticSemantic3DGSConfig:
    values = {
        "model_dir": tmp_path / "model",
        "editor_baseline_name": "editor",
        "method": method,
        "method_root": tmp_path / method,
        "python": Path(sys.executable),
        "native_extensions_dir": tmp_path / "extensions",
    }
    values.update(changes)
    return StaticSemantic3DGSConfig(**values).normalized()


def _paths(tmp_path: Path, method: str) -> StaticSemantic3DGSPaths:
    source = tmp_path / "mapanything"
    family = source / "04_static_semantic_3dgs"
    return StaticSemantic3DGSPaths(
        source_run_dir=source,
        family_dir=family,
        run_dir=family / "runs" / method / "joint",
    )


def test_method_defaults_match_released_training_schedules(tmp_path: Path) -> None:
    hard = _config(tmp_path, "segment_then_splat")
    soft = _config(tmp_path, "gaussian_grouping")

    assert (hard.iterations, hard.densify_until_iter) == (40_000, 20_000)
    assert (soft.iterations, soft.densify_until_iter) == (30_000, 10_000)


def test_gaussian_grouping_rejects_unsafe_rgb_only_downsampling(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="does not resize indexed object masks"):
        _config(tmp_path, "gaussian_grouping", resolution=2)


def test_methods_share_inputs_but_have_parallel_run_folders(tmp_path: Path) -> None:
    hard = _paths(tmp_path, "segment_then_splat")
    soft = _paths(tmp_path, "gaussian_grouping")

    assert hard.shared_dir == soft.shared_dir
    assert hard.run_dir != soft.run_dir
    assert hard.run_dir.parent.name == "segment_then_splat"
    assert soft.run_dir.parent.name == "gaussian_grouping"


def test_common_masks_reserve_zero_and_remap_gapped_region_ids() -> None:
    source = np.asarray([[0, 7, 31], [31, 7, 0]], dtype=np.int32)
    output = _remap_labels(source, {7: 1, 31: 2})

    assert output.dtype == np.uint8
    assert output.tolist() == [[0, 1, 2], [2, 1, 0]]


def test_segment_then_splat_initializer_uses_hard_ids_and_unknown_255(
    tmp_path: Path,
) -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    colors = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    path = tmp_path / "points3D.ply"

    _write_segment_then_splat_ply(
        path,
        points,
        colors,
        np.asarray([0, 255], dtype=np.uint8),
    )
    vertices = PlyData.read(path)["vertex"]

    assert np.asarray(vertices["obj_id_default"]).tolist() == [0, 255]
    assert np.asarray(vertices["obj_id_middle"]).tolist() == [255, 255]
    assert np.asarray(vertices["obj_id_small"]).tolist() == [255, 255]


def test_hard_identity_export_restores_persistent_region_ids(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "ids.pth"
    torch.save(torch.tensor([0, 1, 255, 1]), identity)
    mapping = {
        "regions": [
            {"persistentRegionId": 7, "hardObjectId": 0},
            {"persistentRegionId": 31, "hardObjectId": 1},
        ]
    }

    labels = _segment_then_splat_region_ids(identity, mapping)

    assert labels.tolist() == [7, 31, 0, 31]


def test_soft_identity_export_bakes_classifier_argmax_to_persistent_ids(
    tmp_path: Path,
) -> None:
    vertices = np.empty(
        3,
        dtype=[("x", "f4"), ("obj_dc_0", "f4"), ("obj_dc_1", "f4")],
    )
    vertices["x"] = [0, 1, 2]
    vertices["obj_dc_0"] = [2, 0, -2]
    vertices["obj_dc_1"] = [0, 2, -2]
    classifier = tmp_path / "classifier.pth"
    torch.save(
        {
            "weight": torch.tensor(
                [
                    [[[0.0]], [[0.0]]],
                    [[[1.0]], [[0.0]]],
                    [[[0.0]], [[1.0]]],
                ]
            ),
            "bias": torch.tensor([0.1, 0.0, 0.0]),
        },
        classifier,
    )
    mapping = {
        "regions": [
            {"persistentRegionId": 7, "classId": 1},
            {"persistentRegionId": 31, "classId": 2},
        ]
    }

    labels = _gaussian_grouping_region_ids(vertices, classifier, mapping)

    assert labels.tolist() == [7, 31, 0]
    standard = _standard_gaussian_vertices(vertices)
    assert standard.dtype.names == ("x",)


def test_native_commands_use_same_dataset_and_method_specific_losses(
    tmp_path: Path,
) -> None:
    hard_config = _config(tmp_path, "segment_then_splat")
    soft_config = _config(tmp_path, "gaussian_grouping")
    hard_paths = _paths(tmp_path, "segment_then_splat")
    soft_paths = _paths(tmp_path, "gaussian_grouping")
    soft_paths.shared_region_map.parent.mkdir(parents=True)
    soft_paths.shared_region_map.write_text(
        '{"regions":[{"persistentRegionId":7,"classId":1}]}',
        encoding="utf-8",
    )

    hard_command = _segment_then_splat_command(hard_config, hard_paths)
    soft_command = _gaussian_grouping_command(soft_config, soft_paths)

    assert str(hard_paths.dataset_dir) in hard_command
    assert hard_command[hard_command.index("--stage2_iters") + 1] == "0"
    assert "--partial_mask_iou" in hard_command
    assert str(soft_paths.dataset_dir) in soft_command
    assert "--config_file" in soft_command
    assert soft_command[soft_command.index("--num_classes") + 1] == "2"
