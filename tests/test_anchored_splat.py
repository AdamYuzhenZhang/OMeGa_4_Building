from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from omega_local.reconstruction.backends.anchored_3dgs.contract import (
    AnchoredSplatConfig,
    AnchoredSplatPaths,
)
from omega_local.reconstruction.backends.anchored_3dgs.manager import (
    AnchoredSplatManager,
)
from omega_local.reconstruction.backends.anchored_3dgs.prepare import (
    _write_full_gaussians,
)
from omega_local.reconstruction.backends.anchored_3dgs.refine import (
    refine_masks,
)
from omega_local.reconstruction.backends.anchored_3dgs.train import (
    anchored_train_command,
)
from omega_local.reconstruction.backends.anchored_3dgs.train_launch import (
    _load_view_weights,
    _patched_source,
)
from omega_local.segmentation.split_splat.contract import SplitSplatRunPaths


def _config(tmp_path: Path, **changes) -> AnchoredSplatConfig:
    values = {
        "model_dir": tmp_path / "model",
        "editor_baseline_name": "editor",
        "split_splat_root": tmp_path / "Split_and_Splat",
        "python": tmp_path / "python",
    }
    values.update(changes)
    return AnchoredSplatConfig(**values).normalized()


def _paths(tmp_path: Path) -> AnchoredSplatPaths:
    source = SplitSplatRunPaths(
        tmp_path / "experiments" / "split_splat",
        "source",
        "shared",
        30_000,
    )
    return AnchoredSplatPaths(
        run_dir=tmp_path / "reconstruction" / "anchored",
        source=source,
        run_id="anchored",
    )


def test_no_refinement_is_the_default_and_does_not_execute_sam2(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    result = refine_masks(config, object(), _paths(tmp_path))

    assert config.mask_refinement == "none"
    assert result["policy"] == "none"
    assert result["maskMutation"] is False
    assert result["sam2Executed"] is False


def test_refinement_policy_is_part_of_run_identity(tmp_path: Path) -> None:
    no_refine = _config(tmp_path)
    paper = _config(
        tmp_path,
        run_id="anchored_splat_paper_sam2",
        mask_refinement="paper_sam2",
    )

    assert no_refine.run_id != paper.run_id
    assert no_refine.mask_refinement == "none"
    assert paper.mask_refinement == "paper_sam2"


def test_runtime_patch_loads_gaussians_and_balances_foreground_loss() -> None:
    source = (
        "def training(dataset, opt):\n"
        "    scene = Scene(dataset, gaussians)\n"
        "    gaussians.training_setup(opt)\n"
        "        Ll1_mask = l1_loss(mask, GT_mask)\n"
        "        Ll1 = l1_loss(image, gt_image)\n"
        "        loss = Ll1 + Ll1_mask * 0.25\n"
    )
    patched = _patched_source(source)

    assert "_load_anchored_gaussians(gaussians, scene, dataset)" in patched
    assert "_view_weight(viewpoint_cam.image_name)" in patched
    assert "_sanitize_identity_masks(scene)" in patched
    assert "_balanced_mask_l1(mask, GT_mask)" in patched
    assert "_foreground_l1(image, gt_image, GT_mask)" in patched
    assert "Ll1_mask * 1" in patched


def test_view_weights_are_normalized_over_region_training_views(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps(
            {
                "frames": [
                    {"imageStem": "a", "weight": 8},
                    {"imageStem": "b", "weight": 1},
                    {"imageStem": "c", "weight": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "a.png").touch()
    (masks / "b.png").touch()

    normalized = _load_view_weights(weights, mask_dir=masks)

    assert np.isclose((normalized["a"] + normalized["b"]) / 2.0, 1.0)
    assert np.isclose(normalized["a"] / normalized["b"], 8.0)


def test_full_gaussian_subset_preserves_every_vertex_attribute(
    tmp_path: Path,
) -> None:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("id_0", "f4"),
            ("desc_0", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("rot_0", "f4"),
        ]
    )
    vertices = np.zeros(3, dtype=dtype)
    vertices["x"] = [1, 2, 3]
    vertices["desc_0"] = [4, 5, 6]
    source = PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
        byte_order="<",
    )
    output = tmp_path / "subset.ply"

    _write_full_gaussians(source, vertices[[0, 2]], output)
    loaded = PlyData.read(output)["vertex"].data

    assert loaded.dtype.names == vertices.dtype.names
    assert loaded.shape == (2,)
    assert loaded["x"].tolist() == [1, 3]
    assert loaded["desc_0"].tolist() == [4, 6]


def test_train_command_uses_private_launcher_and_anchor_sidecars(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    command = anchored_train_command(
        config,
        dataset_dir=tmp_path / "dataset",
        model_dir=tmp_path / "model_out",
        initializer_gaussians=tmp_path / "full.ply",
        training_view_weights=tmp_path / "weights.json",
        initial_pass=True,
    )

    assert "omega_local.reconstruction.backends.anchored_3dgs.train_launch" in command
    assert "--initializer-gaussians" in command
    assert "--training-view-weights" in command
    assert "--training-mask-dir" in command
    assert "--init_rec" in command
    assert command[command.index("--densify_from_iter") + 1] == "1000"
    assert command[command.index("--lambda_dssim") + 1] == "0"
    assert command[command.index("--mask-loss-weight") + 1] == "1"


def test_anchored_paths_are_outside_released_experiment_runs(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    assert "reconstruction" in paths.run_dir.parts
    assert paths.run_dir != paths.source.run_dir
    assert not paths.run_dir.is_relative_to(paths.source.experiment_root)


def test_compose_overwrite_clears_models_and_final_outputs(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    (paths.splat_composition / "round_00").mkdir(parents=True)
    paths.splat_outputs.mkdir(parents=True)
    (paths.splat_outputs / "stage.json").write_text("{}", encoding="utf-8")
    paths.splat_refined_models.mkdir(parents=True)

    manager = object.__new__(AnchoredSplatManager)
    manager.paths = paths
    manager._clear_stage("compose")

    assert not paths.splat_composition.exists()
    assert not paths.splat_outputs.exists()
    assert paths.splat_refined_models.is_dir()


def test_anchored_manager_cleans_completed_training_stage(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    manager = object.__new__(AnchoredSplatManager)
    manager.paths = paths
    manager.config = type(
        "Config",
        (),
        {"run_id": "anchored", "instance_iterations": 1_000},
    )()
    manager._initialize_run = lambda: None
    manager._require_prerequisites = lambda stage: None
    manager._announce = lambda message: None

    model = paths.splat_initial_models / "1"
    final = model / "point_cloud" / "iteration_1000" / "point_cloud.ply"

    def train(*, dry_run: bool) -> dict[str, object]:
        assert dry_run is False
        for path in (
            final,
            model / "point_cloud" / "iteration_500" / "point_cloud.ply",
            model / "input.ply",
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"generated")
        return {"stage": "initial"}

    manager._run_initial = train
    result = manager.run_stage("initial")

    assert result["status"] == "complete"
    assert result["storageCleanup"]["removedFiles"] == 2
    assert final.is_file()
    assert not (model / "input.ply").exists()
