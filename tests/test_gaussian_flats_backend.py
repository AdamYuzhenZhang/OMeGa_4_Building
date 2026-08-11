from __future__ import annotations

import sys
from pathlib import Path

from omega_local.reconstruction.backends.gaussian_flats.contract import (
    GaussianFlatsConfig,
    GaussianFlatsPaths,
)
from omega_local.reconstruction.backends.gaussian_flats.prepare import (
    _colmap_image_names,
)
from omega_local.reconstruction.backends.gaussian_flats.runner import (
    mesh_commands,
    train_command,
)


def _config(tmp_path: Path) -> GaussianFlatsConfig:
    return GaussianFlatsConfig(
        model_dir=tmp_path / "model",
        editor_baseline_name="editor",
        gaussian_flats_root=tmp_path / "3dgs-flats",
        python=Path(sys.executable),
    ).normalized()


def test_paper_settings_and_door_are_adapter_defaults(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert config.planar_region == "Door"
    assert config.iterations == 30_000
    assert config.plane_fit_iter == 3_500
    assert config.plane_fit_min_points == 100
    assert config.plane_sigma_res == 0.01
    assert config.plane_sigma_dist == 0.3
    assert config.mesh_voxel_size == 0.02
    assert config.planar_grid_resolution == 0.02


def test_mesh_command_uses_configured_tsdf_voxel_size(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = GaussianFlatsPaths(tmp_path / "run")

    command = mesh_commands(config, paths)[0]

    assert command[command.index("--mesh_voxel_size") + 1] == "0.02"


def test_train_command_uses_released_entrypoint_and_hybrid_losses(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = GaussianFlatsPaths(tmp_path / "run")

    command = train_command(config, paths)

    assert command[2] == str(config.gaussian_flats_root / "train_planar.py")
    assert command[command.index("--mask_root") + 1] == str(paths.plane_masks_dir)
    assert command[command.index("--init_type") + 1] == "sfm"
    assert command[command.index("--planar_mask_loss_weight") + 1] == "0.1"
    assert command[command.index("--depthtv_loss_weight") + 1] == "0.1"


def test_train_command_ignores_partial_checkpoints(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = GaussianFlatsPaths(tmp_path / "run")
    paths.model_dir.mkdir(parents=True)
    (paths.model_dir / "chkpnt7000.pth").touch()

    command = train_command(config, paths)

    assert "--start_checkpoint" not in command
    assert "--checkpoint_iterations" not in command
    assert "7000" not in command


def test_colmap_names_ignore_numeric_points2d_rows(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text(
        "# header\n"
        "2 1 0 0 0 0 0 0 1 scan_000_000001.jpg\n"
        "1.0 2.0 9 3.0 4.0 -1\n"
        "1 1 0 0 0 0 0 0 1 scan_000_000000.jpg\n\n",
        encoding="utf-8",
    )

    assert _colmap_image_names(path) == [
        "scan_000_000000.jpg",
        "scan_000_000001.jpg",
    ]
