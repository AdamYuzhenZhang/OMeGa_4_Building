from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from omega_local.reconstruction.gaussian_pruning import (
    prune_gaussian_floaters,
)
from omega_local.reconstruction.masked_training import (
    balanced_mask_l1,
    foreground_l1,
)

from omega_local.reconstruction.pipelines.mapanything_3dgs.contract import (
    MapAnything3DGSConfig,
    MapAnything3DGSPaths,
)
from omega_local.reconstruction.pipelines.mapanything_3dgs.manager import (
    MapAnything3DGSManager,
)
from omega_local.reconstruction.pipelines.mapanything_3dgs.split import (
    _Frame,
    _vote_points,
)
from omega_local.reconstruction.pipelines.mapanything_3dgs.splat import (
    prepare_region_datasets,
    region_train_command,
)
from omega_local.reconstruction.pipelines.mapanything_3dgs.train_launch import (
    _load_view_weights,
    _patched_source,
)
from omega_local.segmentation.interactive.reconstruction_runs import (
    discover_mapanything_runs,
    live_mapanything_experiment,
)
from omega_local.segmentation.split_splat.contract import write_jsonl


def _config(tmp_path: Path, **changes) -> MapAnything3DGSConfig:
    values = {
        "model_dir": tmp_path / "model",
        "editor_baseline_name": "editor",
        "split_splat_root": tmp_path / "Split_and_Splat",
        "python": Path(sys.executable),
        "point_cloud": tmp_path / "mapanything.ply",
    }
    values.update(changes)
    return MapAnything3DGSConfig(**values).normalized()


def _paths(tmp_path: Path) -> MapAnything3DGSPaths:
    return MapAnything3DGSPaths(
        run_dir=tmp_path / "mapanything_run",
        point_cloud=tmp_path / "mapanything.ply",
    )


def test_mapanything_run_contract_is_separate_from_split_splat(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    assert "mapanything_run" in paths.run_dir.parts
    assert "split_splat" not in paths.run_dir.parts
    assert paths.point_labels.parent == paths.point_labels_dir
    assert paths.composed_scene.parent == paths.splat_outputs


def test_reconstruction_settings_can_overwrite_from_affected_stage(
    tmp_path: Path,
) -> None:
    baseline = (
        tmp_path / "model" / "segmentation" / "baselines" / "editor"
    )
    scan = baseline / "dataset" / "scans" / "scene"
    scan.mkdir(parents=True)
    (baseline / "dataset" / "frame_manifest.jsonl").write_text(
        "", encoding="utf-8"
    )
    (scan / "points.pts").write_text("", encoding="utf-8")
    original = MapAnything3DGSManager(_config(tmp_path))
    original._initialize_run("prepare", overwrite=False)

    quality = MapAnything3DGSManager(
        _config(tmp_path, instance_iterations=30_000)
    )
    quality._initialize_run("splat_prepare", overwrite=True)
    manifest = json.loads(
        quality.paths.run_manifest.read_text(encoding="utf-8")
    )
    assert manifest["instanceIterations"] == 30_000

    invalid = MapAnything3DGSManager(
        _config(
            tmp_path,
            instance_iterations=30_000,
            propagation_method="xmem",
        )
    )
    with pytest.raises(RuntimeError, match="Rerun from \x27split\x27"):
        invalid._initialize_run("splat_prepare", overwrite=True)


def test_train_patch_balances_foreground_rgb_and_alpha() -> None:
    source = (
        "def training():\n"
        "    scene = Scene(dataset, gaussians)\n"
        "        Ll1_mask = l1_loss(mask, GT_mask)\n"
        "        Ll1 = l1_loss(image, gt_image)\n"
        "        loss = Ll1 + Ll1_mask * 0.25\n"
        "            if iteration < opt.densify_until_iter < opt.max_num_splats:\n"
        "                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)\n"
    )
    patched = _patched_source(source)

    assert "_view_weight(viewpoint_cam.image_name)" in patched
    assert "_sanitize_identity_masks(scene)" in patched
    assert "_balanced_mask_l1(mask, GT_mask)" in patched
    assert "_foreground_l1(image, gt_image, GT_mask)" in patched
    assert "Ll1_mask * 1" in patched
    assert "_load_anchored_gaussians" not in patched
    assert "gaussians.get_xyz.shape[0] < opt.max_num_splats" in patched
    assert "_enforce_max_splats(gaussians, opt.max_num_splats)" in patched


def test_balanced_losses_do_not_dilute_small_foregrounds() -> None:
    target_mask = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    rendered_mask = torch.zeros_like(target_mask)
    assert torch.isclose(
        balanced_mask_l1(rendered_mask, target_mask),
        torch.tensor(0.5),
    )

    target_rgb = torch.ones((3, 2, 2))
    rendered_rgb = torch.zeros_like(target_rgb)
    assert torch.isclose(
        foreground_l1(rendered_rgb, target_rgb, target_mask),
        torch.tensor(1.0),
    )


def test_balanced_losses_treat_colored_identity_masks_as_binary_support() -> None:
    target_mask = torch.zeros((3, 2, 2))
    target_mask[:, 0, 0] = torch.tensor([0.2, 0.6, 0.9])
    rendered_mask = torch.zeros_like(target_mask)

    assert torch.isclose(
        balanced_mask_l1(rendered_mask, target_mask),
        torch.tensor((0.2 + 0.6 + 0.9) / 6.0),
    )
    target_rgb = torch.ones((3, 2, 2))
    rendered_rgb = torch.zeros_like(target_rgb)
    assert torch.isclose(
        foreground_l1(rendered_rgb, target_rgb, target_mask),
        torch.tensor(1.0),
    )


def test_mapanything_tied_points_keep_geometry_through_local_completion() -> None:
    points = np.asarray([[0, 0, 1], [1, 0, 1]], dtype=np.float32)
    first = np.zeros((5, 5), dtype=np.uint16)
    second = np.zeros((5, 5), dtype=np.uint16)
    first[2, 2] = 1
    first[2, 3] = 1
    second[2, 2] = 2
    second[2, 3] = 1

    def frame(frame_id: int, labels: np.ndarray) -> _Frame:
        return _Frame(
            frame_id=frame_id,
            image_stem=f"frame_{frame_id:06d}",
            width=5,
            height=5,
            fx=1.0,
            fy=1.0,
            cx=2.0,
            cy=2.0,
            rotation=np.eye(3, dtype=np.float32),
            translation=np.zeros(3, dtype=np.float32),
            labels=labels,
            is_manual=False,
        )

    labels, _confidence, provenance, summary = _vote_points(
        points,
        [frame(0, first), frame(1, second)],
        [1, 2],
        manual_frame_weight=8,
        progress=None,
    )

    assert labels.tolist() == [1, 1]
    assert np.all(provenance > 0)
    assert summary["tiedPointCount"] == 1
    assert summary["completion"]["completedPointCount"] == 1
    assert summary["unassignedPointCount"] == 0


def test_region_view_weights_are_normalized_over_positive_views(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.json"
    weights.write_text(
        json.dumps(
            {
                "frames": [
                    {"imageStem": "manual", "weight": 8},
                    {"imageStem": "propagated", "weight": 1},
                    {"imageStem": "absent", "weight": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "manual.png").touch()
    (masks / "propagated.png").touch()

    normalized = _load_view_weights(weights, masks)

    assert np.isclose(
        (normalized["manual"] + normalized["propagated"]) / 2.0,
        1.0,
    )
    assert np.isclose(
        normalized["manual"] / normalized["propagated"],
        8.0,
    )


def test_region_train_uses_point_initialization_and_densification(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, instance_iterations=10_000)
    command = region_train_command(
        config,
        dataset_dir=tmp_path / "region",
        model_dir=tmp_path / "model_out",
        training_view_weights=tmp_path / "weights.json",
    )

    assert "mapanything_3dgs.train_launch" in " ".join(command)
    assert "--initializer-gaussians" not in command
    assert "--is_instance" in command
    assert "--init_rec" not in command
    assert command[command.index("--densify_from_iter") + 1] == "500"
    assert command[command.index("--densify_until_iter") + 1] == "5000"
    assert command[command.index("--depth_l1_weight_init") + 1] == "0"
    assert command[command.index("--lambda_dssim") + 1] == "0"
    assert command[command.index("--mask-loss-weight") + 1] == "1"


def test_splat_prepare_uses_only_positive_region_views(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        minimum_region_points=2,
        minimum_positive_views=2,
    )
    paths = _paths(tmp_path)
    paths.clean_masks_dir.mkdir(parents=True)
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    source_images = paths.dataset_dir / "images"
    source_images.mkdir(parents=True)
    frames = []
    for frame_id in range(3):
        name = f"frame_{frame_id:06d}.JPEG"
        Image.fromarray(np.zeros((6, 8, 3), dtype=np.uint8)).save(
            source_images / name
        )
        frames.append(
            {
                "frameId": frame_id,
                "imageName": name,
                "width": 8,
                "height": 6,
                "fx": 10.0,
                "fy": 10.0,
                "cx": 4.0,
                "cy": 3.0,
                "colmapImageId": frame_id + 1,
                "qvec": [1, 0, 0, 0],
                "tvec": [0, 0, 0],
            }
        )
    write_jsonl(paths.frame_map, frames)

    region_dir = paths.split_instances_dir / "7"
    mask_dir = region_dir / "masks"
    mask_dir.mkdir(parents=True)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:4, 2:6] = 255
    for frame_id in (0, 2):
        Image.fromarray(mask).save(mask_dir / f"frame_{frame_id:06d}.png")
    points = np.zeros(
        3,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    point_path = region_dir / "region_7.ply"
    PlyData([PlyElement.describe(points, "vertex")]).write(point_path)
    paths.stage_summary("split").write_text(
        json.dumps(
            {
                "instanceIds": [7],
                "points": {
                    "regions": [
                        {
                            "regionId": 7,
                            "pointCount": 3,
                            "pointCloud": str(point_path),
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    paths.training_view_weights.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frameId": frame_id,
                        "imageStem": f"frame_{frame_id:06d}",
                        "isManualAnchor": frame_id == 0,
                        "weight": 8 if frame_id == 0 else 1,
                    }
                    for frame_id in range(3)
                ]
            }
        ),
        encoding="utf-8",
    )

    result = prepare_region_datasets(config, paths)
    dataset = paths.region_datasets / "7"

    assert result["instanceIds"] == [7]
    assert sorted(path.name for path in (dataset / "images").iterdir()) == [
        "frame_000000.JPEG",
        "frame_000002.JPEG",
    ]
    assert not (dataset / "images" / "frame_000001.JPEG").exists()
    assert (dataset / "sparse" / "0" / "points3D.ply").is_symlink()
    region_weights = json.loads(
        (dataset / "training_view_weights.json").read_text(encoding="utf-8")
    )
    assert [row["frameId"] for row in region_weights["frames"]] == [0, 2]


def test_live_pipeline_status_tracks_regions_and_iterations(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    run = (
        interactive
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
        / "mapanything_test"
    )
    (run / "03_splat/01_region_datasets").mkdir(parents=True)
    (run / "01_input").mkdir(parents=True)
    (run / "02_split/clean_masks").mkdir(parents=True)
    for relative, stage in (
        ("01_input/stage.json", "prepare"),
        ("02_split/clean_masks/stage.json", "split"),
    ):
        (run / relative).write_text(
            json.dumps({"stage": stage, "status": "complete"}),
            encoding="utf-8",
        )
    (run / "run.json").write_text(
        json.dumps(
            {
                "runId": "mapanything_test",
                "propagationMethod": "sam2_video",
                "manualFrameWeight": 8,
                "instanceIterations": 10_000,
            }
        ),
        encoding="utf-8",
    )
    (run / "03_splat/01_region_datasets/stage.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "discardedInstanceCount": 0,
                "instances": [{"regionId": 3}, {"regionId": 9}],
            }
        ),
        encoding="utf-8",
    )
    (run / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "message": "Training progress: 25%| 2500/10000",
                "updatedUtc": "2026-07-29T20:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    completed = (
        run
        / "03_splat/02_region_models/3"
        / "point_cloud/iteration_10000/point_cloud.ply"
    )
    completed.parent.mkdir(parents=True)
    completed.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 123\nend_header\n"
    )

    status = discover_mapanything_runs(interactive)[0]

    assert status["status"] == "running"
    assert status["stageIndex"] == 3
    assert status["trainedRegionCount"] == 1
    assert status["currentRegionId"] == 9
    assert status["currentIteration"] == 2500
    assert status["regionProgress"] == 25.0
    assert status["completedRegions"][0]["gaussianCount"] == 123


def test_live_pipeline_exposes_completed_region_gaussians(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    run = (
        interactive
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
        / "mapanything_test"
    )
    (run / "03_splat/01_region_datasets").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "runId": "mapanything_test",
                "propagationMethod": "sam2_video",
                "manualFrameWeight": 8,
                "instanceIterations": 10_000,
            }
        ),
        encoding="utf-8",
    )
    (run / "03_splat/01_region_datasets/stage.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "instances": [{"regionId": 4}],
            }
        ),
        encoding="utf-8",
    )
    point_cloud = (
        run
        / "03_splat/02_region_models/4"
        / "point_cloud/iteration_10000/point_cloud.ply"
    )
    point_cloud.parent.mkdir(parents=True)
    point_cloud.write_bytes(
        b"ply\nformat binary_little_endian 1.0\n"
        b"element vertex 77\nend_header\n"
    )

    payload = live_mapanything_experiment(
        interactive,
        "mapanything_test_splat",
    )

    assert payload is not None
    artifact = payload["gaussianArtifacts"]["rgb_region_4"]
    assert artifact["pointCount"] == 77
    assert artifact["path"] == str(point_cloud)


def test_floater_pruning_keeps_supported_and_removes_unsupported(
    tmp_path: Path,
) -> None:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("f_dc_0", "f4"),
            ("f_dc_1", "f4"),
            ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
        ]
    )
    vertices = np.zeros(3, dtype=dtype)
    vertices["x"] = [0.0, 1.0, 0.0]
    vertices["z"] = 1.0
    vertices["opacity"] = [0.0, 0.0, -10.0]
    point_cloud = tmp_path / "gaussians.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(point_cloud)

    masks = tmp_path / "masks"
    masks.mkdir()
    frames = []
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    for frame_id in range(3):
        stem = f"frame_{frame_id:06d}"
        Image.fromarray(mask).save(masks / f"{stem}.png")
        frames.append(
            {
                "frameId": frame_id,
                "imageName": f"{stem}.jpg",
                "width": 5,
                "height": 5,
                "fx": 1.0,
                "fy": 1.0,
                "cx": 2.0,
                "cy": 2.0,
                "qvec": [1, 0, 0, 0],
                "tvec": [0, 0, 0],
            }
        )
    frame_map = tmp_path / "frame_map.jsonl"
    write_jsonl(frame_map, frames)

    result = prune_gaussian_floaters(
        point_cloud,
        frame_map=frame_map,
        mask_dir=masks,
        summary_path=tmp_path / "pruning.json",
        mask_dilation_pixels=0,
    )
    output = PlyData.read(point_cloud)["vertex"].data

    assert output.shape == (1,)
    assert output["x"].tolist() == [0.0]
    assert result["removalReasons"]["lowOpacity"] == 1
    assert result["removalReasons"]["multiViewUnsupported"] == 1
    assert result["keptGaussianCount"] == 1


def test_floater_pruning_reads_split_splat_pose_schema(
    tmp_path: Path,
) -> None:
    dtype = np.dtype(
        [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]
    )
    vertices = np.zeros(1, dtype=dtype)
    vertices["z"] = 1.0
    point_cloud = tmp_path / "anchored_gaussians.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(point_cloud)

    masks = tmp_path / "anchored_masks"
    masks.mkdir()
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    frames = []
    pose_lines = []
    for frame_id in range(3):
        stem = f"scan_000_{frame_id:06d}"
        Image.fromarray(mask).save(masks / f"{stem}.png")
        frames.append(
            {
                "frameId": frame_id,
                "splitSplatImageName": f"{stem}.JPEG",
                "width": 5,
                "height": 5,
                "fx": 1.0,
                "fy": 1.0,
                "cx": 2.0,
                "cy": 2.0,
            }
        )
        pose_lines.extend(
            [f"{frame_id + 1} 1 0 0 0 0 0 0 {frame_id + 1} {stem}.JPEG", ""]
        )
    frame_map = tmp_path / "anchored_frame_map.jsonl"
    write_jsonl(frame_map, frames)
    images_txt = tmp_path / "images.txt"
    images_txt.write_text("\n".join(pose_lines), encoding="utf-8")

    result = prune_gaussian_floaters(
        point_cloud,
        frame_map=frame_map,
        mask_dir=masks,
        summary_path=tmp_path / "anchored_pruning.json",
        colmap_images=images_txt,
        mask_dilation_pixels=0,
    )

    assert result["keptGaussianCount"] == 1
    assert result["support"]["positiveMaskViewCount"] == 3


def test_floater_pruning_uses_rgba_alpha_not_hidden_rgb(
    tmp_path: Path,
) -> None:
    dtype = np.dtype(
        [("x", "f4"), ("y", "f4"), ("z", "f4"), ("opacity", "f4")]
    )
    vertices = np.zeros(2, dtype=dtype)
    vertices["x"] = [0.0, 1.0]
    vertices["z"] = 1.0
    point_cloud = tmp_path / "rgba_gaussians.ply"
    PlyData([PlyElement.describe(vertices, "vertex")]).write(point_cloud)

    masks = tmp_path / "rgba_masks"
    masks.mkdir()
    frames = []
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[..., :3] = [80, 160, 240]
    rgba[2, 2, 3] = 255
    for frame_id in range(3):
        stem = f"rgba_{frame_id:06d}"
        Image.fromarray(rgba, mode="RGBA").save(masks / f"{stem}.png")
        frames.append(
            {
                "frameId": frame_id,
                "imageName": f"{stem}.jpg",
                "width": 5,
                "height": 5,
                "fx": 1.0,
                "fy": 1.0,
                "cx": 2.0,
                "cy": 2.0,
                "qvec": [1, 0, 0, 0],
                "tvec": [0, 0, 0],
            }
        )
    frame_map = tmp_path / "rgba_frame_map.jsonl"
    write_jsonl(frame_map, frames)

    result = prune_gaussian_floaters(
        point_cloud,
        frame_map=frame_map,
        mask_dir=masks,
        summary_path=tmp_path / "rgba_pruning.json",
        mask_dilation_pixels=0,
    )

    assert result["keptGaussianCount"] == 1
    assert result["removalReasons"]["multiViewUnsupported"] == 1
