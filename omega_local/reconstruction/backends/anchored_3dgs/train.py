"""Train anchored per-region 3DGS models through the released trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from omega_local.reconstruction.gaussian_pruning import (
    prune_gaussian_floaters,
)
from omega_local.segmentation.split_splat.contract import now_utc, read_json
from omega_local.reconstruction.gaussian_io import trained_point_cloud_path
from omega_local.segmentation.split_splat.upstream import run_command

from .contract import AnchoredSplatConfig, AnchoredSplatPaths


ProgressCallback = Callable[[str], None]


def train_initial_regions(
    config: AnchoredSplatConfig,
    runtime_config: Any,
    paths: AnchoredSplatPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _train_regions(
        config,
        runtime_config,
        paths,
        output_dir=paths.splat_initial_models,
        stage="initial",
        initial_pass=True,
        progress=progress,
        command_progress=command_progress,
        dry_run=dry_run,
    )


def train_refined_regions(
    config: AnchoredSplatConfig,
    runtime_config: Any,
    paths: AnchoredSplatPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _train_regions(
        config,
        runtime_config,
        paths,
        output_dir=paths.splat_refined_models,
        stage="refined",
        initial_pass=False,
        progress=progress,
        command_progress=command_progress,
        dry_run=dry_run,
    )


def anchored_train_command(
    config: AnchoredSplatConfig,
    *,
    dataset_dir: Path,
    model_dir: Path,
    initializer_gaussians: Path,
    training_view_weights: Path,
    initial_pass: bool,
) -> list[str]:
    half = max(config.instance_iterations // 2, 1)
    command = [
        str(config.python),
        "-m",
        "omega_local.reconstruction.backends.anchored_3dgs.train_launch",
        "--upstream-train",
        str(config.split_splat_root / "train.py"),
        "--initializer-gaussians",
        str(initializer_gaussians),
        "--training-view-weights",
        str(training_view_weights),
        "--training-mask-dir",
        str(dataset_dir / "masks"),
        "--mask-loss-weight",
        f"{config.mask_loss_weight:.9g}",
        "--",
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(config.instance_iterations),
        "--is_instance",
        "--test_iterations",
        str(half),
        str(config.instance_iterations),
        "--save_iterations",
        str(half),
        str(config.instance_iterations),
        "--densify_from_iter",
        str(config.instance_iterations),
        "--lambda_dssim",
        "0",
        "--disable_viewer",
    ]
    if initial_pass:
        command.append("--init_rec")
    return command


def _train_regions(
    config: AnchoredSplatConfig,
    runtime_config: Any,
    paths: AnchoredSplatPaths,
    *,
    output_dir: Path,
    stage: str,
    initial_pass: bool,
    progress: ProgressCallback | None,
    command_progress: ProgressCallback | None,
    dry_run: bool,
) -> dict[str, Any]:
    prepared = read_json(paths.stage_summary("prepare"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(prepared["instances"]):
        region_id = int(item["regionId"])
        model_dir = output_dir / str(region_id)
        point_cloud = trained_point_cloud_path(
            model_dir,
            config.instance_iterations,
        )
        cache_hit = point_cloud.is_file()
        command = anchored_train_command(
            config,
            dataset_dir=Path(item["datasetDir"]),
            model_dir=model_dir,
            initializer_gaussians=Path(item["initializerGaussians"]),
            training_view_weights=Path(item["trainingViewWeights"]),
            initial_pass=initial_pass,
        )
        if cache_hit:
            result = {"cacheHit": True, "returnCode": 0}
        else:
            result = run_command(
                command,
                cwd=paths.splat_workspace,
                config=runtime_config,
                log_path=paths.logs_dir / stage / f"{region_id}.log",
                progress=command_progress,
                dry_run=dry_run,
            )
        if not dry_run and not point_cloud.is_file():
            raise FileNotFoundError(
                f"Anchored region {region_id} did not produce {point_cloud}"
            )
        pruning = None
        if not dry_run and config.floater_pruning:
            pruning = prune_gaussian_floaters(
                point_cloud,
                frame_map=paths.frame_map,
                mask_dir=Path(item["datasetDir"]) / "masks",
                summary_path=model_dir / "floater_pruning.json",
                colmap_images=(
                    Path(item["datasetDir"]) / "sparse" / "0" / "images.txt"
                ),
                opacity_threshold=config.floater_opacity_threshold,
                min_visible_views=config.floater_min_visible_views,
                mask_dilation_pixels=config.floater_mask_dilation_pixels,
            )
            if progress is not None:
                progress(
                    f"Pruned anchored region {region_id}: "
                    f"{pruning['removedGaussianCount']:,} removed."
                )
        rows.append(
            {
                "instanceId": region_id,
                "regionId": region_id,
                "modelDir": str(model_dir),
                "pointCloud": str(point_cloud),
                "cacheHit": cache_hit,
                "command": result,
                "floaterPruning": pruning,
            }
        )
        if progress is not None:
            progress(
                f"Anchored {stage} regions {index + 1}/{len(prepared['instances'])} "
                f"(ID {region_id})."
            )
    return {
        "schemaVersion": 1,
        "stage": stage,
        "timestampUtc": now_utc(),
        "method": "Anchored full-Gaussian per-region 3DGS optimization",
        "instanceCount": len(rows),
        "instances": rows,
        "settings": {
            "iterations": config.instance_iterations,
            "initialPass": initial_pass,
            "initializer": "exact full-attribute global Gaussian subset",
            "topologyPolicy": (
                "preserve initializer topology during optimization, then "
                "remove low-opacity, oversized, and unsupported Gaussians"
            ),
            "maskLossWeighting": (
                "foreground/background-balanced identity loss; per-frame "
                "anchor weight divided by positive-view mean"
            ),
            "maskLossWeight": config.mask_loss_weight,
            "rgbLossWeighting": (
                "foreground-normalized RGB; uniform across positive views"
            ),
            "floaterPruning": config.floater_pruning,
            "floaterPruningPolicy": (
                "post-training low-opacity, world-scale, and conservative "
                "multi-view semantic support pruning"
            ),
        },
    }
