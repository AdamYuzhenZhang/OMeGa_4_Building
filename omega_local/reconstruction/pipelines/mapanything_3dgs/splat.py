"""Prepare, train, and compose MapAnything-initialized region 3DGS models."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from omega_local.reconstruction.gaussian_pruning import (
    prune_gaussian_floaters,
)
from omega_local.reconstruction.gaussian_io import (
    trained_point_cloud_path,
    write_segmented_gaussian_ply,
)
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
    replace_symlink,
)
from omega_local.segmentation.split_splat.storage import (
    cleanup_completed_models,
)
from omega_local.segmentation.split_splat.upstream import run_command

from .artifacts import register_splat_artifacts
from .contract import MapAnything3DGSConfig, MapAnything3DGSPaths


ProgressCallback = Callable[[str], None]


def prepare_region_datasets(
    config: MapAnything3DGSConfig,
    paths: MapAnything3DGSPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    split = read_json(paths.stage_summary("split"))
    frames = read_jsonl(paths.frame_map)
    if paths.region_datasets.exists():
        shutil.rmtree(paths.region_datasets)
    paths.region_datasets.mkdir(parents=True)

    weights = read_json(paths.training_view_weights)
    weight_by_frame = {
        int(row["frameId"]): row for row in weights.get("frames", [])
    }
    kept = []
    discarded = []
    point_rows = {
        int(row["regionId"]): row
        for row in split["points"].get("regions", [])
    }
    for index, region_id in enumerate(split["instanceIds"]):
        region_id = int(region_id)
        point_row = point_rows.get(region_id)
        source_points = (
            Path(str(point_row["pointCloud"]))
            if point_row is not None
            else Path()
        )
        source_masks = paths.split_instances_dir / str(region_id) / "masks"
        mask_frames = {
            path.stem: path
            for path in source_masks.glob("*.png")
            if _mask_has_foreground(path)
        }
        point_count = (
            int(point_row["pointCount"]) if point_row is not None else 0
        )
        reason = None
        if point_count < config.minimum_region_points:
            reason = "insufficient_mapanything_points"
        elif len(mask_frames) < config.minimum_positive_views:
            reason = "insufficient_positive_views"
        if reason is not None:
            discarded.append(
                {
                    "regionId": region_id,
                    "reason": reason,
                    "pointCount": point_count,
                    "positiveViewCount": len(mask_frames),
                }
            )
            continue

        dataset = paths.region_datasets / str(region_id)
        image_dir = dataset / "images"
        mask_dir = dataset / "masks"
        sparse_dir = dataset / "sparse" / "0"
        for directory in (image_dir, mask_dir, sparse_dir):
            directory.mkdir(parents=True)
        selected_frames = [
            row
            for row in frames
            if Path(str(row["imageName"])).stem in mask_frames
        ]
        camera_lines = [
            "# Camera list with one line of data per camera:",
            "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
            f"# Number of cameras: {len(selected_frames)}",
        ]
        image_lines = [
            "# Image list with two lines of data per image:",
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
            "# POINTS2D[] as (X, Y, POINT3D_ID)",
            f"# Number of images: {len(selected_frames)}, mean observations per image: 0",
        ]
        region_weight_rows = []
        for frame in selected_frames:
            frame_id = int(frame["frameId"])
            image_name = str(frame["imageName"])
            stem = Path(image_name).stem
            camera_id = int(frame["colmapImageId"])
            replace_symlink(
                image_dir / image_name,
                paths.dataset_dir / "images" / image_name,
            )
            shutil.copy2(mask_frames[stem], mask_dir / f"{stem}.png")
            camera_lines.append(
                f"{camera_id} PINHOLE {int(frame['width'])} {int(frame['height'])} "
                f"{float(frame['fx']):.17g} {float(frame['fy']):.17g} "
                f"{float(frame['cx']):.17g} {float(frame['cy']):.17g}"
            )
            image_lines.extend(
                [
                    f"{camera_id} "
                    f"{' '.join(f'{float(value):.17g}' for value in frame['qvec'])} "
                    f"{' '.join(f'{float(value):.17g}' for value in frame['tvec'])} "
                    f"{camera_id} {image_name}",
                    "",
                ]
            )
            source_weight = weight_by_frame.get(
                frame_id,
                {
                    "frameId": frame_id,
                    "imageStem": stem,
                    "isManualAnchor": False,
                    "weight": 1,
                },
            )
            region_weight_rows.append(source_weight)
        (sparse_dir / "cameras.txt").write_text(
            "\n".join(camera_lines) + "\n",
            encoding="utf-8",
        )
        (sparse_dir / "images.txt").write_text(
            "\n".join(image_lines) + "\n",
            encoding="utf-8",
        )
        replace_symlink(sparse_dir / "points3D.ply", source_points)
        region_weights = dataset / "training_view_weights.json"
        atomic_write_json(
            region_weights,
            {
                "schemaVersion": 1,
                "manualFrameWeight": config.manual_frame_weight,
                "frames": region_weight_rows,
            },
        )
        kept.append(
            {
                "regionId": region_id,
                "datasetDir": str(dataset),
                "pointCloud": str(source_points),
                "pointCount": point_count,
                "positiveViewCount": len(selected_frames),
                "manualPositiveViewCount": sum(
                    bool(row.get("isManualAnchor"))
                    for row in region_weight_rows
                ),
                "trainingViewWeights": str(region_weights),
            }
        )
        _report(
            progress,
            f"Prepared MapAnything region {index + 1}/{len(split['instanceIds'])} "
            f"(ID {region_id}, {point_count:,} points, "
            f"{len(selected_frames)} views).",
        )

    if not kept:
        raise ValueError(
            "No persistent region has enough MapAnything points and positive views."
        )
    return {
        "schemaVersion": 1,
        "stage": "splat_prepare",
        "timestampUtc": now_utc(),
        "method": "Per-region masked 3DGS datasets from MapAnything point subsets",
        "instanceCount": len(kept),
        "discardedInstanceCount": len(discarded),
        "instanceIds": [row["regionId"] for row in kept],
        "instances": kept,
        "discarded": discarded,
        "settings": {
            "initializer": "region-owned MapAnything RGB points",
            "minimumRegionPoints": config.minimum_region_points,
            "minimumPositiveViews": config.minimum_positive_views,
            "viewPolicy": "only frames with nonempty region masks",
            "unknownPixelPolicy": "absent frames are excluded, not negative",
            "manualFrameWeight": config.manual_frame_weight,
        },
    }


def train_regions(
    config: MapAnything3DGSConfig,
    paths: MapAnything3DGSPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    prepared = read_json(paths.stage_summary("splat_prepare"))
    paths.region_models.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, item in enumerate(prepared["instances"]):
        region_id = int(item["regionId"])
        model_dir = paths.region_models / str(region_id)
        point_cloud = trained_point_cloud_path(
            model_dir,
            config.instance_iterations,
        )
        command = region_train_command(
            config,
            dataset_dir=Path(item["datasetDir"]),
            model_dir=model_dir,
            training_view_weights=Path(item["trainingViewWeights"]),
        )
        cache_hit = point_cloud.is_file()
        result = (
            {"cacheHit": True, "returnCode": 0}
            if cache_hit
            else run_command(
                command,
                cwd=paths.run_dir,
                config=config,
                log_path=paths.logs_dir / "splat_train" / f"{region_id}.log",
                progress=command_progress,
                dry_run=dry_run,
            )
        )
        if not dry_run and not point_cloud.is_file():
            raise FileNotFoundError(
                f"Region {region_id} did not produce {point_cloud}"
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
            _report(
                progress,
                f"Pruned MapAnything region {region_id}: "
                f"{pruning['removedGaussianCount']:,} removed.",
            )
        rows.append(
            {
                "regionId": region_id,
                "modelDir": str(model_dir),
                "pointCloud": str(point_cloud),
                "cacheHit": cache_hit,
                "command": result,
                "floaterPruning": pruning,
            }
        )
        _report(
            progress,
            f"Trained MapAnything regions {index + 1}/{len(prepared['instances'])} "
            f"(ID {region_id}).",
        )
    return {
        "schemaVersion": 1,
        "stage": "splat_train",
        "timestampUtc": now_utc(),
        "method": "Masked per-region 3DGS from MapAnything RGB point initialization",
        "instanceCount": len(rows),
        "instances": rows,
        "settings": {
            "iterations": config.instance_iterations,
            "densification": True,
            "maximumSplatsPerRegion": config.max_num_splats,
            "densificationCapPolicy": (
                "stop at current Gaussian count; prune one-step overflow by "
                "lowest opacity"
            ),
            "densifyUntilIteration": max(
                min(
                    config.instance_iterations // 2,
                    config.instance_iterations - 1,
                ),
                1,
            ),
            "stabilizationIterations": config.instance_iterations - max(
                min(
                    config.instance_iterations // 2,
                    config.instance_iterations - 1,
                ),
                1,
            ),
            "initializer": "MapAnything RGB points, not pretrained Gaussians",
            "maskLossWeighting": (
                "foreground/background-balanced identity loss; manual frame "
                "weight normalized over each region positive views"
            ),
            "maskLossWeight": config.mask_loss_weight,
            "rgbLossWeighting": (
                "foreground-normalized RGB; uniform across positive views"
            ),
            "depthLoss": False,
            "maskShapeMutation": False,
            "floaterPruning": config.floater_pruning,
            "floaterPruningPolicy": (
                "opacity below threshold, world scale above 0.1 camera extent, "
                "or repeated front visibility with zero mask support"
            ),
        },
    }


def region_train_command(
    config: MapAnything3DGSConfig,
    *,
    dataset_dir: Path,
    model_dir: Path,
    training_view_weights: Path,
) -> list[str]:
    half = max(config.instance_iterations // 2, 1)
    densify_until = max(
        min(config.instance_iterations // 2, config.instance_iterations - 1),
        1,
    )
    densify_from = min(500, max(densify_until - 1, 0))
    return [
        str(config.python),
        "-m",
        (
            "omega_local.reconstruction.pipelines.mapanything_3dgs."
            "train_launch"
        ),
        "--upstream-train",
        str(config.split_splat_root / "train.py"),
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
        "--densify_from_iter",
        str(densify_from),
        "--densify_until_iter",
        str(densify_until),
        "--max_num_splats",
        str(config.max_num_splats),
        "--depth_l1_weight_init",
        "0",
        "--depth_l1_weight_final",
        "0",
        "--lambda_dssim",
        "0",
        "--test_iterations",
        str(half),
        str(config.instance_iterations),
        "--save_iterations",
        str(half),
        str(config.instance_iterations),
        "--disable_viewer",
    ]


def compose_regions(
    config: MapAnything3DGSConfig,
    paths: MapAnything3DGSPaths,
    editor_paths: Any,
) -> dict[str, Any]:
    trained = read_json(paths.stage_summary("splat_train"))
    paths.splat_outputs.mkdir(parents=True, exist_ok=True)
    blocks = []
    label_blocks = []
    rows = []
    source_ply = None
    for item in trained["instances"]:
        region_id = int(item["regionId"])
        point_cloud = Path(item["pointCloud"])
        ply = PlyData.read(point_cloud, mmap="r")
        vertices = np.array(ply["vertex"].data, copy=True)
        if source_ply is None:
            source_ply = ply
        elif vertices.dtype != blocks[0].dtype:
            raise ValueError(
                "Region 3DGS PLY schemas differ and cannot be concatenated: "
                f"{point_cloud}"
            )
        blocks.append(vertices)
        label_blocks.append(
            np.full(vertices.shape[0], region_id, dtype=np.int32)
        )
        rows.append(
            {
                "regionId": region_id,
                "pointCloud": str(point_cloud),
                "gaussianCount": int(vertices.shape[0]),
            }
        )
    if not blocks or source_ply is None:
        raise ValueError("No trained MapAnything region model is available.")
    vertices = np.concatenate(blocks)
    labels = np.concatenate(label_blocks)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=source_ply.text,
        byte_order=getattr(source_ply, "byte_order", "<"),
        comments=list(source_ply.comments),
        obj_info=list(source_ply.obj_info),
    ).write(paths.composed_scene)
    np.save(paths.composed_labels, labels)
    write_segmented_gaussian_ply(
        paths.composed_scene,
        paths.composed_region_ids,
        labels,
    )
    summary = {
        "schemaVersion": 1,
        "stage": "splat_compose",
        "timestampUtc": now_utc(),
        "method": "Direct world-aligned concatenation of independently trained regions",
        "regionCount": len(rows),
        "gaussianCount": int(vertices.shape[0]),
        "regions": rows,
        "outputs": {
            "rgbGaussians": str(paths.composed_scene),
            "regionIdGaussians": str(paths.composed_region_ids),
            "regionLabels": str(paths.composed_labels),
        },
        "settings": {
            "jointOptimization": False,
            "collisionRefinement": False,
            "coordinateSystem": "shared OMeGa world coordinates",
            "note": (
                "This first MapAnything baseline measures independent region "
                "training. Boundary-aware joint refinement is intentionally "
                "kept separate."
            ),
        },
    }
    summary["editorArtifact"] = register_splat_artifacts(
        config,
        paths,
        editor_paths,
        summary,
    )
    return summary


def region_storage_cleanup(
    paths: MapAnything3DGSPaths,
    *,
    final_iteration: int,
) -> dict[str, Any]:
    return cleanup_completed_models(
        paths.region_models,
        final_iteration=final_iteration,
    )



def _mask_has_foreground(path: Path) -> bool:
    image = Image.open(path)
    values = (
        np.asarray(image.getchannel("A"))
        if "A" in image.getbands()
        else np.asarray(image.convert("L"))
    )
    return bool(np.any(values > 0))


def _report(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
