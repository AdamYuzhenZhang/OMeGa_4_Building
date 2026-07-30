"""Paper-faithful, resumable adapter for the Split&Splat Splat stage."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from omega_local.reconstruction.gaussian_diagnostics import (
    summarize_gaussian_ply,
)

from omega_local.segmentation.interactive.paths import EditorPaths

from .adapter import _read_colmap_images_text, _world_from_camera
from omega_local.reconstruction.gaussian_io import (
    ensure_viewer_gaussian_ply,
    trained_point_cloud_path,
    write_segmented_gaussian_ply,
    write_viewer_gaussian_partitions,
)
from .contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    read_json,
    read_jsonl,
    replace_symlink,
)
from .upstream import (
    composition_train_command,
    instance_train_command,
    mask_refinement_command,
    run_command,
)


ProgressCallback = Callable[[str], None]
PostCompositionFilter = Callable[
    [Path, np.ndarray],
    tuple[np.ndarray, dict[str, Any]],
]


@dataclass(frozen=True)
class _CompositionNode:
    name: str
    instance_ids: tuple[int, ...]
    ply_path: Path
    masks_dir: Path
    point_labels: np.ndarray
    cache_key: str = ""


def prepare_splat(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build isolated per-instance datasets from a completed Split run."""
    split_summary = read_json(paths.stage_summary("split"))
    support = _ensure_shared_support(config, paths, progress=progress)
    global_points, global_rgb, global_tree = _global_rgb_initialization(
        paths.global_point_cloud
    )
    training_weights_source = _training_view_weights_source(split_summary, paths)
    instance_ids = [int(value) for value in split_summary.get("instanceIds", [])]
    if not instance_ids:
        raise ValueError("Split produced no instances to reconstruct.")

    if paths.splat_instances_dir.exists():
        shutil.rmtree(paths.splat_instances_dir)
    paths.splat_instances_dir.mkdir(parents=True)
    _prepare_workspace(paths)

    camera_source = paths.dataset_dir / "sparse" / "0"
    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for index, instance_id in enumerate(instance_ids):
        source = paths.split_raw_output / str(instance_id)
        point_source = source / f"label_{instance_id}.ply"
        masks = sorted(
            path
            for path in source.glob("*.png")
            if _mask_has_foreground(path)
        )
        reason = None
        if len(masks) < 2:
            reason = "fewer_than_two_nonempty_masks"
        elif not point_source.is_file():
            reason = "missing_labeled_point_cloud"
        if reason is not None:
            discarded.append({"instanceId": instance_id, "reason": reason})
            continue

        dataset_dir = paths.splat_instances_dir / str(instance_id)
        mask_dir = dataset_dir / "masks"
        sparse_dir = dataset_dir / "sparse" / "0"
        mask_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        replace_symlink(dataset_dir / "images", paths.shared_splat_support_dir / "images")
        replace_symlink(sparse_dir / "cameras.txt", camera_source / "cameras.txt")
        replace_symlink(sparse_dir / "images.txt", camera_source / "images.txt")
        initialization = sparse_dir / "points3D.ply"
        color_summary = _write_colored_instance_initialization(
            point_source,
            initialization,
            global_points=global_points,
            global_rgb=global_rgb,
            global_tree=global_tree,
        )
        for mask_path in masks:
            shutil.copy2(mask_path, mask_dir / mask_path.name)
        staged_training_weights = None
        if training_weights_source is not None:
            staged_training_weights = dataset_dir / "training_view_weights.json"
            replace_symlink(staged_training_weights, training_weights_source)
        kept.append(
            {
                "instanceId": instance_id,
                "datasetDir": str(dataset_dir),
                "pointCloud": str(initialization),
                "sourcePointCloud": str(point_source),
                "initializationColor": color_summary,
                "maskCount": len(masks),
                "trainingViewWeights": (
                    str(staged_training_weights)
                    if staged_training_weights is not None
                    else None
                ),
            }
        )
        _progress_count(progress, "Prepared Splat instances", index + 1, len(instance_ids))

    if not kept:
        raise ValueError("No Split instance has both labeled points and two masks.")
    summary = {
        "schemaVersion": 1,
        "stage": "splat_prepare",
        "timestampUtc": now_utc(),
        "method": "Released Split&Splat per-instance dataset contract",
        "sourceSplitSummary": str(paths.stage_summary("split")),
        "sharedSupport": support,
        "instanceCount": len(kept),
        "discardedInstanceCount": len(discarded),
        "instanceIds": [row["instanceId"] for row in kept],
        "instances": kept,
        "discarded": discarded,
        "settings": {
            "minimumNonemptyViews": 2,
            "initialization": "Split-labeled global Gaussian means",
            "trainingViewWeights": {
                "available": training_weights_source is not None,
                "source": (
                    str(training_weights_source)
                    if training_weights_source is not None
                    else None
                ),
                "appliedByReleasedTrainer": False,
            },
        },
    }
    return summary


def _training_view_weights_source(
    split_summary: dict[str, Any],
    paths: SplitSplatRunPaths,
) -> Path | None:
    """Resolve optional anchor weights without changing released training."""
    candidates = [
        Path(str(split_summary.get("trainingViewWeightsPath") or "")),
        paths.training_view_weights,
    ]
    for candidate in candidates:
        if str(candidate) and candidate.is_file():
            return candidate.resolve()
    return None


def _global_rgb_initialization(
    source: Path,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """Read Gaussian means and DC appearance for colored instance starts."""
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    vertex = PlyData.read(source, mmap="r")["vertex"].data
    names = set(vertex.dtype.names or ())
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"}
    if not required.issubset(names):
        raise ValueError(
            "Global Split&Splat Gaussians cannot seed RGB instances; missing "
            f"{sorted(required.difference(names))} in {source}"
        )
    points = np.column_stack(
        [vertex["x"], vertex["y"], vertex["z"]]
    ).astype(np.float64, copy=False)
    dc = np.column_stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]]
    ).astype(np.float32, copy=False)
    rgb = np.clip(
        np.float32(0.28209479177387814) * dc + np.float32(0.5),
        0.0,
        1.0,
    )
    return points, np.rint(rgb * 255.0).astype(np.uint8), cKDTree(points)


def _write_colored_instance_initialization(
    source: Path,
    destination: Path,
    *,
    global_points: np.ndarray,
    global_rgb: np.ndarray,
    global_tree: Any,
) -> dict[str, Any]:
    """Restore RGB omitted by the released Split point-cloud writer."""
    from plyfile import PlyData, PlyElement

    source_vertex = PlyData.read(source, mmap="r")["vertex"].data
    points = np.column_stack(
        [source_vertex["x"], source_vertex["y"], source_vertex["z"]]
    ).astype(np.float64, copy=False)
    distances, indices = global_tree.query(points, k=1, workers=-1)
    scene_extent = float(np.linalg.norm(np.ptp(global_points, axis=0)))
    tolerance = max(scene_extent * 1e-6, 1e-7)
    matched = np.isfinite(distances) & (distances <= tolerance)
    rgb = np.full((points.shape[0], 3), 128, dtype=np.uint8)
    rgb[matched] = global_rgb[np.asarray(indices[matched], dtype=np.int64)]
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    output = np.zeros(points.shape[0], dtype=dtype)
    output["x"], output["y"], output["z"] = points.T
    output["red"], output["green"], output["blue"] = rgb.T
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(output, "vertex")],
        text=False,
        byte_order="<",
    ).write(destination)
    return {
        "method": "nearest_exact_global_gaussian_dc",
        "matchedPointCount": int(np.count_nonzero(matched)),
        "pointCount": int(points.shape[0]),
        "distanceTolerance": tolerance,
        "maximumMatchedDistance": (
            float(np.max(distances[matched])) if np.any(matched) else 0.0
        ),
    }


def train_initial_instances(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    prepared = read_json(paths.stage_summary("splat_prepare"))
    rows = prepared["instances"]
    paths.splat_initial_models.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        instance_id = int(row["instanceId"])
        model_dir = paths.splat_initial_models / str(instance_id)
        point_cloud = trained_point_cloud_path(
            model_dir,
            config.instance_iterations,
        )
        cache_hit = point_cloud.is_file()
        if not cache_hit:
            command = instance_train_command(
                config,
                dataset_dir=Path(row["datasetDir"]),
                model_dir=model_dir,
                initial_pass=True,
            )
            result = run_command(
                command,
                cwd=paths.splat_workspace,
                config=config,
                log_path=paths.logs_dir / "splat_initial" / f"{instance_id}.log",
                progress=command_progress,
            )
        else:
            result = {"cacheHit": True, "returnCode": 0}
        if not point_cloud.is_file():
            raise FileNotFoundError(
                f"Initial instance {instance_id} did not produce {point_cloud}"
            )
        commands.append(
            {
                "instanceId": instance_id,
                "modelDir": str(model_dir),
                "pointCloud": str(point_cloud),
                "cacheHit": cache_hit,
                "command": result,
            }
        )
        _progress_count(progress, "Initial instance reconstructions", index + 1, len(rows))
    return _training_summary(
        "splat_initial",
        "Released Split&Splat initial per-instance 3DGS",
        config,
        commands,
        initialPass=True,
    )


def refine_instance_masks(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    prepared = read_json(paths.stage_summary("splat_prepare"))
    _prepare_mask_refinement_workspace(config, paths)
    paths.splat_mask_refinement.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(prepared["instances"]):
        instance_id = int(item["instanceId"])
        marker = paths.splat_mask_refinement / f"{instance_id}.json"
        if marker.is_file():
            row = read_json(marker)
            row["cacheHit"] = True
        else:
            command = mask_refinement_command(
                config,
                paths,
                instance_id=instance_id,
            )
            result = run_command(
                command,
                cwd=paths.splat_workspace,
                config=config,
                log_path=paths.logs_dir / "splat_masks" / f"{instance_id}.log",
                progress=command_progress,
            )
            dataset_dir = Path(item["datasetDir"])
            added = _merge_refined_masks(dataset_dir)
            row = {
                "instanceId": instance_id,
                "datasetDir": str(dataset_dir),
                "refinedOrAddedMaskCount": added,
                "command": result,
                "cacheHit": False,
            }
            atomic_write_json(marker, row)
        rows.append(row)
        _progress_count(progress, "Refined instance masks", index + 1, len(prepared["instances"]))
    return {
        "schemaVersion": 1,
        "stage": "splat_masks",
        "timestampUtc": now_utc(),
        "method": "Released Split&Splat Gaussian reprojection and SAM2 mask refinement",
        "instanceCount": len(rows),
        "instances": rows,
        "settings": {
            "promptCount": 5,
            "candidateSelection": "higher IoU against rendered Gaussian mask",
            "cameraCoverage": "all_views",
            "missingMaskMinimumIoU": 0.95,
            "sourceNote": (
                "The compatibility launcher removes the released script's "
                "half-camera limit and replaces its 0.05 missing-mask gate with "
                "the paper's tau_iou=0.95."
            ),
            "maskDilationPixels": 1,
        },
    }


def train_refined_instances(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    prepared = read_json(paths.stage_summary("splat_prepare"))
    paths.splat_refined_models.mkdir(parents=True, exist_ok=True)
    commands: list[dict[str, Any]] = []
    for index, row in enumerate(prepared["instances"]):
        instance_id = int(row["instanceId"])
        model_dir = paths.splat_refined_models / str(instance_id)
        point_cloud = trained_point_cloud_path(model_dir, config.instance_iterations)
        cache_hit = point_cloud.is_file()
        if not cache_hit:
            command = instance_train_command(
                config,
                dataset_dir=Path(row["datasetDir"]),
                model_dir=model_dir,
                initial_pass=False,
            )
            result = run_command(
                command,
                cwd=paths.splat_workspace,
                config=config,
                log_path=paths.logs_dir / "splat_refined" / f"{instance_id}.log",
                progress=command_progress,
            )
        else:
            result = {"cacheHit": True, "returnCode": 0}
        if not point_cloud.is_file():
            raise FileNotFoundError(
                f"Refined instance {instance_id} did not produce {point_cloud}"
            )
        commands.append(
            {
                "instanceId": instance_id,
                "modelDir": str(model_dir),
                "pointCloud": str(point_cloud),
                "cacheHit": cache_hit,
                "command": result,
            }
        )
        _progress_count(progress, "Refined instance reconstructions", index + 1, len(prepared["instances"]))
    return _training_summary(
        "splat_refined",
        "Released Split&Splat refined per-instance 3DGS",
        config,
        commands,
        initialPass=False,
    )


def compose_instances(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None = None,
    command_progress: ProgressCallback | None = None,
    post_composition_filter: PostCompositionFilter | None = None,
    foreground_balanced: bool = False,
) -> dict[str, Any]:
    """Compose refined objects with the paper's collision-driven schedule."""
    prepared = read_json(paths.stage_summary("splat_prepare"))
    paths.splat_composition.mkdir(parents=True, exist_ok=True)
    nodes = []
    for row in prepared["instances"]:
        instance_id = int(row["instanceId"])
        ply_path = trained_point_cloud_path(
            paths.splat_refined_models / str(instance_id),
            config.instance_iterations,
        )
        point_count = _ply_vertex_count(ply_path)
        masks_dir = Path(row["datasetDir"]) / "masks"
        nodes.append(
            _CompositionNode(
                name=str(instance_id),
                instance_ids=(instance_id,),
                ply_path=ply_path,
                masks_dir=masks_dir,
                point_labels=np.full(point_count, instance_id, dtype=np.int32),
                cache_key=_composition_source_key(
                    ply_path,
                    masks_dir,
                    instance_ids=(instance_id,),
                ),
            )
        )

    rounds: list[dict[str, Any]] = []
    round_index = 0
    while len(nodes) > 1:
        pairs = _select_collision_pairs(nodes)
        if not pairs:
            break
        weight = config.composition_mask_weights[
            min(round_index, len(config.composition_mask_weights) - 1)
        ]
        round_dir = paths.splat_composition / f"round_{round_index:02d}"
        next_nodes: list[_CompositionNode] = []
        consumed: set[str] = set()
        pair_rows = []
        for pair_index, (left, right, score) in enumerate(pairs):
            consumed.update((left.name, right.name))
            combined_ids = tuple(
                sorted((*left.instance_ids, *right.instance_ids))
            )
            combined_name = "_".join(str(value) for value in combined_ids)
            storage_key = _composition_storage_key(combined_ids)
            dataset_dir = round_dir / "datasets" / storage_key
            model_dir, output_ply = _composition_model_paths(
                round_dir,
                storage_key=storage_key,
                iterations=config.composition_iterations,
            )
            labels = np.concatenate((left.point_labels, right.point_labels))
            cache_input = _composition_cache_input(
                left,
                right,
                iterations=config.composition_iterations,
                mask_weight=weight,
                foreground_balanced=foreground_balanced,
            )
            cache_manifest = model_dir / "composition_cache.json"
            cache_hit, cache_reason = _composition_cache_status(
                cache_manifest,
                output_ply,
                cache_input,
                expected_point_count=int(labels.shape[0]),
            )
            if cache_hit and not (dataset_dir / "masks").is_dir():
                cache_hit = False
                cache_reason = "composition_masks_missing"
            if not cache_hit:
                if model_dir.exists() or model_dir.is_symlink():
                    if model_dir.is_symlink() or model_dir.is_file():
                        model_dir.unlink()
                    else:
                        shutil.rmtree(model_dir)
                _prepare_composition_dataset(
                    paths,
                    dataset_dir,
                    left,
                    right,
                )
                command = composition_train_command(
                    config,
                    dataset_dir=dataset_dir,
                    model_dir=model_dir,
                    mask_weight=weight,
                    foreground_balanced=foreground_balanced,
                )
                try:
                    result = run_command(
                        command,
                        cwd=paths.splat_workspace,
                        config=config,
                        log_path=(
                            paths.logs_dir
                            / "splat_compose"
                            / f"round_{round_index:02d}_{storage_key}.log"
                        ),
                        progress=command_progress,
                    )
                finally:
                    _cleanup_composition_training_artifacts(
                        dataset_dir,
                        model_dir,
                        iterations=config.composition_iterations,
                    )
            else:
                result = {"cacheHit": True, "returnCode": 0}
                _cleanup_composition_training_artifacts(
                    dataset_dir,
                    model_dir,
                    iterations=config.composition_iterations,
                )
            output_point_count = _ply_vertex_count(output_ply)
            if output_point_count != labels.shape[0]:
                raise RuntimeError(
                    "Composition changed Gaussian count despite disabled "
                    f"densification: expected {labels.shape[0]}, got "
                    f"{output_point_count}: {output_ply}"
                )
            output_identity = _composition_file_identity(output_ply)
            if not cache_hit:
                atomic_write_json(
                    cache_manifest,
                    {
                        "schemaVersion": 1,
                        "input": cache_input,
                        "expectedPointCount": int(labels.shape[0]),
                        "output": output_identity,
                    },
                )
            node_cache_key = _composition_digest(
                {
                    "input": cache_input,
                    "output": output_identity,
                }
            )
            next_nodes.append(
                _CompositionNode(
                    name=combined_name,
                    instance_ids=combined_ids,
                    ply_path=output_ply,
                    masks_dir=dataset_dir / "masks",
                    point_labels=labels,
                    cache_key=node_cache_key,
                )
            )
            pair_rows.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "combined": combined_name,
                    "storageKey": storage_key,
                    "collisionScore": score,
                    "maskLossWeight": weight,
                    "modelDir": str(model_dir),
                    "cacheHit": cache_hit,
                    "cacheReason": cache_reason,
                    "command": result,
                }
            )
            _progress_count(
                progress,
                f"Composition round {round_index + 1}",
                pair_index + 1,
                len(pairs),
            )
        next_nodes.extend(node for node in nodes if node.name not in consumed)
        rounds.append(
            {
                "round": round_index,
                "maskLossWeight": weight,
                "inputNodeCount": len(nodes),
                "outputNodeCount": len(next_nodes),
                "pairs": pair_rows,
            }
        )
        nodes = sorted(next_nodes, key=lambda node: node.instance_ids)
        round_index += 1

    paths.splat_outputs.mkdir(parents=True, exist_ok=True)
    final_model = paths.splat_outputs / "composed_scene.ply"
    final_labels = paths.splat_outputs / "composed_scene_instance_labels.npy"
    final_ids = paths.splat_outputs / "composed_scene_instance_ids.ply"
    _concatenate_nodes(nodes, final_model)
    labels = np.concatenate([node.point_labels for node in nodes])
    post_composition_pruning = None
    if post_composition_filter is not None:
        labels, post_composition_pruning = post_composition_filter(
            final_model,
            labels,
        )
    np.save(final_labels, labels)
    write_segmented_gaussian_ply(final_model, final_ids, labels)
    individual_objects = _export_individual_objects(
        final_model,
        labels,
        paths.splat_outputs / "individual_objects",
        include_unassigned=True,
        semantic_colors=True,
    )
    composed_rgb_objects = _export_individual_objects(
        final_model,
        labels,
        paths.splat_outputs / "composed_rgb_objects",
        include_unassigned=True,
    )
    split_rgb_objects = _export_individual_objects(
        paths.global_point_cloud,
        np.load(paths.point_labels),
        paths.splat_outputs / "split_rgb_objects",
        allowed_ids={int(value) for value in prepared["instanceIds"]},
    )
    point_cache = paths.splat_outputs / "composed_scene_points.npz"
    _write_point_cache(final_model, labels, point_cache)

    summary = {
        "schemaVersion": 1,
        "viewerSchemaVersion": 3,
        "stage": "splat_compose",
        "timestampUtc": now_utc(),
        "method": "Split&Splat collision-driven instance composition",
        "sourceSplitSummary": str(paths.stage_summary("split")),
        "instanceCount": len(prepared["instances"]),
        "roundCount": len(rounds),
        "residualNodeCount": len(nodes),
        "rounds": rounds,
        "settings": {
            "iterationsPerMerge": config.composition_iterations,
            "densification": False,
            "maskLossWeights": list(config.composition_mask_weights),
            "opacityReset": "released GaussianModel.reset_opacity before each merge",
            "supervision": (
                "foreground-balanced RGB and identity masks"
                if foreground_balanced
                else "released full-frame RGB and identity masks"
            ),
            "collisionMetric": (
                "paper Eq. 4 directional AABB containment; unordered pair score "
                "is max(C_ab, C_ba)"
            ),
        },
        "outputs": {
            "fullGaussians": str(final_model),
            "instanceIdGaussians": str(final_ids),
            "instanceLabels": str(final_labels),
            "pointsCache": str(point_cache),
            "individualObjects": individual_objects,
            "composedRgbObjects": composed_rgb_objects,
            "splitRgbObjects": split_rgb_objects,
        },
    }
    if post_composition_pruning is not None:
        summary["postCompositionPruning"] = post_composition_pruning
    _write_splat_experiment_registration(paths, editor_paths, summary)
    return summary


def _composition_storage_key(instance_ids: tuple[int, ...]) -> str:
    """Keep composition path components bounded while retaining stable identity."""
    joined = "_".join(str(value) for value in instance_ids)
    if len(joined) <= 96:
        return joined
    digest = hashlib.sha256(joined.encode("ascii")).hexdigest()[:12]
    return (
        f"n{len(instance_ids):04d}_"
        f"{instance_ids[0]:06d}_{instance_ids[-1]:06d}_{digest}"
    )


def _composition_model_paths(
    round_dir: Path,
    *,
    storage_key: str,
    iterations: int,
) -> tuple[Path, Path]:
    """Return the bounded model and point-cloud paths for one merge."""
    model_dir = round_dir / "models" / storage_key
    return model_dir, trained_point_cloud_path(model_dir, iterations)


def _composition_source_key(
    ply_path: Path,
    masks_dir: Path,
    *,
    instance_ids: tuple[int, ...],
) -> str:
    return _composition_digest(
        {
            "instanceIds": [int(value) for value in instance_ids],
            "pointCloud": _composition_file_identity(ply_path),
            "masks": _composition_mask_digest(masks_dir),
        }
    )


def _composition_cache_input(
    left: _CompositionNode,
    right: _CompositionNode,
    *,
    iterations: int,
    mask_weight: float,
    foreground_balanced: bool = False,
) -> dict[str, Any]:
    left_key = left.cache_key or _composition_source_key(
        left.ply_path,
        left.masks_dir,
        instance_ids=left.instance_ids,
    )
    right_key = right.cache_key or _composition_source_key(
        right.ply_path,
        right.masks_dir,
        instance_ids=right.instance_ids,
    )
    return {
        "protocol": (
            "anchored_foreground_balanced_composition_v1"
            if foreground_balanced
            else "released_split_splat_composition_no_densification_v1"
        ),
        "leftKey": left_key,
        "rightKey": right_key,
        "instanceIds": [
            int(value)
            for value in sorted((*left.instance_ids, *right.instance_ids))
        ],
        "iterations": int(iterations),
        "maskLossWeight": float(mask_weight),
    }


def _composition_cache_status(
    manifest_path: Path,
    output_ply: Path,
    expected_input: dict[str, Any],
    *,
    expected_point_count: int,
) -> tuple[bool, str]:
    if not output_ply.is_file():
        return False, "output_missing"
    if not manifest_path.is_file():
        return False, "manifest_missing"
    try:
        manifest = read_json(manifest_path)
    except Exception as error:
        return False, f"manifest_unreadable:{type(error).__name__}"
    if manifest.get("input") != expected_input:
        return False, "input_changed"
    if int(manifest.get("expectedPointCount", -1)) != int(expected_point_count):
        return False, "expected_count_changed"
    try:
        output_count = _ply_vertex_count(output_ply)
        output_identity = _composition_file_identity(output_ply)
    except Exception as error:
        return False, f"output_unreadable:{type(error).__name__}"
    if output_count != int(expected_point_count):
        return False, "output_count_mismatch"
    if manifest.get("output") != output_identity:
        return False, "output_changed"
    return True, "valid"


def _composition_file_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sizeBytes": int(stat.st_size),
        "modifiedTimeNs": int(stat.st_mtime_ns),
        "pointCount": _ply_vertex_count(path),
    }


def _composition_mask_digest(masks_dir: Path) -> dict[str, Any]:
    paths = sorted(path for path in masks_dir.glob("*.png") if path.is_file())
    if not paths:
        raise ValueError(f"Composition mask directory is empty: {masks_dir}")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "fileCount": len(paths),
        "sha256": digest.hexdigest(),
    }


def _composition_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def register_splat_artifacts(
    paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    summary: dict[str, Any],
) -> None:
    refreshed = refresh_splat_viewer_artifacts(paths, summary)
    atomic_write_json(paths.stage_summary("splat_compose"), refreshed)
    _write_splat_experiment_registration(paths, editor_paths, refreshed)


def refresh_splat_viewer_artifacts(
    paths: SplitSplatRunPaths,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Refresh display artifacts without repeating any Splat optimization."""
    outputs = dict(summary["outputs"])
    outputs.pop("rgbGaussians", None)
    required = [
        Path(str(outputs.get("instanceIdGaussians") or "")),
        paths.splat_outputs / "individual_objects",
        paths.splat_outputs / "composed_rgb_objects",
        paths.splat_outputs / "split_rgb_objects",
    ]
    if int(summary.get("viewerSchemaVersion", 0)) >= 3 and all(
        path.exists() for path in required
    ):
        return {**summary, "outputs": outputs}

    final_model = Path(outputs["fullGaussians"])
    labels = np.load(Path(outputs["instanceLabels"])).astype(np.int32, copy=False)
    final_ids = Path(outputs["instanceIdGaussians"])
    write_segmented_gaussian_ply(final_model, final_ids, labels)
    outputs["individualObjects"] = _export_individual_objects(
        final_model,
        labels,
        paths.splat_outputs / "individual_objects",
        include_unassigned=True,
        semantic_colors=True,
    )
    outputs["composedRgbObjects"] = _export_individual_objects(
        final_model,
        labels,
        paths.splat_outputs / "composed_rgb_objects",
        include_unassigned=True,
    )
    prepared = read_json(paths.stage_summary("splat_prepare"))
    outputs["splitRgbObjects"] = _export_individual_objects(
        paths.global_point_cloud,
        np.load(paths.point_labels),
        paths.splat_outputs / "split_rgb_objects",
        allowed_ids={int(value) for value in prepared["instanceIds"]},
    )
    return {
        **summary,
        "viewerSchemaVersion": 3,
        "outputs": outputs,
    }


def _ensure_shared_support(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
    *,
    progress: ProgressCallback | None,
) -> dict[str, Any]:
    output = paths.shared_splat_support_dir
    summary_path = output / "summary.json"
    if summary_path.is_file():
        return read_json(summary_path)
    output.mkdir(parents=True, exist_ok=True)
    replace_symlink(output / "images", paths.dataset_dir / "images")
    depth_dir = output / "depth_metric_mm"
    pose_dir = output / "pose"
    intrinsic_dir = output / "intrinsic"
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    intrinsic_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = read_jsonl(paths.frame_map)
    metric_depth_source = (
        paths.global_gs_dir / "split_depth_editor_depth"
        if config.depth_source == "editor_depth"
        else paths.dataset_dir / "depth"
    )
    if not metric_depth_source.is_dir():
        raise FileNotFoundError(
            "Splat mask refinement requires the depth visibility maps prepared "
            f"by Split: {metric_depth_source}"
        )
    image_rows = {
        Path(str(row["name"])).stem: row
        for row in _read_colmap_images_text(
            paths.dataset_dir / "sparse" / "0" / "images.txt"
        )
    }
    intrinsics = np.asarray(
        [
            [row["fx"], row["fy"], row["cx"], row["cy"]]
            for row in frame_rows
        ],
        dtype=np.float64,
    )
    max_intrinsic_delta = float(
        np.max(np.abs(intrinsics - intrinsics[:1]))
    )
    if max_intrinsic_delta > 1e-4:
        raise ValueError(
            "Released mask_optimizer_scannet.py accepts one shared intrinsic "
            f"matrix, but this dataset varies by {max_intrinsic_delta:.6g}."
        )
    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = intrinsics[0, 0]
    intrinsic[1, 1] = intrinsics[0, 1]
    intrinsic[0, 2] = intrinsics[0, 2]
    intrinsic[1, 2] = intrinsics[0, 3]
    np.savetxt(intrinsic_dir / "intrinsic_depth.txt", intrinsic, fmt="%.17g")

    clipped_depth_pixels = 0
    for index, row in enumerate(frame_rows):
        stem = Path(str(row["splitSplatImageName"])).stem
        image = image_rows[stem]
        metric = np.asarray(
            np.load(metric_depth_source / f"{stem}_pred.npy"),
            dtype=np.float32,
        )
        metric = np.where(
            np.isfinite(metric) & (metric > 0.0),
            metric,
            0.0,
        )
        clipped_depth_pixels += int(
            np.count_nonzero(metric * 1000.0 > np.iinfo(np.uint16).max)
        )
        millimeters = np.clip(
            np.rint(np.maximum(metric, 0.0) * 1000.0),
            0,
            np.iinfo(np.uint16).max,
        ).astype(np.uint16)
        Image.fromarray(millimeters).save(depth_dir / f"{stem}.png")
        pose = _world_from_camera(
            np.asarray(image["qvec"], dtype=np.float64),
            np.asarray(image["tvec"], dtype=np.float64),
        )
        np.savetxt(pose_dir / f"{stem}.txt", pose, fmt="%.17g")
        _progress_count(progress, "Prepared shared Splat camera support", index + 1, len(frame_rows))

    summary = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "frameCount": len(frame_rows),
        "imagesDir": str(output / "images"),
        "metricDepthMillimetersDir": str(depth_dir),
        "metricDepthSource": str(metric_depth_source),
        "poseDir": str(pose_dir),
        "intrinsicPath": str(intrinsic_dir / "intrinsic_depth.txt"),
        "maxIntrinsicDelta": max_intrinsic_delta,
        "clippedDepthPixels": clipped_depth_pixels,
        "depthEncoding": "uint16_millimeters_for_released_refiner",
        "sharedAcrossMaskRuns": True,
    }
    atomic_write_json(summary_path, summary)
    return summary


def _prepare_workspace(paths: SplitSplatRunPaths) -> None:
    """Restore the released scripts' relative data and output paths."""
    scene = paths.splat_scene_dir
    scene.mkdir(parents=True, exist_ok=True)
    paths.splat_initial_models.mkdir(parents=True, exist_ok=True)
    paths.splat_refined_models.mkdir(parents=True, exist_ok=True)
    support = paths.shared_splat_support_dir
    replace_symlink(scene / "images", support / "images")
    replace_symlink(scene / "depth", support / "depth_metric_mm")
    replace_symlink(scene / "pose", support / "pose")
    replace_symlink(scene / "intrinsic", support / "intrinsic")
    replace_symlink(scene / "masks", paths.splat_instances_dir)
    raw_link = paths.splat_workspace / "output" / paths.run_id / "raw"
    ref_link = paths.splat_workspace / "output" / paths.run_id / "ref"
    replace_symlink(raw_link, paths.splat_initial_models)
    replace_symlink(ref_link, paths.splat_refined_models)


def _prepare_mask_refinement_workspace(
    config: SplitSplatRunConfig,
    paths: SplitSplatRunPaths,
) -> None:
    """Add the released SAM2 checkpoint path required by mask refinement."""
    _prepare_workspace(paths)
    checkpoint_dir = config.split_splat_root / "checkpoints"
    checkpoint = checkpoint_dir / "sam2.1_hiera_large.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "Split&Splat mask refinement requires the released SAM2.1 "
            f"checkpoint: {checkpoint}"
        )
    replace_symlink(paths.splat_workspace / "checkpoints", checkpoint_dir)


def _prepare_composition_dataset(
    paths: SplitSplatRunPaths,
    dataset_dir: Path,
    left: _CompositionNode,
    right: _CompositionNode,
) -> None:
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    (dataset_dir / "masks").mkdir(parents=True)
    sparse_dir = dataset_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True)
    replace_symlink(dataset_dir / "images", paths.shared_splat_support_dir / "images")
    camera_source = paths.dataset_dir / "sparse" / "0"
    replace_symlink(sparse_dir / "cameras.txt", camera_source / "cameras.txt")
    replace_symlink(sparse_dir / "images.txt", camera_source / "images.txt")
    _concatenate_ply([left.ply_path, right.ply_path], sparse_dir / "points3D.ply")
    _compose_mask_directories(
        left.masks_dir,
        right.masks_dir,
        dataset_dir / "masks",
    )


def _cleanup_composition_training_artifacts(
    dataset_dir: Path,
    model_dir: Path,
    *,
    iterations: int,
) -> None:
    """Retain the resumable final model while dropping full-size duplicates."""
    for path in (
        dataset_dir / "sparse" / "0" / "points3D.ply",
        model_dir / "input.ply",
    ):
        path.unlink(missing_ok=True)
    for path in model_dir.glob("events.out.tfevents.*"):
        path.unlink(missing_ok=True)
    for path in model_dir.glob("chkpnt*.pth"):
        path.unlink(missing_ok=True)
    midpoint = max(int(iterations) // 2, 1)
    if midpoint != int(iterations):
        checkpoint = model_dir / "point_cloud" / f"iteration_{midpoint}"
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def _select_collision_pairs(
    nodes: list[_CompositionNode],
) -> list[tuple[_CompositionNode, _CompositionNode, float]]:
    candidates: list[tuple[float, str, str, _CompositionNode, _CompositionNode]] = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1 :]:
            score = _collision_score(left.ply_path, right.ply_path)
            if score > 0.0:
                candidates.append((-score, left.name, right.name, left, right))
    selected = []
    consumed: set[str] = set()
    for negative_score, _, _, left, right in sorted(candidates):
        if left.name in consumed or right.name in consumed:
            continue
        consumed.update((left.name, right.name))
        selected.append((left, right, -negative_score))
    return selected


def _collision_score(first: Path, second: Path) -> float:
    return max(
        _directional_collision_score(first, second),
        _directional_collision_score(second, first),
    )


def _directional_collision_score(source: Path, container: Path) -> float:
    """Paper Eq. 4: fraction of source Gaussians inside container's AABB."""
    source_xyz = _xyz(_read_ply_vertex(source))
    container_xyz = _xyz(_read_ply_vertex(container))
    if source_xyz.shape[0] == 0 or container_xyz.shape[0] == 0:
        return 0.0
    inside = np.all(
        (source_xyz >= np.min(container_xyz, axis=0))
        & (source_xyz <= np.max(container_xyz, axis=0)),
        axis=1,
    )
    return float(np.count_nonzero(inside) / source_xyz.shape[0])


def _compose_mask_directories(first: Path, second: Path, output: Path) -> None:
    names = sorted({path.name for path in first.glob("*.png")} | {path.name for path in second.glob("*.png")})
    for name in names:
        first_path = first / name
        second_path = second / name
        first_image = Image.open(first_path).convert("RGBA") if first_path.is_file() else None
        second_image = Image.open(second_path).convert("RGBA") if second_path.is_file() else None
        if first_image is None:
            assert second_image is not None
            second_image.save(output / name)
            continue
        if second_image is None:
            first_image.save(output / name)
            continue
        first_area = int(np.count_nonzero(np.asarray(first_image.getchannel("A"))))
        second_area = int(np.count_nonzero(np.asarray(second_image.getchannel("A"))))
        background = Image.new("RGBA", first_image.size, (0, 0, 0, 0))
        larger, smaller = (
            (first_image, second_image)
            if first_area >= second_area
            else (second_image, first_image)
        )
        Image.alpha_composite(
            Image.alpha_composite(background, larger),
            smaller,
        ).save(output / name)


def _merge_refined_masks(dataset_dir: Path) -> int:
    source = dataset_dir / "masks_extra"
    if not source.is_dir():
        return 0
    target = dataset_dir / "masks"
    count = 0
    for path in source.glob("*.png"):
        shutil.move(str(path), str(target / path.name))
        count += 1
    shutil.rmtree(source)
    return count


def _training_summary(
    stage: str,
    method: str,
    config: SplitSplatRunConfig,
    rows: list[dict[str, Any]],
    *,
    initialPass: bool,
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "stage": stage,
        "timestampUtc": now_utc(),
        "method": method,
        "instanceCount": len(rows),
        "instances": rows,
        "settings": {
            "iterations": config.instance_iterations,
            "maskLossWeight": 0.25,
            "initialMaskDilation": bool(initialPass),
            "densification": "released defaults",
        },
    }


def _write_splat_experiment_registration(
    paths: SplitSplatRunPaths,
    editor_paths: EditorPaths,
    summary: dict[str, Any],
) -> None:
    outputs = summary["outputs"]
    appearance_source = Path(outputs["fullGaussians"])
    appearance_viewer = (
        appearance_source.parent / "viewer" / appearance_source.name
    )
    ensure_viewer_gaussian_ply(appearance_source, appearance_viewer)
    diagnostics_path = appearance_source.parent / "appearance_diagnostics.json"
    atomic_write_json(
        diagnostics_path,
        {
            "schemaVersion": 1,
            "composedScene": summarize_gaussian_ply(appearance_source),
        },
    )
    gaussian_artifacts = {
        "appearance": {
            "displayName": "Composed RGB 3DGS",
            "path": str(appearance_viewer),
            "format": "3dgs_ply",
            "pointCount": _ply_vertex_count(Path(outputs["fullGaussians"])),
            "labelCount": int(summary["instanceCount"]),
            "artifactGroup": "composed_scene",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        },
        "instance_ids": {
            "displayName": "Composed Instance Geometry",
            "path": outputs["instanceIdGaussians"],
            "format": "3dgs_ply",
            "pointCount": _ply_vertex_count(Path(outputs["instanceIdGaussians"])),
            "labelCount": int(summary["instanceCount"]),
            "artifactGroup": "composed_scene",
            "colorSpace": "split_splat_instance",
            "radiometricAppearance": False,
        },
    }
    for row in outputs.get("individualObjects", []):
        instance_id = int(row["instanceId"])
        object_name = "Unassigned" if instance_id == 0 else f"Composed Object {instance_id}"
        gaussian_artifacts[f"composed_object_{instance_id:06d}"] = {
            "displayName": object_name,
            "path": row["path"],
            "format": "3dgs_ply",
            "pointCount": int(row["pointCount"]),
            "instanceId": instance_id,
            "artifactGroup": "composed_objects",
            "colorSpace": "split_splat_instance",
            "radiometricAppearance": False,
        }
    for row in outputs.get("composedRgbObjects", []):
        instance_id = int(row["instanceId"])
        object_name = "Unassigned RGB" if instance_id == 0 else f"RGB Region {instance_id}"
        gaussian_artifacts[f"composed_rgb_object_{instance_id:06d}"] = {
            "displayName": object_name,
            "path": row["path"],
            "format": "3dgs_ply",
            "pointCount": int(row["pointCount"]),
            "instanceId": instance_id,
            "artifactGroup": "composed_rgb_objects",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    for row in outputs.get("splitRgbObjects", []):
        instance_id = int(row["instanceId"])
        gaussian_artifacts[f"rgb_object_{instance_id:06d}"] = {
            "displayName": f"RGB Object {instance_id}",
            "path": row["path"],
            "format": "3dgs_ply",
            "pointCount": int(row["pointCount"]),
            "instanceId": instance_id,
            "artifactGroup": "rgb_objects",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    run_dir = (
        editor_paths.segmentation3d_dir
        / "runs"
        / f"{paths.run_id}_splat"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "experiment.json",
        {
            "schemaVersion": 1,
            "runId": f"{paths.run_id}_splat",
            "methodId": "split_splat_splat",
            "experimentFamily": "split_splat",
            "artifactRole": "reconstruction",
            "baseRunId": paths.run_id,
            "reconstructionVariant": "released_splat",
            "inputId": paths.run_id,
            "displayName": f"Split&Splat Splat: {paths.run_id}",
            "labelSpace": "split_splat_instance",
            "pointsCachePath": outputs["pointsCache"],
            "geometrySource": "split_splat_composed_instances",
            "ready": True,
            "timestampUtc": now_utc(),
            "canonicalRunDir": str(paths.run_dir),
            "gaussianArtifacts": gaussian_artifacts,
            "appearanceDiagnostics": str(diagnostics_path),
        },
    )


def _export_individual_objects(
    source: Path,
    labels: np.ndarray,
    output_dir: Path,
    *,
    allowed_ids: set[int] | None = None,
    include_unassigned: bool = False,
    semantic_colors: bool = False,
) -> list[dict[str, Any]]:
    rows = write_viewer_gaussian_partitions(
        source,
        output_dir,
        labels,
        allowed_ids=allowed_ids,
        include_unassigned=include_unassigned,
        semantic_colors=semantic_colors,
    )
    source_name = (
        "post_composition_instance_geometry"
        if semantic_colors
        else "shared_global_rgb_gaussians"
    )
    for row in rows:
        row["source"] = source_name
    return rows


def _write_point_cache(ply_path: Path, labels: np.ndarray, output: Path) -> None:
    vertex = _read_ply_vertex(ply_path)
    np.savez_compressed(
        output,
        points=_xyz(vertex).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int32),
    )


def _concatenate_nodes(nodes: list[_CompositionNode], output: Path) -> None:
    _concatenate_ply([node.ply_path for node in nodes], output)


def _concatenate_ply(paths: list[Path], output: Path) -> None:
    from plyfile import PlyData, PlyElement

    if not paths:
        raise ValueError("Cannot compose an empty Gaussian list.")
    first = PlyData.read(paths[0])
    vertices = [first["vertex"].data]
    dtype = vertices[0].dtype
    for path in paths[1:]:
        vertex = PlyData.read(path)["vertex"].data
        if vertex.dtype != dtype:
            raise ValueError(f"Gaussian PLY schemas differ: {paths[0]} and {path}")
        vertices.append(vertex)
    output.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(np.concatenate(vertices), "vertex")],
        text=first.text,
        byte_order=first.byte_order,
    ).write(output)


def _read_ply_vertex(path: Path) -> np.ndarray:
    from plyfile import PlyData

    if not path.is_file():
        raise FileNotFoundError(path)
    return PlyData.read(path, mmap="r")["vertex"].data


def _xyz(vertex: np.ndarray) -> np.ndarray:
    return np.column_stack([vertex["x"], vertex["y"], vertex["z"]])


def _ply_vertex_count(path: Path) -> int:
    return int(_read_ply_vertex(path).shape[0])



def _mask_has_foreground(path: Path) -> bool:
    return bool(np.any(np.asarray(Image.open(path).convert("L")) > 0))


def _progress_count(
    progress: ProgressCallback | None,
    label: str,
    completed: int,
    total: int,
) -> None:
    if progress is not None and (
        completed == 1 or completed % 10 == 0 or completed == total
    ):
        progress(f"{label} {completed}/{total}.")
