"""Prepare per-region datasets from an anchored Split partition."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from omega_local.segmentation.split_splat.contract import (
    now_utc,
    read_json,
    replace_symlink,
)
from omega_local.segmentation.split_splat.splat import (
    _ensure_shared_support,
    _prepare_workspace,
)

from .contract import AnchoredSplatConfig, AnchoredSplatPaths


ProgressCallback = Callable[[str], None]
_SH_C0 = np.float32(0.28209479177387814)


def prepare_regions(
    config: AnchoredSplatConfig,
    runtime_config: Any,
    paths: AnchoredSplatPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Slice full Gaussian state and stage source masks without mutation."""
    split_summary = read_json(paths.source.stage_summary("split"))
    if str(split_summary.get("splitMethod")) != "anchored_3d":
        raise ValueError(
            "Anchored Splat requires an anchored_3d Split result; got "
            f"{split_summary.get('splitMethod')!r}."
        )
    if str(split_summary.get("labelNamespace")) != "persistent_region":
        raise ValueError(
            "Anchored Splat requires persistent-region labels; got "
            f"{split_summary.get('labelNamespace')!r}."
        )

    labels = np.asarray(
        np.load(paths.source.point_labels, allow_pickle=False),
        dtype=np.int64,
    )
    confidence = _optional_array(
        paths.source.point_label_confidence,
        labels.shape,
        np.float32,
        fill=0.0,
    )
    provenance = _optional_array(
        paths.source.point_label_provenance,
        labels.shape,
        np.uint8,
        fill=0,
    )
    global_ply = PlyData.read(paths.global_point_cloud, mmap="r")
    vertices = global_ply["vertex"].data
    if int(vertices.shape[0]) != int(labels.shape[0]):
        raise ValueError(
            "Anchored point labels are not aligned with the shared global "
            f"Gaussians: labels={labels.shape[0]}, points={vertices.shape[0]}."
        )
    _require_gaussian_fields(vertices)
    unassigned_count = int(np.count_nonzero(labels <= 0))
    if unassigned_count:
        raise ValueError(
            "Anchored Splat requires a complete global-Gaussian partition; "
            f"{unassigned_count:,} Gaussians are unassigned."
        )

    support = _ensure_shared_support(
        runtime_config,
        paths,
        progress=progress,
    )
    output = paths.splat_instances_dir
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _prepare_workspace(paths)

    weights_source = Path(
        str(split_summary.get("trainingViewWeightsPath") or "")
    )
    if not weights_source.is_file():
        weights_source = paths.source.training_view_weights
    if not weights_source.is_file():
        raise FileNotFoundError(
            "Anchored Split training weights are missing: "
            f"{weights_source}"
        )

    camera_source = paths.dataset_dir / "sparse" / "0"
    mask_ids = [
        int(value)
        for value in split_summary.get(
            "maskInstanceIds",
            split_summary.get("sourcePersistentRegionIds", []),
        )
    ]
    if not mask_ids:
        raise ValueError("Anchored Split exported no persistent mask IDs.")
    ownership_ids = {
        int(value)
        for value in np.unique(labels)
        if int(value) > 0
    }
    mask_id_set = set(mask_ids)
    if ownership_ids != mask_id_set:
        raise ValueError(
            "Anchored mask and Gaussian ownership IDs differ: "
            f"mask_only={sorted(mask_id_set - ownership_ids)}, "
            f"ownership_only={sorted(ownership_ids - mask_id_set)}."
        )

    kept: list[dict[str, Any]] = []
    for index, region_id in enumerate(mask_ids):
        selected = labels == region_id
        gaussian_count = int(np.count_nonzero(selected))
        source_masks = paths.source.split_raw_output / str(region_id)
        masks = sorted(
            path
            for path in source_masks.glob("*.png")
            if _mask_has_foreground(path)
        )
        reason = None
        if gaussian_count < config.minimum_gaussians:
            reason = "missing_or_too_small_gaussian_initializer"
        elif len(masks) < config.minimum_positive_views:
            reason = "fewer_than_minimum_positive_views"
        if reason is not None:
            raise ValueError(
                f"Persistent region {region_id} cannot be silently omitted: "
                f"{reason} (Gaussians={gaussian_count}, "
                f"positive views={len(masks)})."
            )

        dataset_dir = output / str(region_id)
        mask_dir = dataset_dir / "masks"
        sparse_dir = dataset_dir / "sparse" / "0"
        initializer_dir = dataset_dir / "initializer"
        mask_dir.mkdir(parents=True)
        sparse_dir.mkdir(parents=True)
        initializer_dir.mkdir(parents=True)
        replace_symlink(
            dataset_dir / "images",
            paths.shared_splat_support_dir / "images",
        )
        replace_symlink(
            sparse_dir / "cameras.txt",
            camera_source / "cameras.txt",
        )
        replace_symlink(
            sparse_dir / "images.txt",
            camera_source / "images.txt",
        )
        for mask_path in masks:
            shutil.copy2(mask_path, mask_dir / mask_path.name)
        replace_symlink(
            dataset_dir / "training_view_weights.json",
            weights_source,
        )

        subset = np.array(vertices[selected], copy=True)
        full_initializer = initializer_dir / "gaussians.ply"
        _write_full_gaussians(
            global_ply,
            subset,
            full_initializer,
        )
        point_initializer = sparse_dir / "points3D.ply"
        _write_rgb_points(subset, point_initializer)
        np.save(
            initializer_dir / "label_confidence.npy",
            confidence[selected],
        )
        np.save(
            initializer_dir / "label_provenance.npy",
            provenance[selected],
        )

        kept.append(
            {
                "instanceId": region_id,
                "regionId": region_id,
                "datasetDir": str(dataset_dir),
                "pointCloud": str(point_initializer),
                "initializerGaussians": str(full_initializer),
                "gaussianCount": gaussian_count,
                "maskCount": len(masks),
                "trainingViewWeights": str(
                    dataset_dir / "training_view_weights.json"
                ),
                "sourceLabelConfidence": str(
                    initializer_dir / "label_confidence.npy"
                ),
                "sourceLabelProvenance": str(
                    initializer_dir / "label_provenance.npy"
                ),
            }
        )
        _report(
            progress,
            f"Prepared anchored region {index + 1}/{len(mask_ids)} "
            f"(ID {region_id}, {gaussian_count} Gaussians).",
        )

    if not kept:
        raise ValueError("No anchored region can be reconstructed.")
    prepared_gaussian_count = sum(
        int(row["gaussianCount"])
        for row in kept
    )
    if prepared_gaussian_count != int(vertices.shape[0]):
        raise RuntimeError(
            "Per-region initializers do not conserve the shared global 3DGS: "
            f"prepared={prepared_gaussian_count}, global={vertices.shape[0]}."
        )
    return {
        "schemaVersion": 1,
        "stage": "prepare",
        "timestampUtc": now_utc(),
        "method": "Anchored full-Gaussian region dataset preparation",
        "sourceSplitSummary": str(paths.source.stage_summary("split")),
        "sourceGlobalGaussians": str(paths.global_point_cloud),
        "sharedSupport": support,
        "instanceCount": len(kept),
        "globalGaussianCount": int(vertices.shape[0]),
        "preparedGaussianCount": int(prepared_gaussian_count),
        "discardedInstanceCount": 0,
        "instanceIds": [row["instanceId"] for row in kept],
        "instances": kept,
        "discarded": [],
        "settings": {
            "minimumGaussians": config.minimum_gaussians,
            "minimumPositiveViews": config.minimum_positive_views,
            "initializer": "exact full-attribute global Gaussian subset",
            "initializerPartitionCoverage": 1.0,
            "unassignedGaussianCount": 0,
            "regionPolicy": "every persistent mask region must be reconstructed",
            "maskPolicy": "anchored source masks copied into isolated datasets",
            "sourceMutation": False,
            "trainingViewWeights": str(weights_source.resolve()),
        },
    }


def _optional_array(
    path: Path,
    shape: tuple[int, ...],
    dtype: Any,
    *,
    fill: float | int,
) -> np.ndarray:
    if not path.is_file():
        return np.full(shape, fill, dtype=dtype)
    values = np.asarray(np.load(path, allow_pickle=False), dtype=dtype)
    if values.shape != shape:
        raise ValueError(
            f"Expected {path} to have shape {shape}, got {values.shape}."
        )
    return values


def _require_gaussian_fields(vertices: np.ndarray) -> None:
    names = set(vertices.dtype.names or ())
    required = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
        "id_0",
        "id_1",
        "id_2",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(
            "Global Gaussian PLY is missing required attributes: "
            f"{missing}"
        )
    if not any(name.startswith("desc_") for name in names):
        raise ValueError(
            "Global Gaussian PLY has no descriptor fields required by the "
            "released Split&Splat Gaussian loader."
        )


def _write_full_gaussians(
    source: PlyData,
    vertices: np.ndarray,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=source.text,
        byte_order=getattr(source, "byte_order", "<"),
        comments=list(source.comments),
        obj_info=list(source.obj_info),
    ).write(destination)


def _write_rgb_points(vertices: np.ndarray, destination: Path) -> None:
    dc = np.column_stack(
        [vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]
    ).astype(np.float32, copy=False)
    rgb = np.rint(np.clip(_SH_C0 * dc + 0.5, 0.0, 1.0) * 255.0).astype(
        np.uint8
    )
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
    output = np.zeros(vertices.shape[0], dtype=dtype)
    output["x"] = vertices["x"]
    output["y"] = vertices["y"]
    output["z"] = vertices["z"]
    output["red"], output["green"], output["blue"] = rgb.T
    destination.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(output, "vertex")],
        text=False,
        byte_order="<",
    ).write(destination)


def _mask_has_foreground(path: Path) -> bool:
    alpha = np.asarray(Image.open(path).convert("RGBA").getchannel("A"))
    return bool(np.any(alpha > 0))
def _report(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)
