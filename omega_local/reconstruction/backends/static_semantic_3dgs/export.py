"""Normalize native static 3DGS outputs into one comparable artifact schema."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_complete_stage,
    read_json,
)

from .contract import StaticSemantic3DGSConfig, StaticSemantic3DGSPaths


ProgressCallback = Callable[[str], None]


def export_static_semantic_3dgs(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    train_summary = read_complete_stage(paths.stage_summary("train"))
    if train_summary is None:
        raise RuntimeError(
            f"Static semantic 3DGS training is not complete: {paths.stage_summary('train')}"
        )
    native_ply = Path(train_summary["outputs"]["nativeScenePly"])
    if not native_ply.is_file():
        raise FileNotFoundError(f"Native trained scene is missing: {native_ply}")
    region_map = read_json(paths.shared_region_map)
    native = PlyData.read(native_ply)
    vertices = native["vertex"].data

    if config.method == "segment_then_splat":
        native_identity = Path(train_summary["outputs"]["hardObjectIds"])
        region_ids = _segment_then_splat_region_ids(native_identity, region_map)
    elif config.method == "gaussian_grouping":
        native_identity = Path(train_summary["outputs"]["classifier"])
        region_ids = _gaussian_grouping_region_ids(
            vertices,
            native_identity,
            region_map,
        )
    else:
        raise AssertionError(config.method)
    if region_ids.shape != (vertices.shape[0],):
        raise ValueError(
            f"Identity count {region_ids.shape} does not match Gaussian count "
            f"{vertices.shape[0]}."
        )

    if paths.outputs_dir.exists():
        shutil.rmtree(paths.outputs_dir)
    paths.object_splats_dir.mkdir(parents=True)
    standard_vertices = _standard_gaussian_vertices(vertices)
    _write_vertices(paths.scene_ply, standard_vertices)
    np.save(paths.gaussian_region_ids, region_ids.astype(np.int32))

    metadata = {
        int(row["persistentRegionId"]): dict(row)
        for row in region_map["regions"]
    }
    all_region_ids = [0, *sorted(metadata)]
    region_rows: list[dict[str, Any]] = []
    for index, region_id in enumerate(all_region_ids):
        mask = region_ids == region_id
        count = int(np.count_nonzero(mask))
        output_path: Path | None = None
        if count:
            output_path = paths.object_splats_dir / f"region_{region_id:06d}.ply"
            _write_vertices(output_path, standard_vertices[mask])
        source = metadata.get(region_id, {})
        region_rows.append(
            {
                "persistentRegionId": region_id,
                "name": (
                    "Background / Unassigned"
                    if region_id == 0
                    else source.get("name", f"Region {region_id}")
                ),
                "color": source.get("color"),
                "gaussianCount": count,
                "ply": str(output_path) if output_path is not None else None,
                "role": "background_or_unassigned" if region_id == 0 else "region",
            }
        )
        if progress is not None:
            progress(
                f"Exported static 3DGS region {index + 1}/{len(all_region_ids)}."
            )

    regions_payload = {
        "schemaVersion": 1,
        "labelNamespace": "persistent_region",
        "regions": region_rows,
    }
    atomic_write_json(paths.regions_manifest, regions_payload)
    manifest = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "method": config.display_name,
        "identityModel": (
            "hard_immutable"
            if config.method == "segment_then_splat"
            else "soft_learned_then_argmax"
        ),
        "gaussianCount": int(vertices.shape[0]),
        "foregroundGaussianCount": int(np.count_nonzero(region_ids)),
        "backgroundGaussianCount": int(np.count_nonzero(region_ids == 0)),
        "scenePly": str(paths.scene_ply),
        "gaussianRegionIds": str(paths.gaussian_region_ids),
        "regionsManifest": str(paths.regions_manifest),
        "objectSplatsDir": str(paths.object_splats_dir),
        "nativeScenePly": str(native_ply),
        "nativeIdentityArtifact": str(native_identity),
        "static3DGS": True,
        "labelZeroMeaning": "background_or_unassigned",
    }
    atomic_write_json(paths.output_manifest, manifest)
    return {
        "schemaVersion": 1,
        "stage": "export",
        "timestampUtc": now_utc(),
        "method": config.display_name,
        "outputs": manifest,
        "regions": region_rows,
    }


def _segment_then_splat_region_ids(
    identity_path: Path,
    region_map: dict[str, Any],
) -> np.ndarray:
    hard_ids = torch.load(
        identity_path,
        map_location="cpu",
        weights_only=True,
    )
    hard_ids = np.asarray(hard_ids.detach().cpu(), dtype=np.int64).reshape(-1)
    hard_to_persistent = {
        int(row["hardObjectId"]): int(row["persistentRegionId"])
        for row in region_map["regions"]
    }
    result = np.zeros(hard_ids.shape, dtype=np.int32)
    for hard_id, persistent_id in hard_to_persistent.items():
        result[hard_ids == hard_id] = persistent_id
    unknown = sorted(
        int(value)
        for value in np.unique(hard_ids)
        if int(value) != 255 and int(value) not in hard_to_persistent
    )
    if unknown:
        raise ValueError(f"Segment then Splat emitted unknown hard IDs: {unknown}")
    return result


def _gaussian_grouping_region_ids(
    vertices: np.ndarray,
    classifier_path: Path,
    region_map: dict[str, Any],
) -> np.ndarray:
    feature_names = sorted(
        (name for name in vertices.dtype.names or () if name.startswith("obj_dc_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if not feature_names:
        raise ValueError("Gaussian Grouping PLY contains no obj_dc identity features.")
    state = torch.load(
        classifier_path,
        map_location="cpu",
        weights_only=True,
    )
    weight = np.asarray(state["weight"].detach().cpu(), dtype=np.float32)
    bias = np.asarray(state["bias"].detach().cpu(), dtype=np.float32)
    weight = weight.reshape(weight.shape[0], weight.shape[1])
    features = np.column_stack(
        [np.asarray(vertices[name], dtype=np.float32) for name in feature_names]
    )
    if features.shape[1] != weight.shape[1]:
        raise ValueError(
            f"Gaussian Grouping feature width {features.shape[1]} does not "
            f"match classifier width {weight.shape[1]}."
        )
    class_to_persistent = {
        int(row["classId"]): int(row["persistentRegionId"])
        for row in region_map["regions"]
    }
    result = np.zeros(features.shape[0], dtype=np.int32)
    chunk_size = 250_000
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        logits = features[start:stop] @ weight.T + bias
        class_ids = np.argmax(logits, axis=1)
        chunk_labels = np.zeros(stop - start, dtype=np.int32)
        for class_id, persistent_id in class_to_persistent.items():
            chunk_labels[class_ids == class_id] = persistent_id
        result[start:stop] = chunk_labels
    return result


def _standard_gaussian_vertices(vertices: np.ndarray) -> np.ndarray:
    names = [
        name
        for name in vertices.dtype.names or ()
        if not name.startswith("obj_dc_")
    ]
    if not names:
        raise ValueError("Trained PLY has no standard Gaussian properties.")
    output = np.empty(
        vertices.shape[0],
        dtype=[(name, vertices.dtype.fields[name][0]) for name in names],
    )
    for name in names:
        output[name] = vertices[name]
    return output


def _write_vertices(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices, "vertex")],
        text=False,
    ).write(path)
