"""Train and export the released multilevel static semantic 3DGS."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from omega_local.reconstruction.gaussian_io import ensure_viewer_gaussian_ply
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
)

from .contract import LEVELS, OfficialSegmentThenSplatConfig, OfficialSegmentThenSplatPaths
from .execution import run_command


Progress = Callable[[str], None]
_ID_ARTIFACTS = {
    "large": "default_object_id_{iteration}.pth",
    "middle": "middle_object_id_{iteration}.pth",
    "small": "small_object_id_{iteration}.pth",
}


def train_paper_model(
    config: OfficialSegmentThenSplatConfig,
    paths: OfficialSegmentThenSplatPaths,
    *,
    progress: Progress | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    validate_paper_runtime(config)
    command = [
        str(config.python),
        "-u",
        str(config.source_root / "train.py"),
        "-s",
        str(paths.scene_dir),
        "-m",
        str(paths.model_output),
        "--eval",
        "--iterations",
        str(config.iterations),
        "--resolution",
        "1",
        "--num_sample_objects",
        str(config.num_sample_objects),
        "--densify_until_iter",
        str(config.densify_until_iter),
        "--partial_mask_iou",
        str(config.partial_mask_iou),
        "--load2gpu_on_the_fly",
        "--save_iterations",
        str(config.iterations),
        "--test_iterations",
        str(config.iterations),
    ]
    result = run_command(
        command,
        cwd=config.source_root,
        log_path=paths.model_stage / "logs" / "train.log",
        environment=_training_environment(config),
        progress=progress,
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "schemaVersion": 1,
            "stage": "train",
            "status": "planned",
            "command": command,
        }
    final_ply = (
        paths.model_output
        / "point_cloud"
        / f"iteration_{config.iterations}"
        / "point_cloud.ply"
    )
    if not final_ply.is_file():
        raise FileNotFoundError(f"Paper training did not produce {final_ply}")
    identities = {}
    for level, template in _ID_ARTIFACTS.items():
        path = paths.model_output / template.format(iteration=config.iterations)
        if not path.is_file():
            raise FileNotFoundError(f"Paper {level} object IDs are missing: {path}")
        identities[level] = str(path)
    summary = {
        "schemaVersion": 1,
        "stage": "train",
        "timestampUtc": now_utc(),
        "method": "Released Segment then Splat joint multilevel 3DGS",
        "iterations": int(config.iterations),
        "densifyUntilIteration": int(config.densify_until_iter),
        "stage1Iterations": 0,
        "stage2Iterations": 5000,
        "partialMaskStartIteration": 30000,
        "partialMaskIoU": float(config.partial_mask_iou),
        "nativeScenePly": str(final_ply),
        "identityArtifacts": identities,
        "command": result["command"],
        "log": str(paths.model_stage / "logs" / "train.log"),
    }
    atomic_write_json(paths.stage_summary("train"), summary)
    return summary


def export_paper_model(
    config: OfficialSegmentThenSplatConfig,
    paths: OfficialSegmentThenSplatPaths,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    train_summary = _read_json(paths.stage_summary("train"))
    native_ply = Path(str(train_summary["nativeScenePly"]))
    if paths.outputs_stage.exists():
        shutil.rmtree(paths.outputs_stage)
    paths.object_splats.mkdir(parents=True, exist_ok=True)
    ensure_viewer_gaussian_ply(native_ply, paths.scene_ply)
    vertices = PlyData.read(paths.scene_ply)["vertex"].data
    gaussian_count = int(vertices.shape[0])
    level_rows: dict[str, Any] = {}

    for level in LEVELS:
        ids_path = Path(str(train_summary["identityArtifacts"][level]))
        values = torch.load(
            ids_path,
            map_location="cpu",
            weights_only=True,
        )
        identities = np.asarray(values.detach().cpu(), dtype=np.int64).reshape(-1)
        if identities.shape != (gaussian_count,):
            raise ValueError(
                f"{level} identity count {identities.shape} does not match "
                f"{gaussian_count} Gaussians."
            )
        level_dir = paths.object_splats / level
        level_dir.mkdir(parents=True)
        np.save(paths.outputs_stage / f"ids_{level}.npy", identities)
        parts = []
        native_ids = [255, *sorted(int(value) for value in np.unique(identities) if int(value) != 255)]
        for index, native_id in enumerate(native_ids):
            mask = identities == native_id
            count = int(np.count_nonzero(mask))
            if not count:
                continue
            region_id = 0 if native_id == 255 else native_id + 1
            output = level_dir / f"region_{region_id:06d}.ply"
            _write_vertices(output, vertices[mask])
            parts.append(
                {
                    "regionId": int(region_id),
                    "nativeObjectId": int(native_id),
                    "name": (
                        "Background / Unassigned"
                        if native_id == 255
                        else f"{level.title()} Object {region_id}"
                    ),
                    "gaussianCount": count,
                    "ply": str(output),
                }
            )
            if progress is not None and (
                index == 0 or (index + 1) % 25 == 0 or index + 1 == len(native_ids)
            ):
                progress(
                    f"Exported paper {level} objects {index + 1}/{len(native_ids)}."
                )
        if sum(int(row["gaussianCount"]) for row in parts) != gaussian_count:
            raise RuntimeError(f"Paper {level} object partition is incomplete.")
        level_rows[level] = {
            "objectCount": int(sum(int(row["regionId"]) > 0 for row in parts)),
            "assignedGaussianCount": int(np.count_nonzero(identities != 255)),
            "backgroundGaussianCount": int(np.count_nonzero(identities == 255)),
            "identityArtifact": str(ids_path),
            "objectsDir": str(level_dir),
            "objects": parts,
        }

    manifest = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "method": "Segment then Splat Paper Pipeline",
        "experimentFamily": "segment_then_splat_paper",
        "runId": config.run_id,
        "scenePly": str(paths.scene_ply),
        "gaussianCount": gaussian_count,
        "levels": level_rows,
        "nativeScenePly": str(native_ply),
        "static3DGS": True,
    }
    atomic_write_json(paths.output_manifest, manifest)
    summary = {
        "schemaVersion": 1,
        "stage": "export",
        "timestampUtc": now_utc(),
        "status": "complete",
        "outputs": manifest,
    }
    atomic_write_json(paths.stage_summary("export"), summary)
    return summary


def validate_paper_runtime(config: OfficialSegmentThenSplatConfig) -> dict[str, Any]:
    required = [
        config.python,
        config.source_root / "train.py",
        config.native_extensions_dir,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Segment then Splat runtime is missing: " + ", ".join(missing))
    probe = (
        "import inspect, diff_gaussian_rasterization as d; "
        "print(inspect.signature(d.GaussianRasterizer.forward))"
    )
    process = subprocess.run(
        [str(config.python), "-c", probe],
        cwd=str(config.source_root),
        env=_training_environment(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0 or "means2D_densify" not in process.stdout:
        raise RuntimeError(
            "Segment then Splat CUDA rasterizer is unavailable. Run "
            "scripts/setup_segment_then_splat_runtime.sh.\n"
            + process.stderr.strip()
        )
    return {"rasterizerSignature": process.stdout.strip()}


def _training_environment(
    config: OfficialSegmentThenSplatConfig,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    prior = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        str(config.native_extensions_dir)
        if not prior
        else f"{config.native_extensions_dir}{os.pathsep}{prior}"
    )
    return environment


def _write_vertices(path: Path, vertices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))

