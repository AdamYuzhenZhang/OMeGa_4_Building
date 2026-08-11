"""Official ObjectGS per-object mesh export and editor registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.interactive.paths import EditorPaths
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_json,
)

from .contract import ObjectGSConfig, ObjectGSPaths
from .runner import _run_command


ProgressCallback = Callable[[str], None]


def export_object_meshes(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
    editor_paths: EditorPaths,
    *,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    pointer = read_json(paths.model_pointer)
    model_dir = Path(str(pointer.get("modelDir") or ""))
    if not model_dir.is_dir():
        raise FileNotFoundError(f"ObjectGS model is unavailable: {model_dir}")
    worker = Path(__file__).resolve().parent / "compat" / "export_object_meshes.py"
    command = [
        str(config.python),
        "-u",
        str(worker),
        "--objectgs-root",
        str(config.objectgs_root),
        "--model-dir",
        str(model_dir),
        "--region-map",
        str(paths.region_map),
        "--output-dir",
        str(paths.meshes_dir),
        "--mesh-voxel-size",
        str(config.mesh_voxel_size),
        "--mesh-resolution",
        str(config.mesh_resolution),
        "--mesh-clusters",
        str(config.mesh_clusters),
        "--mesh-max-triangles",
        str(config.mesh_max_triangles),
    ]
    result = _run_command(
        command,
        cwd=config.objectgs_root,
        log_path=paths.logs_dir / "mesh.log",
        compat_root=Path(__file__).resolve().parent / "compat",
        objectgs_root=config.objectgs_root,
        progress=progress,
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "schemaVersion": 1,
            "stage": "mesh",
            "status": "planned",
            "command": command,
            "meshesDir": str(paths.meshes_dir),
        }
    manifest = read_json(paths.mesh_manifest)
    rows = [
        row
        for row in manifest.get("regions", [])
        if isinstance(row, dict)
    ]
    renderable = [row for row in rows if str(row.get("glb") or "")]
    if not renderable:
        raise RuntimeError("ObjectGS mesh export produced no renderable object meshes.")
    _register_mesh_artifact(config, paths, editor_paths, renderable)
    return {
        "schemaVersion": 1,
        "stage": "mesh",
        "timestampUtc": now_utc(),
        "method": "Released ObjectGS object-depth rendering and bounded TSDF",
        "modelDir": str(model_dir),
        "meshManifest": str(paths.mesh_manifest),
        "meshVoxelSize": float(config.mesh_voxel_size),
        "meshResolution": int(config.mesh_resolution),
        "meshClusters": int(config.mesh_clusters),
        "meshMaxTriangles": int(config.mesh_max_triangles),
        "regionCount": len(rows),
        "renderableRegionCount": len(renderable),
        "vertexCount": sum(int(row.get("vertexCount", 0)) for row in rows),
        "triangleCount": sum(int(row.get("triangleCount", 0)) for row in rows),
        "command": result["command"],
    }


def _register_mesh_artifact(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
    editor_paths: EditorPaths,
    rows: list[dict[str, Any]],
) -> None:
    run_id = f"{config.mapanything_run_id}_{config.run_id}_anchors"
    experiment = editor_paths.segmentation3d_dir / "runs" / run_id / "experiment.json"
    if not experiment.is_file():
        raise FileNotFoundError(
            "Run ObjectGS export before mesh registration: "
            f"{experiment}"
        )
    payload = read_json(experiment)
    artifacts = payload.get("meshArtifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
    artifacts["object_meshes"] = {
        "variantId": "object_meshes",
        "displayName": "Official Object Meshes",
        "method": "ObjectGS object render + bounded TSDF",
        "meshVoxelSize": float(config.mesh_voxel_size),
        "meshResolution": int(config.mesh_resolution),
        "meshClusters": int(config.mesh_clusters),
        "meshMaxTriangles": int(config.mesh_max_triangles),
        "manifestPath": str(paths.mesh_manifest),
        "partCount": len(rows),
        "vertexCount": sum(int(row.get("vertexCount", 0)) for row in rows),
        "triangleCount": sum(int(row.get("triangleCount", 0)) for row in rows),
    }
    payload["meshArtifacts"] = artifacts
    atomic_write_json(experiment, payload)
