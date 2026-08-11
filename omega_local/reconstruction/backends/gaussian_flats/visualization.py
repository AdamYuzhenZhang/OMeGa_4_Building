"""Prepare Gaussian Flats artifacts for the interactive reconstruction viewer."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from omega_local.segmentation.split_splat.contract import atomic_write_json, now_utc


_ARTIFACT_LOCK = threading.Lock()


def ensure_visualization_artifacts(run_dir: Path) -> dict[str, Any]:
    """Prepare plane-mask overlays and web-ready meshes from native outputs."""
    with _ARTIFACT_LOCK:
        return _ensure_visualization_artifacts(run_dir)


def _ensure_visualization_artifacts(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    run = _read_json(run_dir / "run.json")
    iterations = int(run.get("iterations", 30_000))
    outputs = run_dir / "03_outputs"
    output_manifest = outputs / "visualization.json"
    interactive_dir = run_dir.parents[3]
    viewer_run_id = f"gaussian_flats_{run_dir.name}"
    viewer_manifest = (
        interactive_dir
        / "3d_segmentation"
        / "runs"
        / viewer_run_id
        / "experiment.json"
    )

    fingerprint = _fingerprint(
        (
            run_dir / "01_input" / "frames.jsonl",
            run_dir / "01_input" / "planes.json",
            run_dir / "02_model" / "cameras.json",
            run_dir / "stages" / "render.json",
            run_dir / "02_model" / "fuse_post.ply",
            run_dir / "02_model" / "planar_mesh.obj",
        )
    )
    cached = _read_json(output_manifest)
    if (
        cached.get("sourceFingerprint") == fingerprint
        and "gaussianArtifacts" not in cached
        and viewer_manifest.is_file()
    ):
        return cached

    outputs.mkdir(parents=True, exist_ok=True)
    region = _read_json(run_dir / "01_input" / "planes.json")
    region_id = int(region.get("regionId", 1))
    region_name = str(
        region.get("regionName") or run.get("planarRegion") or "Planar Region"
    )
    plane_mask_artifacts = _plane_mask_artifacts(
        run_dir,
        outputs,
        region_id=region_id,
        region_name=region_name,
    )

    mesh_artifacts: dict[str, dict[str, Any]] = {}
    mesh_rows = []
    for source, output, instance_id, name in (
        (
            run_dir / "02_model" / "fuse_post.ply",
            outputs / "meshes" / "full_hybrid.glb",
            0,
            "Full Hybrid TSDF",
        ),
        (
            run_dir / "02_model" / "planar_mesh.obj",
            outputs / "meshes" / "flat_region.glb",
            region_id,
            f"{region_name} Flat Surface",
        ),
    ):
        if not source.is_file():
            continue
        vertex_count, triangle_count = _mesh_to_glb(source, output)
        mesh_rows.append(
            {
                "persistentRegionId": instance_id,
                "name": name,
                "glb": str(output),
                "vertexCount": vertex_count,
                "triangleCount": triangle_count,
            }
        )
    if mesh_rows:
        mesh_manifest = outputs / "meshes" / "comparison.json"
        atomic_write_json(
            mesh_manifest,
            {
                "schemaVersion": 1,
                "timestampUtc": now_utc(),
                "regions": mesh_rows,
            },
        )
        mesh_artifacts["mesh_comparison"] = {
            "displayName": "Full + Flat Meshes",
            "manifestPath": str(mesh_manifest),
            "partCount": len(mesh_rows),
            "vertexCount": sum(row["vertexCount"] for row in mesh_rows),
            "triangleCount": sum(row["triangleCount"] for row in mesh_rows),
        }

    payload = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "sourceFingerprint": fingerprint,
        "regionId": region_id,
        "regionName": region_name,
        "meshArtifacts": mesh_artifacts,
        "planeMaskArtifacts": plane_mask_artifacts,
        "renderArtifacts": _render_artifacts(run_dir, iterations),
    }
    atomic_write_json(output_manifest, payload)
    atomic_write_json(
        viewer_manifest,
        {
            "schemaVersion": 1,
            "timestampUtc": now_utc(),
            "runId": viewer_run_id,
            "methodId": "gaussian_flats",
            "experimentFamily": "gaussian_flats",
            "artifactRole": "hybrid_reconstruction",
            "baseRunId": run_dir.name,
            "reconstructionVariant": "released_gaussian_flats",
            "displayName": f"Gaussian Flats · {region_name}",
            "inputId": str(run.get("propagationMethod") or ""),
            "labelSpace": "persistent_region",
            "canonicalRunDir": str(run_dir),
                "meshArtifacts": mesh_artifacts,
            "planeMaskArtifacts": plane_mask_artifacts,
            "renderArtifacts": payload["renderArtifacts"],
        },
    )
    return payload



def _plane_mask_artifacts(
    run_dir: Path,
    outputs: Path,
    *,
    region_id: int,
    region_name: str,
) -> dict[str, dict[str, Any]]:
    """Create overlays for training masks and learned plane assignments."""
    frame_index = run_dir / "01_input" / "frames.jsonl"
    if not frame_index.is_file():
        return {}

    frame_rows = [
        json.loads(line)
        for line in frame_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    color = _label_color(region_id)
    input_dir = outputs / "plane_masks" / "input"
    input_count, input_nonempty = _write_mask_overlays(
        (
            (int(row["frameId"]), Path(str(row.get("planeMask") or "")))
            for row in frame_rows
        ),
        input_dir,
        color,
        soft_alpha=False,
    )
    artifacts: dict[str, dict[str, Any]] = {}
    if input_count:
        artifacts["input_plane_masks"] = {
            "displayName": f"{region_name} Input Masks",
            "overlayDir": str(input_dir),
            "frameCount": input_count,
            "nonemptyFrameCount": input_nonempty,
            "regionId": int(region_id),
            "regionName": region_name,
            "sourceRole": "training_plane_masks",
        }

    rendered_dir = outputs / "plane_masks" / "rendered"
    rendered_count, rendered_nonempty = _write_mask_overlays(
        _rendered_plane_mask_rows(run_dir, frame_rows),
        rendered_dir,
        color,
        soft_alpha=True,
    )
    if rendered_count:
        artifacts["rendered_plane_masks"] = {
            "displayName": f"{region_name} Rendered Assignment",
            "overlayDir": str(rendered_dir),
            "frameCount": rendered_count,
            "nonemptyFrameCount": rendered_nonempty,
            "regionId": int(region_id),
            "regionName": region_name,
            "sourceRole": "trained_plane_assignment",
        }
    return artifacts


def _rendered_plane_mask_rows(
    run_dir: Path,
    frame_rows: list[dict[str, Any]],
) -> list[tuple[int, Path]]:
    cameras_path = run_dir / "02_model" / "cameras.json"
    if not cameras_path.is_file():
        return []
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    frame_id_by_stem = {
        Path(str(row.get("imageName") or "")).stem: int(row["frameId"])
        for row in frame_rows
    }
    iteration = int(_read_json(run_dir / "run.json").get("iterations", 30_000))
    model_dir = run_dir / "02_model"
    test_run = model_dir / "test" / f"ours_{iteration}"
    train_run = model_dir / "train" / f"ours_{iteration}"
    test_count = len(list((test_run / "renders").glob("*.png")))
    rows: list[tuple[int, Path]] = []
    for camera_index, camera in enumerate(cameras):
        frame_id = frame_id_by_stem.get(
            Path(str(camera.get("img_name") or "")).stem
        )
        if frame_id is None:
            continue
        if camera_index < test_count:
            source = test_run / "renders_mask" / "0" / f"{camera_index:05d}.png"
        else:
            source = (
                train_run
                / "renders_mask"
                / "0"
                / f"{camera_index - test_count:05d}.png"
            )
        rows.append((frame_id, source))
    return rows


def _write_mask_overlays(
    rows: Any,
    output_dir: Path,
    color: np.ndarray,
    *,
    soft_alpha: bool,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_count = 0
    nonempty_count = 0
    for frame_id, source in rows:
        if not source.is_file():
            continue
        confidence = np.asarray(Image.open(source).convert("L"), dtype=np.uint8)
        mask = confidence > 0
        overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
        overlay[mask, :3] = color
        overlay[..., 3] = (
            np.rint(confidence.astype(np.float32) * (138.0 / 255.0)).astype(np.uint8)
            if soft_alpha
            else np.where(mask, 138, 0).astype(np.uint8)
        )
        target = output_dir / f"{int(frame_id):06d}.png"
        if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
            Image.fromarray(overlay, mode="RGBA").save(target, compress_level=3)
        frame_count += 1
        nonempty_count += int(np.any(mask))
    return frame_count, nonempty_count


def _label_color(label: int) -> np.ndarray:
    value = (int(label) * 1103515245 + 12345) & 0xFFFFFFFF
    return np.array(
        [
            45 + ((value >> 0) & 0x7F),
            65 + ((value >> 8) & 0x7F),
            85 + ((value >> 16) & 0x7F),
        ],
        dtype=np.uint8,
    )

def _mesh_to_glb(source: Path, output: Path) -> tuple[int, int]:
    import trimesh

    output.parent.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(source, force="scene", process=False)
    scene = loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
    geometries = list(scene.geometry.values())
    vertex_count = sum(len(mesh.vertices) for mesh in geometries)
    triangle_count = sum(len(mesh.faces) for mesh in geometries)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(scene.export(file_type="glb"))
    temporary.replace(output)
    return int(vertex_count), int(triangle_count)


def _render_artifacts(run_dir: Path, iterations: int) -> dict[str, str]:
    train = run_dir / "02_model" / "train" / f"ours_{iterations}"
    test = run_dir / "02_model" / "test" / f"ours_{iterations}"
    rows = {
        "trainRgb": train / "renders",
        "trainPlanes": train / "renders_planes",
        "trainPlaneMasks": train / "renders_mask",
        "testRgb": test / "renders",
        "testPlanes": test / "renders_planes",
        "testPlaneMasks": test / "renders_mask",
    }
    return {key: str(path) for key, path in rows.items() if path.is_dir()}


def _fingerprint(paths: tuple[Path, ...]) -> dict[str, str]:
    result = {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            result[str(path)] = f"{stat.st_size:x}-{stat.st_mtime_ns:x}"
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
