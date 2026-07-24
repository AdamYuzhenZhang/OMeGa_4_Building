"""SAI3D baseline adapter for OMeGa DSLR reconstructions.

SAI3D is ScanNet-oriented: it expects RGB-D frames, integer 2D proposal masks,
camera poses, and 3D superpoints. This adapter keeps SAI3D as an external
baseline while translating OMeGa outputs plus the common 2D proposal initializer
into that shape:

    <model_dir>/segmentation/baselines/sai3d/
      dataset/posed_images/<scene>/
      dataset/2D_masks/<scene>/<mask_name>/
      dataset/scans/<scene>/
      mesh_labels/
      view_masks/
      summaries/

The default proposal source is `view_proposals_1024`. SAI3D's multi-view
affinity and progressive region-growing logic is imported from the external
SAI3D repository; the local code handles staging, mesh superpoint substitution,
and canonical OMeGa outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import scipy.spatial
import trimesh

from omega_local.remesh.mesh_clean import resolve_latest_omega_mesh
from omega_local.segmentation.proposal_source import resolve_proposal_source


OMEGA_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = OMEGA_ROOT.parent


@dataclass(frozen=True)
class SAI3DPaths:
    baseline_dir: Path
    dataset_dir: Path
    posed_dir: Path
    mask_root_dir: Path
    mask_dir: Path
    scan_dir: Path
    mesh_label_dir: Path
    view_mask_dir: Path
    view_mask_masks_dir: Path
    view_mask_overlay_dir: Path
    view_mask_proposal_dir: Path
    view_mask_proposal_masks_dir: Path
    view_mask_proposal_overlay_dir: Path
    view_mask_reject_dir: Path
    view_mask_reject_masks_dir: Path
    view_mask_reject_overlay_dir: Path
    view_mask_projected_dir: Path
    view_mask_projected_masks_dir: Path
    view_mask_projected_overlay_dir: Path
    view_mask_point_support_dir: Path
    view_mask_point_support_masks_dir: Path
    view_mask_point_support_overlay_dir: Path
    view_mask_prompt_dir: Path
    view_mask_prompt_overlay_dir: Path
    view_mask_sam2_dir: Path
    view_mask_sam2_masks_dir: Path
    view_mask_sam2_overlay_dir: Path
    summary_dir: Path
    log_dir: Path
    scene_name: str
    mask_name: str

    @property
    def scene_mesh(self) -> Path:
        return self.scan_dir / f"{self.scene_name}_vh_clean_2.ply"

    @property
    def points_path(self) -> Path:
        return self.scan_dir / "points.pts"

    @property
    def superpoints_path(self) -> Path:
        return self.scan_dir / "omega_superpoints.npy"

    @property
    def observations_path(self) -> Path:
        return self.mesh_label_dir / "sai3d_observations.npz"

    @property
    def frame_manifest(self) -> Path:
        return self.dataset_dir / "frame_manifest.jsonl"

    @property
    def dataset_summary(self) -> Path:
        return self.dataset_dir / "dataset_summary.json"

    @property
    def summary(self) -> Path:
        return self.baseline_dir / "baseline_summary.json"


@dataclass(frozen=True)
class PointSet:
    points: np.ndarray
    source: str
    summary: dict[str, Any]
    face_indices: np.ndarray | None = None
    barycentric: np.ndarray | None = None
    external_path: Path | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _slug_token(value: str) -> str:
    out: list[str] = []
    last = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last = False
        elif not last:
            out.append("_")
            last = True
    return "".join(out).strip("_") or "sai3d"


def resolve_paths(
    model_dir: Path,
    *,
    baseline_name: str = "sai3d",
    scene_name: str | None = None,
    output_dir: Path | None = None,
    mask_name: str = "view_proposals",
) -> SAI3DPaths:
    baseline_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else model_dir / "segmentation" / "baselines" / _slug_token(baseline_name)
    )
    scene_slug = _slug_token(scene_name or f"{model_dir.parent.parent.name}_{model_dir.name}")
    dataset_dir = baseline_dir / "dataset"
    return SAI3DPaths(
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        posed_dir=dataset_dir / "posed_images" / scene_slug,
        mask_root_dir=dataset_dir / "2D_masks" / scene_slug,
        mask_dir=dataset_dir / "2D_masks" / scene_slug / _slug_token(mask_name),
        scan_dir=dataset_dir / "scans" / scene_slug,
        mesh_label_dir=baseline_dir / "mesh_labels",
        view_mask_dir=baseline_dir / "view_masks",
        view_mask_masks_dir=baseline_dir / "view_masks" / "masks",
        view_mask_overlay_dir=baseline_dir / "view_masks" / "overlays",
        view_mask_proposal_dir=baseline_dir / "view_masks" / "proposal_relabel",
        view_mask_proposal_masks_dir=baseline_dir / "view_masks" / "proposal_relabel" / "masks",
        view_mask_proposal_overlay_dir=baseline_dir / "view_masks" / "proposal_relabel" / "overlays",
        view_mask_reject_dir=baseline_dir / "view_masks" / "proposal_rejected",
        view_mask_reject_masks_dir=baseline_dir / "view_masks" / "proposal_rejected" / "masks",
        view_mask_reject_overlay_dir=baseline_dir / "view_masks" / "proposal_rejected" / "overlays",
        view_mask_projected_dir=baseline_dir / "view_masks" / "projected_3d",
        view_mask_projected_masks_dir=baseline_dir / "view_masks" / "projected_3d" / "masks",
        view_mask_projected_overlay_dir=baseline_dir / "view_masks" / "projected_3d" / "overlays",
        view_mask_point_support_dir=baseline_dir / "view_masks" / "projected_points",
        view_mask_point_support_masks_dir=baseline_dir / "view_masks" / "projected_points" / "masks",
        view_mask_point_support_overlay_dir=baseline_dir / "view_masks" / "projected_points" / "overlays",
        view_mask_prompt_dir=baseline_dir / "view_masks" / "sam2_prompts",
        view_mask_prompt_overlay_dir=baseline_dir / "view_masks" / "sam2_prompts" / "overlays",
        view_mask_sam2_dir=baseline_dir / "view_masks" / "sam2_refined",
        view_mask_sam2_masks_dir=baseline_dir / "view_masks" / "sam2_refined" / "masks",
        view_mask_sam2_overlay_dir=baseline_dir / "view_masks" / "sam2_refined" / "overlays",
        summary_dir=baseline_dir / "summaries",
        log_dir=baseline_dir / "logs",
        scene_name=scene_slug,
        mask_name=_slug_token(mask_name),
    )


def _require_file(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _resolve_sai3d_root(path: Path | None) -> Path:
    if path is None:
        fork_root = THIRD_PARTY_ROOT / "SAI3D_DT"
        root = fork_root if fork_root.exists() else THIRD_PARTY_ROOT / "SAI3D"
    else:
        root = path.expanduser().resolve()
    _require_dir(root, "SAI3D root")
    _require_file(root / "sai3d_base.py", "SAI3D sai3d_base.py")
    _require_file(root / "helpers" / "sai3d_utils.py", "SAI3D helpers/sai3d_utils.py")
    return root


def _resolve_input_mesh(model_dir: Path, mesh: Path | None, iteration: int) -> Path:
    if mesh is not None:
        raw = mesh.expanduser()
        candidates = [raw.resolve()] if raw.is_absolute() else [raw.resolve(), (model_dir / raw).resolve()]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        raise SystemExit(f"Input mesh does not exist. Tried: {[str(candidate) for candidate in candidates]}")
    healed = model_dir / "remesh" / "local" / "preclean_healed_mesh.ply"
    if healed.exists() and healed.is_file():
        return healed
    preclean = model_dir / "remesh" / "local" / "preclean_mesh.ply"
    if preclean.exists() and preclean.is_file():
        return preclean
    return resolve_latest_omega_mesh(model_dir, iteration=iteration)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"SAI3D baseline expects a triangle mesh: {path}")
    return mesh


def _mesh_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    colors = getattr(mesh.visual, "vertex_colors", None)
    if colors is not None and len(colors) == len(mesh.vertices):
        colors_u8 = np.asarray(colors, dtype=np.uint8)
        if colors_u8.shape[1] >= 3:
            return colors_u8[:, :3].copy()
    return np.full((len(mesh.vertices), 3), 190, dtype=np.uint8)


def _export_scene_mesh(source_mesh: Path, out_path: Path) -> dict[str, Any]:
    mesh = _load_mesh(source_mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    colors = _mesh_vertex_colors(mesh)

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(vertices)
    out.triangles = o3d.utility.Vector3iVector(faces)
    out.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(out_path), out, write_ascii=False):
        raise RuntimeError(f"Failed to write SAI3D scene mesh: {out_path}")
    np.savetxt(out_path.parent / "points.pts", vertices.astype(np.float32), fmt="%.8f")

    return {
        "sourceMesh": str(source_mesh),
        "sceneMesh": str(out_path),
        "vertexCount": int(vertices.shape[0]),
        "faceCount": int(faces.shape[0]),
        "bounds": np.asarray(mesh.bounds, dtype=float).tolist(),
    }


def _write_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(points, dtype=np.float32), fmt="%.8f")


def _write_intrinsics(path: Path, fx: float, fy: float, cx: float, cy: float) -> None:
    mat = np.array(
        [
            [float(fx), 0.0, float(cx), 0.0],
            [0.0, float(fy), float(cy), 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, mat, fmt="%.10f")


def _build_raycast_scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = _load_mesh(mesh_path)
    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def _render_depth_m(
    scene: o3d.t.geometry.RaycastingScene,
    *,
    pose_world_t_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    chunk_rows: int,
) -> np.ndarray:
    pose = np.asarray(pose_world_t_cam, dtype=np.float64)
    world_to_cam = np.linalg.inv(pose)
    rot = pose[:3, :3]
    origin = pose[:3, 3].astype(np.float32)
    depth = np.zeros((height, width), dtype=np.float32)

    for y0 in range(0, height, max(int(chunk_rows), 1)):
        y1 = min(height, y0 + max(int(chunk_rows), 1))
        yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
        dirs_cam = np.stack(
            [
                (xx + 0.5 - float(cx)) / max(float(fx), 1e-6),
                (yy + 0.5 - float(cy)) / max(float(fy), 1e-6),
                np.ones_like(xx, dtype=np.float32),
            ],
            axis=-1,
        ).reshape(-1, 3)
        dirs_world = dirs_cam @ rot.T.astype(np.float32)
        dirs_world = dirs_world / np.maximum(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-8)
        origins = np.broadcast_to(origin.reshape(1, 3), dirs_world.shape)
        rays = np.concatenate([origins, dirs_world.astype(np.float32)], axis=1)
        answer = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = answer["t_hit"].numpy().reshape(-1)
        hit = np.isfinite(t_hit)
        if not np.any(hit):
            continue
        points_world = origins[hit].astype(np.float64) + dirs_world[hit].astype(np.float64) * t_hit[hit, None]
        points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
        points_cam = (world_to_cam @ points_h.T).T[:, :3]
        z = points_cam[:, 2]
        z_valid = z > 0.0
        flat = depth[y0:y1].reshape(-1)
        hit_indices = np.nonzero(hit)[0][z_valid]
        flat[hit_indices] = z[z_valid].astype(np.float32)
    return depth


def _render_face_label_map(
    scene: o3d.t.geometry.RaycastingScene,
    face_labels: np.ndarray,
    *,
    pose_world_t_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    chunk_rows: int,
) -> np.ndarray:
    pose = np.asarray(pose_world_t_cam, dtype=np.float64)
    rot = pose[:3, :3]
    origin = pose[:3, 3].astype(np.float32)
    labels = np.asarray(face_labels, dtype=np.int64)
    rendered = np.zeros((height, width), dtype=np.uint16)

    for y0 in range(0, height, max(int(chunk_rows), 1)):
        y1 = min(height, y0 + max(int(chunk_rows), 1))
        yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
        dirs_cam = np.stack(
            [
                (xx + 0.5 - float(cx)) / max(float(fx), 1e-6),
                (yy + 0.5 - float(cy)) / max(float(fy), 1e-6),
                np.ones_like(xx, dtype=np.float32),
            ],
            axis=-1,
        ).reshape(-1, 3)
        dirs_world = dirs_cam @ rot.T.astype(np.float32)
        dirs_world = dirs_world / np.maximum(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-8)
        origins = np.broadcast_to(origin.reshape(1, 3), dirs_world.shape)
        rays = np.concatenate([origins, dirs_world.astype(np.float32)], axis=1)
        answer = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = answer["t_hit"].numpy().reshape(-1)
        primitive_ids = answer["primitive_ids"].numpy().reshape(-1).astype(np.int64, copy=False)
        hit = np.isfinite(t_hit) & (primitive_ids >= 0) & (primitive_ids < labels.shape[0])
        if not np.any(hit):
            continue
        flat = rendered[y0:y1].reshape(-1)
        hit_ids = np.nonzero(hit)[0]
        flat[hit_ids] = np.clip(labels[primitive_ids[hit_ids]], 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return rendered


def _write_depth_png(path: Path, depth_m: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 1, np.iinfo(np.uint16).max).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"Failed to write depth png: {path}")
    values = depth_m[valid]
    return {
        "coverage": float(np.count_nonzero(valid) / max(valid.size, 1)),
        "minDepthMeters": float(values.min()) if values.size else 0.0,
        "medianDepthMeters": float(np.median(values)) if values.size else 0.0,
        "maxDepthMeters": float(values.max()) if values.size else 0.0,
    }


def _load_label_map(path: Path, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        labels = np.load(path).astype(np.uint16, copy=False)
    else:
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise ValueError(f"Could not read label map: {path}")
        if raw.ndim == 3:
            raw = raw[:, :, 0]
        labels = raw.astype(np.uint16, copy=False)
    if expected_shape is not None and labels.shape != expected_shape:
        labels = cv2.resize(labels, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return labels.astype(np.uint16, copy=False)


def prepare(args: argparse.Namespace, paths: SAI3DPaths) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    source_mesh = _resolve_input_mesh(model_dir, args.mesh, int(args.iteration))
    proposal_source = resolve_proposal_source(
        model_dir,
        proposal_source_name=str(args.proposal_source_name),
        proposal_source_dir=args.proposal_source_dir,
    )
    source_rows = proposal_source.rows(int(args.source_frame_stride), int(args.max_frames))

    if paths.dataset_dir.exists() and any(paths.dataset_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"SAI3D dataset already exists: {paths.dataset_dir}. Pass --overwrite to replace it.")
    if args.overwrite and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
    for directory in [paths.posed_dir, paths.mask_dir, paths.scan_dir, paths.summary_dir, paths.log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    mesh_summary = _export_scene_mesh(source_mesh, paths.scene_mesh)
    scene = _build_raycast_scene(source_mesh)

    manifest: list[dict[str, Any]] = []
    depth_coverages: list[float] = []
    for out_index, row in enumerate(source_rows):
        source_frame_id = proposal_source.frame_id(row, out_index)
        color_src = proposal_source.image_path(row)
        pose_src = proposal_source.pose_path(row)
        mask_src = proposal_source.mask_path(row, out_index)

        rgb = cv2.imread(str(color_src), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Could not read source RGB frame: {color_src}")
        height, width = rgb.shape[:2]
        fx = float(row["fx"])
        fy = float(row["fy"])
        cx = float(row["cx"])
        cy = float(row["cy"])
        pose = np.loadtxt(pose_src, dtype=np.float64)

        color_path = paths.posed_dir / f"{out_index}.jpg"
        if not cv2.imwrite(str(color_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
            raise RuntimeError(f"Failed to write SAI3D RGB frame: {color_path}")
        pose_path = paths.posed_dir / f"{out_index}.txt"
        np.savetxt(pose_path, pose, fmt="%.10f")

        mask = _load_label_map(mask_src, expected_shape=(height, width))
        mask_path = paths.mask_dir / f"maskraw_{out_index}.png"
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to write SAI3D mask frame: {mask_path}")

        depth = _render_depth_m(
            scene,
            pose_world_t_cam=pose,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            chunk_rows=int(args.raycast_chunk_rows),
        )
        depth_path = paths.posed_dir / f"{out_index}.png"
        depth_stats = _write_depth_png(depth_path, depth)
        depth_coverages.append(float(depth_stats["coverage"]))
        if out_index == 0:
            _write_intrinsics(paths.posed_dir / "intrinsic_color.txt", fx, fy, cx, cy)
            _write_intrinsics(paths.posed_dir / "intrinsic_depth.txt", fx, fy, cx, cy)

        manifest.append(
            {
                "sai3dFrameId": int(out_index),
                "proposalSource": str(proposal_source.baseline_dir),
                "sourceFrameId": int(source_frame_id),
                "scanID": str(row.get("scanID", "")),
                "frameID": int(row.get("frameID", source_frame_id)),
                "imageName": str(row.get("imageName", color_src.name)),
                "sourceImagePath": str(row.get("sourceImagePath", color_src)),
                "colorPath": str(color_path.relative_to(paths.dataset_dir)),
                "depthPath": str(depth_path.relative_to(paths.dataset_dir)),
                "posePath": str(pose_path.relative_to(paths.dataset_dir)),
                "maskPath": str(mask_path.relative_to(paths.dataset_dir)),
                "width": int(width),
                "height": int(height),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "depthStats": depth_stats,
                "inputLabelCount": int(np.count_nonzero(np.unique(mask) > 0)),
                "inputCoverage": float(np.count_nonzero(mask > 0) / max(mask.size, 1)),
                "viewWeight": max(int(row.get("viewWeight", 1)), 1),
                "isManualAnchor": bool(row.get("isManualAnchor", False)),
                "maskSource": str(row.get("maskSource", args.proposal_source_name)),
            }
        )
        print(
            f"[sai3d prepare {out_index + 1:04d}/{len(source_rows):04d}] "
            f"source={source_frame_id} labels={manifest[-1]['inputLabelCount']} "
            f"mask_cov={manifest[-1]['inputCoverage']:.3f} depth_cov={depth_stats['coverage']:.3f}"
        )

    _write_jsonl(paths.frame_manifest, manifest)
    summary = {
        "stage": "prepare",
        "timestampUtc": _now(),
        "method": "SAI3D staged from an external 2D proposal mask source",
        "modelDir": str(model_dir),
        "sceneName": paths.scene_name,
        "sourceMesh": str(source_mesh),
        "proposalSourceName": str(args.proposal_source_name),
        "proposalSourceDir": str(proposal_source.baseline_dir),
        "proposalSourceFrameManifest": str(proposal_source.frame_manifest),
        "proposalSourceMaskDir": str(proposal_source.mask_dir),
        "mesh": mesh_summary,
        "frameCount": int(len(manifest)),
        "maskName": paths.mask_name,
        "depthCoverageMean": float(np.mean(depth_coverages)) if depth_coverages else 0.0,
        "outputs": {
            "datasetDir": str(paths.dataset_dir),
            "posedDir": str(paths.posed_dir),
            "maskDir": str(paths.mask_dir),
            "scanDir": str(paths.scan_dir),
            "sceneMesh": str(paths.scene_mesh),
            "points": str(paths.points_path),
            "frameManifest": str(paths.frame_manifest),
        },
    }
    _write_json(paths.dataset_summary, summary)
    _write_json(paths.summary_dir / "prepare_summary.json", summary)
    return summary


def _mesh_vertex_edges(faces: np.ndarray) -> np.ndarray:
    faces_i = np.asarray(faces, dtype=np.int64)
    edges = np.vstack([faces_i[:, [0, 1]], faces_i[:, [1, 2]], faces_i[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0).astype(np.int64, copy=False)


class _UnionFind:
    def __init__(self, n_items: int):
        self.parent = np.arange(int(n_items), dtype=np.int64)
        self.rank = np.zeros(int(n_items), dtype=np.uint8)

    def find(self, value: int) -> int:
        root = int(value)
        while self.parent[root] != root:
            root = int(self.parent[root])
        cur = int(value)
        while self.parent[cur] != cur:
            nxt = int(self.parent[cur])
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def labels(self) -> np.ndarray:
        roots = np.array([self.find(i) for i in range(self.parent.shape[0])], dtype=np.int64)
        unique = np.unique(roots)
        out = np.zeros(roots.shape[0], dtype=np.int32)
        for label, root in enumerate(unique):
            out[roots == root] = int(label)
        return out


def _natural_labels(labels: np.ndarray) -> np.ndarray:
    labels_i = np.asarray(labels, dtype=np.int64)
    unique = np.unique(labels_i)
    out = np.zeros(labels_i.shape[0], dtype=np.int32)
    for new_label, old_label in enumerate(unique):
        out[labels_i == old_label] = int(new_label)
    return out


def _natural_positive_labels(labels: np.ndarray) -> np.ndarray:
    labels_i = np.asarray(labels, dtype=np.int64)
    out = np.zeros(labels_i.shape[0], dtype=np.int32)
    positive = labels_i > 0
    for new_label, old_label in enumerate(np.unique(labels_i[positive]), start=1):
        out[labels_i == old_label] = int(new_label)
    return out


def _sample_mesh_surface(mesh: trimesh.Trimesh, sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    valid_faces = np.nonzero(np.isfinite(areas) & (areas > 1e-12))[0]
    if valid_faces.size == 0:
        raise ValueError("Cannot area-sample a mesh with no positive-area faces.")
    count = max(int(sample_count), 1)
    probabilities = areas[valid_faces] / float(np.sum(areas[valid_faces]))
    rng = np.random.default_rng(int(seed))
    face_ids = valid_faces[rng.choice(valid_faces.shape[0], size=count, replace=True, p=probabilities)]
    tri = vertices[faces[face_ids]]

    r1 = rng.random(count)
    r2 = rng.random(count)
    sqrt_r1 = np.sqrt(r1)
    bary = np.stack(
        [
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ],
        axis=1,
    )
    points = np.einsum("ni,nij->nj", bary, tri).astype(np.float32, copy=False)
    return points, face_ids.astype(np.int32, copy=False), bary.astype(np.float32, copy=False)


def _resolve_point_cloud_path(model_dir: Path, path: Path | None) -> Path:
    if path is None:
        raise SystemExit("--point-cloud is required when --point-source point_cloud")
    raw = path.expanduser()
    candidates = [raw.resolve()] if raw.is_absolute() else [raw.resolve(), (model_dir / raw).resolve()]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise SystemExit(f"Point cloud does not exist. Tried: {[str(candidate) for candidate in candidates]}")


def _load_point_cloud_points(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.size == 0:
        mesh = trimesh.load(path, force="mesh", process=False)
        if isinstance(mesh, trimesh.Scene):
            parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            if parts:
                mesh = trimesh.util.concatenate(parts)
        if isinstance(mesh, trimesh.Trimesh):
            points = np.asarray(mesh.vertices, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Could not load a non-empty XYZ point set from: {path}")
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud only contains non-finite points: {path}")
    return points.astype(np.float32, copy=False)


def _subsample_points(points: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray | None]:
    count = int(points.shape[0])
    limit = int(max_points)
    if limit <= 0 or count <= limit:
        return points.astype(np.float32, copy=False), None
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(count, size=limit, replace=False)).astype(np.int64, copy=False)
    return points[selected].astype(np.float32, copy=False), selected


def _build_point_set(mesh: trimesh.Trimesh, args: argparse.Namespace) -> PointSet:
    model_dir = args.model_dir.expanduser().resolve()
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    source = str(args.point_source)
    if source == "mesh_vertices":
        return PointSet(
            points=vertices,
            source=source,
            summary={
                "source": source,
                "pointCount": int(vertices.shape[0]),
                "description": "OMeGa mesh vertices used directly as SAI3D graph points.",
            },
        )
    if source == "mesh_area_samples":
        sample_count = int(args.point_sample_count)
        if sample_count <= 0:
            sample_count = int(vertices.shape[0])
        points, face_ids, barycentric = _sample_mesh_surface(mesh, sample_count, int(args.point_sample_seed))
        return PointSet(
            points=points,
            source=source,
            face_indices=face_ids,
            barycentric=barycentric,
            summary={
                "source": source,
                "pointCount": int(points.shape[0]),
                "requestedPointCount": int(sample_count),
                "sampleSeed": int(args.point_sample_seed),
                "sampledFaceCount": int(np.unique(face_ids).shape[0]),
                "meshFaceCount": int(len(mesh.faces)),
                "description": "Triangle-area-weighted surface samples from the staged OMeGa mesh.",
            },
        )
    if source == "point_cloud":
        point_cloud_path = _resolve_point_cloud_path(model_dir, args.point_cloud)
        loaded = _load_point_cloud_points(point_cloud_path)
        points, selected = _subsample_points(
            loaded,
            max_points=int(args.point_cloud_max_points),
            seed=int(args.point_cloud_sample_seed),
        )
        return PointSet(
            points=points,
            source=source,
            external_path=point_cloud_path,
            summary={
                "source": source,
                "sourcePointCloud": str(point_cloud_path),
                "loadedPointCount": int(loaded.shape[0]),
                "pointCount": int(points.shape[0]),
                "maxPointCount": int(args.point_cloud_max_points),
                "sampleSeed": int(args.point_cloud_sample_seed),
                "wasSubsampled": bool(selected is not None),
                "description": "External XYZ point cloud used as the SAI3D graph points.",
            },
        )
    raise SystemExit(f"Unsupported --point-source: {source}")


def _mesh_face_adjacency_lists(mesh: trimesh.Trimesh) -> list[list[int]]:
    faces = np.asarray(mesh.faces, dtype=np.int64)
    neighbors: list[list[int]] = [[] for _ in range(faces.shape[0])]
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if adjacency.size:
        for a, b in adjacency:
            neighbors[int(a)].append(int(b))
            neighbors[int(b)].append(int(a))
    return neighbors


def _voxel_superpoints(vertices: np.ndarray, edges: np.ndarray, target_count: int, voxel_size: float) -> tuple[np.ndarray, dict[str, Any]]:
    vertices_f = np.asarray(vertices, dtype=np.float64)
    extents = vertices_f.max(axis=0) - vertices_f.min(axis=0)
    diag = float(np.linalg.norm(extents))
    if voxel_size <= 0.0:
        safe_extents = np.maximum(extents, max(diag * 0.02, 1e-4))
        voxel_size = float(np.cbrt(float(np.prod(safe_extents)) / max(int(target_count), 1)))
    voxel_size = max(float(voxel_size), 1e-6)
    grid = np.floor((vertices_f - vertices_f.min(axis=0)) / voxel_size).astype(np.int64)
    _, voxel_ids = np.unique(grid, axis=0, return_inverse=True)

    uf = _UnionFind(vertices_f.shape[0])
    same_voxel = voxel_ids[edges[:, 0]] == voxel_ids[edges[:, 1]]
    for a, b in edges[same_voxel]:
        uf.union(int(a), int(b))
    labels = uf.labels()
    summary = {
        "mode": "voxel",
        "targetCount": int(target_count),
        "voxelSize": float(voxel_size),
        "superpointCount": int(np.unique(labels).shape[0]),
        "meanVerticesPerSuperpoint": float(vertices_f.shape[0] / max(np.unique(labels).shape[0], 1)),
    }
    return labels.astype(np.int32, copy=False), summary


def _sample_voxel_superpoints(
    *,
    mesh: trimesh.Trimesh,
    point_set: PointSet,
    target_count: int,
    voxel_size: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(point_set.points, dtype=np.float64)
    face_ids = point_set.face_indices
    if face_ids is None:
        raise ValueError("Area-sampled superpoints require point face indices.")
    extents = points.max(axis=0) - points.min(axis=0)
    diag = float(np.linalg.norm(extents))
    if voxel_size <= 0.0:
        safe_extents = np.maximum(extents, max(diag * 0.02, 1e-4))
        voxel_size = float(np.cbrt(float(np.prod(safe_extents)) / max(int(target_count), 1)))
    voxel_size = max(float(voxel_size), 1e-6)
    grid = np.floor((points - points.min(axis=0)) / voxel_size).astype(np.int64)
    _, voxel_ids = np.unique(grid, axis=0, return_inverse=True)

    uf = _UnionFind(points.shape[0])
    bucket_faces: dict[int, dict[int, int]] = {}
    for point_id, (voxel_id, face_id) in enumerate(zip(voxel_ids, face_ids, strict=True)):
        face_reps = bucket_faces.setdefault(int(voxel_id), {})
        previous = face_reps.get(int(face_id))
        if previous is None:
            face_reps[int(face_id)] = int(point_id)
        else:
            uf.union(previous, int(point_id))

    face_neighbors = _mesh_face_adjacency_lists(mesh)
    for face_reps in bucket_faces.values():
        present = set(face_reps)
        for face_id, rep in face_reps.items():
            for neighbor in face_neighbors[int(face_id)]:
                if neighbor in present and int(face_id) < int(neighbor):
                    uf.union(rep, face_reps[int(neighbor)])

    labels = uf.labels()
    summary = {
        "mode": "voxel",
        "pointSource": str(point_set.source),
        "targetCount": int(target_count),
        "voxelSize": float(voxel_size),
        "superpointCount": int(np.unique(labels).shape[0]),
        "meanPointsPerSuperpoint": float(points.shape[0] / max(np.unique(labels).shape[0], 1)),
        "description": "Connected same-voxel sampled surface patches; adjacency follows original mesh face adjacency.",
    }
    return labels.astype(np.int32, copy=False), summary


def _nonempty_voxel_count(points: np.ndarray, voxel_size: float) -> int:
    points_f = np.asarray(points, dtype=np.float64)
    grid = np.floor((points_f - points_f.min(axis=0)) / max(float(voxel_size), 1e-9)).astype(np.int64)
    return int(np.unique(grid, axis=0).shape[0])


def _derive_point_cloud_voxel_size(points: np.ndarray, target_count: int) -> tuple[float, dict[str, Any]]:
    points_f = np.asarray(points, dtype=np.float64)
    target = max(int(target_count), 1)
    extents = points_f.max(axis=0) - points_f.min(axis=0)
    diag = float(np.linalg.norm(extents))
    if points_f.shape[0] <= target:
        return max(float(diag) * 1e-6, 1e-6), {
            "method": "one_point_per_superpoint",
            "targetCount": int(target),
            "reason": "point count does not exceed target count",
        }

    low = max(float(diag) * 1e-7, 1e-6)
    high = max(float(np.max(extents)), low * 2.0)
    while _nonempty_voxel_count(points_f, high) > target and high < max(float(diag) * 4.0, low * 4.0):
        high *= 2.0

    best_size = high
    best_count = _nonempty_voxel_count(points_f, high)
    for _ in range(24):
        mid = 0.5 * (low + high)
        count = _nonempty_voxel_count(points_f, mid)
        if abs(count - target) < abs(best_count - target):
            best_size = mid
            best_count = count
        if count > target:
            low = mid
        else:
            high = mid

    return float(best_size), {
        "method": "non_empty_voxel_count_binary_search",
        "targetCount": int(target),
        "estimatedVoxelCount": int(best_count),
        "boundsExtent": extents.astype(float).tolist(),
        "boundsDiagonal": float(diag),
    }


def _point_cloud_voxel_superpoints(points: np.ndarray, target_count: int, voxel_size: float) -> tuple[np.ndarray, dict[str, Any]]:
    points_f = np.asarray(points, dtype=np.float64)
    extents = points_f.max(axis=0) - points_f.min(axis=0)
    diag = float(np.linalg.norm(extents))
    voxel_derivation: dict[str, Any]
    if voxel_size <= 0.0:
        voxel_size, voxel_derivation = _derive_point_cloud_voxel_size(points_f, int(target_count))
    else:
        voxel_derivation = {"method": "explicit", "targetCount": int(target_count)}
    voxel_size = max(float(voxel_size), 1e-6)
    grid = np.floor((points_f - points_f.min(axis=0)) / voxel_size).astype(np.int64)
    _, labels = np.unique(grid, axis=0, return_inverse=True)
    labels = _natural_labels(labels)
    unique = np.unique(labels)
    summary = {
        "mode": "voxel",
        "pointSource": "point_cloud",
        "targetCount": int(target_count),
        "voxelSize": float(voxel_size),
        "superpointCount": int(unique.shape[0]),
        "meanPointsPerSuperpoint": float(points_f.shape[0] / max(unique.shape[0], 1)),
        "voxelDerivation": voxel_derivation,
        "description": "Non-empty voxel superpoints over an external point cloud; graph adjacency is computed by SAI3D kNN.",
    }
    return labels.astype(np.int32, copy=False), summary


def _build_superpoints(mesh: trimesh.Trimesh, point_set: PointSet, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    points = np.asarray(point_set.points, dtype=np.float32)
    mode = str(args.superpoint_mode)
    if mode == "vertex":
        labels = np.arange(points.shape[0], dtype=np.int32)
        return labels, {
            "mode": "vertex",
            "pointSource": str(point_set.source),
            "superpointCount": int(labels.shape[0]),
            "meanPointsPerSuperpoint": 1.0,
        }
    if mode == "voxel":
        if point_set.source == "mesh_vertices":
            edges = _mesh_vertex_edges(np.asarray(mesh.faces, dtype=np.int64))
            labels, summary = _voxel_superpoints(
                points,
                edges,
                target_count=int(args.superpoint_target_count),
                voxel_size=float(args.superpoint_voxel_size),
            )
            summary["pointSource"] = str(point_set.source)
            summary["meanPointsPerSuperpoint"] = summary.pop("meanVerticesPerSuperpoint")
            return labels, summary
        if point_set.source == "point_cloud":
            return _point_cloud_voxel_superpoints(
                points,
                target_count=int(args.superpoint_target_count),
                voxel_size=float(args.superpoint_voxel_size),
            )
        return _sample_voxel_superpoints(
            mesh=mesh,
            point_set=point_set,
            target_count=int(args.superpoint_target_count),
            voxel_size=float(args.superpoint_voxel_size),
        )
    raise SystemExit(f"Unsupported --superpoint-mode: {mode}")


def _project_vertices(
    vertices: np.ndarray,
    pose_world_t_cam: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose_world_t_cam, dtype=np.float64)
    world_to_cam = np.linalg.inv(pose)
    points_h = np.concatenate([vertices.astype(np.float64), np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    points_cam = (world_to_cam @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    u = np.full(vertices.shape[0], -1, dtype=np.int32)
    v = np.full(vertices.shape[0], -1, dtype=np.int32)
    valid = np.all(np.isfinite(points_cam), axis=1) & (z > 1e-8)
    if np.any(valid):
        valid_indices = np.nonzero(valid)[0]
        denom = z[valid_indices]
        u_float = points_cam[valid_indices, 0] * float(fx) / denom + float(cx)
        v_float = points_cam[valid_indices, 1] * float(fy) / denom + float(cy)
        finite_uv = np.isfinite(u_float) & np.isfinite(v_float)
        if np.any(finite_uv):
            indices = valid_indices[finite_uv]
            int_bounds = np.iinfo(np.int32)
            u[indices] = np.clip(np.rint(u_float[finite_uv]), int_bounds.min + 1, int_bounds.max).astype(np.int32)
            v[indices] = np.clip(np.rint(v_float[finite_uv]), int_bounds.min + 1, int_bounds.max).astype(np.int32)
    return z.astype(np.float32), u, v


def _point_zbuffer_visible_indices(
    *,
    z: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    bounded: np.ndarray,
    width: int,
    keep_depth_band_m: float,
) -> np.ndarray:
    valid_indices = np.nonzero(bounded)[0]
    if valid_indices.size == 0:
        return valid_indices.astype(np.int64, copy=False)

    linear = (v[valid_indices].astype(np.int64) * int(width)) + u[valid_indices].astype(np.int64)
    depths = z[valid_indices].astype(np.float32, copy=False)
    order = np.lexsort((depths, linear))
    linear_sorted = linear[order]
    depths_sorted = depths[order]
    indices_sorted = valid_indices[order]

    start = np.r_[0, np.nonzero(linear_sorted[1:] != linear_sorted[:-1])[0] + 1]
    nearest = depths_sorted[start]
    end = np.r_[start[1:], linear_sorted.shape[0]]
    keep = np.zeros(indices_sorted.shape[0], dtype=bool)
    depth_band = max(float(keep_depth_band_m), 0.0)
    for lo, hi, near in zip(start, end, nearest, strict=True):
        keep[lo:hi] = depths_sorted[lo:hi] <= float(near) + depth_band
    return indices_sorted[keep].astype(np.int64, copy=False)


def _resolve_visibility_source(point_set: PointSet, args: argparse.Namespace) -> str:
    requested = str(args.point_visibility_source)
    if requested != "auto":
        return requested
    if point_set.source == "point_cloud":
        return "point_zbuffer"
    return "rendered_depth"


def _project_vertices_to_masks(
    *,
    paths: SAI3DPaths,
    vertices: np.ndarray,
    point_set: PointSet,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    view_stride = max(int(args.sai3d_view_stride), 1)
    graph_rows = rows[::view_stride]
    point_labels = np.zeros((vertices.shape[0], len(graph_rows)), dtype=np.uint16)
    point_seen = np.zeros((vertices.shape[0], len(graph_rows)), dtype=bool)
    graph_frame_to_col = {int(row["sai3dFrameId"]): col for col, row in enumerate(graph_rows)}
    projection_cache: list[dict[str, Any]] = []
    frame_stats: list[dict[str, Any]] = []
    visibility_source = _resolve_visibility_source(point_set, args)

    for row_index, row in enumerate(rows):
        frame_id = int(row["sai3dFrameId"])
        width = int(row["width"])
        height = int(row["height"])
        pose = np.loadtxt(paths.dataset_dir / str(row["posePath"]), dtype=np.float64)
        mask = _load_label_map(paths.dataset_dir / str(row["maskPath"]), expected_shape=(height, width))
        z, u, v = _project_vertices(
            vertices,
            pose,
            fx=float(row["fx"]),
            fy=float(row["fy"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
        )
        bounded = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if visibility_source == "point_zbuffer":
            visible_indices = _point_zbuffer_visible_indices(
                z=z,
                u=u,
                v=v,
                bounded=bounded,
                width=width,
                keep_depth_band_m=float(args.point_zbuffer_depth_band),
            )
        else:
            depth_raw = cv2.imread(str(paths.dataset_dir / str(row["depthPath"])), cv2.IMREAD_UNCHANGED)
            if depth_raw is None:
                raise ValueError(f"Could not read rendered depth: {paths.dataset_dir / str(row['depthPath'])}")
            depth = depth_raw.astype(np.float32) / 1000.0
            valid_indices = np.nonzero(bounded)[0]
            sampled_depth = depth[v[valid_indices], u[valid_indices]] if valid_indices.size else np.zeros(0, dtype=np.float32)
            visible_local = sampled_depth > 0.0
            if valid_indices.size:
                visible_local &= np.isclose(
                    z[valid_indices],
                    sampled_depth,
                    rtol=float(args.visibility_rtol),
                    atol=float(args.visibility_atol),
                )
            visible_indices = valid_indices[visible_local]
        labels = mask[v[visible_indices], u[visible_indices]] if visible_indices.size else np.zeros(0, dtype=np.uint16)

        if frame_id in graph_frame_to_col:
            col = graph_frame_to_col[frame_id]
            point_seen[visible_indices, col] = True
            point_labels[visible_indices, col] = labels.astype(np.uint16, copy=False)

        projection_cache.append(
            {
                "frameId": int(frame_id),
                "indices": visible_indices.astype(np.int32, copy=False),
                "u": u[visible_indices].astype(np.int32, copy=False),
                "v": v[visible_indices].astype(np.int32, copy=False),
                "z": z[visible_indices].astype(np.float32, copy=False),
                "labels": labels.astype(np.uint16, copy=False),
            }
        )
        frame_stats.append(
            {
                "frameId": int(frame_id),
                "visiblePointCount": int(visible_indices.size),
                "positivePointCount": int(np.count_nonzero(labels > 0)),
                "inputLabelCount": int(np.count_nonzero(np.unique(mask) > 0)),
                "visibilitySource": visibility_source,
            }
        )
        print(
            f"[sai3d project {row_index + 1:04d}/{len(rows):04d}] "
            f"frame={frame_id} visible={visible_indices.size} positive={int(np.count_nonzero(labels > 0))} "
            f"visibility={visibility_source}"
        )
    return point_labels, point_seen, projection_cache, frame_stats


class OmegaSAI3DAgent:
    def __init__(self, points: np.ndarray, args: argparse.Namespace, initial_seg_ids: np.ndarray | None):
        # SAI3DBase was written around ScanNet argparse fields. The OMeGa
        # adapter bypasses SAI3D's native frame loader, but the base constructor
        # still reads view_freq, so provide the neutral compatibility value.
        if not hasattr(args, "view_freq"):
            args.view_freq = 1
        if str(args.sai3d_root) not in sys.path:
            sys.path.insert(0, str(args.sai3d_root))
        from sai3d_base import SAI3DBase

        class _Agent(SAI3DBase):
            def __init__(self, outer: OmegaSAI3DAgent, points_: np.ndarray, args_: argparse.Namespace):
                self._outer = outer
                super().__init__(points_, args_)

            def get_seg_data(
                self,
                base_dir: str,
                scene_id: str,
                max_neighbor_distance: int,
                seg_ids: np.ndarray | None = None,
                points_obj_labels_path: str | None = None,
                k_graph: int = 8,
                point_level: bool = False,
            ):
                return self._outer.get_seg_data(
                    max_neighbor_distance=max_neighbor_distance,
                    seg_ids=seg_ids,
                    k_graph=k_graph,
                    point_level=point_level,
                )

        self.points = points.astype(np.float32, copy=False)
        self.args = args
        self.initial_seg_ids = None if initial_seg_ids is None else _natural_labels(initial_seg_ids)
        self.agent = _Agent(self, self.points, args)
        self.agent.base_dir = ""
        self.agent.scene_id = "omega"

    def get_seg_data(
        self,
        *,
        max_neighbor_distance: int,
        seg_ids: np.ndarray | None = None,
        k_graph: int = 8,
        point_level: bool = False,
    ):
        def query_point_neighbors() -> np.ndarray:
            points_kdtree = scipy.spatial.KDTree(self.points)
            distances, neighbors = points_kdtree.query(
                self.points,
                max(int(k_graph), 1),
                workers=int(self.args.sai3d_workers),
            )
            distances = np.asarray(distances)
            neighbors = np.asarray(neighbors)
            if neighbors.ndim == 1:
                neighbors = neighbors[:, None]
                distances = distances[:, None]
            max_edge = float(getattr(self.args, "graph_max_edge_length", 0.0))
            if max_edge > 0.0:
                invalid = distances > max_edge
                if np.any(invalid):
                    self_ids = np.arange(self.points.shape[0], dtype=np.int64)[:, None]
                    neighbors = neighbors.copy()
                    neighbors[invalid] = np.broadcast_to(self_ids, neighbors.shape)[invalid]
            return neighbors.astype(np.int64, copy=False)

        if point_level:
            seg_ids_out = np.arange(self.points.shape[0], dtype=np.int32)
            neighbors = query_point_neighbors()
            self.agent.seg_member_count = np.ones(self.points.shape[0], dtype=np.int32)
            return seg_ids_out, int(self.points.shape[0]), seg_ids_out, neighbors

        labels = seg_ids if seg_ids is not None else self.initial_seg_ids
        if labels is None:
            labels = np.arange(self.points.shape[0], dtype=np.int32)
        labels = _natural_labels(np.asarray(labels, dtype=np.int64))
        unique, counts = np.unique(labels, return_counts=True)
        seg_num = int(unique.shape[0])
        seg_members = {int(seg_id): np.nonzero(labels == int(seg_id))[0] for seg_id in unique}

        point_neighbors = query_point_neighbors()
        direct = np.zeros((seg_num, seg_num), dtype=bool)
        for seg_id, members in seg_members.items():
            neighbor_seg_ids = labels[point_neighbors[members]]
            direct[int(seg_id), neighbor_seg_ids] = True
        direct[np.eye(seg_num, dtype=bool)] = False
        direct[direct.T] = True

        max_dist = max(int(max_neighbor_distance), 1)
        indirect = np.zeros((max_dist, seg_num, seg_num), dtype=bool)
        indirect[0] = direct
        for dist in range(1, max_dist):
            for seg_id in range(seg_num):
                prev_neighbors = indirect[dist - 1, seg_id]
                expanded = indirect[dist - 1, prev_neighbors].sum(axis=0) > 0
                indirect[dist, seg_id] = expanded
            indirect[dist, np.eye(seg_num, dtype=bool)] = False
            indirect[dist, indirect[dist - 1]] = True
        self.agent.seg_member_count = counts.astype(np.int32, copy=False)
        return labels.astype(np.int32, copy=False), seg_num, seg_members, indirect

    def assign_from_observations(
        self,
        *,
        point_labels: np.ndarray,
        point_seen: np.ndarray,
        thresholds: np.ndarray,
        view_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        history: list[dict[str, Any]] = []
        points_labels: np.ndarray | None = None
        weighted_labels, weighted_seen = _repeat_weighted_views(
            point_labels,
            point_seen,
            view_weights,
        )
        labels_f = weighted_labels.astype(np.float32, copy=False)
        self.agent.M = int(labels_f.shape[1])

        if float(self.args.from_points_thres) > 0.0:
            initial_labels = np.arange(self.points.shape[0], dtype=np.int32) + 1
            self.agent.seg_ids, self.agent.seg_num, self.agent.seg_members, self.agent.seg_direct_neighbors = self.agent.get_seg_data(
                base_dir="",
                scene_id="omega",
                max_neighbor_distance=1,
                seg_ids=initial_labels,
                point_level=True,
                k_graph=int(self.args.k_graph),
            )
            seg_adj = self.agent.get_seg_dok_adjacency(labels_f, weighted_seen)
            point_stage_labels = self.agent.assign_seg_label(
                seg_adj,
                float(self.args.from_points_thres),
                max_neighbor_distance=1,
                dense_neighbor=True,
            ).astype(np.int32)
            points_labels = _natural_positive_labels(point_stage_labels)
            history.append(
                {
                    "stage": "from_points",
                    "threshold": float(self.args.from_points_thres),
                    "labelCount": int(np.count_nonzero(np.unique(points_labels) > 0)),
                }
            )

        for idx, threshold in enumerate(thresholds):
            self.agent.seg_ids, self.agent.seg_num, self.agent.seg_members, self.agent.seg_indirect_neighbors = self.agent.get_seg_data(
                base_dir="",
                scene_id="omega",
                max_neighbor_distance=int(self.args.max_neighbor_distance),
                seg_ids=points_labels,
                k_graph=int(self.args.k_graph),
            )
            self.agent.seg_direct_neighbors = self.agent.seg_indirect_neighbors[0]
            seg_adj = self.agent.get_seg_adjacency(
                points_any=self.points,
                similar_meric=str(self.args.similar_metric),
                points_label=labels_f,
                points_seen=weighted_seen,
            )
            seg_labels = self.agent.assign_seg_label(
                seg_adj,
                float(threshold),
                max_neighbor_distance=int(self.args.max_neighbor_distance),
            )
            if idx == len(thresholds) - 1 and int(self.args.thres_merge) > 0:
                seg_labels = self.agent.merge_small_segs(seg_labels, int(self.args.thres_merge), seg_adj)

            out = np.zeros(self.points.shape[0], dtype=np.int32)
            for seg_id in range(int(self.agent.seg_num)):
                out[self.agent.seg_members[seg_id]] = int(seg_labels[seg_id])
            points_labels = _natural_positive_labels(out)
            history.append(
                {
                    "stage": "progressive_region_growing",
                    "iteration": int(idx),
                    "threshold": float(threshold),
                    "primitiveCount": int(self.agent.seg_num),
                    "labelCount": int(np.count_nonzero(np.unique(points_labels) > 0)),
                }
            )

        if points_labels is None:
            points_labels = np.zeros(self.points.shape[0], dtype=np.int32)
        return points_labels.astype(np.int32, copy=False), history


def _repeat_weighted_views(
    point_labels: np.ndarray,
    point_seen: np.ndarray,
    view_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(point_labels)
    seen = np.asarray(point_seen)
    if labels.ndim != 2 or seen.shape != labels.shape:
        raise ValueError("SAI3D point labels and visibility must be matching N x M arrays.")
    if view_weights is None:
        return labels, seen
    weights = np.asarray(view_weights, dtype=np.int32).reshape(-1)
    if weights.shape[0] != labels.shape[1]:
        raise ValueError(
            f"SAI3D view weights have length {weights.shape[0]}, expected {labels.shape[1]}."
        )
    if np.any(weights < 1):
        raise ValueError("SAI3D view weights must be positive integers.")
    if np.all(weights == 1):
        return labels, seen
    repeated_columns = np.repeat(np.arange(labels.shape[1], dtype=np.int32), weights)
    return labels[:, repeated_columns], seen[:, repeated_columns]


def _parse_threshold_schedule(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) == 3 and values[2] >= 2.0 and float(values[2]).is_integer():
        return np.linspace(values[0], values[1], int(values[2]), dtype=np.float32)
    if not values:
        raise SystemExit("--thres-connect must contain at least one threshold")
    return np.asarray(values, dtype=np.float32)


def _palette(labels: np.ndarray) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(71)
    table: dict[int, np.ndarray] = {}
    for value in np.unique(labels.astype(np.int64)):
        label = int(value)
        if label <= 0:
            table[label] = np.array([210, 210, 210], dtype=np.uint8)
        else:
            table[label] = rng.integers(35, 255, size=3, dtype=np.uint8)
    return table


def _write_labeled_mesh(path: Path, mesh: trimesh.Trimesh, vertex_labels: np.ndarray) -> None:
    labels = np.asarray(vertex_labels, dtype=np.int64)
    unique, inverse = np.unique(labels, return_inverse=True)
    rng = np.random.default_rng(71)
    table = rng.integers(35, 255, size=(unique.shape[0], 3), dtype=np.uint8)
    table[unique <= 0] = np.array([210, 210, 210], dtype=np.uint8)
    colors = np.empty((len(mesh.vertices), 4), dtype=np.uint8)
    colors[:, :3] = table[inverse]
    colors[:, 3] = 255
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    out.visual.vertex_colors = colors
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(path)


def _write_labeled_point_cloud(path: Path, points: np.ndarray, labels: np.ndarray) -> None:
    labels_i = np.asarray(labels, dtype=np.int64)
    unique, inverse = np.unique(labels_i, return_inverse=True)
    rng = np.random.default_rng(71)
    table = rng.integers(35, 255, size=(unique.shape[0], 3), dtype=np.uint8)
    table[unique <= 0] = np.array([210, 210, 210], dtype=np.uint8)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(table[inverse].astype(np.float64) / 255.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write point cloud: {path}")


def _write_scalar_mesh(path: Path, mesh: trimesh.Trimesh, vertex_values: np.ndarray) -> None:
    values = np.asarray(vertex_values, dtype=np.float32)
    valid = np.isfinite(values)
    colors = np.full((len(mesh.vertices), 4), 210, dtype=np.uint8)
    if np.any(valid):
        lo = float(np.percentile(values[valid], 2.0))
        hi = float(np.percentile(values[valid], 98.0))
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        gray = np.rint(normalized * 255.0).astype(np.uint8).reshape(-1, 1)
        mapped = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, 0, :]
        colors[:, :3] = cv2.cvtColor(mapped.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        colors[:, 3] = 255
        colors[~valid] = np.array([210, 210, 210, 255], dtype=np.uint8)
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)
    out.visual.vertex_colors = colors
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(path)


def _scalar_colors(values: np.ndarray) -> np.ndarray:
    values_f = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values_f)
    colors = np.full((values_f.shape[0], 3), 210, dtype=np.uint8)
    if np.any(valid):
        lo = float(np.percentile(values_f[valid], 2.0))
        hi = float(np.percentile(values_f[valid], 98.0))
        if hi <= lo:
            hi = lo + 1e-6
        normalized = np.clip((values_f - lo) / (hi - lo), 0.0, 1.0)
        gray = np.rint(normalized * 255.0).astype(np.uint8).reshape(-1, 1)
        mapped = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[:, 0, :]
        colors = cv2.cvtColor(mapped.reshape(-1, 1, 3), cv2.COLOR_BGR2RGB).reshape(-1, 3)
        colors[~valid] = np.array([210, 210, 210], dtype=np.uint8)
    return colors


def _write_scalar_point_cloud(path: Path, points: np.ndarray, values: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud.colors = o3d.utility.Vector3dVector(_scalar_colors(values).astype(np.float64) / 255.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
        raise RuntimeError(f"Failed to write scalar point cloud: {path}")


def _superpoint_size_stats(superpoint_labels: np.ndarray) -> dict[str, Any]:
    unique, counts = np.unique(np.asarray(superpoint_labels, dtype=np.int64), return_counts=True)
    return {
        "superpointCount": int(unique.shape[0]),
        "minPointsPerSuperpoint": int(np.min(counts)) if counts.size else 0,
        "medianPointsPerSuperpoint": float(np.median(counts)) if counts.size else 0.0,
        "meanPointsPerSuperpoint": float(np.mean(counts)) if counts.size else 0.0,
        "maxPointsPerSuperpoint": int(np.max(counts)) if counts.size else 0,
    }


def _write_superpoint_debug_outputs(
    *,
    paths: SAI3DPaths,
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    point_set: PointSet,
    superpoint_labels: np.ndarray,
) -> dict[str, Any]:
    paths.mesh_label_dir.mkdir(parents=True, exist_ok=True)
    superpoint_points_path = paths.mesh_label_dir / "superpoints_points.ply"
    _write_labeled_point_cloud(superpoint_points_path, points, superpoint_labels.astype(np.int32, copy=False) + 1)
    summary = {
        **_superpoint_size_stats(superpoint_labels),
        "superpointPointCloud": str(superpoint_points_path),
    }
    if point_set.source == "mesh_vertices" and superpoint_labels.shape[0] == len(mesh.vertices):
        superpoint_mesh_path = paths.mesh_label_dir / "superpoints_mesh.ply"
        _write_labeled_mesh(superpoint_mesh_path, mesh, superpoint_labels.astype(np.int32, copy=False) + 1)
        summary["superpointMesh"] = str(superpoint_mesh_path)
    return summary


def _write_observation_debug_outputs(
    *,
    paths: SAI3DPaths,
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    point_set: PointSet,
    superpoint_labels: np.ndarray,
    point_labels: np.ndarray,
    point_seen: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(superpoint_labels, dtype=np.int64)
    unique = np.unique(labels)
    seen_count = np.zeros(unique.shape[0], dtype=np.int64)
    positive_count = np.zeros(unique.shape[0], dtype=np.int64)
    visible_frame_count = np.zeros(unique.shape[0], dtype=np.int32)
    positive_frame_count = np.zeros(unique.shape[0], dtype=np.int32)

    for out_id, superpoint_id in enumerate(unique):
        members = np.nonzero(labels == int(superpoint_id))[0]
        seen = point_seen[members]
        positive = point_labels[members] > 0
        seen_count[out_id] = int(np.count_nonzero(seen))
        positive_count[out_id] = int(np.count_nonzero(positive & seen))
        visible_frame_count[out_id] = int(np.count_nonzero(np.any(seen, axis=0)))
        positive_frame_count[out_id] = int(np.count_nonzero(np.any(positive & seen, axis=0)))

    positive_ratio = positive_count.astype(np.float32) / np.maximum(seen_count.astype(np.float32), 1.0)
    point_positive_ratio = np.zeros(labels.shape[0], dtype=np.float32)
    for out_id, superpoint_id in enumerate(unique):
        point_positive_ratio[labels == int(superpoint_id)] = positive_ratio[out_id]

    support_points_path = paths.mesh_label_dir / "superpoint_positive_support_points.ply"
    _write_scalar_point_cloud(support_points_path, points, point_positive_ratio)
    support_npz_path = paths.mesh_label_dir / "superpoint_observation_support.npz"
    np.savez_compressed(
        support_npz_path,
        superpoint_ids=unique.astype(np.int32, copy=False),
        seen_count=seen_count.astype(np.int64, copy=False),
        positive_count=positive_count.astype(np.int64, copy=False),
        visible_frame_count=visible_frame_count.astype(np.int32, copy=False),
        positive_frame_count=positive_frame_count.astype(np.int32, copy=False),
        positive_ratio=positive_ratio.astype(np.float32, copy=False),
    )

    supported = positive_count > 0
    summary = {
        "superpointCount": int(unique.shape[0]),
        "observedSuperpointCount": int(np.count_nonzero(seen_count > 0)),
        "positiveSuperpointCount": int(np.count_nonzero(supported)),
        "positiveSuperpointFraction": float(np.count_nonzero(supported) / max(unique.shape[0], 1)),
        "meanPositiveRatio": float(np.mean(positive_ratio[supported])) if np.any(supported) else 0.0,
        "medianPositiveRatio": float(np.median(positive_ratio[supported])) if np.any(supported) else 0.0,
        "supportArrays": str(support_npz_path),
        "positiveSupportPointCloud": str(support_points_path),
    }
    if point_set.source == "mesh_vertices" and point_positive_ratio.shape[0] == len(mesh.vertices):
        support_mesh_path = paths.mesh_label_dir / "superpoint_positive_support_mesh.ply"
        _write_scalar_mesh(support_mesh_path, mesh, point_positive_ratio)
        summary["positiveSupportMesh"] = str(support_mesh_path)
    return summary


def _face_majority_labels(faces: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    face_labels = np.zeros(faces.shape[0], dtype=np.int32)
    for idx, face in enumerate(np.asarray(faces, dtype=np.int64)):
        values = vertex_labels[face]
        values = values[values > 0]
        if values.size == 0:
            continue
        labels, counts = np.unique(values, return_counts=True)
        face_labels[idx] = int(labels[int(np.argmax(counts))])
    return face_labels


def _majority_positive(values: np.ndarray) -> int:
    positive = np.asarray(values, dtype=np.int64)
    positive = positive[positive > 0]
    if positive.size == 0:
        return 0
    labels, counts = np.unique(positive, return_counts=True)
    return int(labels[int(np.argmax(counts))])


def _map_point_labels_to_mesh_vertices(
    *,
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    point_labels: np.ndarray,
    point_set: PointSet,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = np.asarray(point_labels, dtype=np.int32)
    if point_set.source == "mesh_vertices" and labels.shape[0] == vertices.shape[0]:
        vertex_labels = labels.astype(np.int32, copy=False)
        return vertex_labels, _face_majority_labels(faces, vertex_labels)

    face_labels = np.zeros(faces.shape[0], dtype=np.int32)
    face_ids = point_set.face_indices
    if face_ids is not None and face_ids.shape[0] == labels.shape[0]:
        order = np.argsort(face_ids)
        sorted_faces = face_ids[order]
        sorted_labels = labels[order]
        start = 0
        while start < sorted_faces.shape[0]:
            end = start + 1
            while end < sorted_faces.shape[0] and sorted_faces[end] == sorted_faces[start]:
                end += 1
            face_labels[int(sorted_faces[start])] = _majority_positive(sorted_labels[start:end])
            start = end

    vertex_faces = np.asarray(mesh.vertex_faces, dtype=np.int64)
    vertex_labels = np.zeros(vertices.shape[0], dtype=np.int32)
    if vertex_faces.size:
        for vertex_id, incident in enumerate(vertex_faces):
            incident = incident[incident >= 0]
            if incident.size:
                vertex_labels[vertex_id] = _majority_positive(face_labels[incident])

    missing = vertex_labels <= 0
    positive_points = labels > 0
    if np.any(missing) and np.any(positive_points):
        tree = scipy.spatial.cKDTree(np.asarray(points, dtype=np.float32)[positive_points])
        positive_labels = labels[positive_points]
        _, nearest = tree.query(vertices[missing], k=1, workers=-1)
        vertex_labels[missing] = positive_labels[np.asarray(nearest, dtype=np.int64)]
        face_labels = _face_majority_labels(faces, vertex_labels)
    return vertex_labels.astype(np.int32, copy=False), face_labels.astype(np.int32, copy=False)


def _assign_local_mask_majority(
    *,
    local: np.ndarray,
    local_id: int,
    proj: dict[str, Any],
    vertex_labels: np.ndarray,
    min_vertices: int,
    min_majority: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    seen_inside = np.asarray(proj["labels"]) == int(local_id)
    if not np.any(seen_inside):
        return np.zeros(local.shape, dtype=np.uint16), {
            "localLabel": int(local_id),
            "globalLabel": 0,
            "supportVertices": 0,
            "majority": 0.0,
        }
    candidate_vertices = np.asarray(proj["indices"])[seen_inside]
    values = vertex_labels[candidate_vertices]
    values = values[values > 0]
    if values.size < int(min_vertices):
        return np.zeros(local.shape, dtype=np.uint16), {
            "localLabel": int(local_id),
            "globalLabel": 0,
            "supportVertices": int(values.size),
            "majority": 0.0,
        }
    labels, counts = np.unique(values, return_counts=True)
    best = int(labels[int(np.argmax(counts))])
    majority = float(np.max(counts) / max(int(np.sum(counts)), 1))
    if majority < float(min_majority):
        best = 0
    out = np.zeros(local.shape, dtype=np.uint16)
    if best > 0:
        out[local == int(local_id)] = np.uint16(min(best, np.iinfo(np.uint16).max))
    return out, {
        "localLabel": int(local_id),
        "globalLabel": int(best),
        "supportVertices": int(values.size),
        "majority": float(majority),
    }


def _positive_label_hist(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.asarray(values, dtype=np.int64)
    positive = positive[positive > 0]
    if positive.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.unique(positive, return_counts=True)


def _best_positive_label(values: np.ndarray) -> tuple[int, int, float]:
    labels, counts = _positive_label_hist(values)
    if labels.size == 0:
        return 0, 0, 0.0
    best_idx = int(np.argmax(counts))
    total = int(np.sum(counts))
    return int(labels[best_idx]), int(counts[best_idx]), float(counts[best_idx] / max(total, 1))


def _adaptive_split_splat_erode(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return binary
    area_ratio = float(np.count_nonzero(binary) / max(binary.size, 1))
    if area_ratio < 0.05:
        kernel_size, iterations = 3, 2
    elif area_ratio < 0.30:
        kernel_size, iterations = 5, 2
    else:
        kernel_size, iterations = 5, 3
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=iterations) > 0
    return eroded if np.any(eroded) else binary


def _assign_local_mask_split_splat(
    *,
    local: np.ndarray,
    local_id: int,
    projected: np.ndarray,
    point_support: np.ndarray,
    min_support_pixels: int,
    min_majority: float,
    min_support_ratio: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mask = np.asarray(local) == int(local_id)
    out = np.zeros(local.shape, dtype=np.uint16)
    rejected = np.zeros(local.shape, dtype=np.uint16)
    if not np.any(mask):
        return out, rejected, {
            "localLabel": int(local_id),
            "globalLabel": 0,
            "decision": "empty",
            "supportSource": "none",
            "supportPixels": 0,
            "majority": 0.0,
            "supportRatio": 0.0,
        }

    eroded = _adaptive_split_splat_erode(mask)
    eroded_area = int(np.count_nonzero(eroded))
    point_values = np.asarray(point_support)[eroded]
    point_values = point_values[point_values > 0]
    geometry_values = np.asarray(projected)[eroded]
    geometry_values = geometry_values[geometry_values > 0]

    support_source = "point"
    values = point_values
    if values.size < int(min_support_pixels):
        support_source = "geometry"
        values = geometry_values

    best, best_count, majority = _best_positive_label(values)
    support_pixels = int(values.size)
    support_ratio = float(support_pixels / max(eroded_area, 1))
    accepted = (
        best > 0
        and support_pixels >= int(min_support_pixels)
        and majority >= float(min_majority)
        and support_ratio >= float(min_support_ratio)
    )
    decision = "accepted" if accepted else "rejected"
    if accepted:
        out[mask] = np.uint16(min(best, np.iinfo(np.uint16).max))
    else:
        rejected[mask] = np.uint16(min(int(local_id), np.iinfo(np.uint16).max))

    return out, rejected, {
        "localLabel": int(local_id),
        "globalLabel": int(best) if accepted else 0,
        "dominantCandidateLabel": int(best),
        "decision": decision,
        "supportSource": support_source,
        "supportPixels": int(support_pixels),
        "dominantSupportPixels": int(best_count),
        "majority": float(majority),
        "supportRatio": float(support_ratio),
        "proposalPixels": int(np.count_nonzero(mask)),
        "erodedPixels": int(eroded_area),
        "pointSupportPixels": int(point_values.size),
        "geometrySupportPixels": int(geometry_values.size),
    }


def _build_split_splat_proposal_relabel(
    *,
    local: np.ndarray,
    projected: np.ndarray,
    point_support: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    proposal_relabel = np.zeros(local.shape, dtype=np.uint16)
    rejected = np.zeros(local.shape, dtype=np.uint16)
    assignments: list[dict[str, Any]] = []
    for local_label in np.unique(local):
        local_id = int(local_label)
        if local_id <= 0:
            continue
        assigned, rejected_part, assignment = _assign_local_mask_split_splat(
            local=local,
            local_id=local_id,
            projected=projected,
            point_support=point_support,
            min_support_pixels=int(args.view_mask_split_splat_min_support_pixels),
            min_majority=float(args.view_mask_split_splat_min_majority),
            min_support_ratio=float(args.view_mask_split_splat_min_support_ratio),
        )
        fill = assigned > 0
        proposal_relabel[fill] = assigned[fill]
        reject_fill = rejected_part > 0
        rejected[reject_fill] = rejected_part[reject_fill]
        assignments.append(assignment)
    return proposal_relabel, rejected, assignments


def _overlay(rgb_bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.52) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    out = rgb.astype(np.float32, copy=True)
    table = _palette(labels)
    for label, color in table.items():
        if label <= 0:
            continue
        mask = labels == label
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _resolve_sam2_root(path: Path | None) -> Path:
    root = THIRD_PARTY_ROOT / "sam2" if path is None else path.expanduser().resolve()
    _require_dir(root, "SAM2 root")
    _require_file(root / "sam2" / "build_sam.py", "SAM2 build_sam.py")
    return root


def _resolve_sam2_checkpoint(sam2_root: Path, checkpoint: Path | None) -> Path:
    ckpt = checkpoint.expanduser().resolve() if checkpoint is not None else sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    return _require_file(ckpt, "SAM2 checkpoint")


def _build_sam2_image_predictor(args: argparse.Namespace) -> tuple[Any, str, str, str, Any]:
    sam2_root = _resolve_sam2_root(args.view_mask_sam2_root)
    checkpoint = _resolve_sam2_checkpoint(sam2_root, args.view_mask_sam2_checkpoint)
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_context: Any = nullcontext()
    if device.type == "cuda":
        amp_context = torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    amp_context.__enter__()
    model = build_sam2(str(args.view_mask_sam2_config), str(checkpoint), device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor, str(sam2_root), str(checkpoint), str(args.view_mask_sam2_config), amp_context


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return float(np.count_nonzero(a & b) / union)


def _mask_recall(candidate: np.ndarray, reference: np.ndarray) -> float:
    ref_count = np.count_nonzero(reference)
    if ref_count == 0:
        return 0.0
    return float(np.count_nonzero(np.asarray(candidate, dtype=bool) & np.asarray(reference, dtype=bool)) / ref_count)


def _erode_binary(mask: np.ndarray, erode_px: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if int(erode_px) <= 0 or not np.any(binary):
        return binary
    kernel = np.ones((2 * int(erode_px) + 1, 2 * int(erode_px) + 1), dtype=np.uint8)
    eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1) > 0
    return eroded if np.any(eroded) else binary


def _greedy_coreset_2d(points_xy: np.ndarray, count: int, alpha: float) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0 or int(count) <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    if points.shape[0] <= int(count):
        return points.astype(np.float32, copy=False)

    keep = np.ones(points.shape[0], dtype=bool)
    if points.shape[0] > 3:
        try:
            hull = scipy.spatial.ConvexHull(points)
            keep[np.asarray(hull.vertices, dtype=np.int64)] = False
            if np.count_nonzero(keep) >= int(count):
                points = points[keep]
        except scipy.spatial.QhullError:
            pass

    mean = np.mean(points, axis=0)
    selected = np.zeros((int(count), 2), dtype=np.float32)
    first = int(np.argmin(np.linalg.norm(points - mean[None, :], axis=1)))
    selected[0] = points[first]
    available = np.ones(points.shape[0], dtype=bool)
    available[first] = False
    min_dist = np.linalg.norm(points - selected[0][None, :], axis=1)
    dist_from_mean = np.linalg.norm(points - mean[None, :], axis=1)

    out_count = 1
    while out_count < int(count) and np.any(available):
        score = min_dist - float(alpha) * dist_from_mean
        score[~available] = -np.inf
        nxt = int(np.argmax(score))
        selected[out_count] = points[nxt]
        available[nxt] = False
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[nxt][None, :], axis=1))
        out_count += 1
    return selected[:out_count].astype(np.float32, copy=False)


def _sample_prompt_points(
    *,
    label: int,
    geometry_mask: np.ndarray,
    proj: dict[str, Any],
    point_labels: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = geometry_mask.shape
    eroded = _erode_binary(geometry_mask, int(args.view_mask_prompt_erode_px))
    indices = np.asarray(proj["indices"], dtype=np.int64)
    u = np.asarray(proj["u"], dtype=np.int32)
    v = np.asarray(proj["v"], dtype=np.int32)
    candidates = np.zeros((0, 2), dtype=np.float32)
    if indices.size:
        graph_labels = point_labels[indices]
        match = graph_labels == int(label)
        match &= (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if np.any(match):
            inside = eroded[v[match], u[match]]
            if np.any(inside):
                uv = np.stack([u[match][inside], v[match][inside]], axis=1)
                candidates = uv.astype(np.float32, copy=False)

    if candidates.shape[0] < int(args.view_mask_sam2_min_prompt_points):
        ys, xs = np.nonzero(eroded)
        if xs.size:
            sample_limit = min(xs.size, max(5000, int(args.view_mask_sam2_prompt_count) * 200))
            if xs.size > sample_limit:
                rng = np.random.default_rng(int(label) * 7919 + 71)
                chosen = rng.choice(xs.size, size=sample_limit, replace=False)
                xs = xs[chosen]
                ys = ys[chosen]
            candidates = np.stack([xs, ys], axis=1).astype(np.float32, copy=False)

    return _greedy_coreset_2d(
        candidates,
        count=int(args.view_mask_sam2_prompt_count),
        alpha=float(args.view_mask_sam2_prompt_alpha),
    )


def _render_point_support_label_map(
    *,
    proj: dict[str, Any],
    point_labels: np.ndarray,
    width: int,
    height: int,
    radius: int,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint16)
    indices = np.asarray(proj["indices"], dtype=np.int64)
    if indices.size == 0:
        return out
    u = np.asarray(proj["u"], dtype=np.int32)
    v = np.asarray(proj["v"], dtype=np.int32)
    z = np.asarray(proj.get("z", np.zeros(indices.shape[0], dtype=np.float32)), dtype=np.float32)
    labels = point_labels[indices]
    draw = labels > 0
    if not np.any(draw):
        return out
    order = np.nonzero(draw)[0][np.argsort(z[draw])[::-1]]
    r = max(int(radius), 0)
    for item in order:
        label = int(labels[item])
        x = int(u[item])
        y = int(v[item])
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        x0, x1 = max(0, x - r), min(width, x + r + 1)
        y0, y1 = max(0, y - r), min(height, y + r + 1)
        out[y0:y1, x0:x1] = np.uint16(min(label, np.iinfo(np.uint16).max))
    return out


def _draw_prompt_overlay(
    rgb_bgr: np.ndarray,
    geometry_labels: np.ndarray,
    prompt_records: list[dict[str, Any]],
) -> np.ndarray:
    out = _overlay(rgb_bgr, geometry_labels, alpha=0.32)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for record in prompt_records:
        pts = np.asarray(record.get("points", []), dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue
        for x_f, y_f in pts:
            cv2.circle(bgr, (int(round(float(x_f))), int(round(float(y_f)))), 4, (35, 255, 35), thickness=-1)
            cv2.circle(bgr, (int(round(float(x_f))), int(round(float(y_f)))), 6, (255, 255, 255), thickness=1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _predict_sam2_mask(predictor: Any, points_xy: np.ndarray) -> np.ndarray | None:
    if points_xy.shape[0] == 0:
        return None
    labels = np.ones(points_xy.shape[0], dtype=np.int32)
    masks, _scores, _logits = predictor.predict(
        point_coords=points_xy.astype(np.float32, copy=False),
        point_labels=labels,
        multimask_output=False,
    )
    masks_arr = np.asarray(masks)
    if masks_arr.ndim == 3:
        return masks_arr[0].astype(bool, copy=False)
    if masks_arr.ndim == 2:
        return masks_arr.astype(bool, copy=False)
    return None


def _assemble_scored_masks(candidates: list[tuple[int, np.ndarray, float]]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 0), dtype=np.uint16)
    height, width = candidates[0][1].shape
    labels = np.zeros((height, width), dtype=np.uint16)
    scores = np.full((height, width), -np.inf, dtype=np.float32)
    for label, mask, score in candidates:
        if int(label) <= 0 or not np.any(mask):
            continue
        confidence = float(score)
        update = np.asarray(mask, dtype=bool) & (confidence >= scores)
        labels[update] = np.uint16(min(int(label), np.iinfo(np.uint16).max))
        scores[update] = confidence
    return labels


def _write_view_masks(
    *,
    paths: SAI3DPaths,
    mesh: trimesh.Trimesh,
    rows: list[dict[str, Any]],
    projection_cache: list[dict[str, Any]],
    point_labels: np.ndarray,
    face_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if paths.view_mask_dir.exists() and bool(args.overwrite):
        shutil.rmtree(paths.view_mask_dir)
    for directory in [
        paths.view_mask_masks_dir,
        paths.view_mask_overlay_dir,
        paths.view_mask_proposal_masks_dir,
        paths.view_mask_proposal_overlay_dir,
        paths.view_mask_reject_masks_dir,
        paths.view_mask_reject_overlay_dir,
        paths.view_mask_projected_masks_dir,
        paths.view_mask_projected_overlay_dir,
        paths.view_mask_point_support_masks_dir,
        paths.view_mask_point_support_overlay_dir,
        paths.view_mask_prompt_overlay_dir,
        paths.view_mask_sam2_masks_dir,
        paths.view_mask_sam2_overlay_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
    scene = _build_raycast_scene(paths.scene_mesh)
    refine_mode = str(args.view_mask_refine_mode)
    sam_predictor: Any | None = None
    sam_context: Any | None = None
    sam2_meta: dict[str, Any] = {}
    if refine_mode in {"sam2", "split_splat"}:
        sam_predictor, sam2_root, sam2_checkpoint, sam2_config, sam_context = _build_sam2_image_predictor(args)
        sam2_meta = {
            "sam2Root": sam2_root,
            "sam2Checkpoint": sam2_checkpoint,
            "sam2Config": sam2_config,
        }

    pred_rows: list[dict[str, Any]] = []
    aggregate_counts = {
        "labelsConsidered": 0,
        "sam2Attempted": 0,
        "sam2Accepted": 0,
        "proposalSelected": 0,
        "proposalAccepted": 0,
        "proposalRejected": 0,
        "geometrySelected": 0,
    }
    try:
        for row, proj in zip(rows, projection_cache, strict=False):
            frame_id = int(row["sai3dFrameId"])
            width = int(row["width"])
            height = int(row["height"])
            local = _load_label_map(paths.dataset_dir / str(row["maskPath"]), expected_shape=(height, width))
            pose = np.loadtxt(paths.dataset_dir / str(row["posePath"]), dtype=np.float64)
            projected = _render_face_label_map(
                scene,
                face_labels,
                pose_world_t_cam=pose,
                fx=float(row["fx"]),
                fy=float(row["fy"]),
                cx=float(row["cx"]),
                cy=float(row["cy"]),
                width=width,
                height=height,
                chunk_rows=int(args.raycast_chunk_rows),
            )
            point_support = _render_point_support_label_map(
                proj=proj,
                point_labels=point_labels,
                width=width,
                height=height,
                radius=int(args.view_mask_point_radius),
            )
            proposal_relabel = np.zeros(local.shape, dtype=np.uint16)
            rejected_proposals = np.zeros(local.shape, dtype=np.uint16)
            assignments: list[dict[str, Any]] = []
            if refine_mode == "split_splat":
                proposal_relabel, rejected_proposals, assignments = _build_split_splat_proposal_relabel(
                    local=local,
                    projected=projected,
                    point_support=point_support,
                    args=args,
                )
                aggregate_counts["proposalAccepted"] += int(sum(1 for item in assignments if item.get("decision") == "accepted"))
                aggregate_counts["proposalRejected"] += int(sum(1 for item in assignments if item.get("decision") == "rejected"))
            else:
                for local_label in np.unique(local):
                    local_id = int(local_label)
                    if local_id <= 0:
                        continue
                    assigned, assignment = _assign_local_mask_majority(
                        local=local,
                        local_id=local_id,
                        proj=proj,
                        vertex_labels=point_labels,
                        min_vertices=int(args.view_mask_min_vertices),
                        min_majority=float(args.view_mask_min_majority),
                    )
                    fill = assigned > 0
                    proposal_relabel[fill] = assigned[fill]
                    assignments.append(assignment)

            rgb_bgr = cv2.imread(str(paths.dataset_dir / str(row["colorPath"])), cv2.IMREAD_COLOR)
            if sam_predictor is not None and rgb_bgr is not None:
                sam_predictor.set_image(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))

            raw_sam_candidates: list[tuple[int, np.ndarray, float]] = []
            final_candidates: list[tuple[int, np.ndarray, float]] = []
            prompt_records: list[dict[str, Any]] = []
            refinements: list[dict[str, Any]] = []

            label_values = np.unique(np.concatenate([np.unique(projected), np.unique(proposal_relabel), np.unique(point_support)]))
            labels_positive = [int(value) for value in label_values if int(value) > 0]
            labels_positive.sort(key=lambda value: int(np.count_nonzero(projected == value)), reverse=True)
            if int(args.view_mask_max_labels_per_view) > 0:
                labels_positive = labels_positive[: int(args.view_mask_max_labels_per_view)]

            for label in labels_positive:
                geometry_mask = projected == int(label)
                proposal_mask = proposal_relabel == int(label)
                point_mask = point_support == int(label)
                if not np.any(geometry_mask) and np.any(proposal_mask):
                    geometry_mask = proposal_mask.copy()
                elif not np.any(geometry_mask) and np.any(point_mask):
                    geometry_mask = point_mask.copy()
                if not np.any(geometry_mask):
                    continue
                aggregate_counts["labelsConsidered"] += 1

                proposal_iou = _mask_iou(proposal_mask, geometry_mask)
                proposal_recall = _mask_recall(proposal_mask, geometry_mask)
                proposal_point_recall = _mask_recall(proposal_mask, point_mask)
                prompt_points = np.zeros((0, 2), dtype=np.float32)
                sam_mask: np.ndarray | None = None
                sam_iou = 0.0
                sam_recall = 0.0
                sam_area_ratio = 0.0
                sam_accepted = False
                geometry_area = int(np.count_nonzero(geometry_mask))
                proposal_area = int(np.count_nonzero(proposal_mask))
                proposal_area_ratio = float(proposal_area / max(geometry_area, 1))
                proposal_accepted = (
                    refine_mode == "split_splat"
                    and proposal_area > 0
                    and proposal_area_ratio <= float(args.view_mask_split_splat_max_proposal_area_ratio)
                    and (
                        proposal_iou >= float(args.view_mask_proposal_min_iou)
                        or proposal_point_recall >= float(args.view_mask_split_splat_min_point_recall)
                    )
                )
                should_prompt_sam = refine_mode == "sam2" or (
                    refine_mode == "split_splat" and not proposal_accepted
                )

                if (
                    should_prompt_sam
                    and sam_predictor is not None
                    and rgb_bgr is not None
                    and geometry_area >= int(args.view_mask_sam2_min_geometry_area)
                ):
                    prompt_points = _sample_prompt_points(
                        label=label,
                        geometry_mask=geometry_mask,
                        proj=proj,
                        point_labels=point_labels,
                        args=args,
                    )
                    prompt_records.append({"label": int(label), "points": prompt_points.tolist()})
                    if prompt_points.shape[0] >= int(args.view_mask_sam2_min_prompt_points):
                        aggregate_counts["sam2Attempted"] += 1
                        sam_mask = _predict_sam2_mask(sam_predictor, prompt_points)
                        if sam_mask is not None:
                            sam_iou = _mask_iou(sam_mask, geometry_mask)
                            sam_recall = _mask_recall(sam_mask, geometry_mask)
                            sam_area_ratio = float(np.count_nonzero(sam_mask) / max(geometry_area, 1))
                            sam_accepted = (
                                sam_iou >= float(args.view_mask_sam2_min_iou)
                                and sam_recall >= float(args.view_mask_sam2_min_geometry_recall)
                                and sam_area_ratio <= float(args.view_mask_sam2_max_area_ratio)
                                and (sam_iou + float(args.view_mask_sam2_prefer_margin)) >= proposal_iou
                            )
                            raw_sam_candidates.append((label, sam_mask, max(float(sam_iou), 1e-4)))

                if proposal_accepted:
                    selected_mask = proposal_mask
                    selected_kind = "split_splat_proposal"
                    selected_score = 0.80 + min(max(float(proposal_iou), float(proposal_point_recall)), 1.0)
                    aggregate_counts["proposalSelected"] += 1
                elif sam_mask is not None and sam_accepted:
                    selected_mask = sam_mask
                    selected_kind = "sam2_refined"
                    selected_score = 0.70 + min(float(sam_iou), 1.0)
                    aggregate_counts["sam2Accepted"] += 1
                elif refine_mode != "geometry" and proposal_area > 0 and proposal_iou >= float(args.view_mask_proposal_min_iou):
                    selected_mask = proposal_mask
                    selected_kind = "proposal_relabel"
                    selected_score = 0.55 + min(float(proposal_iou), 1.0)
                    aggregate_counts["proposalSelected"] += 1
                else:
                    selected_mask = geometry_mask
                    selected_kind = "geometry_projected"
                    selected_score = 0.40 + min(float(_mask_recall(geometry_mask, proposal_mask)) if proposal_area > 0 else 0.0, 0.25)
                    aggregate_counts["geometrySelected"] += 1
                final_candidates.append((label, selected_mask, selected_score))
                refinements.append(
                    {
                        "globalLabel": int(label),
                        "geometryArea": int(geometry_area),
                        "proposalArea": int(proposal_area),
                        "proposalGeometryIoU": float(proposal_iou),
                        "proposalGeometryRecall": float(proposal_recall),
                        "proposalPointRecall": float(proposal_point_recall),
                        "proposalAreaRatio": float(proposal_area_ratio),
                        "proposalAccepted": bool(proposal_accepted),
                        "promptCount": int(prompt_points.shape[0]),
                        "sam2Attempted": bool(sam_mask is not None),
                        "sam2GeometryIoU": float(sam_iou),
                        "sam2GeometryRecall": float(sam_recall),
                        "sam2AreaRatio": float(sam_area_ratio),
                        "selected": selected_kind,
                    }
                )

            sam2_refined = _assemble_scored_masks(raw_sam_candidates) if raw_sam_candidates else np.zeros((height, width), dtype=np.uint16)
            final_mask = _assemble_scored_masks(final_candidates) if final_candidates else np.zeros((height, width), dtype=np.uint16)

            png_path = paths.view_mask_masks_dir / f"{frame_id:06d}.png"
            npy_path = paths.view_mask_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(png_path), final_mask)
            np.save(npy_path, final_mask)

            proposal_png_path = paths.view_mask_proposal_masks_dir / f"{frame_id:06d}.png"
            proposal_npy_path = paths.view_mask_proposal_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(proposal_png_path), proposal_relabel)
            np.save(proposal_npy_path, proposal_relabel)

            reject_png_path = paths.view_mask_reject_masks_dir / f"{frame_id:06d}.png"
            reject_npy_path = paths.view_mask_reject_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(reject_png_path), rejected_proposals)
            np.save(reject_npy_path, rejected_proposals)

            projected_png_path = paths.view_mask_projected_masks_dir / f"{frame_id:06d}.png"
            projected_npy_path = paths.view_mask_projected_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(projected_png_path), projected)
            np.save(projected_npy_path, projected)

            point_support_png_path = paths.view_mask_point_support_masks_dir / f"{frame_id:06d}.png"
            point_support_npy_path = paths.view_mask_point_support_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(point_support_png_path), point_support)
            np.save(point_support_npy_path, point_support)

            sam2_png_path = paths.view_mask_sam2_masks_dir / f"{frame_id:06d}.png"
            sam2_npy_path = paths.view_mask_sam2_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(sam2_png_path), sam2_refined)
            np.save(sam2_npy_path, sam2_refined)

            overlay_path = paths.view_mask_overlay_dir / f"{frame_id:06d}.png"
            proposal_overlay_path = paths.view_mask_proposal_overlay_dir / f"{frame_id:06d}.png"
            reject_overlay_path = paths.view_mask_reject_overlay_dir / f"{frame_id:06d}.png"
            projected_overlay_path = paths.view_mask_projected_overlay_dir / f"{frame_id:06d}.png"
            point_support_overlay_path = paths.view_mask_point_support_overlay_dir / f"{frame_id:06d}.png"
            prompt_overlay_path = paths.view_mask_prompt_overlay_dir / f"{frame_id:06d}.png"
            sam2_overlay_path = paths.view_mask_sam2_overlay_dir / f"{frame_id:06d}.png"
            if rgb_bgr is not None:
                cv2.imwrite(str(overlay_path), cv2.cvtColor(_overlay(rgb_bgr, final_mask), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(proposal_overlay_path), cv2.cvtColor(_overlay(rgb_bgr, proposal_relabel), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(reject_overlay_path), cv2.cvtColor(_overlay(rgb_bgr, rejected_proposals), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(projected_overlay_path), cv2.cvtColor(_overlay(rgb_bgr, projected), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(point_support_overlay_path), cv2.cvtColor(_overlay(rgb_bgr, point_support), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(prompt_overlay_path), cv2.cvtColor(_draw_prompt_overlay(rgb_bgr, projected, prompt_records), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(sam2_overlay_path), cv2.cvtColor(_overlay(rgb_bgr, sam2_refined), cv2.COLOR_RGB2BGR))

            final_label_count = int(np.count_nonzero(np.unique(final_mask) > 0))
            final_coverage = float(np.count_nonzero(final_mask > 0) / max(final_mask.size, 1))
            proposal_label_count = int(np.count_nonzero(np.unique(proposal_relabel) > 0))
            proposal_coverage = float(np.count_nonzero(proposal_relabel > 0) / max(proposal_relabel.size, 1))
            reject_label_count = int(np.count_nonzero(np.unique(rejected_proposals) > 0))
            reject_coverage = float(np.count_nonzero(rejected_proposals > 0) / max(rejected_proposals.size, 1))
            projected_label_count = int(np.count_nonzero(np.unique(projected) > 0))
            projected_coverage = float(np.count_nonzero(projected > 0) / max(projected.size, 1))
            point_support_label_count = int(np.count_nonzero(np.unique(point_support) > 0))
            point_support_coverage = float(np.count_nonzero(point_support > 0) / max(point_support.size, 1))
            sam2_label_count = int(np.count_nonzero(np.unique(sam2_refined) > 0))
            sam2_coverage = float(np.count_nonzero(sam2_refined > 0) / max(sam2_refined.size, 1))
            pred_rows.append(
                {
                    "sai3dFrameId": int(frame_id),
                    "sourceFrameId": int(row.get("sourceFrameId", frame_id)),
                    "scanID": row.get("scanID", ""),
                    "frameID": int(row.get("frameID", frame_id)),
                    "imageName": row.get("imageName", ""),
                    "maskPng": str(png_path),
                    "maskNpy": str(npy_path),
                    "overlay": str(overlay_path),
                    "labelCount": int(final_label_count),
                    "coverage": float(final_coverage),
                    "maskKind": (
                        "split_splat_proposal_depth_gated"
                        if refine_mode == "split_splat"
                        else "split_splat_geometry_gated_sam2"
                        if refine_mode == "sam2"
                        else "geometry_gated_projection"
                    ),
                    "proposalRelabelMaskPng": str(proposal_png_path),
                    "proposalRelabelMaskNpy": str(proposal_npy_path),
                    "proposalRelabelOverlay": str(proposal_overlay_path),
                    "proposalRelabelLabelCount": int(proposal_label_count),
                    "proposalRelabelCoverage": float(proposal_coverage),
                    "rejectedProposalMaskPng": str(reject_png_path),
                    "rejectedProposalMaskNpy": str(reject_npy_path),
                    "rejectedProposalOverlay": str(reject_overlay_path),
                    "rejectedProposalLabelCount": int(reject_label_count),
                    "rejectedProposalCoverage": float(reject_coverage),
                    "projected3dMaskPng": str(projected_png_path),
                    "projected3dMaskNpy": str(projected_npy_path),
                    "projected3dOverlay": str(projected_overlay_path),
                    "projected3dLabelCount": int(projected_label_count),
                    "projected3dCoverage": float(projected_coverage),
                    "pointSupportMaskPng": str(point_support_png_path),
                    "pointSupportMaskNpy": str(point_support_npy_path),
                    "pointSupportOverlay": str(point_support_overlay_path),
                    "pointSupportLabelCount": int(point_support_label_count),
                    "pointSupportCoverage": float(point_support_coverage),
                    "sam2RefinedMaskPng": str(sam2_png_path),
                    "sam2RefinedMaskNpy": str(sam2_npy_path),
                    "sam2RefinedOverlay": str(sam2_overlay_path),
                    "sam2RefinedLabelCount": int(sam2_label_count),
                    "sam2RefinedCoverage": float(sam2_coverage),
                    "promptOverlay": str(prompt_overlay_path),
                    "localAssignments": assignments,
                    "refinements": refinements,
                }
            )
            print(
                f"[sai3d view-mask {len(pred_rows):04d}/{len(rows):04d}] frame={frame_id} "
                f"final_labels={final_label_count} final_cov={final_coverage:.3f} "
                f"sam2_accept={sum(1 for item in refinements if item['selected'] == 'sam2_refined')}"
            )
    finally:
        if sam_context is not None:
            sam_context.__exit__(None, None, None)

    predictions = paths.view_mask_dir / "predictions.jsonl"
    _write_jsonl(predictions, pred_rows)
    summary = {
        "stage": "view_masks",
        "timestampUtc": _now(),
        "method": (
            "SAI3D proposal-gated finalizer inspired by Split&Splat: geometry-visible SAI3D labels, proposal assignment, SAM2 fallback, and geometry gating."
            if refine_mode == "split_splat"
            else "Split&Splat-inspired projection: geometry-visible SAI3D labels, SAM2 point prompts, and geometry-IoU gating."
        ),
        "frameCount": int(len(pred_rows)),
        "refineMode": refine_mode,
        "refinementCounts": aggregate_counts,
        "sam2": sam2_meta,
        "parameters": {
            "pointSupportRadius": int(args.view_mask_point_radius),
            "promptCount": int(args.view_mask_sam2_prompt_count),
            "promptErodePx": int(args.view_mask_prompt_erode_px),
            "promptAlpha": float(args.view_mask_sam2_prompt_alpha),
            "minPromptPoints": int(args.view_mask_sam2_min_prompt_points),
            "minGeometryArea": int(args.view_mask_sam2_min_geometry_area),
            "minGeometryIoU": float(args.view_mask_sam2_min_iou),
            "minGeometryRecall": float(args.view_mask_sam2_min_geometry_recall),
            "maxAreaRatio": float(args.view_mask_sam2_max_area_ratio),
            "proposalMinIoU": float(args.view_mask_proposal_min_iou),
            "preferMargin": float(args.view_mask_sam2_prefer_margin),
            "splitSplatMinSupportPixels": int(args.view_mask_split_splat_min_support_pixels),
            "splitSplatMinMajority": float(args.view_mask_split_splat_min_majority),
            "splitSplatMinSupportRatio": float(args.view_mask_split_splat_min_support_ratio),
            "splitSplatMinPointRecall": float(args.view_mask_split_splat_min_point_recall),
            "splitSplatMaxProposalAreaRatio": float(args.view_mask_split_splat_max_proposal_area_ratio),
            "maxLabelsPerView": int(args.view_mask_max_labels_per_view),
        },
        "coverage": {
            "min": float(min((row["coverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["coverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["coverage"] for row in pred_rows), default=0.0)),
        },
        "proposalRelabelCoverage": {
            "min": float(min((row["proposalRelabelCoverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["proposalRelabelCoverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["proposalRelabelCoverage"] for row in pred_rows), default=0.0)),
        },
        "rejectedProposalCoverage": {
            "min": float(min((row["rejectedProposalCoverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["rejectedProposalCoverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["rejectedProposalCoverage"] for row in pred_rows), default=0.0)),
        },
        "projected3dCoverage": {
            "min": float(min((row["projected3dCoverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["projected3dCoverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["projected3dCoverage"] for row in pred_rows), default=0.0)),
        },
        "sam2RefinedCoverage": {
            "min": float(min((row["sam2RefinedCoverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["sam2RefinedCoverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["sam2RefinedCoverage"] for row in pred_rows), default=0.0)),
        },
        "outputs": {
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskMasksDir": str(paths.view_mask_masks_dir),
            "viewMaskOverlayDir": str(paths.view_mask_overlay_dir),
            "proposalRelabelMaskDir": str(paths.view_mask_proposal_dir),
            "proposalRelabelMaskMasksDir": str(paths.view_mask_proposal_masks_dir),
            "proposalRelabelMaskOverlayDir": str(paths.view_mask_proposal_overlay_dir),
            "rejectedProposalMaskDir": str(paths.view_mask_reject_dir),
            "rejectedProposalMaskMasksDir": str(paths.view_mask_reject_masks_dir),
            "rejectedProposalMaskOverlayDir": str(paths.view_mask_reject_overlay_dir),
            "projected3dMaskDir": str(paths.view_mask_projected_dir),
            "projected3dMaskMasksDir": str(paths.view_mask_projected_masks_dir),
            "projected3dMaskOverlayDir": str(paths.view_mask_projected_overlay_dir),
            "pointSupportMaskDir": str(paths.view_mask_point_support_dir),
            "pointSupportMaskMasksDir": str(paths.view_mask_point_support_masks_dir),
            "pointSupportMaskOverlayDir": str(paths.view_mask_point_support_overlay_dir),
            "promptOverlayDir": str(paths.view_mask_prompt_overlay_dir),
            "sam2RefinedMaskDir": str(paths.view_mask_sam2_dir),
            "sam2RefinedMaskMasksDir": str(paths.view_mask_sam2_masks_dir),
            "sam2RefinedMaskOverlayDir": str(paths.view_mask_sam2_overlay_dir),
            "viewMaskPredictions": str(predictions),
        },
    }
    _write_json(paths.summary_dir / "view_masks_summary.json", summary)
    return summary


def _compute_observations(args: argparse.Namespace, paths: SAI3DPaths) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    sai3d_root = _resolve_sai3d_root(args.sai3d_root)
    args.sai3d_root = sai3d_root
    rows = _read_jsonl(_require_file(paths.frame_manifest, "SAI3D frame manifest"))
    mesh = _load_mesh(_require_file(paths.scene_mesh, "SAI3D staged mesh"))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    point_set = _build_point_set(mesh, args)
    points = point_set.points.astype(np.float32, copy=False)
    _write_points(paths.points_path, points)

    superpoint_labels, superpoint_summary = _build_superpoints(mesh, point_set, args)
    np.save(paths.superpoints_path, superpoint_labels.astype(np.int32, copy=False))
    superpoint_debug = _write_superpoint_debug_outputs(
        paths=paths,
        mesh=mesh,
        points=points,
        point_set=point_set,
        superpoint_labels=superpoint_labels,
    )
    superpoint_summary = {
        **superpoint_summary,
        **superpoint_debug,
    }
    _write_json(paths.summary_dir / "superpoints_summary.json", superpoint_summary)

    point_labels, point_seen, projection_cache, frame_stats = _project_vertices_to_masks(
        paths=paths,
        vertices=points,
        point_set=point_set,
        rows=rows,
        args=args,
    )
    if not np.any(point_seen):
        raise SystemExit(
            "SAI3D visibility projection found no visible graph points. "
            "Check the staged poses, intrinsics, rendered depth, and input mesh."
        )
    if not np.any(point_labels > 0):
        raise SystemExit(
            "SAI3D projection found visible points but no positive 2D proposal labels. "
            "Check the selected proposal source masks."
        )
    paths.mesh_label_dir.mkdir(parents=True, exist_ok=True)
    graph_rows = rows[:: max(int(args.sai3d_view_stride), 1)]
    view_weights = np.asarray(
        [max(int(row.get("viewWeight", 1)), 1) for row in graph_rows],
        dtype=np.int32,
    )
    np.savez_compressed(
        paths.observations_path,
        points=points.astype(np.float32, copy=False),
        point_labels=point_labels.astype(np.uint16, copy=False),
        point_seen=point_seen.astype(bool, copy=False),
        superpoint_labels=superpoint_labels.astype(np.int32, copy=False),
        point_face_indices=(
            point_set.face_indices.astype(np.int32, copy=False)
            if point_set.face_indices is not None
            else np.full(points.shape[0], -1, dtype=np.int32)
        ),
        point_barycentric=(
            point_set.barycentric.astype(np.float32, copy=False)
            if point_set.barycentric is not None
            else np.full((points.shape[0], 3), np.nan, dtype=np.float32)
        ),
        graph_frame_ids=np.asarray([int(row["sai3dFrameId"]) for row in graph_rows], dtype=np.int32),
        view_weights=view_weights,
    )
    observation_summary = _write_observation_debug_outputs(
        paths=paths,
        mesh=mesh,
        points=points,
        point_set=point_set,
        superpoint_labels=superpoint_labels,
        point_labels=point_labels,
        point_seen=point_seen,
    )
    outputs = {
        "datasetDir": str(paths.dataset_dir),
        "frameManifest": str(paths.frame_manifest),
        "meshLabelDir": str(paths.mesh_label_dir),
        "observations": str(paths.observations_path),
        "points": str(paths.points_path),
        "superpoints": str(paths.superpoints_path),
        "superpointPointCloud": str(paths.mesh_label_dir / "superpoints_points.ply"),
        "positiveSupportPointCloud": str(paths.mesh_label_dir / "superpoint_positive_support_points.ply"),
        "positiveSupportArrays": str(paths.mesh_label_dir / "superpoint_observation_support.npz"),
    }
    if "superpointMesh" in superpoint_summary:
        outputs["superpointMesh"] = str(paths.mesh_label_dir / "superpoints_mesh.ply")
    if "positiveSupportMesh" in observation_summary:
        outputs["positiveSupportMesh"] = str(paths.mesh_label_dir / "superpoint_positive_support_mesh.ply")
    summary = {
        "stage": "observe",
        "timestampUtc": _now(),
        "method": "SAI3D superpoint observation from multi-view 2D proposal masks",
        "modelDir": str(model_dir),
        "sai3dRoot": str(sai3d_root),
        "sceneName": paths.scene_name,
        "frameCount": int(len(rows)),
        "graphFrameCount": int(point_labels.shape[1]),
        "effectiveWeightedViewCount": int(view_weights.sum()),
        "manualAnchorFrameCount": int(sum(bool(row.get("isManualAnchor")) for row in graph_rows)),
        "pointCount": int(points.shape[0]),
        "pointSource": str(point_set.source),
        "vertexCount": int(vertices.shape[0]),
        "faceCount": int(faces.shape[0]),
        "parameters": {
            "pointSource": str(args.point_source),
            "pointSampleCount": int(args.point_sample_count),
            "pointSampleSeed": int(args.point_sample_seed),
            "pointCloud": str(args.point_cloud) if args.point_cloud is not None else "",
            "pointCloudMaxPoints": int(args.point_cloud_max_points),
            "pointCloudSampleSeed": int(args.point_cloud_sample_seed),
            "pointVisibilitySource": str(args.point_visibility_source),
            "resolvedPointVisibilitySource": _resolve_visibility_source(point_set, args),
            "pointZbufferDepthBand": float(args.point_zbuffer_depth_band),
            "superpointMode": str(args.superpoint_mode),
            "superpointTargetCount": int(args.superpoint_target_count),
            "superpointVoxelSize": float(args.superpoint_voxel_size),
            "graphMaxEdgeLength": float(args.graph_max_edge_length),
            "visibilityRtol": float(args.visibility_rtol),
            "visibilityAtol": float(args.visibility_atol),
            "sai3dViewStride": int(args.sai3d_view_stride),
            "viewWeighting": "integer_view_replication",
        },
        "pointSet": point_set.summary,
        "superpoints": superpoint_summary,
        "observations": observation_summary,
        "frameProjectionStats": frame_stats,
        "outputs": outputs,
    }
    _write_json(paths.summary_dir / "observation_summary.json", summary)
    return {
        "summary": summary,
        "rows": rows,
        "mesh": mesh,
        "points": points,
        "point_set": point_set,
        "vertices": vertices,
        "faces": faces,
        "superpoint_labels": superpoint_labels,
        "point_labels": point_labels,
        "point_seen": point_seen,
        "view_weights": view_weights,
        "projection_cache": projection_cache,
        "frame_stats": frame_stats,
    }


def observe(args: argparse.Namespace, paths: SAI3DPaths) -> dict[str, Any]:
    return _compute_observations(args, paths)["summary"]


def segment(args: argparse.Namespace, paths: SAI3DPaths) -> dict[str, Any]:
    observed = _compute_observations(args, paths)
    model_dir = args.model_dir.expanduser().resolve()
    sai3d_root = _resolve_sai3d_root(args.sai3d_root)
    rows = observed["rows"]
    mesh = observed["mesh"]
    points = observed["points"]
    point_set = observed["point_set"]
    vertices = observed["vertices"]
    faces = observed["faces"]
    superpoint_labels = observed["superpoint_labels"]
    point_labels = observed["point_labels"]
    point_seen = observed["point_seen"]
    view_weights = observed["view_weights"]
    projection_cache = observed["projection_cache"]
    frame_stats = observed["frame_stats"]

    thresholds = _parse_threshold_schedule(str(args.thres_connect))
    agent = OmegaSAI3DAgent(points, args, superpoint_labels)
    graph_point_labels, history = agent.assign_from_observations(
        point_labels=point_labels,
        point_seen=point_seen,
        thresholds=thresholds,
        view_weights=view_weights,
    )
    vertex_labels, face_labels = _map_point_labels_to_mesh_vertices(
        mesh=mesh,
        points=points,
        point_labels=graph_point_labels,
        point_set=point_set,
    )
    point_label_path = paths.mesh_label_dir / "point_labels.npy"
    vertex_label_path = paths.mesh_label_dir / "vertex_labels.npy"
    face_label_path = paths.mesh_label_dir / "face_labels.npy"
    np.save(point_label_path, graph_point_labels.astype(np.int32, copy=False))
    np.save(vertex_label_path, vertex_labels.astype(np.int32, copy=False))
    np.save(face_label_path, face_labels.astype(np.int32, copy=False))
    labeled_points_path = paths.mesh_label_dir / "labeled_points.ply"
    _write_labeled_point_cloud(labeled_points_path, points, graph_point_labels)
    labeled_mesh_path = paths.mesh_label_dir / "labeled_mesh.ply"
    _write_labeled_mesh(labeled_mesh_path, mesh, vertex_labels)

    unique_point_labels = np.unique(graph_point_labels)
    unique_vertex_labels = np.unique(vertex_labels)
    unique_face_labels = np.unique(face_labels)
    segment_outputs = {
        "meshLabelDir": str(paths.mesh_label_dir),
        "observations": str(paths.observations_path),
        "points": str(paths.points_path),
        "superpoints": str(paths.superpoints_path),
        "superpointPointCloud": str(paths.mesh_label_dir / "superpoints_points.ply"),
        "positiveSupportPointCloud": str(paths.mesh_label_dir / "superpoint_positive_support_points.ply"),
        "pointLabels": str(point_label_path),
        "labeledPoints": str(labeled_points_path),
        "vertexLabels": str(vertex_label_path),
        "faceLabels": str(face_label_path),
        "labeledMesh": str(labeled_mesh_path),
    }
    if point_set.source == "mesh_vertices":
        segment_outputs["superpointMesh"] = str(paths.mesh_label_dir / "superpoints_mesh.ply")
        segment_outputs["positiveSupportMesh"] = str(paths.mesh_label_dir / "superpoint_positive_support_mesh.ply")
    summary = {
        "stage": "segment",
        "timestampUtc": _now(),
        "method": "SAI3D multi-view mask affinity with OMeGa mesh superpoints",
        "modelDir": str(model_dir),
        "sai3dRoot": str(sai3d_root),
        "sceneName": paths.scene_name,
        "frameCount": int(len(rows)),
        "graphFrameCount": int(point_labels.shape[1]),
        "effectiveWeightedViewCount": int(view_weights.sum()),
        "manualAnchorFrameCount": int(np.count_nonzero(view_weights > 1)),
        "pointCount": int(points.shape[0]),
        "pointSource": str(point_set.source),
        "vertexCount": int(vertices.shape[0]),
        "faceCount": int(faces.shape[0]),
        "pointLabelCount": int(np.count_nonzero(unique_point_labels > 0)),
        "vertexLabelCount": int(np.count_nonzero(unique_vertex_labels > 0)),
        "faceLabelCount": int(np.count_nonzero(unique_face_labels > 0)),
        "labeledPointCount": int(np.count_nonzero(graph_point_labels > 0)),
        "labeledVertexCount": int(np.count_nonzero(vertex_labels > 0)),
        "parameters": {
            "pointSource": str(args.point_source),
            "pointSampleCount": int(args.point_sample_count),
            "pointSampleSeed": int(args.point_sample_seed),
            "pointCloud": str(args.point_cloud) if args.point_cloud is not None else "",
            "pointCloudMaxPoints": int(args.point_cloud_max_points),
            "pointCloudSampleSeed": int(args.point_cloud_sample_seed),
            "pointVisibilitySource": str(args.point_visibility_source),
            "resolvedPointVisibilitySource": _resolve_visibility_source(point_set, args),
            "pointZbufferDepthBand": float(args.point_zbuffer_depth_band),
            "superpointMode": str(args.superpoint_mode),
            "superpointTargetCount": int(args.superpoint_target_count),
            "superpointVoxelSize": float(args.superpoint_voxel_size),
            "graphMaxEdgeLength": float(args.graph_max_edge_length),
            "fromPointsThreshold": float(args.from_points_thres),
            "thresConnect": [float(v) for v in thresholds],
            "thresMerge": int(args.thres_merge),
            "maxNeighborDistance": int(args.max_neighbor_distance),
            "similarMetric": str(args.similar_metric),
            "thresTrunc": float(args.thres_trunc),
            "visibilityRtol": float(args.visibility_rtol),
            "visibilityAtol": float(args.visibility_atol),
            "sai3dViewStride": int(args.sai3d_view_stride),
            "viewWeighting": "integer_view_replication",
        },
        "superpoints": observed["summary"].get("superpoints", {}),
        "observations": observed["summary"].get("observations", {}),
        "history": history,
        "frameProjectionStats": frame_stats,
        "outputs": segment_outputs,
    }
    _write_json(paths.summary_dir / "segment_summary.json", summary)
    return summary


def view_masks(args: argparse.Namespace, paths: SAI3DPaths) -> dict[str, Any]:
    """Regenerate only the final SAI3D-to-view masks from saved 3D labels."""
    rows = _read_jsonl(_require_file(paths.frame_manifest, "SAI3D frame manifest"))
    mesh = _load_mesh(_require_file(paths.scene_mesh, "SAI3D staged mesh"))
    observations_path = _require_file(paths.observations_path, "SAI3D observations")
    point_label_path = paths.mesh_label_dir / "point_labels.npy"
    face_label_path = paths.mesh_label_dir / "face_labels.npy"
    missing = [str(path) for path in (point_label_path, face_label_path) if not path.exists()]
    if missing:
        raise SystemExit(
            "SAI3D view-mask refinement needs saved 3D labels from the segment stage. "
            "Missing:\n  "
            + "\n  ".join(missing)
            + "\nRun the SAI3D --stage segment command first, then rerun --stage view_masks."
        )

    observations = np.load(observations_path)
    if "points" in observations:
        points = observations["points"].astype(np.float32, copy=False)
    else:
        points = np.loadtxt(paths.points_path, dtype=np.float32)
    graph_point_labels = np.load(point_label_path).astype(np.int32, copy=False)
    face_labels = np.load(face_label_path).astype(np.int32, copy=False)
    if graph_point_labels.shape[0] != points.shape[0]:
        raise SystemExit(
            "Saved SAI3D point labels do not match saved graph points. "
            f"labels={graph_point_labels.shape[0]} points={points.shape[0]}. "
            "Rerun --stage segment instead."
        )

    point_source = str(args.point_source)
    for summary_name in ("segment_summary.json", "observation_summary.json"):
        summary_path = paths.summary_dir / summary_name
        if summary_path.exists():
            try:
                stored = _read_json(summary_path)
                point_source = str(stored.get("pointSource", point_source))
                break
            except json.JSONDecodeError:
                pass
    point_set = PointSet(
        points=points,
        source=point_source,
        summary={"source": point_source, "description": "Loaded from saved SAI3D observation graph."},
    )
    _obs_labels, _obs_seen, projection_cache, frame_stats = _project_vertices_to_masks(
        paths=paths,
        vertices=points,
        point_set=point_set,
        rows=rows,
        args=args,
    )
    view_summary = _write_view_masks(
        paths=paths,
        mesh=mesh,
        rows=rows,
        projection_cache=projection_cache,
        point_labels=graph_point_labels,
        face_labels=face_labels,
        args=args,
    )
    summary = {
        "stage": "view_masks",
        "timestampUtc": _now(),
        "method": "Regenerate Split&Splat-inspired SAI3D view masks from saved 3D point/face labels.",
        "frameCount": int(len(rows)),
        "pointCount": int(points.shape[0]),
        "pointSource": point_source,
        "frameProjectionStats": frame_stats,
        "viewMasks": view_summary,
        "outputs": {
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskMasksDir": str(paths.view_mask_masks_dir),
            "viewMaskPredictions": str(paths.view_mask_dir / "predictions.jsonl"),
            "proposalRelabelViewMaskDir": str(paths.view_mask_proposal_dir),
            "rejectedProposalViewMaskDir": str(paths.view_mask_reject_dir),
            "rejectedProposalViewMaskMasksDir": str(paths.view_mask_reject_masks_dir),
            "projected3dViewMaskDir": str(paths.view_mask_projected_dir),
            "pointSupportViewMaskDir": str(paths.view_mask_point_support_dir),
            "promptOverlayDir": str(paths.view_mask_prompt_overlay_dir),
            "sam2RefinedViewMaskDir": str(paths.view_mask_sam2_dir),
        },
    }
    _write_json(paths.summary_dir / "view_mask_refine_summary.json", summary)
    return summary


def _merge_stage_summary(paths: SAI3DPaths, stage_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(paths.summary_dir.glob("*_summary.json")):
        try:
            summaries.append(_read_json(path))
        except json.JSONDecodeError:
            continue
    outputs = {
        "datasetDir": str(paths.dataset_dir),
        "posedDir": str(paths.posed_dir),
        "maskDir": str(paths.mask_dir),
        "frameManifest": str(paths.frame_manifest),
        "meshLabelDir": str(paths.mesh_label_dir),
        "observations": str(paths.observations_path),
        "points": str(paths.points_path),
        "superpoints": str(paths.superpoints_path),
        "superpointPointCloud": str(paths.mesh_label_dir / "superpoints_points.ply"),
        "positiveSupportPointCloud": str(paths.mesh_label_dir / "superpoint_positive_support_points.ply"),
        "positiveSupportArrays": str(paths.mesh_label_dir / "superpoint_observation_support.npz"),
        "pointLabels": str(paths.mesh_label_dir / "point_labels.npy"),
        "labeledPoints": str(paths.mesh_label_dir / "labeled_points.ply"),
        "vertexLabels": str(paths.mesh_label_dir / "vertex_labels.npy"),
        "faceLabels": str(paths.mesh_label_dir / "face_labels.npy"),
        "labeledMesh": str(paths.mesh_label_dir / "labeled_mesh.ply"),
        "viewMaskDir": str(paths.view_mask_dir),
        "viewMaskMasksDir": str(paths.view_mask_masks_dir),
        "proposalRelabelViewMaskDir": str(paths.view_mask_proposal_dir),
        "proposalRelabelViewMaskMasksDir": str(paths.view_mask_proposal_masks_dir),
        "rejectedProposalViewMaskDir": str(paths.view_mask_reject_dir),
        "rejectedProposalViewMaskMasksDir": str(paths.view_mask_reject_masks_dir),
        "projected3dViewMaskDir": str(paths.view_mask_projected_dir),
        "projected3dViewMaskMasksDir": str(paths.view_mask_projected_masks_dir),
        "pointSupportViewMaskDir": str(paths.view_mask_point_support_dir),
        "pointSupportViewMaskMasksDir": str(paths.view_mask_point_support_masks_dir),
        "promptOverlayDir": str(paths.view_mask_prompt_overlay_dir),
        "sam2RefinedViewMaskDir": str(paths.view_mask_sam2_dir),
        "sam2RefinedViewMaskMasksDir": str(paths.view_mask_sam2_masks_dir),
        "viewMaskPredictions": str(paths.view_mask_dir / "predictions.jsonl"),
    }
    if str(args.point_source) == "mesh_vertices":
        outputs["superpointMesh"] = str(paths.mesh_label_dir / "superpoints_mesh.ply")
        outputs["positiveSupportMesh"] = str(paths.mesh_label_dir / "superpoint_positive_support_mesh.ply")
    payload = {
        "stage": "sai3d_baseline",
        "timestampUtc": _now(),
        "modelDir": str(args.model_dir.expanduser().resolve()),
        "baselineDir": str(paths.baseline_dir),
        "sceneName": paths.scene_name,
        "maskName": paths.mask_name,
        "activeStageResults": stage_results,
        "summaries": summaries,
        "outputs": outputs,
    }
    _write_json(paths.summary, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SAI3D baseline in OMeGa conventions.")
    parser.add_argument("--baseline", default="sai3d", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("prepare", "observe", "segment", "view_masks", "all"), default="prepare")
    parser.add_argument("--model-dir", type=Path, required=True, help="OMeGa model/result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Optional mesh. Defaults to healed/preclean/latest OMeGa mesh.")
    parser.add_argument("--iteration", type=int, default=-1, help="OMeGa mesh iteration to use when --mesh is omitted.")
    parser.add_argument("--baseline-name", default="sai3d")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sai3d-root", type=Path, default=None, help="Defaults to third_party/SAI3D_DT when present.")
    parser.add_argument(
        "--proposal-source-name",
        default="view_proposals_1024",
        help="Common 2D proposal source to stage into SAI3D.",
    )
    parser.add_argument(
        "--proposal-source-dir",
        type=Path,
        default=None,
        help="Explicit common 2D proposal source directory.",
    )
    parser.add_argument("--source-frame-stride", type=int, default=1, help="Use every Nth source mask frame.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all selected source frames.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--raycast-chunk-rows", type=int, default=64)
    parser.add_argument(
        "--point-source",
        choices=("mesh_vertices", "mesh_area_samples", "point_cloud"),
        default="mesh_vertices",
        help="3D points used by the SAI3D graph. The mesh is still used for staging, rendered masks, and mesh-label outputs.",
    )
    parser.add_argument(
        "--point-sample-count",
        type=int,
        default=0,
        help="Number of area-balanced surface samples when --point-source mesh_area_samples. 0 matches mesh vertex count.",
    )
    parser.add_argument("--point-sample-seed", type=int, default=71)
    parser.add_argument(
        "--point-cloud",
        type=Path,
        default=None,
        help="External PLY point cloud when --point-source point_cloud.",
    )
    parser.add_argument(
        "--point-cloud-max-points",
        type=int,
        default=0,
        help="Randomly keep at most this many external cloud points. 0 keeps all points.",
    )
    parser.add_argument("--point-cloud-sample-seed", type=int, default=71)
    parser.add_argument(
        "--point-visibility-source",
        choices=("auto", "rendered_depth", "point_zbuffer"),
        default="auto",
        help="Visibility test for projected graph points. auto uses point_zbuffer for external clouds and rendered_depth otherwise.",
    )
    parser.add_argument(
        "--point-zbuffer-depth-band",
        type=float,
        default=0.02,
        help="Meters behind the nearest projected point per pixel to keep when using point_zbuffer visibility.",
    )
    parser.add_argument("--superpoint-mode", choices=("voxel", "vertex"), default="voxel")
    parser.add_argument("--superpoint-target-count", type=int, default=8000)
    parser.add_argument("--superpoint-voxel-size", type=float, default=0.0, help="0 derives a size from --superpoint-target-count.")
    parser.add_argument(
        "--graph-max-edge-length",
        type=float,
        default=0.0,
        help="Optional maximum metric length for kNN graph edges. 0 keeps all kNN edges.",
    )
    parser.add_argument("--sai3d-view-stride", type=int, default=1, help="Use every Nth staged view in SAI3D affinity.")
    parser.add_argument("--visibility-rtol", type=float, default=0.15)
    parser.add_argument("--visibility-atol", type=float, default=0.02)
    parser.add_argument("--thres-connect", default="0.9,0.5,5", help="SAI3D progressive thresholds, or start,end,count.")
    parser.add_argument("--dis-decay", type=float, default=0.5)
    parser.add_argument("--thres-dis", type=float, default=0.15, help="Kept for upstream SAI3D argument compatibility.")
    parser.add_argument("--thres-merge", type=int, default=200)
    parser.add_argument("--max-neighbor-distance", type=int, default=2)
    parser.add_argument("--similar-metric", choices=("2-norm", "1-norm", "inf-norm", "Hellinger"), default="2-norm")
    parser.add_argument("--thres-trunc", type=float, default=0.0)
    parser.add_argument("--from-points-thres", type=float, default=0.0)
    parser.add_argument("--k-graph", type=int, default=8)
    parser.add_argument("--sai3d-workers", type=int, default=20)
    parser.add_argument("--use-torch", action="store_true")
    parser.add_argument("--view-mask-min-vertices", type=int, default=6)
    parser.add_argument("--view-mask-min-majority", type=float, default=0.55)
    parser.add_argument(
        "--view-mask-refine-mode",
        choices=("geometry", "sam2", "split_splat"),
        default="split_splat",
        help=(
            "Final per-view mask mode. geometry only projects 3D labels; sam2 prompts each visible label; "
            "split_splat is a SAI3D proposal-gated finalizer inspired by Split&Splat: "
            "assign automatic proposals with projected 3D support and use SAM2 only as fallback."
        ),
    )
    parser.add_argument("--view-mask-sam2-root", type=Path, default=None, help="Defaults to third_party/sam2.")
    parser.add_argument("--view-mask-sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--view-mask-sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--view-mask-point-radius", type=int, default=2)
    parser.add_argument("--view-mask-sam2-prompt-count", type=int, default=10)
    parser.add_argument("--view-mask-sam2-min-prompt-points", type=int, default=3)
    parser.add_argument("--view-mask-prompt-erode-px", type=int, default=5)
    parser.add_argument(
        "--view-mask-sam2-prompt-alpha",
        type=float,
        default=0.3,
        help="Split&Splat-style coreset penalty for points far from the projected segment centroid.",
    )
    parser.add_argument("--view-mask-sam2-min-geometry-area", type=int, default=96)
    parser.add_argument("--view-mask-sam2-min-iou", type=float, default=0.28)
    parser.add_argument("--view-mask-sam2-min-geometry-recall", type=float, default=0.35)
    parser.add_argument("--view-mask-sam2-max-area-ratio", type=float, default=3.0)
    parser.add_argument("--view-mask-sam2-prefer-margin", type=float, default=0.02)
    parser.add_argument("--view-mask-proposal-min-iou", type=float, default=0.10)
    parser.add_argument(
        "--view-mask-split-splat-min-support-pixels",
        type=int,
        default=16,
        help="Minimum projected 3D-support pixels inside an eroded 2D proposal before the proposal can inherit a global ID.",
    )
    parser.add_argument(
        "--view-mask-split-splat-min-majority",
        type=float,
        default=0.58,
        help="Minimum dominant global-label fraction inside an eroded proposal for Split&Splat-style proposal assignment.",
    )
    parser.add_argument(
        "--view-mask-split-splat-min-support-ratio",
        type=float,
        default=0.002,
        help="Minimum visible 3D-support fraction of an eroded proposal. Keeps tiny accidental overlaps from assigning large masks.",
    )
    parser.add_argument(
        "--view-mask-split-splat-min-point-recall",
        type=float,
        default=0.20,
        help="Accept a relabelled proposal if it covers at least this fraction of sparse projected point support.",
    )
    parser.add_argument(
        "--view-mask-split-splat-max-proposal-area-ratio",
        type=float,
        default=8.0,
        help="Reject proposal masks that are this many times larger than the projected 3D geometry support.",
    )
    parser.add_argument(
        "--view-mask-max-labels-per-view",
        type=int,
        default=0,
        help="0 refines every visible label. Positive values keep largest projected labels only for quick tests.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    mask_name = str(args.proposal_source_name)
    paths = resolve_paths(
        model_dir,
        baseline_name=args.baseline_name,
        scene_name=args.scene_name,
        output_dir=args.output_dir,
        mask_name=mask_name,
    )
    stage_results: list[dict[str, Any]] = []
    stages = ["prepare", "segment", "view_masks"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "prepare":
            stage_results.append(prepare(args, paths))
        elif stage == "observe":
            stage_results.append(observe(args, paths))
        elif stage == "segment":
            stage_results.append(segment(args, paths))
        elif stage == "view_masks":
            stage_results.append(view_masks(args, paths))
        else:  # pragma: no cover
            raise AssertionError(stage)
    summary = _merge_stage_summary(paths, stage_results, args)
    print("=== SAI3D Baseline ===")
    print(f"Stage: {args.stage}")
    print(f"Baseline dir: {paths.baseline_dir}")
    print(f"Scene: {paths.scene_name}")
    print(f"Summary: {paths.summary}")
    print(f"Mesh labels: {summary['outputs']['meshLabelDir']}")
    print(f"View masks: {summary['outputs']['viewMaskDir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
