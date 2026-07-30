"""Path discovery and small IO helpers for the interactive editor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OMEGA_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = OMEGA_ROOT.parents[1]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class EditorPaths:
    model_dir: Path
    baseline_name: str
    baseline_dir: Path
    dataset_dir: Path
    frame_manifest: Path
    point_labels: Path | None
    points_path: Path
    interactive_dir: Path
    point_clouds_dir: Path
    colmap_sparse_source: Path
    colmap_sparse_cache: Path
    colmap_sparse_summary: Path
    colmap_sparse_cleaned_cache: Path
    colmap_sparse_cleaned_ply: Path
    colmap_sparse_cleaned_summary: Path
    colmap_sparse_segmented_cache: Path
    colmap_sparse_segmented_ply: Path
    colmap_sparse_segmented_summary: Path
    colmap_track_model: Path
    colmap_pixel_transform_summary: Path
    feedforward_source: Path
    feedforward_init_mesh: Path
    feedforward_cache: Path
    feedforward_summary: Path
    feedforward_segmented_cache: Path
    feedforward_segmented_ply: Path
    feedforward_segmented_summary: Path
    omega_final_mesh_source: Path
    omega_final_points: Path
    omega_final_cache: Path
    omega_final_summary: Path
    omega_final_surface_cache: Path
    omega_final_surface_summary: Path
    omega_final_hybrid_points: Path
    omega_final_hybrid_cache: Path
    omega_final_hybrid_summary: Path
    omega_final_segmented_cache: Path
    omega_final_segmented_ply: Path
    omega_final_segmented_summary: Path
    keyframes_dir: Path
    keyframes_summary: Path
    regions_dir: Path
    regions_summary: Path
    region_maps_dir: Path
    region_overlays_dir: Path
    region_edit_log: Path
    segmentation3d_dir: Path


def slug_token(value: str) -> str:
    out: list[str] = []
    last = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last = False
        elif not last:
            out.append("_")
            last = True
    return "".join(out).strip("_") or "interactive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def resolve_paths(
    model_dir: Path,
    baseline_name: str,
    *,
    colmap_model: Path | None = None,
    colmap_point_cloud: Path | None = None,
    feedforward_point_cloud: Path | None = None,
    feedforward_init_mesh: Path | None = None,
    omega_final_mesh: Path | None = None,
) -> EditorPaths:
    model_dir = model_dir.expanduser().resolve()
    baseline_name = slug_token(baseline_name)
    baseline_dir = model_dir / "segmentation" / "baselines" / baseline_name
    if not baseline_dir.exists():
        parent = baseline_dir.parent
        available = sorted(path.name for path in parent.iterdir() if path.is_dir()) if parent.exists() else []
        raise FileNotFoundError(
            f"SAI3D baseline does not exist: {baseline_dir}. "
            f"Available baselines: {', '.join(available) if available else 'none'}"
        )

    dataset_dir = baseline_dir / "dataset"
    frame_manifest = dataset_dir / "frame_manifest.jsonl"
    point_labels_candidate = baseline_dir / "mesh_labels" / "point_labels.npy"
    point_labels = point_labels_candidate if point_labels_candidate.exists() else None
    scan_dirs = sorted((dataset_dir / "scans").glob("*"))
    points_candidates = [scan_dir / "points.pts" for scan_dir in scan_dirs]
    points_path = next((path for path in points_candidates if path.exists()), Path())

    missing = [
        ("frame manifest", frame_manifest),
        ("points.pts", points_path),
    ]
    absent = [f"{label}: {path}" for label, path in missing if not path.exists()]
    if absent:
        raise FileNotFoundError("Interactive editor is missing required SAI3D input data:\n" + "\n".join(absent))

    interactive_dir = baseline_dir / "interactive"
    point_clouds_dir = interactive_dir / "data" / "point_clouds"
    capture_root = model_dir.parents[1]
    omega_run_root = model_dir.parent
    resolved_feedforward = (
        omega_run_root / "dataset" / "vfm_sparse" / "0" / "points3D.ply"
        if feedforward_point_cloud is None
        else feedforward_point_cloud.expanduser().resolve()
    )
    resolved_init_mesh = (
        omega_run_root / "init_mesh.ply"
        if feedforward_init_mesh is None
        else feedforward_init_mesh.expanduser().resolve()
    )
    resolved_omega_final_mesh = (
        _find_omega_final_mesh(model_dir)
        if omega_final_mesh is None
        else omega_final_mesh.expanduser().resolve()
    )
    colmap_track_model = (
        _find_colmap_track_model(capture_root, model_dir)
        if colmap_model is None
        else colmap_model.expanduser().resolve()
    )
    resolved_colmap_points = (
        capture_root / "pointcloud" / "colmap_sparse" / "colmap_sparse_points.ply"
        if colmap_point_cloud is None
        else colmap_point_cloud.expanduser().resolve()
    )
    keyframes_dir = interactive_dir / "keyframes"
    regions_dir = interactive_dir / "regions"
    return EditorPaths(
        model_dir=model_dir,
        baseline_name=baseline_name,
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        frame_manifest=frame_manifest,
        point_labels=point_labels,
        points_path=points_path,
        interactive_dir=interactive_dir,
        point_clouds_dir=point_clouds_dir,
        colmap_sparse_source=resolved_colmap_points,
        colmap_sparse_cache=point_clouds_dir / "colmap_sparse.npz",
        colmap_sparse_summary=point_clouds_dir / "colmap_sparse.json",
        colmap_sparse_cleaned_cache=point_clouds_dir / "colmap_sparse_cleaned.npz",
        colmap_sparse_cleaned_ply=point_clouds_dir / "colmap_sparse_cleaned.ply",
        colmap_sparse_cleaned_summary=point_clouds_dir / "colmap_sparse_cleaned.json",
        colmap_sparse_segmented_cache=point_clouds_dir / "colmap_sparse_segmented.npz",
        colmap_sparse_segmented_ply=point_clouds_dir / "colmap_sparse_segmented.ply",
        colmap_sparse_segmented_summary=point_clouds_dir / "colmap_sparse_segmented.json",
        colmap_track_model=colmap_track_model,
        colmap_pixel_transform_summary=(
            capture_root / "metadata" / "pipeline" / "step_D01c_export_dslr_pinhole_package.json"
        ).resolve(),
        feedforward_source=resolved_feedforward.resolve(),
        feedforward_init_mesh=resolved_init_mesh.resolve(),
        feedforward_cache=point_clouds_dir / "feedforward_init.npz",
        feedforward_summary=point_clouds_dir / "feedforward_init.json",
        feedforward_segmented_cache=point_clouds_dir / "feedforward_init_segmented.npz",
        feedforward_segmented_ply=point_clouds_dir / "feedforward_init_segmented.ply",
        feedforward_segmented_summary=point_clouds_dir / "feedforward_init_segmented.json",
        omega_final_mesh_source=resolved_omega_final_mesh,
        omega_final_points=point_clouds_dir / "omega_final_vertices.ply",
        omega_final_cache=point_clouds_dir / "omega_final_vertices.npz",
        omega_final_summary=point_clouds_dir / "omega_final_vertices.json",
        omega_final_surface_cache=point_clouds_dir / "omega_final_surface_completion.npz",
        omega_final_surface_summary=point_clouds_dir / "omega_final_surface_completion.json",
        omega_final_hybrid_points=point_clouds_dir / "omega_final_clean_hybrid.ply",
        omega_final_hybrid_cache=point_clouds_dir / "omega_final_clean_hybrid.npz",
        omega_final_hybrid_summary=point_clouds_dir / "omega_final_clean_hybrid.json",
        omega_final_segmented_cache=point_clouds_dir / "omega_final_clean_hybrid_segmented.npz",
        omega_final_segmented_ply=point_clouds_dir / "omega_final_clean_hybrid_segmented.ply",
        omega_final_segmented_summary=point_clouds_dir / "omega_final_clean_hybrid_segmented.json",
        keyframes_dir=keyframes_dir,
        keyframes_summary=keyframes_dir / "keyframes.json",
        regions_dir=regions_dir,
        regions_summary=regions_dir / "regions.json",
        region_maps_dir=regions_dir / "keyframe_region_maps",
        region_overlays_dir=regions_dir / "overlays",
        region_edit_log=regions_dir / "edit_log.jsonl",
        segmentation3d_dir=interactive_dir / "3d_segmentation",
    )


def _find_colmap_track_model(capture_root: Path, model_dir: Path) -> Path:
    roots = [capture_root]
    if capture_root.name.endswith("_pinhole"):
        roots.append(capture_root.with_name(capture_root.name.removesuffix("_pinhole")))

    candidates: list[Path] = []
    for root in roots:
        sparse_root = root / "cameras" / "sparse_reconstruction"
        candidates.extend(
            [
                sparse_root / "dslr_colmap" / "sparse_bin" / "0",
                sparse_root / "dslr_colmap" / "sparse_text",
                sparse_root / "sparse_bin" / "0",
                sparse_root / "sparse_bin",
                sparse_root / "sparse_text",
            ]
        )
    candidates.append(model_dir.parent / "dataset" / "sparse" / "0")

    for candidate in candidates:
        if _is_colmap_model_dir(candidate):
            return candidate.resolve()
    return candidates[0].resolve()


def _find_omega_final_mesh(model_dir: Path) -> Path:
    exact = model_dir.parent / "model_baseline_strongest_30000" / "plys" / "mesh_29999_rank0.ply"
    candidates = [exact, model_dir / "plys" / "mesh_29999_rank0.ply"]
    candidates.extend(
        sorted(
            model_dir.parent.glob("*strongest*/plys/mesh_*_rank0.ply"),
            key=lambda path: (path.parent.parent.name, path.name),
            reverse=True,
        )
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return exact.resolve()


def _is_colmap_model_dir(path: Path) -> bool:
    has_images = (path / "images.bin").is_file() or (path / "images.txt").is_file()
    has_points = (path / "points3D.bin").is_file() or (path / "points3D.txt").is_file()
    has_cameras = (path / "cameras.bin").is_file() or (path / "cameras.txt").is_file()
    return bool(path.is_dir() and has_images and has_points and has_cameras)
