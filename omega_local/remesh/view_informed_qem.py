"""View-informed QEM remeshing for OMeGa meshes.

This stage consumes the face-aligned scores produced by ``regions.py`` and
turns them into a vertex quality field for weighted QEM simplification.

The important policy is ours, even though the first backend is PyMeshLab:

- high ``face_simplify_weight`` lowers local vertex quality, so broad planar
  interiors are easier to collapse;
- high ``face_protect_weight`` raises vertex quality, so detail, seams, and
  uncertain geometry are harder to collapse;
- high ``face_plane_project_weight`` enables a fitted-plane snap before QEM;
- mesh-only open boundaries, non-manifold groups, and strong dihedral creases
  remain geometric guards.

This module is post-hoc only. It does not mutate checkpoints or OMeGa training
state. The written policy arrays are intentionally explicit so a later local
QEM implementation can reuse the same simplify/protect math without depending
on PyMeshLab's weighted simplifier.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import open3d as o3d
import pymeshlab as pml
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from . import evidence as evidence_utils
from .posthoc import MeshStats, _mesh_stats, compute_vertex_feature_quality, latest_mesh


ProgressFn = Callable[[str], None]


@dataclass
class ViewInformedQEMConfig:
    input_mesh: Path
    scores_npz: Path
    output_mesh: Path
    target_faces: int
    policy_npz: Path | None = None
    preserve_boundary: bool = True
    boundary_weight: float = 5.0
    preserve_normal: bool = True
    preserve_topology: bool = False
    planar_quadric: bool = True
    planar_weight: float = 0.0005
    quality_weight: bool = True
    quality_threshold: float = 0.3
    min_quality: float = 0.04
    quality_gamma: float = 1.25
    view_protect_weight: float = 1.0
    anti_simplify_weight: float = 0.45
    uncertainty_weight: float = 0.60
    detail_weight: float = 0.35
    geometry_weight: float = 1.0
    crease_angle_deg: float = 80.0
    boundary_quality: float = 1.0
    geometry_quality_gamma: float = 1.0
    plane_project: bool = True
    plane_project_threshold: float = 0.60
    plane_project_detail_max: float = 0.45
    plane_project_min_faces: int = 20
    plane_project_vertex_fraction: float = 0.55
    plane_project_strength: float = 1.0
    isotropic_iterations: int = 0
    isotropic_target_length_m: float = 0.0
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass
class ViewInformedQEMSummary:
    stage_name: str
    created_at_utc: str
    input_mesh: str
    scores_npz: str
    output_mesh: str
    policy_npz: str
    target_faces: int
    mode: str
    config: dict[str, Any]
    input_stats: MeshStats
    output_stats: MeshStats
    score_stats: dict[str, dict[str, float]]
    vertex_policy_stats: dict[str, dict[str, float]]
    edge_policy_stats: dict[str, dict[str, float]]
    geometry_feature_stats: dict[str, float]
    debug_meshes: dict[str, str]


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    candidate = (base / path).expanduser().resolve()
    if candidate.exists():
        return candidate
    repo_candidate = (evidence_utils._repo_root() / path).expanduser().resolve()  # noqa: SLF001
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def resolve_scores_npz(model_dir: Path, requested: Path | None = None) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if requested is not None:
        path = _resolve_path(requested, model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Requested region score npz does not exist: {path}")
        return path
    default = model_dir / "remesh" / "evidence" / "mesh_region_scores.npz"
    if default.exists():
        return default
    raise FileNotFoundError(f"Missing region scores: {default}. Run run_omega_mesh_region_scores.py first.")


def _score_summary_path(scores_npz: Path) -> Path:
    return scores_npz.with_name(f"{scores_npz.stem}_summary.json")


def resolve_input_mesh(model_dir: Path, requested: Path | None, scores_npz: Path) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if requested is not None:
        path = _resolve_path(requested, model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Requested input mesh does not exist: {path}")
        return path

    summary_path = _score_summary_path(scores_npz)
    if summary_path.exists():
        summary = _read_json(summary_path)
        mesh_path = summary.get("inputs", {}).get("meshPath")
        if mesh_path:
            path = Path(str(mesh_path)).expanduser().resolve()
            if path.exists():
                return path

    return latest_mesh(model_dir)


def _summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q99": 0.0,
            "max": 0.0,
        }
    qs = np.quantile(arr, [0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "q25": float(qs[0]),
        "median": float(qs[1]),
        "mean": float(np.mean(arr)),
        "q75": float(qs[2]),
        "q90": float(qs[3]),
        "q99": float(qs[4]),
        "max": float(np.max(arr)),
    }


def _load_scores(path: Path, face_count: int) -> dict[str, np.ndarray]:
    data = np.load(path)
    scores = {key: data[key] for key in data.files}
    required = [
        "face_simplify_weight",
        "face_protect_weight",
        "face_type_uncertain_score",
        "face_type_detail_score",
        "face_type_planar_score",
    ]
    missing = [key for key in required if key not in scores]
    if missing:
        raise KeyError(
            "Region scores are missing required view-informed QEM arrays: "
            f"{missing}. Re-run scripts/run_omega_mesh_region_scores.py."
        )
    bad_shapes = [key for key in required if scores[key].shape[0] != face_count]
    if bad_shapes:
        raise ValueError(
            "Region scores do not match the input mesh face count. "
            f"Mesh has {face_count} faces, but these arrays differ: {bad_shapes}. "
            "Re-run evidence and region scoring for this exact mesh."
        )
    return scores


def _face_scalar(scores: dict[str, np.ndarray], key: str, face_count: int, default: float = 0.0) -> np.ndarray:
    if key not in scores:
        return np.full((face_count,), float(default), dtype=np.float32)
    values = np.asarray(scores[key], dtype=np.float32).reshape(-1)
    if values.shape[0] != face_count:
        raise ValueError(f"Score array {key!r} has {values.shape[0]} values, expected {face_count}.")
    return np.nan_to_num(values, nan=float(default), posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32)


def _face_to_vertex_mean(values: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((vertex_count,), dtype=np.float64)
    counts = np.zeros((vertex_count,), dtype=np.float64)
    repeated = np.repeat(np.asarray(values, dtype=np.float64), 3)
    np.add.at(out, faces.reshape(-1), repeated)
    np.add.at(counts, faces.reshape(-1), 1.0)
    return np.divide(out, np.maximum(counts, 1.0), out=np.zeros_like(out), where=counts > 0.0)


def _face_to_vertex_max(values: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((vertex_count,), dtype=np.float64)
    repeated = np.repeat(np.asarray(values, dtype=np.float64), 3)
    np.maximum.at(out, faces.reshape(-1), repeated)
    return out


def _vertex_to_face_mean(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.mean(np.asarray(values, dtype=np.float64)[faces], axis=1).astype(np.float32)


def _unique_edges_with_faces(faces: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    if faces.size == 0:
        return np.zeros((0, 2), dtype=np.int64), []
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_faces = np.tile(np.arange(faces.shape[0], dtype=np.int64), 3)
    sorted_edges = np.sort(edges, axis=1)
    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    sorted_edges = sorted_edges[order]
    sorted_face_ids = edge_faces[order]
    changed = np.ones((len(sorted_edges),), dtype=bool)
    changed[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(changed)
    ends = np.concatenate([starts[1:], np.asarray([len(sorted_edges)], dtype=np.int64)])
    unique_edges = sorted_edges[starts].astype(np.int64, copy=False)
    incident_faces = [sorted_face_ids[start:end].astype(np.int64, copy=False) for start, end in zip(starts, ends)]
    return unique_edges, incident_faces


def _edge_reduce(values: np.ndarray, incident_faces: list[np.ndarray], mode: str) -> np.ndarray:
    out = np.zeros((len(incident_faces),), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    for idx, faces in enumerate(incident_faces):
        local = values[faces]
        if local.size == 0:
            out[idx] = 0.0
        elif mode == "min":
            out[idx] = float(np.min(local))
        elif mode == "max":
            out[idx] = float(np.max(local))
        else:
            out[idx] = float(np.mean(local))
    return out


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, np.maximum(double_area[:, None], 1e-12), out=np.zeros_like(raw, dtype=np.float64))
    centers = np.mean(triangles, axis=1)
    areas = double_area * 0.5
    return centers.astype(np.float64), normals.astype(np.float64), areas.astype(np.float64)


def _fit_weighted_plane(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    points = points[valid]
    weights = weights[valid]
    if points.shape[0] < 3:
        return None
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None
    centroid = np.sum(points * weights[:, None], axis=0) / weight_sum
    centered = points - centroid
    cov = (centered * weights[:, None]).T @ centered / weight_sum
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return centroid.astype(np.float64), (normal / norm).astype(np.float64)


def _selected_face_components(faces: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, int]:
    face_count = int(faces.shape[0])
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    labels = np.full((face_count,), -1, dtype=np.int32)
    selected_ids = np.flatnonzero(selected)
    if selected_ids.size == 0:
        return labels, 0
    if selected_ids.size == 1:
        labels[selected_ids[0]] = 0
        return labels, 1
    local_index = np.full((face_count,), -1, dtype=np.int64)
    local_index[selected_ids] = np.arange(selected_ids.size, dtype=np.int64)
    _, incident_faces = _unique_edges_with_faces(faces)
    pair_rows: list[int] = []
    pair_cols: list[int] = []
    for face_ids in incident_faces:
        if len(face_ids) != 2:
            continue
        a = int(face_ids[0])
        b = int(face_ids[1])
        if selected[a] and selected[b]:
            pair_rows.extend([int(local_index[a]), int(local_index[b])])
            pair_cols.extend([int(local_index[b]), int(local_index[a])])
    if not pair_rows:
        labels[selected_ids] = np.arange(selected_ids.size, dtype=np.int32)
        return labels, int(selected_ids.size)
    graph = sparse.coo_matrix(
        (np.ones((len(pair_rows),), dtype=bool), (np.asarray(pair_rows), np.asarray(pair_cols))),
        shape=(selected_ids.size, selected_ids.size),
        dtype=bool,
    )
    component_count, local_labels = connected_components(graph, directed=False, return_labels=True)
    labels[selected_ids] = local_labels.astype(np.int32, copy=False)
    return labels, int(component_count)


def _apply_planar_projection(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    scores: dict[str, np.ndarray],
    policy_arrays: dict[str, np.ndarray],
    config: ViewInformedQEMConfig,
    progress: ProgressFn,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    vertices = np.asarray(vertices, dtype=np.float64)
    face_count = int(faces.shape[0])
    vertex_count = int(vertices.shape[0])
    if not bool(config.plane_project):
        return vertices.copy(), {"enabled": False}, {
            "vertex_plane_projected": np.zeros((vertex_count,), dtype=np.float32),
            "vertex_plane_project_distance_m": np.zeros((vertex_count,), dtype=np.float32),
            "face_plane_projected": np.zeros((face_count,), dtype=np.float32),
        }

    face_plane = _face_scalar(scores, "face_plane_project_weight", face_count, default=0.0)
    face_detail = _face_scalar(scores, "face_type_detail_score", face_count, default=0.0)
    face_protect = _face_scalar(scores, "face_protect_weight", face_count, default=0.0)
    threshold = float(config.plane_project_threshold)
    detail_max = float(config.plane_project_detail_max)
    selected = (face_plane >= threshold) & (face_detail <= detail_max) & (face_protect <= max(detail_max, 0.65))
    labels, component_count = _selected_face_components(faces, selected)

    vertex_incident_count = np.zeros((vertex_count,), dtype=np.float64)
    vertex_selected_count = np.zeros((vertex_count,), dtype=np.float64)
    vertex_detail_max = np.zeros((vertex_count,), dtype=np.float64)
    vertex_protect_max = np.zeros((vertex_count,), dtype=np.float64)
    repeated_all = faces.reshape(-1)
    np.add.at(vertex_incident_count, repeated_all, 1.0)
    np.add.at(vertex_selected_count, faces[selected].reshape(-1), 1.0)
    np.maximum.at(vertex_detail_max, repeated_all, np.repeat(face_detail, 3))
    np.maximum.at(vertex_protect_max, repeated_all, np.repeat(face_protect, 3))
    selected_fraction = np.divide(
        vertex_selected_count,
        np.maximum(vertex_incident_count, 1.0),
        out=np.zeros_like(vertex_selected_count),
        where=vertex_incident_count > 0.0,
    )
    vertex_plane_project_mean = np.asarray(policy_arrays.get("vertex_plane_project_mean", np.zeros((vertex_count,), dtype=np.float32)), dtype=np.float64)
    global_vertex_ok = (
        (selected_fraction >= float(config.plane_project_vertex_fraction))
        & (vertex_plane_project_mean >= threshold)
        & (vertex_detail_max <= detail_max)
        & (vertex_protect_max <= max(detail_max, 0.65))
    )

    new_vertices = vertices.copy()
    vertex_projected = np.zeros((vertex_count,), dtype=np.float32)
    vertex_distance = np.zeros((vertex_count,), dtype=np.float32)
    components: list[dict[str, Any]] = []
    strength = float(np.clip(config.plane_project_strength, 0.0, 1.0))
    for component_id in range(component_count):
        comp_faces = np.flatnonzero(labels == component_id)
        if comp_faces.size < int(config.plane_project_min_faces):
            continue
        fit_vertices = np.unique(faces[comp_faces].reshape(-1))
        fit_weights = np.maximum(vertex_plane_project_mean[fit_vertices], 1e-4)
        plane = _fit_weighted_plane(vertices[fit_vertices], fit_weights)
        if plane is None:
            continue
        centroid, normal = plane
        comp_vertices = fit_vertices[global_vertex_ok[fit_vertices]]
        if comp_vertices.size == 0:
            continue
        distances = (new_vertices[comp_vertices] - centroid) @ normal
        new_vertices[comp_vertices] -= strength * distances[:, None] * normal[None, :]
        vertex_projected[comp_vertices] = 1.0
        np.maximum.at(vertex_distance, comp_vertices, np.abs((strength * distances).astype(np.float32)))
        components.append(
            {
                "id": int(component_id),
                "faceCount": int(comp_faces.size),
                "projectedVertexCount": int(comp_vertices.size),
                "meanAbsDistanceM": float(np.mean(np.abs(distances))) if distances.size else 0.0,
                "maxAbsDistanceM": float(np.max(np.abs(distances))) if distances.size else 0.0,
                "centroid": [float(v) for v in centroid],
                "normal": [float(v) for v in normal],
            }
        )
    face_plane_projected = _vertex_to_face_mean(vertex_projected, faces)
    info = {
        "enabled": True,
        "threshold": threshold,
        "detailMax": detail_max,
        "minFaces": int(config.plane_project_min_faces),
        "strength": strength,
        "selectedFaceCount": int(np.count_nonzero(selected)),
        "componentCount": int(component_count),
        "projectedComponentCount": int(len(components)),
        "projectedVertexCount": int(np.count_nonzero(vertex_projected > 0.0)),
        "meanProjectedDistanceM": float(np.mean(vertex_distance[vertex_projected > 0.0])) if np.any(vertex_projected > 0.0) else 0.0,
        "maxProjectedDistanceM": float(np.max(vertex_distance)) if vertex_distance.size else 0.0,
        "components": components[:200],
    }
    progress(
        "[view-qem] Plane projection: faces={:,} components={:,} projected_vertices={:,}".format(
            info["selectedFaceCount"], info["projectedComponentCount"], info["projectedVertexCount"]
        )
    )
    arrays = {
        "vertex_plane_projected": vertex_projected.astype(np.float32),
        "vertex_plane_project_distance_m": vertex_distance.astype(np.float32),
        "face_plane_projected": face_plane_projected.astype(np.float32),
    }
    return new_vertices, info, arrays


def _colorize_scalar(values: np.ndarray, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    normalized = np.clip((values - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    mapped = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and values.ndim == 1 and rgb.shape[1] == 1:
        rgb = rgb[:, 0, :]
    rgb[~valid] = np.array([36, 36, 36], dtype=np.uint8)
    return rgb


def _write_face_color_ply(vertices: np.ndarray, faces: np.ndarray, values: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    colors = _colorize_scalar(values)
    rgba = np.concatenate([colors, np.full((faces.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def _write_debug_meshes(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    out_dir: Path,
    face_arrays: dict[str, np.ndarray],
) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes" / "view_informed_qem"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, values in face_arrays.items():
        path = debug_dir / f"{name}.ply"
        _write_face_color_ply(vertices, faces, np.asarray(values, dtype=np.float32), path)
        paths[name] = path
    return paths


def build_view_informed_policy(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    scores: dict[str, np.ndarray],
    config: ViewInformedQEMConfig,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, float]], dict[str, dict[str, float]], dict[str, float]]:
    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])

    face_simplify = _face_scalar(scores, "face_simplify_weight", face_count)
    face_protect = _face_scalar(scores, "face_protect_weight", face_count)
    face_uncertain = _face_scalar(scores, "face_type_uncertain_score", face_count)
    face_detail = _face_scalar(scores, "face_type_detail_score", face_count)
    face_planar = _face_scalar(scores, "face_type_planar_score", face_count)
    face_plane_project = _face_scalar(scores, "face_plane_project_weight", face_count, default=0.0)

    vertex_simplify_mean = _face_to_vertex_mean(face_simplify, faces, vertex_count)
    vertex_protect_max = _face_to_vertex_max(face_protect, faces, vertex_count)
    vertex_uncertain_max = _face_to_vertex_max(face_uncertain, faces, vertex_count)
    vertex_detail_max = _face_to_vertex_max(face_detail, faces, vertex_count)
    vertex_planar_mean = _face_to_vertex_mean(face_planar, faces, vertex_count)
    vertex_plane_project_mean = _face_to_vertex_mean(face_plane_project, faces, vertex_count)

    geometry_quality, geometry_stats = compute_vertex_feature_quality(
        vertices,
        faces,
        crease_angle_deg=float(config.crease_angle_deg),
        boundary_quality=float(config.boundary_quality),
        min_quality=0.0,
        quality_gamma=float(config.geometry_quality_gamma),
    )

    view_quality_raw = np.maximum.reduce(
        [
            float(config.view_protect_weight) * vertex_protect_max,
            float(config.anti_simplify_weight) * (1.0 - vertex_simplify_mean),
            float(config.uncertainty_weight) * vertex_uncertain_max,
            float(config.detail_weight) * vertex_detail_max,
        ]
    ).clip(0.0, 1.0)
    combined_raw = np.maximum(
        view_quality_raw,
        float(config.geometry_weight) * np.asarray(geometry_quality, dtype=np.float64),
    ).clip(0.0, 1.0)

    min_quality = np.clip(float(config.min_quality), 1e-6, 1.0)
    gamma = max(float(config.quality_gamma), 1e-6)
    vertex_quality = min_quality + (1.0 - min_quality) * np.power(combined_raw, gamma)
    vertex_quality = vertex_quality.clip(min_quality, 1.0).astype(np.float64)

    unique_edges, incident_faces = _unique_edges_with_faces(faces)
    edge_simplify_min = _edge_reduce(face_simplify, incident_faces, "min")
    edge_simplify_mean = _edge_reduce(face_simplify, incident_faces, "mean")
    edge_protect_max = _edge_reduce(face_protect, incident_faces, "max")
    edge_uncertain_max = _edge_reduce(face_uncertain, incident_faces, "max")
    edge_detail_max = _edge_reduce(face_detail, incident_faces, "max")
    edge_face_count = np.asarray([len(face_ids) for face_ids in incident_faces], dtype=np.int32)

    arrays = {
        "face_simplify_weight": face_simplify.astype(np.float32),
        "face_protect_weight": face_protect.astype(np.float32),
        "face_type_uncertain_score": face_uncertain.astype(np.float32),
        "face_type_detail_score": face_detail.astype(np.float32),
        "face_type_planar_score": face_planar.astype(np.float32),
        "face_plane_project_weight": face_plane_project.astype(np.float32),
        "vertex_simplify_mean": vertex_simplify_mean.astype(np.float32),
        "vertex_protect_max": vertex_protect_max.astype(np.float32),
        "vertex_uncertain_max": vertex_uncertain_max.astype(np.float32),
        "vertex_detail_max": vertex_detail_max.astype(np.float32),
        "vertex_planar_mean": vertex_planar_mean.astype(np.float32),
        "vertex_plane_project_mean": vertex_plane_project_mean.astype(np.float32),
        "vertex_view_quality_raw": view_quality_raw.astype(np.float32),
        "vertex_geometry_quality": np.asarray(geometry_quality, dtype=np.float32),
        "vertex_combined_quality_raw": combined_raw.astype(np.float32),
        "vertex_quality": vertex_quality.astype(np.float32),
        "face_view_quality_raw": _vertex_to_face_mean(view_quality_raw, faces),
        "face_geometry_quality": _vertex_to_face_mean(geometry_quality, faces),
        "face_combined_quality_raw": _vertex_to_face_mean(combined_raw, faces),
        "face_vertex_quality": _vertex_to_face_mean(vertex_quality, faces),
        "edge_vertices": unique_edges.astype(np.int64),
        "edge_face_count": edge_face_count,
        "edge_simplify_min": edge_simplify_min.astype(np.float32),
        "edge_simplify_mean": edge_simplify_mean.astype(np.float32),
        "edge_protect_max": edge_protect_max.astype(np.float32),
        "edge_uncertain_max": edge_uncertain_max.astype(np.float32),
        "edge_detail_max": edge_detail_max.astype(np.float32),
    }

    score_stats = {
        "face_simplify_weight": _summarize(face_simplify),
        "face_protect_weight": _summarize(face_protect),
        "face_type_uncertain_score": _summarize(face_uncertain),
        "face_type_detail_score": _summarize(face_detail),
        "face_type_planar_score": _summarize(face_planar),
        "face_plane_project_weight": _summarize(face_plane_project),
    }
    vertex_policy_stats = {
        "vertex_simplify_mean": _summarize(vertex_simplify_mean),
        "vertex_protect_max": _summarize(vertex_protect_max),
        "vertex_uncertain_max": _summarize(vertex_uncertain_max),
        "vertex_detail_max": _summarize(vertex_detail_max),
        "vertex_plane_project_mean": _summarize(vertex_plane_project_mean),
        "vertex_view_quality_raw": _summarize(view_quality_raw),
        "vertex_geometry_quality": _summarize(geometry_quality),
        "vertex_combined_quality_raw": _summarize(combined_raw),
        "vertex_quality": _summarize(vertex_quality),
    }
    edge_policy_stats = {
        "edge_simplify_min": _summarize(edge_simplify_min),
        "edge_simplify_mean": _summarize(edge_simplify_mean),
        "edge_protect_max": _summarize(edge_protect_max),
        "edge_uncertain_max": _summarize(edge_uncertain_max),
        "edge_detail_max": _summarize(edge_detail_max),
        "edge_face_count": _summarize(edge_face_count),
    }
    return arrays, score_stats, vertex_policy_stats, edge_policy_stats, geometry_stats


def remesh_with_view_informed_qem(
    config: ViewInformedQEMConfig,
    progress: ProgressFn | None = None,
) -> ViewInformedQEMSummary:
    progress = progress or _progress_default
    input_mesh = config.input_mesh.expanduser().resolve()
    scores_npz = config.scores_npz.expanduser().resolve()
    output_mesh = config.output_mesh.expanduser().resolve()
    policy_npz = (
        config.policy_npz.expanduser().resolve()
        if config.policy_npz is not None
        else output_mesh.with_suffix(".policy.npz")
    )
    summary_path = output_mesh.with_suffix(".summary.json")
    if not input_mesh.exists():
        raise FileNotFoundError(f"Input mesh does not exist: {input_mesh}")
    if not scores_npz.exists():
        raise FileNotFoundError(f"Region scores do not exist: {scores_npz}")
    existing = [path for path in [output_mesh, policy_npz, summary_path] if path.exists()]
    if existing and not bool(config.overwrite):
        raise FileExistsError(f"View-informed QEM outputs already exist: {existing}. Re-run with --overwrite.")
    if int(config.target_faces) <= 0:
        raise ValueError("--target-faces must be positive.")

    vertices, faces = evidence_utils.load_mesh_arrays(input_mesh)
    faces_i32 = np.asarray(faces, dtype=np.int32)
    face_count = int(faces_i32.shape[0])
    scores = _load_scores(scores_npz, face_count)

    progress(f"[view-qem] Mesh: {input_mesh} ({vertices.shape[0]:,} vertices, {face_count:,} faces)")
    progress(f"[view-qem] Scores: {scores_npz}")
    progress(
        "[view-qem] Policy: simplify/protect face scores -> vertex quality; "
        "mesh creases and open boundaries remain geometric guards"
    )

    policy_arrays, score_stats, vertex_policy_stats, edge_policy_stats, geometry_stats = build_view_informed_policy(
        vertices=vertices,
        faces=faces_i32,
        scores=scores,
        config=config,
    )
    qem_vertices, plane_projection_info, plane_projection_arrays = _apply_planar_projection(
        vertices=vertices,
        faces=faces_i32,
        scores=scores,
        policy_arrays=policy_arrays,
        config=config,
        progress=progress,
    )
    policy_arrays.update(plane_projection_arrays)
    vertex_policy_stats.update({name: _summarize(values) for name, values in plane_projection_arrays.items() if name.startswith("vertex_")})
    score_stats.update({name: _summarize(values) for name, values in plane_projection_arrays.items() if name.startswith("face_")})

    policy_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(policy_npz, **policy_arrays)
    progress(f"[view-qem] Wrote policy arrays: {policy_npz}")

    debug_meshes: dict[str, Path] = {}
    if bool(config.write_debug_meshes):
        debug_face_arrays = {
            "face_simplify_weight": policy_arrays["face_simplify_weight"],
            "face_protect_weight": policy_arrays["face_protect_weight"],
            "face_plane_project_weight": policy_arrays["face_plane_project_weight"],
            "face_plane_projected": policy_arrays["face_plane_projected"],
            "face_type_planar_score": policy_arrays["face_type_planar_score"],
            "face_type_detail_score": policy_arrays["face_type_detail_score"],
            "face_type_uncertain_score": policy_arrays["face_type_uncertain_score"],
            "face_view_quality_raw": policy_arrays["face_view_quality_raw"],
            "face_geometry_quality": policy_arrays["face_geometry_quality"],
            "face_combined_quality_raw": policy_arrays["face_combined_quality_raw"],
            "face_vertex_quality": policy_arrays["face_vertex_quality"],
        }
        debug_meshes = _write_debug_meshes(
            vertices=vertices,
            faces=faces_i32,
            out_dir=output_mesh.parent,
            face_arrays=debug_face_arrays,
        )
        if bool(config.plane_project):
            projected_path = output_mesh.parent / f"{output_mesh.stem}_plane_projected_input.ply"
            trimesh.Trimesh(vertices=qem_vertices, faces=faces_i32, process=False).export(projected_path)
            debug_meshes["plane_projected_input"] = projected_path
        progress(f"[view-qem] Wrote {len(debug_meshes)} debug mesh colors under {output_mesh.parent / 'debug_meshes'}")

    input_stats = _mesh_stats(o3d.io.read_triangle_mesh(str(input_mesh)))
    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(
        pml.Mesh(
            vertex_matrix=np.asarray(qem_vertices, dtype=np.float64),
            face_matrix=faces_i32,
            v_scalar_array=np.asarray(policy_arrays["vertex_quality"], dtype=np.float64),
        ),
        mesh_name="omega_view_informed_input",
    )
    progress(f"[view-qem] Running weighted QEM target={int(config.target_faces):,}")
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(config.target_faces),
        qualitythr=float(config.quality_threshold),
        preserveboundary=bool(config.preserve_boundary),
        boundaryweight=float(config.boundary_weight),
        preservenormal=bool(config.preserve_normal),
        preservetopology=bool(config.preserve_topology),
        optimalplacement=True,
        planarquadric=bool(config.planar_quadric),
        planarweight=float(config.planar_weight),
        qualityweight=bool(config.quality_weight),
        autoclean=True,
        selected=False,
    )
    if int(config.isotropic_iterations) > 0:
        if float(config.isotropic_target_length_m) <= 0.0:
            raise ValueError("--isotropic-target-length-m must be positive when isotropic remeshing is enabled.")
        progress(
            "[view-qem] Running optional isotropic pass "
            f"iterations={int(config.isotropic_iterations)} target_length={float(config.isotropic_target_length_m):.6f}m"
        )
        mesh_set.meshing_isotropic_explicit_remeshing(
            iterations=int(config.isotropic_iterations),
            adaptive=True,
            selectedonly=False,
            targetlen=pml.PureValue(float(config.isotropic_target_length_m)),
            featuredeg=float(config.crease_angle_deg),
            checksurfdist=True,
            maxsurfdist=pml.PureValue(float(config.isotropic_target_length_m) * 0.75),
            splitflag=True,
            collapseflag=True,
            swapflag=True,
            smoothflag=True,
            reprojectflag=True,
        )

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh_set.save_current_mesh(str(output_mesh), save_vertex_color=False, save_vertex_quality=True)
    output_stats = _mesh_stats(o3d.io.read_triangle_mesh(str(output_mesh)))
    progress(f"[view-qem] Faces: {input_stats.faces:,} -> {output_stats.faces:,}")
    progress(f"[view-qem] Output: {output_mesh}")

    summary = ViewInformedQEMSummary(
        stage_name="omega_view_informed_qem",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_mesh=str(input_mesh),
        scores_npz=str(scores_npz),
        output_mesh=str(output_mesh),
        policy_npz=str(policy_npz),
        target_faces=int(config.target_faces),
        mode="minimal_planar_detail_qem_with_optional_plane_projection",
        config={
            **asdict(config),
            "input_mesh": str(input_mesh),
            "scores_npz": str(scores_npz),
            "output_mesh": str(output_mesh),
            "policy_npz": str(policy_npz),
        },
        input_stats=input_stats,
        output_stats=output_stats,
        score_stats=score_stats,
        vertex_policy_stats=vertex_policy_stats,
        edge_policy_stats=edge_policy_stats,
        geometry_feature_stats={**geometry_stats, "planeProjection": plane_projection_info},
        debug_meshes={name: str(path) for name, path in debug_meshes.items()},
    )
    _write_json(summary_path, summary_to_json(summary))
    progress(f"[view-qem] Summary: {summary_path}")
    return summary


def summary_to_json(summary: ViewInformedQEMSummary) -> dict[str, Any]:
    return asdict(summary)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run view-informed weighted QEM remeshing from OMeGa Stage-2 region scores."
    )
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--input-mesh", type=Path, default=None, help="Defaults to the mesh recorded in the scores summary, then latest OMeGa mesh.")
    parser.add_argument("--scores-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/evidence/mesh_region_scores.npz.")
    parser.add_argument("--out", type=Path, default=None, help="Output PLY. Defaults to <model-dir>/remesh/mesh_view_informed_qem_<target>.ply.")
    parser.add_argument("--policy-npz", type=Path, default=None, help="Policy arrays output. Defaults beside --out with .policy.npz suffix.")
    parser.add_argument("--target-faces", type=int, default=80_000)
    parser.add_argument("--preserve-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--boundary-weight", type=float, default=5.0)
    parser.add_argument("--preserve-normal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-topology", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--planar-quadric", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--planar-weight", type=float, default=0.0005)
    parser.add_argument("--quality-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-threshold", type=float, default=0.3)
    parser.add_argument("--min-quality", type=float, default=0.04)
    parser.add_argument("--quality-gamma", type=float, default=1.25)
    parser.add_argument("--view-protect-weight", type=float, default=1.0)
    parser.add_argument("--anti-simplify-weight", type=float, default=0.45)
    parser.add_argument("--uncertainty-weight", type=float, default=0.60)
    parser.add_argument("--detail-weight", type=float, default=0.35)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--crease-angle-deg", type=float, default=80.0)
    parser.add_argument("--boundary-quality", type=float, default=1.0)
    parser.add_argument("--geometry-quality-gamma", type=float, default=1.0)
    parser.add_argument("--plane-project", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plane-project-threshold", type=float, default=0.60)
    parser.add_argument("--plane-project-detail-max", type=float, default=0.45)
    parser.add_argument("--plane-project-min-faces", type=int, default=20)
    parser.add_argument("--plane-project-vertex-fraction", type=float, default=0.55)
    parser.add_argument("--plane-project-strength", type=float, default=1.0)
    parser.add_argument("--isotropic-iterations", type=int, default=0)
    parser.add_argument("--isotropic-target-length-m", type=float, default=0.0)
    parser.add_argument("--debug-meshes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> ViewInformedQEMConfig:
    model_dir = args.model_dir.expanduser().resolve()
    scores_npz = resolve_scores_npz(model_dir, args.scores_npz)
    input_mesh = resolve_input_mesh(model_dir, args.input_mesh, scores_npz)
    output_mesh = (
        args.out.expanduser().resolve()
        if args.out is not None
        else model_dir / "remesh" / f"mesh_view_informed_qem_{int(args.target_faces):06d}.ply"
    )
    policy_npz = args.policy_npz.expanduser().resolve() if args.policy_npz is not None else None
    return ViewInformedQEMConfig(
        input_mesh=input_mesh,
        scores_npz=scores_npz,
        output_mesh=output_mesh,
        target_faces=int(args.target_faces),
        policy_npz=policy_npz,
        preserve_boundary=bool(args.preserve_boundary),
        boundary_weight=float(args.boundary_weight),
        preserve_normal=bool(args.preserve_normal),
        preserve_topology=bool(args.preserve_topology),
        planar_quadric=bool(args.planar_quadric),
        planar_weight=float(args.planar_weight),
        quality_weight=bool(args.quality_weight),
        quality_threshold=float(args.quality_threshold),
        min_quality=float(args.min_quality),
        quality_gamma=float(args.quality_gamma),
        view_protect_weight=float(args.view_protect_weight),
        anti_simplify_weight=float(args.anti_simplify_weight),
        uncertainty_weight=float(args.uncertainty_weight),
        detail_weight=float(args.detail_weight),
        geometry_weight=float(args.geometry_weight),
        crease_angle_deg=float(args.crease_angle_deg),
        boundary_quality=float(args.boundary_quality),
        geometry_quality_gamma=float(args.geometry_quality_gamma),
        plane_project=bool(args.plane_project),
        plane_project_threshold=float(args.plane_project_threshold),
        plane_project_detail_max=float(args.plane_project_detail_max),
        plane_project_min_faces=int(args.plane_project_min_faces),
        plane_project_vertex_fraction=float(args.plane_project_vertex_fraction),
        plane_project_strength=float(args.plane_project_strength),
        isotropic_iterations=int(args.isotropic_iterations),
        isotropic_target_length_m=float(args.isotropic_target_length_m),
        write_debug_meshes=bool(args.debug_meshes),
        overwrite=bool(args.overwrite),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = remesh_with_view_informed_qem(config_from_args(args))
    print("=== OMeGa View-Informed QEM Remesh ===", flush=True)
    print(f"Input: {summary.input_mesh}", flush=True)
    print(f"Scores: {summary.scores_npz}", flush=True)
    print(f"Output: {summary.output_mesh}", flush=True)
    print(f"Policy: {summary.policy_npz}", flush=True)
    print(f"Faces: {summary.input_stats.faces:,} -> {summary.output_stats.faces:,}", flush=True)
    print(f"Vertex quality mean: {summary.vertex_policy_stats['vertex_quality']['mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
