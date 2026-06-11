"""Step 3b planar surface adjustment for OMeGa remeshing.

This pass consumes Stage-2 planar/detail policy scores and moves only vertices
belonging to confident plane-like surface components. It does not simplify or
retriangulate. It fits planes from the mesh geometry itself rather than from
view-normal directions, because view normals can disagree across cameras.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from . import evidence as evidence_utils

ProgressFn = Callable[[str], None]


@dataclass
class PlaneAdjustmentConfig:
    input_mesh: Path
    scores_npz: Path
    output_mesh: Path
    policy_npz: Path | None = None
    plane_threshold: float = 0.60
    detail_max: float = 0.45
    protect_max: float = 0.65
    min_component_faces: int = 20
    vertex_plane_fraction: float = 0.55
    fit_trim_quantile: float = 0.90
    fit_iterations: int = 3
    strength: float = 1.0
    max_vertex_move_m: float = 0.25
    max_plane_rmse_m: float = 0.0
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass
class PlaneAdjustmentResult:
    output_mesh: Path
    policy_npz: Path
    summary_json: Path
    debug_meshes: dict[str, Path]
    component_count: int
    adjusted_component_count: int
    adjusted_vertex_count: int


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
            raise FileNotFoundError(f"Requested region scores npz does not exist: {path}")
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
    return evidence_utils.resolve_latest_omega_mesh(model_dir, None, -1)


def _summarize(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0, "min": 0.0, "q25": 0.0, "median": 0.0, "mean": 0.0, "q75": 0.0, "q90": 0.0, "q99": 0.0, "max": 0.0}
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
    required = ["face_plane_project_weight", "face_planar_weight", "face_detail_weight", "face_protect_weight"]
    missing = [key for key in required if key not in scores]
    if missing:
        raise KeyError(f"Region scores missing plane adjustment arrays: {missing}. Re-run Stage 2.")
    bad_shapes = [key for key in required if np.asarray(scores[key]).reshape(-1).shape[0] != face_count]
    if bad_shapes:
        raise ValueError(f"Score arrays do not match mesh face count {face_count}: {bad_shapes}")
    return scores


def _face_scalar(scores: dict[str, np.ndarray], key: str, face_count: int, default: float = 0.0) -> np.ndarray:
    if key not in scores:
        return np.full((face_count,), float(default), dtype=np.float32)
    values = np.asarray(scores[key], dtype=np.float32).reshape(-1)
    if values.shape[0] != face_count:
        raise ValueError(f"Score array {key!r} has {values.shape[0]} values, expected {face_count}.")
    return np.nan_to_num(values, nan=float(default), posinf=1.0, neginf=0.0).clip(0.0, 1.0).astype(np.float32)


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, np.maximum(double_area[:, None], 1e-12), out=np.zeros_like(raw), where=double_area[:, None] > 0.0)
    centers = np.mean(triangles, axis=1)
    return centers.astype(np.float64), normals.astype(np.float64), (0.5 * double_area).astype(np.float64)


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


def _selected_face_components(faces: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, int]:
    face_count = int(faces.shape[0])
    selected = np.asarray(selected, dtype=bool).reshape(-1)
    labels = np.full((face_count,), -1, dtype=np.int32)
    selected_ids = np.flatnonzero(selected)
    if selected_ids.size == 0:
        return labels, 0
    local_index = np.full((face_count,), -1, dtype=np.int64)
    local_index[selected_ids] = np.arange(selected_ids.size, dtype=np.int64)
    _, incident_faces = _unique_edges_with_faces(faces)
    rows: list[int] = []
    cols: list[int] = []
    for face_ids in incident_faces:
        if len(face_ids) != 2:
            continue
        a = int(face_ids[0])
        b = int(face_ids[1])
        if selected[a] and selected[b]:
            rows.extend([int(local_index[a]), int(local_index[b])])
            cols.extend([int(local_index[b]), int(local_index[a])])
    if not rows:
        labels[selected_ids] = np.arange(selected_ids.size, dtype=np.int32)
        return labels, int(selected_ids.size)
    graph = sparse.coo_matrix((np.ones((len(rows),), dtype=bool), (np.asarray(rows), np.asarray(cols))), shape=(selected_ids.size, selected_ids.size), dtype=bool)
    component_count, local_labels = connected_components(graph, directed=False, return_labels=True)
    labels[selected_ids] = local_labels.astype(np.int32, copy=False)
    return labels, int(component_count)


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


def _fit_robust_plane(points: np.ndarray, weights: np.ndarray, trim_quantile: float, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    active = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(active) < 3:
        return None
    trim_q = float(np.clip(trim_quantile, 0.50, 1.0))
    plane: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(max(int(iterations), 1)):
        plane = _fit_weighted_plane(points[active], weights[active])
        if plane is None:
            return None
        centroid, normal = plane
        residual = np.abs((points - centroid) @ normal)
        finite_active = active & np.isfinite(residual)
        if np.count_nonzero(finite_active) < 6 or trim_q >= 0.999:
            break
        cutoff = float(np.quantile(residual[finite_active], trim_q))
        next_active = finite_active & (residual <= max(cutoff, 1e-9))
        if np.count_nonzero(next_active) < 3:
            break
        active = next_active
    if plane is None:
        return None
    centroid, normal = plane
    residual = np.abs((points - centroid) @ normal)
    return centroid, normal, residual


def _component_samples(vertices: np.ndarray, faces: np.ndarray, face_ids: np.ndarray, face_weights: np.ndarray, face_areas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces[face_ids]]
    centers = np.mean(tri, axis=1)
    weights = np.maximum(face_weights[face_ids].astype(np.float64), 1e-4) * np.maximum(face_areas[face_ids], 1e-12)
    vertex_points = tri.reshape(-1, 3)
    vertex_weights = np.repeat(weights / 3.0, 3)
    center_weights = weights
    return np.concatenate([vertex_points, centers], axis=0), np.concatenate([vertex_weights, center_weights], axis=0)


def _face_to_vertex_mean(values: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((vertex_count,), dtype=np.float64)
    counts = np.zeros((vertex_count,), dtype=np.float64)
    np.add.at(out, faces.reshape(-1), np.repeat(np.asarray(values, dtype=np.float64), 3))
    np.add.at(counts, faces.reshape(-1), 1.0)
    return np.divide(out, np.maximum(counts, 1.0), out=np.zeros_like(out), where=counts > 0.0)


def _face_to_vertex_max(values: np.ndarray, faces: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((vertex_count,), dtype=np.float64)
    np.maximum.at(out, faces.reshape(-1), np.repeat(np.asarray(values, dtype=np.float64), 3))
    return out


def _vertex_to_face_mean(values: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if faces.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return np.mean(np.asarray(values, dtype=np.float64)[faces], axis=1).astype(np.float32)


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


def _component_colors(component_id: np.ndarray) -> np.ndarray:
    ids = np.maximum(np.asarray(component_id, dtype=np.int64), -1)
    x = ids.astype(np.uint32) * np.uint32(1664525) + np.uint32(1013904223)
    colors = np.stack([((x >> np.uint32(16)) & np.uint32(255)).astype(np.uint8), ((x >> np.uint32(8)) & np.uint32(255)).astype(np.uint8), (x & np.uint32(255)).astype(np.uint8)], axis=1)
    colors[ids < 0] = np.array([36, 36, 36], dtype=np.uint8)
    return colors


def _write_face_color_ply(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate([np.asarray(colors, dtype=np.uint8), np.full((faces.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def _write_debug_meshes(vertices: np.ndarray, faces: np.ndarray, arrays: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes" / "plane_adjustment"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    scalar_keys = [
        "face_plane_selected",
        "face_plane_adjusted",
        "face_plane_project_weight",
        "face_planar_weight",
        "face_detail_weight",
        "face_protect_weight",
        "face_plane_residual_before_score",
        "face_plane_residual_after_score",
        "face_plane_adjust_distance_score",
    ]
    for key in scalar_keys:
        if key not in arrays:
            continue
        path = debug_dir / f"{key}.ply"
        _write_face_color_ply(vertices, faces, _colorize_scalar(arrays[key], vmin=0.0, vmax=1.0), path)
        paths[key] = path
    if "face_plane_component_id" in arrays:
        path = debug_dir / "face_plane_component_id.ply"
        _write_face_color_ply(vertices, faces, _component_colors(arrays["face_plane_component_id"]), path)
        paths["face_plane_component_id"] = path
    return paths


def adjust_planar_surfaces(config: PlaneAdjustmentConfig, progress: ProgressFn | None = None) -> PlaneAdjustmentResult:
    progress = progress or _progress_default
    input_mesh = config.input_mesh.expanduser().resolve()
    scores_npz = config.scores_npz.expanduser().resolve()
    output_mesh = config.output_mesh.expanduser().resolve()
    policy_npz = config.policy_npz.expanduser().resolve() if config.policy_npz is not None else output_mesh.with_suffix(".plane_adjustment.npz")
    summary_json = output_mesh.with_suffix(".plane_adjustment_summary.json")
    existing = [path for path in [output_mesh, policy_npz, summary_json] if path.exists()]
    if existing and not bool(config.overwrite):
        raise FileExistsError(f"Plane adjustment outputs already exist: {existing}. Re-run with --overwrite.")

    vertices, faces = evidence_utils.load_mesh_arrays(input_mesh)
    faces_i64 = np.asarray(faces, dtype=np.int64)
    vertex_count = int(vertices.shape[0])
    face_count = int(faces_i64.shape[0])
    scores = _load_scores(scores_npz, face_count)
    face_centers, _, face_areas = _face_geometry(vertices, faces_i64)

    face_plane = _face_scalar(scores, "face_plane_project_weight", face_count)
    face_planar = _face_scalar(scores, "face_planar_weight", face_count)
    face_detail = _face_scalar(scores, "face_detail_weight", face_count)
    face_protect = _face_scalar(scores, "face_protect_weight", face_count)
    selected = (face_plane >= float(config.plane_threshold)) & (face_detail <= float(config.detail_max)) & (face_protect <= float(config.protect_max))
    component_id, component_count = _selected_face_components(faces_i64, selected)

    vertex_incident_count = np.zeros((vertex_count,), dtype=np.float64)
    vertex_selected_count = np.zeros((vertex_count,), dtype=np.float64)
    repeated = faces_i64.reshape(-1)
    np.add.at(vertex_incident_count, repeated, 1.0)
    if np.any(selected):
        np.add.at(vertex_selected_count, faces_i64[selected].reshape(-1), 1.0)
    vertex_selected_fraction = np.divide(vertex_selected_count, np.maximum(vertex_incident_count, 1.0), out=np.zeros_like(vertex_selected_count), where=vertex_incident_count > 0.0)
    vertex_plane_mean = _face_to_vertex_mean(face_plane, faces_i64, vertex_count)
    vertex_detail_max = _face_to_vertex_max(face_detail, faces_i64, vertex_count)
    vertex_protect_max = _face_to_vertex_max(face_protect, faces_i64, vertex_count)
    vertex_allowed = (
        (vertex_selected_fraction >= float(config.vertex_plane_fraction))
        & (vertex_plane_mean >= float(config.plane_threshold))
        & (vertex_detail_max <= float(config.detail_max))
        & (vertex_protect_max <= float(config.protect_max))
    )

    new_vertices = np.asarray(vertices, dtype=np.float64).copy()
    vertex_adjusted = np.zeros((vertex_count,), dtype=np.float32)
    vertex_adjust_distance = np.zeros((vertex_count,), dtype=np.float32)
    face_adjusted = np.zeros((face_count,), dtype=np.float32)
    face_residual_before = np.zeros((face_count,), dtype=np.float32)
    face_residual_after = np.zeros((face_count,), dtype=np.float32)
    component_records: list[dict[str, Any]] = []

    progress(f"[plane-adjust] Mesh: {input_mesh} ({vertex_count:,} vertices, {face_count:,} faces)")
    progress(f"[plane-adjust] Scores: {scores_npz}")
    progress(f"[plane-adjust] Selected plane-like faces: {int(np.count_nonzero(selected)):,} in {component_count:,} components")

    strength = float(np.clip(config.strength, 0.0, 1.0))
    max_move = float(config.max_vertex_move_m)
    for cid in range(component_count):
        comp_faces = np.flatnonzero(component_id == cid)
        if comp_faces.size < int(config.min_component_faces):
            continue
        sample_points, sample_weights = _component_samples(vertices, faces_i64, comp_faces, face_plane, face_areas)
        fit = _fit_robust_plane(sample_points, sample_weights, float(config.fit_trim_quantile), int(config.fit_iterations))
        if fit is None:
            continue
        centroid, normal, sample_residual = fit
        comp_center_residual = np.abs((face_centers[comp_faces] - centroid) @ normal)
        rmse = float(np.sqrt(np.mean(comp_center_residual ** 2))) if comp_center_residual.size else 0.0
        if float(config.max_plane_rmse_m) > 0.0 and rmse > float(config.max_plane_rmse_m):
            continue

        comp_vertices_all = np.unique(faces_i64[comp_faces].reshape(-1))
        comp_vertices = comp_vertices_all[vertex_allowed[comp_vertices_all]]
        if comp_vertices.size == 0:
            continue
        distances = (new_vertices[comp_vertices] - centroid) @ normal
        move = strength * distances
        if max_move > 0.0:
            move = np.clip(move, -max_move, max_move)
        new_vertices[comp_vertices] -= move[:, None] * normal[None, :]
        vertex_adjusted[comp_vertices] = 1.0
        np.maximum.at(vertex_adjust_distance, comp_vertices, np.abs(move.astype(np.float32)))

        after_residual = np.abs((np.mean(new_vertices[faces_i64[comp_faces]], axis=1) - centroid) @ normal)
        face_residual_before[comp_faces] = comp_center_residual.astype(np.float32)
        face_residual_after[comp_faces] = after_residual.astype(np.float32)
        face_adjusted[comp_faces] = float(comp_vertices.size > 0)
        component_records.append(
            {
                "id": int(cid),
                "faceCount": int(comp_faces.size),
                "allVertexCount": int(comp_vertices_all.size),
                "adjustedVertexCount": int(comp_vertices.size),
                "rmseBeforeM": rmse,
                "meanAbsResidualBeforeM": float(np.mean(comp_center_residual)) if comp_center_residual.size else 0.0,
                "meanAbsResidualAfterM": float(np.mean(after_residual)) if after_residual.size else 0.0,
                "maxAbsMoveM": float(np.max(np.abs(move))) if move.size else 0.0,
                "centroid": [float(v) for v in centroid],
                "normal": [float(v) for v in normal],
                "sampleResidualQ90M": float(np.quantile(sample_residual[np.isfinite(sample_residual)], 0.90)) if np.any(np.isfinite(sample_residual)) else 0.0,
            }
        )

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=new_vertices, faces=faces_i64, process=False).export(output_mesh)

    residual_scale = max(float(np.quantile(face_residual_before[face_residual_before > 0.0], 0.95)) if np.any(face_residual_before > 0.0) else 1e-6, 1e-6)
    move_scale = max(float(np.quantile(vertex_adjust_distance[vertex_adjust_distance > 0.0], 0.95)) if np.any(vertex_adjust_distance > 0.0) else 1e-6, 1e-6)
    arrays = {
        "face_plane_project_weight": face_plane.astype(np.float32),
        "face_planar_weight": face_planar.astype(np.float32),
        "face_detail_weight": face_detail.astype(np.float32),
        "face_protect_weight": face_protect.astype(np.float32),
        "face_plane_selected": selected.astype(np.float32),
        "face_plane_component_id": component_id.astype(np.int32),
        "face_plane_adjusted": face_adjusted.astype(np.float32),
        "face_plane_residual_before_m": face_residual_before.astype(np.float32),
        "face_plane_residual_after_m": face_residual_after.astype(np.float32),
        "face_plane_residual_before_score": np.clip(face_residual_before / residual_scale, 0.0, 1.0).astype(np.float32),
        "face_plane_residual_after_score": np.clip(face_residual_after / residual_scale, 0.0, 1.0).astype(np.float32),
        "vertex_plane_selected_fraction": vertex_selected_fraction.astype(np.float32),
        "vertex_plane_mean": vertex_plane_mean.astype(np.float32),
        "vertex_detail_max": vertex_detail_max.astype(np.float32),
        "vertex_protect_max": vertex_protect_max.astype(np.float32),
        "vertex_plane_adjusted": vertex_adjusted.astype(np.float32),
        "vertex_plane_adjust_distance_m": vertex_adjust_distance.astype(np.float32),
        "face_plane_adjust_distance_score": np.clip(_vertex_to_face_mean(vertex_adjust_distance, faces_i64) / move_scale, 0.0, 1.0).astype(np.float32),
    }
    policy_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(policy_npz, **arrays)

    debug_meshes: dict[str, Path] = {}
    if bool(config.write_debug_meshes):
        debug_meshes = _write_debug_meshes(new_vertices, faces_i64, arrays, output_mesh.parent)

    summary = {
        "stageName": "omega_plane_adjustment",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "inputMesh": str(input_mesh),
        "scoresNpz": str(scores_npz),
        "outputMesh": str(output_mesh),
        "policyNpz": str(policy_npz),
        "config": {**asdict(config), "input_mesh": str(input_mesh), "scores_npz": str(scores_npz), "output_mesh": str(output_mesh), "policy_npz": str(policy_npz)},
        "mesh": {"vertexCount": vertex_count, "faceCount": face_count},
        "selection": {"selectedFaceCount": int(np.count_nonzero(selected)), "componentCount": int(component_count), "minComponentFaces": int(config.min_component_faces)},
        "adjustment": {
            "adjustedComponentCount": int(len(component_records)),
            "adjustedVertexCount": int(np.count_nonzero(vertex_adjusted > 0.0)),
            "maxVertexMoveM": float(np.max(vertex_adjust_distance)) if vertex_adjust_distance.size else 0.0,
            "meanAdjustedVertexMoveM": float(np.mean(vertex_adjust_distance[vertex_adjusted > 0.0])) if np.any(vertex_adjusted > 0.0) else 0.0,
            "componentRecords": component_records[:500],
        },
        "arrayStats": {name: _summarize(values) for name, values in arrays.items() if np.asarray(values).dtype.kind in {"f", "i", "u", "b"}},
        "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
    }
    _write_json(summary_json, summary)
    progress(f"[plane-adjust] Adjusted components: {len(component_records):,}; vertices: {int(np.count_nonzero(vertex_adjusted > 0.0)):,}")
    progress(f"[plane-adjust] Output mesh: {output_mesh}")
    progress(f"[plane-adjust] Policy arrays: {policy_npz}")
    progress(f"[plane-adjust] Summary: {summary_json}")
    return PlaneAdjustmentResult(
        output_mesh=output_mesh,
        policy_npz=policy_npz,
        summary_json=summary_json,
        debug_meshes=debug_meshes,
        component_count=component_count,
        adjusted_component_count=len(component_records),
        adjusted_vertex_count=int(np.count_nonzero(vertex_adjusted > 0.0)),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adjust confident plane-like OMeGa mesh regions before QEM.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--input-mesh", type=Path, default=None, help="Defaults to the mesh recorded in Stage-2 summary, then latest OMeGa mesh.")
    parser.add_argument("--scores-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/evidence/mesh_region_scores.npz.")
    parser.add_argument("--out", type=Path, default=None, help="Output PLY. Defaults to <model-dir>/remesh/mesh_plane_adjusted.ply.")
    parser.add_argument("--policy-npz", type=Path, default=None, help="Output policy npz. Defaults beside --out with .plane_adjustment.npz suffix.")
    parser.add_argument("--plane-threshold", type=float, default=0.60)
    parser.add_argument("--detail-max", type=float, default=0.45)
    parser.add_argument("--protect-max", type=float, default=0.65)
    parser.add_argument("--min-component-faces", type=int, default=20)
    parser.add_argument("--vertex-plane-fraction", type=float, default=0.55)
    parser.add_argument("--fit-trim-quantile", type=float, default=0.90)
    parser.add_argument("--fit-iterations", type=int, default=3)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--max-vertex-move-m", type=float, default=0.25, help="0 disables clamping.")
    parser.add_argument("--max-plane-rmse-m", type=float, default=0.0, help="0 disables component RMSE rejection.")
    parser.add_argument("--debug-meshes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> PlaneAdjustmentConfig:
    model_dir = args.model_dir.expanduser().resolve()
    scores_npz = resolve_scores_npz(model_dir, args.scores_npz)
    input_mesh = resolve_input_mesh(model_dir, args.input_mesh, scores_npz)
    output_mesh = args.out.expanduser().resolve() if args.out is not None else model_dir / "remesh" / "mesh_plane_adjusted.ply"
    policy_npz = args.policy_npz.expanduser().resolve() if args.policy_npz is not None else None
    return PlaneAdjustmentConfig(
        input_mesh=input_mesh,
        scores_npz=scores_npz,
        output_mesh=output_mesh,
        policy_npz=policy_npz,
        plane_threshold=float(args.plane_threshold),
        detail_max=float(args.detail_max),
        protect_max=float(args.protect_max),
        min_component_faces=int(args.min_component_faces),
        vertex_plane_fraction=float(args.vertex_plane_fraction),
        fit_trim_quantile=float(args.fit_trim_quantile),
        fit_iterations=int(args.fit_iterations),
        strength=float(args.strength),
        max_vertex_move_m=float(args.max_vertex_move_m),
        max_plane_rmse_m=float(args.max_plane_rmse_m),
        write_debug_meshes=bool(args.debug_meshes),
        overwrite=bool(args.overwrite),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    adjust_planar_surfaces(config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
