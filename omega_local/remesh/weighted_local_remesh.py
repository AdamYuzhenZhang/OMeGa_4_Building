"""Phase 4B weighted local remeshing.

This stage consumes the Phase 3 continuous weights and performs conservative
post-hoc topology simplification.  The first mutating operator is a batched
edge-collapse pass:

- planar weight increases collapse pressure and proxy-plane projection;
- detail weight suppresses collapse;
- boundary weight and mesh boundaries are hard no-cross constraints;
- uncertainty weight makes all collapse decisions conservative.

The pass is global in coverage, but each accepted edit is a local edge
contraction with quality, normal, plane, and displacement gates.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

from omega_local.remesh.local_ops import (
    build_vertex_faces,
    edge_face_max,
    edge_face_mean,
    edge_face_min,
    face_geometry,
    face_max_from_edges,
    load_mesh_arrays,
    simulate_collapse_gates,
    simulate_flip_quality,
    triangle_quality_from_vertices,
    write_face_scalar_mesh,
)
from omega_local.remesh.local_qem import (
    _compute_final_target_lengths,
    _load_proxy_entries,
    _proxy_plane_arrays,
    _safe_positive,
    _summarize,
)
from omega_local.remesh.policy import build_mesh_topology


ProgressFn = Callable[[str], None]


REJECT_CODE_TO_NAME = {
    0: "not_evaluated",
    1: "accepted",
    2: "reject_boundary",
    3: "reject_density",
    4: "reject_topology",
    5: "reject_support",
    6: "reject_plane",
    7: "reject_normal",
    8: "reject_uncertainty",
    9: "reject_quality",
    10: "reject_batch_conflict",
    11: "reject_projection",
}
REJECT_NAME_TO_CODE = {name: code for code, name in REJECT_CODE_TO_NAME.items()}


@dataclass(frozen=True)
class WeightedRemeshConfig:
    mesh_path: Path
    policy_npz: Path
    proxies_npz: Path
    output_dir: Path
    output_name: str = "mesh_weighted_remesh.ply"
    diagnostics_name: str = "weighted_remesh_diagnostics"
    operations_name: str = "weighted_remesh_operations"
    proxies_json: Path | None = None
    max_passes: int = 8
    max_collapses_per_pass: int = 100000
    max_flips_per_pass: int = 50000
    max_jsonl_rows: int = 50000
    collapse_threshold: float = 0.01
    support_threshold: float = 0.08
    boundary_threshold: float = 0.50
    uncertainty_threshold: float = 1.10
    alpha_collapse: float = 0.75
    alpha_planar_collapse: float = 5.00
    alpha_planar_transition_collapse: float = 3.00
    alpha_feature_edge_collapse: float = 3.00
    alpha_detail_collapse: float = 1.75
    detail_mesh_target_factor: float = 2.25
    detail_simplify_min_weight: float = 0.25
    q_min_hard: float = 0.03
    q_min_soft: float = 0.12
    max_normal_error: float = 2.0
    max_quality_error: float = 1.5
    max_projection_distance_factor: float = 8.0
    plane_residual_factor: float = 2.0
    pre_project_before_collapse: bool = True
    feature_edge_collapse: bool = True
    feature_edge_min_planar: float = 0.35
    feature_edge_max_uncertain: float = 1.10
    planar_boundary_softening: float = 0.85
    planar_transition_min_weight: float = 0.35
    planar_projection_strength: float = 1.50
    planar_project_iterations: int = 3
    planar_project_min_weight: float = 0.25
    planar_project_max_distance_factor: float = 8.0
    detail_relax_iterations: int = 4
    detail_relax_strength: float = 0.45
    detail_relax_min_weight: float = 0.18
    detail_relax_max_distance_factor: float = 1.25
    flip_min_quality_improvement: float = 0.002
    planar_area_length_factor: float = 0.20
    planar_boundary_length_factor: float = 6.0
    scene_max_length_factor: float = 12.0
    proxy_min_length_factor: float = 1.5
    proxy_confidence_floor: float = 0.05
    lambda_proxy: float = 1.0
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class WeightedRemeshResult:
    output_mesh: Path
    diagnostics_npz: Path
    operations_jsonl: Path
    summary_json: Path
    report_md: Path
    debug_meshes: dict[str, Path]
    input_face_count: int
    output_face_count: int
    accepted_collapses: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _format_number(value: Any) -> str:
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6g}"
        return str(value)
    return str(value)


def _format_summary(summary: dict[str, Any]) -> str:
    if not summary:
        return "n/a"
    keys = ["count", "mean", "median", "q90", "max"]
    parts = []
    for key in keys:
        if key in summary:
            parts.append(f"{key}={_format_number(summary[key])}")
    return ", ".join(parts) if parts else "n/a"


def _reject_counts(reject_code: np.ndarray, candidate_ids: np.ndarray) -> dict[str, int]:
    if candidate_ids.size == 0:
        return {}
    codes = np.asarray(reject_code, dtype=np.uint8)[np.asarray(candidate_ids, dtype=np.int64)]
    counts: dict[str, int] = {}
    for code in np.unique(codes):
        name = REJECT_CODE_TO_NAME.get(int(code), f"unknown_{int(code)}")
        counts[name] = int(np.count_nonzero(codes == code))
    return counts


def _required(data: dict[str, np.ndarray], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} is missing required arrays: {missing}")


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _candidate_ids(pressure: np.ndarray, limit: int) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.float32)
    ids = np.flatnonzero(np.isfinite(pressure) & (pressure > 0.0))
    if ids.size == 0:
        return ids.astype(np.int64)
    limit = int(limit)
    if limit > 0 and ids.size > limit:
        local = np.argpartition(pressure[ids], -limit)[-limit:]
        ids = ids[local]
    order = np.argsort(-pressure[ids], kind="mergesort")
    return ids[order].astype(np.int64)


def _face_values(
    data: dict[str, np.ndarray],
    key: str,
    face_source: np.ndarray,
    *,
    fallback: np.ndarray | float,
    dtype: Any = np.float32,
) -> np.ndarray:
    fallback_arr = np.asarray(fallback, dtype=dtype)
    out = np.full(face_source.shape, fallback_arr.item() if fallback_arr.ndim == 0 else 0, dtype=dtype)
    values = data.get(key)
    if values is None:
        if fallback_arr.ndim == 0:
            return out
        return np.asarray(fallback_arr, dtype=dtype).copy()
    values = np.asarray(values, dtype=dtype)
    valid = (face_source >= 0) & (face_source < values.shape[0])
    if values.ndim == 1:
        out[valid] = values[face_source[valid]]
        return out
    return values[face_source]


def _edge_proxy_boundaries(edge_faces: np.ndarray, face_proxy_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_faces = np.asarray(edge_faces, dtype=np.int64)
    face_proxy_id = np.asarray(face_proxy_id, dtype=np.int32)
    edge_count = int(edge_faces.shape[0])
    proxy_id = np.full((edge_count,), -1, dtype=np.int32)
    boundary = np.zeros((edge_count,), dtype=np.float32)
    face_a = edge_faces[:, 0]
    face_b = edge_faces[:, 1]
    has_two = face_b >= 0
    if np.any(has_two):
        pa = face_proxy_id[face_a[has_two]]
        pb = face_proxy_id[face_b[has_two]]
        same = (pa >= 0) & (pa == pb)
        diff = (pa >= 0) & (pb >= 0) & (pa != pb)
        ids = np.flatnonzero(has_two)
        proxy_id[ids[same]] = pa[same]
        boundary[ids[diff]] = 1.0
    return proxy_id, boundary.astype(np.float32)


def _edge_plane_project(
    *,
    midpoint: np.ndarray,
    proxy_id: int,
    proxy_normals: np.ndarray,
    proxy_offsets: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, float]:
    if proxy_id < 0 or proxy_id >= proxy_normals.shape[0]:
        return midpoint, 0.0
    normal = proxy_normals[int(proxy_id)]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return midpoint, 0.0
    signed = float(np.dot(normal, midpoint) + float(proxy_offsets[int(proxy_id)]))
    projected = midpoint - signed * normal
    strength = float(np.clip(strength, 0.0, 1.0))
    candidate = midpoint + strength * (projected - midpoint)
    return candidate, float(np.linalg.norm(candidate - midpoint))


def _edge_feature_project(
    *,
    midpoint: np.ndarray,
    proxy_a: int,
    proxy_b: int,
    proxy_normals: np.ndarray,
    proxy_offsets: np.ndarray,
    strength: float,
) -> tuple[np.ndarray, float, bool]:
    if proxy_a < 0 or proxy_b < 0 or proxy_a == proxy_b:
        return midpoint, 0.0, False
    if proxy_a >= proxy_normals.shape[0] or proxy_b >= proxy_normals.shape[0]:
        return midpoint, 0.0, False
    n1 = np.asarray(proxy_normals[int(proxy_a)], dtype=np.float64)
    n2 = np.asarray(proxy_normals[int(proxy_b)], dtype=np.float64)
    norm1 = float(np.linalg.norm(n1))
    norm2 = float(np.linalg.norm(n2))
    if norm1 <= 1e-12 or norm2 <= 1e-12:
        return midpoint, 0.0, False
    n1 = n1 / norm1
    n2 = n2 / norm2
    gram = np.asarray([[np.dot(n1, n1), np.dot(n1, n2)], [np.dot(n2, n1), np.dot(n2, n2)]], dtype=np.float64)
    det = float(np.linalg.det(gram))
    if abs(det) <= 1e-8:
        return midpoint, 0.0, False
    rhs = np.asarray(
        [
            float(np.dot(n1, midpoint) + float(proxy_offsets[int(proxy_a)])),
            float(np.dot(n2, midpoint) + float(proxy_offsets[int(proxy_b)])),
        ],
        dtype=np.float64,
    )
    lambdas = np.linalg.solve(gram, rhs)
    projected = midpoint - lambdas[0] * n1 - lambdas[1] * n2
    strength = float(np.clip(strength, 0.0, 1.0))
    candidate = midpoint + strength * (projected - midpoint)
    return candidate, float(np.linalg.norm(candidate - midpoint)), True


def _mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    if vertices.size == 0 or faces.size == 0:
        return {
            "vertexCount": int(vertices.shape[0]),
            "faceCount": int(faces.shape[0]),
            "edgeCount": 0,
            "boundaryEdgeCount": 0,
            "nonmanifoldEdgeCount": 0,
            "surfaceArea": 0.0,
            "edgeLength": _summarize(np.zeros((0,), dtype=np.float32)),
        }
    topology = build_mesh_topology(faces)
    edge_length = np.linalg.norm(vertices[topology.edge_vertices[:, 1]] - vertices[topology.edge_vertices[:, 0]], axis=1)
    _, _, face_area, _ = face_geometry(vertices, faces)
    return {
        "vertexCount": int(vertices.shape[0]),
        "faceCount": int(faces.shape[0]),
        "edgeCount": int(topology.edge_vertices.shape[0]),
        "boundaryEdgeCount": int(np.count_nonzero(topology.boundary_edges)),
        "nonmanifoldEdgeCount": int(topology.nonmanifold_edge_count),
        "surfaceArea": float(np.sum(face_area)),
        "edgeLength": _summarize(edge_length.astype(np.float32)),
    }


def _unique_faces(faces: np.ndarray, face_source: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.size == 0:
        return faces, face_source, np.zeros((0,), dtype=bool)
    keys = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    _, unique_indices = np.unique(keys, axis=0, return_index=True)
    keep = np.zeros((faces.shape[0],), dtype=bool)
    keep[np.sort(unique_indices)] = True
    return faces[keep], face_source[keep], keep


def _remove_unreferenced(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if faces.size == 0:
        return vertices[:0].copy(), faces.copy(), np.zeros((vertices.shape[0],), dtype=np.int64) - 1
    used = np.unique(faces.reshape(-1))
    remap = np.full((vertices.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return vertices[used].copy(), remap[faces], remap


def _build_vertex_neighbors(edge_vertices: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(int(vertex_count))]
    for a_raw, b_raw in np.asarray(edge_vertices, dtype=np.int64):
        a = int(a_raw)
        b = int(b_raw)
        buckets[a].append(b)
        buckets[b].append(a)
    return [np.asarray(sorted(set(values)), dtype=np.int64) for values in buckets]


def _vertex_face_mean(faces: np.ndarray, values: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((int(vertex_count),), dtype=np.float64)
    count = np.zeros((int(vertex_count),), dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    for local_id in range(3):
        np.add.at(out, faces[:, local_id], values)
        np.add.at(count, faces[:, local_id], 1.0)
    return np.divide(out, np.maximum(count, 1.0), out=np.zeros_like(out)).astype(np.float32)


def _vertex_face_max(faces: np.ndarray, values: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((int(vertex_count),), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    for local_id in range(3):
        np.maximum.at(out, np.asarray(faces, dtype=np.int64)[:, local_id], values)
    return out.astype(np.float32)


def _vertex_edge_max(edge_vertices: np.ndarray, values: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((int(vertex_count),), dtype=np.float32)
    edge_vertices = np.asarray(edge_vertices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    if edge_vertices.size == 0:
        return out
    np.maximum.at(out, edge_vertices[:, 0], values)
    np.maximum.at(out, edge_vertices[:, 1], values)
    return out.astype(np.float32)


def _face_mean_from_vertices(faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.mean(values[np.asarray(faces, dtype=np.int64)], axis=1).astype(np.float32)


def _face_max_from_vertices(faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.max(values[np.asarray(faces, dtype=np.int64)], axis=1).astype(np.float32)


def _vertex_normals(faces: np.ndarray, face_normals: np.ndarray, face_area: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((int(vertex_count), 3), dtype=np.float64)
    weights = np.maximum(np.asarray(face_area, dtype=np.float64), 1e-12)
    weighted = np.asarray(face_normals, dtype=np.float64) * weights[:, None]
    for local_id in range(3):
        np.add.at(out, np.asarray(faces, dtype=np.int64)[:, local_id], weighted)
    norms = np.linalg.norm(out, axis=1)
    valid = norms > 1e-12
    out[valid] /= norms[valid, None]
    return out


def _dominant_vertex_proxy(
    faces: np.ndarray,
    face_proxy_id: np.ndarray,
    face_weight: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    proxy_ids = np.asarray(face_proxy_id, dtype=np.int32)
    out_proxy = np.full((int(vertex_count),), -1, dtype=np.int32)
    out_fraction = np.zeros((int(vertex_count),), dtype=np.float32)
    buckets: list[dict[int, float]] = [dict() for _ in range(int(vertex_count))]
    for face_id, tri in enumerate(np.asarray(faces, dtype=np.int64)):
        proxy_id = int(proxy_ids[int(face_id)])
        if proxy_id < 0:
            continue
        weight = float(max(np.asarray(face_weight, dtype=np.float32)[int(face_id)], 1e-4))
        for vertex_id in tri:
            local = buckets[int(vertex_id)]
            local[proxy_id] = local.get(proxy_id, 0.0) + weight
    for vertex_id, values in enumerate(buckets):
        if not values:
            continue
        total = max(sum(values.values()), 1e-12)
        proxy_id, weight = max(values.items(), key=lambda item: item[1])
        out_proxy[vertex_id] = int(proxy_id)
        out_fraction[vertex_id] = np.float32(weight / total)
    return out_proxy, out_fraction


def _vertex_quality_ok(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_faces: list[np.ndarray],
    face_normals: np.ndarray,
    vertex_id: int,
    candidate: np.ndarray,
    q_min_hard: float,
) -> bool:
    incident = vertex_faces[int(vertex_id)]
    if incident.size == 0:
        return False
    triangles = vertices[faces[incident]].copy()
    local_faces = faces[incident]
    mask = local_faces == int(vertex_id)
    triangles[mask] = candidate
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw_normals, axis=1)
    if np.any(lengths <= 1e-12):
        return False
    normals = raw_normals / np.maximum(lengths[:, None], 1e-12)
    old = np.asarray(face_normals, dtype=np.float64)[incident]
    if np.any(np.sum(normals * old, axis=1) <= 0.0):
        return False
    quality = triangle_quality_from_vertices(triangles)
    return bool(quality.size > 0 and float(np.min(quality)) >= float(q_min_hard))


def _collapse_link_condition_ok(
    *,
    faces: np.ndarray,
    vertex_faces: list[np.ndarray],
    edge_faces: np.ndarray,
    edge_vertices: np.ndarray,
    edge_id: int,
) -> bool:
    vi, vj = [int(v) for v in edge_vertices[int(edge_id)]]
    incident_faces = [int(v) for v in edge_faces[int(edge_id)] if int(v) >= 0]
    if len(incident_faces) != 2:
        return False
    ni: set[int] = set()
    nj: set[int] = set()
    for face_id in vertex_faces[vi]:
        for value in faces[int(face_id)]:
            value = int(value)
            if value != vi:
                ni.add(value)
    for face_id in vertex_faces[vj]:
        for value in faces[int(face_id)]:
            value = int(value)
            if value != vj:
                nj.add(value)
    opposite: set[int] = set()
    for face_id in incident_faces:
        for value in faces[int(face_id)]:
            value = int(value)
            if value not in {vi, vj}:
                opposite.add(value)
    return bool((ni & nj) == opposite)


def _build_pass_fields(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_source: np.ndarray,
    policy: dict[str, np.ndarray],
    proxies: dict[str, np.ndarray],
    proxy_entries: dict[int, dict[str, Any]],
    config: WeightedRemeshConfig,
    pass_index: int,
    original_face_count: int,
) -> dict[str, Any]:
    topology = build_mesh_topology(faces)
    face_centers, face_normals, face_area, mean_edge_length = face_geometry(vertices, faces)
    edge_vertices = topology.edge_vertices
    edge_faces = topology.edge_faces
    edge_count = int(edge_vertices.shape[0])
    source_is_initial = (
        pass_index == 0
        and face_source.shape[0] == original_face_count
        and np.array_equal(face_source, np.arange(original_face_count, dtype=np.int64))
    )
    edge_length = np.linalg.norm(vertices[edge_vertices[:, 1]] - vertices[edge_vertices[:, 0]], axis=1).astype(np.float32)

    support = _clip01(_face_values(policy, "face_support", face_source, fallback=0.0))
    face_planar = _clip01(_face_values(policy, "face_weight_planar", face_source, fallback=0.0))
    face_detail = _clip01(_face_values(policy, "face_weight_detail", face_source, fallback=0.0))
    face_boundary = _clip01(_face_values(policy, "face_weight_boundary", face_source, fallback=0.0))
    face_uncertain = _clip01(_face_values(policy, "face_weight_uncertain", face_source, fallback=1.0))
    face_eps = _safe_positive(
        _face_values(policy, "face_position_tolerance", face_source, fallback=np.maximum(0.25 * mean_edge_length, 1e-6)),
        np.maximum(0.25 * mean_edge_length, 1e-6),
    )
    face_tau_a = _safe_positive(
        _face_values(policy, "face_tau_a", face_source, fallback=np.full(faces.shape[0], math.radians(5.0), dtype=np.float32)),
        np.float32(math.radians(5.0)),
    )
    face_direction_conf = _clip01(_face_values(policy, "face_direction_confidence", face_source, fallback=0.0))
    face_target = _safe_positive(
        _face_values(policy, "face_detail_target_length", face_source, fallback=np.maximum(mean_edge_length, 1e-6)),
        np.maximum(mean_edge_length, 1e-6),
    )
    source_mean_edge_length = _safe_positive(
        _face_values(policy, "face_mean_edge_length", face_source, fallback=np.maximum(mean_edge_length, 1e-6)),
        np.maximum(mean_edge_length, 1e-6),
    )
    face_proxy_values = _face_values(proxies, "face_proxy_id", face_source, fallback=-1, dtype=np.int32)
    face_proxy_id = np.asarray(face_proxy_values, dtype=np.int32)

    edge_proxy_id, edge_proxy_boundary = _edge_proxy_boundaries(edge_faces, face_proxy_id)
    edge_proxy_a = np.full((edge_count,), -1, dtype=np.int32)
    edge_proxy_b = np.full((edge_count,), -1, dtype=np.int32)
    valid_a = edge_faces[:, 0] >= 0
    edge_proxy_a[valid_a] = face_proxy_id[edge_faces[valid_a, 0]]
    valid_b = edge_faces[:, 1] >= 0
    edge_proxy_b[valid_b] = face_proxy_id[edge_faces[valid_b, 1]]
    edge_project_proxy_id = edge_proxy_id.copy()
    has_two_faces = edge_faces[:, 1] >= 0
    unresolved_proxy = edge_project_proxy_id < 0
    if np.any(has_two_faces & unresolved_proxy):
        ids = np.flatnonzero(has_two_faces & unresolved_proxy)
        proxy_a = face_proxy_id[edge_faces[ids, 0]]
        proxy_b = face_proxy_id[edge_faces[ids, 1]]
        one_sided_a = (proxy_a >= 0) & (proxy_b < 0)
        one_sided_b = (proxy_b >= 0) & (proxy_a < 0)
        edge_project_proxy_id[ids[one_sided_a]] = proxy_a[one_sided_a]
        edge_project_proxy_id[ids[one_sided_b]] = proxy_b[one_sided_b]
    target_policy = {
        "face_detail_target_length": face_target.astype(np.float32),
        "face_mean_edge_length": mean_edge_length.astype(np.float32),
        "face_planar_score": face_planar.astype(np.float32),
    }
    target_proxies = {
        "face_proxy_id": face_proxy_id.astype(np.int32),
        "edge_proxy_boundary": edge_proxy_boundary.astype(np.uint8),
    }
    face_final_target, proxy_target_length, target_stats = _compute_final_target_lengths(
        vertices=vertices,
        faces=faces,
        face_centers=face_centers,
        face_area=face_area,
        edge_vertices=edge_vertices,
        edge_faces=edge_faces,
        edge_length=edge_length,
        policy=target_policy,
        proxies=target_proxies,
        config=config,
    )

    edge_support = edge_face_min(edge_faces, support)
    edge_planar = edge_face_min(edge_faces, face_planar)
    edge_planar_any = edge_face_max(edge_faces, face_planar)
    edge_planar_mean = edge_face_mean(edge_faces, face_planar)
    edge_detail = edge_face_max(edge_faces, face_detail)
    edge_uncertain = edge_face_max(edge_faces, face_uncertain)
    edge_eps = edge_face_min(edge_faces, face_eps)
    edge_tau_a = edge_face_mean(edge_faces, face_tau_a)
    original_edge_tau_a = policy.get("edge_tau_a")
    if source_is_initial and original_edge_tau_a is not None and np.asarray(original_edge_tau_a).shape[0] == edge_count:
        edge_tau_a = _safe_positive(
            np.asarray(original_edge_tau_a, dtype=np.float32),
            np.float32(math.radians(5.0)),
        )
    edge_direction_conf = edge_face_mean(edge_faces, face_direction_conf)
    edge_target = _safe_positive(edge_face_min(edge_faces, face_final_target), np.maximum(edge_length, 1e-6))
    edge_target_mean = _safe_positive(edge_face_mean(edge_faces, face_final_target), np.maximum(edge_target, 1e-6))
    detail_mesh_target = np.maximum(
        face_target,
        source_mean_edge_length * float(config.detail_mesh_target_factor),
    ).astype(np.float32)
    detail_simplify_target = np.where(
        face_detail >= float(config.detail_simplify_min_weight),
        detail_mesh_target,
        face_target,
    ).astype(np.float32)
    edge_detail_target = _safe_positive(edge_face_mean(edge_faces, detail_simplify_target), np.maximum(edge_target, 1e-6))

    original_edge_weight = policy.get("edge_weight_boundary")
    if source_is_initial and original_edge_weight is not None and np.asarray(original_edge_weight).shape[0] == edge_count:
        edge_boundary = _clip01(np.asarray(original_edge_weight, dtype=np.float32))
    else:
        edge_boundary = edge_face_max(edge_faces, face_boundary)
    edge_boundary = np.maximum(edge_boundary, edge_proxy_boundary).clip(0.0, 1.0).astype(np.float32)

    support_gate = np.clip(
        (edge_support - float(config.support_threshold)) / max(1.0 - float(config.support_threshold), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    certainty_gate = (1.0 - 0.25 * edge_uncertain).clip(0.0, 1.0).astype(np.float32)
    planar_transition_gate = np.clip(
        (edge_planar_any - float(config.planar_transition_min_weight))
        / max(1.0 - float(config.planar_transition_min_weight), 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)
    planar_transition_gate *= (edge_proxy_boundary < 0.5).astype(np.float32)
    planar_transition_gate *= (topology.boundary_edges.astype(bool) == 0).astype(np.float32)
    planar_transition_gate *= (1.0 - 0.75 * edge_detail).clip(0.0, 1.0).astype(np.float32)
    feature_edge_allowed = (
        bool(config.feature_edge_collapse)
        & (topology.boundary_edges.astype(bool) == 0)
        & (edge_proxy_boundary >= 0.5)
        & (edge_proxy_a >= 0)
        & (edge_proxy_b >= 0)
        & (edge_proxy_a != edge_proxy_b)
        & (edge_planar_mean >= float(config.feature_edge_min_planar))
        & (edge_support >= float(config.support_threshold))
        & (edge_uncertain < float(config.feature_edge_max_uncertain))
    )
    boundary_softener = (
        float(config.planar_boundary_softening)
        * planar_transition_gate
        * support_gate
        * certainty_gate
    ).clip(0.0, 1.0).astype(np.float32)
    edge_effective_boundary = (edge_boundary * (1.0 - boundary_softener)).clip(0.0, 1.0).astype(np.float32)
    protected = (
        topology.boundary_edges.astype(bool)
        | ((edge_proxy_boundary >= 0.5) & (feature_edge_allowed == 0))
        | ((edge_effective_boundary >= float(config.boundary_threshold)) & (feature_edge_allowed == 0))
    )
    boundary_gate = np.where(
        feature_edge_allowed,
        1.0,
        (1.0 - edge_effective_boundary).clip(0.0, 1.0) ** 2,
    ).astype(np.float32)
    shortness = np.clip(
        (float(config.alpha_collapse) * edge_target - edge_length) / np.maximum(float(config.alpha_collapse) * edge_target, 1e-9),
        0.0,
        1.0,
    ).astype(np.float32)
    planar_shortness = np.clip(
        (float(config.alpha_planar_collapse) * edge_target - edge_length)
        / np.maximum(float(config.alpha_planar_collapse) * edge_target, 1e-9),
        0.0,
        1.0,
    ).astype(np.float32)
    planar_pressure = planar_shortness * edge_planar * support_gate
    planar_transition_shortness = np.clip(
        (float(config.alpha_planar_transition_collapse) * edge_target_mean - edge_length)
        / np.maximum(float(config.alpha_planar_transition_collapse) * edge_target_mean, 1e-9),
        0.0,
        1.0,
    ).astype(np.float32)
    planar_transition_pressure = (
        planar_transition_shortness
        * planar_transition_gate
        * edge_planar_any
        * support_gate
        * certainty_gate
    ).astype(np.float32)
    feature_shortness = np.clip(
        (float(config.alpha_feature_edge_collapse) * edge_target_mean - edge_length)
        / np.maximum(float(config.alpha_feature_edge_collapse) * edge_target_mean, 1e-9),
        0.0,
        1.0,
    ).astype(np.float32)
    feature_edge_pressure = (
        feature_shortness
        * feature_edge_allowed.astype(np.float32)
        * edge_planar_mean
        * support_gate
        * certainty_gate
    ).astype(np.float32)
    cleanup_pressure = shortness * (1.0 - edge_detail).clip(0.0, 1.0) * support_gate * (0.25 + 0.75 * edge_planar)
    detail_shortness = np.clip(
        (float(config.alpha_detail_collapse) * edge_detail_target - edge_length)
        / np.maximum(float(config.alpha_detail_collapse) * edge_detail_target, 1e-9),
        0.0,
        1.0,
    ).astype(np.float32)
    detail_simplify_pressure = (
        detail_shortness
        * (edge_detail >= float(config.detail_simplify_min_weight)).astype(np.float32)
        * edge_detail
        * support_gate
        * (0.35 + 0.65 * edge_direction_conf)
    ).astype(np.float32)
    collapse_pressure = np.maximum.reduce([planar_pressure, planar_transition_pressure, feature_edge_pressure, cleanup_pressure, detail_simplify_pressure]) * boundary_gate * certainty_gate
    collapse_pressure[protected] = 0.0
    collapse_pressure[edge_uncertain >= float(config.uncertainty_threshold)] = 0.0
    collapse_pressure = collapse_pressure.clip(0.0, 1.0).astype(np.float32)

    proxy_count = max(
        (max(proxy_entries.keys()) + 1 if proxy_entries else 0),
        int(np.max(face_proxy_id)) + 1 if np.any(face_proxy_id >= 0) else 0,
    )
    proxy_normals, proxy_offsets, proxy_rmse_q90, proxy_confidence = _proxy_plane_arrays(proxy_entries, proxy_count)
    return {
        "topology": topology,
        "vertex_faces": build_vertex_faces(faces, int(vertices.shape[0])),
        "face_normals": face_normals,
        "mean_edge_length": mean_edge_length,
        "face_final_target": face_final_target,
        "proxy_target_length": proxy_target_length,
        "target_stats": target_stats,
        "edge_vertices": edge_vertices,
        "edge_faces": edge_faces,
        "edge_length": edge_length,
        "edge_target": edge_target,
        "edge_target_mean": edge_target_mean,
        "edge_detail_target": edge_detail_target,
        "edge_support": edge_support,
        "edge_planar": edge_planar,
        "edge_planar_any": edge_planar_any,
        "edge_planar_mean": edge_planar_mean,
        "edge_detail": edge_detail,
        "edge_boundary": edge_boundary,
        "edge_effective_boundary": edge_effective_boundary,
        "edge_uncertain": edge_uncertain,
        "edge_eps": edge_eps,
        "edge_tau_a": edge_tau_a,
        "edge_direction_conf": edge_direction_conf,
        "edge_proxy_id": edge_proxy_id,
        "edge_project_proxy_id": edge_project_proxy_id,
        "edge_proxy_a": edge_proxy_a,
        "edge_proxy_b": edge_proxy_b,
        "edge_proxy_boundary": edge_proxy_boundary,
        "edge_feature_collapse_allowed": feature_edge_allowed.astype(np.uint8),
        "edge_is_mesh_boundary": topology.boundary_edges.astype(np.uint8),
        "edge_collapse_pressure": collapse_pressure,
        "edge_planar_transition_pressure": (planar_transition_pressure * boundary_gate).clip(0.0, 1.0).astype(np.float32),
        "edge_feature_collapse_pressure": (feature_edge_pressure * boundary_gate).clip(0.0, 1.0).astype(np.float32),
        "edge_detail_simplify_pressure": (detail_simplify_pressure * boundary_gate * certainty_gate).clip(0.0, 1.0).astype(np.float32),
        "proxy_normals": proxy_normals,
        "proxy_offsets": proxy_offsets,
        "proxy_rmse_q90": proxy_rmse_q90,
        "proxy_confidence": proxy_confidence,
        "face_weight_planar": face_planar,
        "face_weight_detail": face_detail,
        "face_weight_boundary": face_boundary,
        "face_weight_uncertain": face_uncertain,
        "face_support": support,
        "face_eps": face_eps,
        "face_proxy_id": face_proxy_id,
        "face_detail_simplify_target": detail_simplify_target,
    }


def _evaluate_collapse_candidates(
    *,
    pass_index: int,
    candidate_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_source: np.ndarray,
    fields: dict[str, Any],
    config: WeightedRemeshConfig,
    source_accum: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    edge_vertices = fields["edge_vertices"]
    edge_faces = fields["edge_faces"]
    edge_count = int(edge_vertices.shape[0])
    accepted = np.zeros((edge_count,), dtype=np.uint8)
    reject_code = np.zeros((edge_count,), dtype=np.uint8)
    candidate_positions = np.zeros((edge_count, 3), dtype=np.float64)
    projection_distance = np.zeros((edge_count,), dtype=np.float32)
    touched_vertices = np.zeros((vertices.shape[0],), dtype=bool)
    accepted_count = 0
    max_rows = int(config.max_jsonl_rows)

    for rank, edge_id_raw in enumerate(candidate_ids):
        edge_id = int(edge_id_raw)
        vi, vj = [int(v) for v in edge_vertices[edge_id]]
        reason = "accepted"
        feature_edge = bool(fields["edge_feature_collapse_allowed"][edge_id])
        if float(fields["edge_collapse_pressure"][edge_id]) < float(config.collapse_threshold):
            reason = "reject_density"
        elif (
            bool(fields["edge_is_mesh_boundary"][edge_id])
            or (float(fields["edge_proxy_boundary"][edge_id]) >= 0.5 and not feature_edge)
            or (float(fields["edge_effective_boundary"][edge_id]) >= float(config.boundary_threshold) and not feature_edge)
        ):
            reason = "reject_boundary"
        elif float(fields["edge_support"][edge_id]) < float(config.support_threshold):
            reason = "reject_support"
        elif float(fields["edge_uncertain"][edge_id]) >= float(config.uncertainty_threshold):
            reason = "reject_uncertainty"
        elif bool(touched_vertices[vi]) or bool(touched_vertices[vj]):
            reason = "reject_batch_conflict"
        elif not _collapse_link_condition_ok(
            faces=faces,
            vertex_faces=fields["vertex_faces"],
            edge_faces=edge_faces,
            edge_vertices=edge_vertices,
            edge_id=edge_id,
        ):
            reason = "reject_topology"

        midpoint = 0.5 * (vertices[vi] + vertices[vj])
        if feature_edge:
            projection_strength = (
                float(config.planar_projection_strength)
                * float(fields["edge_planar_mean"][edge_id])
                * (1.0 - 0.25 * float(fields["edge_uncertain"][edge_id]))
            )
            candidate, projected_distance, projected_ok = _edge_feature_project(
                midpoint=midpoint,
                proxy_a=int(fields["edge_proxy_a"][edge_id]),
                proxy_b=int(fields["edge_proxy_b"][edge_id]),
                proxy_normals=fields["proxy_normals"],
                proxy_offsets=fields["proxy_offsets"],
                strength=projection_strength,
            )
            if reason == "accepted" and not projected_ok:
                reason = "reject_plane"
        else:
            projection_strength = (
                float(config.planar_projection_strength)
                * max(float(fields["edge_planar"][edge_id]), 0.75 * float(fields["edge_planar_any"][edge_id]))
                * (1.0 - 0.25 * float(fields["edge_uncertain"][edge_id]))
                * (1.0 - 0.5 * float(fields["edge_detail"][edge_id]))
            )
            candidate, projected_distance = _edge_plane_project(
                midpoint=midpoint,
                proxy_id=int(fields["edge_project_proxy_id"][edge_id]),
                proxy_normals=fields["proxy_normals"],
                proxy_offsets=fields["proxy_offsets"],
                strength=projection_strength,
            )
        candidate_positions[edge_id] = candidate
        projection_distance[edge_id] = np.float32(projected_distance)
        if reason == "accepted":
            max_projection = (
                float(fields["edge_eps"][edge_id])
                * (1.0 + float(config.max_projection_distance_factor) * float(fields["edge_planar"][edge_id]))
                * max(0.25, 1.0 - 0.75 * float(fields["edge_uncertain"][edge_id]))
            )
            if projected_distance > max(max_projection, 1e-9):
                reason = "reject_projection"

        plane_error = 0.0
        proxy_ids_to_check = (
            [int(fields["edge_proxy_a"][edge_id]), int(fields["edge_proxy_b"][edge_id])]
            if feature_edge
            else [int(fields["edge_project_proxy_id"][edge_id])]
        )
        for proxy_id in proxy_ids_to_check:
            if reason == "accepted" and proxy_id >= 0 and proxy_id < fields["proxy_normals"].shape[0]:
                normal = fields["proxy_normals"][proxy_id]
                if float(np.linalg.norm(normal)) > 1e-12:
                    residual = abs(float(np.dot(normal, candidate) + float(fields["proxy_offsets"][proxy_id])))
                    tolerance = max(
                        float(fields["proxy_rmse_q90"][proxy_id]) if proxy_id < fields["proxy_rmse_q90"].shape[0] else 0.0,
                        float(fields["edge_eps"][edge_id]),
                        1e-9,
                    )
                    plane_error = max(float(plane_error), residual / tolerance)
                    if plane_error > float(config.plane_residual_factor):
                        reason = "reject_plane"

        gates = simulate_collapse_gates(
            vertices=vertices,
            faces=faces,
            vertex_faces=fields["vertex_faces"],
            face_normals=fields["face_normals"],
            edge_vertices=edge_vertices,
            edge_faces=edge_faces,
            edge_id=edge_id,
            candidate_position=candidate,
            detail_weight=float(fields["edge_detail"][edge_id]),
            tau_a=float(fields["edge_tau_a"][edge_id]),
            q_min_hard=float(config.q_min_hard),
            q_min_soft=float(config.q_min_soft),
        )
        if reason == "accepted" and not bool(gates["topologyOk"]):
            reason = "reject_topology"
        normal_limit = float(config.max_normal_error) * (1.0 + 1.5 * float(fields["edge_planar_any"][edge_id]) + (1.0 if feature_edge else 0.0))
        quality_limit = float(config.max_quality_error) * (1.0 + 0.75 * float(fields["edge_planar_any"][edge_id]) + (0.5 if feature_edge else 0.0))
        if reason == "accepted" and float(gates["normalError"]) > normal_limit:
            reason = "reject_normal"
        if reason == "accepted" and (not bool(gates["qualityOk"]) or float(gates["qualityError"]) > quality_limit):
            reason = "reject_quality"

        if reason == "accepted":
            accepted[edge_id] = 1
            touched_vertices[vi] = True
            touched_vertices[vj] = True
            accepted_count += 1
        reject_code[edge_id] = REJECT_NAME_TO_CODE[reason]

        incident = [int(v) for v in edge_faces[edge_id] if int(v) >= 0]
        for face_id in incident:
            src = int(face_source[face_id])
            if src >= 0:
                source_accum["source_face_collapse_pressure"][src] = max(
                    float(source_accum["source_face_collapse_pressure"][src]),
                    float(fields["edge_collapse_pressure"][edge_id]),
                )
                if reason == "accepted":
                    source_accum["source_face_collapse_accept"][src] = 1.0
                elif reason in source_accum:
                    source_accum[reason][src] = 1.0
                source_accum["source_face_detail_simplify_pressure"][src] = max(
                    float(source_accum["source_face_detail_simplify_pressure"][src]),
                    float(fields["edge_detail_simplify_pressure"][edge_id]),
                )
                source_accum["source_face_planar_transition_pressure"][src] = max(
                    float(source_accum["source_face_planar_transition_pressure"][src]),
                    float(fields["edge_planar_transition_pressure"][edge_id]),
                )
                source_accum["source_face_feature_collapse_pressure"][src] = max(
                    float(source_accum["source_face_feature_collapse_pressure"][src]),
                    float(fields["edge_feature_collapse_pressure"][edge_id]),
                )

        if max_rows <= 0 or len(rows) < max_rows or reason == "accepted":
            rows.append(
                {
                    "operation": "collapse",
                    "pass": int(pass_index),
                    "rank": int(rank),
                    "edgeId": edge_id,
                    "vertices": [vi, vj],
                    "faces": incident,
                    "sourceFaces": [int(face_source[fid]) for fid in incident],
                    "accepted": bool(reason == "accepted"),
                    "reason": reason,
                    "candidatePosition": [float(v) for v in candidate],
                    "gates": {
                        "plane": float(plane_error),
                        "normal": float(gates["normalError"]) if np.isfinite(gates["normalError"]) else 1.0e9,
                        "quality": float(gates["qualityError"]) if np.isfinite(gates["qualityError"]) else 1.0e9,
                        "minQuality": float(gates["minQuality"]),
                        "projectionDistance": float(projected_distance),
                    },
                    "signals": {
                        "pressure": float(fields["edge_collapse_pressure"][edge_id]),
                        "length": float(fields["edge_length"][edge_id]),
                        "targetLength": float(fields["edge_target"][edge_id]),
                        "targetLengthMean": float(fields["edge_target_mean"][edge_id]),
                        "detailTargetLength": float(fields["edge_detail_target"][edge_id]),
                        "planarTransitionPressure": float(fields["edge_planar_transition_pressure"][edge_id]),
                        "featureCollapsePressure": float(fields["edge_feature_collapse_pressure"][edge_id]),
                        "featureEdge": bool(feature_edge),
                        "detailSimplifyPressure": float(fields["edge_detail_simplify_pressure"][edge_id]),
                        "support": float(fields["edge_support"][edge_id]),
                        "planar": float(fields["edge_planar"][edge_id]),
                        "planarAny": float(fields["edge_planar_any"][edge_id]),
                        "planarMean": float(fields["edge_planar_mean"][edge_id]),
                        "detail": float(fields["edge_detail"][edge_id]),
                        "boundary": float(fields["edge_boundary"][edge_id]),
                        "effectiveBoundary": float(fields["edge_effective_boundary"][edge_id]),
                        "proxyBoundary": float(fields["edge_proxy_boundary"][edge_id]),
                        "proxyA": int(fields["edge_proxy_a"][edge_id]),
                        "proxyB": int(fields["edge_proxy_b"][edge_id]),
                        "uncertain": float(fields["edge_uncertain"][edge_id]),
                        "projectionStrength": float(projection_strength),
                    },
                }
            )

    return accepted, reject_code, candidate_positions, projection_distance, accepted_count


def _apply_batched_collapses(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_source: np.ndarray,
    edge_vertices: np.ndarray,
    accepted: np.ndarray,
    candidate_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    selected = np.flatnonzero(np.asarray(accepted, dtype=np.uint8) > 0)
    if selected.size == 0:
        return vertices, faces, face_source, {"degenerateFacesRemoved": 0, "duplicateFacesRemoved": 0, "unreferencedVerticesRemoved": 0}

    representative = np.arange(vertices.shape[0], dtype=np.int64)
    new_vertices = vertices.copy()
    for edge_id in selected:
        a, b = [int(v) for v in edge_vertices[int(edge_id)]]
        keep, drop = (a, b) if a < b else (b, a)
        representative[drop] = keep
        new_vertices[keep] = candidate_positions[int(edge_id)]

    remapped_faces = representative[faces]
    degenerate = (
        (remapped_faces[:, 0] == remapped_faces[:, 1])
        | (remapped_faces[:, 1] == remapped_faces[:, 2])
        | (remapped_faces[:, 2] == remapped_faces[:, 0])
    )
    faces_kept = remapped_faces[~degenerate]
    source_kept = face_source[~degenerate]
    degenerate_removed = int(np.count_nonzero(degenerate))
    before_unique = int(faces_kept.shape[0])
    faces_unique, source_unique, unique_keep = _unique_faces(faces_kept, source_kept)
    duplicate_removed = before_unique - int(np.count_nonzero(unique_keep))
    compact_vertices, compact_faces, vertex_remap = _remove_unreferenced(new_vertices, faces_unique)
    unreferenced_removed = int(np.count_nonzero(vertex_remap < 0))
    return compact_vertices, compact_faces, source_unique, {
        "degenerateFacesRemoved": degenerate_removed,
        "duplicateFacesRemoved": duplicate_removed,
        "unreferencedVerticesRemoved": unreferenced_removed,
    }


def _apply_flip_pass(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_source: np.ndarray,
    fields: dict[str, Any],
    config: WeightedRemeshConfig,
    pass_index: int,
    source_accum: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, int, dict[str, float]]:
    edge_vertices = fields["edge_vertices"]
    edge_faces = fields["edge_faces"]
    edge_count = int(edge_vertices.shape[0])
    if edge_count == 0 or int(config.max_flips_per_pass) == 0:
        return faces, 0, {"candidateEdges": 0.0, "acceptedFlips": 0.0}

    has_two = edge_faces[:, 1] >= 0
    eligible = (
        has_two
        & (fields["edge_is_mesh_boundary"].astype(bool) == 0)
        & (fields["edge_proxy_boundary"] < 0.5)
        & (fields["edge_effective_boundary"] < float(config.boundary_threshold))
        & (fields["edge_uncertain"] < float(config.uncertainty_threshold))
        & (fields["edge_support"] >= float(config.support_threshold))
    )
    seed_score = (
        eligible.astype(np.float32)
        * fields["edge_support"]
        * (0.4 + 0.6 * fields["edge_detail"])
        * np.square((1.0 - fields["edge_effective_boundary"]).clip(0.0, 1.0))
        * (1.0 - 0.25 * fields["edge_uncertain"]).clip(0.0, 1.0)
    )
    candidate_ids = _candidate_ids(seed_score, int(config.max_flips_per_pass) * 4)
    edge_keys = {tuple(sorted((int(a), int(b)))) for a, b in edge_vertices}
    touched_vertices = np.zeros((vertices.shape[0],), dtype=bool)
    touched_faces = np.zeros((faces.shape[0],), dtype=bool)
    new_faces = faces.copy()
    accepted_count = 0
    improvements: list[float] = []
    max_rows = int(config.max_jsonl_rows)

    for rank, edge_id_raw in enumerate(candidate_ids):
        if accepted_count >= int(config.max_flips_per_pass):
            break
        edge_id = int(edge_id_raw)
        if seed_score[edge_id] <= 0.0:
            continue
        face_a, face_b = [int(v) for v in edge_faces[edge_id]]
        if face_a < 0 or face_b < 0 or touched_faces[face_a] or touched_faces[face_b]:
            continue
        va, vb = [int(v) for v in edge_vertices[edge_id]]
        if touched_vertices[va] or touched_vertices[vb]:
            continue
        fa = new_faces[face_a]
        fb = new_faces[face_b]
        opp_a = [int(v) for v in fa if int(v) not in {va, vb}]
        opp_b = [int(v) for v in fb if int(v) not in {va, vb}]
        if len(opp_a) != 1 or len(opp_b) != 1:
            continue
        vc = opp_a[0]
        vd = opp_b[0]
        if len({va, vb, vc, vd}) != 4 or touched_vertices[vc] or touched_vertices[vd]:
            continue
        new_edge_key = tuple(sorted((vc, vd)))
        if new_edge_key in edge_keys:
            continue
        gates = simulate_flip_quality(vertices=vertices, faces=new_faces, edge_vertices=edge_vertices, edge_faces=edge_faces, edge_id=edge_id)
        if not bool(gates["valid"]) or float(gates["qualityImprovement"]) < float(config.flip_min_quality_improvement):
            continue

        new_faces[face_a] = np.asarray([vc, vd, vb], dtype=np.int64)
        new_faces[face_b] = np.asarray([vd, vc, va], dtype=np.int64)
        touched_faces[face_a] = True
        touched_faces[face_b] = True
        touched_vertices[[va, vb, vc, vd]] = True
        accepted_count += 1
        improvements.append(float(gates["qualityImprovement"]))
        for face_id in (face_a, face_b):
            src = int(face_source[face_id])
            if src >= 0:
                source_accum["source_face_flip_accept"][src] = 1.0
                source_accum["source_face_flip_pressure"][src] = max(
                    float(source_accum["source_face_flip_pressure"][src]),
                    float(seed_score[edge_id]),
                )
        if max_rows <= 0 or len(rows) < max_rows:
            rows.append(
                {
                    "operation": "flip",
                    "pass": int(pass_index),
                    "rank": int(rank),
                    "edgeId": int(edge_id),
                    "vertices": [int(va), int(vb)],
                    "faces": [int(face_a), int(face_b)],
                    "sourceFaces": [int(face_source[face_a]), int(face_source[face_b])],
                    "accepted": True,
                    "reason": "accepted",
                    "gates": {
                        "qualityImprovement": float(gates["qualityImprovement"]),
                        "oldMinQuality": float(gates["oldMinQuality"]),
                        "newMinQuality": float(gates["newMinQuality"]),
                    },
                    "signals": {
                        "pressure": float(seed_score[edge_id]),
                        "support": float(fields["edge_support"][edge_id]),
                        "detail": float(fields["edge_detail"][edge_id]),
                        "planar": float(fields["edge_planar"][edge_id]),
                        "boundary": float(fields["edge_boundary"][edge_id]),
                        "effectiveBoundary": float(fields["edge_effective_boundary"][edge_id]),
                        "uncertain": float(fields["edge_uncertain"][edge_id]),
                    },
                }
            )
    return new_faces, accepted_count, {
        "candidateEdges": float(candidate_ids.size),
        "acceptedFlips": float(accepted_count),
        "qualityImprovement": _summarize(np.asarray(improvements, dtype=np.float32)) if improvements else _summarize(np.zeros((0,), dtype=np.float32)),
    }


def _apply_relax_project(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_source: np.ndarray,
    fields: dict[str, Any],
    config: WeightedRemeshConfig,
    source_accum: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    if vertices.size == 0 or faces.size == 0:
        return vertices, {"planarProjectedVertices": 0, "detailRelaxedVertices": 0}

    topology = fields["topology"]
    vertex_faces = fields["vertex_faces"]
    neighbors = _build_vertex_neighbors(topology.edge_vertices, int(vertices.shape[0]))
    face_normals = np.asarray(fields["face_normals"], dtype=np.float64)
    _, _, face_area, _ = face_geometry(vertices, faces)
    vertex_normals = _vertex_normals(faces, face_normals, face_area, int(vertices.shape[0]))
    vertex_planar = _vertex_face_mean(faces, fields["face_weight_planar"], int(vertices.shape[0]))
    vertex_detail = _vertex_face_max(faces, fields["face_weight_detail"], int(vertices.shape[0]))
    vertex_uncertain = _vertex_face_max(faces, fields["face_weight_uncertain"], int(vertices.shape[0]))
    vertex_support = _vertex_face_mean(faces, fields["face_support"], int(vertices.shape[0]))
    edge_boundary_for_vertices = np.maximum(
        fields.get("edge_effective_boundary", fields["edge_boundary"]),
        np.maximum(fields["edge_proxy_boundary"], fields["edge_is_mesh_boundary"].astype(np.float32)),
    )
    vertex_boundary = np.maximum(
        0.5 * _vertex_face_max(faces, fields["face_weight_boundary"], int(vertices.shape[0])),
        _vertex_edge_max(topology.edge_vertices, edge_boundary_for_vertices, int(vertices.shape[0])),
    ).clip(0.0, 1.0)
    vertex_eps = _safe_positive(_vertex_face_mean(faces, fields["face_eps"], int(vertices.shape[0])), 1e-6)
    vertex_proxy, vertex_proxy_fraction = _dominant_vertex_proxy(
        faces,
        fields["face_proxy_id"],
        fields["face_weight_planar"],
        int(vertices.shape[0]),
    )

    out = vertices.copy()
    planar_total_distance = np.zeros((vertices.shape[0],), dtype=np.float32)
    detail_total_distance = np.zeros((vertices.shape[0],), dtype=np.float32)
    planar_projected = np.zeros((vertices.shape[0],), dtype=np.float32)
    detail_relaxed = np.zeros((vertices.shape[0],), dtype=np.float32)

    for _ in range(max(int(config.detail_relax_iterations), 0)):
        next_vertices = out.copy()
        for vertex_id, nbrs in enumerate(neighbors):
            if nbrs.size == 0:
                continue
            gate = (
                float(vertex_detail[vertex_id])
                * float(vertex_support[vertex_id])
                * (1.0 - 0.25 * float(vertex_uncertain[vertex_id]))
                * np.square(max(0.0, 1.0 - float(vertex_boundary[vertex_id])))
                * (1.0 - 0.35 * float(vertex_planar[vertex_id]))
            )
            if gate < float(config.detail_relax_min_weight):
                continue
            normal = vertex_normals[vertex_id]
            if float(np.linalg.norm(normal)) <= 1e-12:
                continue
            target = np.mean(out[nbrs], axis=0)
            delta = target - out[vertex_id]
            tangential = delta - float(np.dot(delta, normal)) * normal
            step = float(config.detail_relax_strength) * gate * tangential
            length = float(np.linalg.norm(step))
            max_step = float(vertex_eps[vertex_id]) * float(config.detail_relax_max_distance_factor)
            if length <= 1e-12 or max_step <= 0.0:
                continue
            if length > max_step:
                step *= max_step / length
                length = max_step
            candidate = out[vertex_id] + step
            if not _vertex_quality_ok(
                vertices=out,
                faces=faces,
                vertex_faces=vertex_faces,
                face_normals=face_normals,
                vertex_id=vertex_id,
                candidate=candidate,
                q_min_hard=float(config.q_min_hard),
            ):
                continue
            next_vertices[vertex_id] = candidate
            detail_total_distance[vertex_id] += np.float32(length)
            detail_relaxed[vertex_id] = 1.0
        out = next_vertices

    for _ in range(max(int(config.planar_project_iterations), 0)):
        next_vertices = out.copy()
        for vertex_id in range(out.shape[0]):
            proxy_id = int(vertex_proxy[vertex_id])
            if proxy_id < 0 or proxy_id >= fields["proxy_normals"].shape[0]:
                continue
            gate = (
                float(vertex_planar[vertex_id])
                * float(vertex_support[vertex_id])
                * float(vertex_proxy_fraction[vertex_id])
                * (1.0 - 0.25 * float(vertex_uncertain[vertex_id]))
                * (1.0 - 0.35 * float(vertex_detail[vertex_id]))
                * max(0.0, 1.0 - 0.5 * float(vertex_boundary[vertex_id]))
            )
            if gate < float(config.planar_project_min_weight):
                continue
            normal = fields["proxy_normals"][proxy_id]
            if float(np.linalg.norm(normal)) <= 1e-12:
                continue
            signed = float(np.dot(normal, out[vertex_id]) + float(fields["proxy_offsets"][proxy_id]))
            max_step = float(vertex_eps[vertex_id]) * float(config.planar_project_max_distance_factor)
            step_scalar = np.clip(float(config.planar_projection_strength) * gate * signed, -max_step, max_step)
            step = -step_scalar * normal
            length = float(np.linalg.norm(step))
            if length <= 1e-12:
                continue
            candidate = out[vertex_id] + step
            if not _vertex_quality_ok(
                vertices=out,
                faces=faces,
                vertex_faces=vertex_faces,
                face_normals=face_normals,
                vertex_id=vertex_id,
                candidate=candidate,
                q_min_hard=float(config.q_min_hard),
            ):
                continue
            next_vertices[vertex_id] = candidate
            planar_total_distance[vertex_id] += np.float32(length)
            planar_projected[vertex_id] = 1.0
        out = next_vertices

    face_planar_projected = _face_max_from_vertices(faces, planar_projected)
    face_detail_relaxed = _face_max_from_vertices(faces, detail_relaxed)
    face_planar_distance = _face_mean_from_vertices(faces, planar_total_distance)
    face_detail_distance = _face_mean_from_vertices(faces, detail_total_distance)
    for face_id, source_id_raw in enumerate(face_source):
        source_id = int(source_id_raw)
        if source_id < 0:
            continue
        source_accum["source_face_planar_projected"][source_id] = max(
            float(source_accum["source_face_planar_projected"][source_id]),
            float(face_planar_projected[face_id]),
        )
        source_accum["source_face_detail_relaxed"][source_id] = max(
            float(source_accum["source_face_detail_relaxed"][source_id]),
            float(face_detail_relaxed[face_id]),
        )
        source_accum["source_face_planar_projection_distance"][source_id] = max(
            float(source_accum["source_face_planar_projection_distance"][source_id]),
            float(face_planar_distance[face_id]),
        )
        source_accum["source_face_detail_relax_distance"][source_id] = max(
            float(source_accum["source_face_detail_relax_distance"][source_id]),
            float(face_detail_distance[face_id]),
        )

    return out, {
        "planarProjectedVertices": int(np.count_nonzero(planar_projected > 0.0)),
        "detailRelaxedVertices": int(np.count_nonzero(detail_relaxed > 0.0)),
        "planarProjectionDistance": _summarize(planar_total_distance, planar_projected > 0.0),
        "detailRelaxDistance": _summarize(detail_total_distance, detail_relaxed > 0.0),
    }


def _write_debug_meshes(
    vertices: np.ndarray,
    faces: np.ndarray,
    arrays: dict[str, np.ndarray],
    output_dir: Path,
) -> dict[str, Path]:
    debug_dir = output_dir / "debug_meshes" / "weighted_remesh"
    fields = {
        "collapse_pressure": "source_face_collapse_pressure",
        "collapse_accept": "source_face_collapse_accept",
        "detail_simplify_pressure": "source_face_detail_simplify_pressure",
        "planar_transition_pressure": "source_face_planar_transition_pressure",
        "feature_collapse_pressure": "source_face_feature_collapse_pressure",
        "flip_pressure": "source_face_flip_pressure",
        "flip_accept": "source_face_flip_accept",
        "planar_projected": "source_face_planar_projected",
        "planar_projection_distance": "source_face_planar_projection_distance",
        "detail_relaxed": "source_face_detail_relaxed",
        "detail_relax_distance": "source_face_detail_relax_distance",
        "removed_by_collapse": "source_face_removed_by_collapse",
        "survived": "source_face_survived",
        "reject_boundary": "source_face_reject_boundary",
        "reject_uncertainty": "source_face_reject_uncertainty",
        "reject_quality": "source_face_reject_quality",
        "reject_normal": "source_face_reject_normal",
        "reject_projection": "source_face_reject_projection",
        "weight_planar": "source_face_weight_planar",
        "weight_detail": "source_face_weight_detail",
        "weight_boundary": "source_face_weight_boundary",
        "weight_uncertain": "source_face_weight_uncertain",
    }
    paths: dict[str, Path] = {}
    for name, key in fields.items():
        if key not in arrays:
            continue
        values = np.asarray(arrays[key], dtype=np.float32)
        path = debug_dir / f"{name}.ply"
        write_face_scalar_mesh(path, vertices, faces, values, np.isfinite(values), vmin=0.0, vmax=1.0)
        paths[name] = path
    return paths


def _build_report(summary: dict[str, Any]) -> str:
    counts = summary.get("counts", {})
    input_faces = int(counts.get("inputFaces", 0))
    output_faces = int(counts.get("outputFaces", 0))
    input_vertices = int(counts.get("inputVertices", 0))
    output_vertices = int(counts.get("outputVertices", 0))
    face_reduction = 0.0 if input_faces <= 0 else 1.0 - float(output_faces) / float(input_faces)
    vertex_reduction = 0.0 if input_vertices <= 0 else 1.0 - float(output_vertices) / float(input_vertices)
    config = summary.get("config", {})
    lines = [
        "# Weighted Local Remesh Report",
        "",
        "## Overall",
        "",
        f"- Input mesh: `{summary.get('inputs', {}).get('meshPath', '')}`",
        f"- Output mesh: `{summary.get('outputs', {}).get('outputMesh', '')}`",
        f"- Faces: {input_faces:,} -> {output_faces:,} ({face_reduction:.2%} reduction)",
        f"- Vertices: {input_vertices:,} -> {output_vertices:,} ({vertex_reduction:.2%} reduction)",
        f"- Accepted collapses: {int(counts.get('acceptedCollapses', 0)):,}",
        f"- Accepted flips: {int(counts.get('acceptedFlips', 0)):,}",
        f"- Removed source faces: {int(counts.get('removedSourceFaces', 0)):,}",
        f"- Planar-projected source faces: {int(counts.get('planarProjectedSourceFaces', 0)):,}",
        f"- Detail-relaxed source faces: {int(counts.get('detailRelaxedSourceFaces', 0)):,}",
        "",
        "## Key Settings",
        "",
        f"- Pre-project before collapse: {bool(config.get('pre_project_before_collapse', False))}",
        f"- Collapse threshold: {_format_number(config.get('collapse_threshold', 0.0))}",
        f"- Alpha planar collapse: {_format_number(config.get('alpha_planar_collapse', 0.0))}",
        f"- Alpha planar transition collapse: {_format_number(config.get('alpha_planar_transition_collapse', 0.0))}",
        f"- Feature-edge collapse: {bool(config.get('feature_edge_collapse', False))}",
        f"- Alpha feature-edge collapse: {_format_number(config.get('alpha_feature_edge_collapse', 0.0))}",
        f"- Planar boundary softening: {_format_number(config.get('planar_boundary_softening', 0.0))}",
        f"- Alpha detail collapse: {_format_number(config.get('alpha_detail_collapse', 0.0))}",
        f"- Detail mesh target factor: {_format_number(config.get('detail_mesh_target_factor', 0.0))}",
        f"- Detail relax iterations/strength: {config.get('detail_relax_iterations', 0)} / {_format_number(config.get('detail_relax_strength', 0.0))}",
        "",
    ]

    pre_project = summary.get("preProject", {})
    if pre_project:
        lines.extend(
            [
                "## Pre-Projection",
                "",
                f"- Planar-projected vertices: {int(pre_project.get('planarProjectedVertices', 0)):,}",
                f"- Planar projection distance: {_format_summary(pre_project.get('planarProjectionDistance', {}))}",
                "",
            ]
        )

    lines.extend(["## Passes", ""])
    for pass_summary in summary.get("passes", []):
        pass_id = int(pass_summary.get("pass", 0)) + 1
        before_faces = int(pass_summary.get("facesBefore", 0))
        after_faces = int(pass_summary.get("facesAfter", 0))
        lines.extend(
            [
                f"### Pass {pass_id}",
                "",
                f"- Faces: {before_faces:,} -> {after_faces:,}",
                f"- Collapse candidates: {int(pass_summary.get('candidateEdges', 0)):,}",
                f"- Accepted collapses: {int(pass_summary.get('acceptedCollapses', 0)):,}",
                f"- Accepted flips: {int(pass_summary.get('acceptedFlips', 0)):,}",
            ]
        )
        reject_counts = pass_summary.get("rejectCounts", {})
        if reject_counts:
            reject_text = ", ".join(f"{name}={count:,}" for name, count in sorted(reject_counts.items()))
            lines.append(f"- Reject counts: {reject_text}")
        stats = pass_summary.get("stats", {})
        if stats:
            lines.extend(
                [
                    f"- Collapse pressure: {_format_summary(stats.get('collapsePressure', {}))}",
                    f"- Planar transition pressure: {_format_summary(stats.get('planarTransitionPressure', {}))}",
                    f"- Feature collapse pressure: {_format_summary(stats.get('featureCollapsePressure', {}))}",
                    f"- Detail simplify pressure: {_format_summary(stats.get('detailSimplifyPressure', {}))}",
                    f"- Accepted projection distance: {_format_summary(stats.get('acceptedProjectionDistance', {}))}",
                ]
            )
        cleanup = pass_summary.get("cleanup", {})
        if cleanup:
            lines.append(
                "- Cleanup: "
                + ", ".join(f"{key}={value:,}" for key, value in cleanup.items())
            )
        lines.append("")

    relax_project = summary.get("relaxProject", {})
    if relax_project:
        lines.extend(
            [
                "## Final Relax/Project",
                "",
                f"- Planar-projected vertices: {int(relax_project.get('planarProjectedVertices', 0)):,}",
                f"- Detail-relaxed vertices: {int(relax_project.get('detailRelaxedVertices', 0)):,}",
                f"- Planar projection distance: {_format_summary(relax_project.get('planarProjectionDistance', {}))}",
                f"- Detail relax distance: {_format_summary(relax_project.get('detailRelaxDistance', {}))}",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- If planar transition pressure is high but accepted collapses are low, inspect reject counts; boundary, projection, quality, and topology rejects identify the blocking gate.",
            "- If feature collapse pressure is high but accepted collapses are low, seam simplification is being blocked by topology, normal, quality, or plane-intersection gates.",
            "- If pre-projection moved many vertices but planar collapses remain low, the target length or quality gates are probably limiting face reduction.",
            "- If detail simplify pressure is high but detail collapses are low, detail regions are being protected by boundary, normal, quality, or uncertainty gates.",
            "- The operation JSONL records individual accepted operations and sampled rejected candidates; this report summarizes the full pass-level behavior.",
        ]
    )
    return "\n".join(lines)


def compute_weighted_remesh(config: WeightedRemeshConfig, progress: ProgressFn | None = None) -> WeightedRemeshResult:
    progress = progress or _progress_default
    mesh_path = config.mesh_path.expanduser().resolve()
    policy_npz = config.policy_npz.expanduser().resolve()
    proxies_npz = config.proxies_npz.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_mesh = output_dir / str(config.output_name)
    diagnostics_npz = output_dir / f"{config.diagnostics_name}.npz"
    operations_jsonl = output_dir / f"{config.operations_name}.jsonl"
    summary_json = output_dir / f"{Path(config.output_name).stem}_summary.json"
    report_md = output_dir / f"{Path(config.output_name).stem}_report.md"
    for path in (output_mesh, diagnostics_npz, operations_jsonl, summary_json, report_md):
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"Weighted remesh output exists: {path}. Re-run with overwrite=True.")

    progress(f"[weighted-remesh] Mesh: {mesh_path}")
    original_vertices, original_faces = load_mesh_arrays(mesh_path)
    policy = _load_npz(policy_npz)
    proxies = _load_npz(proxies_npz)
    _required(
        policy,
        [
            "face_support",
            "face_weight_planar",
            "face_weight_detail",
            "face_weight_boundary",
            "face_weight_uncertain",
            "face_detail_target_length",
            "face_position_tolerance",
        ],
        "Policy npz",
    )
    _required(proxies, ["face_proxy_id"], "Proxies npz")
    original_face_count = int(original_faces.shape[0])
    if np.asarray(policy["face_weight_planar"]).shape[0] != original_face_count:
        raise ValueError("Policy face arrays do not match the input mesh. Re-run Phase 3 for this mesh.")
    if np.asarray(proxies["face_proxy_id"]).shape[0] != original_face_count:
        raise ValueError("Proxy face arrays do not match the input mesh. Re-run Phase 3 for this mesh.")

    proxy_entries = _load_proxy_entries(config.proxies_json)
    vertices = original_vertices.copy()
    faces = original_faces.copy()
    face_source = np.arange(original_face_count, dtype=np.int64)
    input_stats = _mesh_stats(vertices, faces)
    rows: list[dict[str, Any]] = []
    pass_summaries: list[dict[str, Any]] = []
    total_accepted = 0
    total_flips = 0
    source_face_final_target = _safe_positive(policy["face_detail_target_length"], 1.0).astype(np.float32)
    pre_project_summary: dict[str, Any] = {}

    source_accum: dict[str, np.ndarray] = {
        "source_face_collapse_pressure": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_collapse_accept": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_detail_simplify_pressure": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_planar_transition_pressure": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_feature_collapse_pressure": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_flip_pressure": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_flip_accept": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_planar_projected": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_planar_projection_distance": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_detail_relaxed": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_detail_relax_distance": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_removed_by_collapse": np.zeros((original_face_count,), dtype=np.float32),
        "source_face_survived": np.ones((original_face_count,), dtype=np.float32),
        "reject_boundary": np.zeros((original_face_count,), dtype=np.float32),
        "reject_density": np.zeros((original_face_count,), dtype=np.float32),
        "reject_topology": np.zeros((original_face_count,), dtype=np.float32),
        "reject_support": np.zeros((original_face_count,), dtype=np.float32),
        "reject_plane": np.zeros((original_face_count,), dtype=np.float32),
        "reject_normal": np.zeros((original_face_count,), dtype=np.float32),
        "reject_uncertainty": np.zeros((original_face_count,), dtype=np.float32),
        "reject_quality": np.zeros((original_face_count,), dtype=np.float32),
        "reject_batch_conflict": np.zeros((original_face_count,), dtype=np.float32),
        "reject_projection": np.zeros((original_face_count,), dtype=np.float32),
    }

    if bool(config.pre_project_before_collapse) and int(config.planar_project_iterations) > 0:
        pre_fields = _build_pass_fields(
            vertices=vertices,
            faces=faces,
            face_source=face_source,
            policy=policy,
            proxies=proxies,
            proxy_entries=proxy_entries,
            config=config,
            pass_index=-1,
            original_face_count=original_face_count,
        )
        progress("[weighted-remesh] Pre-project: snapping confident planar vertices before collapse.")
        vertices, pre_project_summary = _apply_relax_project(
            vertices=vertices,
            faces=faces,
            face_source=face_source,
            fields=pre_fields,
            config=replace(config, detail_relax_iterations=0),
            source_accum=source_accum,
        )

    for pass_index in range(max(int(config.max_passes), 0)):
        fields = _build_pass_fields(
            vertices=vertices,
            faces=faces,
            face_source=face_source,
            policy=policy,
            proxies=proxies,
            proxy_entries=proxy_entries,
            config=config,
            pass_index=pass_index,
            original_face_count=original_face_count,
        )
        if pass_index == 0 and face_source.shape[0] == original_face_count:
            source_face_final_target = np.asarray(fields["face_final_target"], dtype=np.float32).copy()
        candidates = _candidate_ids(fields["edge_collapse_pressure"], int(config.max_collapses_per_pass))
        candidates = candidates[fields["edge_collapse_pressure"][candidates] >= float(config.collapse_threshold)]
        progress(
            f"[weighted-remesh] Pass {pass_index + 1}: edges={fields['edge_vertices'].shape[0]:,}; "
            f"collapse candidates={candidates.size:,}"
        )
        accepted, reject_code, candidate_positions, projection_distance, accepted_count = _evaluate_collapse_candidates(
            pass_index=pass_index,
            candidate_ids=candidates,
            vertices=vertices,
            faces=faces,
            face_source=face_source,
            fields=fields,
            config=config,
            source_accum=source_accum,
            rows=rows,
        )
        reject_summary = _reject_counts(reject_code, candidates)
        vertices_next, faces_next, source_next, cleanup = _apply_batched_collapses(
            vertices,
            faces,
            face_source,
            fields["edge_vertices"],
            accepted,
            candidate_positions,
        )
        flip_fields = _build_pass_fields(
            vertices=vertices_next,
            faces=faces_next,
            face_source=source_next,
            policy=policy,
            proxies=proxies,
            proxy_entries=proxy_entries,
            config=config,
            pass_index=pass_index + 1,
            original_face_count=original_face_count,
        )
        faces_flipped, flip_count, flip_summary = _apply_flip_pass(
            vertices=vertices_next,
            faces=faces_next,
            face_source=source_next,
            fields=flip_fields,
            config=config,
            pass_index=pass_index,
            source_accum=source_accum,
            rows=rows,
        )
        pass_summaries.append(
            {
                "pass": int(pass_index),
                "candidateEdges": int(candidates.size),
                "acceptedCollapses": int(accepted_count),
                "acceptedFlips": int(flip_count),
                "facesBefore": int(faces.shape[0]),
                "facesAfter": int(faces_flipped.shape[0]),
                "verticesBefore": int(vertices.shape[0]),
                "verticesAfter": int(vertices_next.shape[0]),
                "cleanup": cleanup,
                "flip": flip_summary,
                "rejectCounts": reject_summary,
                "stats": {
                    "collapsePressure": _summarize(fields["edge_collapse_pressure"], fields["edge_collapse_pressure"] > 0.0),
                    "planarTransitionPressure": _summarize(fields["edge_planar_transition_pressure"], fields["edge_planar_transition_pressure"] > 0.0),
                    "featureCollapsePressure": _summarize(fields["edge_feature_collapse_pressure"], fields["edge_feature_collapse_pressure"] > 0.0),
                    "detailSimplifyPressure": _summarize(fields["edge_detail_simplify_pressure"], fields["edge_detail_simplify_pressure"] > 0.0),
                    "acceptedProjectionDistance": _summarize(projection_distance, accepted > 0),
                },
            }
        )
        total_accepted += int(accepted_count)
        total_flips += int(flip_count)
        vertices, faces, face_source = vertices_next, faces_flipped, source_next
        if accepted_count == 0 and flip_count == 0:
            break

    final_fields = _build_pass_fields(
        vertices=vertices,
        faces=faces,
        face_source=face_source,
        policy=policy,
        proxies=proxies,
        proxy_entries=proxy_entries,
        config=config,
        pass_index=max(int(config.max_passes), 0) + 1,
        original_face_count=original_face_count,
    )
    vertices, relax_project_summary = _apply_relax_project(
        vertices=vertices,
        faces=faces,
        face_source=face_source,
        fields=final_fields,
        config=config,
        source_accum=source_accum,
    )

    survived = np.zeros((original_face_count,), dtype=np.float32)
    if face_source.size:
        valid_source = face_source[(face_source >= 0) & (face_source < original_face_count)]
        survived[np.unique(valid_source)] = 1.0
    source_accum["source_face_survived"] = survived
    source_accum["source_face_removed_by_collapse"] = (1.0 - survived).astype(np.float32)

    diagnostics: dict[str, np.ndarray] = {
        "final_face_source_id": face_source.astype(np.int64),
        "source_face_collapse_pressure": source_accum["source_face_collapse_pressure"].astype(np.float32),
        "source_face_collapse_accept": source_accum["source_face_collapse_accept"].astype(np.float32),
        "source_face_detail_simplify_pressure": source_accum["source_face_detail_simplify_pressure"].astype(np.float32),
        "source_face_planar_transition_pressure": source_accum["source_face_planar_transition_pressure"].astype(np.float32),
        "source_face_feature_collapse_pressure": source_accum["source_face_feature_collapse_pressure"].astype(np.float32),
        "source_face_flip_pressure": source_accum["source_face_flip_pressure"].astype(np.float32),
        "source_face_flip_accept": source_accum["source_face_flip_accept"].astype(np.float32),
        "source_face_planar_projected": source_accum["source_face_planar_projected"].astype(np.float32),
        "source_face_planar_projection_distance": source_accum["source_face_planar_projection_distance"].astype(np.float32),
        "source_face_detail_relaxed": source_accum["source_face_detail_relaxed"].astype(np.float32),
        "source_face_detail_relax_distance": source_accum["source_face_detail_relax_distance"].astype(np.float32),
        "source_face_removed_by_collapse": source_accum["source_face_removed_by_collapse"].astype(np.float32),
        "source_face_survived": source_accum["source_face_survived"].astype(np.float32),
        "source_face_reject_boundary": source_accum["reject_boundary"].astype(np.float32),
        "source_face_reject_density": source_accum["reject_density"].astype(np.float32),
        "source_face_reject_topology": source_accum["reject_topology"].astype(np.float32),
        "source_face_reject_support": source_accum["reject_support"].astype(np.float32),
        "source_face_reject_plane": source_accum["reject_plane"].astype(np.float32),
        "source_face_reject_normal": source_accum["reject_normal"].astype(np.float32),
        "source_face_reject_uncertainty": source_accum["reject_uncertainty"].astype(np.float32),
        "source_face_reject_quality": source_accum["reject_quality"].astype(np.float32),
        "source_face_reject_batch_conflict": source_accum["reject_batch_conflict"].astype(np.float32),
        "source_face_reject_projection": source_accum["reject_projection"].astype(np.float32),
        "source_face_weight_planar": _clip01(policy["face_weight_planar"]),
        "source_face_weight_detail": _clip01(policy["face_weight_detail"]),
        "source_face_weight_boundary": _clip01(policy["face_weight_boundary"]),
        "source_face_weight_uncertain": _clip01(policy["face_weight_uncertain"]),
        "source_face_support": _clip01(policy["face_support"]),
    }
    diagnostics["source_face_final_target_length"] = source_face_final_target.astype(np.float32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)
    np.savez_compressed(diagnostics_npz, **diagnostics)
    _write_jsonl(operations_jsonl, rows)
    debug_meshes = _write_debug_meshes(original_vertices, original_faces, diagnostics, output_dir) if config.write_debug_meshes else {}

    output_stats = _mesh_stats(vertices, faces)
    summary = {
        "stageName": "omega_weighted_local_remesh",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "mesh_path": str(mesh_path),
            "policy_npz": str(policy_npz),
            "proxies_npz": str(proxies_npz),
            "proxies_json": None if config.proxies_json is None else str(config.proxies_json),
            "output_dir": str(output_dir),
        },
        "inputs": {
            "meshPath": str(mesh_path),
            "policyNpz": str(policy_npz),
            "proxiesNpz": str(proxies_npz),
        },
        "counts": {
            "acceptedCollapses": int(total_accepted),
            "acceptedFlips": int(total_flips),
            "planarProjectedSourceFaces": int(np.count_nonzero(diagnostics["source_face_planar_projected"] > 0.0)),
            "detailRelaxedSourceFaces": int(np.count_nonzero(diagnostics["source_face_detail_relaxed"] > 0.0)),
            "inputFaces": int(original_face_count),
            "outputFaces": int(faces.shape[0]),
            "removedSourceFaces": int(np.count_nonzero(diagnostics["source_face_removed_by_collapse"] > 0.0)),
            "inputVertices": int(original_vertices.shape[0]),
            "outputVertices": int(vertices.shape[0]),
        },
        "inputMesh": input_stats,
        "outputMesh": output_stats,
        "preProject": pre_project_summary,
        "passes": pass_summaries,
        "relaxProject": relax_project_summary,
        "stats": {
            "source_face_collapse_pressure": _summarize(diagnostics["source_face_collapse_pressure"], diagnostics["source_face_collapse_pressure"] > 0.0),
            "source_face_collapse_accept": _summarize(diagnostics["source_face_collapse_accept"], diagnostics["source_face_collapse_accept"] > 0.0),
            "source_face_detail_simplify_pressure": _summarize(diagnostics["source_face_detail_simplify_pressure"], diagnostics["source_face_detail_simplify_pressure"] > 0.0),
            "source_face_planar_transition_pressure": _summarize(diagnostics["source_face_planar_transition_pressure"], diagnostics["source_face_planar_transition_pressure"] > 0.0),
            "source_face_feature_collapse_pressure": _summarize(diagnostics["source_face_feature_collapse_pressure"], diagnostics["source_face_feature_collapse_pressure"] > 0.0),
            "source_face_flip_accept": _summarize(diagnostics["source_face_flip_accept"], diagnostics["source_face_flip_accept"] > 0.0),
            "source_face_planar_projected": _summarize(diagnostics["source_face_planar_projected"], diagnostics["source_face_planar_projected"] > 0.0),
            "source_face_planar_projection_distance": _summarize(diagnostics["source_face_planar_projection_distance"], diagnostics["source_face_planar_projection_distance"] > 0.0),
            "source_face_detail_relaxed": _summarize(diagnostics["source_face_detail_relaxed"], diagnostics["source_face_detail_relaxed"] > 0.0),
            "source_face_detail_relax_distance": _summarize(diagnostics["source_face_detail_relax_distance"], diagnostics["source_face_detail_relax_distance"] > 0.0),
            "source_face_removed_by_collapse": _summarize(diagnostics["source_face_removed_by_collapse"], diagnostics["source_face_removed_by_collapse"] > 0.0),
        },
        "rejectCodes": REJECT_CODE_TO_NAME,
        "outputs": {
            "outputMesh": str(output_mesh),
            "diagnosticsNpz": str(diagnostics_npz),
            "operationsJsonl": str(operations_jsonl),
            "summaryJson": str(summary_json),
            "reportMd": str(report_md),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
        "notes": [
            "This mutating Phase 4 pass performs collapse, flip, and bounded relax/project; it does not split/densify.",
            "Planar projection is stronger than collapse-only mode and is bounded by face_position_tolerance.",
            "Proxy-boundary feature edges may collapse along the two-plane intersection; they still cannot collapse across mesh boundaries.",
            "Detail relaxation is tangential smoothing with local quality gates, so detail regions can be beautified without simplification.",
            "High-detail regions may still simplify when local edges are shorter than the adaptive detail target from normal evidence and source-mesh scale.",
            "edge_weight_boundary and mesh boundary edges are hard no-cross constraints for collapse.",
            "edge_weight_uncertain suppresses collapse pressure and rejects highly uncertain edges.",
            "Diagnostics are stored on the original source-face indexing so existing Stage 2 view buffers can visualize the edits.",
        ],
    }
    _write_markdown(report_md, _build_report(summary))
    _write_json(summary_json, summary)
    progress(f"[weighted-remesh] Wrote {output_mesh}")
    progress(f"[weighted-remesh] Wrote {diagnostics_npz}")
    progress(f"[weighted-remesh] Wrote {report_md}")
    return WeightedRemeshResult(
        output_mesh=output_mesh,
        diagnostics_npz=diagnostics_npz,
        operations_jsonl=operations_jsonl,
        summary_json=summary_json,
        report_md=report_md,
        debug_meshes=debug_meshes,
        input_face_count=original_face_count,
        output_face_count=int(faces.shape[0]),
        accepted_collapses=int(total_accepted),
    )


def _resolve_path(path: Path, base: Path) -> Path:
    raw = path.expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [raw.resolve(), (base / raw).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _optional_path(path: Path | None) -> Path | None:
    if path is None or str(path).strip() in {"", "."}:
        return None
    return path


def _existing_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{description} does not exist: {path}")
    if path.is_dir():
        raise SystemExit(
            f"{description} points to a directory, not a file: {path}\n"
            "If this came from an environment variable, re-export it or omit the flag to use the default."
        )
    return path


def _config_from_args(args: argparse.Namespace) -> WeightedRemeshConfig:
    model_dir = args.model_dir.expanduser().resolve()
    mesh_arg = _optional_path(args.mesh)
    policy_arg = _optional_path(args.policy_npz)
    proxies_arg = _optional_path(args.proxies_npz)
    proxies_json_arg = _optional_path(args.proxies_json)
    output_arg = _optional_path(args.output_dir)
    proxies_json = None
    if proxies_json_arg is not None:
        proxies_json = _existing_file(_resolve_path(proxies_json_arg, model_dir), "Proxies json")
    elif (model_dir / "remesh" / "local" / "proxies.json").exists():
        proxies_json = model_dir / "remesh" / "local" / "proxies.json"
    return WeightedRemeshConfig(
        mesh_path=_existing_file(
            _resolve_path(mesh_arg, model_dir) if mesh_arg is not None else model_dir / "remesh" / "local" / "preclean_mesh.ply",
            "Mesh",
        ),
        policy_npz=_existing_file(
            _resolve_path(policy_arg, model_dir) if policy_arg is not None else model_dir / "remesh" / "local" / "policy.npz",
            "Policy npz",
        ),
        proxies_npz=_existing_file(
            _resolve_path(proxies_arg, model_dir) if proxies_arg is not None else model_dir / "remesh" / "local" / "proxies.npz",
            "Proxies npz",
        ),
        proxies_json=proxies_json,
        output_dir=output_arg.expanduser().resolve() if output_arg is not None else model_dir / "remesh" / "local",
        output_name=str(args.output_name),
        diagnostics_name=str(args.diagnostics_name),
        operations_name=str(args.operations_name),
        max_passes=int(args.max_passes),
        max_collapses_per_pass=int(args.max_collapses_per_pass),
        max_flips_per_pass=int(args.max_flips_per_pass),
        max_jsonl_rows=int(args.max_jsonl_rows),
        collapse_threshold=float(args.collapse_threshold),
        support_threshold=float(args.support_threshold),
        boundary_threshold=float(args.boundary_threshold),
        uncertainty_threshold=float(args.uncertainty_threshold),
        alpha_collapse=float(args.alpha_collapse),
        alpha_planar_collapse=float(args.alpha_planar_collapse),
        alpha_planar_transition_collapse=float(args.alpha_planar_transition_collapse),
        alpha_feature_edge_collapse=float(args.alpha_feature_edge_collapse),
        alpha_detail_collapse=float(args.alpha_detail_collapse),
        detail_mesh_target_factor=float(args.detail_mesh_target_factor),
        detail_simplify_min_weight=float(args.detail_simplify_min_weight),
        q_min_hard=float(args.q_min_hard),
        q_min_soft=float(args.q_min_soft),
        max_normal_error=float(args.max_normal_error),
        max_quality_error=float(args.max_quality_error),
        max_projection_distance_factor=float(args.max_projection_distance_factor),
        plane_residual_factor=float(args.plane_residual_factor),
        pre_project_before_collapse=bool(args.pre_project_before_collapse),
        feature_edge_collapse=bool(args.feature_edge_collapse),
        feature_edge_min_planar=float(args.feature_edge_min_planar),
        feature_edge_max_uncertain=float(args.feature_edge_max_uncertain),
        planar_boundary_softening=float(args.planar_boundary_softening),
        planar_transition_min_weight=float(args.planar_transition_min_weight),
        planar_projection_strength=float(args.planar_projection_strength),
        planar_project_iterations=int(args.planar_project_iterations),
        planar_project_min_weight=float(args.planar_project_min_weight),
        planar_project_max_distance_factor=float(args.planar_project_max_distance_factor),
        detail_relax_iterations=int(args.detail_relax_iterations),
        detail_relax_strength=float(args.detail_relax_strength),
        detail_relax_min_weight=float(args.detail_relax_min_weight),
        detail_relax_max_distance_factor=float(args.detail_relax_max_distance_factor),
        flip_min_quality_improvement=float(args.flip_min_quality_improvement),
        planar_area_length_factor=float(args.planar_area_length_factor),
        planar_boundary_length_factor=float(args.planar_boundary_length_factor),
        scene_max_length_factor=float(args.scene_max_length_factor),
        proxy_min_length_factor=float(args.proxy_min_length_factor),
        proxy_confidence_floor=float(args.proxy_confidence_floor),
        lambda_proxy=float(args.lambda_proxy),
        write_debug_meshes=not bool(args.no_debug_meshes),
        overwrite=bool(args.overwrite),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 4B weighted local remesh from Phase 3 policy weights.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--proxies-npz", type=Path, default=None)
    parser.add_argument("--proxies-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default="mesh_weighted_remesh.ply")
    parser.add_argument("--diagnostics-name", default="weighted_remesh_diagnostics")
    parser.add_argument("--operations-name", default="weighted_remesh_operations")
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--max-collapses-per-pass", type=int, default=100000)
    parser.add_argument("--max-flips-per-pass", type=int, default=50000)
    parser.add_argument("--max-jsonl-rows", type=int, default=50000)
    parser.add_argument("--collapse-threshold", type=float, default=0.01)
    parser.add_argument("--support-threshold", type=float, default=0.08)
    parser.add_argument("--boundary-threshold", type=float, default=0.50)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.10)
    parser.add_argument("--alpha-collapse", type=float, default=0.75)
    parser.add_argument("--alpha-planar-collapse", type=float, default=5.00)
    parser.add_argument("--alpha-planar-transition-collapse", type=float, default=3.00)
    parser.add_argument("--alpha-feature-edge-collapse", type=float, default=3.00)
    parser.add_argument("--alpha-detail-collapse", type=float, default=1.75)
    parser.add_argument("--detail-mesh-target-factor", type=float, default=2.25)
    parser.add_argument("--detail-simplify-min-weight", type=float, default=0.25)
    parser.add_argument("--q-min-hard", type=float, default=0.03)
    parser.add_argument("--q-min-soft", type=float, default=0.12)
    parser.add_argument("--max-normal-error", type=float, default=2.0)
    parser.add_argument("--max-quality-error", type=float, default=1.5)
    parser.add_argument("--max-projection-distance-factor", type=float, default=8.0)
    parser.add_argument("--plane-residual-factor", type=float, default=2.0)
    parser.add_argument("--pre-project-before-collapse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-edge-collapse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-edge-min-planar", type=float, default=0.35)
    parser.add_argument("--feature-edge-max-uncertain", type=float, default=1.10)
    parser.add_argument("--planar-boundary-softening", type=float, default=0.85)
    parser.add_argument("--planar-transition-min-weight", type=float, default=0.35)
    parser.add_argument("--planar-projection-strength", type=float, default=1.50)
    parser.add_argument("--planar-project-iterations", type=int, default=3)
    parser.add_argument("--planar-project-min-weight", type=float, default=0.25)
    parser.add_argument("--planar-project-max-distance-factor", type=float, default=8.0)
    parser.add_argument("--detail-relax-iterations", type=int, default=4)
    parser.add_argument("--detail-relax-strength", type=float, default=0.45)
    parser.add_argument("--detail-relax-min-weight", type=float, default=0.18)
    parser.add_argument("--detail-relax-max-distance-factor", type=float, default=1.25)
    parser.add_argument("--flip-min-quality-improvement", type=float, default=0.002)
    parser.add_argument("--planar-area-length-factor", type=float, default=0.20)
    parser.add_argument("--planar-boundary-length-factor", type=float, default=6.0)
    parser.add_argument("--scene-max-length-factor", type=float, default=12.0)
    parser.add_argument("--proxy-min-length-factor", type=float, default=1.5)
    parser.add_argument("--proxy-confidence-floor", type=float, default=0.05)
    parser.add_argument("--lambda-proxy", type=float, default=1.0)
    parser.add_argument("--no-debug-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compute_weighted_remesh(_config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
