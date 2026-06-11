"""Simplification-first planar patch remeshing.

This pass is intentionally simpler than the weighted local remesher:

- high planar-weight faces are grouped into connected planar patches;
- each patch is projected to one fitted plane;
- patch interiors are simplified aggressively;
- patch boundaries and adjacent planes are hard stops;
- high detail/uncertainty outside planar patches is preserved.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

from omega_local.remesh.local_ops import (
    build_vertex_faces,
    face_geometry,
    load_mesh_arrays,
    simulate_collapse_gates,
    triangle_quality_from_vertices,
    write_face_scalar_mesh,
)
from omega_local.remesh.policy import build_mesh_topology
from omega_local.remesh.weighted_local_remesh import (
    _apply_batched_collapses,
    _collapse_link_condition_ok,
    _mesh_stats,
)


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class PlanarPatchRemeshConfig:
    mesh_path: Path
    policy_npz: Path
    proxies_npz: Path
    output_dir: Path
    output_name: str = "mesh_planar_patch_remesh.ply"
    diagnostics_name: str = "planar_patch_remesh_diagnostics"
    operations_name: str = "planar_patch_remesh_operations"
    planar_threshold: float = 0.30
    feature_threshold: float = 0.45
    boundary_stop_threshold: float = 0.35
    min_patch_faces: int = 12
    patch_normal_degrees: float = 8.0
    patch_distance_factor: float = 2.5
    patch_trim_distance_factor: float = 2.5
    planar_target_factor: float = 7.0
    max_passes: int = 12
    max_collapses_per_pass: int = 150000
    collapse_threshold: float = 0.02
    q_min_hard: float = 0.005
    q_min_soft: float = 0.03
    max_quality_error: float = 2.0
    project_regularization: float = 1.0e-6
    max_jsonl_rows: int = 50000
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class PlanarPatchRemeshResult:
    output_mesh: Path
    diagnostics_npz: Path
    operations_jsonl: Path
    summary_json: Path
    report_md: Path
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


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _summarize(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is not None:
        arr = arr[np.asarray(valid, dtype=bool).reshape(-1)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0, "min": 0.0, "median": 0.0, "mean": 0.0, "q90": 0.0, "q99": 0.0, "max": 0.0}
    qs = np.quantile(arr, [0.5, 0.9, 0.99])
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "median": float(qs[0]),
        "mean": float(np.mean(arr)),
        "q90": float(qs[1]),
        "q99": float(qs[2]),
        "max": float(np.max(arr)),
    }


def _fit_plane(points: np.ndarray, weights: np.ndarray, fallback_normal: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 1.0e-12)
    centroid = np.sum(points * weights[:, None], axis=0) / max(float(np.sum(weights)), 1.0e-12)
    centered = points - centroid
    cov = (centered * weights[:, None]).T @ centered / max(float(np.sum(weights)), 1.0e-12)
    _evals, evecs = np.linalg.eigh(cov)
    normal = np.asarray(evecs[:, 0], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        normal = np.asarray(fallback_normal if fallback_normal is not None else [0.0, 0.0, 1.0], dtype=np.float64)
        norm = max(float(np.linalg.norm(normal)), 1.0e-12)
    normal = normal / norm
    if fallback_normal is not None and float(np.dot(normal, fallback_normal)) < 0.0:
        normal = -normal
    return normal, -float(np.dot(normal, centroid))


def _plane_residual(normal: np.ndarray, offset: float, points: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(points, dtype=np.float64) @ np.asarray(normal, dtype=np.float64) + float(offset))


def _edge_array(data: dict[str, np.ndarray], key: str, edge_count: int) -> np.ndarray:
    value = data.get(key)
    if value is None or np.asarray(value).shape[0] != edge_count:
        return np.zeros((edge_count,), dtype=np.float32)
    return _clip01(np.asarray(value, dtype=np.float32))


def _shared_edge_id(face_edges: np.ndarray, face_a: int, face_b: int) -> int:
    edges_a = set(int(v) for v in np.asarray(face_edges[int(face_a)], dtype=np.int64) if int(v) >= 0)
    for edge_id_raw in np.asarray(face_edges[int(face_b)], dtype=np.int64):
        edge_id = int(edge_id_raw)
        if edge_id in edges_a:
            return edge_id
    return -1


def _reference_plane_for_face(
    *,
    face_id: int,
    face_proxy_id: np.ndarray,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    proxy_normals: np.ndarray,
    proxy_offsets: np.ndarray,
) -> tuple[np.ndarray, float]:
    proxy_id = int(face_proxy_id[int(face_id)])
    if proxy_id >= 0 and proxy_id < proxy_normals.shape[0]:
        normal = np.asarray(proxy_normals[proxy_id], dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-12:
            return normal / norm, float(proxy_offsets[proxy_id])
    normal = np.asarray(face_normals[int(face_id)], dtype=np.float64)
    norm = max(float(np.linalg.norm(normal)), 1.0e-12)
    normal = normal / norm
    return normal, -float(np.dot(normal, face_centers[int(face_id)]))


def _face_compatible_with_plane(
    *,
    face_id: int,
    reference_normal: np.ndarray,
    reference_offset: float,
    face_centers: np.ndarray,
    face_normals: np.ndarray,
    face_eps: np.ndarray,
    cos_limit: float,
    distance_factor: float,
) -> bool:
    normal = np.asarray(face_normals[int(face_id)], dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1.0e-12:
        return False
    normal = normal / norm
    if abs(float(np.dot(reference_normal, normal))) < float(cos_limit):
        return False
    residual = abs(float(np.dot(reference_normal, face_centers[int(face_id)]) + float(reference_offset)))
    tolerance = float(distance_factor) * max(float(face_eps[int(face_id)]), 1.0e-6)
    return bool(residual <= tolerance)


def _edge_stop_values(
    *,
    topology: Any,
    policy: dict[str, np.ndarray],
    proxies: dict[str, np.ndarray],
) -> np.ndarray:
    edge_count = int(topology.edge_faces.shape[0])
    edge_stop = np.maximum(
        _edge_array(policy, "edge_weight_boundary", edge_count),
        _edge_array(policy, "edge_boundary_score", edge_count),
    )
    edge_stop = np.maximum(edge_stop, _edge_array(proxies, "edge_proxy_boundary", edge_count))
    edge_stop = np.maximum(edge_stop, np.asarray(topology.boundary_edges, dtype=np.float32))
    return edge_stop.astype(np.float32)


def _patch_boundary_stop_mask(
    *,
    faces: np.ndarray,
    source_patch_id: np.ndarray,
    policy: dict[str, np.ndarray],
    proxies: dict[str, np.ndarray],
    threshold: float,
) -> np.ndarray:
    topology = build_mesh_topology(faces)
    edge_stop = _edge_stop_values(topology=topology, policy=policy, proxies=proxies)
    out = np.zeros((int(faces.shape[0]),), dtype=np.float32)
    for edge_id, (fa_raw, fb_raw) in enumerate(topology.edge_faces):
        fa = int(fa_raw)
        fb = int(fb_raw)
        if fa < 0:
            continue
        pa = int(source_patch_id[fa])
        pb = int(source_patch_id[fb]) if fb >= 0 else -1
        hard_stop = float(edge_stop[int(edge_id)]) >= float(threshold)
        if pa >= 0 and (hard_stop or pb != pa):
            out[fa] = 1.0
        if fb >= 0 and pb >= 0 and (hard_stop or pa != pb):
            out[fb] = 1.0
    return out


def _proxy_plane_arrays(proxies: dict[str, np.ndarray], proxy_count: int) -> tuple[np.ndarray, np.ndarray]:
    # Plane values are not stored in proxies.npz; this fallback lets component
    # grouping use local face normals when proxies.json is unavailable.
    return np.zeros((int(proxy_count), 3), dtype=np.float64), np.zeros((int(proxy_count),), dtype=np.float64)


def _load_proxy_json_planes(path: Path | None, proxy_count: int) -> tuple[np.ndarray, np.ndarray]:
    normals = np.zeros((int(proxy_count), 3), dtype=np.float64)
    offsets = np.zeros((int(proxy_count),), dtype=np.float64)
    if path is None or not path.exists() or path.is_dir():
        return normals, offsets
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("proxies", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        iterable = entries.values()
    else:
        iterable = entries
    for entry in iterable:
        proxy_id = int(entry.get("id", -1))
        if proxy_id < 0 or proxy_id >= proxy_count:
            continue
        normal = np.asarray(entry.get("normal", [0.0, 0.0, 0.0]), dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-12:
            normals[proxy_id] = normal / norm
            offsets[proxy_id] = float(entry.get("d", 0.0))
    return normals, offsets


def _connected_planar_patches(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    policy: dict[str, np.ndarray],
    proxies: dict[str, np.ndarray],
    proxy_normals: np.ndarray,
    proxy_offsets: np.ndarray,
    config: PlanarPatchRemeshConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    topology = build_mesh_topology(faces)
    centers, normals, areas, mean_edge = face_geometry(vertices, faces)
    face_count = int(faces.shape[0])
    planar = _clip01(policy["face_weight_planar"])
    detail = _clip01(policy.get("face_weight_detail", np.zeros(face_count, dtype=np.float32)))
    uncertain = _clip01(policy.get("face_weight_uncertain", np.zeros(face_count, dtype=np.float32)))
    feature = np.maximum(detail, uncertain).astype(np.float32)
    face_eps = np.asarray(policy.get("face_position_tolerance", np.maximum(0.25 * mean_edge, 1.0e-6)), dtype=np.float32)
    face_proxy_id = np.asarray(proxies.get("face_proxy_id", np.full(face_count, -1, dtype=np.int32)), dtype=np.int32)
    candidate = planar >= float(config.planar_threshold)
    edge_stop = _edge_stop_values(topology=topology, policy=policy, proxies=proxies)
    cos_limit = math.cos(math.radians(float(config.patch_normal_degrees)))
    visited = np.zeros((face_count,), dtype=bool)
    patch_id = np.full((face_count,), -1, dtype=np.int32)
    patch_normals: list[np.ndarray] = []
    patch_offsets: list[float] = []
    patch_target: list[float] = []
    patch_rows: list[dict[str, Any]] = []

    for seed in np.flatnonzero(candidate):
        seed = int(seed)
        if visited[seed]:
            continue
        ref_normal, ref_offset = _reference_plane_for_face(
            face_id=seed,
            face_proxy_id=face_proxy_id,
            face_centers=centers,
            face_normals=normals,
            proxy_normals=proxy_normals,
            proxy_offsets=proxy_offsets,
        )
        stack = [seed]
        visited[seed] = True
        component: list[int] = []
        while stack:
            face_id = stack.pop()
            component.append(face_id)
            for nbr_raw in topology.face_neighbors[face_id]:
                nbr = int(nbr_raw)
                if visited[nbr] or not bool(candidate[nbr]):
                    continue
                shared_edge = _shared_edge_id(topology.face_edges, face_id, nbr)
                if shared_edge < 0 or float(edge_stop[shared_edge]) >= float(config.boundary_stop_threshold):
                    continue
                if not _face_compatible_with_plane(
                    face_id=nbr,
                    reference_normal=ref_normal,
                    reference_offset=ref_offset,
                    face_centers=centers,
                    face_normals=normals,
                    face_eps=face_eps,
                    cos_limit=cos_limit,
                    distance_factor=float(config.patch_distance_factor),
                ):
                    continue
                visited[nbr] = True
                stack.append(nbr)

        if len(component) < int(config.min_patch_faces):
            continue
        comp = np.asarray(component, dtype=np.int64)
        fallback = np.sum(normals[comp] * np.maximum(areas[comp], 1.0e-12)[:, None], axis=0)
        normal, offset = _fit_plane(centers[comp], np.maximum(areas[comp] * planar[comp], 1.0e-8), fallback)
        keep = np.asarray(
            [
                _face_compatible_with_plane(
                    face_id=int(face_id),
                    reference_normal=normal,
                    reference_offset=offset,
                    face_centers=centers,
                    face_normals=normals,
                    face_eps=face_eps,
                    cos_limit=cos_limit,
                    distance_factor=float(config.patch_trim_distance_factor),
                )
                for face_id in comp
            ],
            dtype=bool,
        )
        comp = comp[keep]
        if comp.size < int(config.min_patch_faces):
            continue
        fallback = np.sum(normals[comp] * np.maximum(areas[comp], 1.0e-12)[:, None], axis=0)
        normal, offset = _fit_plane(centers[comp], np.maximum(areas[comp] * planar[comp], 1.0e-8), fallback)
        pid = len(patch_normals)
        patch_id[comp] = int(pid)
        patch_normals.append(normal)
        patch_offsets.append(offset)
        comp_edges = topology.face_edges[comp].reshape(-1)
        comp_edges = comp_edges[comp_edges >= 0]
        lengths = np.linalg.norm(vertices[topology.edge_vertices[comp_edges, 1]] - vertices[topology.edge_vertices[comp_edges, 0]], axis=1)
        base_length = float(np.median(lengths)) if lengths.size else float(np.median(mean_edge[comp]))
        target = max(base_length * float(config.planar_target_factor), 1.0e-6)
        patch_target.append(target)
        residual = _plane_residual(normal, offset, centers[comp])
        patch_rows.append(
            {
                "patchId": int(pid),
                "faceCount": int(comp.size),
                "area": float(np.sum(areas[comp])),
                "targetLength": float(target),
                "planarMean": float(np.mean(planar[comp])),
                "featureMean": float(np.mean(feature[comp])),
                "residual": _summarize(residual),
            }
        )

    if patch_normals:
        normals_out = np.stack(patch_normals, axis=0).astype(np.float64)
        offsets_out = np.asarray(patch_offsets, dtype=np.float64)
        target_out = np.asarray(patch_target, dtype=np.float32)
    else:
        normals_out = np.zeros((0, 3), dtype=np.float64)
        offsets_out = np.zeros((0,), dtype=np.float64)
        target_out = np.zeros((0,), dtype=np.float32)
    return patch_id, normals_out, offsets_out, target_out, patch_rows


def _project_vertices_to_patches(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_patch_id: np.ndarray,
    patch_normals: np.ndarray,
    patch_offsets: np.ndarray,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    memberships: list[set[int]] = [set() for _ in range(int(vertices.shape[0]))]
    touches_unpatched = np.zeros((int(vertices.shape[0]),), dtype=bool)
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        patch_id = int(face_patch_id[int(face_id)])
        if patch_id < 0:
            for vertex_id in face:
                touches_unpatched[int(vertex_id)] = True
            continue
        for vertex_id in face:
            memberships[int(vertex_id)].add(patch_id)

    out = vertices.copy()
    moved = np.zeros((vertices.shape[0],), dtype=np.float32)
    for vertex_id, patches in enumerate(memberships):
        if bool(touches_unpatched[vertex_id]):
            continue
        valid = [pid for pid in patches if 0 <= pid < patch_normals.shape[0]]
        if not valid:
            continue
        normals = patch_normals[valid]
        offsets = patch_offsets[valid]
        if len(valid) == 1:
            normal = normals[0]
            candidate = out[vertex_id] - (float(np.dot(normal, out[vertex_id])) + float(offsets[0])) * normal
        else:
            reg = float(max(regularization, 1.0e-12))
            lhs = normals.T @ normals + reg * np.eye(3)
            rhs = normals.T @ (-offsets) + reg * out[vertex_id]
            candidate = np.linalg.solve(lhs, rhs)
        moved[vertex_id] = np.float32(np.linalg.norm(candidate - out[vertex_id]))
        out[vertex_id] = candidate
    return out, moved


def _vertex_patch_memberships(faces: np.ndarray, face_patch_id: np.ndarray, vertex_count: int) -> tuple[list[set[int]], np.ndarray]:
    memberships: list[set[int]] = [set() for _ in range(int(vertex_count))]
    touches_unpatched = np.zeros((int(vertex_count),), dtype=bool)
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        patch_id = int(face_patch_id[int(face_id)])
        if patch_id < 0:
            for vertex_id in face:
                touches_unpatched[int(vertex_id)] = True
            continue
        for vertex_id in face:
            memberships[int(vertex_id)].add(patch_id)
    return memberships, touches_unpatched


def _evaluate_patch_collapses(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_source: np.ndarray,
    source_patch_id: np.ndarray,
    patch_normals: np.ndarray,
    patch_offsets: np.ndarray,
    patch_target: np.ndarray,
    config: PlanarPatchRemeshConfig,
    source_accum: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    pass_index: int,
) -> tuple[np.ndarray, np.ndarray, int, dict[str, int]]:
    topology = build_mesh_topology(faces)
    vertex_faces = build_vertex_faces(faces, int(vertices.shape[0]))
    _centers, face_normals, _areas, _mean_edge = face_geometry(vertices, faces)
    edge_vertices = topology.edge_vertices
    edge_faces = topology.edge_faces
    edge_count = int(edge_vertices.shape[0])
    edge_length = np.linalg.norm(vertices[edge_vertices[:, 1]] - vertices[edge_vertices[:, 0]], axis=1)
    current_patch = np.full((faces.shape[0],), -1, dtype=np.int32)
    valid_source = (face_source >= 0) & (face_source < source_patch_id.shape[0])
    current_patch[valid_source] = source_patch_id[face_source[valid_source]]
    vertex_patches, vertex_touches_unpatched = _vertex_patch_memberships(faces, current_patch, int(vertices.shape[0]))

    pressure = np.zeros((edge_count,), dtype=np.float32)
    for edge_id, (fa_raw, fb_raw) in enumerate(edge_faces):
        fa = int(fa_raw)
        fb = int(fb_raw)
        if fa < 0 or fb < 0:
            continue
        pa = int(current_patch[fa])
        pb = int(current_patch[fb])
        if pa < 0 or pb < 0:
            continue
        if pa != pb:
            continue
        target = float(patch_target[pa])
        value = (target - float(edge_length[edge_id])) / max(target, 1.0e-9)
        if value > 0.0:
            pressure[edge_id] = np.float32(value)
    candidates = np.flatnonzero(pressure >= float(config.collapse_threshold))
    if candidates.size > int(config.max_collapses_per_pass) > 0:
        local = np.argpartition(pressure[candidates], -int(config.max_collapses_per_pass))[-int(config.max_collapses_per_pass) :]
        candidates = candidates[local]
    candidates = candidates[np.argsort(-pressure[candidates], kind="mergesort")]

    accepted = np.zeros((edge_count,), dtype=np.uint8)
    candidate_positions = np.zeros((edge_count, 3), dtype=np.float64)
    touched = np.zeros((vertices.shape[0],), dtype=bool)
    counts = {"candidate": int(candidates.size), "accepted": 0, "reject_batch": 0, "reject_boundary": 0, "reject_topology": 0, "reject_quality": 0}
    max_rows = int(config.max_jsonl_rows)
    for rank, edge_id_raw in enumerate(candidates):
        edge_id = int(edge_id_raw)
        vi, vj = [int(v) for v in edge_vertices[edge_id]]
        fa, fb = [int(v) for v in edge_faces[edge_id]]
        pa = int(current_patch[fa])
        pb = int(current_patch[fb])
        reason = "accepted"
        midpoint = 0.5 * (vertices[vi] + vertices[vj])
        if bool(touched[vi]) or bool(touched[vj]):
            reason = "reject_batch"
            candidate = midpoint
        elif (
            bool(vertex_touches_unpatched[vi])
            or bool(vertex_touches_unpatched[vj])
            or vertex_patches[vi] != {pa}
            or vertex_patches[vj] != {pa}
        ):
            reason = "reject_boundary"
            candidate = midpoint
        elif not _collapse_link_condition_ok(faces=faces, vertex_faces=vertex_faces, edge_faces=edge_faces, edge_vertices=edge_vertices, edge_id=edge_id):
            reason = "reject_topology"
            candidate = midpoint
        else:
            normal = patch_normals[pa]
            candidate = midpoint - (float(np.dot(normal, midpoint)) + float(patch_offsets[pa])) * normal
        candidate_positions[edge_id] = candidate
        gates = simulate_collapse_gates(
            vertices=vertices,
            faces=faces,
            vertex_faces=vertex_faces,
            face_normals=face_normals,
            edge_vertices=edge_vertices,
            edge_faces=edge_faces,
            edge_id=edge_id,
            candidate_position=candidate,
            detail_weight=0.0,
            tau_a=math.radians(90.0),
            q_min_hard=float(config.q_min_hard),
            q_min_soft=float(config.q_min_soft),
        )
        if reason == "accepted" and (not bool(gates["topologyOk"])):
            reason = "reject_topology"
        if reason == "accepted" and (not bool(gates["qualityOk"]) or float(gates["qualityError"]) > float(config.max_quality_error)):
            reason = "reject_quality"

        if reason == "accepted":
            accepted[edge_id] = 1
            touched[vi] = True
            touched[vj] = True
            counts["accepted"] += 1
        else:
            counts[reason] = counts.get(reason, 0) + 1

        for face_id in (fa, fb):
            src = int(face_source[face_id])
            if src < 0:
                continue
            source_accum["source_face_collapse_pressure"][src] = max(float(source_accum["source_face_collapse_pressure"][src]), float(pressure[edge_id]))
            source_accum["source_face_patch_internal_pressure"][src] = max(float(source_accum["source_face_patch_internal_pressure"][src]), float(pressure[edge_id]))
            if reason == "accepted":
                source_accum["source_face_collapse_accept"][src] = 1.0
            elif reason == "reject_boundary":
                source_accum["source_face_reject_boundary"][src] = 1.0
            elif reason == "reject_quality":
                source_accum["source_face_reject_quality"][src] = 1.0
        if max_rows <= 0 or len(rows) < max_rows or reason == "accepted":
            rows.append(
                {
                    "operation": "patch_collapse",
                    "pass": int(pass_index),
                    "rank": int(rank),
                    "edgeId": int(edge_id),
                    "kind": "patch_interior",
                    "accepted": bool(reason == "accepted"),
                    "reason": reason,
                    "vertices": [vi, vj],
                    "faces": [fa, fb],
                    "sourceFaces": [int(face_source[fa]), int(face_source[fb])],
                    "patches": [pa, pb],
                    "signals": {"pressure": float(pressure[edge_id]), "length": float(edge_length[edge_id])},
                    "gates": {"quality": float(gates["qualityError"]), "minQuality": float(gates["minQuality"])},
                }
            )
    return accepted, candidate_positions, int(counts["accepted"]), counts


def _write_debug_meshes(vertices: np.ndarray, faces: np.ndarray, diagnostics: dict[str, np.ndarray], output_dir: Path) -> dict[str, Path]:
    fields = {
        "planar_patch_mask": "source_face_planar_patch_mask",
        "patch_boundary_stop": "source_face_patch_boundary_stop",
        "feature_preserve_weight": "source_face_feature_preserve_weight",
        "feature_preserve_mask": "source_face_feature_preserve_mask",
        "patch_internal_pressure": "source_face_patch_internal_pressure",
        "patch_feature_pressure": "source_face_patch_feature_pressure",
        "collapse_accept": "source_face_collapse_accept",
        "reject_boundary": "source_face_reject_boundary",
        "reject_quality": "source_face_reject_quality",
        "removed_by_collapse": "source_face_removed_by_collapse",
        "weight_planar": "source_face_weight_planar",
        "weight_detail": "source_face_weight_detail",
        "weight_uncertain": "source_face_weight_uncertain",
    }
    out: dict[str, Path] = {}
    debug_dir = output_dir / "debug_meshes" / "planar_patch_remesh"
    for name, key in fields.items():
        if key not in diagnostics:
            continue
        path = debug_dir / f"{name}.ply"
        values = np.asarray(diagnostics[key], dtype=np.float32)
        write_face_scalar_mesh(path, vertices, faces, values, np.isfinite(values), vmin=0.0, vmax=1.0)
        out[name] = path
    return out


def compute_planar_patch_remesh(config: PlanarPatchRemeshConfig, progress: ProgressFn | None = None) -> PlanarPatchRemeshResult:
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
        if path.exists() and not bool(config.overwrite):
            raise FileExistsError(f"Planar patch remesh output exists: {path}. Re-run with overwrite=True.")

    vertices, faces = load_mesh_arrays(mesh_path)
    original_vertices = vertices.copy()
    original_faces = faces.copy()
    policy = _load_npz(policy_npz)
    proxies = _load_npz(proxies_npz)
    face_count = int(faces.shape[0])
    proxy_count = int(np.max(proxies["face_proxy_id"])) + 1 if np.any(np.asarray(proxies["face_proxy_id"]) >= 0) else 0
    proxy_normals, proxy_offsets = _proxy_plane_arrays(proxies, proxy_count)
    json_path = proxies_npz.with_name("proxies.json")
    json_normals, json_offsets = _load_proxy_json_planes(json_path if json_path.exists() else None, proxy_count)
    valid_json = np.linalg.norm(json_normals, axis=1) > 0.0 if json_normals.size else np.zeros((0,), dtype=bool)
    if proxy_count:
        proxy_normals[valid_json] = json_normals[valid_json]
        proxy_offsets[valid_json] = json_offsets[valid_json]

    source_patch_id, patch_normals, patch_offsets, patch_target, patch_rows = _connected_planar_patches(
        vertices=vertices,
        faces=faces,
        policy=policy,
        proxies=proxies,
        proxy_normals=proxy_normals,
        proxy_offsets=proxy_offsets,
        config=config,
    )
    progress(f"[planar-patch-remesh] Planar patches: {len(patch_rows):,}; faces in patches={np.count_nonzero(source_patch_id >= 0):,}")
    feature_weight = np.maximum(_clip01(policy.get("face_weight_detail", np.zeros(face_count))), _clip01(policy.get("face_weight_uncertain", np.zeros(face_count)))).astype(np.float32)
    source_face_patch_mask = (source_patch_id >= 0).astype(np.float32)
    source_face_patch_boundary_stop = _patch_boundary_stop_mask(
        faces=faces,
        source_patch_id=source_patch_id,
        policy=policy,
        proxies=proxies,
        threshold=float(config.boundary_stop_threshold),
    )
    source_face_feature_preserve_mask = ((source_patch_id < 0) & (feature_weight >= float(config.feature_threshold))).astype(np.float32)
    source_accum: dict[str, np.ndarray] = {
        "source_face_collapse_pressure": np.zeros((face_count,), dtype=np.float32),
        "source_face_patch_internal_pressure": np.zeros((face_count,), dtype=np.float32),
        "source_face_patch_feature_pressure": np.zeros((face_count,), dtype=np.float32),
        "source_face_collapse_accept": np.zeros((face_count,), dtype=np.float32),
        "source_face_reject_boundary": np.zeros((face_count,), dtype=np.float32),
        "source_face_reject_quality": np.zeros((face_count,), dtype=np.float32),
    }
    rows: list[dict[str, Any]] = []
    face_source = np.arange(face_count, dtype=np.int64)
    pass_rows: list[dict[str, Any]] = []
    total_accepted = 0
    if patch_normals.shape[0] > 0:
        vertices, moved = _project_vertices_to_patches(vertices, faces, source_patch_id, patch_normals, patch_offsets, float(config.project_regularization))
    else:
        moved = np.zeros((vertices.shape[0],), dtype=np.float32)
    pre_project_summary = _summarize(moved, moved > 0.0)

    for pass_index in range(max(int(config.max_passes), 0)):
        current_source_patch = np.full((faces.shape[0],), -1, dtype=np.int32)
        valid = (face_source >= 0) & (face_source < source_patch_id.shape[0])
        current_source_patch[valid] = source_patch_id[face_source[valid]]
        vertices, moved_pass = _project_vertices_to_patches(vertices, faces, current_source_patch, patch_normals, patch_offsets, float(config.project_regularization))
        accepted, candidate_positions, accepted_count, counts = _evaluate_patch_collapses(
            vertices=vertices,
            faces=faces,
            face_source=face_source,
            source_patch_id=source_patch_id,
            patch_normals=patch_normals,
            patch_offsets=patch_offsets,
            patch_target=patch_target,
            config=config,
            source_accum=source_accum,
            rows=rows,
            pass_index=pass_index,
        )
        topology = build_mesh_topology(faces)
        vertices_next, faces_next, source_next, cleanup = _apply_batched_collapses(
            vertices,
            faces,
            face_source,
            topology.edge_vertices,
            accepted,
            candidate_positions,
        )
        pass_rows.append(
            {
                "pass": int(pass_index),
                "facesBefore": int(faces.shape[0]),
                "facesAfter": int(faces_next.shape[0]),
                "verticesBefore": int(vertices.shape[0]),
                "verticesAfter": int(vertices_next.shape[0]),
                "acceptedCollapses": int(accepted_count),
                "counts": counts,
                "projectionMovedVertices": int(np.count_nonzero(moved_pass > 0.0)),
                "cleanup": cleanup,
            }
        )
        total_accepted += int(accepted_count)
        vertices, faces, face_source = vertices_next, faces_next, source_next
        progress(f"[planar-patch-remesh] Pass {pass_index + 1}: accepted={accepted_count:,}; faces={faces.shape[0]:,}")
        if accepted_count == 0:
            break

    final_source_patch = np.full((faces.shape[0],), -1, dtype=np.int32)
    valid = (face_source >= 0) & (face_source < source_patch_id.shape[0])
    final_source_patch[valid] = source_patch_id[face_source[valid]]
    vertices, final_moved = _project_vertices_to_patches(vertices, faces, final_source_patch, patch_normals, patch_offsets, float(config.project_regularization))
    survived = np.zeros((face_count,), dtype=np.float32)
    if face_source.size:
        valid_source = face_source[(face_source >= 0) & (face_source < face_count)]
        survived[np.unique(valid_source)] = 1.0
    removed = (1.0 - survived).astype(np.float32)
    diagnostics = {
        "final_face_source_id": face_source.astype(np.int64),
        "source_face_patch_id": source_patch_id.astype(np.int32),
        "source_face_planar_patch_mask": source_face_patch_mask.astype(np.float32),
        "source_face_patch_boundary_stop": source_face_patch_boundary_stop.astype(np.float32),
        "source_face_feature_preserve_weight": feature_weight.astype(np.float32),
        "source_face_feature_preserve_mask": source_face_feature_preserve_mask.astype(np.float32),
        "source_face_collapse_pressure": source_accum["source_face_collapse_pressure"].astype(np.float32),
        "source_face_patch_internal_pressure": source_accum["source_face_patch_internal_pressure"].astype(np.float32),
        "source_face_patch_feature_pressure": source_accum["source_face_patch_feature_pressure"].astype(np.float32),
        "source_face_collapse_accept": source_accum["source_face_collapse_accept"].astype(np.float32),
        "source_face_removed_by_collapse": removed,
        "source_face_survived": survived,
        "source_face_weight_planar": _clip01(policy["face_weight_planar"]),
        "source_face_weight_detail": _clip01(policy.get("face_weight_detail", np.zeros(face_count))),
        "source_face_weight_boundary": _clip01(policy.get("face_weight_boundary", np.zeros(face_count))),
        "source_face_weight_uncertain": _clip01(policy.get("face_weight_uncertain", np.zeros(face_count))),
        "source_face_reject_boundary": source_accum["source_face_reject_boundary"].astype(np.float32),
        "source_face_reject_uncertainty": np.zeros((face_count,), dtype=np.float32),
        "source_face_reject_quality": source_accum["source_face_reject_quality"].astype(np.float32),
        "source_face_reject_normal": np.zeros((face_count,), dtype=np.float32),
        "source_face_reject_projection": np.zeros((face_count,), dtype=np.float32),
        "source_face_final_target_length": np.asarray(policy.get("face_detail_target_length", np.ones(face_count)), dtype=np.float32),
    }

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)
    np.savez_compressed(diagnostics_npz, **diagnostics)
    _write_jsonl(operations_jsonl, rows)
    debug_meshes = _write_debug_meshes(original_vertices, original_faces, diagnostics, output_dir) if bool(config.write_debug_meshes) else {}
    input_stats = _mesh_stats(original_vertices, original_faces)
    output_stats = _mesh_stats(vertices, faces)
    summary = {
        "stageName": "omega_planar_patch_remesh",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(config), "mesh_path": str(mesh_path), "policy_npz": str(policy_npz), "proxies_npz": str(proxies_npz), "output_dir": str(output_dir)},
        "counts": {
            "inputFaces": int(original_faces.shape[0]),
            "outputFaces": int(faces.shape[0]),
            "inputVertices": int(original_vertices.shape[0]),
            "outputVertices": int(vertices.shape[0]),
            "planarPatchCount": int(len(patch_rows)),
            "planarPatchSourceFaces": int(np.count_nonzero(source_patch_id >= 0)),
            "planarPatchBoundaryStopSourceFaces": int(np.count_nonzero(source_face_patch_boundary_stop > 0.0)),
            "featurePreserveSourceFaces": int(np.count_nonzero(source_face_feature_preserve_mask > 0.0)),
            "acceptedCollapses": int(total_accepted),
            "removedSourceFaces": int(np.count_nonzero(removed > 0.0)),
        },
        "inputMesh": input_stats,
        "outputMesh": output_stats,
        "preProject": pre_project_summary,
        "finalProject": _summarize(final_moved, final_moved > 0.0),
        "passes": pass_rows,
        "patches": patch_rows,
        "stats": {name: _summarize(value, value > 0.0) for name, value in diagnostics.items() if np.asarray(value).ndim == 1 and np.asarray(value).dtype.kind in {"f", "i", "u"}},
        "outputs": {
            "outputMesh": str(output_mesh),
            "diagnosticsNpz": str(diagnostics_npz),
            "operationsJsonl": str(operations_jsonl),
            "summaryJson": str(summary_json),
            "reportMd": str(report_md),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
        "notes": [
            "This pass is region-first: high planar-weight faces become planar patches.",
            "Patch-interior vertices are forced to fitted planes before and after interior simplification.",
            "Vertices touching non-patch faces are protected so planar cleanup does not deform neighboring detail faces.",
            "Patch boundaries, proxy seams, strong normal boundaries, and adjacent fitted planes are hard no-cross stops.",
            "High detail and uncertainty are merged into source_face_feature_preserve_weight and preserved outside planar patches.",
            "The output remains a triangle mesh; source_face_patch_id records the merged planar patch identity.",
        ],
    }
    _write_json(summary_json, summary)
    report = [
        "# Planar Patch Remesh Report",
        "",
        f"- Faces: {original_faces.shape[0]:,} -> {faces.shape[0]:,}",
        f"- Vertices: {original_vertices.shape[0]:,} -> {vertices.shape[0]:,}",
        f"- Planar patches: {len(patch_rows):,}",
        f"- Planar source faces: {int(np.count_nonzero(source_patch_id >= 0)):,}",
        f"- Planar patch boundary-stop source faces: {int(np.count_nonzero(source_face_patch_boundary_stop > 0.0)):,}",
        f"- Feature-preserve source faces outside planar patches: {int(np.count_nonzero(source_face_feature_preserve_mask > 0.0)):,}",
        f"- Accepted collapses: {int(total_accepted):,}",
        f"- Removed source faces: {int(np.count_nonzero(removed > 0.0)):,}",
        "",
        "## Passes",
    ]
    for row in pass_rows:
        report.append(
            f"- Pass {int(row['pass']) + 1}: faces {row['facesBefore']:,} -> {row['facesAfter']:,}; "
            f"accepted {row['acceptedCollapses']:,}; candidates {row['counts'].get('candidate', 0):,}"
        )
    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    progress(f"[planar-patch-remesh] Wrote {output_mesh}")
    progress(f"[planar-patch-remesh] Wrote {diagnostics_npz}")
    progress(f"[planar-patch-remesh] Wrote {report_md}")
    return PlanarPatchRemeshResult(
        output_mesh=output_mesh,
        diagnostics_npz=diagnostics_npz,
        operations_jsonl=operations_jsonl,
        summary_json=summary_json,
        report_md=report_md,
        input_face_count=int(original_faces.shape[0]),
        output_face_count=int(faces.shape[0]),
        accepted_collapses=int(total_accepted),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run simplification-first planar patch remesh.")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--proxies-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-name", default="mesh_planar_patch_remesh.ply")
    parser.add_argument("--diagnostics-name", default="planar_patch_remesh_diagnostics")
    parser.add_argument("--operations-name", default="planar_patch_remesh_operations")
    parser.add_argument("--planar-threshold", type=float, default=0.22)
    parser.add_argument("--feature-threshold", type=float, default=0.50)
    parser.add_argument("--boundary-stop-threshold", type=float, default=0.35)
    parser.add_argument("--min-patch-faces", type=int, default=12)
    parser.add_argument("--patch-normal-degrees", type=float, default=12.0)
    parser.add_argument("--patch-distance-factor", type=float, default=4.0)
    parser.add_argument("--patch-trim-distance-factor", type=float, default=2.5)
    parser.add_argument("--planar-target-factor", type=float, default=7.0)
    parser.add_argument("--max-passes", type=int, default=12)
    parser.add_argument("--max-collapses-per-pass", type=int, default=150000)
    parser.add_argument("--collapse-threshold", type=float, default=0.02)
    parser.add_argument("--q-min-hard", type=float, default=0.005)
    parser.add_argument("--q-min-soft", type=float, default=0.03)
    parser.add_argument("--max-quality-error", type=float, default=2.0)
    parser.add_argument("--max-jsonl-rows", type=int, default=50000)
    parser.add_argument("--no-debug-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path, base: Path) -> Path:
    raw = path.expanduser()
    return raw.resolve() if raw.is_absolute() else (base / raw).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    mesh_path = _resolve(args.mesh, model_dir) if args.mesh is not None else model_dir / "remesh" / "local" / "preclean_mesh.ply"
    output_dir = _resolve(args.output_dir, model_dir) if args.output_dir is not None else model_dir / "remesh" / "local"
    compute_planar_patch_remesh(
        PlanarPatchRemeshConfig(
            mesh_path=mesh_path,
            policy_npz=_resolve(args.policy_npz, model_dir) if args.policy_npz is not None else model_dir / "remesh" / "local" / "policy.npz",
            proxies_npz=_resolve(args.proxies_npz, model_dir) if args.proxies_npz is not None else model_dir / "remesh" / "local" / "proxies.npz",
            output_dir=output_dir,
            output_name=str(args.output_name),
            diagnostics_name=str(args.diagnostics_name),
            operations_name=str(args.operations_name),
            planar_threshold=float(args.planar_threshold),
            feature_threshold=float(args.feature_threshold),
            boundary_stop_threshold=float(args.boundary_stop_threshold),
            min_patch_faces=int(args.min_patch_faces),
            patch_normal_degrees=float(args.patch_normal_degrees),
            patch_distance_factor=float(args.patch_distance_factor),
            patch_trim_distance_factor=float(args.patch_trim_distance_factor),
            planar_target_factor=float(args.planar_target_factor),
            max_passes=int(args.max_passes),
            max_collapses_per_pass=int(args.max_collapses_per_pass),
            collapse_threshold=float(args.collapse_threshold),
            q_min_hard=float(args.q_min_hard),
            q_min_soft=float(args.q_min_soft),
            max_quality_error=float(args.max_quality_error),
            max_jsonl_rows=int(args.max_jsonl_rows),
            write_debug_meshes=not bool(args.no_debug_meshes),
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
