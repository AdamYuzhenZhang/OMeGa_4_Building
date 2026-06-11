"""Per-vertex density export for OMeGa-guided Geogram remeshing.

Geogram/Vorpalite already supports scalar vertex weights in its CVT objective.
This module converts Phase 3 continuous remesh weights into that scalar density:
large planar regions receive low density, while low-planarity/detail/boundary
regions receive higher density so anisotropic CVT does not simplify them away.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omega_local.remesh.local_ops import face_geometry, load_mesh_arrays
from omega_local.remesh.mesh_clean import resolve_latest_omega_mesh
from omega_local.remesh.transfer import summarize


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class OmegaGeogramDensityConfig:
    model_dir: Path
    mesh_path: Path | None = None
    policy_npz: Path | None = None
    proxies_npz: Path | None = None
    evidence_npz: Path | None = None
    output_dir: Path | None = None
    density_name: str = "omega_geogram_density"
    iteration: int = -1
    min_density: float = 0.25
    max_density: float = 4.0
    signal_gamma: float = 1.25
    detail_influence: float = 1.0
    boundary_influence: float = 0.5
    uncertain_influence: float = 0.0
    auto_normal_error_degrees: float = 0.0
    auto_h_min_factor: float = 0.50
    auto_h_max_factor: float = 6.0
    auto_target_point_scale: float = 1.0
    use_fused_normals: bool = True
    fused_normal_blend: float = 0.70
    fused_normal_min_weight: float = 8.0
    planar_gradation_iterations: int = 0
    planar_gradation_strength: float = 0.45
    planar_gradation_planar_power: float = 2.0
    planar_gradation_min_edge_weight: float = 0.05
    planar_gradation_max_density_ratio: float = 1.75
    planar_gradation_normal_lift_scale: float = 1.0
    planar_gradation_metric_tau_factor: float = 2.0
    planar_gradation_min_normal_gate: float = 0.05
    planar_gradation_use_proxy_boundaries: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class OmegaGeogramDensityResult:
    density_txt: Path
    normal_txt: Path | None
    density_npz: Path
    summary_json: Path
    vertex_count: int
    face_count: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_existing(path: Path | None, fallback: Path, label: str) -> Path:
    resolved = path.expanduser().resolve() if path is not None else fallback.expanduser().resolve()
    if not resolved.exists() or resolved.is_dir():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def _face_to_vertex_area_mean(
    faces: np.ndarray,
    face_values: np.ndarray,
    face_area: np.ndarray,
    vertex_count: int,
) -> np.ndarray:
    numer = np.zeros((int(vertex_count),), dtype=np.float64)
    denom = np.zeros((int(vertex_count),), dtype=np.float64)
    weighted = np.asarray(face_values, dtype=np.float64) * np.asarray(face_area, dtype=np.float64)
    for local in range(3):
        ids = np.asarray(faces[:, local], dtype=np.int64)
        np.add.at(numer, ids, weighted)
        np.add.at(denom, ids, face_area)
    return np.divide(numer, np.maximum(denom, 1.0e-12)).astype(np.float32)


def _face_to_vertex_max(faces: np.ndarray, face_values: np.ndarray, vertex_count: int) -> np.ndarray:
    out = np.zeros((int(vertex_count),), dtype=np.float32)
    values = np.asarray(face_values, dtype=np.float32)
    for local in range(3):
        np.maximum.at(out, np.asarray(faces[:, local], dtype=np.int64), values)
    return out


def _mesh_edge_topology(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=np.int64)
    edge_to_id: dict[tuple[int, int], int] = {}
    edge_vertices: list[tuple[int, int]] = []
    edge_faces: list[list[int]] = []
    face_edges = np.full((int(faces.shape[0]), 3), -1, dtype=np.int64)
    for face_id, face in enumerate(faces):
        corners = (int(face[0]), int(face[1]), int(face[2]))
        for local_id, (u, v) in enumerate(((corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[0]))):
            key = (u, v) if u <= v else (v, u)
            edge_id = edge_to_id.get(key)
            if edge_id is None:
                edge_id = len(edge_vertices)
                edge_to_id[key] = edge_id
                edge_vertices.append(key)
                edge_faces.append([face_id, -1])
            elif edge_faces[edge_id][1] < 0:
                edge_faces[edge_id][1] = face_id
            face_edges[face_id, local_id] = edge_id
    return (
        np.asarray(edge_vertices, dtype=np.int64),
        np.asarray(edge_faces, dtype=np.int64),
        face_edges,
    )


def _edge_topology_from_policy(policy: dict[str, np.ndarray], faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    face_count = int(faces.shape[0])
    if {"edge_vertices", "edge_faces", "face_edges"}.issubset(policy):
        edge_vertices = np.asarray(policy["edge_vertices"], dtype=np.int64)
        edge_faces = np.asarray(policy["edge_faces"], dtype=np.int64)
        face_edges = np.asarray(policy["face_edges"], dtype=np.int64)
        if (
            edge_vertices.ndim == 2
            and edge_vertices.shape[1] == 2
            and edge_faces.ndim == 2
            and edge_faces.shape == (edge_vertices.shape[0], 2)
            and face_edges.ndim == 2
            and face_edges.shape == (face_count, 3)
        ):
            return edge_vertices, edge_faces, face_edges
    return _mesh_edge_topology(faces)


def _edge_to_face_max(face_edges: np.ndarray, edge_values: np.ndarray, face_count: int) -> np.ndarray:
    face_edges = np.asarray(face_edges, dtype=np.int64)
    edge_values = np.asarray(edge_values, dtype=np.float32)
    out = np.zeros((int(face_count),), dtype=np.float32)
    for local_id in range(3):
        ids = face_edges[:, local_id]
        valid = ids >= 0
        current = np.zeros_like(out)
        current[valid] = edge_values[ids[valid]]
        out = np.maximum(out, current)
    return out


def _edge_to_face_mean(face_edges: np.ndarray, edge_values: np.ndarray, face_count: int) -> np.ndarray:
    face_edges = np.asarray(face_edges, dtype=np.int64)
    edge_values = np.asarray(edge_values, dtype=np.float32)
    numer = np.zeros((int(face_count),), dtype=np.float64)
    denom = np.zeros((int(face_count),), dtype=np.float64)
    for local_id in range(3):
        ids = face_edges[:, local_id]
        valid = ids >= 0
        numer[valid] += edge_values[ids[valid]]
        denom[valid] += 1.0
    return np.divide(numer, np.maximum(denom, 1.0e-12)).astype(np.float32)


def _load_optional_npz(path: Path | None) -> dict[str, np.ndarray]:
    if path is None or not path.exists() or path.is_dir():
        return {}
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _vertex_policy_value(
    policy: dict[str, np.ndarray],
    key: str,
    face_key: str,
    faces: np.ndarray,
    face_area: np.ndarray,
    vertex_count: int,
    *,
    reducer: str,
    fallback: float = 0.0,
) -> np.ndarray:
    if key in policy and np.asarray(policy[key]).shape[0] == int(vertex_count):
        return _clip01(policy[key])
    if face_key in policy and np.asarray(policy[face_key]).shape[0] == int(faces.shape[0]):
        if reducer == "max":
            return _clip01(_face_to_vertex_max(faces, policy[face_key], vertex_count))
        return _clip01(_face_to_vertex_area_mean(faces, policy[face_key], face_area, vertex_count))
    return np.full((int(vertex_count),), float(fallback), dtype=np.float32)


def _write_density_txt(
    path: Path,
    *,
    density: np.ndarray,
    planar: np.ndarray,
    detail: np.ndarray,
    boundary: np.ndarray,
    uncertain: np.ndarray,
    signal: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("omega_geogram_density_v1\n")
        handle.write(f"vertex_count {int(density.shape[0])}\n")
        handle.write("columns index density planar detail boundary uncertain signal\n")
        for idx in range(int(density.shape[0])):
            handle.write(
                "{} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(
                    idx,
                    float(density[idx]),
                    float(planar[idx]),
                    float(detail[idx]),
                    float(boundary[idx]),
                    float(uncertain[idx]),
                    float(signal[idx]),
                )
            )


def _write_normal_txt(
    path: Path,
    *,
    normals: np.ndarray,
    weights: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("omega_geogram_normals_v1\n")
        handle.write(f"vertex_count {int(normals.shape[0])}\n")
        handle.write("columns index nx ny nz weight\n")
        for idx in range(int(normals.shape[0])):
            handle.write(
                "{} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(
                    idx,
                    float(normals[idx, 0]),
                    float(normals[idx, 1]),
                    float(normals[idx, 2]),
                    float(weights[idx]),
                )
            )


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, np.maximum(norm, eps), out=np.zeros_like(vectors), where=norm > 0.0)


def _face_to_vertex_normal_mean(
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_weights: np.ndarray,
    face_area: np.ndarray,
    vertex_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    numer = np.zeros((int(vertex_count), 3), dtype=np.float64)
    denom = np.zeros((int(vertex_count),), dtype=np.float64)
    weighted = np.asarray(face_normals, dtype=np.float64) * (np.asarray(face_weights, dtype=np.float64) * np.asarray(face_area, dtype=np.float64))[:, None]
    scalar_weight = np.asarray(face_weights, dtype=np.float64) * np.asarray(face_area, dtype=np.float64)
    for local_id in range(3):
        ids = np.asarray(faces[:, local_id], dtype=np.int64)
        np.add.at(numer, ids, weighted)
        np.add.at(denom, ids, scalar_weight)
    normals = _safe_normalize(np.divide(numer, np.maximum(denom[:, None], 1.0e-12), out=np.zeros_like(numer), where=denom[:, None] > 0.0))
    weights = np.divide(denom, np.maximum(denom + np.median(face_area), 1.0e-12), out=np.zeros_like(denom), where=denom > 0.0)
    return normals.astype(np.float32), np.clip(weights, 0.0, 1.0).astype(np.float32)


def _compute_auto_sizing(
    *,
    evidence: dict[str, np.ndarray],
    policy: dict[str, np.ndarray],
    faces: np.ndarray,
    face_area: np.ndarray,
    face_mean_edge: np.ndarray,
    face_count: int,
    vertex_count: int,
    normal_error_degrees: float,
    h_min_factor: float,
    h_max_factor: float,
    target_point_scale: float,
) -> dict[str, Any] | None:
    if float(normal_error_degrees) <= 0.0 or not evidence:
        return None
    required = ("face_detail_target_length", "face_tau_n")
    if any(key not in evidence for key in required):
        return None

    normal_count = np.asarray(evidence.get("normal_count", policy.get("normal_count", np.zeros(face_count))), dtype=np.int32).reshape(-1)
    tau_n = np.asarray(evidence["face_tau_n"], dtype=np.float64).reshape(-1)
    target = np.asarray(evidence["face_detail_target_length"], dtype=np.float64).reshape(-1)
    mean_edge = np.asarray(face_mean_edge, dtype=np.float64).reshape(-1)
    valid = (
        (normal_count[:face_count] > 0)
        & np.isfinite(tau_n[:face_count])
        & np.isfinite(target[:face_count])
        & (tau_n[:face_count] > 1.0e-8)
        & (target[:face_count] > 1.0e-8)
    )
    if not np.any(valid):
        return None

    normal_error = np.deg2rad(float(normal_error_degrees))
    h_face = np.asarray(mean_edge, dtype=np.float64).copy()
    h_face[valid] = target[:face_count][valid] * normal_error / np.maximum(tau_n[:face_count][valid], 1.0e-8)

    positive_edges = mean_edge[np.isfinite(mean_edge) & (mean_edge > 1.0e-8)]
    edge_ref = float(np.median(positive_edges)) if positive_edges.size else 1.0
    h_min = max(float(h_min_factor), 1.0e-3) * float(np.quantile(positive_edges, 0.10) if positive_edges.size else edge_ref)
    h_max = max(h_min * 1.01, float(h_max_factor) * float(np.quantile(positive_edges, 0.90) if positive_edges.size else edge_ref))
    h_face = np.clip(h_face, h_min, h_max)

    h_vertex = _face_to_vertex_area_mean(faces, h_face.astype(np.float32), face_area, vertex_count).astype(np.float64)
    h_ref = float(np.median(h_vertex[np.isfinite(h_vertex) & (h_vertex > 1.0e-8)])) if np.any(h_vertex > 1.0e-8) else edge_ref
    density = np.square(h_ref / np.maximum(h_vertex, 1.0e-8)).astype(np.float32)

    equilateral_vertex_area = np.sqrt(3.0) * 0.5 * np.square(np.maximum(h_face, 1.0e-8))
    estimated_points = int(round(float(target_point_scale) * float(np.sum(np.asarray(face_area, dtype=np.float64) / equilateral_vertex_area))))
    estimated_points = max(4, estimated_points)
    return {
        "face_h": h_face.astype(np.float32),
        "vertex_h": h_vertex.astype(np.float32),
        "vertex_density": density,
        "estimated_points": int(estimated_points),
        "estimated_faces": int(max(4, round(2.0 * estimated_points))),
        "summary": {
            "enabled": True,
            "normalErrorDegrees": float(normal_error_degrees),
            "normalErrorRadians": float(normal_error),
            "hMin": float(h_min),
            "hMax": float(h_max),
            "hReference": float(h_ref),
            "targetPointScale": float(target_point_scale),
            "estimatedTargetPoints": int(estimated_points),
            "estimatedTargetFaces": int(max(4, round(2.0 * estimated_points))),
            "validFaceCount": int(np.count_nonzero(valid)),
            "validFaceFraction": float(np.count_nonzero(valid) / max(face_count, 1)),
            "faceTargetLength": summarize(h_face.astype(np.float32), valid),
            "vertexTargetLength": summarize(h_vertex.astype(np.float32)),
            "vertexDensity": summarize(density),
        },
    }


def _apply_planar_gradation(
    *,
    density: np.ndarray,
    planar: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    policy: dict[str, np.ndarray],
    proxies: dict[str, np.ndarray],
    iterations: int,
    strength: float,
    planar_power: float,
    min_edge_weight: float,
    max_density_ratio: float,
    normal_lift_scale: float,
    metric_tau_factor: float,
    min_normal_gate: float,
    use_proxy_boundaries: bool,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Smooth density over same-plane edges without erasing the signal.

    The scalar density is smoothed in log space.  Edge weights are continuous:
    endpoint planarity makes an edge eligible, while the normal-lifted metric
    and proxy boundaries suppress smoothing across corners or adjacent planes.
    """

    iterations = max(int(iterations), 0)
    strength = max(float(strength), 0.0)
    if iterations == 0 or strength <= 0.0:
        return density.astype(np.float32, copy=True), {
            "enabled": False,
            "iterations": int(iterations),
            "edgeCount": 0,
            "activeEdgeCount": 0,
        }, {}

    edge_vertices, edge_faces, face_edges = _edge_topology_from_policy(policy, np.asarray(faces, dtype=np.int64))
    if edge_vertices.size == 0:
        return density.astype(np.float32, copy=True), {
            "enabled": False,
            "iterations": int(iterations),
            "edgeCount": 0,
            "activeEdgeCount": 0,
        }, {}

    a = edge_vertices[:, 0]
    b = edge_vertices[:, 1]
    edge_planar = np.power(
        np.clip(np.minimum(planar[a], planar[b]), 0.0, 1.0),
        max(float(planar_power), 1.0e-6),
    ).astype(np.float64)
    edge_planar[edge_planar < float(min_edge_weight)] = 0.0

    vertices64 = np.asarray(vertices, dtype=np.float64)
    edge_length = np.linalg.norm(vertices64[a] - vertices64[b], axis=1)
    positive_lengths = edge_length[edge_length > 1.0e-12]
    median_edge = float(np.median(positive_lengths)) if positive_lengths.size else 1.0
    bbox_diag = float(np.linalg.norm(np.max(vertices64, axis=0) - np.min(vertices64, axis=0)))
    lift_alpha = 0.02 * max(float(normal_lift_scale), 0.0) * max(bbox_diag, 1.0e-12)

    normal_gate = np.ones((int(edge_vertices.shape[0]),), dtype=np.float64)
    lifted_normal_length = np.zeros_like(normal_gate)
    metric_ratio = np.zeros_like(normal_gate)
    has_two = np.asarray(edge_faces[:, 1] >= 0, dtype=bool)
    if np.any(has_two) and lift_alpha > 0.0:
        face_normals64 = np.asarray(face_normals, dtype=np.float64)
        fa = edge_faces[has_two, 0]
        fb = edge_faces[has_two, 1]
        dot = np.abs(np.sum(face_normals64[fa] * face_normals64[fb], axis=1)).clip(0.0, 1.0)
        normal_delta = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dot))
        lifted = lift_alpha * normal_delta
        tau = max(float(metric_tau_factor), 1.0e-6) * np.maximum(edge_length[has_two], 0.25 * median_edge)
        gate = np.exp(-np.square(lifted / np.maximum(tau, 1.0e-12)))
        gate[gate < float(min_normal_gate)] = 0.0
        normal_gate[has_two] = gate
        lifted_normal_length[has_two] = lifted
        metric_ratio[has_two] = lifted / np.maximum(tau, 1.0e-12)
    elif np.any(has_two):
        face_normals64 = np.asarray(face_normals, dtype=np.float64)
        fa = edge_faces[has_two, 0]
        fb = edge_faces[has_two, 1]
        dot = np.abs(np.sum(face_normals64[fa] * face_normals64[fb], axis=1)).clip(0.0, 1.0)
        normal_delta = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dot))
        lifted_normal_length[has_two] = normal_delta

    proxy_gate = np.ones((int(edge_vertices.shape[0]),), dtype=np.float64)
    proxy_stop_count = 0
    if bool(use_proxy_boundaries) and proxies:
        edge_proxy_boundary = np.asarray(proxies.get("edge_proxy_boundary", np.zeros((0,), dtype=np.uint8))).reshape(-1)
        proxy_edge_faces = np.asarray(proxies.get("edge_faces", np.zeros((0, 2), dtype=np.int64)), dtype=np.int64)
        proxy_edge_topology_ok = proxy_edge_faces.size == 0 or (
            proxy_edge_faces.shape == edge_faces.shape and np.array_equal(proxy_edge_faces, edge_faces)
        )
        if edge_proxy_boundary.shape[0] == edge_vertices.shape[0] and proxy_edge_topology_ok:
            stop = edge_proxy_boundary > 0
            proxy_gate[stop] = 0.0
            proxy_stop_count += int(np.count_nonzero(stop))
        face_proxy_id = np.asarray(proxies.get("face_proxy_id", np.zeros((0,), dtype=np.int32)), dtype=np.int64).reshape(-1)
        if face_proxy_id.shape[0] == int(faces.shape[0]) and np.any(has_two):
            fa = edge_faces[has_two, 0]
            fb = edge_faces[has_two, 1]
            pa = face_proxy_id[fa]
            pb = face_proxy_id[fb]
            stop_two = (pa >= 0) & (pb >= 0) & (pa != pb)
            proxy_gate[np.flatnonzero(has_two)[stop_two]] = 0.0
            proxy_stop_count += int(np.count_nonzero(stop_two))

    edge_weight = edge_planar * normal_gate * proxy_gate
    edge_weight[edge_weight < float(min_edge_weight)] = 0.0
    active = edge_weight > 0.0
    debug = {
        "edge_planar_gradation_weight": edge_weight.astype(np.float32),
        "edge_planar_gradation_planar_weight": edge_planar.astype(np.float32),
        "edge_planar_gradation_normal_gate": normal_gate.astype(np.float32),
        "edge_planar_gradation_proxy_gate": proxy_gate.astype(np.float32),
        "edge_planar_gradation_lifted_normal_length": lifted_normal_length.astype(np.float32),
        "edge_planar_gradation_metric_ratio": metric_ratio.astype(np.float32),
        "edge_vertices": edge_vertices.astype(np.int64),
        "edge_faces": edge_faces.astype(np.int64),
        "face_edges": face_edges.astype(np.int64),
        "face_planar_gradation_weight": _edge_to_face_max(face_edges, edge_weight, int(faces.shape[0])).astype(np.float32),
        "face_planar_gradation_planar_weight": _edge_to_face_max(face_edges, edge_planar, int(faces.shape[0])).astype(np.float32),
        "face_planar_gradation_normal_gate": _edge_to_face_mean(face_edges, normal_gate, int(faces.shape[0])).astype(np.float32),
        "face_planar_gradation_proxy_gate": _edge_to_face_mean(face_edges, proxy_gate, int(faces.shape[0])).astype(np.float32),
        "face_planar_gradation_metric_ratio": _edge_to_face_max(face_edges, metric_ratio, int(faces.shape[0])).astype(np.float32),
    }
    if not np.any(active):
        return density.astype(np.float32, copy=True), {
            "enabled": False,
            "iterations": int(iterations),
            "edgeCount": int(edge_vertices.shape[0]),
            "activeEdgeCount": 0,
            "normalLiftScale": float(normal_lift_scale),
            "normalLiftAlpha": float(lift_alpha),
            "metricTauFactor": float(metric_tau_factor),
            "minNormalGate": float(min_normal_gate),
            "proxyBoundaryStops": int(proxy_stop_count),
        }, debug

    a = a[active]
    b = b[active]
    edge_weight = edge_weight[active]
    base_log = np.log(np.maximum(np.asarray(density, dtype=np.float64), 1.0e-12))
    current = base_log.copy()
    max_log_ratio = np.log(max(float(max_density_ratio), 1.0))
    for _ in range(iterations):
        numer = current.copy()
        denom = np.ones_like(current)
        w = float(strength) * edge_weight
        np.add.at(numer, a, w * current[b])
        np.add.at(denom, a, w)
        np.add.at(numer, b, w * current[a])
        np.add.at(denom, b, w)
        current = numer / np.maximum(denom, 1.0e-12)
        if max_log_ratio > 0.0:
            current = np.clip(current, base_log - max_log_ratio, base_log + max_log_ratio)

    smoothed = np.exp(current).astype(np.float32)
    change_ratio = np.divide(
        smoothed,
        np.maximum(np.asarray(density, dtype=np.float32), 1.0e-12),
        out=np.ones_like(smoothed, dtype=np.float32),
        where=np.asarray(density, dtype=np.float32) > 0.0,
    )
    return smoothed, {
        "enabled": True,
        "iterations": int(iterations),
        "strength": float(strength),
        "planarPower": float(planar_power),
        "minEdgeWeight": float(min_edge_weight),
        "maxDensityRatio": float(max_density_ratio),
        "normalLiftScale": float(normal_lift_scale),
        "normalLiftAlpha": float(lift_alpha),
        "metricTauFactor": float(metric_tau_factor),
        "minNormalGate": float(min_normal_gate),
        "usesProxyBoundaries": bool(use_proxy_boundaries),
        "proxyBoundaryStops": int(proxy_stop_count),
        "medianEdgeLength": float(median_edge),
        "edgeCount": int(edge_vertices.shape[0]),
        "activeEdgeCount": int(edge_weight.shape[0]),
        "activeEdgeFraction": float(edge_weight.shape[0] / max(edge_vertices.shape[0], 1)),
        "planarEdgeWeight": summarize(debug["edge_planar_gradation_planar_weight"]),
        "normalGate": summarize(debug["edge_planar_gradation_normal_gate"]),
        "proxyGate": summarize(debug["edge_planar_gradation_proxy_gate"]),
        "metricRatio": summarize(debug["edge_planar_gradation_metric_ratio"]),
        "changeRatio": summarize(change_ratio),
    }, debug


def compute_omega_geogram_density(
    config: OmegaGeogramDensityConfig,
    progress: ProgressFn | None = None,
) -> OmegaGeogramDensityResult:
    progress = progress or _progress_default
    model_dir = config.model_dir.expanduser().resolve()
    mesh_path = (
        config.mesh_path.expanduser().resolve()
        if config.mesh_path is not None
        else resolve_latest_omega_mesh(model_dir, iteration=int(config.iteration))
    )
    policy_npz = _resolve_existing(config.policy_npz, model_dir / "remesh" / "local" / "policy.npz", "policy npz")
    proxies_path = (
        config.proxies_npz.expanduser().resolve()
        if config.proxies_npz is not None
        else model_dir / "remesh" / "local" / "proxies.npz"
    )
    evidence_path = (
        config.evidence_npz.expanduser().resolve()
        if config.evidence_npz is not None
        else model_dir / "remesh" / "local" / "evidence.npz"
    )
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else model_dir / "remesh" / "geogram_density"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    density_txt = output_dir / f"{config.density_name}.txt"
    normal_txt = output_dir / f"{config.density_name}_normals.txt"
    density_npz = output_dir / f"{config.density_name}.npz"
    summary_json = output_dir / f"{config.density_name}_summary.json"
    for path in (density_txt, normal_txt, density_npz, summary_json):
        if path.exists() and not bool(config.overwrite):
            raise FileExistsError(f"Output exists: {path}. Re-run with --overwrite.")

    vertices, faces = load_mesh_arrays(mesh_path)
    _centers, face_normals, face_area, _mean_edge = face_geometry(vertices, faces)
    policy_data = np.load(policy_npz)
    policy = {key: policy_data[key] for key in policy_data.files}
    proxies = _load_optional_npz(proxies_path)
    evidence = _load_optional_npz(evidence_path)
    face_count = int(faces.shape[0])
    vertex_count = int(vertices.shape[0])
    if "face_weight_planar" in policy and int(np.asarray(policy["face_weight_planar"]).shape[0]) != face_count:
        raise ValueError("Policy face count does not match the density-export mesh.")

    planar = _vertex_policy_value(
        policy,
        "vertex_weight_planar",
        "face_weight_planar",
        faces,
        face_area,
        vertex_count,
        reducer="mean",
        fallback=0.0,
    )
    detail = _vertex_policy_value(
        policy,
        "vertex_weight_detail",
        "face_weight_detail",
        faces,
        face_area,
        vertex_count,
        reducer="max",
        fallback=0.0,
    )
    boundary = _vertex_policy_value(
        policy,
        "vertex_weight_boundary",
        "face_weight_boundary",
        faces,
        face_area,
        vertex_count,
        reducer="max",
        fallback=0.0,
    )
    uncertain = _vertex_policy_value(
        policy,
        "vertex_weight_uncertain",
        "face_weight_uncertain",
        faces,
        face_area,
        vertex_count,
        reducer="max",
        fallback=0.0,
    )

    nonplanar = 1.0 - planar
    signal = np.maximum(nonplanar, float(config.detail_influence) * detail)
    signal = np.maximum(signal, float(config.boundary_influence) * boundary)
    if float(config.uncertain_influence) != 0.0:
        signal = np.maximum(signal, float(config.uncertain_influence) * uncertain)
    signal = np.clip(signal, 0.0, 1.0)
    signal = np.power(signal, max(float(config.signal_gamma), 1.0e-6)).astype(np.float32)

    min_density = max(float(config.min_density), 1.0e-6)
    max_density = max(float(config.max_density), min_density + 1.0e-6)
    auto_sizing = _compute_auto_sizing(
        evidence=evidence,
        policy=policy,
        faces=faces,
        face_area=face_area,
        face_mean_edge=_mean_edge,
        face_count=face_count,
        vertex_count=vertex_count,
        normal_error_degrees=float(config.auto_normal_error_degrees),
        h_min_factor=float(config.auto_h_min_factor),
        h_max_factor=float(config.auto_h_max_factor),
        target_point_scale=float(config.auto_target_point_scale),
    )
    if auto_sizing is not None:
        raw_density = np.clip(np.asarray(auto_sizing["vertex_density"], dtype=np.float32), min_density, max_density)
    else:
        raw_density = (min_density * np.power(max_density / min_density, signal)).astype(np.float32)
    density, gradation_summary, gradation_debug = _apply_planar_gradation(
        density=raw_density,
        planar=planar,
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        policy=policy,
        proxies=proxies,
        iterations=int(config.planar_gradation_iterations),
        strength=float(config.planar_gradation_strength),
        planar_power=float(config.planar_gradation_planar_power),
        min_edge_weight=float(config.planar_gradation_min_edge_weight),
        max_density_ratio=float(config.planar_gradation_max_density_ratio),
        normal_lift_scale=float(config.planar_gradation_normal_lift_scale),
        metric_tau_factor=float(config.planar_gradation_metric_tau_factor),
        min_normal_gate=float(config.planar_gradation_min_normal_gate),
        use_proxy_boundaries=bool(config.planar_gradation_use_proxy_boundaries),
    )

    face_density = np.mean(density[np.asarray(faces, dtype=np.int64)], axis=1).astype(np.float32)
    face_signal = np.mean(signal[np.asarray(faces, dtype=np.int64)], axis=1).astype(np.float32)
    _write_density_txt(
        density_txt,
        density=density,
        planar=planar,
        detail=detail,
        boundary=boundary,
        uncertain=uncertain,
        signal=signal,
    )
    normal_output: Path | None = None
    vertex_fused_normal = np.zeros((vertex_count, 3), dtype=np.float32)
    vertex_fused_normal_weight = np.zeros((vertex_count,), dtype=np.float32)
    if bool(config.use_fused_normals) and "fused_face_normal" in evidence:
        face_fused = np.asarray(evidence["fused_face_normal"], dtype=np.float32)
        face_fused_weight = np.asarray(evidence.get("fused_normal_weight", np.zeros((face_count,), dtype=np.float32)), dtype=np.float32)
        if face_fused.shape == (face_count, 3) and face_fused_weight.shape[0] == face_count:
            face_conf = np.clip(face_fused_weight / max(float(config.fused_normal_min_weight), 1.0e-6), 0.0, 1.0).astype(np.float32)
            vertex_fused_normal, vertex_fused_normal_weight = _face_to_vertex_normal_mean(
                faces,
                face_fused,
                face_conf,
                face_area,
                vertex_count,
            )
            if np.any(vertex_fused_normal_weight > 0.0):
                _write_normal_txt(normal_txt, normals=vertex_fused_normal, weights=vertex_fused_normal_weight)
                normal_output = normal_txt
    np.savez_compressed(
        density_npz,
        vertex_density=density.astype(np.float32),
        vertex_density_raw=raw_density.astype(np.float32),
        vertex_density_change_ratio=np.divide(
            density,
            np.maximum(raw_density, 1.0e-12),
            out=np.ones_like(density, dtype=np.float32),
            where=raw_density > 0.0,
        ).astype(np.float32),
        vertex_density_signal=signal.astype(np.float32),
        vertex_planar=planar.astype(np.float32),
        vertex_detail=detail.astype(np.float32),
        vertex_boundary=boundary.astype(np.float32),
        vertex_uncertain=uncertain.astype(np.float32),
        face_density=face_density.astype(np.float32),
        face_density_raw=np.mean(raw_density[np.asarray(faces, dtype=np.int64)], axis=1).astype(np.float32),
        face_density_change_ratio=np.divide(
            face_density,
            np.maximum(np.mean(raw_density[np.asarray(faces, dtype=np.int64)], axis=1), 1.0e-12),
            out=np.ones_like(face_density, dtype=np.float32),
            where=np.mean(raw_density[np.asarray(faces, dtype=np.int64)], axis=1) > 0.0,
        ).astype(np.float32),
        face_density_signal=face_signal.astype(np.float32),
        face_planar=np.mean(planar[faces], axis=1).astype(np.float32),
        face_detail=np.max(detail[faces], axis=1).astype(np.float32),
        face_boundary=np.max(boundary[faces], axis=1).astype(np.float32),
        face_uncertain=np.max(uncertain[faces], axis=1).astype(np.float32),
        face_target_h=(
            np.asarray(auto_sizing["face_h"], dtype=np.float32)
            if auto_sizing is not None
            else np.zeros((face_count,), dtype=np.float32)
        ),
        vertex_target_h=(
            np.asarray(auto_sizing["vertex_h"], dtype=np.float32)
            if auto_sizing is not None
            else np.zeros((vertex_count,), dtype=np.float32)
        ),
        vertex_fused_normal=vertex_fused_normal.astype(np.float32),
        vertex_fused_normal_weight=vertex_fused_normal_weight.astype(np.float32),
        mesh_vertices=vertices.astype(np.float32),
        mesh_faces=faces.astype(np.int64),
        **gradation_debug,
    )

    summary = {
        "stageName": "omega_geogram_density_export",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "model_dir": str(model_dir),
            "mesh_path": str(mesh_path),
            "policy_npz": str(policy_npz),
            "proxies_npz": str(proxies_path) if proxies_path.exists() and proxies_path.is_file() else "",
            "evidence_npz": str(evidence_path) if evidence_path.exists() and evidence_path.is_file() else "",
            "output_dir": str(output_dir),
        },
        "vertexCount": vertex_count,
        "faceCount": face_count,
        "stats": {
            "vertex_density": summarize(density),
            "vertex_density_raw": summarize(raw_density),
            "vertex_density_signal": summarize(signal),
            "vertex_planar": summarize(planar),
            "vertex_detail": summarize(detail),
            "vertex_boundary": summarize(boundary),
            "vertex_uncertain": summarize(uncertain),
            "face_density": summarize(face_density),
            "face_density_raw": summarize(np.mean(raw_density[np.asarray(faces, dtype=np.int64)], axis=1).astype(np.float32)),
            "face_density_signal": summarize(face_signal),
        },
        "autoSizing": (
            auto_sizing["summary"]
            if auto_sizing is not None
            else {
                "enabled": False,
                "normalErrorDegrees": float(config.auto_normal_error_degrees),
            }
        ),
        "planarGradation": gradation_summary,
        "outputs": {
            "densityTxt": str(density_txt),
            "normalTxt": str(normal_output) if normal_output is not None else "",
            "densityNpz": str(density_npz),
            "summaryJson": str(summary_json),
        },
        "notes": [
            "High planar weight lowers Geogram density.",
            "Low planarity, detail weight, and boundary weight raise density.",
            "Optional planar gradation smooths log density only across planar-compatible mesh edges and clamps per-vertex change.",
            "Planar gradation uses Geogram's normal-lift scale to stop smoothing across normal breaks in the same metric family as anisotropic CVT.",
            "When auto sizing is enabled, the user normal-error tolerance rescales StableNormal target lengths into h(x), then h(x)^-2 becomes Geogram density.",
            "When fused normals are available from Stage 2, a soft normal sidecar can guide Geogram's 6D normal-lifted metric.",
            "Geogram uses this scalar as vertex mass in CVT sampling and optimization; larger values receive more samples.",
        ],
    }
    _write_json(summary_json, summary)
    progress(f"[geogram-density] Mesh: {mesh_path} ({vertex_count:,} vertices, {face_count:,} faces)")
    progress(f"[geogram-density] Policy: {policy_npz}")
    progress(f"[geogram-density] Wrote: {density_txt}")
    return OmegaGeogramDensityResult(
        density_txt=density_txt,
        normal_txt=normal_output,
        density_npz=density_npz,
        summary_json=summary_json,
        vertex_count=vertex_count,
        face_count=face_count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export OMeGa per-vertex density for Geogram/Vorpalite.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Defaults to latest OMeGa mesh.")
    parser.add_argument("--policy-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/policy.npz.")
    parser.add_argument("--proxies-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/proxies.npz when present.")
    parser.add_argument("--evidence-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/evidence.npz when present.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/geogram_density.")
    parser.add_argument("--density-name", default="omega_geogram_density")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--min-density", type=float, default=0.25)
    parser.add_argument("--max-density", type=float, default=4.0)
    parser.add_argument("--signal-gamma", type=float, default=1.25)
    parser.add_argument("--detail-influence", type=float, default=1.0)
    parser.add_argument("--boundary-influence", type=float, default=0.5)
    parser.add_argument("--uncertain-influence", type=float, default=0.0)
    parser.add_argument(
        "--auto-normal-error-degrees",
        type=float,
        default=0.0,
        help="Enable h(x)-driven density from StableNormal target lengths using this allowed normal error.",
    )
    parser.add_argument("--auto-h-min-factor", type=float, default=0.50)
    parser.add_argument("--auto-h-max-factor", type=float, default=6.0)
    parser.add_argument("--auto-target-point-scale", type=float, default=1.0)
    parser.add_argument("--use-fused-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-normal-blend", type=float, default=0.70)
    parser.add_argument("--fused-normal-min-weight", type=float, default=8.0)
    parser.add_argument("--planar-gradation-iterations", type=int, default=0)
    parser.add_argument("--planar-gradation-strength", type=float, default=0.45)
    parser.add_argument("--planar-gradation-planar-power", type=float, default=2.0)
    parser.add_argument("--planar-gradation-min-edge-weight", type=float, default=0.05)
    parser.add_argument("--planar-gradation-max-density-ratio", type=float, default=1.75)
    parser.add_argument(
        "--planar-gradation-normal-lift-scale",
        type=float,
        default=1.0,
        help="Normal-lift scale used by the same-plane gate. Matches Geogram remesh:anisotropy by default.",
    )
    parser.add_argument(
        "--planar-gradation-metric-tau-factor",
        type=float,
        default=2.0,
        help="Allowed lifted-normal displacement measured in local edge lengths.",
    )
    parser.add_argument("--planar-gradation-min-normal-gate", type=float, default=0.05)
    parser.add_argument("--planar-gradation-use-proxy-boundaries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compute_omega_geogram_density(
        OmegaGeogramDensityConfig(
            model_dir=args.model_dir,
            mesh_path=args.mesh,
            policy_npz=args.policy_npz,
            proxies_npz=args.proxies_npz,
            evidence_npz=args.evidence_npz,
            output_dir=args.output_dir,
            density_name=str(args.density_name),
            iteration=int(args.iteration),
            min_density=float(args.min_density),
            max_density=float(args.max_density),
            signal_gamma=float(args.signal_gamma),
            detail_influence=float(args.detail_influence),
            boundary_influence=float(args.boundary_influence),
            uncertain_influence=float(args.uncertain_influence),
            auto_normal_error_degrees=float(args.auto_normal_error_degrees),
            auto_h_min_factor=float(args.auto_h_min_factor),
            auto_h_max_factor=float(args.auto_h_max_factor),
            auto_target_point_scale=float(args.auto_target_point_scale),
            use_fused_normals=bool(args.use_fused_normals),
            fused_normal_blend=float(args.fused_normal_blend),
            fused_normal_min_weight=float(args.fused_normal_min_weight),
            planar_gradation_iterations=int(args.planar_gradation_iterations),
            planar_gradation_strength=float(args.planar_gradation_strength),
            planar_gradation_planar_power=float(args.planar_gradation_planar_power),
            planar_gradation_min_edge_weight=float(args.planar_gradation_min_edge_weight),
            planar_gradation_max_density_ratio=float(args.planar_gradation_max_density_ratio),
            planar_gradation_normal_lift_scale=float(args.planar_gradation_normal_lift_scale),
            planar_gradation_metric_tau_factor=float(args.planar_gradation_metric_tau_factor),
            planar_gradation_min_normal_gate=float(args.planar_gradation_min_normal_gate),
            planar_gradation_use_proxy_boundaries=bool(args.planar_gradation_use_proxy_boundaries),
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
