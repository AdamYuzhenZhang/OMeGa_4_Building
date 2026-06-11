"""Phase 3 policy fields for redesigned OMeGa local remeshing.

This stage converts full view-normal evidence into continuous face and edge
fields.  It does not modify mesh geometry.  The important distinction is that
stable multiview normal detail may drive local detail preservation, while
inconsistent offset gradients are treated as unexplained foreground/occlusion
evidence and used as protection rather than subdivision pressure.
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
import trimesh

from omega_local.remesh.weights import compute_phase3_weights


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class RemeshPolicyConfig:
    mesh_path: Path
    evidence_npz: Path
    output_dir: Path
    policy_name: str = "policy"
    planarity_rings: int = 2
    support_min: float = 0.08
    planar_score_threshold: float = 0.35
    detail_max_for_planar: float = 0.75
    unexplained_max_for_planar: float = 0.55
    min_tau_a_degrees: float = 5.0
    boundary_score_threshold: float = 0.50
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class RemeshPolicyResult:
    policy_npz: Path
    summary_json: Path
    debug_meshes: dict[str, Path]
    face_count: int
    edge_count: int


@dataclass(frozen=True)
class MeshTopology:
    edge_vertices: np.ndarray
    edge_faces: np.ndarray
    face_edges: np.ndarray
    face_neighbors: list[np.ndarray]
    boundary_edges: np.ndarray
    nonmanifold_edge_count: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summarize(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is not None:
        arr = arr[np.asarray(valid, dtype=bool).reshape(-1)]
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


def _load_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh object from {mesh_path}: {type(mesh)!r}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces in {mesh_path}, got shape {faces.shape}")
    return vertices, faces


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw_normals, axis=1)
    normals = np.divide(
        raw_normals,
        np.maximum(double_area[:, None], 1e-12),
        out=np.zeros_like(raw_normals, dtype=np.float64),
    )
    centers = np.mean(triangles, axis=1)
    areas = 0.5 * double_area
    edge_lengths = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2)
    return centers, normals, areas, np.mean(edge_lengths, axis=1)


def build_mesh_topology(faces: np.ndarray) -> MeshTopology:
    face_count = int(faces.shape[0])
    edge_records: dict[tuple[int, int], list[int]] = {}
    face_edges = np.full((face_count, 3), -1, dtype=np.int64)
    edge_vertices_list: list[tuple[int, int]] = []
    edge_faces_list: list[list[int]] = []
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for local_id, (a, b) in enumerate(((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0])))):
            key = (a, b) if a < b else (b, a)
            edge_id = edge_records.get(key)
            if edge_id is None:
                edge_records[key] = [len(edge_vertices_list)]
                edge_vertices_list.append(key)
                edge_faces_list.append([face_id])
                face_edges[face_id, local_id] = len(edge_vertices_list) - 1
            else:
                idx = int(edge_id[0])
                edge_faces_list[idx].append(face_id)
                face_edges[face_id, local_id] = idx

    edge_count = len(edge_vertices_list)
    edge_faces = np.full((edge_count, 2), -1, dtype=np.int64)
    nonmanifold_count = 0
    neighbor_sets: list[set[int]] = [set() for _ in range(face_count)]
    for edge_id, incident in enumerate(edge_faces_list):
        if len(incident) == 1:
            edge_faces[edge_id, 0] = int(incident[0])
        elif len(incident) == 2:
            a, b = int(incident[0]), int(incident[1])
            edge_faces[edge_id] = (a, b)
            neighbor_sets[a].add(b)
            neighbor_sets[b].add(a)
        else:
            nonmanifold_count += 1
            kept = [int(incident[0]), int(incident[1])]
            edge_faces[edge_id] = kept
            for a in incident:
                for b in incident:
                    if a != b:
                        neighbor_sets[int(a)].add(int(b))
    face_neighbors = [np.asarray(sorted(values), dtype=np.int64) for values in neighbor_sets]
    return MeshTopology(
        edge_vertices=np.asarray(edge_vertices_list, dtype=np.int64),
        edge_faces=edge_faces,
        face_edges=face_edges,
        face_neighbors=face_neighbors,
        boundary_edges=edge_faces[:, 1] < 0,
        nonmanifold_edge_count=int(nonmanifold_count),
    )


def _load_evidence(path: Path, face_count: int) -> dict[str, np.ndarray]:
    data = np.load(path)
    evidence = {key: data[key] for key in data.files}
    required = [
        "face_support",
        "face_view_disagreement",
        "face_normal_kappa",
        "face_detail_target_length",
        "face_position_tolerance",
        "face_tau_a",
        "offset_high_gradient_pixel_count",
        "offset_normal_gradient_max",
        "normal_count",
        "view_count",
    ]
    missing = [key for key in required if key not in evidence]
    if missing:
        raise KeyError(f"Evidence file is missing required Phase 2 arrays: {missing}")
    for key in required:
        arr = np.asarray(evidence[key])
        if arr.ndim > 0 and arr.shape[0] != face_count:
            raise ValueError(f"Evidence array {key!r} has {arr.shape[0]} values, expected {face_count}.")
    return evidence


def _robust_scale(values: np.ndarray, fallback: float = 1.0) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(fallback)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    return max(1.4826 * mad, float(fallback), 1e-6)


def _normalize_by_quantile(values: np.ndarray, valid: np.ndarray, q: float = 0.95) -> np.ndarray:
    out = np.zeros_like(np.asarray(values, dtype=np.float32))
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return out
    scale = float(np.quantile(np.asarray(values, dtype=np.float64)[mask], float(q)))
    scale = max(scale, 1e-6)
    out[mask] = np.clip(np.asarray(values, dtype=np.float32)[mask] / scale, 0.0, 1.0)
    return out.astype(np.float32)


def _inverse_normalize_by_quantiles(values: np.ndarray, valid: np.ndarray, q_low: float = 0.10, q_high: float = 0.90) -> np.ndarray:
    out = np.zeros_like(np.asarray(values, dtype=np.float32))
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values) & (np.asarray(values) > 0.0)
    if not np.any(mask):
        return out
    arr = np.asarray(values, dtype=np.float64)[mask]
    lo = float(np.quantile(arr, float(q_low)))
    hi = float(np.quantile(arr, float(q_high)))
    if hi <= lo + 1e-9:
        return out
    score = 1.0 - (np.asarray(values, dtype=np.float32)[mask] - lo) / (hi - lo)
    out[mask] = np.clip(score, 0.0, 1.0)
    return out.astype(np.float32)


def _fit_two_gaussian_mixture(values: np.ndarray, weights: np.ndarray, iterations: int = 40) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    posterior = np.zeros(values.shape[0], dtype=np.float32)
    if np.count_nonzero(valid) < 20:
        if np.any(np.isfinite(values)):
            threshold = float(np.quantile(values[np.isfinite(values)], 0.75))
            posterior[np.isfinite(values)] = (values[np.isfinite(values)] >= threshold).astype(np.float32)
        return posterior, {"mode": "quantile_fallback", "lowMean": 0.0, "highMean": 0.0, "highWeight": 0.0}

    x = values[valid]
    w = weights[valid]
    w = w / max(float(np.sum(w)), 1e-12)
    mu = np.asarray([np.quantile(x, 0.30), np.quantile(x, 0.80)], dtype=np.float64)
    sigma = np.full((2,), max(float(np.std(x)), 1e-3), dtype=np.float64)
    pi = np.asarray([0.55, 0.45], dtype=np.float64)
    for _ in range(max(int(iterations), 1)):
        density = []
        for comp in range(2):
            var = max(float(sigma[comp] * sigma[comp]), 1e-8)
            density.append(pi[comp] * np.exp(-0.5 * (x - mu[comp]) ** 2 / var) / math.sqrt(2.0 * math.pi * var))
        dens = np.stack(density, axis=1)
        resp = dens / np.maximum(np.sum(dens, axis=1, keepdims=True), 1e-12)
        weighted_resp = resp * w[:, None]
        comp_mass = np.maximum(np.sum(weighted_resp, axis=0), 1e-12)
        pi = comp_mass / np.sum(comp_mass)
        mu = np.sum(weighted_resp * x[:, None], axis=0) / comp_mass
        sigma = np.sqrt(np.maximum(np.sum(weighted_resp * (x[:, None] - mu[None, :]) ** 2, axis=0) / comp_mass, 1e-8))
    high = int(np.argmax(mu))
    posterior[valid] = resp[:, high].astype(np.float32)
    return posterior, {
        "mode": "weighted_two_gaussian_em",
        "lowMean": float(np.min(mu)),
        "highMean": float(np.max(mu)),
        "lowSigma": float(sigma[int(np.argmin(mu))]),
        "highSigma": float(sigma[high]),
        "highWeight": float(pi[high]),
    }


def _local_planarity(
    *,
    centers: np.ndarray,
    areas: np.ndarray,
    support: np.ndarray,
    position_tolerance: np.ndarray,
    neighbors: list[np.ndarray],
    rings: int,
) -> tuple[np.ndarray, np.ndarray]:
    face_count = centers.shape[0]
    planarity = np.zeros(face_count, dtype=np.float32)
    residual = np.zeros(face_count, dtype=np.float32)
    area = np.asarray(areas, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64).clip(0.0, 1.0)
    eps_pos = np.maximum(np.asarray(position_tolerance, dtype=np.float64), 1e-6)

    for face_id in range(face_count):
        visited = {face_id}
        frontier = {face_id}
        for _ in range(max(int(rings), 0)):
            next_frontier: set[int] = set()
            for item in frontier:
                next_frontier.update(int(n) for n in neighbors[item])
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        ids = np.fromiter(visited, dtype=np.int64)
        if ids.size < 3:
            planarity[face_id] = 0.0
            continue
        weights = area[ids] * np.maximum(support[ids], 0.05)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            planarity[face_id] = 0.0
            continue
        centroid = np.sum(centers[ids] * weights[:, None], axis=0) / weight_sum
        centered = centers[ids] - centroid
        cov = (centered * weights[:, None]).T @ centered / weight_sum
        try:
            eigenvalues = np.linalg.eigvalsh(cov)
        except np.linalg.LinAlgError:
            planarity[face_id] = 0.0
            continue
        r_geo = math.sqrt(max(float(eigenvalues[0]), 0.0))
        residual[face_id] = float(r_geo)
        planarity[face_id] = float(math.exp(-(r_geo * r_geo) / max(float(eps_pos[face_id] * eps_pos[face_id]), 1e-12)))
    return planarity, residual


def _edge_arrays(
    *,
    topology: MeshTopology,
    face_normals: np.ndarray,
    face_tau_a: np.ndarray,
    face_support: np.ndarray,
    detail: np.ndarray,
    unexplained: np.ndarray,
    planar_score: np.ndarray,
    target_length: np.ndarray,
    position_tolerance: np.ndarray,
    min_tau_a_degrees: float,
) -> dict[str, np.ndarray]:
    edge_faces = topology.edge_faces
    face_a = edge_faces[:, 0]
    face_b = edge_faces[:, 1]
    has_two = face_b >= 0
    edge_count = edge_faces.shape[0]
    angle = np.zeros(edge_count, dtype=np.float32)
    tau_a = np.zeros(edge_count, dtype=np.float32)
    if np.any(has_two):
        a = face_a[has_two]
        b = face_b[has_two]
        dot = np.abs(np.sum(face_normals[a] * face_normals[b], axis=1)).clip(0.0, 1.0)
        angle[has_two] = np.arccos(dot).astype(np.float32)
        tau_a[has_two] = np.maximum(0.5 * (face_tau_a[a] + face_tau_a[b]), math.radians(float(min_tau_a_degrees))).astype(np.float32)
    boundary = topology.boundary_edges
    tau_a[boundary] = np.maximum(face_tau_a[face_a[boundary]], math.radians(float(min_tau_a_degrees))).astype(np.float32)
    mesh_feature = np.zeros(edge_count, dtype=np.float32)
    mesh_feature[has_two] = (1.0 - np.exp(-(angle[has_two] * angle[has_two]) / np.maximum(tau_a[has_two] * tau_a[has_two], 1e-12))).astype(np.float32)

    edge_support = np.zeros(edge_count, dtype=np.float32)
    edge_detail = np.zeros(edge_count, dtype=np.float32)
    edge_unexplained = np.zeros(edge_count, dtype=np.float32)
    edge_planar = np.zeros(edge_count, dtype=np.float32)
    edge_planar_jump = np.zeros(edge_count, dtype=np.float32)
    edge_detail_jump = np.zeros(edge_count, dtype=np.float32)
    edge_target = np.zeros(edge_count, dtype=np.float32)
    edge_eps = np.zeros(edge_count, dtype=np.float32)
    edge_support[boundary] = face_support[face_a[boundary]]
    edge_detail[boundary] = detail[face_a[boundary]]
    edge_unexplained[boundary] = unexplained[face_a[boundary]]
    edge_planar[boundary] = planar_score[face_a[boundary]]
    edge_target[boundary] = target_length[face_a[boundary]]
    edge_eps[boundary] = position_tolerance[face_a[boundary]]
    if np.any(has_two):
        a = face_a[has_two]
        b = face_b[has_two]
        edge_support[has_two] = np.minimum(face_support[a], face_support[b])
        edge_detail[has_two] = np.maximum(detail[a], detail[b])
        edge_unexplained[has_two] = np.maximum(unexplained[a], unexplained[b])
        edge_planar[has_two] = np.minimum(planar_score[a], planar_score[b])
        edge_planar_jump[has_two] = np.abs(planar_score[a] - planar_score[b])
        edge_detail_jump[has_two] = np.abs(detail[a] - detail[b])
        edge_target[has_two] = np.minimum(target_length[a], target_length[b])
        edge_eps[has_two] = np.minimum(position_tolerance[a], position_tolerance[b])

    # Boundary weight should mark likely feature/seam transitions, not every
    # face that merely contains stable detail.  Mesh boundaries are exported
    # separately as topology constraints and are not folded into this score.
    boundary_score = 1.0 - (
        (1.0 - mesh_feature)
        * (1.0 - edge_detail_jump)
        * (1.0 - edge_planar_jump)
    )
    boundary_score[boundary] = 0.0
    return {
        "edge_dihedral_angle": angle,
        "edge_tau_a": tau_a,
        "edge_support": edge_support,
        "edge_feature_score": mesh_feature.astype(np.float32),
        "edge_detail_score": edge_detail.astype(np.float32),
        "edge_unexplained_offset": edge_unexplained.astype(np.float32),
        "edge_planar_score": edge_planar.astype(np.float32),
        "edge_planar_jump": edge_planar_jump.astype(np.float32),
        "edge_detail_jump": edge_detail_jump.astype(np.float32),
        "edge_boundary_score": np.clip(boundary_score, 0.0, 1.0).astype(np.float32),
        "edge_preliminary_target_length": edge_target.astype(np.float32),
        "edge_position_tolerance": edge_eps.astype(np.float32),
        "edge_is_mesh_boundary": boundary.astype(np.uint8),
    }


def _face_max_from_edges(face_edges: np.ndarray, edge_values: np.ndarray) -> np.ndarray:
    out = np.zeros(face_edges.shape[0], dtype=np.float32)
    for local_id in range(3):
        values = edge_values[face_edges[:, local_id]]
        out = np.maximum(out, values.astype(np.float32))
    return out


def _colorize_scalar(values: np.ndarray, valid: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    normalized = np.clip((values - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    mapped = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and values.ndim == 1 and rgb.shape[1] == 1:
        rgb = rgb[:, 0, :]
    rgb[~valid] = np.array([38, 38, 38], dtype=np.uint8)
    return rgb


def _scalar_range(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.asarray(valid, dtype=bool) & np.isfinite(arr)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(finite, 0.02))
    hi = float(np.quantile(finite, 0.98))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    return lo, hi


def _write_face_color_ply(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate(
        [np.asarray(colors, dtype=np.uint8), np.full((faces.shape[0], 1), 255, dtype=np.uint8)],
        axis=1,
    )
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def write_policy_debug_meshes(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    policy: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes" / "policy"
    debug_dir.mkdir(parents=True, exist_ok=True)
    normal_valid = policy["normal_count"] > 0
    support_valid = policy["view_count"] > 0
    specs = {
        "face_support": (policy["face_support"], support_valid, 0.0, 1.0),
        "face_detail_posterior": (policy["face_detail_posterior"], normal_valid, 0.0, 1.0),
        "face_unexplained_offset": (policy["face_unexplained_offset"], policy["offset_high_gradient_pixel_count"] > 0, 0.0, 1.0),
        "face_target_length_detail_score": (policy["face_target_length_detail_score"], normal_valid, 0.0, 1.0),
        "face_directional_target_detail": (policy["face_directional_target_detail"], normal_valid, 0.0, 1.0),
        "face_low_gradient_probability": (policy["face_low_gradient_probability"], normal_valid, 0.0, 1.0),
        "face_planarity_geo": (policy["face_planarity_geo"], support_valid, 0.0, 1.0),
        "face_planar_score": (policy["face_planar_score"], support_valid, 0.0, 1.0),
        "face_protect_detail": (policy["face_protect_detail"], normal_valid, 0.0, 1.0),
        "face_protect_missing_geometry": (policy["face_protect_missing_geometry"], policy["offset_high_gradient_pixel_count"] > 0, 0.0, 1.0),
        "face_protect_disagreement": (policy["face_protect_disagreement"], normal_valid, 0.0, 1.0),
        "face_protect_boundary": (policy["face_protect_boundary"], support_valid, 0.0, 1.0),
        "face_protect_score": (policy["face_protect_score"], support_valid, 0.0, 1.0),
        "face_boundary_score": (policy["face_boundary_score"], support_valid, 0.0, 1.0),
        "face_weight_planar": (policy["face_weight_planar"], support_valid, 0.0, 1.0),
        "face_weight_detail": (policy["face_weight_detail"], normal_valid, 0.0, 1.0),
        "face_weight_boundary": (policy["face_weight_boundary"], support_valid, 0.0, 1.0),
        "face_weight_uncertain": (policy["face_weight_uncertain"], support_valid, 0.0, 1.0),
        "face_offset_detail_pressure": (policy["face_offset_detail_pressure"], support_valid, 0.0, 1.0),
        "face_residual_unexplained_offset": (policy["face_residual_unexplained_offset"], support_valid, 0.0, 1.0),
        "face_stable_explanation": (policy["face_stable_explanation"], support_valid, 0.0, 1.0),
        "face_boundary_from_edges": (policy["face_boundary_from_edges"], support_valid, 0.0, 1.0),
        "face_low_support_uncertainty": (policy["face_low_support_uncertainty"], support_valid, 0.0, 1.0),
        "face_disagreement_uncertainty": (policy["face_disagreement_uncertainty"], normal_valid, 0.0, 1.0),
        "face_normal_kappa": (
            policy["face_normal_kappa"],
            normal_valid,
            *_scalar_range(policy["face_normal_kappa"], normal_valid),
        ),
        "face_detail_target_length": (
            policy["face_detail_target_length"],
            normal_valid,
            *_scalar_range(policy["face_detail_target_length"], normal_valid),
        ),
        "face_planarity_residual": (
            policy["face_planarity_residual"],
            support_valid,
            *_scalar_range(policy["face_planarity_residual"], support_valid),
        ),
    }
    paths: dict[str, Path] = {}
    for name, (values, valid, vmin, vmax) in specs.items():
        path = debug_dir / f"{name}.ply"
        colors = _colorize_scalar(values, valid, vmin=float(vmin), vmax=float(vmax))
        _write_face_color_ply(vertices, faces, colors, path)
        paths[name] = path
    return paths


def compute_remesh_policy(config: RemeshPolicyConfig, progress: ProgressFn | None = None) -> RemeshPolicyResult:
    progress = progress or _progress_default
    mesh_path = config.mesh_path.expanduser().resolve()
    evidence_npz = config.evidence_npz.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_npz = output_dir / f"{config.policy_name}.npz"
    summary_json = output_dir / f"{config.policy_name}_summary.json"
    if policy_npz.exists() and not config.overwrite:
        raise FileExistsError(f"Policy output exists: {policy_npz}. Re-run with overwrite=True.")

    vertices, faces = _load_mesh_arrays(mesh_path)
    centers, mesh_normals, face_area, mean_edge_length = _face_geometry(vertices, faces)
    topology = build_mesh_topology(faces)
    evidence = _load_evidence(evidence_npz, int(faces.shape[0]))
    face_count = int(faces.shape[0])
    progress(f"[policy] Mesh: {mesh_path} ({vertices.shape[0]:,} vertices, {face_count:,} faces)")
    progress(f"[policy] Evidence: {evidence_npz}")
    progress(f"[policy] Edges: {topology.edge_vertices.shape[0]:,} boundary={np.count_nonzero(topology.boundary_edges):,}")

    support = np.asarray(evidence["face_support"], dtype=np.float32).clip(0.0, 1.0)
    view_count = np.asarray(evidence["view_count"], dtype=np.int32)
    normal_count = np.asarray(evidence["normal_count"], dtype=np.int32)
    kappa = np.asarray(evidence["face_normal_kappa"], dtype=np.float32)
    disagreement = np.asarray(evidence["face_view_disagreement"], dtype=np.float32)
    target_length = np.asarray(evidence["face_detail_target_length"], dtype=np.float32)
    target_length = np.where(target_length > 0.0, target_length, mean_edge_length.astype(np.float32))
    position_tolerance = np.asarray(evidence["face_position_tolerance"], dtype=np.float32)
    position_tolerance = np.where(position_tolerance > 0.0, position_tolerance, (0.25 * mean_edge_length).astype(np.float32))
    tau_a = np.asarray(evidence["face_tau_a"], dtype=np.float32)
    tau_a = np.where(tau_a > 0.0, tau_a, math.radians(float(config.min_tau_a_degrees))).astype(np.float32)
    offset_count = np.asarray(evidence["offset_high_gradient_pixel_count"], dtype=np.int64)
    offset_gradient = np.asarray(evidence["offset_normal_gradient_max"], dtype=np.float32)
    direction_confidence = np.asarray(evidence.get("face_direction_confidence", np.zeros(face_count)), dtype=np.float32).clip(0.0, 1.0)

    normal_valid = normal_count > 0
    detail_weights = support.astype(np.float64)
    tau_g_median = float(np.median(np.asarray(evidence.get("face_tau_g", np.zeros(face_count)), dtype=np.float32)[normal_valid])) if np.any(normal_valid) else 1e-3
    mixture_input = np.log(np.maximum(kappa, 0.0) + max(tau_g_median, 1e-6))
    raw_detail_posterior, mixture_summary = _fit_two_gaussian_mixture(mixture_input, detail_weights)
    disagreement_scale = _robust_scale(disagreement[normal_valid], fallback=float(np.quantile(disagreement[normal_valid], 0.75)) if np.any(normal_valid) else 1.0)
    agreement = np.exp(-(disagreement * disagreement) / max(disagreement_scale * disagreement_scale, 1e-12)).astype(np.float32)
    offset_strength = _normalize_by_quantile(offset_gradient, offset_count > 0, q=0.95)
    target_length_detail = _inverse_normalize_by_quantiles(target_length, normal_valid, q_low=0.10, q_high=0.90)
    directional_target_detail = target_length_detail * (0.25 + 0.75 * direction_confidence)
    stable_detail = (
        0.65 * raw_detail_posterior
        + 0.35 * directional_target_detail
    ) * np.sqrt(np.clip(support, 0.0, 1.0)) * agreement
    offset_detail = np.zeros_like(offset_strength, dtype=np.float32)
    unexplained_offset = offset_strength * (1.0 - support * agreement)
    detail_posterior = stable_detail.clip(0.0, 1.0).astype(np.float32)
    unexplained_offset = np.clip(unexplained_offset, 0.0, 1.0).astype(np.float32)
    low_gradient_probability = (1.0 - raw_detail_posterior).clip(0.0, 1.0).astype(np.float32)

    progress("[policy] Computing local geometric planarity...")
    planarity_geo, planarity_residual = _local_planarity(
        centers=centers,
        areas=face_area,
        support=support,
        position_tolerance=position_tolerance,
        neighbors=topology.face_neighbors,
        rings=int(config.planarity_rings),
    )
    planar_base_score = (support * planarity_geo).clip(0.0, 1.0).astype(np.float32)
    planar_score = planar_base_score
    disagreement_score = _normalize_by_quantile(disagreement, normal_valid, q=0.95)

    edge_policy = _edge_arrays(
        topology=topology,
        face_normals=mesh_normals,
        face_tau_a=tau_a,
        face_support=support,
        detail=detail_posterior,
        unexplained=unexplained_offset,
        planar_score=planar_score,
        target_length=target_length,
        position_tolerance=position_tolerance,
        min_tau_a_degrees=float(config.min_tau_a_degrees),
    )
    face_boundary_score = _face_max_from_edges(topology.face_edges, edge_policy["edge_boundary_score"])
    protect_detail = detail_posterior.astype(np.float32)
    protect_missing_geometry = unexplained_offset.astype(np.float32)
    protect_disagreement = disagreement_score.astype(np.float32)
    protect_boundary = face_boundary_score.astype(np.float32)
    protect_score = np.maximum.reduce([
        protect_detail,
        protect_disagreement,
        protect_boundary,
    ]).astype(np.float32)
    planar_candidate = (
        (support >= float(config.support_min))
        & (planar_score >= float(config.planar_score_threshold))
        & (detail_posterior <= float(config.detail_max_for_planar))
    )

    policy: dict[str, np.ndarray] = {
        "face_centers": centers.astype(np.float32),
        "mesh_face_normals": mesh_normals.astype(np.float32),
        "face_area": face_area.astype(np.float32),
        "face_mean_edge_length": mean_edge_length.astype(np.float32),
        "view_count": view_count,
        "normal_count": normal_count,
        "offset_high_gradient_pixel_count": offset_count,
        "face_support": support.astype(np.float32),
        "face_view_disagreement": disagreement.astype(np.float32),
        "face_normal_kappa": kappa.astype(np.float32),
        "face_raw_detail_posterior": raw_detail_posterior.astype(np.float32),
        "face_target_length_detail_score": target_length_detail.astype(np.float32),
        "face_directional_target_detail": directional_target_detail.astype(np.float32),
        "face_stable_detail": stable_detail.astype(np.float32),
        "face_detail_posterior": detail_posterior,
        "face_low_gradient_probability": low_gradient_probability,
        "face_unexplained_offset": unexplained_offset,
        "face_missing_geometry_likelihood": unexplained_offset,
        "face_offset_detail": offset_detail.astype(np.float32),
        "face_offset_strength": offset_strength.astype(np.float32),
        "face_planarity_geo": planarity_geo.astype(np.float32),
        "face_planarity_residual": planarity_residual.astype(np.float32),
        "face_planar_base_score": planar_base_score,
        "face_planar_score": planar_score,
        "face_planar_candidate": planar_candidate.astype(np.uint8),
        "face_protect_detail": protect_detail,
        "face_protect_missing_geometry": protect_missing_geometry,
        "face_protect_disagreement": protect_disagreement,
        "face_protect_boundary": protect_boundary,
        "face_protect_score": protect_score.astype(np.float32),
        "face_detail_target_length": target_length.astype(np.float32),
        "face_position_tolerance": position_tolerance.astype(np.float32),
        "face_tau_a": tau_a.astype(np.float32),
        "face_boundary_score": face_boundary_score.astype(np.float32),
        "face_direction_tensor": np.asarray(evidence.get("face_direction_tensor", np.zeros((face_count, 3))), dtype=np.float32),
        "face_direction_confidence": direction_confidence.astype(np.float32),
        "edge_vertices": topology.edge_vertices.astype(np.int64),
        "edge_faces": topology.edge_faces.astype(np.int64),
        "face_edges": topology.face_edges.astype(np.int64),
        **edge_policy,
    }
    base_weights, weight_summary = compute_phase3_weights(
        policy=policy,
        faces=faces,
        vertex_count=int(vertices.shape[0]),
        proxies=None,
    )
    policy.update(base_weights)
    np.savez_compressed(policy_npz, **policy)

    debug_meshes: dict[str, Path] = {}
    if config.write_debug_meshes:
        debug_meshes = write_policy_debug_meshes(vertices=vertices, faces=faces, policy=policy, out_dir=output_dir)

    summary = {
        "stageName": "omega_remesh_policy",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "mesh_path": str(mesh_path),
            "evidence_npz": str(evidence_npz),
            "output_dir": str(output_dir),
        },
        "inputs": {
            "meshPath": str(mesh_path),
            "evidenceNpz": str(evidence_npz),
        },
        "mesh": {
            "vertexCount": int(vertices.shape[0]),
            "faceCount": face_count,
            "edgeCount": int(topology.edge_vertices.shape[0]),
            "boundaryEdgeCount": int(np.count_nonzero(topology.boundary_edges)),
            "nonmanifoldEdgeCount": int(topology.nonmanifold_edge_count),
        },
        "mixture": mixture_summary,
        "counts": {
            "supportedFaces": int(np.count_nonzero(support >= float(config.support_min))),
            "normalFaces": int(np.count_nonzero(normal_valid)),
            "planarCandidateFaces": int(np.count_nonzero(planar_candidate)),
            "detailFaces": int(np.count_nonzero(detail_posterior >= 0.50)),
            "unexplainedOffsetFaces": int(np.count_nonzero(unexplained_offset >= 0.50)),
            "strongBoundaryEdges": int(np.count_nonzero(edge_policy["edge_boundary_score"] >= float(config.boundary_score_threshold))),
            "highPlanarWeightFaces": int(np.count_nonzero(policy["face_weight_planar"] >= 0.50)),
            "highDetailWeightFaces": int(np.count_nonzero(policy["face_weight_detail"] >= 0.50)),
            "highBoundaryWeightFaces": int(np.count_nonzero(policy["face_weight_boundary"] >= 0.50)),
            "highUncertaintyWeightFaces": int(np.count_nonzero(policy["face_weight_uncertain"] >= 0.50)),
        },
        "stats": {
            "face_support": _summarize(support, view_count > 0),
            "face_normal_kappa": _summarize(kappa, normal_valid),
            "face_view_disagreement": _summarize(disagreement, normal_valid),
            "face_detail_posterior": _summarize(detail_posterior, normal_valid),
            "face_target_length_detail_score": _summarize(target_length_detail, normal_valid),
            "face_directional_target_detail": _summarize(directional_target_detail, normal_valid),
            "face_unexplained_offset": _summarize(unexplained_offset, offset_count > 0),
            "face_planarity_geo": _summarize(planarity_geo, view_count > 0),
            "face_planar_score": _summarize(planar_score, view_count > 0),
            "face_protect_detail": _summarize(protect_detail, normal_valid),
            "face_protect_missing_geometry": _summarize(protect_missing_geometry, offset_count > 0),
            "face_protect_disagreement": _summarize(protect_disagreement, normal_valid),
            "face_protect_boundary": _summarize(protect_boundary, view_count > 0),
            "face_protect_score": _summarize(protect_score, view_count > 0),
            "face_weight_planar": _summarize(policy["face_weight_planar"], view_count > 0),
            "face_weight_detail": _summarize(policy["face_weight_detail"], normal_valid),
            "face_weight_boundary": _summarize(policy["face_weight_boundary"], view_count > 0),
            "face_weight_uncertain": _summarize(policy["face_weight_uncertain"], view_count > 0),
            "face_offset_detail_pressure": _summarize(policy["face_offset_detail_pressure"], view_count > 0),
            "face_residual_unexplained_offset": _summarize(policy["face_residual_unexplained_offset"], view_count > 0),
            "face_stable_explanation": _summarize(policy["face_stable_explanation"], view_count > 0),
            "face_boundary_from_edges": _summarize(policy["face_boundary_from_edges"], view_count > 0),
            "face_low_support_uncertainty": _summarize(policy["face_low_support_uncertainty"], view_count > 0),
            "face_disagreement_uncertainty": _summarize(policy["face_disagreement_uncertainty"], normal_valid),
            "edge_weight_planar": _summarize(policy["edge_weight_planar"]),
            "edge_weight_detail": _summarize(policy["edge_weight_detail"]),
            "edge_weight_boundary": _summarize(policy["edge_weight_boundary"]),
            "edge_weight_uncertain": _summarize(policy["edge_weight_uncertain"]),
            "edge_boundary_score": _summarize(edge_policy["edge_boundary_score"]),
            "edge_feature_score": _summarize(edge_policy["edge_feature_score"]),
        },
        "weights": {
            "finalizedWithProxies": bool(weight_summary.finalized_with_proxies),
            "stats": weight_summary.stats,
        },
        "outputs": {
            "policyNpz": str(policy_npz),
            "summaryJson": str(summary_json),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
        "notes": [
            "Stable multiview normal gradients, short target length, and direction confidence become detail evidence.",
            "Offset gradients never increase face_detail_posterior in this policy.",
            "Offset gradients with weak support or high disagreement become face_missing_geometry_likelihood / face_unexplained_offset.",
            "Unexplained offset is exported separately and does not suppress planar score or planar candidates.",
            "The combined face_protect_score excludes missing-geometry protection; use face_protect_missing_geometry separately.",
            "Phase 3 writes face/edge/vertex planar, detail, boundary, and uncertainty weights into policy.npz.",
            "Offset evidence that coincides with stable detail becomes face_offset_detail_pressure; only the residual offset contributes to uncertainty.",
            "Uncertainty is decomposed into unexplained offset, disagreement uncertainty, and low-support uncertainty after stable planar/detail explanations are considered.",
            "Boundary weight is edge-centric; face_weight_boundary is an averaged diagnostic projection, not the hard operation boundary.",
            "The combined Phase 3 runner finalizes boundary weights again after planar proxies are available.",
            "This stage writes policy arrays only; no topology or geometry is changed.",
        ],
    }
    _write_json(summary_json, summary)
    progress(f"[policy] Wrote {policy_npz}")
    progress(f"[policy] Wrote {summary_json}")
    return RemeshPolicyResult(
        policy_npz=policy_npz,
        summary_json=summary_json,
        debug_meshes=debug_meshes,
        face_count=face_count,
        edge_count=int(topology.edge_vertices.shape[0]),
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


def _config_from_args(args: argparse.Namespace) -> RemeshPolicyConfig:
    model_dir = args.model_dir.expanduser().resolve()
    mesh_arg = _optional_path(args.mesh)
    evidence_arg = _optional_path(args.evidence_npz)
    output_arg = _optional_path(args.output_dir)
    mesh_path = _existing_file(
        _resolve_path(mesh_arg, model_dir) if mesh_arg is not None else model_dir / "remesh" / "local" / "preclean_mesh.ply",
        "Mesh",
    )
    evidence_npz = _existing_file(
        _resolve_path(evidence_arg, model_dir) if evidence_arg is not None else model_dir / "remesh" / "local" / "evidence.npz",
        "Evidence npz",
    )
    output_dir = output_arg.expanduser().resolve() if output_arg is not None else model_dir / "remesh" / "local"
    return RemeshPolicyConfig(
        mesh_path=mesh_path,
        evidence_npz=evidence_npz,
        output_dir=output_dir,
        policy_name=str(args.policy_name),
        planarity_rings=int(args.planarity_rings),
        support_min=float(args.support_min),
        planar_score_threshold=float(args.planar_score_threshold),
        detail_max_for_planar=float(args.detail_max_for_planar),
        unexplained_max_for_planar=float(args.unexplained_max_for_planar),
        min_tau_a_degrees=float(args.min_tau_a_degrees),
        boundary_score_threshold=float(args.boundary_score_threshold),
        write_debug_meshes=not bool(args.no_debug_meshes),
        overwrite=bool(args.overwrite),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Phase 3 remesh policy fields from full view-normal evidence.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/preclean_mesh.ply.")
    parser.add_argument("--evidence-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/evidence.npz.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/local.")
    parser.add_argument("--policy-name", default="policy")
    parser.add_argument("--planarity-rings", type=int, default=2)
    parser.add_argument("--support-min", type=float, default=0.08)
    parser.add_argument("--planar-score-threshold", type=float, default=0.35)
    parser.add_argument("--detail-max-for-planar", type=float, default=0.75)
    parser.add_argument("--unexplained-max-for-planar", type=float, default=0.55)
    parser.add_argument("--min-tau-a-degrees", type=float, default=5.0)
    parser.add_argument("--boundary-score-threshold", type=float, default=0.50)
    parser.add_argument("--no-debug-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compute_remesh_policy(_config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
