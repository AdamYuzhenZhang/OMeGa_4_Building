"""Phase 4A dry-run local remesh operation proposals.

This module does not change topology.  It converts Phase 3 policy/proxy fields
into collapse, split, and flip proposal pressure, then evaluates detailed gates
for the strongest candidates so the behavior can be inspected before the real
mutating remesher is enabled.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
    write_face_scalar_mesh,
)
from omega_local.remesh.policy import build_mesh_topology


ProgressFn = Callable[[str], None]


REJECT_CODE_TO_NAME = {
    0: "not_evaluated",
    1: "accepted",
    2: "reject_boundary",
    3: "reject_density",
    4: "reject_topology",
    5: "reject_fit",
    6: "reject_plane",
    7: "reject_normal",
    8: "reject_missing_geometry_protection",
    9: "reject_quality",
    10: "reject_not_improved",
}
REJECT_NAME_TO_CODE = {name: code for code, name in REJECT_CODE_TO_NAME.items()}


@dataclass(frozen=True)
class OperationProposalConfig:
    mesh_path: Path
    policy_npz: Path
    proxies_npz: Path
    output_dir: Path
    proposals_name: str = "operation_proposals"
    proxies_json: Path | None = None
    support_threshold: float = 0.08
    planar_threshold: float = 0.35
    boundary_threshold: float = 0.50
    detail_collapse_threshold: float = 0.35
    missing_geometry_protect_threshold: float = 0.70
    alpha_collapse: float = 0.75
    alpha_planar_collapse: float = 1.50
    alpha_split: float = 1.50
    q_min_hard: float = 0.03
    q_min_soft: float = 0.12
    max_detailed_collapses: int = 50000
    max_detailed_splits: int = 50000
    max_detailed_flips: int = 50000
    max_jsonl_rows_per_type: int = 20000
    enable_split_proposals: bool = False
    planar_area_length_factor: float = 0.20
    planar_boundary_length_factor: float = 6.0
    scene_max_length_factor: float = 12.0
    proxy_min_length_factor: float = 1.5
    proxy_confidence_floor: float = 0.05
    lambda_proxy: float = 1.0
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class OperationProposalResult:
    proposals_npz: Path
    operations_jsonl: Path
    summary_json: Path
    debug_meshes: dict[str, Path]
    edge_count: int


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


def _json_float(value: Any, fallback: float = 1.0e9) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return out if math.isfinite(out) else float(fallback)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _load_proxy_entries(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists() or path.is_dir():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("proxies", []) if isinstance(payload, dict) else []
    return {int(entry["id"]): dict(entry) for entry in entries if "id" in entry}


def _required(data: dict[str, np.ndarray], keys: list[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} is missing required arrays: {missing}")


def _safe_positive(values: np.ndarray, fallback: np.ndarray | float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    fb = np.asarray(fallback, dtype=np.float32)
    return np.where(np.isfinite(values) & (values > 0.0), values, fb).astype(np.float32)


def _proxy_plane_arrays(proxy_entries: dict[int, dict[str, Any]], proxy_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals = np.zeros((proxy_count, 3), dtype=np.float64)
    offsets = np.zeros((proxy_count,), dtype=np.float64)
    rmse_q90 = np.zeros((proxy_count,), dtype=np.float64)
    confidence = np.zeros((proxy_count,), dtype=np.float64)
    for proxy_id, entry in proxy_entries.items():
        if proxy_id < 0 or proxy_id >= proxy_count:
            continue
        normal = np.asarray(entry.get("normal", [0.0, 0.0, 0.0]), dtype=np.float64)
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            normals[proxy_id] = normal / norm
        offsets[proxy_id] = float(entry.get("d", 0.0))
        rmse_q90[proxy_id] = float(entry.get("residualQ90", entry.get("rmse", 0.0)))
        confidence[proxy_id] = float(entry.get("supportMean", 0.0)) * float(entry.get("planarScoreMean", 0.0))
    return normals, offsets, rmse_q90, confidence


def _compute_final_target_lengths(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_centers: np.ndarray,
    face_area: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    edge_length: np.ndarray,
    policy: dict[str, np.ndarray],
    proxies: dict[str, np.ndarray],
    config: OperationProposalConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    detail_length = _safe_positive(
        np.asarray(policy["face_detail_target_length"], dtype=np.float32),
        np.asarray(policy.get("face_mean_edge_length", np.ones(faces.shape[0], dtype=np.float32)), dtype=np.float32),
    )
    planar_score = np.asarray(policy["face_planar_score"], dtype=np.float32).clip(0.0, 1.0)
    face_proxy_id = np.asarray(proxies["face_proxy_id"], dtype=np.int32)
    proxy_ids = np.asarray([pid for pid in np.unique(face_proxy_id) if int(pid) >= 0], dtype=np.int32)
    proxy_plane_length = np.zeros((max(int(np.max(proxy_ids)) + 1, 0) if proxy_ids.size else 0,), dtype=np.float32)
    if edge_length.size:
        scene_max = max(float(np.quantile(edge_length, 0.95)) * float(config.scene_max_length_factor), 1e-6)
    else:
        scene_max = max(float(np.quantile(detail_length, 0.95)) * float(config.scene_max_length_factor), 1e-6)
    proxy_boundary = np.asarray(proxies.get("edge_proxy_boundary", np.zeros(edge_length.shape[0], dtype=np.uint8)), dtype=np.uint8) > 0
    for proxy_id in proxy_ids:
        mask = face_proxy_id == int(proxy_id)
        area = float(np.sum(face_area[mask]))
        if area <= 0.0:
            continue
        faces_for_proxy = np.flatnonzero(mask)
        centers = face_centers[faces_for_proxy]
        if centers.shape[0] >= 2:
            extent = float(np.linalg.norm(np.max(centers, axis=0) - np.min(centers, axis=0)))
        else:
            extent = math.sqrt(area)
        boundary_edges = proxy_boundary & np.any(np.isin(edge_faces, faces_for_proxy), axis=1)
        boundary_len = float(np.median(edge_length[boundary_edges])) if np.any(boundary_edges) else 0.0
        area_len = float(config.planar_area_length_factor) * math.sqrt(area)
        extent_len = 0.20 * extent if extent > 0.0 else area_len
        if boundary_len > 0.0:
            boundary_cap = float(config.planar_boundary_length_factor) * boundary_len
        else:
            boundary_cap = scene_max
        conservative_detail = float(np.quantile(detail_length[mask], 0.75)) * float(config.proxy_min_length_factor)
        plane_len = max(min(area_len, extent_len, boundary_cap, scene_max), conservative_detail, 1e-6)
        proxy_plane_length[int(proxy_id)] = np.float32(plane_len)

    final_length = detail_length.copy()
    valid_proxy = face_proxy_id >= 0
    if np.any(valid_proxy) and proxy_plane_length.size:
        plane_len_per_face = np.zeros_like(final_length)
        in_range = valid_proxy & (face_proxy_id < proxy_plane_length.shape[0])
        plane_len_per_face[in_range] = proxy_plane_length[face_proxy_id[in_range]]
        blend = planar_score[in_range]
        final_length[in_range] = np.exp(
            blend * np.log(np.maximum(plane_len_per_face[in_range], 1e-9))
            + (1.0 - blend) * np.log(np.maximum(detail_length[in_range], 1e-9))
        ).astype(np.float32)
    stats = {
        "sceneMaxLength": float(scene_max),
        "proxyCountWithTargetLength": int(np.count_nonzero(proxy_plane_length > 0.0)),
    }
    return final_length.astype(np.float32), proxy_plane_length.astype(np.float32), stats


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


def _collapse_candidate_position(
    *,
    vertices: np.ndarray,
    edge_vertices: np.ndarray,
    edge_id: int,
    proxy_id: int,
    proxy_normals: np.ndarray,
    proxy_offsets: np.ndarray,
) -> np.ndarray:
    vi, vj = [int(v) for v in edge_vertices[int(edge_id)]]
    midpoint = 0.5 * (vertices[vi] + vertices[vj])
    if proxy_id >= 0 and proxy_id < proxy_normals.shape[0]:
        normal = proxy_normals[proxy_id]
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            return midpoint - (float(np.dot(normal, midpoint)) + float(proxy_offsets[proxy_id])) * normal
    return midpoint


def _evaluate_collapse_candidates(
    *,
    candidate_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_faces: list[np.ndarray],
    face_normals: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    face_proxy_id: np.ndarray,
    proxy_normals: np.ndarray,
    proxy_offsets: np.ndarray,
    proxy_rmse_q90: np.ndarray,
    proxy_confidence: np.ndarray,
    edge_planar_interior: np.ndarray,
    edge_boundary_score: np.ndarray,
    edge_missing: np.ndarray,
    edge_detail: np.ndarray,
    edge_tau_a: np.ndarray,
    edge_eps: np.ndarray,
    config: OperationProposalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    edge_count = edge_vertices.shape[0]
    accepted = np.zeros(edge_count, dtype=np.uint8)
    reject_code = np.zeros(edge_count, dtype=np.uint8)
    normal_error = np.zeros(edge_count, dtype=np.float32)
    quality_error = np.zeros(edge_count, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    max_rows = int(config.max_jsonl_rows_per_type)
    for rank, edge_id in enumerate(candidate_ids):
        edge_id = int(edge_id)
        reason = "accepted"
        face_a, face_b = [int(v) for v in edge_faces[edge_id]]
        proxy_id = -1
        if face_a >= 0 and face_b >= 0:
            pa = int(face_proxy_id[face_a])
            pb = int(face_proxy_id[face_b])
            proxy_id = pa if pa == pb else -1

        if edge_boundary_score[edge_id] >= float(config.boundary_threshold) and not bool(edge_planar_interior[edge_id]):
            reason = "reject_boundary"
        elif edge_missing[edge_id] >= float(config.missing_geometry_protect_threshold) and edge_detail[edge_id] < float(config.detail_collapse_threshold):
            reason = "reject_missing_geometry_protection"

        candidate = _collapse_candidate_position(
            vertices=vertices,
            edge_vertices=edge_vertices,
            edge_id=edge_id,
            proxy_id=proxy_id,
            proxy_normals=proxy_normals,
            proxy_offsets=proxy_offsets,
        )
        plane_error = 0.0
        if reason == "accepted" and bool(edge_planar_interior[edge_id]) and proxy_id >= 0 and proxy_id < proxy_normals.shape[0]:
            eps_r = max(
                float(proxy_rmse_q90[proxy_id]) if proxy_id < proxy_rmse_q90.shape[0] else 0.0,
                float(edge_eps[edge_id]),
                1e-9,
            )
            conf = max(float(proxy_confidence[proxy_id]), float(config.proxy_confidence_floor))
            plane_error = conf * (float(np.dot(proxy_normals[proxy_id], candidate)) + float(proxy_offsets[proxy_id])) ** 2 / max(eps_r * eps_r, 1e-12)
            if plane_error > 1.0:
                reason = "reject_plane"

        gates = simulate_collapse_gates(
            vertices=vertices,
            faces=faces,
            vertex_faces=vertex_faces,
            face_normals=face_normals,
            edge_vertices=edge_vertices,
            edge_faces=edge_faces,
            edge_id=edge_id,
            candidate_position=candidate,
            detail_weight=float(edge_detail[edge_id]),
            tau_a=float(edge_tau_a[edge_id]),
            q_min_hard=float(config.q_min_hard),
            q_min_soft=float(config.q_min_soft),
        )
        normal_error[edge_id] = np.float32(gates["normalError"] if np.isfinite(gates["normalError"]) else 1e9)
        quality_error[edge_id] = np.float32(gates["qualityError"] if np.isfinite(gates["qualityError"]) else 1e9)
        if reason == "accepted" and not bool(gates["topologyOk"]):
            reason = "reject_topology"
        if reason == "accepted" and float(gates["normalError"]) > 1.0:
            reason = "reject_normal"
        if reason == "accepted" and (not bool(gates["qualityOk"]) or float(gates["qualityError"]) > 1.0):
            reason = "reject_quality"

        if reason == "accepted":
            accepted[edge_id] = 1
        reject_code[edge_id] = REJECT_NAME_TO_CODE[reason]
        if max_rows <= 0 or rank < max_rows:
            rows.append(
                {
                    "operation": "collapse",
                    "rank": int(rank),
                    "edgeId": edge_id,
                    "vertices": [int(v) for v in edge_vertices[edge_id]],
                    "faces": [face_a, face_b],
                    "accepted": bool(reason == "accepted"),
                    "reason": reason,
                    "proxyId": int(proxy_id),
                    "candidatePosition": [float(v) for v in candidate],
                    "gates": {
                        "plane": _json_float(plane_error),
                        "normal": _json_float(gates["normalError"]),
                        "quality": _json_float(gates["qualityError"]),
                        "minQuality": _json_float(gates["minQuality"], fallback=0.0),
                    },
                    "signals": {
                        "boundary": float(edge_boundary_score[edge_id]),
                        "detail": float(edge_detail[edge_id]),
                        "missingGeometry": float(edge_missing[edge_id]),
                    },
                }
            )
    return accepted, reject_code, normal_error, quality_error, rows


def _evaluate_split_candidates(
    *,
    candidate_ids: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    edge_boundary_score: np.ndarray,
    edge_missing: np.ndarray,
    edge_detail: np.ndarray,
    edge_length: np.ndarray,
    edge_target: np.ndarray,
    config: OperationProposalConfig,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    edge_count = edge_vertices.shape[0]
    accepted = np.zeros(edge_count, dtype=np.uint8)
    reject_code = np.zeros(edge_count, dtype=np.uint8)
    rows: list[dict[str, Any]] = []
    max_rows = int(config.max_jsonl_rows_per_type)
    for rank, edge_id in enumerate(candidate_ids):
        edge_id = int(edge_id)
        reason = "accepted"
        if edge_boundary_score[edge_id] >= float(config.boundary_threshold):
            reason = "reject_boundary"
        elif edge_missing[edge_id] >= float(config.missing_geometry_protect_threshold) and edge_detail[edge_id] < float(config.detail_collapse_threshold):
            reason = "reject_missing_geometry_protection"
        accepted[edge_id] = 1 if reason == "accepted" else 0
        reject_code[edge_id] = REJECT_NAME_TO_CODE[reason]
        if max_rows <= 0 or rank < max_rows:
            rows.append(
                {
                    "operation": "split",
                    "rank": int(rank),
                    "edgeId": edge_id,
                    "vertices": [int(v) for v in edge_vertices[edge_id]],
                    "faces": [int(v) for v in edge_faces[edge_id]],
                    "accepted": bool(reason == "accepted"),
                    "reason": reason,
                    "signals": {
                        "length": float(edge_length[edge_id]),
                        "targetLength": float(edge_target[edge_id]),
                        "lengthRatio": _json_float(edge_length[edge_id] / max(float(edge_target[edge_id]), 1e-9)),
                        "boundary": float(edge_boundary_score[edge_id]),
                        "detail": float(edge_detail[edge_id]),
                        "missingGeometry": float(edge_missing[edge_id]),
                    },
                }
            )
    return accepted, reject_code, rows


def _evaluate_flip_candidates(
    *,
    candidate_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    edge_boundary_score: np.ndarray,
    config: OperationProposalConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    edge_count = edge_vertices.shape[0]
    accepted = np.zeros(edge_count, dtype=np.uint8)
    reject_code = np.zeros(edge_count, dtype=np.uint8)
    quality_improvement = np.zeros(edge_count, dtype=np.float32)
    rows: list[dict[str, Any]] = []
    max_rows = int(config.max_jsonl_rows_per_type)
    for rank, edge_id in enumerate(candidate_ids):
        edge_id = int(edge_id)
        reason = "accepted"
        if edge_boundary_score[edge_id] >= float(config.boundary_threshold):
            reason = "reject_boundary"
        gates = simulate_flip_quality(vertices=vertices, faces=faces, edge_vertices=edge_vertices, edge_faces=edge_faces, edge_id=edge_id)
        quality_improvement[edge_id] = np.float32(gates["qualityImprovement"])
        if reason == "accepted" and not bool(gates["valid"]):
            reason = "reject_topology"
        if reason == "accepted" and float(gates["qualityImprovement"]) <= 0.0:
            reason = "reject_not_improved"
        accepted[edge_id] = 1 if reason == "accepted" else 0
        reject_code[edge_id] = REJECT_NAME_TO_CODE[reason]
        if max_rows <= 0 or rank < max_rows:
            rows.append(
                {
                    "operation": "flip",
                    "rank": int(rank),
                    "edgeId": edge_id,
                    "vertices": [int(v) for v in edge_vertices[edge_id]],
                    "faces": [int(v) for v in edge_faces[edge_id]],
                    "accepted": bool(reason == "accepted"),
                    "reason": reason,
                    "gates": {
                        "qualityImprovement": _json_float(gates["qualityImprovement"]),
                        "oldMinQuality": _json_float(gates["oldMinQuality"], fallback=0.0),
                        "newMinQuality": _json_float(gates["newMinQuality"], fallback=0.0),
                    },
                    "signals": {"boundary": float(edge_boundary_score[edge_id])},
                }
            )
    return accepted, reject_code, quality_improvement, rows


def _write_debug_meshes(vertices: np.ndarray, faces: np.ndarray, arrays: dict[str, np.ndarray], output_dir: Path) -> dict[str, Path]:
    debug_dir = output_dir / "debug_meshes" / "operations"
    fields = {
        "collapse_pressure": "face_collapse_pressure",
        "split_pressure": "face_split_pressure",
        "flip_pressure": "face_flip_pressure",
        "operation_pressure": "face_operation_pressure",
        "collapse_accept": "face_collapse_accept",
        "split_accept": "face_split_accept",
        "flip_accept": "face_flip_accept",
        "reject_boundary": "face_reject_boundary",
        "reject_missing_geometry": "face_reject_missing_geometry",
        "reject_quality": "face_reject_quality",
        "final_target_length": "face_final_target_length",
    }
    paths: dict[str, Path] = {}
    for name, key in fields.items():
        if key not in arrays:
            continue
        values = np.asarray(arrays[key], dtype=np.float32)
        valid = np.isfinite(values)
        path = debug_dir / f"{name}.ply"
        if name == "final_target_length" and np.any(valid & (values > 0.0)):
            finite = values[valid & (values > 0.0)]
            vmin = float(np.quantile(finite, 0.02))
            vmax = float(np.quantile(finite, 0.98))
            if vmax <= vmin + 1e-9:
                vmax = vmin + 1.0
        else:
            vmin, vmax = 0.0, 1.0
        write_face_scalar_mesh(path, vertices, faces, values, valid, vmin=vmin, vmax=vmax)
        paths[name] = path
    return paths


def compute_operation_proposals(config: OperationProposalConfig, progress: ProgressFn | None = None) -> OperationProposalResult:
    progress = progress or _progress_default
    mesh_path = config.mesh_path.expanduser().resolve()
    policy_npz = config.policy_npz.expanduser().resolve()
    proxies_npz = config.proxies_npz.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    proposals_npz = output_dir / f"{config.proposals_name}.npz"
    operations_jsonl = output_dir / f"{config.proposals_name}.jsonl"
    summary_json = output_dir / f"{config.proposals_name}.summary.json"
    if proposals_npz.exists() and not config.overwrite:
        raise FileExistsError(f"Operation proposal output exists: {proposals_npz}. Re-run with overwrite=True.")

    progress(f"[ops] Mesh: {mesh_path}")
    vertices, faces = load_mesh_arrays(mesh_path)
    face_centers, face_normals, face_area, mean_edge_length = face_geometry(vertices, faces)
    topology = build_mesh_topology(faces)
    vertex_faces = build_vertex_faces(faces, int(vertices.shape[0]))
    policy = _load_npz(policy_npz)
    proxies = _load_npz(proxies_npz)
    _required(
        policy,
        [
            "face_support",
            "face_planar_score",
            "face_detail_posterior",
            "face_unexplained_offset",
            "face_detail_target_length",
            "face_position_tolerance",
            "face_direction_confidence",
            "edge_boundary_score",
            "face_weight_planar",
            "face_weight_detail",
            "face_weight_boundary",
            "face_weight_uncertain",
        ],
        "Policy npz",
    )
    _required(proxies, ["face_proxy_id", "edge_proxy_boundary"], "Proxies npz")
    face_count = int(faces.shape[0])
    edge_count = int(topology.edge_vertices.shape[0])
    if np.asarray(policy["face_support"]).shape[0] != face_count:
        raise ValueError(f"Policy face count does not match mesh: {np.asarray(policy['face_support']).shape[0]} vs {face_count}")
    if np.asarray(proxies["face_proxy_id"]).shape[0] != face_count:
        raise ValueError(f"Proxy face count does not match mesh: {np.asarray(proxies['face_proxy_id']).shape[0]} vs {face_count}")

    proxy_entries = _load_proxy_entries(config.proxies_json)
    proxy_count = max((max(proxy_entries.keys()) + 1 if proxy_entries else 0), int(np.max(proxies["face_proxy_id"])) + 1 if np.any(proxies["face_proxy_id"] >= 0) else 0)
    proxy_normals, proxy_offsets, proxy_rmse_q90, proxy_confidence = _proxy_plane_arrays(proxy_entries, proxy_count)

    edge_vertices = topology.edge_vertices
    edge_faces = topology.edge_faces
    edge_length = np.linalg.norm(vertices[edge_vertices[:, 1]] - vertices[edge_vertices[:, 0]], axis=1).astype(np.float32)
    target_policy = {**policy, "face_planar_score": np.asarray(policy["face_weight_planar"], dtype=np.float32).clip(0.0, 1.0)}
    face_final_target, proxy_target_length, target_stats = _compute_final_target_lengths(
        vertices=vertices,
        faces=faces,
        face_centers=face_centers,
        face_area=face_area,
        edge_vertices=edge_vertices,
        edge_faces=edge_faces,
        edge_length=edge_length,
        policy=target_policy,
        proxies=proxies,
        config=config,
    )

    face_support = np.asarray(policy["face_support"], dtype=np.float32).clip(0.0, 1.0)
    face_planar = np.asarray(policy["face_weight_planar"], dtype=np.float32).clip(0.0, 1.0)
    face_detail = np.asarray(policy["face_weight_detail"], dtype=np.float32).clip(0.0, 1.0)
    face_uncertain = np.asarray(policy["face_weight_uncertain"], dtype=np.float32).clip(0.0, 1.0)
    face_missing = np.asarray(policy.get("face_residual_unexplained_offset", policy["face_unexplained_offset"]), dtype=np.float32).clip(0.0, 1.0)
    face_eps = _safe_positive(policy["face_position_tolerance"], np.maximum(0.25 * mean_edge_length, 1e-6))
    face_direction_conf = np.asarray(policy["face_direction_confidence"], dtype=np.float32).clip(0.0, 1.0)
    face_proxy_id = np.asarray(proxies["face_proxy_id"], dtype=np.int32)

    edge_support = edge_face_min(edge_faces, face_support)
    edge_planar = np.asarray(policy.get("edge_weight_planar", edge_face_min(edge_faces, face_planar)), dtype=np.float32).clip(0.0, 1.0)
    edge_detail = np.asarray(policy.get("edge_weight_detail", edge_face_max(edge_faces, face_detail)), dtype=np.float32).clip(0.0, 1.0)
    edge_uncertain = np.asarray(policy.get("edge_weight_uncertain", edge_face_max(edge_faces, face_uncertain)), dtype=np.float32).clip(0.0, 1.0)
    if edge_planar.shape[0] != edge_count:
        edge_planar = edge_face_min(edge_faces, face_planar)
    if edge_detail.shape[0] != edge_count:
        edge_detail = edge_face_max(edge_faces, face_detail)
    if edge_uncertain.shape[0] != edge_count:
        edge_uncertain = edge_face_max(edge_faces, face_uncertain)
    edge_missing = edge_face_max(edge_faces, face_missing)
    edge_eps = edge_face_min(edge_faces, face_eps)
    edge_tau_a = _safe_positive(
        np.asarray(policy.get("edge_tau_a", np.full((edge_count,), math.radians(5.0), dtype=np.float32)), dtype=np.float32),
        np.float32(math.radians(5.0)),
    )
    edge_direction_conf = edge_face_mean(edge_faces, face_direction_conf)
    edge_target = edge_face_min(edge_faces, face_final_target)
    edge_target = _safe_positive(edge_target, np.maximum(edge_length, 1e-6))
    edge_boundary_score = np.asarray(policy.get("edge_weight_boundary", policy["edge_boundary_score"]), dtype=np.float32).clip(0.0, 1.0)
    if edge_boundary_score.shape[0] != edge_count:
        edge_boundary_score = edge_face_max(edge_faces, np.asarray(policy["face_weight_boundary"], dtype=np.float32).clip(0.0, 1.0))
    edge_proxy_boundary = np.asarray(proxies["edge_proxy_boundary"], dtype=np.float32)
    if edge_proxy_boundary.shape[0] != edge_count:
        edge_proxy_boundary = np.zeros(edge_count, dtype=np.float32)
    edge_is_boundary = topology.boundary_edges.astype(bool)
    combined_boundary = np.maximum.reduce([
        edge_boundary_score,
        edge_proxy_boundary.clip(0.0, 1.0),
    ]).astype(np.float32)
    protected_edge = edge_is_boundary | (combined_boundary >= float(config.boundary_threshold))

    face_a = edge_faces[:, 0]
    face_b = edge_faces[:, 1]
    has_two = face_b >= 0
    same_proxy = np.zeros(edge_count, dtype=bool)
    same_proxy[has_two] = (face_proxy_id[face_a[has_two]] >= 0) & (face_proxy_id[face_a[has_two]] == face_proxy_id[face_b[has_two]])
    planar_interior = (
        same_proxy
        & (edge_planar >= float(config.planar_threshold))
        & (edge_support >= float(config.support_threshold))
        & (~protected_edge)
    )
    oversampled = edge_length < float(config.alpha_collapse) * edge_target
    planar_oversampled = planar_interior & (edge_length < float(config.alpha_planar_collapse) * edge_target)
    low_detail_oversampled = oversampled & (edge_detail < float(config.detail_collapse_threshold)) & (~protected_edge)
    undersampled = edge_length > float(config.alpha_split) * edge_target

    support_gate = np.clip((edge_support - float(config.support_threshold)) / max(1.0 - float(config.support_threshold), 1e-6), 0.0, 1.0)
    certainty_gate = (1.0 - edge_uncertain).clip(0.0, 1.0)
    boundary_gate = (1.0 - combined_boundary).clip(0.0, 1.0) ** 2
    shortness = np.clip((float(config.alpha_collapse) * edge_target - edge_length) / np.maximum(float(config.alpha_collapse) * edge_target, 1e-9), 0.0, 1.0)
    planar_shortness = np.clip((float(config.alpha_planar_collapse) * edge_target - edge_length) / np.maximum(float(config.alpha_planar_collapse) * edge_target, 1e-9), 0.0, 1.0)
    collapse_pressure = (
        np.maximum(
            planar_oversampled.astype(np.float32) * planar_shortness * edge_planar * support_gate,
            low_detail_oversampled.astype(np.float32) * shortness * (1.0 - edge_detail).clip(0.0, 1.0) * support_gate * (0.25 + 0.75 * edge_planar),
        )
        * boundary_gate
        * certainty_gate
    ).clip(0.0, 1.0)
    collapse_pressure[protected_edge] = 0.0
    split_pressure = (undersampled.astype(np.float32) * support_gate * edge_detail * boundary_gate * certainty_gate).clip(0.0, 1.0)
    split_pressure[protected_edge] = 0.0

    flip_quality_probe = np.zeros(edge_count, dtype=np.float32)
    flip_seed_ids = np.flatnonzero(has_two & (~protected_edge))
    if flip_seed_ids.size:
        probe_limit = min(int(config.max_detailed_flips) if int(config.max_detailed_flips) > 0 else flip_seed_ids.size, flip_seed_ids.size)
        seed_score = edge_support[flip_seed_ids] * edge_detail[flip_seed_ids] * edge_direction_conf[flip_seed_ids]
        if flip_seed_ids.size > probe_limit:
            local = np.argpartition(seed_score, -probe_limit)[-probe_limit:]
            flip_seed_ids = flip_seed_ids[local]
        for edge_id in flip_seed_ids:
            flip_quality_probe[int(edge_id)] = max(
                0.0,
                float(simulate_flip_quality(vertices=vertices, faces=faces, edge_vertices=edge_vertices, edge_faces=edge_faces, edge_id=int(edge_id))["qualityImprovement"]),
            )
    if np.any(flip_quality_probe > 0.0):
        flip_improvement_norm = np.clip(flip_quality_probe / max(float(np.quantile(flip_quality_probe[flip_quality_probe > 0.0], 0.95)), 1e-6), 0.0, 1.0)
    else:
        flip_improvement_norm = flip_quality_probe
    flip_pressure = (support_gate * edge_detail * edge_direction_conf * flip_improvement_norm * boundary_gate * certainty_gate).astype(np.float32).clip(0.0, 1.0)
    flip_pressure[protected_edge] = 0.0

    collapse_ids = _candidate_ids(collapse_pressure, int(config.max_detailed_collapses))
    split_ids = _candidate_ids(split_pressure, int(config.max_detailed_splits)) if bool(config.enable_split_proposals) else np.zeros((0,), dtype=np.int64)
    flip_ids = _candidate_ids(flip_pressure, int(config.max_detailed_flips))
    progress(f"[ops] Edges: {edge_count:,}; collapse candidates={collapse_ids.size:,}; split candidates={split_ids.size:,}; flip candidates={flip_ids.size:,}")

    collapse_accept, collapse_reject, collapse_normal_error, collapse_quality_error, collapse_rows = _evaluate_collapse_candidates(
        candidate_ids=collapse_ids,
        vertices=vertices,
        faces=faces,
        vertex_faces=vertex_faces,
        face_normals=face_normals,
        edge_vertices=edge_vertices,
        edge_faces=edge_faces,
        face_proxy_id=face_proxy_id,
        proxy_normals=proxy_normals,
        proxy_offsets=proxy_offsets,
        proxy_rmse_q90=proxy_rmse_q90,
        proxy_confidence=np.maximum(proxy_confidence, float(config.proxy_confidence_floor)),
        edge_planar_interior=planar_interior,
        edge_boundary_score=np.maximum(combined_boundary, edge_is_boundary.astype(np.float32)),
        edge_missing=edge_missing,
        edge_detail=edge_detail,
        edge_tau_a=edge_tau_a,
        edge_eps=edge_eps,
        config=config,
    )
    split_accept, split_reject, split_rows = _evaluate_split_candidates(
        candidate_ids=split_ids,
        edge_vertices=edge_vertices,
        edge_faces=edge_faces,
        edge_boundary_score=np.maximum(combined_boundary, edge_is_boundary.astype(np.float32)),
        edge_missing=edge_missing,
        edge_detail=edge_detail,
        edge_length=edge_length,
        edge_target=edge_target,
        config=config,
    )
    flip_accept, flip_reject, flip_quality_improvement, flip_rows = _evaluate_flip_candidates(
        candidate_ids=flip_ids,
        vertices=vertices,
        faces=faces,
        edge_vertices=edge_vertices,
        edge_faces=edge_faces,
        edge_boundary_score=np.maximum(combined_boundary, edge_is_boundary.astype(np.float32)),
        config=config,
    )
    rows = collapse_rows + split_rows + flip_rows

    reject_boundary = ((collapse_reject == REJECT_NAME_TO_CODE["reject_boundary"]) | (split_reject == REJECT_NAME_TO_CODE["reject_boundary"]) | (flip_reject == REJECT_NAME_TO_CODE["reject_boundary"])).astype(np.float32)
    reject_missing = ((collapse_reject == REJECT_NAME_TO_CODE["reject_missing_geometry_protection"]) | (split_reject == REJECT_NAME_TO_CODE["reject_missing_geometry_protection"])).astype(np.float32)
    reject_quality = ((collapse_reject == REJECT_NAME_TO_CODE["reject_quality"]) | (flip_reject == REJECT_NAME_TO_CODE["reject_not_improved"])).astype(np.float32)
    reject_topology = ((collapse_reject == REJECT_NAME_TO_CODE["reject_topology"]) | (flip_reject == REJECT_NAME_TO_CODE["reject_topology"])).astype(np.float32)
    reject_normal = (collapse_reject == REJECT_NAME_TO_CODE["reject_normal"]).astype(np.float32)
    face_arrays = {
        "face_final_target_length": face_final_target.astype(np.float32),
        "face_collapse_pressure": face_max_from_edges(topology.face_edges, collapse_pressure),
        "face_split_pressure": face_max_from_edges(topology.face_edges, split_pressure),
        "face_flip_pressure": face_max_from_edges(topology.face_edges, flip_pressure),
        "face_operation_pressure": face_max_from_edges(topology.face_edges, np.maximum.reduce([collapse_pressure, split_pressure, flip_pressure])),
        "face_collapse_accept": face_max_from_edges(topology.face_edges, collapse_accept.astype(np.float32)),
        "face_split_accept": face_max_from_edges(topology.face_edges, split_accept.astype(np.float32)),
        "face_flip_accept": face_max_from_edges(topology.face_edges, flip_accept.astype(np.float32)),
        "face_reject_boundary": face_max_from_edges(topology.face_edges, reject_boundary),
        "face_reject_missing_geometry": face_max_from_edges(topology.face_edges, reject_missing),
        "face_reject_quality": face_max_from_edges(topology.face_edges, reject_quality),
        "face_reject_topology": face_max_from_edges(topology.face_edges, reject_topology),
        "face_reject_normal": face_max_from_edges(topology.face_edges, reject_normal),
    }

    arrays: dict[str, np.ndarray] = {
        "edge_vertices": edge_vertices.astype(np.int64),
        "edge_faces": edge_faces.astype(np.int64),
        "face_edges": topology.face_edges.astype(np.int64),
        "edge_length": edge_length.astype(np.float32),
        "edge_final_target_length": edge_target.astype(np.float32),
        "edge_position_tolerance": edge_eps.astype(np.float32),
        "edge_support": edge_support.astype(np.float32),
        "edge_planar_score": edge_planar.astype(np.float32),
        "edge_detail_score": edge_detail.astype(np.float32),
        "edge_unexplained_offset": edge_missing.astype(np.float32),
        "edge_uncertainty_score": edge_uncertain.astype(np.float32),
        "edge_boundary_score": combined_boundary.astype(np.float32),
        "edge_proxy_boundary": edge_proxy_boundary.astype(np.float32),
        "edge_is_mesh_boundary": edge_is_boundary.astype(np.uint8),
        "edge_planar_interior": planar_interior.astype(np.uint8),
        "edge_planar_oversampled": planar_oversampled.astype(np.uint8),
        "edge_low_detail_oversampled": low_detail_oversampled.astype(np.uint8),
        "edge_undersampled": undersampled.astype(np.uint8),
        "edge_collapse_pressure": collapse_pressure.astype(np.float32),
        "edge_split_pressure": split_pressure.astype(np.float32),
        "edge_flip_pressure": flip_pressure.astype(np.float32),
        "edge_collapse_accept": collapse_accept,
        "edge_split_accept": split_accept,
        "edge_flip_accept": flip_accept,
        "edge_collapse_reject_code": collapse_reject,
        "edge_split_reject_code": split_reject,
        "edge_flip_reject_code": flip_reject,
        "edge_collapse_normal_error": collapse_normal_error,
        "edge_collapse_quality_error": collapse_quality_error,
        "edge_flip_quality_improvement": np.maximum(flip_quality_probe, flip_quality_improvement).astype(np.float32),
        "proxy_target_length": proxy_target_length.astype(np.float32),
        **face_arrays,
    }
    np.savez_compressed(proposals_npz, **arrays)
    _write_jsonl(operations_jsonl, rows)
    debug_meshes = _write_debug_meshes(vertices, faces, arrays, output_dir) if config.write_debug_meshes else {}
    summary = {
        "stageName": "omega_local_operation_proposals",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "mesh_path": str(mesh_path),
            "policy_npz": str(policy_npz),
            "proxies_npz": str(proxies_npz),
            "output_dir": str(output_dir),
            "proxies_json": None if config.proxies_json is None else str(config.proxies_json),
        },
        "inputs": {
            "meshPath": str(mesh_path),
            "policyNpz": str(policy_npz),
            "proxiesNpz": str(proxies_npz),
        },
        "mesh": {
            "vertexCount": int(vertices.shape[0]),
            "faceCount": face_count,
            "edgeCount": edge_count,
            "boundaryEdgeCount": int(np.count_nonzero(edge_is_boundary)),
        },
        "counts": {
            "collapseCandidates": int(collapse_ids.size),
            "splitCandidates": int(split_ids.size),
            "splitPressureEdges": int(np.count_nonzero(split_pressure > 0.0)),
            "flipCandidates": int(flip_ids.size),
            "collapseAcceptedDryRun": int(np.count_nonzero(collapse_accept)),
            "splitAcceptedDryRun": int(np.count_nonzero(split_accept)),
            "flipAcceptedDryRun": int(np.count_nonzero(flip_accept)),
            "planarOversampledEdges": int(np.count_nonzero(planar_oversampled)),
            "lowDetailOversampledEdges": int(np.count_nonzero(low_detail_oversampled)),
            "undersampledEdges": int(np.count_nonzero(undersampled)),
        },
        "targetLength": target_stats,
        "stats": {
            "edge_length": _summarize(edge_length),
            "edge_final_target_length": _summarize(edge_target),
            "edge_collapse_pressure": _summarize(collapse_pressure, collapse_pressure > 0.0),
            "edge_split_pressure": _summarize(split_pressure, split_pressure > 0.0),
            "edge_flip_pressure": _summarize(flip_pressure, flip_pressure > 0.0),
            "face_operation_pressure": _summarize(face_arrays["face_operation_pressure"], face_arrays["face_operation_pressure"] > 0.0),
        },
        "rejectCodes": REJECT_CODE_TO_NAME,
        "outputs": {
            "proposalsNpz": str(proposals_npz),
            "operationsJsonl": str(operations_jsonl),
            "summaryJson": str(summary_json),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
        "notes": [
            "This is a dry-run proposal stage. It does not mutate vertices, faces, splats, or topology.",
            "Collapse proposals are restricted to planar-oversampled or low-detail oversampled edges.",
            "Split proposals come from detail-weighted undersampling.",
            "Split proposals are disabled by default for post-hoc simplification; split pressure remains a diagnostic/training cue.",
            "Flip proposals are quality/alignment probes and never cross protected boundaries.",
            "Image-space reprojection remains a later sampled audit gate; this stage uses Phase 2/3 world-space target lengths and tolerances.",
        ],
    }
    _write_json(summary_json, summary)
    progress(f"[ops] Wrote {proposals_npz}")
    progress(f"[ops] Wrote {operations_jsonl}")
    return OperationProposalResult(
        proposals_npz=proposals_npz,
        operations_jsonl=operations_jsonl,
        summary_json=summary_json,
        debug_meshes=debug_meshes,
        edge_count=edge_count,
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


def _config_from_args(args: argparse.Namespace) -> OperationProposalConfig:
    model_dir = args.model_dir.expanduser().resolve()
    mesh_arg = _optional_path(args.mesh)
    policy_arg = _optional_path(args.policy_npz)
    proxies_arg = _optional_path(args.proxies_npz)
    proxies_json_arg = _optional_path(args.proxies_json)
    output_arg = _optional_path(args.output_dir)
    mesh_path = _existing_file(
        _resolve_path(mesh_arg, model_dir) if mesh_arg is not None else model_dir / "remesh" / "local" / "preclean_mesh.ply",
        "Mesh",
    )
    policy_npz = _existing_file(
        _resolve_path(policy_arg, model_dir) if policy_arg is not None else model_dir / "remesh" / "local" / "policy.npz",
        "Policy npz",
    )
    proxies_npz = _existing_file(
        _resolve_path(proxies_arg, model_dir) if proxies_arg is not None else model_dir / "remesh" / "local" / "proxies.npz",
        "Proxies npz",
    )
    proxies_json = None
    if proxies_json_arg is not None:
        proxies_json = _existing_file(_resolve_path(proxies_json_arg, model_dir), "Proxies json")
    elif (model_dir / "remesh" / "local" / "proxies.json").exists():
        proxies_json = model_dir / "remesh" / "local" / "proxies.json"
    output_dir = output_arg.expanduser().resolve() if output_arg is not None else model_dir / "remesh" / "local"
    return OperationProposalConfig(
        mesh_path=mesh_path,
        policy_npz=policy_npz,
        proxies_npz=proxies_npz,
        proxies_json=proxies_json,
        output_dir=output_dir,
        proposals_name=str(args.proposals_name),
        support_threshold=float(args.support_threshold),
        planar_threshold=float(args.planar_threshold),
        boundary_threshold=float(args.boundary_threshold),
        detail_collapse_threshold=float(args.detail_collapse_threshold),
        missing_geometry_protect_threshold=float(args.missing_geometry_protect_threshold),
        alpha_collapse=float(args.alpha_collapse),
        alpha_planar_collapse=float(args.alpha_planar_collapse),
        alpha_split=float(args.alpha_split),
        q_min_hard=float(args.q_min_hard),
        q_min_soft=float(args.q_min_soft),
        max_detailed_collapses=int(args.max_detailed_collapses),
        max_detailed_splits=int(args.max_detailed_splits),
        max_detailed_flips=int(args.max_detailed_flips),
        max_jsonl_rows_per_type=int(args.max_jsonl_rows_per_type),
        enable_split_proposals=bool(args.enable_split_proposals),
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
    parser = argparse.ArgumentParser(description="Dry-run local remesh operation proposals from OMeGa Phase 3 policy fields.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--proxies-npz", type=Path, default=None)
    parser.add_argument("--proxies-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--proposals-name", default="operation_proposals")
    parser.add_argument("--support-threshold", type=float, default=0.08)
    parser.add_argument("--planar-threshold", type=float, default=0.35)
    parser.add_argument("--boundary-threshold", type=float, default=0.50)
    parser.add_argument("--detail-collapse-threshold", type=float, default=0.35)
    parser.add_argument("--missing-geometry-protect-threshold", type=float, default=0.70)
    parser.add_argument("--alpha-collapse", type=float, default=0.75)
    parser.add_argument("--alpha-planar-collapse", type=float, default=1.50)
    parser.add_argument("--alpha-split", type=float, default=1.50)
    parser.add_argument("--q-min-hard", type=float, default=0.03)
    parser.add_argument("--q-min-soft", type=float, default=0.12)
    parser.add_argument("--max-detailed-collapses", type=int, default=50000)
    parser.add_argument("--max-detailed-splits", type=int, default=50000)
    parser.add_argument("--max-detailed-flips", type=int, default=50000)
    parser.add_argument("--max-jsonl-rows-per-type", type=int, default=20000)
    parser.add_argument(
        "--enable-split-proposals",
        action="store_true",
        help="Enable accepted dry-run split proposals. Default keeps split pressure diagnostic for post-hoc simplification.",
    )
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
    compute_operation_proposals(_config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
