"""Continuous Phase 3 remesh weights.

The policy stage estimates evidence fields such as support, stable detail,
planarity, disagreement, and unexplained offset.  This module converts those
fields into the four weights consumed by Phase 4:

``W_planar``, ``W_detail``, ``W_boundary``, and ``W_uncertain``.

The weights are deliberately continuous.  Phase 4 should use them as operation
costs and gates rather than turning the mesh into hard region labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh


@dataclass(frozen=True)
class WeightComputationSummary:
    finalized_with_proxies: bool
    face_count: int
    edge_count: int
    vertex_count: int
    stats: dict[str, dict[str, float]]


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _smooth_max(*values: np.ndarray) -> np.ndarray:
    """Probabilistic union: smooth, monotonic, and bounded in [0, 1]."""

    if not values:
        raise ValueError("smooth max requires at least one array")
    out = np.ones_like(_clip01(values[0]), dtype=np.float32)
    for value in values:
        out *= 1.0 - _clip01(value)
    return (1.0 - out).clip(0.0, 1.0).astype(np.float32)


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


def _normalize_by_quantile(values: np.ndarray, valid: np.ndarray, q: float = 0.95) -> np.ndarray:
    out = np.zeros_like(np.asarray(values, dtype=np.float32))
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if not np.any(mask):
        return out
    scale = float(np.quantile(np.asarray(values, dtype=np.float64)[mask], float(q)))
    scale = max(scale, 1e-6)
    out[mask] = np.clip(np.asarray(values, dtype=np.float32)[mask] / scale, 0.0, 1.0)
    return out.astype(np.float32)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


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
    return vertices, faces


def _face_max_from_edges(face_edges: np.ndarray, edge_values: np.ndarray) -> np.ndarray:
    face_edges = np.asarray(face_edges, dtype=np.int64)
    edge_values = np.asarray(edge_values, dtype=np.float32)
    out = np.zeros((face_edges.shape[0],), dtype=np.float32)
    for local_id in range(face_edges.shape[1]):
        valid = face_edges[:, local_id] >= 0
        current = np.zeros_like(out)
        current[valid] = edge_values[face_edges[valid, local_id]]
        out = np.maximum(out, current)
    return out.astype(np.float32)


def _face_mean_from_edges(face_edges: np.ndarray, edge_values: np.ndarray) -> np.ndarray:
    face_edges = np.asarray(face_edges, dtype=np.int64)
    edge_values = np.asarray(edge_values, dtype=np.float32)
    accum = np.zeros((face_edges.shape[0],), dtype=np.float32)
    count = np.zeros((face_edges.shape[0],), dtype=np.float32)
    for local_id in range(face_edges.shape[1]):
        valid = face_edges[:, local_id] >= 0
        accum[valid] += edge_values[face_edges[valid, local_id]]
        count[valid] += 1.0
    return np.divide(accum, np.maximum(count, 1.0), out=np.zeros_like(accum)).astype(np.float32)


def _edge_from_faces(edge_faces: np.ndarray, values: np.ndarray, reducer: str) -> np.ndarray:
    edge_faces = np.asarray(edge_faces, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    a = np.clip(edge_faces[:, 0], 0, max(len(values) - 1, 0))
    b = edge_faces[:, 1]
    out = values[a].copy()
    has_two = b >= 0
    if np.any(has_two):
        other = values[b[has_two]]
        if reducer == "min":
            out[has_two] = np.minimum(out[has_two], other)
        elif reducer == "max":
            out[has_two] = np.maximum(out[has_two], other)
        elif reducer == "mean":
            out[has_two] = 0.5 * (out[has_two] + other)
        else:
            raise ValueError(f"Unknown edge reducer: {reducer}")
    return out.astype(np.float32)


def _vertex_area_mean(faces: np.ndarray, values: np.ndarray, face_area: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    weights = np.maximum(np.asarray(face_area, dtype=np.float64), 1e-12)
    accum = np.zeros((int(vertex_count),), dtype=np.float64)
    denom = np.zeros((int(vertex_count),), dtype=np.float64)
    weighted = values * weights
    for local_id in range(3):
        np.add.at(accum, faces[:, local_id], weighted)
        np.add.at(denom, faces[:, local_id], weights)
    return np.divide(accum, np.maximum(denom, 1e-12), out=np.zeros_like(accum)).clip(0.0, 1.0).astype(np.float32)


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


def compute_phase3_weights(
    *,
    policy: dict[str, np.ndarray],
    faces: np.ndarray,
    vertex_count: int,
    proxies: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], WeightComputationSummary]:
    """Compute continuous face/edge/vertex remesh weights.

    ``proxies`` is optional because ``policy.py`` runs before proxy extraction.
    The combined Phase 3 script calls this function again after proxies exist so
    the finalized boundary weight includes proxy boundaries.
    """

    faces = np.asarray(faces, dtype=np.int64)
    face_count = int(faces.shape[0])
    support = _clip01(policy["face_support"])
    planar_score = _clip01(policy["face_planar_score"])
    detail = _clip01(policy["face_detail_posterior"])
    unexplained = _clip01(policy["face_unexplained_offset"])
    direction_conf = _clip01(policy.get("face_direction_confidence", np.zeros(face_count, dtype=np.float32)))
    face_area = np.asarray(policy.get("face_area", np.ones(face_count, dtype=np.float32)), dtype=np.float32)
    edge_faces = np.asarray(policy["edge_faces"], dtype=np.int64)
    edge_vertices = np.asarray(policy["edge_vertices"], dtype=np.int64)
    edge_count = int(edge_faces.shape[0])

    if "face_protect_disagreement" in policy:
        disagreement = _clip01(policy["face_protect_disagreement"])
    else:
        normal_count = np.asarray(policy.get("normal_count", np.ones(face_count, dtype=np.int32)))
        disagreement = _normalize_by_quantile(policy["face_view_disagreement"], normal_count > 0, q=0.95)

    base_edge_boundary = _clip01(policy.get("edge_boundary_score", np.zeros(edge_count, dtype=np.float32)))
    proxy_face_boundary = np.zeros((face_count,), dtype=np.float32)
    proxy_edge_boundary = np.zeros((edge_count,), dtype=np.float32)
    proxy_weight = np.zeros((face_count,), dtype=np.float32)
    if proxies is not None:
        if "face_proxy_boundary" in proxies and np.asarray(proxies["face_proxy_boundary"]).shape[0] == face_count:
            proxy_face_boundary = _clip01(proxies["face_proxy_boundary"])
        if "edge_proxy_boundary" in proxies and np.asarray(proxies["edge_proxy_boundary"]).shape[0] == edge_count:
            proxy_edge_boundary = _clip01(proxies["edge_proxy_boundary"])
        if "face_proxy_weight" in proxies and np.asarray(proxies["face_proxy_weight"]).shape[0] == face_count:
            proxy_weight = _clip01(proxies["face_proxy_weight"])

    stable_detail = detail * (0.25 + 0.75 * direction_conf)
    stable_planar = planar_score * support
    stable_explanation = np.maximum(stable_planar, stable_detail).clip(0.0, 1.0).astype(np.float32)
    offset_detail_pressure = (unexplained * stable_detail).clip(0.0, 1.0).astype(np.float32)
    residual_unexplained = (unexplained * (1.0 - stable_detail)).clip(0.0, 1.0).astype(np.float32)
    low_support_uncertainty = ((1.0 - support) * (1.0 - stable_explanation) ** 2).clip(0.0, 1.0).astype(np.float32)
    disagreement_uncertainty = (disagreement * (1.0 - 0.75 * stable_detail)).clip(0.0, 1.0).astype(np.float32)
    # In the simplification-first pass, residual offset/disagreement are not a
    # freeze mask. They are feature/detail evidence unless the face has almost
    # no support and no stable planar/detail explanation.
    face_weight_uncertain = (
        0.35
        * low_support_uncertainty
        * (1.0 - 0.75 * stable_planar)
        * (1.0 - 0.50 * stable_detail)
    ).clip(0.0, 1.0).astype(np.float32)

    # Direction confidence controls edge-flow alignment, but density should not
    # disappear completely when the direction tensor is weak.
    direction_gate = 0.25 + 0.75 * direction_conf
    support_gate = 0.25 + 0.75 * np.sqrt(np.clip(support, 0.0, 1.0))
    feature_detail_pressure = (
        _smooth_max(residual_unexplained, disagreement_uncertainty)
        * support_gate
        * (1.0 - 0.65 * stable_planar)
    ).clip(0.0, 1.0).astype(np.float32)
    base_detail_weight = (
        detail
        * direction_gate
        * support_gate
    ).clip(0.0, 1.0).astype(np.float32)
    face_weight_detail = _smooth_max(base_detail_weight, offset_detail_pressure * support_gate, feature_detail_pressure)

    proxy_planar_boost = 0.5 + 0.5 * proxy_weight
    face_weight_planar = (
        planar_score
        * support
        * (1.0 - 0.5 * detail)
        * (1.0 - 0.20 * face_weight_uncertain)
        * proxy_planar_boost
    ).clip(0.0, 1.0).astype(np.float32)

    edge_weight_planar = _edge_from_faces(edge_faces, face_weight_planar, "min")
    edge_weight_detail = _edge_from_faces(edge_faces, face_weight_detail, "max")
    edge_weight_uncertain = _edge_from_faces(edge_faces, face_weight_uncertain, "max")
    edge_weight_boundary = _smooth_max(
        base_edge_boundary,
        proxy_edge_boundary,
    )
    face_boundary_from_edges = _face_mean_from_edges(np.asarray(policy["face_edges"], dtype=np.int64), edge_weight_boundary)
    face_proxy_boundary_fallback = 0.33 * proxy_face_boundary

    vertex_weight_planar = _vertex_area_mean(faces, face_weight_planar, face_area, int(vertex_count))
    vertex_weight_detail = _vertex_face_max(faces, face_weight_detail, int(vertex_count))
    vertex_weight_uncertain = _vertex_face_max(faces, face_weight_uncertain, int(vertex_count))
    vertex_weight_boundary = _vertex_edge_max(edge_vertices, edge_weight_boundary, int(vertex_count)).astype(np.float32)
    face_weight_boundary = np.maximum(face_boundary_from_edges, face_proxy_boundary_fallback).clip(0.0, 1.0).astype(np.float32)

    weights = {
        "face_weight_planar": face_weight_planar.astype(np.float32),
        "face_weight_detail": face_weight_detail.astype(np.float32),
        "face_weight_boundary": face_weight_boundary.astype(np.float32),
        "face_weight_uncertain": face_weight_uncertain.astype(np.float32),
        "face_base_detail_weight": base_detail_weight.astype(np.float32),
        "face_feature_detail_pressure": feature_detail_pressure.astype(np.float32),
        "face_offset_detail_pressure": offset_detail_pressure.astype(np.float32),
        "face_residual_unexplained_offset": residual_unexplained.astype(np.float32),
        "face_low_support_uncertainty": low_support_uncertainty.astype(np.float32),
        "face_disagreement_uncertainty": disagreement_uncertainty.astype(np.float32),
        "face_stable_explanation": stable_explanation.astype(np.float32),
        "face_boundary_from_edges": face_boundary_from_edges.astype(np.float32),
        "edge_weight_planar": edge_weight_planar.astype(np.float32),
        "edge_weight_detail": edge_weight_detail.astype(np.float32),
        "edge_weight_boundary": edge_weight_boundary.astype(np.float32),
        "edge_weight_uncertain": edge_weight_uncertain.astype(np.float32),
        "vertex_weight_planar": vertex_weight_planar.astype(np.float32),
        "vertex_weight_detail": vertex_weight_detail.astype(np.float32),
        "vertex_weight_boundary": vertex_weight_boundary.astype(np.float32),
        "vertex_weight_uncertain": vertex_weight_uncertain.astype(np.float32),
    }
    stats = {name: _summarize(value) for name, value in weights.items()}
    summary = WeightComputationSummary(
        finalized_with_proxies=bool(proxies is not None),
        face_count=face_count,
        edge_count=edge_count,
        vertex_count=int(vertex_count),
        stats=stats,
    )
    return weights, summary


def _colorize_scalar(values: np.ndarray, valid: np.ndarray | None = None, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(values)
    else:
        valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    normalized = np.clip((values - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    mapped = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and values.ndim == 1 and rgb.shape[1] == 1:
        rgb = rgb[:, 0, :]
    rgb[~valid] = np.array([38, 38, 38], dtype=np.uint8)
    return rgb


def _write_face_color_ply(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate([np.asarray(colors, dtype=np.uint8), np.full((faces.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def write_weight_debug_meshes(vertices: np.ndarray, faces: np.ndarray, weights: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes" / "weights"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in [
        "face_weight_planar",
        "face_weight_detail",
        "face_weight_boundary",
        "face_weight_uncertain",
        "face_boundary_from_edges",
        "face_stable_explanation",
            "face_offset_detail_pressure",
            "face_feature_detail_pressure",
            "face_residual_unexplained_offset",
        "face_low_support_uncertainty",
        "face_disagreement_uncertainty",
    ]:
        if name not in weights:
            continue
        path = debug_dir / f"{name}.ply"
        _write_face_color_ply(vertices, faces, _colorize_scalar(weights[name], vmin=0.0, vmax=1.0), path)
        paths[name] = path
    return paths


def finalize_policy_weights(
    *,
    mesh_path: Path,
    policy_npz: Path,
    proxies_npz: Path | None,
    output_dir: Path,
    write_debug_meshes: bool = True,
) -> WeightComputationSummary:
    """Rewrite ``policy.npz`` with finalized Phase 3 weight arrays."""

    vertices, faces = _load_mesh_arrays(mesh_path.expanduser().resolve())
    policy_path = policy_npz.expanduser().resolve()
    policy = _load_npz(policy_path)
    proxies = _load_npz(proxies_npz.expanduser().resolve()) if proxies_npz is not None else None
    weights, summary = compute_phase3_weights(
        policy=policy,
        faces=faces,
        vertex_count=int(vertices.shape[0]),
        proxies=proxies,
    )
    merged = {**policy, **weights}
    np.savez_compressed(policy_path, **merged)
    output_dir = output_dir.expanduser().resolve()
    debug_meshes: dict[str, Path] = {}
    if write_debug_meshes:
        debug_meshes = write_weight_debug_meshes(vertices, faces, weights, output_dir)
    summary_payload: dict[str, Any] = {
        "stageName": "omega_remesh_phase3_weights",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "finalizedWithProxies": bool(summary.finalized_with_proxies),
        "inputs": {
            "meshPath": str(mesh_path),
            "policyNpz": str(policy_path),
            "proxiesNpz": str(proxies_npz) if proxies_npz is not None else None,
        },
        "counts": {
            "faceCount": int(summary.face_count),
            "edgeCount": int(summary.edge_count),
            "vertexCount": int(summary.vertex_count),
        },
        "stats": summary.stats,
        "outputs": {
            "policyNpz": str(policy_path),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
        "notes": [
            "Weights are Phase 3 policy outputs consumed by Phase 4.",
            "Uncertainty is a low-support/no-explanation fallback, not a hard freeze mask for residual offset or disagreement.",
            "Residual offset and disagreement can become feature/detail pressure when the face has enough support.",
            "Proxy boundaries are included only when proxies_npz is available.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_summary_path = output_dir / "weights_summary.json"
    weights_summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    policy_summary_path = output_dir / f"{policy_path.stem}_summary.json"
    if policy_summary_path.exists() and policy_summary_path.is_file():
        try:
            policy_summary = json.loads(policy_summary_path.read_text(encoding="utf-8"))
            policy_summary.setdefault("stats", {}).update(summary.stats)
            policy_summary["weights"] = {
                "finalizedWithProxies": bool(summary.finalized_with_proxies),
                "stats": summary.stats,
            }
            counts = policy_summary.setdefault("counts", {})
            count_keys = {
                "planar": "highPlanarWeightFaces",
                "detail": "highDetailWeightFaces",
                "boundary": "highBoundaryWeightFaces",
                "uncertain": "highUncertaintyWeightFaces",
            }
            for name, key in count_keys.items():
                counts[key] = int(np.count_nonzero(weights[f"face_weight_{name}"] >= 0.50))
            policy_summary.setdefault("outputs", {})["weightsSummaryJson"] = str(weights_summary_path)
            notes = policy_summary.setdefault("notes", [])
            note = "Policy weights were finalized after planar proxy extraction; proxy boundaries are included in boundary weights."
            if note not in notes:
                notes.append(note)
            policy_summary_path.write_text(json.dumps(policy_summary, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            warning_path = output_dir / "weights_summary_update_warning.txt"
            warning_path.write_text(f"Could not update {policy_summary_path}: {exc}\n", encoding="utf-8")
    return summary
