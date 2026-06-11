"""Local mesh operation utilities for OMeGa dry-run remesh proposals.

The helpers in this module simulate small one-ring edits without mutating the
mesh.  Phase 4A uses them to decide whether proposal pressure is plausible
before the real topology-changing operator is enabled.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh


def load_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
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


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw_normals, axis=1)
    normals = np.divide(raw_normals, np.maximum(double_area[:, None], 1e-12), out=np.zeros_like(raw_normals))
    centers = np.mean(triangles, axis=1)
    areas = 0.5 * double_area
    edge_lengths = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2)
    return centers, normals, areas, np.mean(edge_lengths, axis=1)


def triangle_quality_from_vertices(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.size == 0:
        return np.zeros((0,), dtype=np.float64)
    edges = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2)
    double_area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    area = 0.5 * double_area
    denom = np.maximum(np.sum(edges * edges, axis=1), 1e-18)
    return (4.0 * math.sqrt(3.0) * area / denom).astype(np.float64)


def build_vertex_faces(faces: np.ndarray, vertex_count: int) -> list[np.ndarray]:
    buckets: list[list[int]] = [[] for _ in range(int(vertex_count))]
    for face_id, face in enumerate(np.asarray(faces, dtype=np.int64)):
        buckets[int(face[0])].append(face_id)
        buckets[int(face[1])].append(face_id)
        buckets[int(face[2])].append(face_id)
    return [np.asarray(values, dtype=np.int64) for values in buckets]


def face_max_from_edges(face_edges: np.ndarray, edge_values: np.ndarray) -> np.ndarray:
    face_edges = np.asarray(face_edges, dtype=np.int64)
    edge_values = np.asarray(edge_values, dtype=np.float32)
    out = np.zeros((face_edges.shape[0],), dtype=np.float32)
    for local_id in range(face_edges.shape[1]):
        valid = face_edges[:, local_id] >= 0
        current = np.zeros_like(out)
        current[valid] = edge_values[face_edges[valid, local_id]]
        out = np.maximum(out, current)
    return out


def edge_face_min(edge_faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    edge_faces = np.asarray(edge_faces, dtype=np.int64)
    a = edge_faces[:, 0]
    b = edge_faces[:, 1]
    out = values[np.clip(a, 0, len(values) - 1)]
    has_two = b >= 0
    out[has_two] = np.minimum(out[has_two], values[b[has_two]])
    return out.astype(np.float32)


def edge_face_max(edge_faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    edge_faces = np.asarray(edge_faces, dtype=np.int64)
    a = edge_faces[:, 0]
    b = edge_faces[:, 1]
    out = values[np.clip(a, 0, len(values) - 1)]
    has_two = b >= 0
    out[has_two] = np.maximum(out[has_two], values[b[has_two]])
    return out.astype(np.float32)


def edge_face_mean(edge_faces: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    edge_faces = np.asarray(edge_faces, dtype=np.int64)
    a = edge_faces[:, 0]
    b = edge_faces[:, 1]
    out = values[np.clip(a, 0, len(values) - 1)]
    has_two = b >= 0
    out[has_two] = 0.5 * (out[has_two] + values[b[has_two]])
    return out.astype(np.float32)


def simulate_collapse_gates(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_faces: list[np.ndarray],
    face_normals: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    edge_id: int,
    candidate_position: np.ndarray,
    detail_weight: float,
    tau_a: float,
    q_min_hard: float,
    q_min_soft: float,
) -> dict[str, Any]:
    """Evaluate local normal and triangle-quality gates for one edge collapse."""

    vi, vj = [int(v) for v in edge_vertices[int(edge_id)]]
    incident = np.unique(np.concatenate([vertex_faces[vi], vertex_faces[vj]])).astype(np.int64)
    if incident.size == 0:
        return {"topologyOk": False, "qualityOk": False, "normalError": float("inf"), "qualityError": float("inf"), "minQuality": 0.0}

    new_triangles: list[np.ndarray] = []
    old_normals: list[np.ndarray] = []
    topology_ok = True
    for face_id in incident:
        face = faces[int(face_id)].copy()
        has_i = bool(np.any(face == vi))
        has_j = bool(np.any(face == vj))
        if has_i and has_j:
            continue
        face[face == vi] = -1
        face[face == vj] = -1
        coords = vertices[face.clip(min=0)].copy()
        coords[face == -1] = candidate_position
        double_area = float(np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0])))
        if double_area <= 1e-12:
            topology_ok = False
            continue
        new_triangles.append(coords)
        old_normals.append(face_normals[int(face_id)])

    if not new_triangles:
        return {"topologyOk": False, "qualityOk": False, "normalError": float("inf"), "qualityError": float("inf"), "minQuality": 0.0}

    triangles = np.stack(new_triangles, axis=0)
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.divide(raw_normals, np.maximum(np.linalg.norm(raw_normals, axis=1, keepdims=True), 1e-12))
    old = np.stack(old_normals, axis=0)
    dots = np.sum(normals * old, axis=1)
    flipped = dots <= 0.0
    if np.any(flipped):
        topology_ok = False
    normal_deviation = np.maximum(0.0, 1.0 - np.clip(dots, -1.0, 1.0))
    normal_den = max(1.0 - math.cos(max(float(tau_a), 1e-6)), 1e-9)
    normal_error = float(max(float(detail_weight), 0.0) * np.max(normal_deviation) / normal_den)

    quality = triangle_quality_from_vertices(triangles)
    min_quality = float(np.min(quality)) if quality.size else 0.0
    quality_ok = bool(min_quality >= float(q_min_hard))
    quality_error = float(np.sum(np.maximum(0.0, float(q_min_soft) - quality) ** 2))
    return {
        "topologyOk": bool(topology_ok),
        "qualityOk": quality_ok,
        "normalError": normal_error,
        "qualityError": quality_error,
        "minQuality": min_quality,
    }


def simulate_flip_quality(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    edge_id: int,
) -> dict[str, Any]:
    """Evaluate whether flipping an interior edge improves local triangle quality."""

    face_a, face_b = [int(v) for v in edge_faces[int(edge_id)]]
    if face_a < 0 or face_b < 0:
        return {"valid": False, "qualityImprovement": 0.0, "oldMinQuality": 0.0, "newMinQuality": 0.0}
    va, vb = [int(v) for v in edge_vertices[int(edge_id)]]
    fa = faces[face_a]
    fb = faces[face_b]
    opp_a = [int(v) for v in fa if int(v) not in {va, vb}]
    opp_b = [int(v) for v in fb if int(v) not in {va, vb}]
    if len(opp_a) != 1 or len(opp_b) != 1 or opp_a[0] == opp_b[0]:
        return {"valid": False, "qualityImprovement": 0.0, "oldMinQuality": 0.0, "newMinQuality": 0.0}
    vc = opp_a[0]
    vd = opp_b[0]
    if len({va, vb, vc, vd}) != 4:
        return {"valid": False, "qualityImprovement": 0.0, "oldMinQuality": 0.0, "newMinQuality": 0.0}

    old_tris = vertices[np.stack([fa, fb], axis=0)]
    new_tris = np.stack(
        [
            np.stack([vertices[vc], vertices[vd], vertices[vb]], axis=0),
            np.stack([vertices[vd], vertices[vc], vertices[va]], axis=0),
        ],
        axis=0,
    )
    old_q = triangle_quality_from_vertices(old_tris)
    new_q = triangle_quality_from_vertices(new_tris)
    old_min = float(np.min(old_q))
    new_min = float(np.min(new_q))
    raw_normals = np.cross(new_tris[:, 1] - new_tris[:, 0], new_tris[:, 2] - new_tris[:, 0])
    if np.any(np.linalg.norm(raw_normals, axis=1) <= 1e-12):
        return {"valid": False, "qualityImprovement": 0.0, "oldMinQuality": old_min, "newMinQuality": 0.0}
    return {
        "valid": True,
        "qualityImprovement": float(new_min - old_min),
        "oldMinQuality": old_min,
        "newMinQuality": new_min,
    }


def scalar_to_face_colors(values: np.ndarray, valid: np.ndarray | None = None, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if valid is None:
        valid_mask = np.isfinite(values)
    else:
        valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    normalized = np.clip((values - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    mapped = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and values.ndim == 1 and rgb.shape[1] == 1:
        rgb = rgb[:, 0, :]
    colors = np.full((values.shape[0], 4), 255, dtype=np.uint8)
    colors[:, :3] = rgb
    colors[~valid_mask, :3] = np.array([38, 38, 38], dtype=np.uint8)
    return colors


def write_face_scalar_mesh(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual.face_colors = scalar_to_face_colors(values, valid, vmin=float(vmin), vmax=float(vmax))
    mesh.export(path)
