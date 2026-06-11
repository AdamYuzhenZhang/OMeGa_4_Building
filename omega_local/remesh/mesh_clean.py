"""Minimal preprocessing for OMeGa local remeshing.

This module implements the Phase 1 cleanup from REMESH_REDESIGN.md. It is a
topology-safety pass, not a geometry repair pass:

- exact duplicate vertices are merged;
- duplicate faces are removed by vertex set, preserving the first orientation;
- zero-area faces are removed;
- isolated vertices are compacted away;
- bow-tie vertices are split into separate one-ring fans;
- holes, open boundaries, and components are preserved.

No tolerance-based vertex snapping, hole filling, smoothing, or component
filtering happens here.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh


ProgressFn = Callable[[str], None]

FACE_STATUS_KEPT = 0
FACE_STATUS_INVALID_INDEX = 1
FACE_STATUS_NONFINITE_VERTEX = 2
FACE_STATUS_ZERO_AREA = 3
FACE_STATUS_DUPLICATE = 4

FACE_STATUS_NAMES = {
    FACE_STATUS_KEPT: "kept",
    FACE_STATUS_INVALID_INDEX: "invalid_index",
    FACE_STATUS_NONFINITE_VERTEX: "nonfinite_vertex",
    FACE_STATUS_ZERO_AREA: "zero_area",
    FACE_STATUS_DUPLICATE: "duplicate",
}

FACE_STATUS_COLORS = {
    FACE_STATUS_KEPT: (160, 160, 160, 255),
    FACE_STATUS_INVALID_INDEX: (210, 32, 39, 255),
    FACE_STATUS_NONFINITE_VERTEX: (240, 125, 35, 255),
    FACE_STATUS_ZERO_AREA: (255, 205, 0, 255),
    FACE_STATUS_DUPLICATE: (118, 42, 131, 255),
}


@dataclass(frozen=True)
class MeshStats:
    vertices: int
    faces: int
    unique_edges: int
    boundary_edges: int
    nonmanifold_edge_groups: int
    vertices_with_multiple_fans: int
    max_vertex_fan_count: int
    connected_face_components: int
    euler_characteristic: int
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_extent: list[float]
    surface_area: float
    face_area_quantiles: list[float]
    edge_length_quantiles: list[float]


@dataclass(frozen=True)
class MeshPreprocessConfig:
    input_mesh: Path
    output_dir: Path
    output_name: str = "preclean_mesh.ply"
    summary_name: str = "preclean_summary.json"
    diagnostics_name: str = "preclean_diagnostics.npz"
    area_epsilon: float = 0.0
    split_nonmanifold_vertices: bool = True
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class MeshPreprocessResult:
    input_mesh: Path
    output_mesh: Path
    summary_json: Path
    diagnostics_npz: Path
    debug_meshes: dict[str, Path]
    input_stats: MeshStats
    output_stats: MeshStats
    summary: dict[str, Any]


@dataclass
class _MeshState:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_source: np.ndarray
    face_source: np.ndarray


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(int(size), dtype=np.int64)
        self.rank = np.zeros((int(size),), dtype=np.int8)

    def find(self, item: int) -> int:
        item = int(item)
        parent = int(self.parent[item])
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = int(self.rank[root_a])
        rank_b = int(self.rank[root_b])
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _latest_step_file(folder: Path, prefix: str) -> Path | None:
    found: list[tuple[int, Path]] = []
    for path in folder.glob(f"{prefix}_*_rank0.ply"):
        try:
            found.append((int(path.name.split("_")[1]), path))
        except (IndexError, ValueError):
            continue
    return sorted(found)[-1][1] if found else None


def resolve_latest_omega_mesh(model_dir: Path, requested: Path | None = None, iteration: int = -1) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if requested is not None:
        path = requested.expanduser()
        candidates = [path.resolve()] if path.is_absolute() else [
            path.resolve(),
            (model_dir / path).resolve(),
            (_repo_root() / path).resolve(),
        ]
        for resolved in candidates:
            if resolved.exists():
                return resolved
        raise FileNotFoundError(f"Requested mesh does not exist. Tried: {[str(candidate) for candidate in candidates]}")

    plys_dir = model_dir / "plys"
    if int(iteration) > 0:
        for step in (int(iteration), int(iteration) - 1, int(iteration) + 1):
            if step < 0:
                continue
            candidate = plys_dir / f"mesh_{step}_rank0.ply"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Requested mesh iteration was not found under {plys_dir}: {iteration}")

    latest = _latest_step_file(plys_dir, "mesh")
    if latest is not None:
        return latest
    raise FileNotFoundError(f"No mesh_*_rank0.ply found under {plys_dir}")


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


def _stable_unique_rows(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.ascontiguousarray(array)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array.")
    if array.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    key_dtype = np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    keys = array.view(key_dtype).reshape(-1)
    _, first, inverse, counts = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first, kind="mergesort")
    sorted_unique_to_stable = np.empty_like(order)
    sorted_unique_to_stable[order] = np.arange(len(order), dtype=np.int64)
    stable_first = first[order].astype(np.int64, copy=False)
    stable_inverse = sorted_unique_to_stable[inverse].astype(np.int64, copy=False)
    stable_counts = counts[order].astype(np.int64, copy=False)
    return stable_first, stable_inverse, stable_counts


def _triangle_double_area(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float64)
    triangles = vertices[faces]
    return np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return 0.5 * _triangle_double_area(vertices, faces)


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float64)
    triangles = vertices[faces]
    return np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2).reshape(-1)


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
    q25, q50, q75, q90, q99 = np.quantile(arr, [0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "q25": float(q25),
        "median": float(q50),
        "mean": float(np.mean(arr)),
        "q75": float(q75),
        "q90": float(q90),
        "q99": float(q99),
        "max": float(np.max(arr)),
    }


def _quantile_list(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [0.0] * 6
    return [float(v) for v in np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99])]


def _edge_groups(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(faces) == 0:
        empty_edges = np.zeros((0, 2), dtype=np.int64)
        empty_int = np.zeros((0,), dtype=np.int64)
        return empty_edges, empty_int, empty_int, empty_int
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    edge_faces = np.tile(np.arange(len(faces), dtype=np.int64), 3)
    sorted_edges = np.sort(edges, axis=1)
    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    sorted_edges = sorted_edges[order]
    sorted_face_ids = edge_faces[order]
    changed = np.ones((len(sorted_edges),), dtype=bool)
    changed[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(changed).astype(np.int64)
    ends = np.concatenate([starts[1:], np.asarray([len(sorted_edges)], dtype=np.int64)])
    unique_edges = sorted_edges[starts]
    return unique_edges, starts, ends, sorted_face_ids


def _face_component_count(face_count: int, starts: np.ndarray, ends: np.ndarray, sorted_face_ids: np.ndarray) -> int:
    if face_count == 0:
        return 0
    union = _UnionFind(face_count)
    for start, end in zip(starts, ends):
        incident = sorted_face_ids[int(start) : int(end)]
        if len(incident) <= 1:
            continue
        first = int(incident[0])
        for other in incident[1:]:
            union.union(first, int(other))
    roots = {union.find(face_id) for face_id in range(face_count)}
    return len(roots)


def _incident_faces(vertex_count: int, faces: np.ndarray) -> list[list[int]]:
    incident: list[list[int]] = [[] for _ in range(int(vertex_count))]
    for face_id, face in enumerate(faces):
        for vertex_id in face:
            incident[int(vertex_id)].append(int(face_id))
    return incident


def _fan_components_for_vertex(vertex_id: int, incident: list[int], faces: np.ndarray) -> list[list[int]]:
    if len(incident) <= 1:
        return [list(incident)] if incident else []

    local_index = {int(face_id): local for local, face_id in enumerate(incident)}
    union = _UnionFind(len(incident))
    edge_buckets: dict[int, list[int]] = {}
    for face_id in incident:
        face = faces[int(face_id)]
        others = [int(v) for v in face if int(v) != int(vertex_id)]
        for other in others:
            edge_buckets.setdefault(other, []).append(local_index[int(face_id)])

    for bucket in edge_buckets.values():
        if len(bucket) <= 1:
            continue
        first = int(bucket[0])
        for other in bucket[1:]:
            union.union(first, int(other))

    groups: dict[int, list[int]] = {}
    for local, face_id in enumerate(incident):
        groups.setdefault(union.find(local), []).append(int(face_id))
    components = list(groups.values())
    components.sort(key=lambda group: min(group))
    return components


def _vertex_fan_counts(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    incident = _incident_faces(vertex_count, faces)
    counts = np.zeros((int(vertex_count),), dtype=np.int64)
    for vertex_id, face_ids in enumerate(incident):
        counts[vertex_id] = len(_fan_components_for_vertex(vertex_id, face_ids, faces))
    return counts


def mesh_stats(vertices: np.ndarray, faces: np.ndarray) -> MeshStats:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0:
        bbox_min = bbox_max = bbox_extent = np.zeros((3,), dtype=np.float64)
    else:
        bbox_min = np.nanmin(vertices, axis=0)
        bbox_max = np.nanmax(vertices, axis=0)
        bbox_extent = bbox_max - bbox_min
    areas = _face_areas(vertices, faces)
    edge_lengths = _edge_lengths(vertices, faces)
    unique_edges, starts, ends, sorted_face_ids = _edge_groups(faces)
    edge_counts = ends - starts
    fan_counts = _vertex_fan_counts(len(vertices), faces)
    return MeshStats(
        vertices=int(len(vertices)),
        faces=int(len(faces)),
        unique_edges=int(len(unique_edges)),
        boundary_edges=int(np.count_nonzero(edge_counts == 1)),
        nonmanifold_edge_groups=int(np.count_nonzero(edge_counts > 2)),
        vertices_with_multiple_fans=int(np.count_nonzero(fan_counts > 1)),
        max_vertex_fan_count=int(np.max(fan_counts)) if len(fan_counts) else 0,
        connected_face_components=int(_face_component_count(len(faces), starts, ends, sorted_face_ids)),
        euler_characteristic=int(len(vertices) - len(unique_edges) + len(faces)),
        bbox_min=[float(v) for v in bbox_min],
        bbox_max=[float(v) for v in bbox_max],
        bbox_extent=[float(v) for v in bbox_extent],
        surface_area=float(np.sum(areas)),
        face_area_quantiles=_quantile_list(areas),
        edge_length_quantiles=_quantile_list(edge_lengths),
    )


def _filter_faces(
    state: _MeshState,
    keep: np.ndarray,
    *,
    original_face_status: np.ndarray,
    removed_status: int,
) -> _MeshState:
    keep = np.asarray(keep, dtype=bool)
    removed_sources = state.face_source[~keep]
    original_face_status[removed_sources] = int(removed_status)
    return _MeshState(
        vertices=state.vertices,
        faces=state.faces[keep],
        vertex_source=state.vertex_source,
        face_source=state.face_source[keep],
    )


def _remove_invalid_faces(state: _MeshState, original_face_status: np.ndarray) -> tuple[_MeshState, dict[str, int]]:
    faces = state.faces
    vertex_count = len(state.vertices)
    index_valid = np.all((faces >= 0) & (faces < vertex_count), axis=1)
    nonfinite_vertex = np.zeros((len(faces),), dtype=bool)
    if np.any(index_valid):
        finite_vertices = np.isfinite(state.vertices).all(axis=1)
        valid_face_ids = np.nonzero(index_valid)[0]
        nonfinite_vertex[valid_face_ids] = ~np.all(finite_vertices[faces[valid_face_ids]], axis=1)

    original_face_status[state.face_source[~index_valid]] = FACE_STATUS_INVALID_INDEX
    original_face_status[state.face_source[index_valid & nonfinite_vertex]] = FACE_STATUS_NONFINITE_VERTEX
    keep = index_valid & ~nonfinite_vertex
    return (
        _MeshState(
            vertices=state.vertices,
            faces=state.faces[keep],
            vertex_source=state.vertex_source,
            face_source=state.face_source[keep],
        ),
        {
            "invalidIndexFacesRemoved": int(np.count_nonzero(~index_valid)),
            "nonfiniteVertexFacesRemoved": int(np.count_nonzero(index_valid & nonfinite_vertex)),
        },
    )


def _merge_exact_duplicate_vertices(state: _MeshState) -> tuple[_MeshState, dict[str, Any]]:
    first, inverse, counts = _stable_unique_rows(state.vertices)
    if len(first) == len(state.vertices):
        return state, {
            "duplicateVerticesMerged": 0,
            "duplicateVertexGroups": 0,
            "maxDuplicateVertexGroupSize": 1,
        }
    return (
        _MeshState(
            vertices=state.vertices[first],
            faces=inverse[state.faces],
            vertex_source=state.vertex_source[first],
            face_source=state.face_source,
        ),
        {
            "duplicateVerticesMerged": int(len(state.vertices) - len(first)),
            "duplicateVertexGroups": int(np.count_nonzero(counts > 1)),
            "maxDuplicateVertexGroupSize": int(np.max(counts)) if len(counts) else 0,
        },
    )


def _remove_zero_area_faces(
    state: _MeshState,
    original_face_status: np.ndarray,
    *,
    area_epsilon: float,
) -> tuple[_MeshState, dict[str, int]]:
    repeated_vertex = (
        (state.faces[:, 0] == state.faces[:, 1])
        | (state.faces[:, 1] == state.faces[:, 2])
        | (state.faces[:, 2] == state.faces[:, 0])
    )
    double_area = _triangle_double_area(state.vertices, state.faces)
    threshold = max(float(area_epsilon), 0.0) * 2.0
    if threshold == 0.0:
        zero_area = repeated_vertex | (double_area == 0.0)
    else:
        zero_area = repeated_vertex | (double_area <= threshold)
    return (
        _filter_faces(state, ~zero_area, original_face_status=original_face_status, removed_status=FACE_STATUS_ZERO_AREA),
        {"zeroAreaFacesRemoved": int(np.count_nonzero(zero_area))},
    )


def _remove_duplicate_faces(state: _MeshState, original_face_status: np.ndarray) -> tuple[_MeshState, dict[str, int]]:
    if len(state.faces) == 0:
        return state, {"duplicateFacesRemoved": 0, "duplicateFaceGroups": 0}
    sorted_faces = np.sort(state.faces, axis=1)
    first, _, counts = _stable_unique_rows(sorted_faces)
    keep = np.zeros((len(state.faces),), dtype=bool)
    keep[first] = True
    return (
        _filter_faces(state, keep, original_face_status=original_face_status, removed_status=FACE_STATUS_DUPLICATE),
        {
            "duplicateFacesRemoved": int(np.count_nonzero(~keep)),
            "duplicateFaceGroups": int(np.count_nonzero(counts > 1)),
        },
    )


def _remove_isolated_vertices(state: _MeshState) -> tuple[_MeshState, dict[str, int]]:
    if len(state.faces) == 0:
        return (
            _MeshState(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int64),
                vertex_source=np.zeros((0,), dtype=np.int64),
                face_source=state.face_source,
            ),
            {"isolatedVerticesRemoved": int(len(state.vertices))},
        )
    used = np.zeros((len(state.vertices),), dtype=bool)
    used[np.unique(state.faces.reshape(-1))] = True
    remap = np.full((len(state.vertices),), -1, dtype=np.int64)
    remap[used] = np.arange(np.count_nonzero(used), dtype=np.int64)
    return (
        _MeshState(
            vertices=state.vertices[used],
            faces=remap[state.faces],
            vertex_source=state.vertex_source[used],
            face_source=state.face_source,
        ),
        {"isolatedVerticesRemoved": int(np.count_nonzero(~used))},
    )


def _split_nonmanifold_vertex_fans(state: _MeshState) -> tuple[_MeshState, dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    vertices = state.vertices.tolist()
    vertex_source = state.vertex_source.tolist()
    faces = state.faces.copy()
    incident = _incident_faces(len(vertices), faces)
    face_touched = np.zeros((len(faces),), dtype=bool)
    created_vertex_ids: list[int] = []
    created_vertex_sources: list[int] = []
    split_source_vertices: list[int] = []
    split_existing_vertices: list[int] = []
    split_fan_counts: list[int] = []
    split_incident_face_counts: list[int] = []

    for vertex_id, face_ids in enumerate(incident):
        components = _fan_components_for_vertex(vertex_id, face_ids, faces)
        if len(components) <= 1:
            continue
        components.sort(key=lambda group: int(np.min(state.face_source[group])))
        split_existing_vertices.append(int(vertex_id))
        split_source_vertices.append(int(state.vertex_source[vertex_id]))
        split_fan_counts.append(int(len(components)))
        split_incident_face_counts.append(int(len(face_ids)))
        for component in components:
            face_touched[np.asarray(component, dtype=np.int64)] = True
        for component in components[1:]:
            new_vertex_id = len(vertices)
            vertices.append(list(state.vertices[vertex_id]))
            vertex_source.append(int(state.vertex_source[vertex_id]))
            created_vertex_ids.append(new_vertex_id)
            created_vertex_sources.append(int(state.vertex_source[vertex_id]))
            for face_id in component:
                local = np.nonzero(faces[int(face_id)] == int(vertex_id))[0]
                if len(local) != 1:
                    raise RuntimeError("Degenerate face reached vertex fan split.")
                faces[int(face_id), int(local[0])] = new_vertex_id

    diagnostics = {
        "splitExistingVertices": np.asarray(split_existing_vertices, dtype=np.int64),
        "splitSourceVertices": np.asarray(split_source_vertices, dtype=np.int64),
        "splitFanCounts": np.asarray(split_fan_counts, dtype=np.int64),
        "splitIncidentFaceCounts": np.asarray(split_incident_face_counts, dtype=np.int64),
        "splitCreatedVertices": np.asarray(created_vertex_ids, dtype=np.int64),
        "splitCreatedVertexSources": np.asarray(created_vertex_sources, dtype=np.int64),
    }
    new_state = _MeshState(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=faces,
        vertex_source=np.asarray(vertex_source, dtype=np.int64),
        face_source=state.face_source,
    )
    return (
        new_state,
        {
            "verticesSplitIntoFans": int(len(split_existing_vertices)),
            "verticesCreatedByFanSplit": int(len(created_vertex_ids)),
            "facesTouchedByFanSplit": int(np.count_nonzero(face_touched)),
            "maxFanCountBeforeSplit": int(max(split_fan_counts)) if split_fan_counts else 1,
        },
        face_touched,
        diagnostics,
    )


def _export_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), process=False)
    if face_colors is not None and len(face_colors) == len(faces):
        mesh.visual.face_colors = np.asarray(face_colors, dtype=np.uint8)
    mesh.export(path)


def _write_debug_meshes(
    *,
    output_dir: Path,
    original_vertices: np.ndarray,
    original_faces: np.ndarray,
    original_face_status: np.ndarray,
    preclean_vertices: np.ndarray,
    preclean_faces: np.ndarray,
    preclean_face_changed_by_split: np.ndarray,
) -> dict[str, Path]:
    debug_dir = output_dir / "debug_meshes"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_paths: dict[str, Path] = {}

    changed_colors = np.tile(np.asarray([[160, 160, 160, 255]], dtype=np.uint8), (len(preclean_faces), 1))
    changed_colors[np.asarray(preclean_face_changed_by_split, dtype=bool)] = np.asarray([0, 114, 189, 255], dtype=np.uint8)
    changed_path = debug_dir / "preclean_changed_faces.ply"
    _export_mesh(changed_path, preclean_vertices, preclean_faces, changed_colors)
    debug_paths["changedFacesMesh"] = changed_path

    status = np.asarray(original_face_status, dtype=np.int64)
    removed = status != FACE_STATUS_KEPT
    valid_index = np.all((original_faces >= 0) & (original_faces < len(original_vertices)), axis=1)
    finite_face = np.zeros((len(original_faces),), dtype=bool)
    if np.any(valid_index):
        finite_vertices = np.isfinite(original_vertices).all(axis=1)
        finite_face[valid_index] = np.all(finite_vertices[original_faces[valid_index]], axis=1)
    renderable_removed = removed & valid_index & finite_face
    if np.any(renderable_removed):
        removed_faces = original_faces[renderable_removed]
        removed_colors = np.asarray([FACE_STATUS_COLORS[int(code)] for code in status[renderable_removed]], dtype=np.uint8)
        removed_path = debug_dir / "preclean_removed_faces.ply"
        _export_mesh(removed_path, original_vertices, removed_faces, removed_colors)
        debug_paths["removedFacesMesh"] = removed_path

    return debug_paths


def preprocess_mesh(config: MeshPreprocessConfig, progress: ProgressFn = _progress_default) -> MeshPreprocessResult:
    input_mesh = config.input_mesh.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_mesh = output_dir / config.output_name
    summary_json = output_dir / config.summary_name
    diagnostics_npz = output_dir / config.diagnostics_name

    if output_mesh.exists() and not config.overwrite:
        raise FileExistsError(f"Output mesh already exists: {output_mesh}. Re-run with --overwrite.")
    if summary_json.exists() and not config.overwrite:
        raise FileExistsError(f"Summary already exists: {summary_json}. Re-run with --overwrite.")

    progress(f"[preprocess] Loading mesh: {input_mesh}")
    original_vertices, original_faces = load_mesh_arrays(input_mesh)
    input_stats = mesh_stats(original_vertices, original_faces)
    original_face_status = np.zeros((len(original_faces),), dtype=np.int16)

    state = _MeshState(
        vertices=original_vertices,
        faces=original_faces,
        vertex_source=np.arange(len(original_vertices), dtype=np.int64),
        face_source=np.arange(len(original_faces), dtype=np.int64),
    )

    operations: dict[str, Any] = {}
    progress("[preprocess] Removing invalid face references and non-finite face references")
    state, op = _remove_invalid_faces(state, original_face_status)
    operations.update(op)

    progress("[preprocess] Merging exact duplicate vertices")
    state, op = _merge_exact_duplicate_vertices(state)
    operations.update(op)

    progress("[preprocess] Removing zero-area faces")
    state, op = _remove_zero_area_faces(state, original_face_status, area_epsilon=float(config.area_epsilon))
    operations.update(op)

    progress("[preprocess] Removing duplicate faces by vertex set")
    state, op = _remove_duplicate_faces(state, original_face_status)
    operations.update(op)

    progress("[preprocess] Removing isolated vertices")
    state, op = _remove_isolated_vertices(state)
    operations.update(op)

    pre_split_stats = mesh_stats(state.vertices, state.faces)
    preclean_face_changed_by_split = np.zeros((len(state.faces),), dtype=bool)
    split_diagnostics: dict[str, np.ndarray] = {
        "splitExistingVertices": np.zeros((0,), dtype=np.int64),
        "splitSourceVertices": np.zeros((0,), dtype=np.int64),
        "splitFanCounts": np.zeros((0,), dtype=np.int64),
        "splitIncidentFaceCounts": np.zeros((0,), dtype=np.int64),
        "splitCreatedVertices": np.zeros((0,), dtype=np.int64),
        "splitCreatedVertexSources": np.zeros((0,), dtype=np.int64),
    }
    if config.split_nonmanifold_vertices:
        progress("[preprocess] Splitting bow-tie vertices into one-ring fans")
        state, op, preclean_face_changed_by_split, split_diagnostics = _split_nonmanifold_vertex_fans(state)
        operations.update(op)
    else:
        operations.update(
            {
                "verticesSplitIntoFans": 0,
                "verticesCreatedByFanSplit": 0,
                "facesTouchedByFanSplit": 0,
                "maxFanCountBeforeSplit": int(pre_split_stats.max_vertex_fan_count),
            }
        )

    output_stats = mesh_stats(state.vertices, state.faces)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"[preprocess] Writing mesh: {output_mesh}")
    _export_mesh(output_mesh, state.vertices, state.faces)

    debug_meshes: dict[str, Path] = {}
    if config.write_debug_meshes:
        progress("[preprocess] Writing debug meshes")
        debug_meshes = _write_debug_meshes(
            output_dir=output_dir,
            original_vertices=original_vertices,
            original_faces=original_faces,
            original_face_status=original_face_status,
            preclean_vertices=state.vertices,
            preclean_faces=state.faces,
            preclean_face_changed_by_split=preclean_face_changed_by_split,
        )

    progress(f"[preprocess] Writing diagnostics: {diagnostics_npz}")
    np.savez_compressed(
        diagnostics_npz,
        original_face_status=original_face_status.astype(np.int16, copy=False),
        preclean_face_source=state.face_source.astype(np.int64, copy=False),
        preclean_vertex_source=state.vertex_source.astype(np.int64, copy=False),
        preclean_face_changed_by_split=preclean_face_changed_by_split.astype(bool, copy=False),
        preclean_vertex_created_by_split=np.isin(
            np.arange(len(state.vertices), dtype=np.int64),
            split_diagnostics["splitCreatedVertices"],
        ),
        **split_diagnostics,
    )

    status_counts = {
        FACE_STATUS_NAMES[int(code)]: int(np.count_nonzero(original_face_status == int(code)))
        for code in sorted(FACE_STATUS_NAMES)
    }
    summary = {
        "stageName": "omega_local.remesh.mesh_clean.preprocess",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "inputMesh": str(input_mesh),
        "outputMesh": str(output_mesh),
        "operationOrder": [
            "remove invalid face references",
            "remove faces referencing non-finite vertices",
            "merge exact duplicate vertices",
            "remove zero-area faces",
            "remove duplicate faces by vertex set",
            "remove isolated vertices",
            "split bow-tie vertices into one-ring fans",
        ],
        "nonGoals": [
            "no tolerance-based vertex snapping",
            "no smoothing",
            "no hole filling",
            "no component deletion",
            "no watertightness enforcement",
        ],
        "areaEpsilon": float(config.area_epsilon),
        "splitNonmanifoldVertices": bool(config.split_nonmanifold_vertices),
        "operations": operations,
        "faceStatusCounts": status_counts,
        "inputStats": asdict(input_stats),
        "preFanSplitStats": asdict(pre_split_stats),
        "outputStats": asdict(output_stats),
        "outputs": {
            "precleanMesh": str(output_mesh),
            "summaryJson": str(summary_json),
            "diagnosticsNpz": str(diagnostics_npz),
            "debugMeshes": {key: str(path) for key, path in debug_meshes.items()},
        },
    }
    progress(f"[preprocess] Writing summary: {summary_json}")
    _write_json(summary_json, summary)
    progress(
        "[preprocess] Done: "
        f"V {input_stats.vertices}->{output_stats.vertices}, "
        f"F {input_stats.faces}->{output_stats.faces}, "
        f"multi-fan vertices {input_stats.vertices_with_multiple_fans}->{output_stats.vertices_with_multiple_fans}"
    )

    return MeshPreprocessResult(
        input_mesh=input_mesh,
        output_mesh=output_mesh,
        summary_json=summary_json,
        diagnostics_npz=diagnostics_npz,
        debug_meshes=debug_meshes,
        input_stats=input_stats,
        output_stats=output_stats,
        summary=summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess an OMeGa mesh for local remeshing.")
    parser.add_argument("--model-dir", type=Path, default=None, help="OMeGa model/result directory.")
    parser.add_argument("--mesh", "--input-mesh", dest="mesh", type=Path, default=None, help="Mesh to preprocess.")
    parser.add_argument("--iteration", type=int, default=-1, help="Mesh iteration to use when --mesh is omitted.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/local.")
    parser.add_argument("--output-name", default="preclean_mesh.ply")
    parser.add_argument("--summary-name", default="preclean_summary.json")
    parser.add_argument("--diagnostics-name", default="preclean_diagnostics.npz")
    parser.add_argument(
        "--area-epsilon",
        type=float,
        default=0.0,
        help="Area threshold in square world units. Default 0 removes only exactly zero-area triangles.",
    )
    parser.add_argument(
        "--skip-nonmanifold-vertex-split",
        action="store_true",
        help="Disable bow-tie vertex fan splitting.",
    )
    parser.add_argument("--no-debug-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.model_dir is None and args.mesh is None:
        raise SystemExit("Pass --model-dir, --mesh, or both.")
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir is not None else None
    if args.mesh is None:
        if model_dir is None:
            raise SystemExit("--model-dir is required when --mesh is omitted.")
        input_mesh = resolve_latest_omega_mesh(model_dir, iteration=int(args.iteration))
    else:
        input_mesh = resolve_latest_omega_mesh(model_dir, requested=args.mesh) if model_dir is not None else args.mesh.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif model_dir is not None:
        output_dir = model_dir / "remesh" / "local"
    else:
        output_dir = input_mesh.parent / "remesh" / "local"

    preprocess_mesh(
        MeshPreprocessConfig(
            input_mesh=input_mesh,
            output_dir=output_dir,
            output_name=str(args.output_name),
            summary_name=str(args.summary_name),
            diagnostics_name=str(args.diagnostics_name),
            area_epsilon=float(args.area_epsilon),
            split_nonmanifold_vertices=not bool(args.skip_nonmanifold_vertex_split),
            write_debug_meshes=not bool(args.no_debug_meshes),
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
