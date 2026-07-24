"""Prepare raw and cleaned hybrid point clouds from an optimized OMeGa mesh."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_SCHEMA_VERSION = 1
_EXTRACTION_METHOD = "optimized_mesh_vertices"
_SURFACE_SCHEMA_VERSION = 1
_SURFACE_EXTRACTION_METHOD = "area_sampled_voxel_surface_candidates"
_HYBRID_SCHEMA_VERSION = 1
_HYBRID_EXTRACTION_METHOD = "cleaned_vertices_with_surface_completion"


@dataclass(frozen=True)
class OmegaMeshPointCloudConfig:
    mesh_path: Path
    points_path: Path
    cache_path: Path
    summary_path: Path


@dataclass(frozen=True)
class OmegaMeshSurfacePointCloudConfig:
    mesh_path: Path
    cache_path: Path
    summary_path: Path
    target_point_count: int = 800_000
    oversample_factor: int = 3
    seed: int = 71


@dataclass(frozen=True)
class OmegaMeshHybridPointCloudConfig:
    mesh_path: Path
    points_path: Path
    cache_path: Path
    summary_path: Path
    surface_cache_path: Path
    surface_summary_path: Path
    surface_target_point_count: int = 800_000
    surface_oversample_factor: int = 3
    min_component_faces: int = 25
    min_component_area_fraction: float = 1e-5
    seed: int = 71


def source_signature(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtimeNs": int(stat.st_mtime_ns),
    }


def omega_mesh_point_cloud_status(config: OmegaMeshPointCloudConfig) -> dict[str, Any]:
    mesh_path = config.mesh_path.expanduser().resolve()
    summary = _read_json(config.summary_path)
    ready = bool(
        mesh_path.is_file()
        and config.points_path.is_file()
        and config.cache_path.is_file()
        and summary is not None
        and int(summary.get("schemaVersion", -1)) == _SCHEMA_VERSION
        and summary.get("extractionMethod") == _EXTRACTION_METHOD
        and summary.get("source") == source_signature(mesh_path)
    )
    return {
        "available": bool(mesh_path.is_file()),
        "ready": ready,
        "sourcePath": str(mesh_path),
        "pointsPath": str(config.points_path),
        "cachePath": str(config.cache_path),
        "summaryPath": str(config.summary_path),
        "pointCount": int(summary.get("outputPointCount", 0)) if ready and summary else 0,
        "summary": summary if ready else None,
    }


def ensure_omega_mesh_point_cloud(config: OmegaMeshPointCloudConfig) -> dict[str, Any]:
    status = omega_mesh_point_cloud_status(config)
    if status["ready"]:
        return dict(status["summary"])

    mesh_path = config.mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Final OMeGa mesh is missing: {mesh_path}")
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Open3D is required to extract final OMeGa mesh vertices.") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"Could not read non-empty mesh vertices from {mesh_path}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"Could not read non-empty triangle faces from {mesh_path}")

    source_count = int(vertices.shape[0])
    source_indices = np.arange(source_count, dtype=np.int64)
    finite = np.all(np.isfinite(vertices), axis=1)
    vertices = vertices[finite]
    source_indices = source_indices[finite]
    if vertices.shape[0] == 0:
        raise ValueError(f"Final OMeGa mesh contains no finite vertices: {mesh_path}")

    vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    colors_preserved = vertex_colors.shape == (source_count, 3)
    colors = (
        np.clip(np.rint(vertex_colors[finite] * 255.0), 0, 255).astype(np.uint8)
        if colors_preserved
        else np.full(vertices.shape, 255, dtype=np.uint8)
    )
    positions = vertices.astype(np.float32, copy=False)
    bounds_min = np.min(positions, axis=0)
    bounds_max = np.max(positions, axis=0)

    for path in (config.points_path, config.cache_path, config.summary_path):
        path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    temporary_cache = config.cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        positions=positions,
        colors=colors,
        source_indices=source_indices,
    )
    temporary_cache.replace(config.cache_path)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64, copy=False))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    temporary_points = config.points_path.with_name(config.points_path.stem + ".tmp.ply")
    if not o3d.io.write_point_cloud(
        str(temporary_points),
        cloud,
        write_ascii=False,
        compressed=True,
        print_progress=False,
    ):
        temporary_points.unlink(missing_ok=True)
        raise RuntimeError(f"Open3D could not write final OMeGa points: {temporary_points}")
    temporary_points.replace(config.points_path)

    summary = {
        "schemaVersion": _SCHEMA_VERSION,
        "extractionMethod": _EXTRACTION_METHOD,
        "source": source_signature(mesh_path),
        "sourceVertexCount": source_count,
        "sourceTriangleCount": int(triangles.shape[0]),
        "outputPointCount": int(positions.shape[0]),
        "removedNonFinite": source_count - int(positions.shape[0]),
        "preservesEveryFiniteMeshVertex": True,
        "surfaceResampling": False,
        "colorsPreserved": bool(colors_preserved),
        "fallbackColor": None if colors_preserved else [255, 255, 255],
        "outputBounds": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
        },
        "pointsPath": str(config.points_path.expanduser().resolve()),
        "cachePath": str(config.cache_path.expanduser().resolve()),
    }
    temporary_summary = config.summary_path.with_suffix(config.summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(config.summary_path)
    return summary


def omega_mesh_surface_point_cloud_status(
    config: OmegaMeshSurfacePointCloudConfig,
) -> dict[str, Any]:
    mesh_path = config.mesh_path.expanduser().resolve()
    summary = _read_json(config.summary_path)
    expected_parameters = {
        "targetPointCount": int(config.target_point_count),
        "oversampleFactor": int(config.oversample_factor),
        "seed": int(config.seed),
    }
    ready = bool(
        mesh_path.is_file()
        and config.cache_path.is_file()
        and summary is not None
        and int(summary.get("schemaVersion", -1)) == _SURFACE_SCHEMA_VERSION
        and summary.get("extractionMethod") == _SURFACE_EXTRACTION_METHOD
        and summary.get("source") == source_signature(mesh_path)
        and summary.get("parameters") == expected_parameters
    )
    return {
        "available": bool(mesh_path.is_file()),
        "ready": ready,
        "sourcePath": str(mesh_path),
        "cachePath": str(config.cache_path),
        "summaryPath": str(config.summary_path),
        "pointCount": int(summary.get("outputPointCount", 0)) if ready and summary else 0,
        "summary": summary if ready else None,
    }


def ensure_omega_mesh_surface_point_cloud(
    config: OmegaMeshSurfacePointCloudConfig,
) -> dict[str, Any]:
    status = omega_mesh_surface_point_cloud_status(config)
    if status["ready"]:
        return dict(status["summary"])

    target_count = max(int(config.target_point_count), 1)
    oversample_factor = max(int(config.oversample_factor), 1)
    mesh_path = config.mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Final OMeGa mesh is missing: {mesh_path}")
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Open3D is required to sample the final OMeGa mesh.") from exc

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"Could not read non-empty mesh vertices from {mesh_path}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"Could not read non-empty triangle faces from {mesh_path}")

    tri = vertices[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    valid_faces = np.flatnonzero(np.isfinite(areas) & (areas > 1e-12))
    if valid_faces.size == 0:
        raise ValueError(f"Final OMeGa mesh has no finite positive-area faces: {mesh_path}")

    candidate_count = target_count * oversample_factor
    rng = np.random.default_rng(int(config.seed))
    probabilities = areas[valid_faces] / float(np.sum(areas[valid_faces]))
    candidate_faces = valid_faces[
        rng.choice(valid_faces.shape[0], size=candidate_count, replace=True, p=probabilities)
    ]
    candidate_tri = tri[candidate_faces]
    r1 = rng.random(candidate_count)
    r2 = rng.random(candidate_count)
    sqrt_r1 = np.sqrt(r1)
    barycentric = np.stack(
        (
            1.0 - sqrt_r1,
            sqrt_r1 * (1.0 - r2),
            sqrt_r1 * r2,
        ),
        axis=1,
    )
    candidates = np.einsum("ni,nij->nj", barycentric, candidate_tri).astype(
        np.float32,
        copy=False,
    )

    voxel_size, voxel_keys, inverse = _adaptive_voxel_assignment(
        candidates,
        target_count=target_count,
        surface_area=float(np.sum(areas[valid_faces])),
    )
    voxel_count = int(np.max(inverse)) + 1
    centroids = np.column_stack(
        [
            np.bincount(inverse, weights=candidates[:, axis], minlength=voxel_count)
            for axis in range(3)
        ]
    )
    counts = np.bincount(inverse, minlength=voxel_count)
    centroids /= np.maximum(counts[:, None], 1)
    distances = np.sum((candidates - centroids[inverse]) ** 2, axis=1)
    order = np.lexsort((distances, inverse))
    first = np.r_[True, np.diff(inverse[order]) != 0]
    selected = order[first]

    positions = candidates[selected].astype(np.float32, copy=False)
    selected_faces = candidate_faces[selected].astype(np.int64, copy=False)
    vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    colors_preserved = vertex_colors.shape == vertices.shape
    if colors_preserved:
        selected_vertices = triangles[selected_faces]
        selected_barycentric = barycentric[selected]
        colors_float = np.einsum(
            "ni,nij->nj",
            selected_barycentric,
            vertex_colors[selected_vertices],
        )
        colors = np.clip(np.rint(colors_float * 255.0), 0, 255).astype(np.uint8)
    else:
        colors = np.full(positions.shape, 255, dtype=np.uint8)

    for path in (config.cache_path, config.summary_path):
        path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

    temporary_cache = config.cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        positions=positions,
        colors=colors,
        source_indices=selected_faces,
    )
    temporary_cache.replace(config.cache_path)

    component_labels, component_counts, component_areas = mesh.cluster_connected_triangles()
    del component_labels
    component_counts = np.asarray(component_counts, dtype=np.int64)
    component_areas = np.asarray(component_areas, dtype=np.float64)
    total_area = float(np.sum(component_areas))
    bounds_min = np.min(positions, axis=0)
    bounds_max = np.max(positions, axis=0)
    summary = {
        "schemaVersion": _SURFACE_SCHEMA_VERSION,
        "extractionMethod": _SURFACE_EXTRACTION_METHOD,
        "source": source_signature(mesh_path),
        "parameters": {
            "targetPointCount": target_count,
            "oversampleFactor": oversample_factor,
            "seed": int(config.seed),
        },
        "sourceVertexCount": int(vertices.shape[0]),
        "sourceTriangleCount": int(triangles.shape[0]),
        "validTriangleCount": int(valid_faces.shape[0]),
        "candidatePointCount": candidate_count,
        "outputPointCount": int(positions.shape[0]),
        "voxelSize": float(voxel_size),
        "surfaceArea": float(np.sum(areas[valid_faces])),
        "surfaceResampling": True,
        "voxelUniformization": True,
        "representative": "closest sampled surface point to each occupied-voxel centroid",
        "sourceIndexMeaning": "source triangle index",
        "componentFiltering": False,
        "connectedComponentCount": int(component_counts.shape[0]),
        "largestComponentFaceFraction": (
            float(np.max(component_counts) / triangles.shape[0])
            if component_counts.size
            else 0.0
        ),
        "tinyComponentCountAtMost10Faces": int(np.count_nonzero(component_counts <= 10)),
        "tinyComponentAreaFractionAtMost10Faces": (
            float(np.sum(component_areas[component_counts <= 10]) / total_area)
            if total_area > 0.0
            else 0.0
        ),
        "colorsPreserved": bool(colors_preserved),
        "fallbackColor": None if colors_preserved else [255, 255, 255],
        "outputBounds": {
            "min": bounds_min.astype(float).tolist(),
            "max": bounds_max.astype(float).tolist(),
        },
        "cachePath": str(config.cache_path.expanduser().resolve()),
    }
    temporary_summary = config.summary_path.with_suffix(config.summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(config.summary_path)
    return summary


def omega_mesh_hybrid_point_cloud_status(
    config: OmegaMeshHybridPointCloudConfig,
) -> dict[str, Any]:
    mesh_path = config.mesh_path.expanduser().resolve()
    summary = _read_json(config.summary_path)
    expected_parameters = _hybrid_parameters(config)
    ready = bool(
        mesh_path.is_file()
        and config.points_path.is_file()
        and config.cache_path.is_file()
        and summary is not None
        and int(summary.get("schemaVersion", -1)) == _HYBRID_SCHEMA_VERSION
        and summary.get("extractionMethod") == _HYBRID_EXTRACTION_METHOD
        and summary.get("source") == source_signature(mesh_path)
        and summary.get("parameters") == expected_parameters
    )
    return {
        "available": bool(mesh_path.is_file()),
        "ready": ready,
        "sourcePath": str(mesh_path),
        "pointsPath": str(config.points_path),
        "cachePath": str(config.cache_path),
        "summaryPath": str(config.summary_path),
        "pointCount": int(summary.get("outputPointCount", 0)) if ready and summary else 0,
        "summary": summary if ready else None,
    }


def ensure_omega_mesh_hybrid_point_cloud(
    config: OmegaMeshHybridPointCloudConfig,
) -> dict[str, Any]:
    """Keep cleaned mesh vertices and fill only surface voxels they do not occupy."""

    status = omega_mesh_hybrid_point_cloud_status(config)
    if status["ready"]:
        return dict(status["summary"])

    mesh_path = config.mesh_path.expanduser().resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Final OMeGa mesh is missing: {mesh_path}")
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Open3D is required to prepare final OMeGa points.") from exc

    surface_config = OmegaMeshSurfacePointCloudConfig(
        mesh_path=mesh_path,
        cache_path=config.surface_cache_path,
        summary_path=config.surface_summary_path,
        target_point_count=int(config.surface_target_point_count),
        oversample_factor=int(config.surface_oversample_factor),
        seed=int(config.seed),
    )
    surface_summary = ensure_omega_mesh_surface_point_cloud(surface_config)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] == 0:
        raise ValueError(f"Could not read non-empty mesh vertices from {mesh_path}")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.shape[0] == 0:
        raise ValueError(f"Could not read non-empty triangle faces from {mesh_path}")

    component_labels, component_counts, component_areas = mesh.cluster_connected_triangles()
    component_labels = np.asarray(component_labels, dtype=np.int64)
    component_counts = np.asarray(component_counts, dtype=np.int64)
    component_areas = np.asarray(component_areas, dtype=np.float64)
    total_component_area = float(np.sum(component_areas))
    minimum_area = float(config.min_component_area_fraction) * total_component_area
    keep_components = (
        (component_counts >= int(config.min_component_faces))
        | (component_areas >= minimum_area)
    )

    finite_vertices = np.all(np.isfinite(vertices), axis=1)
    triangle_vertices_finite = np.all(finite_vertices[triangles], axis=1)
    tri = vertices[triangles]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    valid_faces = triangle_vertices_finite & np.isfinite(face_areas) & (face_areas > 1e-12)
    kept_faces = valid_faces & keep_components[component_labels]
    used_vertices = np.zeros(vertices.shape[0], dtype=bool)
    used_vertices[triangles[kept_faces].reshape(-1)] = True
    retained_vertex_indices = np.flatnonzero(used_vertices)
    retained_vertices = vertices[retained_vertex_indices].astype(np.float32, copy=False)
    if retained_vertices.shape[0] == 0:
        raise ValueError(f"Mesh cleanup removed every OMeGa vertex: {mesh_path}")

    vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    colors_preserved = vertex_colors.shape == vertices.shape
    retained_colors = (
        np.clip(np.rint(vertex_colors[retained_vertex_indices] * 255.0), 0, 255).astype(np.uint8)
        if colors_preserved
        else np.full(retained_vertices.shape, 255, dtype=np.uint8)
    )

    with np.load(config.surface_cache_path) as payload:
        surface_positions = np.asarray(payload["positions"], dtype=np.float32)
        surface_colors = np.asarray(payload["colors"], dtype=np.uint8)
        surface_faces = np.asarray(payload["source_indices"], dtype=np.int64)
    valid_surface = (
        (surface_faces >= 0)
        & (surface_faces < triangles.shape[0])
        & kept_faces[np.clip(surface_faces, 0, triangles.shape[0] - 1)]
        & np.all(np.isfinite(surface_positions), axis=1)
    )
    surface_positions = surface_positions[valid_surface]
    surface_colors = surface_colors[valid_surface]
    surface_faces = surface_faces[valid_surface]

    voxel_size = float(surface_summary["voxelSize"])
    origin = np.minimum(
        np.min(retained_vertices, axis=0),
        np.min(surface_positions, axis=0),
    ).astype(np.float64)
    vertex_voxels = np.floor(
        (retained_vertices.astype(np.float64) - origin) / voxel_size
    ).astype(np.int32)
    surface_voxels = np.floor(
        (surface_positions.astype(np.float64) - origin) / voxel_size
    ).astype(np.int32)
    occupied_vertex_voxels = _structured_voxel_keys(vertex_voxels)
    structured_surface_voxels = _structured_voxel_keys(surface_voxels)
    _, first_surface = np.unique(structured_surface_voxels, return_index=True)
    first_surface = np.sort(first_surface)
    add_surface = ~np.isin(
        structured_surface_voxels[first_surface],
        np.unique(occupied_vertex_voxels),
        assume_unique=False,
    )
    completion_indices = first_surface[add_surface]
    completion_positions = surface_positions[completion_indices]
    completion_colors = surface_colors[completion_indices]
    completion_faces = surface_faces[completion_indices]

    positions = np.concatenate((retained_vertices, completion_positions), axis=0)
    colors = np.concatenate((retained_colors, completion_colors), axis=0)
    point_kinds = np.concatenate(
        (
            np.zeros(retained_vertices.shape[0], dtype=np.uint8),
            np.ones(completion_positions.shape[0], dtype=np.uint8),
        )
    )
    source_vertex_indices = np.concatenate(
        (
            retained_vertex_indices.astype(np.int64, copy=False),
            np.full(completion_positions.shape[0], -1, dtype=np.int64),
        )
    )
    source_face_indices = np.concatenate(
        (
            np.full(retained_vertices.shape[0], -1, dtype=np.int64),
            completion_faces.astype(np.int64, copy=False),
        )
    )

    for path in (config.points_path, config.cache_path, config.summary_path):
        path.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    temporary_cache = config.cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_cache,
        positions=positions,
        colors=colors,
        source_indices=np.arange(positions.shape[0], dtype=np.int64),
        point_kinds=point_kinds,
        source_vertex_indices=source_vertex_indices,
        source_face_indices=source_face_indices,
    )
    temporary_cache.replace(config.cache_path)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(positions.astype(np.float64, copy=False))
    cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    temporary_points = config.points_path.with_name(config.points_path.stem + ".tmp.ply")
    if not o3d.io.write_point_cloud(
        str(temporary_points),
        cloud,
        write_ascii=False,
        compressed=True,
        print_progress=False,
    ):
        temporary_points.unlink(missing_ok=True)
        raise RuntimeError(f"Open3D could not write hybrid OMeGa points: {temporary_points}")
    temporary_points.replace(config.points_path)

    kept_component_count = int(np.count_nonzero(keep_components))
    retained_area = float(np.sum(face_areas[kept_faces]))
    valid_area = float(np.sum(face_areas[valid_faces]))
    summary = {
        "schemaVersion": _HYBRID_SCHEMA_VERSION,
        "extractionMethod": _HYBRID_EXTRACTION_METHOD,
        "source": source_signature(mesh_path),
        "parameters": _hybrid_parameters(config),
        "sourceVertexCount": int(vertices.shape[0]),
        "sourceTriangleCount": int(triangles.shape[0]),
        "connectedComponentCount": int(component_counts.shape[0]),
        "keptComponentCount": kept_component_count,
        "removedComponentCount": int(component_counts.shape[0] - kept_component_count),
        "keptTriangleCount": int(np.count_nonzero(kept_faces)),
        "removedTriangleCount": int(triangles.shape[0] - np.count_nonzero(kept_faces)),
        "retainedSurfaceAreaFraction": retained_area / valid_area if valid_area > 0.0 else 0.0,
        "retainedMeshVertexCount": int(retained_vertices.shape[0]),
        "removedUnreferencedOrFilteredVertexCount": int(
            vertices.shape[0] - retained_vertices.shape[0]
        ),
        "surfaceCompletionCandidateCount": int(surface_positions.shape[0]),
        "surfaceCompletionPointCount": int(completion_positions.shape[0]),
        "outputPointCount": int(positions.shape[0]),
        "nativeVertexFraction": float(retained_vertices.shape[0] / positions.shape[0]),
        "voxelSize": voxel_size,
        "cleanupPolicy": (
            "retain finite positive-area faces whose component has enough faces OR enough area; "
            "discard vertices unreferenced by retained faces"
        ),
        "completionPolicy": (
            "retain every cleaned native vertex; add one sampled surface point only in a voxel "
            "with no retained native vertex"
        ),
        "pointKindMeaning": {"0": "retained native mesh vertex", "1": "surface completion"},
        "colorsPreserved": bool(colors_preserved),
        "fallbackColor": None if colors_preserved else [255, 255, 255],
        "pointsPath": str(config.points_path.expanduser().resolve()),
        "cachePath": str(config.cache_path.expanduser().resolve()),
    }
    temporary_summary = config.summary_path.with_suffix(config.summary_path.suffix + ".tmp")
    temporary_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary_summary.replace(config.summary_path)
    return summary


def _hybrid_parameters(config: OmegaMeshHybridPointCloudConfig) -> dict[str, Any]:
    return {
        "surfaceTargetPointCount": int(config.surface_target_point_count),
        "surfaceOversampleFactor": int(config.surface_oversample_factor),
        "minComponentFaces": int(config.min_component_faces),
        "minComponentAreaFraction": float(config.min_component_area_fraction),
        "seed": int(config.seed),
    }


def _structured_voxel_keys(keys: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(keys, dtype=np.int32)
    return contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).reshape(-1)


def _adaptive_voxel_assignment(
    points: np.ndarray,
    *,
    target_count: int,
    surface_area: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    origin = np.min(points, axis=0).astype(np.float64)
    target = max(int(target_count), 1)
    initial = max(np.sqrt(max(float(surface_area), 1e-12) / target), 1e-6)
    low = initial * 0.25
    high = initial * 4.0

    def assignment(voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        keys = np.floor((points.astype(np.float64) - origin) / voxel_size).astype(np.int32)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        return unique, inverse.astype(np.int32, copy=False)

    low_keys, low_inverse = assignment(low)
    for _ in range(8):
        if low_keys.shape[0] >= target:
            break
        low *= 0.5
        low_keys, low_inverse = assignment(low)

    high_keys, high_inverse = assignment(high)
    for _ in range(8):
        if high_keys.shape[0] <= target:
            break
        high *= 2.0
        high_keys, high_inverse = assignment(high)

    candidates = [
        (abs(int(low_keys.shape[0]) - target), low, low_keys, low_inverse),
        (abs(int(high_keys.shape[0]) - target), high, high_keys, high_inverse),
    ]
    for _ in range(10):
        middle = float(np.sqrt(low * high))
        keys, inverse = assignment(middle)
        candidates.append((abs(int(keys.shape[0]) - target), middle, keys, inverse))
        if keys.shape[0] > target:
            low = middle
        else:
            high = middle
    _, voxel_size, keys, inverse = min(candidates, key=lambda row: row[0])
    return float(voxel_size), keys, inverse


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None
