"""Post-hoc structure-aware remeshing for OMeGa meshes.

The first remeshing target is conservative: keep OMeGa's reconstructed surface
as the reference, simplify it with a planar-aware quadric error metric, and use
local normal variation as a vertex quality field so creases, sculptural detail,
and open boundaries survive more strongly than broad planar interiors.

This is intentionally a post-processing utility. It does not alter checkpoints,
training configs, or the baseline trainer.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
import pymeshlab as pml


@dataclass
class MeshStats:
    vertices: int
    faces: int
    bbox_min: list[float]
    bbox_max: list[float]
    bbox_extent: list[float]
    surface_area: float
    face_area_quantiles: list[float]
    edge_length_quantiles: list[float]
    edge_manifold: bool
    vertex_manifold: bool
    watertight: bool


@dataclass
class RemeshSummary:
    input_mesh: str
    output_mesh: str
    mode: str
    target_faces: int
    preserve_boundary: bool
    boundary_weight: float
    preserve_normal: bool
    preserve_topology: bool
    planar_quadric: bool
    planar_weight: float
    quality_weight: bool
    crease_angle_deg: float
    boundary_quality: float
    min_quality: float
    quality_gamma: float
    isotropic_iterations: int
    isotropic_target_length_m: float
    input_stats: MeshStats
    cleaned_stats: MeshStats
    output_stats: MeshStats
    quality_stats: dict[str, float]


def _as_float_list(values: np.ndarray) -> list[float]:
    return [float(v) for v in values]


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float64)
    triangles = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )


def _edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float64)
    triangles = vertices[faces]
    return np.linalg.norm(
        triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]],
        axis=2,
    ).reshape(-1)


def _mesh_stats(mesh: o3d.geometry.TriangleMesh) -> MeshStats:
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if len(vertices) == 0:
        bbox_min = bbox_max = bbox_extent = np.zeros((3,), dtype=np.float64)
    else:
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_extent = bbox_max - bbox_min
    areas = _face_areas(vertices, faces)
    edge_lengths = _edge_lengths(vertices, faces)
    return MeshStats(
        vertices=int(len(vertices)),
        faces=int(len(faces)),
        bbox_min=_as_float_list(bbox_min),
        bbox_max=_as_float_list(bbox_max),
        bbox_extent=_as_float_list(bbox_extent),
        surface_area=float(areas.sum()),
        face_area_quantiles=_as_float_list(np.quantile(areas, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]) if len(areas) else np.zeros(6)),
        edge_length_quantiles=_as_float_list(np.quantile(edge_lengths, [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]) if len(edge_lengths) else np.zeros(6)),
        edge_manifold=bool(mesh.is_edge_manifold()),
        vertex_manifold=bool(mesh.is_vertex_manifold()),
        watertight=bool(mesh.is_watertight()),
    )


def load_and_clean_mesh(path: Path) -> tuple[o3d.geometry.TriangleMesh, MeshStats, MeshStats]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    input_stats = _mesh_stats(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    cleaned_stats = _mesh_stats(mesh)
    return mesh, input_stats, cleaned_stats


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    normals_out = np.zeros_like(normals, dtype=np.float64)
    normals_out[valid] = normals[valid] / norms[valid, None]
    return normals_out


def compute_vertex_feature_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    crease_angle_deg: float,
    boundary_quality: float,
    min_quality: float,
    quality_gamma: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute vertex quality used by PyMeshLab weighted QEM.

    PyMeshLab's weighted simplification treats high vertex quality as an error
    amplifier. We therefore assign low quality to planar interiors and high
    quality to vertices near creases, boundaries, non-manifold edge groups, or
    high normal variation. This is a lightweight approximation of
    structure-aware decimation that remains robust on open reconstruction meshes.
    """

    vertex_count = int(vertices.shape[0])
    face_count = int(faces.shape[0])
    if vertex_count == 0 or face_count == 0:
        return np.zeros((vertex_count,), dtype=np.float64), {"mean_quality": 0.0}

    normals = _face_normals(vertices, faces)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]],
        axis=0,
    )
    edge_faces = np.tile(np.arange(face_count, dtype=np.int64), 3)
    sorted_edges = np.sort(edges, axis=1)
    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    sorted_edges = sorted_edges[order]
    sorted_face_ids = edge_faces[order]

    changed = np.ones((len(sorted_edges),), dtype=bool)
    changed[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(changed)
    ends = np.concatenate([starts[1:], np.asarray([len(sorted_edges)], dtype=np.int64)])

    crease_angle_rad = math.radians(max(float(crease_angle_deg), 1e-3))
    hard_angle_rad = max(math.radians(90.0), crease_angle_rad + 1e-3)
    vertex_score = np.zeros((vertex_count,), dtype=np.float64)
    boundary_edges = 0
    crease_edges = 0
    nonmanifold_edges = 0

    for start, end in zip(starts, ends):
        edge_vertices = sorted_edges[start]
        incident = sorted_face_ids[start:end]
        if len(incident) == 1:
            score = float(boundary_quality)
            boundary_edges += 1
        elif len(incident) == 2:
            dot = float(np.clip(abs(np.dot(normals[incident[0]], normals[incident[1]])), 0.0, 1.0))
            angle = math.acos(dot)
            score = 0.0
            if angle >= crease_angle_rad:
                score = min((angle - crease_angle_rad) / (hard_angle_rad - crease_angle_rad), 1.0)
                crease_edges += 1
        else:
            score = 1.0
            nonmanifold_edges += 1

        if score > 0.0:
            np.maximum.at(vertex_score, edge_vertices, score)

    min_quality = max(float(min_quality), 1e-6)
    gamma = max(float(quality_gamma), 1e-6)
    quality = min_quality + (1.0 - min_quality) * np.power(vertex_score.clip(0.0, 1.0), gamma)
    stats = {
        "mean_quality": float(np.mean(quality)),
        "min_quality": float(np.min(quality)),
        "max_quality": float(np.max(quality)),
        "q50_quality": float(np.quantile(quality, 0.5)),
        "q90_quality": float(np.quantile(quality, 0.9)),
        "q99_quality": float(np.quantile(quality, 0.99)),
        "boundary_edges": float(boundary_edges),
        "crease_edges": float(crease_edges),
        "nonmanifold_edge_groups": float(nonmanifold_edges),
    }
    return quality.astype(np.float64, copy=False), stats


def remesh_with_planar_qem(
    *,
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int,
    preserve_boundary: bool,
    boundary_weight: float,
    preserve_normal: bool,
    preserve_topology: bool,
    planar_quadric: bool,
    planar_weight: float,
    quality_weight: bool,
    crease_angle_deg: float,
    boundary_quality: float,
    min_quality: float,
    quality_gamma: float,
    isotropic_iterations: int,
    isotropic_target_length_m: float,
) -> RemeshSummary:
    mesh, input_stats, cleaned_stats = load_and_clean_mesh(input_mesh)
    vertices = np.asarray(mesh.vertices).astype(np.float64, copy=False)
    faces = np.asarray(mesh.triangles).astype(np.int32, copy=False)
    quality, quality_stats = compute_vertex_feature_quality(
        vertices,
        faces,
        crease_angle_deg=crease_angle_deg,
        boundary_quality=boundary_quality,
        min_quality=min_quality,
        quality_gamma=quality_gamma,
    )

    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(
        pml.Mesh(
            vertex_matrix=vertices,
            face_matrix=faces,
            v_scalar_array=quality,
        ),
        mesh_name="omega_input",
    )
    mesh_set.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        qualitythr=0.3,
        preserveboundary=bool(preserve_boundary),
        boundaryweight=float(boundary_weight),
        preservenormal=bool(preserve_normal),
        preservetopology=bool(preserve_topology),
        optimalplacement=True,
        planarquadric=bool(planar_quadric),
        planarweight=float(planar_weight),
        qualityweight=bool(quality_weight),
        autoclean=True,
        selected=False,
    )
    if int(isotropic_iterations) > 0:
        if float(isotropic_target_length_m) <= 0.0:
            raise ValueError("--isotropic-target-length-m must be positive when isotropic remeshing is enabled.")
        mesh_set.meshing_isotropic_explicit_remeshing(
            iterations=int(isotropic_iterations),
            adaptive=True,
            selectedonly=False,
            targetlen=pml.PureValue(float(isotropic_target_length_m)),
            featuredeg=float(crease_angle_deg),
            checksurfdist=True,
            maxsurfdist=pml.PureValue(float(isotropic_target_length_m) * 0.75),
            splitflag=True,
            collapseflag=True,
            swapflag=True,
            smoothflag=True,
            reprojectflag=True,
        )

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh_set.save_current_mesh(str(output_mesh), save_vertex_color=False, save_vertex_quality=True)
    output = o3d.io.read_triangle_mesh(str(output_mesh))
    output_stats = _mesh_stats(output)
    return RemeshSummary(
        input_mesh=str(input_mesh),
        output_mesh=str(output_mesh),
        mode="structure_weighted_planar_qem",
        target_faces=int(target_faces),
        preserve_boundary=bool(preserve_boundary),
        boundary_weight=float(boundary_weight),
        preserve_normal=bool(preserve_normal),
        preserve_topology=bool(preserve_topology),
        planar_quadric=bool(planar_quadric),
        planar_weight=float(planar_weight),
        quality_weight=bool(quality_weight),
        crease_angle_deg=float(crease_angle_deg),
        boundary_quality=float(boundary_quality),
        min_quality=float(min_quality),
        quality_gamma=float(quality_gamma),
        isotropic_iterations=int(isotropic_iterations),
        isotropic_target_length_m=float(isotropic_target_length_m),
        input_stats=input_stats,
        cleaned_stats=cleaned_stats,
        output_stats=output_stats,
        quality_stats=quality_stats,
    )


def latest_mesh(result_dir: Path) -> Path:
    candidates = sorted((result_dir / "plys").glob("mesh_*_rank0.ply"))
    if not candidates:
        raise SystemExit(f"No mesh_*_rank0.ply files found under {result_dir / 'plys'}")

    def step(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=step)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc remesh an OMeGa output mesh.")
    parser.add_argument("result_dir", type=Path, help="OMeGa result directory containing plys/mesh_*_rank0.ply.")
    parser.add_argument("--input-mesh", type=Path, default=None, help="Specific input mesh. Defaults to latest rank0 mesh in result_dir/plys.")
    parser.add_argument("--out", type=Path, default=None, help="Output mesh path. Defaults to result_dir/remesh/mesh_posthoc_structure_qem.ply.")
    parser.add_argument("--target-faces", type=int, default=200_000, help="Target face count for QEM simplification.")
    parser.add_argument("--preserve-boundary", action=argparse.BooleanOptionalAction, default=True, help="Preserve open boundaries during QEM.")
    parser.add_argument("--boundary-weight", type=float, default=5.0, help="Boundary preservation weight.")
    parser.add_argument("--preserve-normal", action=argparse.BooleanOptionalAction, default=True, help="Avoid normal flipping during QEM.")
    parser.add_argument("--preserve-topology", action=argparse.BooleanOptionalAction, default=False, help="Avoid topology-changing collapses. Usually off for OMeGa open meshes.")
    parser.add_argument("--planar-quadric", action=argparse.BooleanOptionalAction, default=True, help="Enable MeshLab planar quadric simplification.")
    parser.add_argument("--planar-weight", type=float, default=0.0005, help="Lower values simplify planar regions more aggressively.")
    parser.add_argument("--quality-weight", action=argparse.BooleanOptionalAction, default=True, help="Use computed vertex quality to protect creases/details.")
    parser.add_argument("--crease-angle-deg", type=float, default=80.0, help="Adjacent face angle that should be treated as a protected feature. OMeGa meshes can be noisy, so the default is deliberately high.")
    parser.add_argument("--boundary-quality", type=float, default=1.0, help="Feature score assigned to open boundary vertices.")
    parser.add_argument("--min-quality", type=float, default=0.05, help="Quality assigned to flat planar interiors.")
    parser.add_argument("--quality-gamma", type=float, default=1.5, help="Gamma applied to feature scores before quality weighting. Values above 1 keep weak normal noise closer to planar-interior quality.")
    parser.add_argument("--isotropic-iterations", type=int, default=0, help="Optional final isotropic remeshing iterations. Keep 0 for the main planar simplification baseline.")
    parser.add_argument("--isotropic-target-length-m", type=float, default=0.0, help="Target edge length for optional isotropic remeshing.")
    return parser.parse_args(argv)


def summary_to_json(summary: RemeshSummary) -> dict[str, Any]:
    return asdict(summary)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result_dir = args.result_dir.expanduser().resolve()
    input_mesh = (args.input_mesh.expanduser().resolve() if args.input_mesh else latest_mesh(result_dir))
    output_mesh = (
        args.out.expanduser().resolve()
        if args.out
        else result_dir / "remesh" / "mesh_posthoc_structure_qem.ply"
    )
    summary = remesh_with_planar_qem(
        input_mesh=input_mesh,
        output_mesh=output_mesh,
        target_faces=int(args.target_faces),
        preserve_boundary=bool(args.preserve_boundary),
        boundary_weight=float(args.boundary_weight),
        preserve_normal=bool(args.preserve_normal),
        preserve_topology=bool(args.preserve_topology),
        planar_quadric=bool(args.planar_quadric),
        planar_weight=float(args.planar_weight),
        quality_weight=bool(args.quality_weight),
        crease_angle_deg=float(args.crease_angle_deg),
        boundary_quality=float(args.boundary_quality),
        min_quality=float(args.min_quality),
        quality_gamma=float(args.quality_gamma),
        isotropic_iterations=int(args.isotropic_iterations),
        isotropic_target_length_m=float(args.isotropic_target_length_m),
    )
    summary_path = output_mesh.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_to_json(summary), indent=2), encoding="utf-8")
    print("=== OMeGa Post-hoc Remesh ===")
    print(f"Input: {input_mesh}")
    print(f"Output: {output_mesh}")
    print(f"Summary: {summary_path}")
    print(f"Faces: {summary.input_stats.faces} -> {summary.output_stats.faces}")
    print(f"Vertices: {summary.input_stats.vertices} -> {summary.output_stats.vertices}")
    print(f"Mean quality: {summary.quality_stats['mean_quality']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

