"""In-training remeshing and splat reparenting for OMeGa-4-Building.

This module is deliberately separate from ``examples/simple_trainer_meshgs.py``.
The clean baseline can still run unchanged; the remesh-aware trainer imports this
module and swaps in ``TrainingRemeshMeshGSStrategy``.

The central operation is topology replacement:

1. Bake current mesh-attached splats to world-space centers, scales, and frames.
2. Run the same planar-aware QEM remesh utility used by post-hoc inspection.
3. For every existing splat, find a nearby triangle on the simplified mesh.
4. Invert the baked world transform back into OMeGa's face-local parameters:

   - ``uv_sum`` and ``u_ratio`` encode the closest point's barycentric position.
   - ``scale_lambda`` preserves the splat's world in-plane size as much as the
     new triangle's allowed relative scale range permits.
   - ``rot_2d`` projects the old major axis onto the new triangle tangent plane.

No appearance or opacity parameters are recreated. They stay attached to the
same splat identity, while only the parent face and geometric face-local
coordinates are rewritten.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import open3d as o3d
import pymeshlab as pml
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from examples.utils import calculate_Rs, calculate_scales, rotation_matrix_to_quaternion
from gsplat.strategy.meshgs import MeshGSStrategy
from gsplat.utils import normalized_quat_to_rotmat
from omega_local.remesh.posthoc import remesh_with_planar_qem, summary_to_json


_EPS = 1e-6


def _logit_clamped(values: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(values.clamp(float(eps), 1.0 - float(eps)))


@torch.no_grad()
def _update_derived_splats(cfg: Any, mesh_params: Dict[str, torch.Tensor], gs_params: Dict[str, torch.Tensor]) -> None:
    """Recompute OMeGa's derived ``means/scales/quats`` from mesh-local params."""

    faces = mesh_params["faces"]
    vertices = mesh_params["vertices"]
    gs2mesh_index = gs_params["index"]

    uv_sum = gs_params["uv_sum"]
    u_ratio = gs_params["u_ratio"]
    u = torch.sigmoid(uv_sum) * torch.sigmoid(u_ratio)
    v = torch.sigmoid(uv_sum) - u

    triangles = vertices[faces][gs2mesh_index]
    means = triangles[:, 0] + u * (triangles[:, 1] - triangles[:, 0]) + v * (triangles[:, 2] - triangles[:, 0])

    scale_range = float(cfg.max_rel_scale) - float(cfg.min_rel_scale)
    base_scales = calculate_scales(triangles).clamp_min(_EPS)
    scales_xy = base_scales * (torch.sigmoid(gs_params["scale_lambda"]) * scale_range + float(cfg.min_rel_scale))
    scales = torch.log(torch.cat([scales_xy, torch.ones_like(scales_xy[:, :1])], dim=-1))

    Rs = calculate_Rs(triangles)
    rot_2d = F.normalize(gs_params["rot_2d"], dim=-1, eps=_EPS)
    R_0 = rot_2d[..., 0:1] * Rs[..., 0] + rot_2d[..., 1:2] * Rs[..., 1]
    R_1 = -rot_2d[..., 1:2] * Rs[..., 0] + rot_2d[..., 0:1] * Rs[..., 1]
    R_2 = Rs[..., 2]
    R = torch.cat([R_0[..., None], R_1[..., None], R_2[..., None]], dim=-1)
    quats = rotation_matrix_to_quaternion(R)

    gs_params["means"] = means
    gs_params["scales"] = scales
    gs_params["quats"] = quats


def _write_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise RuntimeError(f"Failed to write mesh: {path}")


def _load_mesh(path: Path, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError(f"Remeshed output has no vertices/faces: {path}")
    return torch.as_tensor(vertices, dtype=torch.float32, device=device), torch.as_tensor(faces, dtype=torch.long, device=device)


def _mesh_topology_snapshot(mesh: o3d.geometry.TriangleMesh) -> dict[str, Any]:
    """Small topology report used to check whether OMeGa subdivision can run."""

    def _edge_manifold(allow_boundary: bool) -> bool:
        try:
            return bool(mesh.is_edge_manifold(allow_boundary_edges=allow_boundary))
        except TypeError:
            return bool(mesh.is_edge_manifold(allow_boundary))

    def _non_manifold_edges(allow_boundary: bool) -> int:
        try:
            edges = mesh.get_non_manifold_edges(allow_boundary_edges=allow_boundary)
        except TypeError:
            edges = mesh.get_non_manifold_edges(allow_boundary)
        return int(np.asarray(edges).shape[0])

    try:
        non_manifold_vertices = int(np.asarray(mesh.get_non_manifold_vertices()).shape[0])
    except Exception:
        non_manifold_vertices = -1
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.triangles)),
        "edgeManifoldAllowBoundary": _edge_manifold(True),
        "edgeManifoldClosed": _edge_manifold(False),
        "vertexManifold": bool(mesh.is_vertex_manifold()),
        "watertight": bool(mesh.is_watertight()),
        "nonManifoldEdgesAllowBoundary": _non_manifold_edges(True),
        "nonManifoldEdgesClosed": _non_manifold_edges(False),
        "nonManifoldVertices": non_manifold_vertices,
    }


def _repair_remesh_output_for_subdivision(path: Path, *, enabled: bool) -> dict[str, Any]:
    """Repair remesh output so later PyMeshLab midpoint subdivision can run.

    Training-time remeshing is followed by OMeGa's normal subdivision schedule.
    PyMeshLab midpoint subdivision is stricter than rendering: it rejects
    selected patches with non-2-manifold topology. QEM simplification can create
    those cases, so we clean duplicate/null elements and repair non-manifold
    edges/vertices before reparenting splats onto the new mesh.
    """

    mesh_before = o3d.io.read_triangle_mesh(str(path))
    before = _mesh_topology_snapshot(mesh_before)
    if not enabled:
        return {"enabled": False, "changed": False, "before": before, "after": before, "error": None}

    error = None
    try:
        mesh_set = pml.MeshSet()
        mesh_set.load_new_mesh(str(path))
        mesh_set.meshing_remove_duplicate_vertices()
        mesh_set.meshing_remove_duplicate_faces()
        mesh_set.meshing_remove_null_faces()
        mesh_set.meshing_repair_non_manifold_edges(method="Remove Faces")
        mesh_set.meshing_repair_non_manifold_vertices(vertdispratio=0.0)
        mesh_set.meshing_remove_unreferenced_vertices()
        mesh_set.save_current_mesh(str(path), save_vertex_color=False, save_vertex_quality=True)
    except Exception as exc:
        error = str(exc)
        # Fall back to Open3D's conservative cleanup. It cannot fix every
        # vertex-manifold issue, but it often removes the bad edge groups.
        mesh = mesh_before
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        if not o3d.io.write_triangle_mesh(str(path), mesh):
            raise RuntimeError(f"Failed to write repaired remesh output: {path}") from exc

    mesh_after = o3d.io.read_triangle_mesh(str(path))
    after = _mesh_topology_snapshot(mesh_after)
    changed = before != after
    if changed or error is not None or not after["edgeManifoldAllowBoundary"] or not after["vertexManifold"]:
        print(
            "[training-remesh] topology cleanup "
            f"faces {before['faces']:,}->{after['faces']:,}, "
            f"edge_manifold={after['edgeManifoldAllowBoundary']}, "
            f"vertex_manifold={after['vertexManifold']}, "
            f"nonmanifold_edges={after['nonManifoldEdgesAllowBoundary']}, "
            f"nonmanifold_vertices={after['nonManifoldVertices']}"
        )
        if error is not None:
            print(f"[training-remesh][warning] PyMeshLab topology repair failed; used Open3D fallback: {error}")
    return {"enabled": True, "changed": changed, "before": before, "after": after, "error": error}


def _optimizer_replace_parameter(
    optimizers: Dict[str, torch.optim.Optimizer],
    name: str,
    new_value: torch.Tensor,
    *,
    reset_state: bool,
) -> torch.nn.Parameter:
    """Replace a single-parameter Adam optimizer target safely.

    OMeGa stores one optimizer per trainable tensor. During topology replacement
    we need new Parameter objects so future gradients attach to the remeshed
    tensors. Geometry states are reset by default because their old moments were
    accumulated in the previous mesh parameterization.
    """

    new_param = torch.nn.Parameter(new_value.detach().clone())
    optimizer = optimizers.get(name)
    if optimizer is None:
        return new_param

    for group in optimizer.param_groups:
        old_param = group["params"][0]
        old_state = optimizer.state.pop(old_param, {})
        new_state: dict[str, Any] = {}
        for key, value in old_state.items():
            if key == "step":
                new_state[key] = value
            elif isinstance(value, torch.Tensor):
                if (not reset_state) and tuple(value.shape) == tuple(new_param.shape):
                    new_state[key] = value.detach().clone()
                else:
                    new_state[key] = torch.zeros_like(new_param, memory_format=torch.preserve_format)
            else:
                new_state[key] = value
        group["params"] = [new_param]
        optimizer.state[new_param] = new_state
    return new_param


def _closest_point_barycentric(points: torch.Tensor, triangles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Closest point on triangles with barycentric coordinates.

    Vectorized Ericson triangle-distance test. Returns squared distances and
    barycentric weights ordered as ``(A, B, C)``.
    """

    p = points
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = torch.sum(ab * ap, dim=-1)
    d2 = torch.sum(ac * ap, dim=-1)
    bary = torch.zeros((len(points), 3), dtype=points.dtype, device=points.device)
    closest = torch.empty_like(points)
    assigned = torch.zeros((len(points),), dtype=torch.bool, device=points.device)

    def assign(mask: torch.Tensor, values: torch.Tensor) -> None:
        active = mask & (~assigned)
        if active.any():
            bary[active] = values[active]
            assigned[active] = True

    values = torch.zeros_like(bary)
    values[:, 0] = 1.0
    assign((d1 <= 0.0) & (d2 <= 0.0), values)

    bp = p - b
    d3 = torch.sum(ab * bp, dim=-1)
    d4 = torch.sum(ac * bp, dim=-1)
    values = torch.zeros_like(bary)
    values[:, 1] = 1.0
    assign((d3 >= 0.0) & (d4 <= d3), values)

    vc = d1 * d4 - d3 * d2
    denom = (d1 - d3).clamp_min(_EPS)
    v_ab = (d1 / denom).clamp(0.0, 1.0)
    values = torch.zeros_like(bary)
    values[:, 0] = 1.0 - v_ab
    values[:, 1] = v_ab
    assign((vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0), values)

    cp = p - c
    d5 = torch.sum(ab * cp, dim=-1)
    d6 = torch.sum(ac * cp, dim=-1)
    values = torch.zeros_like(bary)
    values[:, 2] = 1.0
    assign((d6 >= 0.0) & (d5 <= d6), values)

    vb = d5 * d2 - d1 * d6
    denom = (d2 - d6).clamp_min(_EPS)
    w_ac = (d2 / denom).clamp(0.0, 1.0)
    values = torch.zeros_like(bary)
    values[:, 0] = 1.0 - w_ac
    values[:, 2] = w_ac
    assign((vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0), values)

    va = d3 * d6 - d5 * d4
    denom = ((d4 - d3) + (d5 - d6)).clamp_min(_EPS)
    w_bc = ((d4 - d3) / denom).clamp(0.0, 1.0)
    values = torch.zeros_like(bary)
    values[:, 1] = 1.0 - w_bc
    values[:, 2] = w_bc
    assign((va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0), values)

    face_mask = ~assigned
    denom = (va + vb + vc).clamp_min(_EPS)
    v = vb / denom
    w = vc / denom
    u = 1.0 - v - w
    values = torch.stack([u, v, w], dim=-1)
    assign(face_mask, values)

    closest = bary[:, :1] * a + bary[:, 1:2] * b + bary[:, 2:3] * c
    dist2 = torch.sum((closest - p) ** 2, dim=-1)
    return dist2, bary.clamp(0.0, 1.0)


@torch.no_grad()
def _closest_triangles_for_points(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    k: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Approximate closest triangle lookup using a face-center KD tree.

    The KD tree narrows each query to ``k`` nearby face centers; within those
    candidates we compute exact closest-point-on-triangle distances. This is a
    practical compromise for training-time remeshes where full mesh BVHs are not
    guaranteed to be available in the environment.
    """

    device = points.device
    vertices_cpu = vertices.detach().cpu().numpy()
    faces_cpu = faces.detach().cpu().numpy()
    centers = vertices_cpu[faces_cpu].mean(axis=1)
    tree = cKDTree(centers)
    k = max(1, min(int(k), int(len(faces_cpu))))
    _, candidate_ids = tree.query(points.detach().cpu().numpy(), k=k, workers=-1)
    if candidate_ids.ndim == 1:
        candidate_ids = candidate_ids[:, None]

    n = int(points.shape[0])
    out_faces = torch.empty((n,), dtype=torch.long, device=device)
    out_bary = torch.empty((n, 3), dtype=points.dtype, device=device)
    out_dist2 = torch.empty((n,), dtype=points.dtype, device=device)
    chunk_size = max(1, int(chunk_size))
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        ids = torch.as_tensor(candidate_ids[start:end], dtype=torch.long, device=device)
        m, kk = ids.shape
        candidate_triangles = vertices[faces[ids.reshape(-1)]].reshape(m, kk, 3, 3)
        query = points[start:end, None, :].expand(m, kk, 3).reshape(-1, 3)
        dist2, bary = _closest_point_barycentric(query, candidate_triangles.reshape(-1, 3, 3))
        dist2 = dist2.reshape(m, kk)
        bary = bary.reshape(m, kk, 3)
        best = torch.argmin(dist2, dim=1)
        row = torch.arange(m, dtype=torch.long, device=device)
        out_faces[start:end] = ids[row, best]
        out_bary[start:end] = bary[row, best]
        out_dist2[start:end] = dist2[row, best]
    return out_faces, out_bary, out_dist2


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    values_cpu = values.detach().float().cpu()
    if values_cpu.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values_cpu.mean()),
        "p50": float(torch.quantile(values_cpu, 0.50)),
        "p90": float(torch.quantile(values_cpu, 0.90)),
        "p95": float(torch.quantile(values_cpu, 0.95)),
        "max": float(values_cpu.max()),
    }


@torch.no_grad()
def _reparent_splats_to_mesh(
    *,
    cfg: Any,
    mesh_params: Dict[str, torch.Tensor],
    gs_params: Dict[str, torch.Tensor],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Any],
    mesh_state: Dict[str, Any],
    new_vertices: torch.Tensor,
    new_faces: torch.Tensor,
) -> dict[str, Any]:
    device = mesh_params["vertices"].device
    _update_derived_splats(cfg, mesh_params, gs_params)

    old_means = gs_params["means"].detach()
    old_scales_xy = torch.exp(gs_params["scales"].detach()[:, :2])
    old_rotmat = normalized_quat_to_rotmat(F.normalize(gs_params["quats"].detach(), dim=-1, eps=_EPS))
    old_major_axis = F.normalize(old_rotmat[..., 0], dim=-1, eps=_EPS)
    old_normal = F.normalize(old_rotmat[..., 2], dim=-1, eps=_EPS)

    new_vertices = new_vertices.to(device=device, dtype=torch.float32)
    new_faces = new_faces.to(device=device, dtype=torch.long)
    nearest_faces, bary, dist2 = _closest_triangles_for_points(
        old_means,
        new_vertices,
        new_faces,
        k=int(getattr(cfg, "training_remesh_reparent_k", 32)),
        chunk_size=int(getattr(cfg, "training_remesh_reparent_chunk_size", 65536)),
    )

    new_triangles = new_vertices[new_faces[nearest_faces]]
    bary_b = bary[:, 1:2]
    bary_c = bary[:, 2:3]
    uv_sum_prob = (bary_b + bary_c).clamp(0.0, 1.0)
    u_ratio_prob = torch.where(
        uv_sum_prob > _EPS,
        (bary_b / uv_sum_prob.clamp_min(_EPS)).clamp(0.0, 1.0),
        torch.full_like(uv_sum_prob, 0.5),
    )
    new_uv_sum = _logit_clamped(uv_sum_prob)
    new_u_ratio = _logit_clamped(u_ratio_prob)

    base_scales = calculate_scales(new_triangles).clamp_min(_EPS)
    scale_range = max(float(cfg.max_rel_scale) - float(cfg.min_rel_scale), _EPS)
    rel_scale = (old_scales_xy / base_scales - float(cfg.min_rel_scale)) / scale_range
    scale_clamped = (rel_scale < 1e-4) | (rel_scale > 1.0 - 1e-4)
    new_scale_lambda = _logit_clamped(rel_scale)

    new_Rs = calculate_Rs(new_triangles)
    new_normal = F.normalize(new_Rs[..., 2], dim=-1, eps=_EPS)
    projected_major = old_major_axis - torch.sum(old_major_axis * new_normal, dim=-1, keepdim=True) * new_normal
    projected_norm = torch.linalg.norm(projected_major, dim=-1, keepdim=True)
    projected_major = torch.where(projected_norm > 1e-5, projected_major / projected_norm.clamp_min(_EPS), new_Rs[..., 0])
    new_rot_2d = torch.stack(
        [
            torch.sum(projected_major * new_Rs[..., 0], dim=-1),
            torch.sum(projected_major * new_Rs[..., 1], dim=-1),
        ],
        dim=-1,
    )
    new_rot_2d = F.normalize(new_rot_2d, dim=-1, eps=_EPS)

    mesh_params["vertices"] = _optimizer_replace_parameter(optimizers, "vertices", new_vertices, reset_state=True)
    mesh_params["faces"] = new_faces
    gs_params["uv_sum"] = _optimizer_replace_parameter(optimizers, "uv_sum", new_uv_sum, reset_state=True)
    gs_params["u_ratio"] = _optimizer_replace_parameter(optimizers, "u_ratio", new_u_ratio, reset_state=True)
    gs_params["scale_lambda"] = _optimizer_replace_parameter(optimizers, "scale_lambda", new_scale_lambda, reset_state=True)
    gs_params["rot_2d"] = _optimizer_replace_parameter(optimizers, "rot_2d", new_rot_2d, reset_state=True)
    gs_params["index"] = nearest_faces.detach().long()
    _update_derived_splats(cfg, mesh_params, gs_params)

    n_splats = int(gs_params["index"].shape[0])
    for key, value in list(state.items()):
        if isinstance(value, torch.Tensor) and value.shape[:1] == (n_splats,):
            state[key] = torch.zeros_like(value)
    for key, value in list(mesh_state.items()):
        if isinstance(value, torch.Tensor):
            mesh_state[key] = torch.zeros((int(new_faces.shape[0]), *value.shape[1:]), dtype=value.dtype, device=value.device)
        else:
            mesh_state[key] = None

    normal_dot = torch.sum(old_normal * new_normal, dim=-1).abs().clamp(0.0, 1.0)
    normal_angle = torch.rad2deg(torch.acos(normal_dot))
    unique_faces = int(torch.unique(nearest_faces).numel())
    return {
        "reparentDistanceMeters": _tensor_stats(torch.sqrt(dist2.clamp_min(0.0))),
        "normalAngleDegrees": _tensor_stats(normal_angle),
        "scaleClampFraction": float(scale_clamped.float().mean().detach().cpu()),
        "splats": n_splats,
        "usedNewFaces": unique_faces,
        "usedNewFaceFraction": float(unique_faces / max(int(new_faces.shape[0]), 1)),
    }


def _resolve_target_faces(cfg: Any, current_faces: int, step: int) -> tuple[int, dict[str, Any]]:
    fixed = int(getattr(cfg, "training_remesh_target_faces", 0))
    floor = max(0, int(getattr(cfg, "training_remesh_min_target_faces", 0)))
    ratio_used: float | None = None
    event_index: int | None = None

    if fixed > 0:
        target = fixed
    else:
        ratios = list(getattr(cfg, "training_remesh_target_ratios", []) or [])
        if ratios:
            start = int(getattr(cfg, "training_remesh_start_iter", 0))
            every = max(1, int(getattr(cfg, "training_remesh_every", 1)))
            event_index = max(0, int((int(step) - start) // every))
            ratio_used = float(ratios[min(event_index, len(ratios) - 1)])
        else:
            ratio_used = float(getattr(cfg, "training_remesh_target_ratio", 0.75))
        if ratio_used <= 0.0:
            ratio_used = 0.75
        target = int(round(current_faces * ratio_used))

    if floor > 0:
        target = max(target, floor)
    target = max(4, target)
    return target, {
        "fixedTargetFaces": int(fixed),
        "minTargetFaces": int(floor),
        "targetRatio": None if ratio_used is None else float(ratio_used),
        "targetRatioEventIndex": event_index,
    }


def _remesh_guard_failures(cfg: Any, remesh_summary: Any) -> list[dict[str, float | str]]:
    failures: list[dict[str, float | str]] = []
    max_area_delta = float(getattr(cfg, "training_remesh_max_surface_area_change_fraction", 0.0))
    if max_area_delta > 0.0:
        area_in = float(remesh_summary.cleaned_stats.surface_area)
        area_out = float(remesh_summary.output_stats.surface_area)
        if area_in > _EPS:
            delta = abs(area_out / area_in - 1.0)
            if delta > max_area_delta:
                failures.append({
                    "name": "surface_area_change_fraction",
                    "value": float(delta),
                    "limit": float(max_area_delta),
                })

    max_edge_growth = float(getattr(cfg, "training_remesh_max_edge_p99_growth", 0.0))
    if max_edge_growth > 0.0:
        input_edges = remesh_summary.cleaned_stats.edge_length_quantiles
        output_edges = remesh_summary.output_stats.edge_length_quantiles
        edge_in = float(input_edges[-1]) if input_edges else 0.0
        edge_out = float(output_edges[-1]) if output_edges else 0.0
        if edge_in > _EPS:
            growth = edge_out / edge_in
            if growth > max_edge_growth:
                failures.append({
                    "name": "edge_p99_growth",
                    "value": float(growth),
                    "limit": float(max_edge_growth),
                })
    return failures


@torch.no_grad()
def run_training_remesh(
    *,
    cfg: Any,
    mesh_params: Dict[str, torch.Tensor],
    gs_params: Dict[str, torch.Tensor],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Any],
    mesh_state: Dict[str, Any],
    step: int,
) -> dict[str, Any]:
    """Run one in-training remesh and rewrite splat bindings."""

    current_vertices = mesh_params["vertices"].detach().cpu().numpy()
    current_faces = mesh_params["faces"].detach().cpu().numpy()
    current_face_count = int(len(current_faces))
    target_faces, target_policy = _resolve_target_faces(cfg, current_face_count, int(step))
    remesh_dir = Path(str(cfg.result_dir)) / "training_remesh" / f"step_{int(step):05d}"
    input_mesh = remesh_dir / "mesh_before_remesh.ply"
    output_mesh = remesh_dir / "mesh_after_remesh.ply"
    summary_path = remesh_dir / "training_remesh_summary.json"

    if target_faces >= current_face_count and int(getattr(cfg, "training_remesh_isotropic_iterations", 0)) <= 0:
        payload = {
            "step": int(step),
            "skipped": True,
            "reason": "target_faces_not_less_than_current_faces",
            "inputFaces": current_face_count,
            "targetFaces": int(target_faces),
            "targetPolicy": target_policy,
        }
        remesh_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[training-remesh] step={step} skipped: target_faces={target_faces:,} "
            f">= current_faces={current_face_count:,}"
        )
        return payload

    _write_mesh(input_mesh, current_vertices, current_faces)
    print(
        f"[training-remesh] step={step} remeshing {current_face_count:,} faces "
        f"toward target={target_faces:,}"
    )
    remesh_summary = remesh_with_planar_qem(
        input_mesh=input_mesh,
        output_mesh=output_mesh,
        target_faces=int(target_faces),
        preserve_boundary=bool(getattr(cfg, "training_remesh_preserve_boundary", True)),
        boundary_weight=float(getattr(cfg, "training_remesh_boundary_weight", 5.0)),
        preserve_normal=bool(getattr(cfg, "training_remesh_preserve_normal", True)),
        preserve_topology=bool(getattr(cfg, "training_remesh_preserve_topology", True)),
        planar_quadric=bool(getattr(cfg, "training_remesh_planar_quadric", True)),
        planar_weight=float(getattr(cfg, "training_remesh_planar_weight", 0.0005)),
        quality_weight=bool(getattr(cfg, "training_remesh_quality_weight", True)),
        crease_angle_deg=float(getattr(cfg, "training_remesh_crease_angle_deg", 80.0)),
        boundary_quality=float(getattr(cfg, "training_remesh_boundary_quality", 1.0)),
        min_quality=float(getattr(cfg, "training_remesh_min_quality", 0.05)),
        quality_gamma=float(getattr(cfg, "training_remesh_quality_gamma", 1.5)),
        isotropic_iterations=int(getattr(cfg, "training_remesh_isotropic_iterations", 0)),
        isotropic_target_length_m=float(getattr(cfg, "training_remesh_isotropic_target_length_m", 0.0)),
    )
    guard_failures = _remesh_guard_failures(cfg, remesh_summary)
    if guard_failures:
        payload = {
            "step": int(step),
            "skipped": True,
            "reason": "remesh_quality_guard_failed",
            "inputMesh": str(input_mesh),
            "outputMesh": str(output_mesh),
            "inputFaces": current_face_count,
            "targetFaces": int(target_faces),
            "targetPolicy": target_policy,
            "remesh": summary_to_json(remesh_summary),
            "guardFailures": guard_failures,
        }
        summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"[training-remesh][warning] step={step} skipped after QEM: "
            f"quality guard failed {guard_failures}"
        )
        return payload

    topology_cleanup = _repair_remesh_output_for_subdivision(
        output_mesh,
        enabled=bool(getattr(cfg, "training_remesh_topology_cleanup", True)),
    )
    new_vertices, new_faces = _load_mesh(output_mesh, mesh_params["vertices"].device)
    reparent_stats = _reparent_splats_to_mesh(
        cfg=cfg,
        mesh_params=mesh_params,
        gs_params=gs_params,
        optimizers=optimizers,
        state=state,
        mesh_state=mesh_state,
        new_vertices=new_vertices,
        new_faces=new_faces,
    )
    payload = {
        "step": int(step),
        "skipped": False,
        "inputMesh": str(input_mesh),
        "outputMesh": str(output_mesh),
        "inputFaces": current_face_count,
        "targetFaces": int(target_faces),
        "targetPolicy": target_policy,
        "outputFaces": int(new_faces.shape[0]),
        "inputVertices": int(current_vertices.shape[0]),
        "outputVertices": int(new_vertices.shape[0]),
        "remesh": summary_to_json(remesh_summary),
        "topologyCleanup": topology_cleanup,
        "reparent": reparent_stats,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"[training-remesh] step={step} faces {current_face_count:,}->{int(new_faces.shape[0]):,}; "
        f"splats={reparent_stats['splats']:,}; used_faces={reparent_stats['usedNewFaces']:,}; "
        f"reparent_p95={reparent_stats['reparentDistanceMeters']['p95'] * 100.0:.2f}cm; "
        f"normal_p95={reparent_stats['normalAngleDegrees']['p95']:.1f}deg"
    )
    return payload


class TrainingRemeshMeshGSStrategy(MeshGSStrategy):
    """MeshGS strategy with an optional topology-simplifying remesh interval."""

    remesh_before_preview_callback: Any = None
    remesh_preview_callback: Any = None

    @torch.no_grad()
    def _split_meshes(
        self,
        mesh_params: Dict,
        gs_params: Dict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        mesh_state: Dict[str, Any],
        step: int,
        cfg,
    ) -> int:
        """Run OMeGa midpoint subdivision, but keep remesh tests alive.

        The in-training remesh experiment still uses the original OMeGa
        subdivision schedule before and between remesh events. PyMeshLab
        selected-face midpoint subdivision can abort if the current mesh, or the
        selected patch, is non-manifold. That can happen after an experimental
        QEM remesh even when the mesh is perfectly renderable.

        For this experimental branch, skip only the failed subdivision event so
        the run can continue and write diagnostics. The clean baseline strategy
        remains unchanged.
        """

        try:
            result = super()._split_meshes(
                mesh_params=mesh_params,
                gs_params=gs_params,
                optimizers=optimizers,
                state=state,
                mesh_state=mesh_state,
                step=step,
                cfg=cfg,
            )
        except Exception as exc:
            print(
                "[training-remesh][warning] OMeGa midpoint subdivision failed "
                f"at step {step}: {exc}. Skipping this subdivision event so "
                "the in-training remesh branch can continue."
            )
            for key in ("grad", "count"):
                value = mesh_state.get(key)
                if isinstance(value, torch.Tensor):
                    value.zero_()
            return 0
        return 0 if result is None else int(result)

    def step_post_backward(
        self,
        mesh_params,
        gs_params,
        cfg,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        mesh_state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        super().step_post_backward(
            mesh_params=mesh_params,
            gs_params=gs_params,
            cfg=cfg,
            optimizers=optimizers,
            state=state,
            mesh_state=mesh_state,
            step=step,
            info=info,
            packed=packed,
        )

        if not bool(getattr(cfg, "training_remesh_on", False)):
            return
        every = int(getattr(cfg, "training_remesh_every", 0))
        start = int(getattr(cfg, "training_remesh_start_iter", 0))
        stop = int(getattr(cfg, "training_remesh_stop_iter", 0))
        if every <= 0 or step <= 0 or step < start:
            return
        if stop > 0 and step > stop:
            return
        # Anchor the cadence at training_remesh_start_iter. This lets us place
        # remesh events between OMeGa's own subdivision steps instead of being
        # forced to absolute multiples of ``training_remesh_every``.
        if (step - start) % every != 0:
            return

        before_preview = None
        if self.remesh_before_preview_callback is not None:
            before_preview = self.remesh_before_preview_callback(step=int(step))

        remesh_payload = run_training_remesh(
            cfg=cfg,
            mesh_params=mesh_params,
            gs_params=gs_params,
            optimizers=optimizers,
            state=state,
            mesh_state=mesh_state,
            step=step,
        )
        if self.remesh_preview_callback is not None:
            self.remesh_preview_callback(
                step=int(step),
                remesh_payload=remesh_payload,
                before_preview=before_preview,
            )
