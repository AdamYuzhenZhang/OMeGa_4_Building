"""Dynamic coplane regularization for OMeGa-4-Building.

This module keeps the grouping logic separate from the trainer.  The idea is:

1. Periodically inspect the current mesh without gradients.
2. Optionally fit broad plane hypotheses from a monocular depth guide.
3. Use those broad depth planes to merge fragmented mesh-face buckets.
4. Fall back to mesh-only grouping for faces not covered by depth planes.
5. During normal backprop, penalize current face vertices and normals against
   cached stop-gradient planes.

The cached targets make the structural hypothesis stable for a few iterations,
while periodic refresh lets groups merge, drop out, or refit as the mesh moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class CoplaneTargets:
    """Per-face target planes produced by a non-differentiable refresh step."""

    normals: Tensor
    offsets: Tensor
    weights: Tensor
    group_ids: Tensor
    face_count: int
    last_refresh_step: int
    stats: dict[str, float]


@dataclass
class DepthPlaneHypotheses:
    """Stop-gradient planes fitted from a monocular depth guide."""

    normals: Tensor
    offsets: Tensor
    weights: Tensor
    stats: dict[str, float]


def _huber(error: Tensor, delta: float) -> Tensor:
    """Huber penalty used for vertex-to-plane distances in meters."""

    abs_error = torch.abs(error)
    delta_t = torch.as_tensor(max(float(delta), 1e-6), device=error.device, dtype=error.dtype)
    quadratic = 0.5 * error.square() / delta_t
    linear = abs_error - 0.5 * delta_t
    return torch.where(abs_error <= delta_t, quadratic, linear)


def _face_geometry(vertices: Tensor, faces: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return face vertices, centers, unit normals, and areas."""

    face_vertices = vertices[faces]
    edge_01 = face_vertices[:, 1] - face_vertices[:, 0]
    edge_02 = face_vertices[:, 2] - face_vertices[:, 0]
    raw_normals = torch.cross(edge_01, edge_02, dim=1)
    double_area = torch.linalg.norm(raw_normals, dim=1).clamp_min(1e-12)
    normals = raw_normals / double_area[:, None]
    centers = face_vertices.mean(dim=1)
    areas = 0.5 * double_area
    return face_vertices, centers, normals, areas


def _canonicalize_normals(normals: Tensor) -> Tensor:
    """Use one orientation for opposite normals on the same geometric plane."""

    major_axis = torch.argmax(torch.abs(normals), dim=1, keepdim=True)
    major_value = torch.gather(normals, 1, major_axis).squeeze(1)
    sign = torch.where(major_value < 0.0, -torch.ones_like(major_value), torch.ones_like(major_value))
    return normals * sign[:, None]


def _canonicalize_planes(normals: Tensor, offsets: Tensor) -> tuple[Tensor, Tensor]:
    """Canonicalize plane orientation while preserving n dot x + d = 0."""

    major_axis = torch.argmax(torch.abs(normals), dim=1, keepdim=True)
    major_value = torch.gather(normals, 1, major_axis).squeeze(1)
    sign = torch.where(major_value < 0.0, -torch.ones_like(major_value), torch.ones_like(major_value))
    return normals * sign[:, None], offsets * sign


def _canonicalize_np(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float64)
    major = int(np.argmax(np.abs(normal)))
    if normal[major] < 0:
        normal = -normal
    return normal


def _fit_plane_np(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit n dot x + d = 0 to points and return normal, offset, residuals."""

    centroid = np.mean(points, axis=0)
    centered = points - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = _canonicalize_np(vh[-1])
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(points @ normal + offset)
    return normal, offset, residuals


def _empty_depth_planes(device: str | torch.device, dtype: torch.dtype) -> DepthPlaneHypotheses:
    return DepthPlaneHypotheses(
        normals=torch.zeros((0, 3), device=device, dtype=dtype),
        offsets=torch.zeros((0,), device=device, dtype=dtype),
        weights=torch.zeros((0,), device=device, dtype=dtype),
        stats={
            "depth_candidate_tiles": 0.0,
            "depth_accepted_tiles": 0.0,
            "depth_plane_hypotheses": 0.0,
            "depth_plane_points": 0.0,
        },
    )


@torch.no_grad()
def build_depth_plane_hypotheses(
    *,
    depth: np.ndarray,
    valid: np.ndarray,
    p_planar: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    device: str | torch.device,
    dtype: torch.dtype,
    min_depth_m: float,
    max_depth_m: float,
    planar_threshold: float,
    sample_stride: int,
    tile_size_px: int,
    min_tile_points: int,
    tile_fit_max_error_m: float,
    normal_angle_deg: float,
    plane_distance_m: float,
    min_plane_points: int,
    max_planes: int,
) -> DepthPlaneHypotheses:
    """Fit broad plane hypotheses from one monocular depth guide.

    This creates structural evidence, not a direct loss.  Only pixels with high
    local-planar confidence are sampled.  Small image tiles must fit a plane;
    then compatible tile planes are merged into broader depth-supported plane
    hypotheses.  Mesh coplane grouping can use these broad hypotheses to join
    fragmented face buckets, while still validating faces by distance/normal.
    """

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.asarray(valid).astype(bool)
    p_planar = np.asarray(p_planar, dtype=np.float32)
    height, width = depth.shape[:2]
    empty = _empty_depth_planes(device, dtype)
    if height == 0 or width == 0:
        return empty

    depth_ok = valid & np.isfinite(depth) & (depth >= float(min_depth_m)) & (p_planar >= float(planar_threshold))
    if float(max_depth_m) > 0.0:
        depth_ok &= depth <= float(max_depth_m)
    if not np.any(depth_ok):
        return empty

    stride = max(int(sample_stride), 1)
    tile_size = max(int(tile_size_px), stride)
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    sample_ok = depth_ok[yy, xx]
    if not np.any(sample_ok):
        return empty
    yy = yy[sample_ok]
    xx = xx[sample_ok]
    zz = depth[yy, xx].astype(np.float64)

    K = np.asarray(K, dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x_cam = (xx.astype(np.float64) - cx) / max(fx, 1e-8) * zz
    y_cam = (yy.astype(np.float64) - cy) / max(fy, 1e-8) * zz
    points_cam = np.stack([x_cam, y_cam, zz], axis=1)
    camtoworld = np.asarray(camtoworld, dtype=np.float64)
    points_world = (camtoworld[:3, :3] @ points_cam.T).T + camtoworld[:3, 3]

    tile_normals: list[np.ndarray] = []
    tile_offsets: list[float] = []
    tile_weights: list[float] = []
    tile_points: list[np.ndarray] = []
    candidate_tiles = 0
    accepted_tiles = 0
    for y0 in range(0, height, tile_size):
        y1 = y0 + tile_size
        in_y = (yy >= y0) & (yy < y1)
        if not np.any(in_y):
            continue
        for x0 in range(0, width, tile_size):
            in_tile = in_y & (xx >= x0) & (xx < x0 + tile_size)
            count = int(np.count_nonzero(in_tile))
            if count < int(min_tile_points):
                continue
            candidate_tiles += 1
            points = points_world[in_tile]
            normal, offset, residuals = _fit_plane_np(points)
            if float(np.median(residuals)) > float(tile_fit_max_error_m):
                continue
            accepted_tiles += 1
            tile_normals.append(normal)
            tile_offsets.append(offset)
            tile_weights.append(float(count))
            tile_points.append(points)

    if not tile_normals:
        return empty

    angle_rad = math.radians(max(float(normal_angle_deg), 1e-3))
    normal_bin = max(2.0 * math.sin(0.5 * angle_rad), 1e-4)
    offset_bin = max(float(plane_distance_m), 1e-5)
    normals_np = np.stack(tile_normals, axis=0)
    offsets_np = np.asarray(tile_offsets, dtype=np.float64)
    keys = np.concatenate(
        [
            np.round(normals_np / normal_bin).astype(np.int32),
            np.round(offsets_np[:, None] / offset_bin).astype(np.int32),
        ],
        axis=1,
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    sorted_inverse = inverse[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_inverse)) + 1])
    ends = np.concatenate([starts[1:], [len(sorted_inverse)]])

    plane_normals: list[np.ndarray] = []
    plane_offsets: list[float] = []
    plane_weights: list[float] = []
    for start, end in zip(starts, ends):
        tile_ids = order[start:end]
        support = float(np.sum([tile_weights[i] for i in tile_ids]))
        if support < float(min_plane_points):
            continue
        points = np.concatenate([tile_points[i] for i in tile_ids], axis=0)
        normal, offset, residuals = _fit_plane_np(points)
        if float(np.median(residuals)) > float(plane_distance_m):
            continue
        plane_normals.append(normal)
        plane_offsets.append(offset)
        plane_weights.append(support)

    if not plane_normals:
        return empty
    ranked = np.argsort(np.asarray(plane_weights))[::-1]
    if int(max_planes) > 0:
        ranked = ranked[: int(max_planes)]
    normals_t = torch.as_tensor(np.stack([plane_normals[i] for i in ranked], axis=0), device=device, dtype=dtype)
    offsets_t = torch.as_tensor(np.asarray([plane_offsets[i] for i in ranked]), device=device, dtype=dtype)
    weights_t = torch.as_tensor(np.asarray([plane_weights[i] for i in ranked]), device=device, dtype=dtype)
    return DepthPlaneHypotheses(
        normals=normals_t,
        offsets=offsets_t,
        weights=weights_t,
        stats={
            "depth_candidate_tiles": float(candidate_tiles),
            "depth_accepted_tiles": float(accepted_tiles),
            "depth_plane_hypotheses": float(normals_t.shape[0]),
            "depth_plane_points": float(weights_t.sum().detach().cpu()),
        },
    )


@torch.no_grad()
def build_coplane_targets(
    *,
    vertices: Tensor,
    faces: Tensor,
    step: int,
    normal_angle_deg: float,
    plane_distance_m: float,
    min_faces: int,
    min_group_area_m2: float,
    min_face_area_m2: float,
    max_groups: int,
    depth_planes: DepthPlaneHypotheses | None = None,
    depth_plane_distance_m: float = 0.08,
    depth_normal_angle_deg: float = 12.0,
) -> CoplaneTargets:
    """Group coplanar faces and cache target planes.

    If depth planes are provided, they are used first as broad structural
    hypotheses.  Faces still have to agree with each depth plane by current mesh
    distance and normal, so noisy depth does not blindly assign every face.  Any
    unassigned faces then use the original mesh-only normal/offset bucketing.
    """

    device = vertices.device
    dtype = vertices.dtype
    face_count = int(faces.shape[0])
    target_normals = torch.zeros((face_count, 3), device=device, dtype=dtype)
    target_offsets = torch.zeros((face_count,), device=device, dtype=dtype)
    target_weights = torch.zeros((face_count,), device=device, dtype=dtype)
    target_group_ids = torch.full((face_count,), -1, device=device, dtype=torch.long)

    def finish(groups: list[np.ndarray], group_area_sums: list[float], candidate_groups: int, depth_assisted_groups: int, depth_assisted_faces: int) -> CoplaneTargets:
        accepted_faces = int(torch.count_nonzero(target_weights > 0).detach().cpu())
        stats = {
            "refreshed_step": float(step),
            "total_faces": float(face_count),
            "candidate_groups": float(candidate_groups),
            "depth_plane_hypotheses": float(0.0 if depth_planes is None else depth_planes.normals.shape[0]),
            "depth_assisted_groups": float(depth_assisted_groups),
            "depth_assisted_faces": float(depth_assisted_faces),
            "accepted_groups": float(len(groups)),
            "accepted_faces": float(accepted_faces),
            "accepted_face_fraction": float(accepted_faces / max(face_count, 1)),
            "mean_group_faces": float(np.mean([len(group) for group in groups])) if groups else 0.0,
            "mean_group_area_m2": float(np.mean(group_area_sums)) if groups else 0.0,
        }
        if depth_planes is not None:
            stats.update(depth_planes.stats)
        return CoplaneTargets(target_normals, target_offsets, target_weights, target_group_ids, face_count, int(step), stats)

    if face_count == 0:
        return finish([], [], 0, 0, 0)

    _, centers, normals, areas = _face_geometry(vertices.detach(), faces.detach())
    normals = _canonicalize_normals(normals)
    finite = torch.isfinite(centers).all(dim=1) & torch.isfinite(normals).all(dim=1) & torch.isfinite(areas)
    valid = finite & (areas >= float(min_face_area_m2))
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    if valid_indices.numel() == 0:
        return finish([], [], 0, 0, 0)

    groups: list[np.ndarray] = []
    group_area_sums: list[float] = []
    assigned = torch.zeros((face_count,), device=device, dtype=torch.bool)
    depth_assisted_groups = 0
    depth_assisted_faces = 0

    if depth_planes is not None and depth_planes.normals.numel() > 0:
        prior_normals, prior_offsets = _canonicalize_planes(
            depth_planes.normals.to(device=device, dtype=dtype),
            depth_planes.offsets.to(device=device, dtype=dtype),
        )
        prior_weights = depth_planes.weights.to(device=device, dtype=dtype)
        prior_order = torch.argsort(prior_weights, descending=True)
        prior_cos = math.cos(math.radians(max(float(depth_normal_angle_deg), 1e-3)))
        for prior_idx in prior_order.tolist():
            plane_normal = prior_normals[prior_idx]
            plane_offset = prior_offsets[prior_idx]
            unassigned_valid = valid & (~assigned)
            face_distance = torch.abs(centers @ plane_normal + plane_offset)
            normal_agreement = torch.abs(normals @ plane_normal)
            face_ids = torch.nonzero(
                unassigned_valid
                & (face_distance <= float(depth_plane_distance_m))
                & (normal_agreement >= float(prior_cos)),
                as_tuple=False,
            ).squeeze(1)
            if int(face_ids.numel()) < int(min_faces):
                continue
            kept_area = areas[face_ids].sum()
            if float(kept_area.detach().cpu()) < float(min_group_area_m2):
                continue
            group_id = len(groups)
            groups.append(face_ids.cpu().numpy())
            group_area_sums.append(float(kept_area.detach().cpu()))
            target_normals[face_ids] = plane_normal.to(dtype=dtype)
            target_offsets[face_ids] = plane_offset.to(dtype=dtype)
            target_weights[face_ids] = areas[face_ids].to(dtype=dtype)
            target_group_ids[face_ids] = int(group_id)
            assigned[face_ids] = True
            depth_assisted_groups += 1
            depth_assisted_faces += int(face_ids.numel())
            if int(max_groups) > 0 and len(groups) >= int(max_groups):
                return finish(groups, group_area_sums, 0, depth_assisted_groups, depth_assisted_faces)

    remaining_valid_indices = torch.nonzero(valid & (~assigned), as_tuple=False).squeeze(1)
    if remaining_valid_indices.numel() == 0:
        return finish(groups, group_area_sums, 0, depth_assisted_groups, depth_assisted_faces)

    angle_rad = math.radians(max(float(normal_angle_deg), 1e-3))
    cos_threshold = math.cos(angle_rad)
    normal_bin = max(2.0 * math.sin(0.5 * angle_rad), 1e-4)
    offset_bin = max(float(plane_distance_m) * 0.5, 1e-5)
    offsets = -torch.sum(normals * centers, dim=1)
    q_normals = torch.round(normals[remaining_valid_indices] / normal_bin).to(torch.int32).cpu().numpy()
    q_offsets = torch.round(offsets[remaining_valid_indices] / offset_bin).to(torch.int32).cpu().numpy()[:, None]
    keys = np.concatenate([q_normals, q_offsets], axis=1)
    valid_indices_np = remaining_valid_indices.cpu().numpy()

    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="mergesort")
    sorted_inverse = inverse[order]
    sorted_faces = valid_indices_np[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(sorted_inverse)) + 1])
    ends = np.concatenate([starts[1:], [len(sorted_inverse)]])

    candidate_groups = 0
    for start, end in zip(starts, ends):
        count = int(end - start)
        if count < int(min_faces):
            continue
        candidate_groups += 1
        group_ids = torch.as_tensor(sorted_faces[start:end], device=device, dtype=torch.long)
        group_areas = areas[group_ids]
        area_sum = group_areas.sum()
        if float(area_sum.detach().cpu()) < float(min_group_area_m2):
            continue

        group_normals = normals[group_ids]
        avg_normal = torch.sum(group_normals * group_areas[:, None], dim=0)
        avg_norm = torch.linalg.norm(avg_normal)
        if avg_norm <= 1e-12:
            continue
        plane_normal = avg_normal / avg_norm
        group_centers = centers[group_ids]
        plane_offset = -torch.sum(group_areas * (group_centers @ plane_normal)) / area_sum.clamp_min(1e-12)

        point_distance = torch.abs(group_centers @ plane_normal + plane_offset)
        normal_agreement = torch.abs(group_normals @ plane_normal)
        keep = (point_distance <= float(plane_distance_m)) & (normal_agreement >= float(cos_threshold))
        if torch.count_nonzero(keep) < int(min_faces):
            continue
        kept_ids = group_ids[keep]
        kept_area = areas[kept_ids].sum()
        if float(kept_area.detach().cpu()) < float(min_group_area_m2):
            continue

        kept_centers = centers[kept_ids]
        kept_normals = normals[kept_ids]
        kept_areas = areas[kept_ids]
        plane_normal = torch.sum(kept_normals * kept_areas[:, None], dim=0)
        plane_normal = plane_normal / torch.linalg.norm(plane_normal).clamp_min(1e-12)
        plane_offset = -torch.sum(kept_areas * (kept_centers @ plane_normal)) / kept_areas.sum().clamp_min(1e-12)

        groups.append(kept_ids.cpu().numpy())
        group_area_sums.append(float(kept_areas.sum().detach().cpu()))
        group_id = len(groups) - 1
        target_normals[kept_ids] = plane_normal.to(dtype=dtype)
        target_offsets[kept_ids] = plane_offset.to(dtype=dtype)
        target_weights[kept_ids] = kept_areas.to(dtype=dtype)
        target_group_ids[kept_ids] = int(group_id)
        if int(max_groups) > 0 and len(groups) >= int(max_groups):
            break

    return finish(groups, group_area_sums, candidate_groups, depth_assisted_groups, depth_assisted_faces)


def should_refresh_coplane_targets(
    *,
    targets: CoplaneTargets | None,
    face_count: int,
    step: int,
    refresh_every: int,
) -> bool:
    """Decide whether cached target planes are stale."""

    if targets is None:
        return True
    if targets.face_count != int(face_count):
        return True
    return int(refresh_every) > 0 and int(step) - int(targets.last_refresh_step) >= int(refresh_every)


def compute_coplane_regularization_loss(
    *,
    vertices: Tensor,
    faces: Tensor,
    targets: CoplaneTargets | None,
    point_lambda: float,
    normal_lambda: float,
    huber_delta_m: float,
) -> tuple[Tensor, Tensor, Tensor, dict[str, float]]:
    """Compute the differentiable coplane loss for cached plane targets.

    For each accepted face f with target plane (n_g, d_g):

        L_point = mean_v rho(n_g dot x_{f,v} + d_g)
        L_normal = 1 - |normal_f dot n_g|
        L_coplane = lambda_p * mean_f L_point + lambda_n * mean_f L_normal

    The target planes are detached from the graph; gradients update the current
    mesh vertices and, through OMeGa's mesh-to-splat update, the attached splats.
    """

    device = vertices.device
    dtype = vertices.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if targets is None or targets.face_count != int(faces.shape[0]):
        return zero, zero, zero, {"active_faces": 0.0, "mean_abs_plane_distance_m": 0.0}

    weights = targets.weights.to(device=device, dtype=dtype)
    active = weights > 0
    if torch.count_nonzero(active) == 0:
        return zero, zero, zero, {"active_faces": 0.0, "mean_abs_plane_distance_m": 0.0}

    plane_normals = targets.normals.to(device=device, dtype=dtype)
    plane_offsets = targets.offsets.to(device=device, dtype=dtype)
    face_vertices, _, face_normals, _ = _face_geometry(vertices, faces)

    vertex_distance = torch.sum(face_vertices * plane_normals[:, None, :], dim=-1) + plane_offsets[:, None]
    point_penalty = _huber(vertex_distance, float(huber_delta_m)).mean(dim=1)
    normal_penalty = 1.0 - torch.abs(torch.sum(face_normals * plane_normals, dim=1)).clamp(0.0, 1.0)

    denom = weights[active].sum().clamp_min(1e-12)
    point_loss = torch.sum(weights[active] * point_penalty[active]) / denom
    normal_loss = torch.sum(weights[active] * normal_penalty[active]) / denom
    scaled_loss = float(point_lambda) * point_loss + float(normal_lambda) * normal_loss

    mean_abs_distance = torch.sum(weights[active] * torch.mean(torch.abs(vertex_distance[active]), dim=1)) / denom
    stats = {
        "active_faces": float(torch.count_nonzero(active).detach().cpu()),
        "mean_abs_plane_distance_m": float(mean_abs_distance.detach().cpu()),
    }
    return scaled_loss, point_loss, normal_loss, stats
