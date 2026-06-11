"""CPU mesh view-buffer rendering for the redesigned local remesh pipeline.

This module renders a triangle mesh into the same undistorted/cropped camera
views used by OMeGa.  The first implementation is intentionally deterministic
and CPU-only: it writes face-id, depth, mesh-normal preview, and mesh edge masks
that can be inspected before any remeshing policy is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ViewBuffer:
    face_id: np.ndarray
    depth: np.ndarray
    mesh_normal_rgb: np.ndarray
    mesh_edge_mask: np.ndarray
    visible_face_ids: np.ndarray
    width: int
    height: int


def resize_camera_inputs(
    *,
    image: np.ndarray,
    normal_map: np.ndarray | None,
    normal_valid: np.ndarray | None,
    K: np.ndarray,
    max_width: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray, float]:
    """Resize image-space inputs and intrinsics by the same factor."""

    image = np.asarray(image[..., :3], dtype=np.uint8)
    K_scaled = np.asarray(K, dtype=np.float64).copy()
    if int(max_width) <= 0 or image.shape[1] <= int(max_width):
        return image, normal_map, normal_valid, K_scaled, 1.0

    scale = float(max_width) / float(image.shape[1])
    width = int(round(image.shape[1] * scale))
    height = int(round(image.shape[0] * scale))
    image_small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    normal_small = None
    valid_small = None
    if normal_map is not None:
        normal_small = cv2.resize(np.asarray(normal_map, dtype=np.float32), (width, height), interpolation=cv2.INTER_AREA)
        norm = np.linalg.norm(normal_small, axis=2, keepdims=True)
        normal_small = np.divide(normal_small, np.maximum(norm, 1e-12), out=np.zeros_like(normal_small))
    if normal_valid is not None:
        valid_small = cv2.resize(np.asarray(normal_valid, dtype=np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale
    return image_small, normal_small, valid_small, K_scaled, scale


def project_vertices(
    vertices: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world vertices to image pixels with OMeGa's z-forward cameras."""

    world_to_cam = np.linalg.inv(np.asarray(camtoworld, dtype=np.float64))
    vertices_h = np.concatenate([np.asarray(vertices, dtype=np.float64), np.ones((len(vertices), 1))], axis=1)
    vertices_cam = (world_to_cam @ vertices_h.T).T[:, :3]
    z = vertices_cam[:, 2]
    projected = np.empty((len(vertices), 2), dtype=np.float64)
    safe_z = np.maximum(z, 1e-8)
    projected[:, 0] = K[0, 0] * (vertices_cam[:, 0] / safe_z) + K[0, 2]
    projected[:, 1] = K[1, 1] * (vertices_cam[:, 1] / safe_z) + K[1, 2]
    return projected, vertices_cam, z


def visible_face_candidates(
    *,
    projected_vertices: np.ndarray,
    vertex_depth: np.ndarray,
    faces: np.ndarray,
    width: int,
    height: int,
    near_plane: float,
) -> np.ndarray:
    projected_faces = projected_vertices[faces]
    z_faces = vertex_depth[faces]
    finite = np.isfinite(projected_faces).all(axis=(1, 2)) & np.isfinite(z_faces).all(axis=1)
    in_front = np.all(z_faces > float(near_plane), axis=1)
    min_xy = np.min(projected_faces, axis=1)
    max_xy = np.max(projected_faces, axis=1)
    in_bounds = (
        (max_xy[:, 0] >= 0.0)
        & (max_xy[:, 1] >= 0.0)
        & (min_xy[:, 0] < float(width))
        & (min_xy[:, 1] < float(height))
    )
    return np.nonzero(finite & in_front & in_bounds)[0].astype(np.int64)


def _rasterize_one_triangle(
    *,
    face_id_value: int,
    pts: np.ndarray,
    depths: np.ndarray,
    normal_rgb: np.ndarray,
    face_id_buffer: np.ndarray,
    depth_buffer: np.ndarray,
    normal_buffer: np.ndarray,
) -> bool:
    height, width = face_id_buffer.shape
    x0 = max(int(np.floor(np.min(pts[:, 0]))), 0)
    x1 = min(int(np.ceil(np.max(pts[:, 0]))) + 1, width)
    y0 = max(int(np.floor(np.min(pts[:, 1]))), 0)
    y1 = min(int(np.ceil(np.max(pts[:, 1]))) + 1, height)
    if x1 <= x0 or y1 <= y0:
        return False

    p0, p1, p2 = pts.astype(np.float64)
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(float(denom)) < 1e-12:
        return False

    yy, xx = np.mgrid[y0:y1, x0:x1]
    px = xx.astype(np.float64) + 0.5
    py = yy.astype(np.float64) + 0.5
    w0 = ((p1[1] - p2[1]) * (px - p2[0]) + (p2[0] - p1[0]) * (py - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (px - p2[0]) + (p0[0] - p2[0]) * (py - p2[1])) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(inside):
        return False

    inv_depth = (w0 / depths[0]) + (w1 / depths[1]) + (w2 / depths[2])
    tri_depth = np.divide(1.0, np.maximum(inv_depth, 1e-12))
    current = depth_buffer[y0:y1, x0:x1]
    update = inside & (tri_depth < current)
    if not np.any(update):
        return False

    current[update] = tri_depth[update].astype(np.float32)
    face_id_view = face_id_buffer[y0:y1, x0:x1]
    face_id_view[update] = int(face_id_value)
    normal_view = normal_buffer[y0:y1, x0:x1]
    normal_view[update] = normal_rgb.astype(np.uint8)
    return True


def _edge_mask_from_face_ids(face_id: np.ndarray, thickness: int) -> np.ndarray:
    valid = face_id >= 0
    edge = np.zeros(face_id.shape, dtype=np.uint8)
    diff_x = valid[:, 1:] & valid[:, :-1] & (face_id[:, 1:] != face_id[:, :-1])
    edge_right = edge[:, 1:]
    edge_left = edge[:, :-1]
    edge_right[diff_x] = 1
    edge_left[diff_x] = 1
    diff_y = valid[1:, :] & valid[:-1, :] & (face_id[1:, :] != face_id[:-1, :])
    edge_down = edge[1:, :]
    edge_up = edge[:-1, :]
    edge_down[diff_y] = 1
    edge_up[diff_y] = 1
    silhouette_x = valid[:, 1:] != valid[:, :-1]
    edge_right[silhouette_x] = 1
    edge_left[silhouette_x] = 1
    silhouette_y = valid[1:, :] != valid[:-1, :]
    edge_down[silhouette_y] = 1
    edge_up[silhouette_y] = 1
    if int(thickness) > 1:
        kernel = np.ones((int(thickness), int(thickness)), dtype=np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
    return edge.astype(bool)


def render_mesh_view_buffer(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals_world: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    image_shape: tuple[int, int],
    near_plane: float,
    edge_thickness: int = 1,
) -> ViewBuffer:
    """Rasterize nearest visible triangles into face-id/depth buffers."""

    height, width = int(image_shape[0]), int(image_shape[1])
    projected, _, vertex_depth = project_vertices(vertices, K, camtoworld)
    candidates = visible_face_candidates(
        projected_vertices=projected,
        vertex_depth=vertex_depth,
        faces=faces,
        width=width,
        height=height,
        near_plane=float(near_plane),
    )
    face_id = np.full((height, width), -1, dtype=np.int32)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    normal_rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
    normal_rgb_by_face = np.clip((np.asarray(face_normals_world, dtype=np.float64) * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    for face_id_value in candidates:
        tri = faces[int(face_id_value)]
        _rasterize_one_triangle(
            face_id_value=int(face_id_value),
            pts=projected[tri],
            depths=vertex_depth[tri].astype(np.float64),
            normal_rgb=normal_rgb_by_face[int(face_id_value)],
            face_id_buffer=face_id,
            depth_buffer=depth,
            normal_buffer=normal_rgb_buffer,
        )

    visible_face_ids = np.unique(face_id[face_id >= 0]).astype(np.int32)
    edge_mask = _edge_mask_from_face_ids(face_id, int(edge_thickness))
    depth[~np.isfinite(depth)] = 0.0
    return ViewBuffer(
        face_id=face_id,
        depth=depth.astype(np.float32, copy=False),
        mesh_normal_rgb=normal_rgb_buffer,
        mesh_edge_mask=edge_mask,
        visible_face_ids=visible_face_ids,
        width=width,
        height=height,
    )


def save_view_buffer_npz(path: Path, buffer: ViewBuffer, *, image_name: str, image_scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        face_id=buffer.face_id,
        depth=buffer.depth,
        mesh_normal_rgb=buffer.mesh_normal_rgb,
        mesh_edge_mask=buffer.mesh_edge_mask.astype(np.uint8),
        visible_face_ids=buffer.visible_face_ids,
        image_name=np.array(str(image_name)),
        image_scale=np.array(float(image_scale), dtype=np.float32),
        width=np.array(int(buffer.width), dtype=np.int32),
        height=np.array(int(buffer.height), dtype=np.int32),
    )
