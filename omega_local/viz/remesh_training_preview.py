"""Preview panels for in-training OMeGa remesh events.

The remesh preview is CPU-only and mirrors the post-hoc remesh sweep diagnostic:
for one fixed camera frame it renders edges, face normals, and one-sided
candidate-to-before-mesh deviation for the mesh before and after the remesh.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from omega_local.viz.mesh_training_preview import resize_rgb_and_intrinsics_for_preview


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(str(path))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Mesh has no triangles: {path}")
    return vertices, faces


def _rotate_clockwise(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(np.asarray(image[..., :3], dtype=np.uint8), k=3))


def _label_image(image: np.ndarray, label: str) -> np.ndarray:
    out = np.asarray(image[..., :3], dtype=np.uint8).copy()
    height, width = out.shape[:2]
    label_h = max(34, min(70, height // 18))
    font_scale = max(0.38, min(1.05, width / 1700.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    while font_scale > 0.34:
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        if text_size[0] <= width - 20:
            break
        font_scale *= 0.9
        thickness = max(1, int(round(font_scale * 2.0)))
    cv2.rectangle(out, (0, 0), (width, label_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        label,
        (10, max(23, min(label_h - 9, label_h // 2 + 9))),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def _fit_image_to_shape(image: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    image = np.asarray(image[..., :3], dtype=np.uint8)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _project_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    near_plane: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world_to_cam = np.linalg.inv(np.asarray(camtoworld, dtype=np.float64))
    vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    vertices_cam = (world_to_cam @ vertices_h.T).T[:, :3]
    z = vertices_cam[:, 2]
    projected = np.empty((len(vertices), 2), dtype=np.float32)
    projected[:, 0] = K[0, 0] * (vertices_cam[:, 0] / np.maximum(z, 1e-8)) + K[0, 2]
    projected[:, 1] = K[1, 1] * (vertices_cam[:, 1] / np.maximum(z, 1e-8)) + K[1, 2]
    face_vertices_cam = vertices_cam[faces]
    valid_faces = np.all(face_vertices_cam[:, :, 2] > float(near_plane), axis=1)
    mean_depth = np.mean(face_vertices_cam[:, :, 2], axis=1)
    return projected, face_vertices_cam, valid_faces, mean_depth


def _visible_face_ids(projected_faces: np.ndarray, valid_faces: np.ndarray, width: int, height: int) -> np.ndarray:
    min_xy = np.min(projected_faces, axis=1)
    max_xy = np.max(projected_faces, axis=1)
    in_bounds = (
        (max_xy[:, 0] >= 0)
        & (max_xy[:, 1] >= 0)
        & (min_xy[:, 0] < width)
        & (min_xy[:, 1] < height)
    )
    finite = np.isfinite(projected_faces).all(axis=(1, 2))
    return np.nonzero(valid_faces & in_bounds & finite)[0]


def _downsample_faces_evenly(faces: np.ndarray, max_faces: int) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64)
    if int(max_faces) <= 0 or len(faces) <= int(max_faces):
        return faces
    stride = int(np.ceil(len(faces) / float(max_faces)))
    return faces[::stride]


def _reference_surface_points(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> np.ndarray:
    faces = _downsample_faces_evenly(faces, int(max_faces))
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    midpoints = np.concatenate(
        [
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ],
        axis=0,
    )
    return np.vstack([vertices, centers, midpoints]).astype(np.float64, copy=False)


def _face_deviation_to_reference(vertices: np.ndarray, faces: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros((0,), dtype=np.float64)
    triangles = vertices[faces]
    centers = triangles.mean(axis=1, keepdims=True)
    midpoints = np.stack(
        [
            0.5 * (triangles[:, 0] + triangles[:, 1]),
            0.5 * (triangles[:, 1] + triangles[:, 2]),
            0.5 * (triangles[:, 2] + triangles[:, 0]),
        ],
        axis=1,
    )
    samples = np.concatenate([triangles, centers, midpoints], axis=1)
    distances, _ = cKDTree(reference_points).query(samples.reshape(-1, 3), k=1, workers=-1)
    distances = np.asarray(distances, dtype=np.float64).reshape(len(faces), -1)
    return np.sqrt(np.mean(distances * distances, axis=1))


def _summarize_errors(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": float(len(finite)),
        "mean": float(np.mean(finite)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def _render_mesh_diagnostics(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    face_errors: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    image_shape: tuple[int, int],
    near_plane: float,
    max_faces: int,
    wire_thickness: int,
    error_vmax_m: float,
) -> dict[str, Any]:
    height, width = int(image_shape[0]), int(image_shape[1])
    wire = np.zeros((height, width, 3), dtype=np.uint8)
    normal = np.zeros((height, width, 3), dtype=np.uint8)
    error = np.zeros((height, width, 3), dtype=np.uint8)
    valid_u8 = np.zeros((height, width), dtype=np.uint8)

    faces_to_render = faces
    face_errors_to_render = face_errors
    if max_faces > 0 and len(faces) > max_faces:
        stride = int(np.ceil(len(faces) / float(max_faces)))
        faces_to_render = faces[::stride]
        face_errors_to_render = face_errors[::stride]

    projected, face_vertices_cam, valid_faces, mean_depth = _project_mesh(
        vertices,
        faces_to_render,
        K,
        camtoworld,
        near_plane,
    )
    projected_faces = projected[faces_to_render]
    visible_ids = _visible_face_ids(projected_faces, valid_faces, width, height)
    if len(visible_ids) > 0:
        order = visible_ids[np.argsort(mean_depth[visible_ids])[::-1]]
        tri_cam = face_vertices_cam[order]
        normals = _safe_normalize(np.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0]))
        normal_colors = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        vmax = float(error_vmax_m)
        if vmax <= 0.0:
            finite_visible = face_errors_to_render[order][np.isfinite(face_errors_to_render[order])]
            vmax = float(np.quantile(finite_visible, 0.95)) if len(finite_visible) else 0.05
        vmax = max(vmax, 1e-6)
        normalized_error = np.clip(face_errors_to_render[order] / vmax, 0.0, 1.0)
        error_bgr = cv2.applyColorMap(np.rint(normalized_error * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        error_colors = error_bgr[:, 0, ::-1]
        wire_thickness = max(1, int(wire_thickness))
        drawn_edges: set[tuple[int, int]] = set()
        for face_id, normal_color, error_color in zip(order, normal_colors, error_colors):
            pts = np.round(projected_faces[face_id]).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], -2 * width, 3 * width)
            pts[:, 1] = np.clip(pts[:, 1], -2 * height, 3 * height)
            cv2.fillConvexPoly(normal, pts, color=normal_color.tolist(), lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(error, pts, color=error_color.tolist(), lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(valid_u8, pts, color=1, lineType=cv2.LINE_AA)
            face_vertices = faces_to_render[face_id]
            for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
                vertex_a = int(face_vertices[local_a])
                vertex_b = int(face_vertices[local_b])
                edge_key = (vertex_a, vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                cv2.line(
                    wire,
                    tuple(pts[local_a]),
                    tuple(pts[local_b]),
                    color=(255, 255, 255),
                    thickness=wire_thickness,
                    lineType=cv2.LINE_8,
                )
    return {
        "wire": wire,
        "normal": normal,
        "error": error,
        "valid": valid_u8 > 0,
        "visible_faces": int(len(visible_ids)),
    }


def _compose_panel(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    before_label: str,
    after_label: str,
    before_splat_rgb: np.ndarray | None,
    after_splat_rgb: np.ndarray | None,
    rotate_clockwise: bool,
) -> np.ndarray:
    rows = []
    image_shape = before["wire"].shape[:2]
    row_specs = [
        (_fit_image_to_shape(before_splat_rgb, image_shape), _fit_image_to_shape(after_splat_rgb, image_shape), "splat RGB"),
        (before["wire"], after["wire"], "edges"),
        (before["normal"], after["normal"], "normal"),
        (before["error"], after["error"], "error"),
    ]
    for left, right, name in row_specs:
        if rotate_clockwise:
            left = _rotate_clockwise(left)
            right = _rotate_clockwise(right)
        rows.append(
            np.concatenate(
                [
                    _label_image(left, f"{before_label} | {name}"),
                    _label_image(right, f"{after_label} | {name}"),
                ],
                axis=1,
            )
        )
    width = max(row.shape[1] for row in rows)
    rows = [row if row.shape[1] == width else np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))) for row in rows]
    return np.concatenate(rows, axis=0)


def write_training_remesh_preview(
    *,
    out_dir: Path,
    step: int,
    frame_index: int,
    before_mesh: Path,
    after_mesh: Path,
    rgb: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    before_splat_rgb: np.ndarray | None = None,
    after_splat_rgb: np.ndarray | None = None,
    max_width: int,
    near_plane: float,
    max_faces: int,
    rotate_clockwise: bool,
    wire_thickness: int,
    error_vmax_m: float,
    error_reference_max_faces: int,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    before_vertices, before_faces = _load_mesh(before_mesh)
    after_vertices, after_faces = _load_mesh(after_mesh)
    rgb_small, K_small, image_scale = resize_rgb_and_intrinsics_for_preview(
        np.asarray(rgb[..., :3], dtype=np.uint8),
        np.asarray(K, dtype=np.float64),
        max_width=int(max_width),
    )
    image_shape = rgb_small.shape[:2]
    reference_points = _reference_surface_points(before_vertices, before_faces, int(error_reference_max_faces))
    before_errors = np.zeros((len(before_faces),), dtype=np.float64)
    after_errors = _face_deviation_to_reference(after_vertices, after_faces, reference_points)
    before_render = _render_mesh_diagnostics(
        vertices=before_vertices,
        faces=before_faces,
        face_errors=before_errors,
        K=K_small,
        camtoworld=camtoworld,
        image_shape=image_shape,
        near_plane=float(near_plane),
        max_faces=int(max_faces),
        wire_thickness=int(wire_thickness),
        error_vmax_m=float(error_vmax_m),
    )
    after_render = _render_mesh_diagnostics(
        vertices=after_vertices,
        faces=after_faces,
        face_errors=after_errors,
        K=K_small,
        camtoworld=camtoworld,
        image_shape=image_shape,
        near_plane=float(near_plane),
        max_faces=int(max_faces),
        wire_thickness=int(wire_thickness),
        error_vmax_m=float(error_vmax_m),
    )
    after_stats = _summarize_errors(after_errors)
    before_label = f"before step {step} faces={len(before_faces):,}"
    after_label = f"after faces={len(after_faces):,} err95={after_stats['p95'] * 100.0:.1f}cm"
    panel = _compose_panel(
        before_render,
        after_render,
        before_label=before_label,
        after_label=after_label,
        before_splat_rgb=before_splat_rgb,
        after_splat_rgb=after_splat_rgb,
        rotate_clockwise=bool(rotate_clockwise),
    )
    panel_path = out_dir / f"training_remesh_preview_step_{int(step):06d}.png"
    imageio.imwrite(panel_path, panel)
    outputs = {"panel": str(panel_path)}
    before_splat_image = _fit_image_to_shape(before_splat_rgb, image_shape)
    after_splat_image = _fit_image_to_shape(after_splat_rgb, image_shape)
    imageio.imwrite(out_dir / "before_splat_rgb.png", before_splat_image)
    imageio.imwrite(out_dir / "after_splat_rgb.png", after_splat_image)
    outputs["before_splat_rgb"] = str(out_dir / "before_splat_rgb.png")
    outputs["after_splat_rgb"] = str(out_dir / "after_splat_rgb.png")
    for prefix, render in (("before", before_render), ("after", after_render)):
        for key in ("wire", "normal", "error"):
            image_path = out_dir / f"{prefix}_{key}.png"
            imageio.imwrite(image_path, render[key])
            outputs[f"{prefix}_{key}"] = str(image_path)
    summary = {
        "step": int(step),
        "frame_index": int(frame_index),
        "image_scale": float(image_scale),
        "before_mesh": str(before_mesh),
        "after_mesh": str(after_mesh),
        "before_faces": int(len(before_faces)),
        "after_faces": int(len(after_faces)),
        "before_visible_faces": int(before_render["visible_faces"]),
        "after_visible_faces": int(after_render["visible_faces"]),
        "error_vmax_m": float(error_vmax_m),
        "error_reference_samples": int(len(reference_points)),
        "after_error_to_before_m": after_stats,
        "outputs": outputs,
    }
    (out_dir / "training_remesh_preview_summary.json").write_text(
        __import__("json").dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary
