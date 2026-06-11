"""Fixed-camera mesh previews for local OMeGa training diagnostics.

The preview renderer is intentionally simple and CPU-only. It avoids Open3D
offscreen rendering because that backend can segfault on some CUDA/EGL setups.
The output is diagnostic, not a physically correct render: a painter-style mesh
render is enough to monitor whether the optimized mesh is flattening, drifting,
or gaining unwanted detail over training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def resize_rgb_and_intrinsics_for_preview(
    rgb: np.ndarray,
    K: np.ndarray,
    max_width: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    if max_width <= 0 or rgb.shape[1] <= max_width:
        return rgb, K, 1.0
    scale = float(max_width) / float(rgb.shape[1])
    new_width = int(round(rgb.shape[1] * scale))
    new_height = int(round(rgb.shape[0] * scale))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
    K_scaled = K.copy()
    K_scaled[0, :] *= scale
    K_scaled[1, :] *= scale
    return resized, K_scaled, scale


def _rotate_clockwise(panel: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(panel, k=3))


def _project_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    near_plane: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    world_to_cam = np.linalg.inv(camtoworld)
    vertices_h = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=vertices.dtype)], axis=1)
    vertices_cam = (world_to_cam @ vertices_h.T).T[:, :3]
    z = vertices_cam[:, 2]

    projected = np.empty((len(vertices), 2), dtype=np.float32)
    projected[:, 0] = K[0, 0] * (vertices_cam[:, 0] / np.maximum(z, 1e-8)) + K[0, 2]
    projected[:, 1] = K[1, 1] * (vertices_cam[:, 1] / np.maximum(z, 1e-8)) + K[1, 2]

    face_vertices_cam = vertices_cam[faces]
    valid_faces = np.all(face_vertices_cam[:, :, 2] > float(near_plane), axis=1)
    mean_depth = np.mean(face_vertices_cam[:, :, 2], axis=1)
    return projected, face_vertices_cam, valid_faces, mean_depth


def _visible_face_mask(
    projected_faces: np.ndarray,
    valid_faces: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    min_xy = np.min(projected_faces, axis=1)
    max_xy = np.max(projected_faces, axis=1)
    in_bounds = (
        (max_xy[:, 0] >= 0)
        & (max_xy[:, 1] >= 0)
        & (min_xy[:, 0] < width)
        & (min_xy[:, 1] < height)
    )
    finite = np.isfinite(projected_faces).all(axis=(1, 2))
    return valid_faces & in_bounds & finite


def render_mesh_training_preview(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    rgb: np.ndarray,
    splat_rgb: Optional[np.ndarray] = None,
    max_width: int = 900,
    near_plane: float = 0.05,
    max_faces: int = 0,
    wire_thickness: int = 1,
    wire_alpha: float = 0.30,
) -> Dict[str, np.ndarray]:
    """Render RGB, shaded mesh, normal, splat render, and wireframe panels."""

    rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
    if splat_rgb is not None:
        splat_rgb = np.asarray(splat_rgb[..., :3], dtype=np.uint8)
    K = np.asarray(K, dtype=np.float64)
    camtoworld = np.asarray(camtoworld, dtype=np.float64)
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if max_faces > 0 and len(faces) > max_faces:
        stride = int(np.ceil(len(faces) / float(max_faces)))
        faces = faces[::stride]

    rgb_small, K_small, image_scale = resize_rgb_and_intrinsics_for_preview(rgb, K, max_width=max_width)
    if splat_rgb is None:
        splat_small = np.zeros_like(rgb_small)
    elif splat_rgb.shape[:2] != rgb_small.shape[:2]:
        splat_small = cv2.resize(splat_rgb, (rgb_small.shape[1], rgb_small.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        splat_small = splat_rgb
    height, width = rgb_small.shape[:2]
    mesh_render = np.zeros((height, width, 3), dtype=np.uint8)
    normal_render = np.zeros((height, width, 3), dtype=np.uint8)
    wireframe_render = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)

    projected, face_vertices_cam, valid_faces, mean_depth = _project_mesh(
        vertices,
        faces,
        K_small,
        camtoworld,
        near_plane=near_plane,
    )
    projected_faces = projected[faces]
    visible = _visible_face_mask(projected_faces, valid_faces, width=width, height=height)
    visible_ids = np.nonzero(visible)[0]
    if len(visible_ids) > 0:
        # Draw far-to-near. This painter pass is robust and fast enough for
        # fixed-view training diagnostics, even though it is not a true z-buffer.
        order = visible_ids[np.argsort(mean_depth[visible_ids])[::-1]]
        tri_cam = face_vertices_cam[order]
        normals = _safe_normalize(np.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0]))
        light = np.asarray([0.25, -0.35, 1.0], dtype=np.float64)
        light = light / np.linalg.norm(light)
        diffuse = 0.35 + 0.65 * np.abs(normals @ light)
        mesh_colors = np.clip(np.asarray([170, 178, 188], dtype=np.float64)[None, :] * diffuse[:, None], 0, 255).astype(np.uint8)
        normal_colors = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

        # Use white mesh edges so the wire panel is easy to read. Thickness
        # controls visual weight; alpha is kept in the summary/config for
        # comparability with older runs.
        wire_color = (255, 255, 255)
        wire_thickness = max(1, int(wire_thickness))
        drawn_edges: set[tuple[int, int]] = set()

        for face_id, mesh_color, normal_color in zip(order, mesh_colors, normal_colors):
            pts = np.round(projected_faces[face_id]).astype(np.int32)
            # Keep huge projected coordinates from creating expensive fills.
            pts[:, 0] = np.clip(pts[:, 0], -2 * width, 3 * width)
            pts[:, 1] = np.clip(pts[:, 1], -2 * height, 3 * height)
            cv2.fillConvexPoly(mesh_render, pts, color=mesh_color.tolist(), lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(normal_render, pts, color=normal_color.tolist(), lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(mask, pts, color=255, lineType=cv2.LINE_AA)

            # Draw each shared mesh edge only once. Dense meshes otherwise turn
            # the wire panel into a thick high-contrast texture because every
            # triangle redraws the same interior edges.
            face_vertices = faces[face_id]
            for local_a, local_b in ((0, 1), (1, 2), (2, 0)):
                vertex_a = int(face_vertices[local_a])
                vertex_b = int(face_vertices[local_b])
                edge = (vertex_a, vertex_b) if vertex_a < vertex_b else (vertex_b, vertex_a)
                if edge in drawn_edges:
                    continue
                drawn_edges.add(edge)
                cv2.line(
                    wireframe_render,
                    tuple(pts[local_a]),
                    tuple(pts[local_b]),
                    color=wire_color,
                    thickness=wire_thickness,
                    lineType=cv2.LINE_8,
                )

    return {
        "rgb": rgb_small,
        "splat": splat_small,
        "mesh": mesh_render,
        "normal": normal_render,
        "wireframe": wireframe_render,
        "mask": mask,
        "image_scale": np.asarray([image_scale], dtype=np.float32),
        "rendered_face_count": np.asarray([len(visible_ids)], dtype=np.int64),
    }


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (max(180, len(label) * 11), 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _prepare_panel(panel: np.ndarray, label: str, rotate_clockwise: bool) -> np.ndarray:
    if rotate_clockwise:
        panel = _rotate_clockwise(panel)
    return _label_panel(panel, label)


def make_preview_canvas(
    preview: Dict[str, np.ndarray],
    *,
    step: int,
    frame_index: int,
    phase: str = "pre",
    rotate_clockwise: bool = True,
) -> np.ndarray:
    step_label = f"{step} {phase}" if phase else str(step)
    panels = [
        _prepare_panel(preview["rgb"], f"RGB frame {frame_index}", rotate_clockwise),
        _prepare_panel(preview["splat"], f"Splat step {step_label}", rotate_clockwise),
        _prepare_panel(preview["mesh"], "Mesh", rotate_clockwise),
        _prepare_panel(preview["normal"], "Mesh normal", rotate_clockwise),
        _prepare_panel(preview["wireframe"], "Mesh wire", rotate_clockwise),
    ]
    return np.concatenate(panels, axis=1)


def write_mesh_training_preview(
    *,
    out_dir: Path,
    step: int,
    frame_index: int,
    phase: str = "pre",
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    rgb: np.ndarray,
    splat_rgb: Optional[np.ndarray] = None,
    max_width: int = 900,
    near_plane: float = 0.05,
    max_faces: int = 0,
    rotate_clockwise: bool = True,
    wire_thickness: int = 1,
    wire_alpha: float = 0.30,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preview = render_mesh_training_preview(
        vertices=vertices,
        faces=faces,
        K=K,
        camtoworld=camtoworld,
        rgb=rgb,
        splat_rgb=splat_rgb,
        max_width=max_width,
        near_plane=near_plane,
        max_faces=max_faces,
        wire_thickness=wire_thickness,
        wire_alpha=wire_alpha,
    )
    canvas = make_preview_canvas(preview, step=step, frame_index=frame_index, phase=phase, rotate_clockwise=rotate_clockwise)
    safe_phase = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in phase)
    out_path = out_dir / f"mesh_preview_frame_{frame_index:04d}_step_{step:06d}_{safe_phase}.png"
    imageio.imwrite(out_path, canvas)
    return {
        "path": str(out_path),
        "step": int(step),
        "phase": str(phase),
        "frame_index": int(frame_index),
        "rendered_face_count": int(preview["rendered_face_count"][0]),
        "image_scale": float(preview["image_scale"][0]),
        "wire_thickness": int(wire_thickness),
        "wire_alpha": float(wire_alpha),
    }


def update_mesh_training_contact_sheet(
    *,
    preview_dir: Path,
    out_path: Optional[Path] = None,
    max_images: int = 24,
    columns: int = 1,
) -> Optional[Path]:
    files = sorted(preview_dir.glob("mesh_preview_frame_*_step_*.png"))
    if not files:
        return None
    if max_images > 0:
        files = files[-max_images:]
    images = [imageio.imread(path)[..., :3] for path in files]
    if not images:
        return None

    min_height = min(image.shape[0] for image in images)
    resized: List[np.ndarray] = []
    for image in images:
        if image.shape[0] != min_height:
            scale = min_height / image.shape[0]
            new_width = int(round(image.shape[1] * scale))
            image = cv2.resize(image, (new_width, min_height), interpolation=cv2.INTER_AREA)
        resized.append(image)

    columns = max(int(columns), 1)
    rows: List[np.ndarray] = []
    for start in range(0, len(resized), columns):
        row_images = resized[start : start + columns]
        max_h = max(image.shape[0] for image in row_images)
        max_w = max(image.shape[1] for image in row_images)
        padded = []
        for image in row_images:
            canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
            canvas[: image.shape[0], : image.shape[1]] = image
            padded.append(canvas)
        while len(padded) < columns:
            padded.append(np.zeros((max_h, max_w, 3), dtype=np.uint8))
        rows.append(np.concatenate(padded, axis=1))

    # Old preview folders can contain images produced with different preview
    # widths, and portrait rotation can also change row width when source
    # images differ. Pad rows before vertical concatenation so a debug contact
    # sheet never interrupts training.
    sheet_width = max(row.shape[1] for row in rows)
    rows = [
        row
        if row.shape[1] == sheet_width
        else np.pad(row, ((0, 0), (0, sheet_width - row.shape[1]), (0, 0)), mode="constant")
        for row in rows
    ]
    sheet = np.concatenate(rows, axis=0)
    if out_path is None:
        out_path = preview_dir / "mesh_preview_contact_sheet.png"
    imageio.imwrite(out_path, sheet)
    return out_path
