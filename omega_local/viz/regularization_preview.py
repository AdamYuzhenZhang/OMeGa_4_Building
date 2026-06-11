"""Fixed-frame previews for depth and coplane regularization.

These images are diagnostic overlays, not evaluation renders.  They answer the
question "what is the extension seeing at this iteration?" by putting the depth
prior, rendered depth, weighted residual, coplane groups, and coplane residuals
next to the RGB frame used for the fixed preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np

from omega_local.viz.mesh_training_preview import resize_rgb_and_intrinsics_for_preview


def _rotate_clockwise(panel: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(panel, k=3))


def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
    out = np.asarray(panel[..., :3], dtype=np.uint8).copy()
    cv2.rectangle(out, (0, 0), (max(180, len(label) * 10), 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _prepare_panel(panel: np.ndarray, label: str, rotate_clockwise: bool) -> np.ndarray:
    if rotate_clockwise:
        panel = _rotate_clockwise(panel)
    return _label_panel(panel, label)


def _as_hw(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if array is None:
        return None
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[0, ..., 0]
    elif array.ndim == 4 and array.shape[1] == 1:
        array = array[0, 0]
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array, dtype=np.float32)


def _resize_map(array: Optional[np.ndarray], width: int, height: int, *, nearest: bool = False) -> Optional[np.ndarray]:
    array = _as_hw(array)
    if array is None:
        return None
    if array.shape[:2] == (height, width):
        return array
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.resize(array, (width, height), interpolation=interpolation)


def _depth_color(depth: Optional[np.ndarray], valid: Optional[np.ndarray], vmin: float, vmax: float) -> np.ndarray:
    if depth is None:
        return np.zeros((*valid.shape, 3), dtype=np.uint8) if valid is not None else np.zeros((32, 32, 3), dtype=np.uint8)
    depth = np.asarray(depth, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(depth)
    else:
        valid = np.asarray(valid).astype(bool) & np.isfinite(depth)
    denom = max(float(vmax) - float(vmin), 1e-6)
    normalized = np.clip((depth - float(vmin)) / denom, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def _scalar_color(values: np.ndarray, *, vmax: float, cmap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(values)
    normalized = np.clip(values / max(float(vmax), 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cmap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def _signed_residual_color(residual: np.ndarray, valid: np.ndarray, scale: float) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    valid = np.asarray(valid).astype(bool) & np.isfinite(residual)
    normalized = np.clip(residual / max(float(scale), 1e-6), -1.0, 1.0)
    out = np.zeros((*residual.shape, 3), dtype=np.float32)
    positive = normalized > 0
    negative = normalized < 0
    out[..., :] = 245.0
    out[positive, 0] = 255.0
    out[positive, 1] = 245.0 * (1.0 - normalized[positive])
    out[positive, 2] = 245.0 * (1.0 - normalized[positive])
    out[negative, 0] = 245.0 * (1.0 + normalized[negative])
    out[negative, 1] = 245.0 * (1.0 + normalized[negative])
    out[negative, 2] = 255.0
    out[~valid] = 0.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _guide_category_color(
    p_planar: Optional[np.ndarray],
    p_detail: Optional[np.ndarray],
    p_ridge: Optional[np.ndarray],
    valid: Optional[np.ndarray],
    width: int,
    height: int,
) -> np.ndarray:
    if p_planar is None or p_detail is None or p_ridge is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    p_planar = _resize_map(p_planar, width, height)
    p_detail = _resize_map(p_detail, width, height)
    p_ridge = _resize_map(p_ridge, width, height)
    valid = np.ones((height, width), dtype=bool) if valid is None else _resize_map(valid, width, height, nearest=True).astype(bool)
    out = np.zeros((height, width, 3), dtype=np.float32)
    # Green = local planar, orange = detail, magenta = ridge.
    out += p_planar[..., None] * np.asarray([50, 210, 110], dtype=np.float32)
    out += p_detail[..., None] * np.asarray([255, 150, 35], dtype=np.float32)
    out += p_ridge[..., None] * np.asarray([230, 70, 255], dtype=np.float32)
    out[~valid] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def _project_mesh(vertices: np.ndarray, faces: np.ndarray, K: np.ndarray, camtoworld: np.ndarray, near_plane: float):
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
    return projected[faces], face_vertices_cam, valid_faces, mean_depth


def _visible_face_mask(projected_faces: np.ndarray, valid_faces: np.ndarray, width: int, height: int) -> np.ndarray:
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


def _group_palette(group_ids: np.ndarray) -> np.ndarray:
    """Deterministic bright colors for integer group ids."""

    ids = np.maximum(group_ids.astype(np.int64), 0)
    r = (37 * ids + 71) % 255
    g = (83 * ids + 149) % 255
    b = (131 * ids + 211) % 255
    return np.stack([r, g, b], axis=1).astype(np.uint8)


def _render_face_colors(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    face_colors: np.ndarray,
    active: np.ndarray,
    width: int,
    height: int,
    near_plane: float,
) -> tuple[np.ndarray, int]:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if len(faces) == 0:
        return image, 0
    projected_faces, _, valid_faces, mean_depth = _project_mesh(vertices, faces, K, camtoworld, near_plane)
    visible = _visible_face_mask(projected_faces, valid_faces & active.astype(bool), width=width, height=height)
    visible_ids = np.nonzero(visible)[0]
    if len(visible_ids) == 0:
        return image, 0
    order = visible_ids[np.argsort(mean_depth[visible_ids])[::-1]]
    for face_id in order:
        pts = np.round(projected_faces[face_id]).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], -2 * width, 3 * width)
        pts[:, 1] = np.clip(pts[:, 1], -2 * height, 3 * height)
        color = tuple(int(channel) for channel in face_colors[face_id])
        cv2.fillConvexPoly(image, pts, color=color, lineType=cv2.LINE_AA)
    return image, int(len(visible_ids))


def _coplane_panels(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    targets: object | None,
    width: int,
    height: int,
    near_plane: float,
    residual_vmax_m: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    if targets is None or len(faces) == 0:
        empty = np.zeros((height, width, 3), dtype=np.uint8)
        return empty, empty.copy(), {"visible_coplane_faces": 0.0, "active_coplane_faces": 0.0}

    weights = targets.weights.detach().cpu().numpy()
    group_ids = targets.group_ids.detach().cpu().numpy()
    normals = targets.normals.detach().cpu().numpy()
    offsets = targets.offsets.detach().cpu().numpy()
    active = weights > 0

    group_colors = np.zeros((len(faces), 3), dtype=np.uint8)
    group_colors[active] = _group_palette(group_ids[active])
    group_image, visible_groups = _render_face_colors(
        vertices=vertices,
        faces=faces,
        K=K,
        camtoworld=camtoworld,
        face_colors=group_colors,
        active=active,
        width=width,
        height=height,
        near_plane=near_plane,
    )

    face_vertices = vertices[faces]
    distances = np.abs(np.sum(face_vertices * normals[:, None, :], axis=-1) + offsets[:, None])
    face_error = np.mean(distances, axis=1)
    normalized = np.clip(face_error / max(float(residual_vmax_m), 1e-6), 0.0, 1.0)
    error_colors = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    error_colors = cv2.cvtColor(error_colors, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    error_colors[~active] = 0
    error_image, visible_errors = _render_face_colors(
        vertices=vertices,
        faces=faces,
        K=K,
        camtoworld=camtoworld,
        face_colors=error_colors,
        active=active,
        width=width,
        height=height,
        near_plane=near_plane,
    )
    stats = {
        "visible_coplane_faces": float(max(visible_groups, visible_errors)),
        "active_coplane_faces": float(np.count_nonzero(active)),
        "mean_face_plane_error_m": float(np.mean(face_error[active])) if np.any(active) else 0.0,
    }
    return group_image, error_image, stats


def render_regularization_preview(
    *,
    rgb: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    rendered_depth: Optional[np.ndarray],
    rendered_alpha: Optional[np.ndarray],
    depth_prior: Optional[np.ndarray],
    depth_valid: Optional[np.ndarray],
    p_planar: Optional[np.ndarray],
    p_detail: Optional[np.ndarray],
    p_ridge: Optional[np.ndarray],
    vertices: np.ndarray,
    faces: np.ndarray,
    coplane_targets: object | None,
    max_width: int,
    near_plane: float,
    depth_min_m: float,
    depth_max_m: float,
    depth_alpha_min: float,
    depth_planar_weight: float,
    depth_detail_weight: float,
    depth_ridge_weight: float,
    depth_distance_decay_m: float,
    residual_scale_m: float,
    coplane_residual_vmax_m: float,
) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    rgb_small, K_small, _ = resize_rgb_and_intrinsics_for_preview(np.asarray(rgb[..., :3], dtype=np.uint8), K, max_width=max_width)
    height, width = rgb_small.shape[:2]
    rendered_depth = _resize_map(rendered_depth, width, height)
    rendered_alpha = _resize_map(rendered_alpha, width, height)
    depth_prior = _resize_map(depth_prior, width, height)
    depth_valid = _resize_map(depth_valid, width, height, nearest=True)
    p_planar = _resize_map(p_planar, width, height)
    p_detail = _resize_map(p_detail, width, height)
    p_ridge = _resize_map(p_ridge, width, height)

    if depth_prior is None:
        depth_prior = np.zeros((height, width), dtype=np.float32)
    if rendered_depth is None:
        rendered_depth = np.zeros((height, width), dtype=np.float32)
    valid = np.ones((height, width), dtype=bool) if depth_valid is None else depth_valid.astype(bool)
    finite = np.isfinite(depth_prior) & np.isfinite(rendered_depth)
    valid &= finite & (depth_prior >= float(depth_min_m))
    if float(depth_max_m) > 0.0:
        valid &= depth_prior <= float(depth_max_m)
    if rendered_alpha is not None and float(depth_alpha_min) > 0.0:
        valid &= rendered_alpha >= float(depth_alpha_min)

    if p_planar is None:
        p_planar = np.zeros((height, width), dtype=np.float32)
    if p_detail is None:
        p_detail = np.zeros((height, width), dtype=np.float32)
    if p_ridge is None:
        p_ridge = np.zeros((height, width), dtype=np.float32)
    depth_weight = (
        float(depth_planar_weight) * np.clip(p_planar, 0.0, 1.0)
        + float(depth_detail_weight) * np.clip(p_detail, 0.0, 1.0)
        + float(depth_ridge_weight) * np.clip(p_ridge, 0.0, 1.0)
    )
    if float(depth_distance_decay_m) > 0.0:
        depth_weight *= np.exp(-np.maximum(depth_prior, 0.0) / float(depth_distance_decay_m))
    depth_weight = np.where(valid, depth_weight, 0.0).astype(np.float32)

    depth_values = np.concatenate([depth_prior[valid], rendered_depth[valid]]) if np.any(valid) else np.asarray([], dtype=np.float32)
    if depth_values.size > 0 and float(depth_max_m) <= 0.0:
        dmin, dmax = np.percentile(depth_values, [2.0, 98.0]).astype(np.float32)
    else:
        dmin = float(depth_min_m)
        dmax = float(depth_max_m) if float(depth_max_m) > 0.0 else float(np.nanmax(depth_values)) if depth_values.size else 1.0
    if dmax <= dmin:
        dmax = dmin + 1.0

    residual = rendered_depth - depth_prior
    weighted_abs_residual = np.where(valid, np.abs(residual) * depth_weight, 0.0).astype(np.float32)
    groups_panel, coplane_error_panel, coplane_stats = _coplane_panels(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        K=K_small,
        camtoworld=np.asarray(camtoworld, dtype=np.float64),
        targets=coplane_targets,
        width=width,
        height=height,
        near_plane=float(near_plane),
        residual_vmax_m=float(coplane_residual_vmax_m),
    )

    panels = {
        "rgb": rgb_small,
        "depth_prior": _depth_color(depth_prior, valid, dmin, dmax),
        "depth_render": _depth_color(rendered_depth, valid, dmin, dmax),
        "depth_residual": _signed_residual_color(residual, valid, float(residual_scale_m)),
        "depth_weighted_abs": _scalar_color(weighted_abs_residual, vmax=float(residual_scale_m), cmap=cv2.COLORMAP_MAGMA),
        "guide_category": _guide_category_color(p_planar, p_detail, p_ridge, valid, width, height),
        "coplane_groups": groups_panel,
        "coplane_error": coplane_error_panel,
    }
    stats = {
        "depth_used_pixels": float(np.count_nonzero(depth_weight > 0)),
        "depth_mean_weight": float(depth_weight[depth_weight > 0].mean()) if np.any(depth_weight > 0) else 0.0,
        "depth_mean_abs_error_m": float((np.abs(residual) * depth_weight).sum() / max(depth_weight.sum(), 1e-8)),
        **coplane_stats,
    }
    return panels, stats


def make_regularization_canvas(
    panels: Dict[str, np.ndarray],
    *,
    step: int,
    frame_index: int,
    phase: str,
    rotate_clockwise: bool,
) -> np.ndarray:
    step_label = f"{step} {phase}" if phase else str(step)
    prepared = [
        _prepare_panel(panels["rgb"], f"RGB frame {frame_index}", rotate_clockwise),
        _prepare_panel(panels["depth_prior"], "Depth prior", rotate_clockwise),
        _prepare_panel(panels["depth_render"], f"Render depth {step_label}", rotate_clockwise),
        _prepare_panel(panels["depth_residual"], "Depth residual", rotate_clockwise),
        _prepare_panel(panels["depth_weighted_abs"], "Weighted |depth error|", rotate_clockwise),
        _prepare_panel(panels["guide_category"], "Guide categories", rotate_clockwise),
        _prepare_panel(panels["coplane_groups"], "Coplane groups", rotate_clockwise),
        _prepare_panel(panels["coplane_error"], "Coplane plane error", rotate_clockwise),
    ]
    return np.concatenate(prepared, axis=1)


def write_regularization_preview(
    *,
    out_dir: Path,
    step: int,
    frame_index: int,
    phase: str,
    rotate_clockwise: bool,
    **kwargs,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panels, stats = render_regularization_preview(**kwargs)
    canvas = make_regularization_canvas(
        panels,
        step=step,
        frame_index=frame_index,
        phase=phase,
        rotate_clockwise=rotate_clockwise,
    )
    safe_phase = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in phase)
    out_path = out_dir / f"regularization_preview_frame_{frame_index:04d}_step_{step:06d}_{safe_phase}.png"
    imageio.imwrite(out_path, canvas)
    return {
        "path": str(out_path),
        "step": int(step),
        "phase": str(phase),
        "frame_index": int(frame_index),
        **stats,
    }


def update_regularization_contact_sheet(
    *,
    preview_dir: Path,
    out_path: Optional[Path] = None,
    max_images: int = 24,
) -> Optional[Path]:
    files = sorted(preview_dir.glob("regularization_preview_frame_*_step_*.png"))
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
            image = cv2.resize(image, (int(round(image.shape[1] * scale)), min_height), interpolation=cv2.INTER_AREA)
        resized.append(image)
    sheet_width = max(image.shape[1] for image in resized)
    padded = [
        image
        if image.shape[1] == sheet_width
        else np.pad(image, ((0, 0), (0, sheet_width - image.shape[1]), (0, 0)), mode="constant")
        for image in resized
    ]
    sheet = np.concatenate(padded, axis=0)
    if out_path is None:
        out_path = preview_dir / "regularization_preview_contact_sheet.png"
    imageio.imwrite(out_path, sheet)
    return out_path
