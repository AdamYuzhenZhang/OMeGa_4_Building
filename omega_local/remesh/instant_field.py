"""View-normal orientation-field export for guided Instant Meshes.

This stage converts saved per-view StableNormal evidence into soft per-vertex
orientation constraints for Instant Meshes. The OMeGa mesh remains the 3D
surface anchor; normal maps only contribute tangent-flow evidence.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from omega_local.remesh.evidence import resolve_latest_omega_mesh
from omega_local.remesh.local_ops import face_geometry, load_mesh_arrays
from omega_local.remesh.transfer import summarize


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class OmegaInstantFieldConfig:
    model_dir: Path
    mesh_path: Path | None = None
    evidence_npz: Path | None = None
    frame_manifest: Path | None = None
    output_dir: Path | None = None
    field_name: str = "omega_instant_field"
    iteration: int = -1
    frame_stride: int = 1
    max_frames: int = 0
    pixel_stride: int = 2
    min_direction_confidence: float = 0.15
    min_gradient_normalized: float = 0.35
    gradient_mode: str = "point"
    weight_quantile: float = 0.90
    min_vertex_weight: float = 0.02
    overwrite: bool = False


@dataclass(frozen=True)
class OmegaInstantFieldResult:
    field_txt: Path
    field_npz: Path
    summary_json: Path
    vertex_count: int
    constrained_vertex_count: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_existing(path: Path | None, fallback: Path, label: str) -> Path:
    resolved = path.expanduser().resolve() if path is not None else fallback.expanduser().resolve()
    if not resolved.exists() or resolved.is_dir():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _resolve_frame_buffer(path_raw: str, model_dir: Path) -> Path:
    path = Path(str(path_raw)).expanduser()
    candidates = [path.resolve()] if path.is_absolute() else [
        path.resolve(),
        (model_dir / path).resolve(),
        (model_dir.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing view-buffer npz. Tried: {[str(v) for v in candidates]}")


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors, dtype=np.float64))


def _tangent_basis_from_normals(normals: np.ndarray) -> np.ndarray:
    normals = _safe_normalize(normals)
    axis_ids = np.argmin(np.abs(normals), axis=1)
    reference = np.zeros_like(normals)
    reference[np.arange(normals.shape[0]), axis_ids] = 1.0
    basis_u = _safe_normalize(np.cross(reference, normals))
    basis_v = _safe_normalize(np.cross(normals, basis_u))
    return np.stack([basis_u, basis_v], axis=2)


def _principal_image_tangent(tensor: np.ndarray) -> np.ndarray:
    """Return image-space line direction perpendicular to strongest normal change."""

    tensor = np.asarray(tensor, dtype=np.float64)
    j11 = tensor[:, 0]
    j12 = tensor[:, 1]
    j22 = tensor[:, 2]
    theta = 0.5 * np.arctan2(2.0 * j12, j11 - j22)
    gx = np.cos(theta)
    gy = np.sin(theta)
    return np.stack([-gy, gx], axis=1).astype(np.float64)


def _pullback_image_tangent_to_world(
    *,
    tangents_img: np.ndarray,
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    camtoworld: np.ndarray,
    normals_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift image-space tangent lines to mesh tangents with the projection Jacobian.

    Returns the lifted tangent, a confidence term in [0, 1], and the screen-space
    least-squares residual. The confidence suppresses grazing or ill-conditioned
    observations instead of letting every valid 2D line become a hard 3D guide.
    """

    tangents_img = _safe_normalize(np.asarray(tangents_img, dtype=np.float64))
    pixel_x = np.asarray(pixel_x, dtype=np.float64)
    pixel_y = np.asarray(pixel_y, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    camtoworld = np.asarray(camtoworld, dtype=np.float64)
    normals_world = _safe_normalize(normals_world)

    fx = max(float(K[0, 0]), 1e-12)
    fy = max(float(K[1, 1]), 1e-12)
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    z = np.maximum(depth, 1e-12)
    x_cam = (pixel_x - cx) * z / fx
    y_cam = (pixel_y - cy) * z / fy

    jac = np.zeros((tangents_img.shape[0], 2, 3), dtype=np.float64)
    jac[:, 0, 0] = fx / z
    jac[:, 0, 2] = -fx * x_cam / (z * z)
    jac[:, 1, 1] = fy / z
    jac[:, 1, 2] = -fy * y_cam / (z * z)

    world_to_cam = camtoworld[:3, :3].T
    basis_world = _tangent_basis_from_normals(normals_world)
    basis_cam = np.einsum("ij,njk->nik", world_to_cam, basis_world)
    projection_tangent = np.einsum("nij,njk->nik", jac, basis_cam)

    ata = np.einsum("nji,njk->nik", projection_tangent, projection_tangent)
    atb = np.einsum("nji,nj->ni", projection_tangent, tangents_img)
    a00 = ata[:, 0, 0]
    a01 = ata[:, 0, 1]
    a11 = ata[:, 1, 1]
    b0 = atb[:, 0]
    b1 = atb[:, 1]
    det = a00 * a11 - a01 * a01
    good = np.isfinite(det) & (np.abs(det) > 1e-18) & np.isfinite(depth) & (depth > 0.0)

    coeff = np.zeros((tangents_img.shape[0], 2), dtype=np.float64)
    coeff[good, 0] = (a11[good] * b0[good] - a01[good] * b1[good]) / det[good]
    coeff[good, 1] = (-a01[good] * b0[good] + a00[good] * b1[good]) / det[good]
    tangent_world = basis_world[:, :, 0] * coeff[:, 0, None] + basis_world[:, :, 1] * coeff[:, 1, None]

    predicted = np.einsum("nij,nj->ni", projection_tangent, coeff)
    residual = np.linalg.norm(predicted - tangents_img, axis=1)
    residual[~good] = 1.0
    residual_quality = np.clip(1.0 - residual, 0.0, 1.0)

    eig_disc = np.sqrt(np.maximum((a00 - a11) * (a00 - a11) + 4.0 * a01 * a01, 0.0))
    eig_min = np.maximum(0.5 * (a00 + a11 - eig_disc), 0.0)
    eig_max = np.maximum(0.5 * (a00 + a11 + eig_disc), 1e-24)
    condition_quality = np.sqrt(np.clip(eig_min / eig_max, 0.0, 1.0))

    points_cam = np.stack([x_cam, y_cam, z], axis=1)
    rays_cam = _safe_normalize(points_cam)
    normals_cam = np.einsum("ij,nj->ni", world_to_cam, normals_world)
    view_cos = np.abs(np.sum(_safe_normalize(normals_cam) * rays_cam, axis=1))
    view_quality = np.sqrt(np.clip((view_cos - 0.12) / 0.88, 0.0, 1.0))
    lift_quality = residual_quality * condition_quality * view_quality
    lift_quality[~good] = 0.0

    if not np.all(good):
        # Fallback for grazing or numerically singular views. This matches the
        # old small-angle camera-axis lift but is only used when the projection
        # Jacobian cannot identify a stable surface tangent. Its confidence is
        # zero, so it will not add guide weight; keeping a vector here only
        # avoids NaNs in diagnostics.
        camera_to_world = camtoworld[:3, :3]
        axis_u = camera_to_world[:, 0] / fx
        axis_v = camera_to_world[:, 1] / fy
        fallback = tangents_img[:, 0, None] * axis_u[None, :] + tangents_img[:, 1, None] * axis_v[None, :]
        fallback = fallback - np.sum(fallback * normals_world, axis=1, keepdims=True) * normals_world
        tangent_world[~good] = fallback[~good]

    tangent_world = tangent_world - np.sum(tangent_world * normals_world, axis=1, keepdims=True) * normals_world
    return _safe_normalize(tangent_world), lift_quality.astype(np.float32), residual.astype(np.float32)


def _tensor6_from_vectors(vectors: np.ndarray) -> np.ndarray:
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    return np.stack([x * x, x * y, x * z, y * y, y * z, z * z], axis=1)


def _matrix_from_tensor6(tensor6: np.ndarray) -> np.ndarray:
    tensor6 = np.asarray(tensor6, dtype=np.float64)
    out = np.zeros((tensor6.shape[0], 3, 3), dtype=np.float64)
    out[:, 0, 0] = tensor6[:, 0]
    out[:, 0, 1] = out[:, 1, 0] = tensor6[:, 1]
    out[:, 0, 2] = out[:, 2, 0] = tensor6[:, 2]
    out[:, 1, 1] = tensor6[:, 3]
    out[:, 1, 2] = out[:, 2, 1] = tensor6[:, 4]
    out[:, 2, 2] = tensor6[:, 5]
    return out


def _dominant_tensor_direction(tensor6: np.ndarray, normals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = _matrix_from_tensor6(tensor6)
    evals, evecs = np.linalg.eigh(matrices)
    direction = evecs[:, :, 2]
    normals = _safe_normalize(normals)
    direction = direction - np.sum(direction * normals, axis=1, keepdims=True) * normals
    direction = _safe_normalize(direction)
    trace = np.maximum(np.sum(evals, axis=1), 0.0)
    line_conf = np.divide(
        np.maximum(evals[:, 2] - evals[:, 1], 0.0),
        np.maximum(evals[:, 2] + evals[:, 1], 1e-12),
        out=np.zeros((evals.shape[0],), dtype=np.float64),
    )
    valid = np.linalg.norm(direction, axis=1) > 0.0
    line_conf[~valid] = 0.0
    return direction.astype(np.float32), trace.astype(np.float32), line_conf.astype(np.float32)


def _normalize_trace_weight(trace: np.ndarray, quantile: float) -> np.ndarray:
    trace = np.asarray(trace, dtype=np.float64)
    valid = trace[np.isfinite(trace) & (trace > 0.0)]
    if valid.size == 0:
        return np.zeros(trace.shape, dtype=np.float32)
    scale = float(np.quantile(valid, float(np.clip(quantile, 0.1, 0.99))))
    scale = max(scale, 1e-12)
    return np.clip(trace / scale, 0.0, 1.0).astype(np.float32)


def _buffer_direction_inputs(buffer: Any, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    requested = str(mode).strip().lower()
    if requested not in {"point", "local", "combined"}:
        raise ValueError(f"Unsupported Instant field gradient mode: {mode}")

    point_tensor = np.asarray(buffer["normal_structure_tensor"], dtype=np.float32)
    point_conf = np.asarray(buffer["normal_direction_confidence"], dtype=np.float32)
    point_grad = np.asarray(buffer["normal_gradient_normalized"], dtype=np.float32)
    has_local = (
        "normal_local_structure_tensor" in buffer
        and "normal_local_direction_confidence" in buffer
        and "normal_local_gradient_normalized" in buffer
    )
    if requested == "point" or not has_local:
        return point_tensor, point_conf, point_grad, "point"

    local_tensor = np.asarray(buffer["normal_local_structure_tensor"], dtype=np.float32)
    local_conf = np.asarray(buffer["normal_local_direction_confidence"], dtype=np.float32)
    local_grad = np.asarray(buffer["normal_local_gradient_normalized"], dtype=np.float32)
    if requested == "local":
        return local_tensor, local_conf, local_grad, "local"

    point_score = point_conf * np.sqrt(np.maximum(point_grad, 0.0))
    local_score = local_conf * np.sqrt(np.maximum(local_grad, 0.0))
    use_local = local_score > point_score
    tensor = np.where(use_local[..., None], local_tensor, point_tensor).astype(np.float32)
    conf = np.where(use_local, local_conf, point_conf).astype(np.float32)
    grad = np.where(use_local, local_grad, point_grad).astype(np.float32)
    return tensor, conf, grad, "combined"


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray, face_normals: np.ndarray, face_areas: np.ndarray) -> np.ndarray:
    normals = np.zeros((vertices.shape[0], 3), dtype=np.float64)
    weighted = np.asarray(face_normals, dtype=np.float64) * np.asarray(face_areas, dtype=np.float64)[:, None]
    for local in range(3):
        np.add.at(normals, faces[:, local], weighted)
    return _safe_normalize(normals).astype(np.float32)


def _write_field_txt(
    path: Path,
    *,
    vertex_direction: np.ndarray,
    vertex_weight: np.ndarray,
    vertex_normal: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("omega_instant_field_v1\n")
        handle.write(f"vertex_count {int(vertex_direction.shape[0])}\n")
        handle.write("columns index qx qy qz qw nx ny nz nw\n")
        for idx, (direction, weight, normal) in enumerate(zip(vertex_direction, vertex_weight, vertex_normal)):
            handle.write(
                "{} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g} {:.9g}\n".format(
                    idx,
                    float(direction[0]),
                    float(direction[1]),
                    float(direction[2]),
                    float(weight),
                    float(normal[0]),
                    float(normal[1]),
                    float(normal[2]),
                    0.0,
                )
            )


def compute_omega_instant_field(
    config: OmegaInstantFieldConfig,
    progress: ProgressFn | None = None,
) -> OmegaInstantFieldResult:
    progress = progress or _progress_default
    model_dir = config.model_dir.expanduser().resolve()
    mesh_path = (
        config.mesh_path.expanduser().resolve()
        if config.mesh_path is not None
        else resolve_latest_omega_mesh(model_dir, iteration=int(config.iteration))
    )
    evidence_npz = _resolve_existing(config.evidence_npz, model_dir / "remesh" / "local" / "evidence.npz", "evidence npz")
    frame_manifest = _resolve_existing(
        config.frame_manifest,
        model_dir / "remesh" / "local" / "evidence_frames.jsonl",
        "evidence frame manifest",
    )
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else model_dir / "remesh" / "instant_field"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    field_txt = output_dir / f"{config.field_name}.txt"
    field_npz = output_dir / f"{config.field_name}.npz"
    summary_json = output_dir / f"{config.field_name}_summary.json"
    for path in (field_txt, field_npz, summary_json):
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"Output exists: {path}. Re-run with --overwrite.")

    vertices, faces = load_mesh_arrays(mesh_path)
    _face_centers, face_normals, face_areas, _mean_edges = face_geometry(vertices, faces)
    vertex_normal = _vertex_normals(vertices, faces, face_normals, face_areas)
    evidence = np.load(evidence_npz)
    if int(evidence["face_area"].shape[0]) != int(faces.shape[0]):
        raise ValueError("Evidence face count does not match the field-export mesh.")

    rows = _read_jsonl(frame_manifest)[:: max(int(config.frame_stride), 1)]
    if int(config.max_frames) > 0:
        rows = rows[: int(config.max_frames)]
    if not rows:
        raise ValueError("No evidence frames selected for Instant field export.")

    face_tensor = np.zeros((faces.shape[0], 6), dtype=np.float64)
    face_sample_weight = np.zeros((faces.shape[0],), dtype=np.float64)
    face_sample_count = np.zeros((faces.shape[0],), dtype=np.int64)
    face_lift_quality_sum = np.zeros((faces.shape[0],), dtype=np.float64)
    face_lift_quality_count = np.zeros((faces.shape[0],), dtype=np.int64)

    progress(f"[instant-field] Mesh: {mesh_path} ({vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces)")
    progress(f"[instant-field] Evidence: {evidence_npz}")
    progress(f"[instant-field] Frames: {len(rows):,}")
    progress(f"[instant-field] Pixel stride: {max(int(config.pixel_stride), 1)}")
    progress(f"[instant-field] Gradient mode: {str(config.gradient_mode).strip().lower()}")

    pixel_stride = max(int(config.pixel_stride), 1)
    resolved_gradient_modes: dict[str, int] = {}
    for ordinal, row in enumerate(rows, start=1):
        buffer_path = _resolve_frame_buffer(str(row.get("viewBufferPath", "")), model_dir)
        buffer = np.load(buffer_path)
        face_id = np.asarray(buffer["face_id"], dtype=np.int64)
        tensor_img, direction_conf, gradient_norm, resolved_mode = _buffer_direction_inputs(buffer, str(config.gradient_mode))
        resolved_gradient_modes[resolved_mode] = resolved_gradient_modes.get(resolved_mode, 0) + 1
        normal_valid = np.asarray(buffer["normal_valid"], dtype=np.uint8) > 0
        sample = np.zeros(face_id.shape, dtype=bool)
        sample[::pixel_stride, ::pixel_stride] = True
        valid = (
            sample
            & normal_valid
            & (face_id >= 0)
            & (direction_conf >= float(config.min_direction_confidence))
            & (gradient_norm >= float(config.min_gradient_normalized))
        )
        if not np.any(valid):
            progress(f"[instant-field {ordinal:04d}/{len(rows):04d}] {row.get('imageName', '')} no valid direction pixels")
            continue

        pixel_y, pixel_x = np.nonzero(valid)
        fids = face_id[valid]
        tangents_img = _principal_image_tangent(tensor_img[valid])
        camtoworld = np.asarray(buffer["camtoworld"], dtype=np.float64)
        K = np.asarray(buffer["K"], dtype=np.float64)
        normals = face_normals[fids]
        tangent_world, lift_quality, _lift_residual = _pullback_image_tangent_to_world(
            tangents_img=tangents_img,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            depth=np.asarray(buffer["depth"], dtype=np.float32)[valid],
            K=K,
            camtoworld=camtoworld,
            normals_world=normals,
        )
        nonzero = (np.linalg.norm(tangent_world, axis=1) > 0.0) & (lift_quality > 0.0)
        if not np.any(nonzero):
            progress(f"[instant-field {ordinal:04d}/{len(rows):04d}] {row.get('imageName', '')} degenerate tangents")
            continue

        fids = fids[nonzero]
        tangent_world = tangent_world[nonzero]
        lift_quality = lift_quality[nonzero].astype(np.float64)
        grad_gate = np.clip(
            (gradient_norm[valid][nonzero].astype(np.float64) - float(config.min_gradient_normalized))
            / max(2.0 - float(config.min_gradient_normalized), 1e-6),
            0.0,
            1.0,
        )
        weights = direction_conf[valid][nonzero].astype(np.float64) * np.sqrt(grad_gate) * lift_quality
        keep = weights > 1e-8
        if not np.any(keep):
            progress(f"[instant-field {ordinal:04d}/{len(rows):04d}] {row.get('imageName', '')} zero weights")
            continue

        fids = fids[keep]
        weights = weights[keep]
        lift_quality = lift_quality[keep]
        tensor6 = _tensor6_from_vectors(tangent_world[keep])
        for channel in range(6):
            face_tensor[:, channel] += np.bincount(
                fids,
                weights=weights * tensor6[:, channel],
                minlength=faces.shape[0],
            )
        face_sample_weight += np.bincount(fids, weights=weights, minlength=faces.shape[0])
        face_sample_count += np.bincount(fids, minlength=faces.shape[0]).astype(np.int64)
        face_lift_quality_sum += np.bincount(fids, weights=lift_quality, minlength=faces.shape[0])
        face_lift_quality_count += np.bincount(fids, minlength=faces.shape[0]).astype(np.int64)
        progress(
            f"[instant-field {ordinal:04d}/{len(rows):04d}] {row.get('imageName', '')} "
            f"pixels={int(fids.size):,} faces={int(np.unique(fids).size):,}"
        )

    face_direction, face_trace, face_line_conf = _dominant_tensor_direction(face_tensor, face_normals)
    trace_weight = _normalize_trace_weight(face_trace, float(config.weight_quantile))
    face_support = np.asarray(evidence["face_support"], dtype=np.float32)
    face_direction_conf = np.asarray(evidence["face_direction_confidence"], dtype=np.float32)
    face_weight = (
        trace_weight
        * np.asarray(face_line_conf, dtype=np.float32)
        * np.sqrt(np.clip(face_support, 0.0, 1.0))
        * (0.25 + 0.75 * np.clip(face_direction_conf, 0.0, 1.0))
    ).astype(np.float32)

    vertex_tensor = np.zeros((vertices.shape[0], 6), dtype=np.float64)
    vertex_weight_max = np.zeros((vertices.shape[0],), dtype=np.float32)
    weighted_face_tensor = face_tensor * face_weight[:, None].astype(np.float64)
    for local in range(3):
        np.add.at(vertex_tensor, faces[:, local], weighted_face_tensor)
        np.maximum.at(vertex_weight_max, faces[:, local], face_weight)

    vertex_direction, vertex_trace, vertex_line_conf = _dominant_tensor_direction(vertex_tensor, vertex_normal)
    vertex_trace_weight = _normalize_trace_weight(vertex_trace, float(config.weight_quantile))
    vertex_weight = (vertex_weight_max * vertex_trace_weight * vertex_line_conf).astype(np.float32)
    vertex_weight = np.clip(vertex_weight, 0.0, 1.0)
    vertex_weight[vertex_weight < float(config.min_vertex_weight)] = 0.0
    vertex_direction[vertex_weight <= 0.0] = 0.0
    face_lift_quality = np.divide(
        face_lift_quality_sum,
        np.maximum(face_lift_quality_count, 1),
        out=np.zeros_like(face_lift_quality_sum),
        where=face_lift_quality_count > 0,
    ).astype(np.float32)

    constrained_count = int(np.count_nonzero(vertex_weight > 0.0))
    _write_field_txt(
        field_txt,
        vertex_direction=vertex_direction,
        vertex_weight=vertex_weight,
        vertex_normal=vertex_normal,
    )
    np.savez_compressed(
        field_npz,
        vertex_direction=vertex_direction.astype(np.float32),
        vertex_direction_weight=vertex_weight.astype(np.float32),
        vertex_normal=vertex_normal.astype(np.float32),
        face_direction=face_direction.astype(np.float32),
        face_direction_weight=face_weight.astype(np.float32),
        face_direction_trace=face_trace.astype(np.float32),
        face_direction_line_confidence=face_line_conf.astype(np.float32),
        face_sample_weight=face_sample_weight.astype(np.float32),
        face_sample_count=face_sample_count.astype(np.int64),
        face_lift_quality=face_lift_quality.astype(np.float32),
        mesh_vertices=vertices.astype(np.float32),
        mesh_faces=faces.astype(np.int64),
    )

    summary = {
        "stageName": "omega_instant_field_export",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_dir": str(model_dir),
            "mesh_path": str(mesh_path),
            "evidence_npz": str(evidence_npz),
            "frame_manifest": str(frame_manifest),
            "output_dir": str(output_dir),
            "fieldName": str(config.field_name),
            "iteration": int(config.iteration),
            "frameStride": int(config.frame_stride),
            "maxFrames": int(config.max_frames),
            "pixelStride": int(config.pixel_stride),
            "minDirectionConfidence": float(config.min_direction_confidence),
            "minGradientNormalized": float(config.min_gradient_normalized),
            "gradientMode": str(config.gradient_mode).strip().lower(),
            "resolvedGradientModes": resolved_gradient_modes,
            "weightQuantile": float(config.weight_quantile),
            "minVertexWeight": float(config.min_vertex_weight),
            "overwrite": bool(config.overwrite),
        },
        "mesh": {
            "vertices": int(vertices.shape[0]),
            "faces": int(faces.shape[0]),
            "surfaceArea": float(np.sum(face_areas)),
        },
        "frames": {
            "selectedCount": int(len(rows)),
            "frameStride": int(config.frame_stride),
            "maxFrames": int(config.max_frames),
            "pixelStride": int(pixel_stride),
        },
        "stats": {
            "faceSampleCount": summarize(face_sample_count),
            "faceSampleWeight": summarize(face_sample_weight),
            "faceLiftQuality": summarize(face_lift_quality),
            "faceDirectionWeight": summarize(face_weight),
            "vertexDirectionWeight": summarize(vertex_weight),
        },
        "coverage": {
            "constrainedVertexCount": int(constrained_count),
            "constrainedVertexFraction": float(constrained_count / max(int(vertices.shape[0]), 1)),
            "constrainedFaceCount": int(np.count_nonzero(face_weight > 0.0)),
            "constrainedFaceFraction": float(np.count_nonzero(face_weight > 0.0) / max(int(faces.shape[0]), 1)),
        },
        "outputs": {
            "fieldTxt": str(field_txt),
            "fieldNpz": str(field_npz),
            "summaryJson": str(summary_json),
        },
        "notes": [
            "This stage exports soft tangent-flow constraints only; it does not move vertices.",
            "Per-view normal-map directions are lifted to the mesh tangent plane and accumulated as unoriented tensors.",
            "The lift uses projection-Jacobian conditioning and view angle to suppress unstable 2D observations.",
            "Low confidence or cross-view unstable evidence becomes low CQw instead of a hard 3D constraint.",
        ],
    }
    _write_json(summary_json, summary)
    progress(f"[instant-field] Wrote {field_txt}")
    progress(f"[instant-field] Wrote {field_npz}")
    progress(f"[instant-field] Constrained vertices: {constrained_count:,} / {vertices.shape[0]:,}")
    return OmegaInstantFieldResult(
        field_txt=field_txt,
        field_npz=field_npz,
        summary_json=summary_json,
        vertex_count=int(vertices.shape[0]),
        constrained_vertex_count=constrained_count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export OMeGa view-normal tangent-flow constraints for guided Instant Meshes.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/preclean_mesh.ply, then latest OMeGa mesh.")
    parser.add_argument("--evidence-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/evidence.npz.")
    parser.add_argument("--frame-manifest", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/evidence_frames.jsonl.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/instant_field.")
    parser.add_argument("--field-name", default="omega_instant_field")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every selected evidence frame.")
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--min-direction-confidence", type=float, default=0.15)
    parser.add_argument("--min-gradient-normalized", type=float, default=0.35)
    parser.add_argument("--gradient-mode", choices=("point", "local", "combined"), default="point")
    parser.add_argument("--weight-quantile", type=float, default=0.90)
    parser.add_argument("--min-vertex-weight", type=float, default=0.02)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compute_omega_instant_field(
        OmegaInstantFieldConfig(
            model_dir=args.model_dir,
            mesh_path=args.mesh,
            evidence_npz=args.evidence_npz,
            frame_manifest=args.frame_manifest,
            output_dir=args.output_dir,
            field_name=str(args.field_name),
            iteration=int(args.iteration),
            frame_stride=int(args.frame_stride),
            max_frames=int(args.max_frames),
            pixel_stride=int(args.pixel_stride),
            min_direction_confidence=float(args.min_direction_confidence),
            min_gradient_normalized=float(args.min_gradient_normalized),
            gradient_mode=str(args.gradient_mode),
            weight_quantile=float(args.weight_quantile),
            min_vertex_weight=float(args.min_vertex_weight),
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
