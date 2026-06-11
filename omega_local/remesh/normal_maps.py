"""Per-view normal-map geometry for OMeGa local remeshing.

StableNormal/PromptDA normals are treated as image-space evidence.  This module
computes normal derivatives, robust per-frame noise tolerances, target lengths
in pixels, and a 2D structure tensor before anything is transferred to the
mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class NormalGeometryConfig:
    target_edge_min_px: float = 2.0
    target_edge_max_px: float = 96.0
    eps: float = 1e-6
    high_gradient_quantile: float = 0.85
    local_variance_radius_px: int = 2


@dataclass(frozen=True)
class NormalGeometry:
    normal_valid: np.ndarray
    normal_du: np.ndarray
    normal_dv: np.ndarray
    gradient_g: np.ndarray
    gradient_normalized: np.ndarray
    noise_sigma_g: float
    tolerance_tau_g: float
    tolerance_tau_n: float
    tolerance_tau_a: float
    target_edge_length_px: np.ndarray
    structure_tensor: np.ndarray
    direction_confidence: np.ndarray
    local_normal_variance: np.ndarray
    local_gradient_g: np.ndarray
    local_gradient_normalized: np.ndarray
    local_structure_tensor: np.ndarray
    local_direction_confidence: np.ndarray
    local_high_gradient_mask: np.ndarray
    high_gradient_mask: np.ndarray
    high_gradient_threshold: float


def safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors, dtype=np.float64))


def _quantile(values: np.ndarray, q: float, fallback: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(fallback)
    return float(np.quantile(values, float(q)))


def _mad_sigma(values: np.ndarray, eps: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(eps)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, float(eps))


def _finite_differences(normal: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(normal, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    du = np.zeros_like(normal, dtype=np.float32)
    dv = np.zeros_like(normal, dtype=np.float32)
    du_valid = np.zeros(valid.shape, dtype=bool)
    dv_valid = np.zeros(valid.shape, dtype=bool)

    if normal.shape[1] > 1:
        du[:, 1:-1] = 0.5 * (normal[:, 2:] - normal[:, :-2])
        du[:, 0] = normal[:, 1] - normal[:, 0]
        du[:, -1] = normal[:, -1] - normal[:, -2]
        du_valid[:, 1:-1] = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
        du_valid[:, 0] = valid[:, 1] & valid[:, 0]
        du_valid[:, -1] = valid[:, -1] & valid[:, -2]
    if normal.shape[0] > 1:
        dv[1:-1, :] = 0.5 * (normal[2:, :] - normal[:-2, :])
        dv[0, :] = normal[1, :] - normal[0, :]
        dv[-1, :] = normal[-1, :] - normal[-2, :]
        dv_valid[1:-1, :] = valid[2:, :] & valid[1:-1, :] & valid[:-2, :]
        dv_valid[0, :] = valid[1, :] & valid[0, :]
        dv_valid[-1, :] = valid[-1, :] & valid[-2, :]

    du[~du_valid] = 0.0
    dv[~dv_valid] = 0.0
    return du, dv, du_valid, dv_valid


def _neighbor_normal_deltas(
    normal: np.ndarray,
    valid: np.ndarray,
    low_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(valid, dtype=bool) & np.asarray(low_gradient, dtype=bool)
    deltas: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    if normal.shape[1] > 1:
        pair_valid = valid[:, 1:] & valid[:, :-1]
        if np.any(pair_valid):
            diff = normal[:, 1:] - normal[:, :-1]
            delta = np.linalg.norm(diff, axis=2)[pair_valid]
            dot = np.sum(normal[:, 1:] * normal[:, :-1], axis=2)
            angle = np.arccos(np.clip(dot[pair_valid], -1.0, 1.0))
            deltas.append(delta)
            angles.append(angle)
    if normal.shape[0] > 1:
        pair_valid = valid[1:, :] & valid[:-1, :]
        if np.any(pair_valid):
            diff = normal[1:, :] - normal[:-1, :]
            delta = np.linalg.norm(diff, axis=2)[pair_valid]
            dot = np.sum(normal[1:, :] * normal[:-1, :], axis=2)
            angle = np.arccos(np.clip(dot[pair_valid], -1.0, 1.0))
            deltas.append(delta)
            angles.append(angle)
    if not deltas:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(deltas).astype(np.float32), np.concatenate(angles).astype(np.float32)


def _structure_confidence(j11: np.ndarray, j12: np.ndarray, j22: np.ndarray, eps: float) -> np.ndarray:
    trace = j11 + j22
    det_term = np.sqrt(np.maximum((j11 - j22) * (j11 - j22) + 4.0 * j12 * j12, 0.0))
    lambda_max = 0.5 * (trace + det_term)
    lambda_min = 0.5 * (trace - det_term)
    return np.divide(
        lambda_max - lambda_min,
        np.maximum(lambda_max + lambda_min, float(eps)),
        out=np.zeros_like(lambda_max, dtype=np.float32),
    ).astype(np.float32)


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    if int(radius) <= 0:
        return np.asarray(values, dtype=np.float32)
    ksize = 2 * int(radius) + 1
    return cv2.boxFilter(
        np.asarray(values, dtype=np.float32),
        ddepth=-1,
        ksize=(ksize, ksize),
        normalize=False,
        borderType=cv2.BORDER_REFLECT_101,
    )


def _local_normal_variance(normal: np.ndarray, valid: np.ndarray, radius: int, eps: float) -> tuple[np.ndarray, np.ndarray]:
    valid_f = np.asarray(valid, dtype=np.float32)
    count = _box_sum(valid_f, radius)
    normal_sum = np.stack(
        [_box_sum(normal[..., channel] * valid_f, radius) for channel in range(3)],
        axis=2,
    )
    mean_normal = np.divide(
        normal_sum,
        np.maximum(count[..., None], float(eps)),
        out=np.zeros_like(normal_sum, dtype=np.float32),
    )
    variance = 1.0 - np.sum(mean_normal * mean_normal, axis=2)
    variance = np.clip(variance, 0.0, 1.0).astype(np.float32)
    variance[count <= 0.0] = 0.0
    return variance, count


def _local_structure_tensor(
    j11: np.ndarray,
    j12: np.ndarray,
    j22: np.ndarray,
    valid: np.ndarray,
    radius: int,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_f = np.asarray(valid, dtype=np.float32)
    count = _box_sum(valid_f, radius)
    local_j11 = np.divide(_box_sum(j11 * valid_f, radius), np.maximum(count, float(eps)), out=np.zeros_like(j11, dtype=np.float32))
    local_j12 = np.divide(_box_sum(j12 * valid_f, radius), np.maximum(count, float(eps)), out=np.zeros_like(j12, dtype=np.float32))
    local_j22 = np.divide(_box_sum(j22 * valid_f, radius), np.maximum(count, float(eps)), out=np.zeros_like(j22, dtype=np.float32))
    local_tensor = np.stack([local_j11, local_j12, local_j22], axis=2).astype(np.float32)
    local_tensor[count <= 0.0] = 0.0
    return local_tensor, count


def compute_normal_geometry(
    normal_map_cam: np.ndarray | None,
    normal_valid: np.ndarray | None,
    config: NormalGeometryConfig | None = None,
) -> NormalGeometry:
    """Compute Stage B normal geometry from one camera-frame normal map."""

    cfg = config or NormalGeometryConfig()
    eps = max(float(cfg.eps), 1e-12)
    if normal_map_cam is None:
        raise ValueError("Stage 2 evidence extraction requires normal maps.")

    normal = safe_normalize(np.asarray(normal_map_cam, dtype=np.float64)).astype(np.float32)
    if normal_valid is None:
        valid = np.isfinite(normal).all(axis=2) & (np.linalg.norm(normal, axis=2) > 0.25)
    else:
        valid = np.asarray(normal_valid, dtype=bool) & np.isfinite(normal).all(axis=2)

    du, dv, du_valid, dv_valid = _finite_differences(normal, valid)
    derivative_valid = valid & du_valid & dv_valid
    gradient_g = np.sqrt(np.sum(du * du + dv * dv, axis=2)).astype(np.float32)
    gradient_g[~derivative_valid] = 0.0

    valid_g = gradient_g[derivative_valid]
    q20 = _quantile(valid_g, 0.20, eps)
    low_gradient = derivative_valid & (gradient_g <= q20)
    low_values = gradient_g[low_gradient]
    sigma_g = _mad_sigma(low_values, eps)
    tau_g = max(_quantile(low_values, 0.95, eps), eps)

    delta_n, delta_a = _neighbor_normal_deltas(normal, valid, low_gradient)
    tau_n = max(_quantile(delta_n, 0.95, eps), eps)
    tau_a = max(_quantile(delta_a, 0.95, eps), eps)

    denominator = np.maximum(gradient_g, sigma_g)
    target_px = np.divide(tau_n, denominator, out=np.full_like(gradient_g, float(cfg.target_edge_max_px)))
    target_px = np.clip(target_px, float(cfg.target_edge_min_px), float(cfg.target_edge_max_px)).astype(np.float32)
    target_px[~derivative_valid] = 0.0

    gradient_normalized = np.divide(gradient_g, tau_g, out=np.zeros_like(gradient_g, dtype=np.float32))
    gradient_normalized[~derivative_valid] = 0.0
    j11 = np.sum(du * du, axis=2).astype(np.float32)
    j12 = np.sum(du * dv, axis=2).astype(np.float32)
    j22 = np.sum(dv * dv, axis=2).astype(np.float32)
    structure_tensor = np.stack([j11, j12, j22], axis=2).astype(np.float32)
    direction_confidence = _structure_confidence(j11, j12, j22, eps)
    direction_confidence[~derivative_valid] = 0.0

    local_radius = max(int(cfg.local_variance_radius_px), 0)
    local_variance, local_count = _local_normal_variance(normal, derivative_valid, local_radius, eps)
    local_tensor, _local_tensor_count = _local_structure_tensor(j11, j12, j22, derivative_valid, local_radius, eps)
    local_j11 = local_tensor[..., 0]
    local_j12 = local_tensor[..., 1]
    local_j22 = local_tensor[..., 2]
    local_gradient_g = np.sqrt(np.maximum(local_j11 + local_j22, 0.0)).astype(np.float32)
    local_gradient_g[local_count <= 0.0] = 0.0
    local_gradient_normalized = np.divide(local_gradient_g, tau_g, out=np.zeros_like(local_gradient_g, dtype=np.float32))
    local_gradient_normalized[local_count <= 0.0] = 0.0
    local_direction_confidence = _structure_confidence(local_j11, local_j12, local_j22, eps)
    local_direction_confidence[local_count <= 0.0] = 0.0

    high_q = _quantile(valid_g, float(cfg.high_gradient_quantile), tau_g)
    high_threshold = max(tau_g, high_q)
    high_gradient = derivative_valid & (gradient_g >= high_threshold)
    local_valid_g = local_gradient_g[local_count > 0.0]
    local_high_q = _quantile(local_valid_g, float(cfg.high_gradient_quantile), high_threshold)
    local_high_threshold = max(tau_g, local_high_q)
    local_high_gradient = (local_count > 0.0) & (local_gradient_g >= local_high_threshold)

    return NormalGeometry(
        normal_valid=derivative_valid,
        normal_du=du,
        normal_dv=dv,
        gradient_g=gradient_g,
        gradient_normalized=gradient_normalized.astype(np.float32),
        noise_sigma_g=float(sigma_g),
        tolerance_tau_g=float(tau_g),
        tolerance_tau_n=float(tau_n),
        tolerance_tau_a=float(tau_a),
        target_edge_length_px=target_px,
        structure_tensor=structure_tensor,
        direction_confidence=direction_confidence,
        local_normal_variance=local_variance.astype(np.float32),
        local_gradient_g=local_gradient_g.astype(np.float32),
        local_gradient_normalized=local_gradient_normalized.astype(np.float32),
        local_structure_tensor=local_tensor.astype(np.float32),
        local_direction_confidence=local_direction_confidence.astype(np.float32),
        local_high_gradient_mask=local_high_gradient,
        high_gradient_mask=high_gradient,
        high_gradient_threshold=float(high_threshold),
    )


def normal_map_to_rgb(normal_map_cam: np.ndarray | None, valid: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[:, :] = np.array([24, 24, 24], dtype=np.uint8)
    if normal_map_cam is None:
        return out
    normal = safe_normalize(np.asarray(normal_map_cam, dtype=np.float64)).astype(np.float32)
    rgb = np.clip(np.rint((normal * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    if rgb.shape[:2] != (height, width):
        rgb = np.asarray(rgb)
    if valid is None:
        out = rgb
    else:
        mask = np.asarray(valid, dtype=bool)
        out[mask] = rgb[mask]
    return out
