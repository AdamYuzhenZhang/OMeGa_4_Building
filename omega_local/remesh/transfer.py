"""Transfer rendered view-normal evidence onto mesh faces.

The Stage 2 transfer is deliberately evidence-only: it writes normalized face
fields and debug counters, but it never changes mesh geometry or topology.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage

from omega_local.remesh.normal_maps import NormalGeometry
from omega_local.remesh.view_buffers import ViewBuffer


@dataclass(frozen=True)
class TransferConfig:
    coverage_threshold: float = 0.15
    support_view_k0: float = 3.0
    min_offset_radius_px: float = 2.0
    max_offset_radius_px: float = 16.0


@dataclass(frozen=True)
class FrameTransferSummary:
    visible_face_count: int
    visible_pixel_count: int
    normal_face_count: int
    normal_pixel_count: int
    high_gradient_pixel_count: int
    offset_delta_px: float
    offset_associated_pixel_count: int


def safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors, dtype=np.float64))


def face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw_normals, axis=1)
    normals = np.divide(
        raw_normals,
        np.maximum(double_area[:, None], 1e-12),
        out=np.zeros_like(raw_normals, dtype=np.float64),
    )
    centers = np.mean(triangles, axis=1)
    areas = 0.5 * double_area
    edge_lengths = np.linalg.norm(triangles[:, [1, 2, 0]] - triangles[:, [0, 1, 2]], axis=2)
    mean_edge_lengths = np.mean(edge_lengths, axis=1)
    return centers.astype(np.float64), normals.astype(np.float64), areas.astype(np.float64), mean_edge_lengths.astype(np.float64)


def summarize(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is not None:
        arr = arr[np.asarray(valid, dtype=bool).reshape(-1)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "count": 0.0,
            "min": 0.0,
            "q25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "q75": 0.0,
            "q90": 0.0,
            "q99": 0.0,
            "max": 0.0,
        }
    qs = np.quantile(arr, [0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "q25": float(qs[0]),
        "median": float(qs[1]),
        "mean": float(np.mean(arr)),
        "q75": float(qs[2]),
        "q90": float(qs[3]),
        "q99": float(qs[4]),
        "max": float(np.max(arr)),
    }


def _group_quantile(face_ids: np.ndarray, values: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    face_ids = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = (face_ids >= 0) & np.isfinite(values)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    face_ids = face_ids[valid]
    values = values[valid]
    order = np.argsort(face_ids, kind="mergesort")
    face_ids = face_ids[order]
    values = values[order]
    unique, starts, counts = np.unique(face_ids, return_index=True, return_counts=True)
    out = np.zeros(unique.shape[0], dtype=np.float32)
    for idx, (start, count) in enumerate(zip(starts, counts)):
        out[idx] = float(np.quantile(values[start : start + count], float(quantile)))
    return unique.astype(np.int32), out


def _face_value_mad(face_chunks: list[np.ndarray], value_chunks: list[np.ndarray], face_count: int) -> np.ndarray:
    out = np.zeros(face_count, dtype=np.float32)
    if not face_chunks:
        return out
    face_ids = np.concatenate(face_chunks).astype(np.int64, copy=False)
    values = np.concatenate(value_chunks).astype(np.float32, copy=False)
    valid = (face_ids >= 0) & (face_ids < face_count) & np.isfinite(values)
    if not np.any(valid):
        return out
    face_ids = face_ids[valid]
    values = values[valid]
    order = np.argsort(face_ids, kind="mergesort")
    face_ids = face_ids[order]
    values = values[order]
    unique, starts, counts = np.unique(face_ids, return_index=True, return_counts=True)
    for face_id, start, count in zip(unique, starts, counts):
        group = values[start : start + count]
        med = float(np.median(group))
        out[int(face_id)] = float(np.median(np.abs(group - med)))
    return out


def _face_value_quantile(
    face_chunks: list[np.ndarray],
    value_chunks: list[np.ndarray],
    face_count: int,
    quantile: float,
) -> np.ndarray:
    out = np.zeros(face_count, dtype=np.float32)
    if not face_chunks:
        return out
    face_ids = np.concatenate(face_chunks).astype(np.int64, copy=False)
    values = np.concatenate(value_chunks).astype(np.float32, copy=False)
    valid = (face_ids >= 0) & (face_ids < face_count) & np.isfinite(values)
    if not np.any(valid):
        return out
    face_ids = face_ids[valid]
    values = values[valid]
    order = np.argsort(face_ids, kind="mergesort")
    face_ids = face_ids[order]
    values = values[order]
    unique, starts, counts = np.unique(face_ids, return_index=True, return_counts=True)
    for face_id, start, count in zip(unique, starts, counts):
        out[int(face_id)] = float(np.quantile(values[start : start + count], float(quantile)))
    return out


def estimate_offset_delta_px(
    *,
    mesh_edge_mask: np.ndarray,
    high_gradient_mask: np.ndarray,
    min_radius_px: float,
    max_radius_px: float,
) -> tuple[float, np.ndarray]:
    """Estimate a per-view pixel offset from normal ridges to mesh edges."""

    edge = np.asarray(mesh_edge_mask, dtype=bool)
    high = np.asarray(high_gradient_mask, dtype=bool)
    if not np.any(edge) or not np.any(high):
        return float(min_radius_px), np.zeros(edge.shape, dtype=np.float32)
    distance_to_edge = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)
    samples = distance_to_edge[high]
    if samples.size == 0:
        delta = float(min_radius_px)
    else:
        delta = float(np.quantile(samples, 0.80))
    delta = float(np.clip(delta, float(min_radius_px), float(max_radius_px)))
    return delta, distance_to_edge.astype(np.float32)


class EvidenceAccumulator:
    def __init__(self, *, vertices: np.ndarray, faces: np.ndarray, config: TransferConfig | None = None) -> None:
        self.config = config or TransferConfig()
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.face_count = int(self.faces.shape[0])
        self.centers, self.normals, self.areas, self.mean_edge_lengths = face_geometry(self.vertices, self.faces)

        n = self.face_count
        self.visible_view_count = np.zeros(n, dtype=np.int32)
        self.normal_view_count = np.zeros(n, dtype=np.int32)
        self.visible_pixel_count = np.zeros(n, dtype=np.int64)
        self.normal_pixel_count = np.zeros(n, dtype=np.int64)
        self.coverage_sum = np.zeros(n, dtype=np.float64)
        self.coverage_observed_count = np.zeros(n, dtype=np.int32)
        self.gradient_sum = np.zeros(n, dtype=np.float64)
        self.gradient_norm_sum = np.zeros(n, dtype=np.float64)
        self.gradient_max = np.zeros(n, dtype=np.float32)
        self.target_px_sum = np.zeros(n, dtype=np.float64)
        self.target_world_sum = np.zeros(n, dtype=np.float64)
        self.mpp_sum = np.zeros(n, dtype=np.float64)
        self.abs_view_cos_sum = np.zeros(n, dtype=np.float64)
        self.tau_g_sum = np.zeros(n, dtype=np.float64)
        self.tau_n_sum = np.zeros(n, dtype=np.float64)
        self.tau_a_sum = np.zeros(n, dtype=np.float64)
        self.tensor_sum = np.zeros((n, 3), dtype=np.float64)
        self.direction_conf_sum = np.zeros(n, dtype=np.float64)
        self.normal_vector_sum = np.zeros((n, 3), dtype=np.float64)
        self.normal_vector_weight_sum = np.zeros(n, dtype=np.float64)
        self.offset_high_gradient_count = np.zeros(n, dtype=np.int64)
        self.offset_gradient_sum = np.zeros(n, dtype=np.float64)
        self.offset_gradient_max = np.zeros(n, dtype=np.float32)
        self.offset_target_px_sum = np.zeros(n, dtype=np.float64)
        self._kappa_face_chunks: list[np.ndarray] = []
        self._kappa_value_chunks: list[np.ndarray] = []
        self._target_px_face_chunks: list[np.ndarray] = []
        self._target_px_value_chunks: list[np.ndarray] = []
        self._target_world_face_chunks: list[np.ndarray] = []
        self._target_world_value_chunks: list[np.ndarray] = []

    def add_frame(
        self,
        *,
        view_buffer: ViewBuffer,
        normal_geometry: NormalGeometry,
        K: np.ndarray,
        camtoworld: np.ndarray,
        normal_map_cam: np.ndarray | None = None,
    ) -> FrameTransferSummary:
        face_id = view_buffer.face_id
        visible = face_id >= 0
        visible_faces, visible_counts = np.unique(face_id[visible], return_counts=True) if np.any(visible) else (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
        )
        if visible_faces.size:
            self.visible_view_count[visible_faces] += 1
            self.visible_pixel_count[visible_faces] += visible_counts.astype(np.int64)

        normal_valid = visible & np.asarray(normal_geometry.normal_valid, dtype=bool)
        normal_faces, normal_counts = np.unique(face_id[normal_valid], return_counts=True) if np.any(normal_valid) else (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int64),
        )
        if visible_faces.size:
            coverage_by_visible = np.zeros(visible_faces.shape[0], dtype=np.float64)
            if normal_faces.size:
                normal_count_by_face = dict(zip(normal_faces.tolist(), normal_counts.tolist()))
                coverage_by_visible = np.array(
                    [normal_count_by_face.get(int(fid), 0) / max(int(count), 1) for fid, count in zip(visible_faces, visible_counts)],
                    dtype=np.float64,
                )
            self.coverage_sum[visible_faces] += coverage_by_visible
            self.coverage_observed_count[visible_faces] += 1
            accepted = visible_faces[coverage_by_visible >= float(self.config.coverage_threshold)]
            self.normal_view_count[accepted] += 1
        if normal_faces.size:
            self.normal_pixel_count[normal_faces] += normal_counts.astype(np.int64)

        if np.any(normal_valid):
            fids = face_id[normal_valid].astype(np.int64)
            depth = view_buffer.depth[normal_valid].astype(np.float64)
            fx = max(float(K[0, 0]), 1e-6)
            fy = max(float(K[1, 1]), 1e-6)
            mpp = depth / np.sqrt(fx * fy)
            gradient = normal_geometry.gradient_g[normal_valid].astype(np.float64)
            gradient_norm = normal_geometry.gradient_normalized[normal_valid].astype(np.float64)
            target_px = normal_geometry.target_edge_length_px[normal_valid].astype(np.float64)
            target_world = target_px * mpp

            self.gradient_sum += np.bincount(fids, weights=gradient, minlength=self.face_count)
            self.gradient_norm_sum += np.bincount(fids, weights=gradient_norm, minlength=self.face_count)
            self.target_px_sum += np.bincount(fids, weights=target_px, minlength=self.face_count)
            self.target_world_sum += np.bincount(fids, weights=target_world, minlength=self.face_count)
            self.mpp_sum += np.bincount(fids, weights=mpp, minlength=self.face_count)
            np.maximum.at(self.gradient_max, fids, gradient.astype(np.float32))

            tensor = normal_geometry.structure_tensor[normal_valid].astype(np.float64)
            for channel in range(3):
                self.tensor_sum[:, channel] += np.bincount(fids, weights=tensor[:, channel], minlength=self.face_count)
            self.direction_conf_sum += np.bincount(
                fids,
                weights=normal_geometry.direction_confidence[normal_valid].astype(np.float64),
                minlength=self.face_count,
            )
            if normal_map_cam is not None:
                normal_cam = np.asarray(normal_map_cam, dtype=np.float64)[normal_valid]
                normal_world = safe_normalize(normal_cam @ np.asarray(camtoworld[:3, :3], dtype=np.float64).T)
                mesh_normal = self.normals[fids]
                sign = np.where(np.sum(normal_world * mesh_normal, axis=1) < 0.0, -1.0, 1.0)
                normal_world = normal_world * sign[:, None]
                align = np.abs(np.sum(normal_world * mesh_normal, axis=1)).clip(0.0, 1.0)
                weight = np.maximum(align, 0.25)
                for channel in range(3):
                    self.normal_vector_sum[:, channel] += np.bincount(
                        fids,
                        weights=normal_world[:, channel] * weight,
                        minlength=self.face_count,
                    )
                self.normal_vector_weight_sum += np.bincount(fids, weights=weight, minlength=self.face_count)

            q_faces, q_gradient = _group_quantile(fids, gradient.astype(np.float32), 0.80)
            if q_faces.size:
                self._kappa_face_chunks.append(q_faces)
                self._kappa_value_chunks.append(q_gradient)
            q_faces, q_target_px = _group_quantile(fids, target_px.astype(np.float32), 0.30)
            if q_faces.size:
                self._target_px_face_chunks.append(q_faces)
                self._target_px_value_chunks.append(q_target_px)
            q_faces, q_target_world = _group_quantile(fids, target_world.astype(np.float32), 0.30)
            if q_faces.size:
                self._target_world_face_chunks.append(q_faces)
                self._target_world_value_chunks.append(q_target_world)

            if normal_faces.size:
                camera_pos = np.asarray(camtoworld[:3, 3], dtype=np.float64)
                view_dirs = safe_normalize(camera_pos[None, :] - self.centers[normal_faces])
                abs_cos = np.abs(np.sum(self.normals[normal_faces] * view_dirs, axis=1))
                self.abs_view_cos_sum[normal_faces] += abs_cos
                self.tau_g_sum[normal_faces] += float(normal_geometry.tolerance_tau_g)
                self.tau_n_sum[normal_faces] += float(normal_geometry.tolerance_tau_n)
                self.tau_a_sum[normal_faces] += float(normal_geometry.tolerance_tau_a)

        offset_delta, _ = estimate_offset_delta_px(
            mesh_edge_mask=view_buffer.mesh_edge_mask,
            high_gradient_mask=normal_geometry.high_gradient_mask,
            min_radius_px=float(self.config.min_offset_radius_px),
            max_radius_px=float(self.config.max_offset_radius_px),
        )
        offset_count = self._add_offset_high_gradient_pixels(
            view_buffer=view_buffer,
            normal_geometry=normal_geometry,
            offset_delta=float(offset_delta),
        )

        return FrameTransferSummary(
            visible_face_count=int(visible_faces.size),
            visible_pixel_count=int(np.count_nonzero(visible)),
            normal_face_count=int(normal_faces.size),
            normal_pixel_count=int(np.count_nonzero(normal_valid)),
            high_gradient_pixel_count=int(np.count_nonzero(normal_geometry.high_gradient_mask)),
            offset_delta_px=float(offset_delta),
            offset_associated_pixel_count=int(offset_count),
        )

    def _add_offset_high_gradient_pixels(
        self,
        *,
        view_buffer: ViewBuffer,
        normal_geometry: NormalGeometry,
        offset_delta: float,
    ) -> int:
        visible = view_buffer.face_id >= 0
        high = np.asarray(normal_geometry.high_gradient_mask, dtype=bool)
        if not np.any(visible) or not np.any(high):
            return 0
        distance, indices = ndimage.distance_transform_edt(~visible, return_indices=True)
        associate = high & (distance <= float(offset_delta))
        if not np.any(associate):
            return 0
        nearest_y = indices[0][associate]
        nearest_x = indices[1][associate]
        nearest_face = view_buffer.face_id[nearest_y, nearest_x].astype(np.int64)
        valid = nearest_face >= 0
        if not np.any(valid):
            return 0
        nearest_face = nearest_face[valid]
        dist = distance[associate][valid].astype(np.float64)
        sigma = max(float(offset_delta), 1e-6)
        weight = np.exp(-(dist * dist) / (2.0 * sigma * sigma))
        gradient = normal_geometry.gradient_g[associate][valid].astype(np.float64)
        target_px = normal_geometry.target_edge_length_px[associate][valid].astype(np.float64)

        self.offset_high_gradient_count += np.bincount(nearest_face, minlength=self.face_count).astype(np.int64)
        self.offset_gradient_sum += np.bincount(nearest_face, weights=gradient * weight, minlength=self.face_count)
        self.offset_target_px_sum += np.bincount(nearest_face, weights=target_px * weight, minlength=self.face_count)
        np.maximum.at(self.offset_gradient_max, nearest_face, gradient.astype(np.float32))
        return int(nearest_face.size)

    def finalize(self, selected_frame_count: int) -> dict[str, np.ndarray]:
        normal_pixels = np.maximum(self.normal_pixel_count.astype(np.float64), 1.0)
        normal_views = np.maximum(self.normal_view_count.astype(np.float64), 1.0)
        observed_views = np.maximum(self.coverage_observed_count.astype(np.float64), 1.0)
        normal_face_valid = self.normal_pixel_count > 0
        support_view_count = self.normal_view_count.astype(np.float32)
        coverage_mean = np.divide(
            self.coverage_sum,
            observed_views,
            out=np.zeros(self.face_count, dtype=np.float64),
            where=observed_views > 0,
        ).astype(np.float32)
        support_raw = coverage_mean * (1.0 - np.exp(-support_view_count / max(float(self.config.support_view_k0), 1e-6)))

        face_kappa = _face_value_quantile(self._kappa_face_chunks, self._kappa_value_chunks, self.face_count, 0.80)
        disagreement = _face_value_mad(self._kappa_face_chunks, self._kappa_value_chunks, self.face_count)
        supported_disagreement = disagreement[(support_raw > 0.0) & np.isfinite(disagreement)]
        tau_i = float(np.quantile(supported_disagreement, 0.75)) if supported_disagreement.size else 1.0
        tau_i = max(tau_i, 1e-6)
        support = support_raw * np.exp(-(disagreement * disagreement) / (tau_i * tau_i))

        target_px_q30 = _face_value_quantile(self._target_px_face_chunks, self._target_px_value_chunks, self.face_count, 0.30)
        target_world_q30 = _face_value_quantile(
            self._target_world_face_chunks,
            self._target_world_value_chunks,
            self.face_count,
            0.30,
        )
        mean_gradient = np.divide(
            self.gradient_sum,
            normal_pixels,
            out=np.zeros(self.face_count, dtype=np.float64),
            where=normal_pixels > 0,
        ).astype(np.float32)
        mean_gradient_norm = np.divide(
            self.gradient_norm_sum,
            normal_pixels,
            out=np.zeros(self.face_count, dtype=np.float64),
            where=normal_pixels > 0,
        ).astype(np.float32)
        mean_target_px = np.divide(self.target_px_sum, normal_pixels, out=np.zeros(self.face_count), where=normal_pixels > 0).astype(np.float32)
        mean_target_world = np.divide(
            self.target_world_sum,
            normal_pixels,
            out=np.zeros(self.face_count),
            where=normal_pixels > 0,
        ).astype(np.float32)
        mean_mpp = np.divide(self.mpp_sum, normal_pixels, out=np.zeros(self.face_count), where=normal_pixels > 0).astype(np.float32)
        mean_abs_view_cos = np.divide(
            self.abs_view_cos_sum,
            normal_views,
            out=np.zeros(self.face_count),
            where=normal_views > 0,
        ).astype(np.float32)
        mean_tau_g = np.divide(self.tau_g_sum, normal_views, out=np.zeros(self.face_count), where=normal_views > 0).astype(np.float32)
        mean_tau_n = np.divide(self.tau_n_sum, normal_views, out=np.zeros(self.face_count), where=normal_views > 0).astype(np.float32)
        mean_tau_a = np.divide(self.tau_a_sum, normal_views, out=np.zeros(self.face_count), where=normal_views > 0).astype(np.float32)
        direction_tensor = np.divide(
            self.tensor_sum,
            normal_pixels[:, None],
            out=np.zeros((self.face_count, 3), dtype=np.float64),
            where=normal_pixels[:, None] > 0,
        ).astype(np.float32)
        direction_conf = np.divide(
            self.direction_conf_sum,
            normal_pixels,
            out=np.zeros(self.face_count),
            where=normal_pixels > 0,
        ).astype(np.float32)
        fused_normal = safe_normalize(
            np.divide(
                self.normal_vector_sum,
                np.maximum(self.normal_vector_weight_sum[:, None], 1e-12),
                out=np.zeros_like(self.normal_vector_sum),
                where=self.normal_vector_weight_sum[:, None] > 0,
            )
        )
        has_fused = self.normal_vector_weight_sum > 0.0
        fused_normal[~has_fused] = self.normals[~has_fused]
        fused_alignment = np.sum(fused_normal * self.normals, axis=1).clip(-1.0, 1.0).astype(np.float32)
        offset_pixels = np.maximum(self.offset_high_gradient_count.astype(np.float64), 1.0)
        offset_gradient_mean = np.divide(
            self.offset_gradient_sum,
            offset_pixels,
            out=np.zeros(self.face_count),
            where=offset_pixels > 0,
        ).astype(np.float32)
        offset_target_px_mean = np.divide(
            self.offset_target_px_sum,
            offset_pixels,
            out=np.zeros(self.face_count),
            where=offset_pixels > 0,
        ).astype(np.float32)

        return {
            "face_centers": self.centers.astype(np.float32),
            "mesh_face_normals": self.normals.astype(np.float32),
            "face_area": self.areas.astype(np.float32),
            "face_mean_edge_length": self.mean_edge_lengths.astype(np.float32),
            "visible_view_count": self.visible_view_count,
            "normal_view_count": self.normal_view_count,
            "view_count": self.visible_view_count,
            "normal_count": self.normal_view_count,
            "visible_pixel_count": self.visible_pixel_count,
            "normal_pixel_count": self.normal_pixel_count,
            "normal_coverage_mean": coverage_mean,
            "face_support_raw": support_raw.astype(np.float32),
            "face_support": support.astype(np.float32),
            "mean_valid_support": support.astype(np.float32),
            "face_view_disagreement": disagreement.astype(np.float32),
            "face_normal_kappa": face_kappa.astype(np.float32),
            "normal_gradient_q80": face_kappa.astype(np.float32),
            "face_normal_gradient_mean": mean_gradient,
            "mean_normal_gradient": mean_gradient,
            "face_normal_gradient_normalized_mean": mean_gradient_norm,
            "face_normal_gradient_max": np.where(normal_face_valid, self.gradient_max, 0.0).astype(np.float32),
            "max_normal_gradient": np.where(normal_face_valid, self.gradient_max, 0.0).astype(np.float32),
            "face_target_edge_length_px_mean": mean_target_px,
            "mean_target_edge_length_px": mean_target_px,
            "face_target_edge_length_px_q30": target_px_q30.astype(np.float32),
            "min_target_edge_length_px": target_px_q30.astype(np.float32),
            "face_target_edge_length_world_mean": mean_target_world,
            "mean_target_edge_length_world_conservative": mean_target_world,
            "face_detail_target_length": target_world_q30.astype(np.float32),
            "face_target_edge_length_world_q30": target_world_q30.astype(np.float32),
            "min_target_edge_length_world_conservative": target_world_q30.astype(np.float32),
            "face_position_tolerance": np.maximum(2.0 * mean_mpp, 0.25 * self.mean_edge_lengths).astype(np.float32),
            "face_pixel_world_scale": mean_mpp,
            "mean_fronto_pixel_world_scale": mean_mpp,
            "face_abs_view_cos_mean": mean_abs_view_cos,
            "mean_abs_view_cos": mean_abs_view_cos,
            "face_tau_g": mean_tau_g,
            "face_tau_n": mean_tau_n,
            "face_tau_a": mean_tau_a,
            "face_direction_tensor": direction_tensor,
            "face_direction_confidence": direction_conf,
            "fused_face_normal": fused_normal.astype(np.float32),
            "fused_normal_weight": self.normal_vector_weight_sum.astype(np.float32),
            "fused_normal_alignment": fused_alignment,
            "offset_high_gradient_pixel_count": self.offset_high_gradient_count,
            "offset_normal_gradient_mean": offset_gradient_mean,
            "offset_normal_gradient_max": self.offset_gradient_max.astype(np.float32),
            "offset_target_edge_length_px_mean": offset_target_px_mean,
            "selected_frame_count": np.array(int(selected_frame_count), dtype=np.int32),
            "support_disagreement_tau": np.array(float(tau_i), dtype=np.float32),
        }
