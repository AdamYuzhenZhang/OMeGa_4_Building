"""Joint 2D/3D CRF recovery from persistent-region point observations.

This is a controlled adaptation of Xie et al., CVPR 2016, "Semantic
Instance Annotation of Street Scenes by 3D to 2D Label Transfer." It retains
separate image and point fields, within-field Potts terms, cross-field
projection terms, and mean-field inference. Fine superpixels and local k-NN
graphs approximate the paper's dense kernels because its implementation and
learned parameters were not released.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .colmap_dense_propagation import (
    _boundary_graph,
    _load_frame_evidence,
    _resize_depth,
    _resize_normal,
    _resize_rgb,
    _segment_centroids,
    _seed_superpixels,
)
from .colmap_point_field import (
    ColmapPointFieldConfig,
    ColmapProjectedPointSource,
    ProjectedPointField,
    colmap_point_field_availability,
)
from .dense_recovery_source import load_sparse_labels, sparse_source_availability, validate_sparse_source
from .feedforward_point_field import (
    FeedForwardPointFieldConfig,
    FeedForwardProjectedPointSource,
    feedforward_point_field_availability,
)
from .sam2_video_propagation import PropagationFrame


@dataclass(frozen=True)
class MultiFieldCrfConfig:
    source_method_id: str
    source_run_dir: Path
    evidence_root: Path
    point_field: ColmapPointFieldConfig | FeedForwardPointFieldConfig
    source_name: str = "COLMAP Tracks"
    point_visibility_policy: str = "recorded_colmap_track_observations"
    target_superpixels: int = 6_000
    compactness: float = 8.0
    mean_field_iterations: int = 8
    pixel_pairwise_weight: float = 1.15
    point_pairwise_weight: float = 0.70
    cross_field_weight: float = 1.80
    damping: float = 0.55
    max_seed_distance_pixels: float = 220.0
    min_confidence: float = 0.50
    min_competitor_margin: float = 0.08
    max_normalized_entropy: float = 0.82
    pixel_neighbors: int = 8
    point_neighbors: int = 8


class MultiFieldCrfRecoverySession:
    """Runs the paper-inspired decoder over a completed projected-point layer."""

    def __init__(self, config: MultiFieldCrfConfig) -> None:
        self.config = config
        self._run_lock = threading.Lock()

    def recover_from_source(
        self,
        *,
        run_input: Any,
        progress_callback: Callable[[int, int, int], None] | None = None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None = None,
    ) -> list[dict[str, Any]]:
        with self._run_lock:
            validate_sparse_source(
                self.config.source_run_dir,
                source_name=str(self.config.source_name),
                expected_fingerprint=str(run_input.fingerprint),
            )
            frames = list(run_input.frames)
            point_source = _make_projected_point_source(self.config.point_field, frames)
            source_by_frame = {int(source.frame_id): source for source in run_input.sources}
            rows: list[dict[str, Any]] = []
            total = len(frames)
            for index, frame in enumerate(frames):
                sparse_labels = load_sparse_labels(
                    self.config.source_run_dir,
                    frame,
                    source_name=str(self.config.source_name),
                )
                source = source_by_frame.get(int(frame.frame_id))
                if source is not None:
                    dense_labels = np.asarray(source.labels, dtype=np.uint16).copy()
                    stats: dict[str, Any] = {
                        "recoveryPolicy": "manual_anchor",
                        "sparsePixelCount": int(np.count_nonzero(sparse_labels)),
                        "densePixelCount": int(np.count_nonzero(dense_labels)),
                        "denseCoverage": float(np.count_nonzero(dense_labels) / max(dense_labels.size, 1)),
                    }
                else:
                    rgb, depth, depth_valid, normal, normal_valid = _load_frame_evidence(
                        frame,
                        self.config.evidence_root,
                    )
                    projected_points = point_source.project(frame)
                    dense_labels, stats = densify_multifield_crf(
                        rgb=rgb,
                        depth=depth,
                        depth_valid=depth_valid,
                        normal=normal,
                        normal_valid=normal_valid,
                        sparse_labels=sparse_labels,
                        point_field=projected_points,
                        config=self.config,
                    )
                    stats["recoveryPolicy"] = "xie16_multifield_crf_adaptation"

                rows.append(
                    {
                        "frameId": int(frame.frame_id),
                        "sourceMethodId": str(self.config.source_method_id),
                        "sparseCoverage": float(np.count_nonzero(sparse_labels) / max(sparse_labels.size, 1)),
                        "coverage": float(stats["denseCoverage"]),
                        **stats,
                    }
                )
                if save_callback is not None:
                    save_callback(frame, dense_labels)
                if progress_callback is not None:
                    progress_callback(index + 1, total, int(frame.frame_id))
            return rows


def multifield_crf_availability(config: MultiFieldCrfConfig) -> tuple[bool, str]:
    sparse_ready, sparse_reason = sparse_source_availability(
        config.source_run_dir,
        source_name=str(config.source_name),
    )
    if not sparse_ready:
        return False, sparse_reason
    point_ready, point_reason = _point_field_availability(config.point_field)
    if not point_ready:
        return False, point_reason
    root = config.evidence_root.expanduser().resolve()
    for label, path in (
        ("Depth Anything evidence", root / "depth_npz"),
        ("StableNormal evidence", root / "normal_npz"),
    ):
        if not path.is_dir() or not any(path.glob("*.npz")):
            return False, f"{label} is missing: {path}"
    try:
        import cv2  # noqa: F401
        from scipy import sparse  # noqa: F401
        from scipy.spatial import cKDTree  # noqa: F401
        from skimage.color import rgb2lab  # noqa: F401
        from skimage.segmentation import slic  # noqa: F401
    except ImportError as exc:
        return False, f"2D/3D CRF dependency is missing: {exc}"
    return True, "Ready"


def _make_projected_point_source(
    config: ColmapPointFieldConfig | FeedForwardPointFieldConfig,
    frames: list[PropagationFrame],
) -> ColmapProjectedPointSource | FeedForwardProjectedPointSource:
    if isinstance(config, ColmapPointFieldConfig):
        return ColmapProjectedPointSource(config, frames)
    if isinstance(config, FeedForwardPointFieldConfig):
        return FeedForwardProjectedPointSource(config)
    raise TypeError(f"Unsupported projected point-field config: {type(config).__name__}")


def _point_field_availability(
    config: ColmapPointFieldConfig | FeedForwardPointFieldConfig,
) -> tuple[bool, str]:
    if isinstance(config, ColmapPointFieldConfig):
        return colmap_point_field_availability(config)
    if isinstance(config, FeedForwardPointFieldConfig):
        return feedforward_point_field_availability(config)
    return False, f"Unsupported projected point-field config: {type(config).__name__}"


def densify_multifield_crf(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    normal: np.ndarray,
    normal_valid: np.ndarray,
    sparse_labels: np.ndarray,
    point_field: ProjectedPointField,
    config: MultiFieldCrfConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Infer image labels while jointly regularizing projected 3D points."""

    try:
        import cv2
        from scipy import sparse as scipy_sparse
        from scipy.spatial import cKDTree
        from skimage.color import rgb2lab
        from skimage.segmentation import slic
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("2D/3D CRF requires OpenCV, SciPy, and scikit-image.") from exc

    sparse_labels = np.asarray(sparse_labels, dtype=np.uint16)
    if sparse_labels.ndim != 2:
        raise ValueError(f"Sparse labels must be a 2D array, got {sparse_labels.shape}.")
    _validate_point_field(point_field)
    height, width = sparse_labels.shape
    rgb_u8 = _resize_rgb(np.asarray(rgb), width, height)
    depth_f32, depth_ok = _resize_depth(cv2, depth, depth_valid, width, height)
    normal_f32, normal_ok = _resize_normal(cv2, normal, normal_valid, width, height)
    lab = rgb2lab(rgb_u8.astype(np.float32) / 255.0).astype(np.float32)

    point_xy = np.asarray(point_field.pixel_xy, dtype=np.int32)
    inside = (
        (point_xy[:, 0] >= 0)
        & (point_xy[:, 0] < width)
        & (point_xy[:, 1] >= 0)
        & (point_xy[:, 1] < height)
    )
    point_xy = point_xy[inside]
    point_positions = np.asarray(point_field.positions, dtype=np.float32)[inside]
    point_normals = np.asarray(point_field.normals, dtype=np.float32)[inside]
    point_labels = np.asarray(point_field.labels, dtype=np.uint16)[inside]

    region_ids = np.unique(
        np.concatenate((sparse_labels[sparse_labels > 0], point_labels[point_labels > 0]))
    ).astype(np.uint16)
    sparse_pixel_count = int(np.count_nonzero(sparse_labels))
    if region_ids.size == 0:
        return sparse_labels.copy(), _empty_stats(
            sparse_pixel_count,
            point_xy.shape[0],
            sparse_labels.size,
        )

    requested = int(np.clip(config.target_superpixels, 500, max(500, sparse_labels.size // 16)))
    segments = slic(
        rgb_u8,
        n_segments=requested,
        compactness=float(config.compactness),
        sigma=0.8,
        start_label=0,
        enforce_connectivity=True,
        min_size_factor=0.25,
        max_size_factor=3.0,
        channel_axis=-1,
    ).astype(np.int32, copy=False)
    node_count = int(segments.max()) + 1
    centroids = _segment_centroids(segments, node_count)
    node_lab = _segment_mean(lab, segments, node_count)
    node_depth = _segment_mean(depth_f32[..., None], segments, node_count)[:, 0]
    node_normal = _normalized_rows(_segment_mean(normal_f32, segments, node_count))
    seed_labels, ambiguous_seed_count = _seed_superpixels(segments, sparse_labels, node_count)

    point_nodes = segments[point_xy[:, 1], point_xy[:, 0]] if point_xy.size else np.empty(0, dtype=np.int32)
    label_to_column = {int(region_id): index + 1 for index, region_id in enumerate(region_ids.tolist())}
    label_count = int(region_ids.size) + 1
    pixel_unary, nearest_seed_distance = _pixel_unary(
        lab=lab,
        sparse_labels=sparse_labels,
        point_xy=point_xy,
        point_labels=point_labels,
        centroids=centroids,
        node_lab=node_lab,
        region_ids=region_ids,
        max_seed_distance=float(config.max_seed_distance_pixels),
    )
    point_unary = _point_unary(point_labels, label_to_column, label_count)

    boundary = _boundary_graph(
        lab,
        depth_f32,
        depth_ok,
        normal_f32,
        normal_ok,
        segments,
        node_count,
    )
    pixel_affinity = _pixel_affinity(
        scipy_sparse=scipy_sparse,
        cKDTree=cKDTree,
        boundary=boundary,
        centroids=centroids,
        node_lab=node_lab,
        node_depth=node_depth,
        node_normal=node_normal,
        neighbors=int(config.pixel_neighbors),
    )
    point_affinity = _point_affinity(
        scipy_sparse=scipy_sparse,
        cKDTree=cKDTree,
        positions=point_positions,
        normals=point_normals,
        neighbors=int(config.point_neighbors),
    )

    q_pixel = _softmax(-pixel_unary)
    q_point = _softmax(-point_unary) if point_unary.shape[0] else np.empty((0, label_count), dtype=np.float32)
    damping = float(np.clip(config.damping, 0.0, 0.95))
    for _iteration in range(max(int(config.mean_field_iterations), 1)):
        point_to_pixel = _aggregate_point_probabilities(q_point, point_nodes, node_count, label_count)
        pixel_logits = (
            -pixel_unary
            + float(config.pixel_pairwise_weight) * np.asarray(pixel_affinity @ q_pixel)
            + float(config.cross_field_weight) * point_to_pixel
        )
        next_pixel = _softmax(pixel_logits)
        q_pixel = damping * q_pixel + (1.0 - damping) * next_pixel
        if q_point.shape[0]:
            point_logits = (
                -point_unary
                + float(config.point_pairwise_weight) * np.asarray(point_affinity @ q_point)
                + float(config.cross_field_weight) * q_pixel[point_nodes]
            )
            next_point = _softmax(point_logits)
            q_point = damping * q_point + (1.0 - damping) * next_point

    node_labels, accepted, confidence, margin, entropy = _decode_nodes(
        q_pixel,
        region_ids,
        nearest_seed_distance,
        config,
    )
    seeded_nodes = np.flatnonzero(seed_labels > 0)
    node_labels[seeded_nodes] = seed_labels[seeded_nodes]
    dense = node_labels[segments]
    dense[sparse_labels > 0] = sparse_labels[sparse_labels > 0]
    dense_pixel_count = int(np.count_nonzero(dense))
    positive_points = int(np.count_nonzero(point_labels))
    return dense.astype(np.uint16, copy=False), {
        "sparsePixelCount": sparse_pixel_count,
        "densePixelCount": dense_pixel_count,
        "denseCoverage": float(dense_pixel_count / max(dense.size, 1)),
        "coverageGain": float(dense_pixel_count / max(sparse_pixel_count, 1)),
        "regionCount": int(region_ids.size),
        "superpixelCount": node_count,
        "seedSuperpixelCount": int(seeded_nodes.size),
        "ambiguousSeedSuperpixelCount": int(ambiguous_seed_count),
        "acceptedSuperpixelCount": int(np.count_nonzero(accepted)),
        "visiblePointCount": int(point_xy.shape[0]),
        "labeledVisiblePointCount": positive_points,
        "meanAcceptedConfidence": float(np.mean(confidence[accepted])) if np.any(accepted) else 0.0,
        "meanAcceptedMargin": float(np.mean(margin[accepted])) if np.any(accepted) else 0.0,
        "meanAcceptedNormalizedEntropy": float(np.mean(entropy[accepted])) if np.any(accepted) else 0.0,
        "meanFieldIterations": int(max(int(config.mean_field_iterations), 1)),
        "paperTermsRetained": ["2d_unary", "3d_unary", "2d_pairwise", "3d_pairwise", "2d_3d_pairwise"],
        "paperTermsSubstituted": {
            "2d_field": "fine_superpixels_with_local_and_bilateral_knn_affinities",
            "3d_initialization": "persistent_labels_voted_from_manual_keyframes",
            "visibility": str(config.point_visibility_policy),
            "geometric_image_cues": "depth_anything_v2_and_stable_normal_pairwise_boundaries",
        },
        "paperTermsOmitted": ["manual_3d_primitives", "street_fold_curb_unary", "learned_crf_weights"],
    }


def _pixel_unary(
    *,
    lab: np.ndarray,
    sparse_labels: np.ndarray,
    point_xy: np.ndarray,
    point_labels: np.ndarray,
    centroids: np.ndarray,
    node_lab: np.ndarray,
    region_ids: np.ndarray,
    max_seed_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    label_count = int(region_ids.size) + 1
    unary = np.full((centroids.shape[0], label_count), 6.0, dtype=np.float32)
    unary[:, 0] = 1.20
    nearest_any = np.full(centroids.shape[0], np.inf, dtype=np.float32)
    yy, xx = np.nonzero(sparse_labels > 0)
    sparse_xy = np.column_stack((xx, yy)).astype(np.float32)
    sparse_values = sparse_labels[yy, xx]

    for column, raw_region_id in enumerate(region_ids.tolist(), start=1):
        region_id = int(raw_region_id)
        sparse_region_xy = sparse_xy[sparse_values == region_id]
        point_region_xy = point_xy[point_labels == region_id].astype(np.float32, copy=False)
        sample_xy = np.concatenate((sparse_region_xy, point_region_xy), axis=0)
        if sample_xy.size == 0:
            continue
        sample_pixels = np.rint(sample_xy).astype(np.int32)
        sample_lab = lab[sample_pixels[:, 1], sample_pixels[:, 0]]
        center = np.median(sample_lab, axis=0)
        spread = np.maximum(np.median(np.abs(sample_lab - center[None, :]), axis=0) * 1.4826, 5.0)
        appearance = np.mean(((node_lab - center[None, :]) / spread[None, :]) ** 2, axis=1)
        distance = cKDTree(sample_xy).query(centroids, k=1, workers=-1)[0].astype(np.float32)
        nearest_any = np.minimum(nearest_any, distance)
        spatial = (distance / max(float(max_seed_distance), 1.0)) ** 2
        unary[:, column] = np.clip(0.65 * appearance + 0.55 * spatial, 0.0, 8.0)
    return unary, nearest_any


def _point_unary(point_labels: np.ndarray, label_to_column: dict[int, int], label_count: int) -> np.ndarray:
    count = int(point_labels.shape[0])
    # Unlabeled projected points are missing evidence, not semantic background.
    # A flat unary lets labeled neighbors and image coupling decide their state.
    unary = np.zeros((count, label_count), dtype=np.float32)
    if count == 0:
        return unary
    for index, raw_label in enumerate(point_labels.tolist()):
        label = int(raw_label)
        if label <= 0:
            continue
        unary[index] = 10.0
        unary[index, label_to_column[label]] = 0.0
    return unary


def _pixel_affinity(
    *,
    scipy_sparse: Any,
    cKDTree: Any,
    boundary: list[list[tuple[int, float]]],
    centroids: np.ndarray,
    node_lab: np.ndarray,
    node_depth: np.ndarray,
    node_normal: np.ndarray,
    neighbors: int,
) -> Any:
    node_count = int(centroids.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for node, edges in enumerate(boundary):
        for neighbor, cost in edges:
            rows.append(node)
            cols.append(int(neighbor))
            values.append(float(np.exp(-max(float(cost), 0.0))))

    features = np.column_stack(
        (
            centroids / 80.0,
            node_lab / np.array([18.0, 12.0, 12.0], dtype=np.float32),
            node_depth[:, None] / 0.35,
            node_normal / 0.45,
        )
    ).astype(np.float32)
    k = min(max(int(neighbors), 1) + 1, node_count)
    if k > 1:
        distances, indices = cKDTree(features).query(features, k=k, workers=-1)
        distances = np.asarray(distances)[:, 1:]
        indices = np.asarray(indices)[:, 1:]
        repeated = np.repeat(np.arange(node_count), indices.shape[1])
        rows.extend(repeated.tolist())
        cols.extend(indices.reshape(-1).astype(np.int64).tolist())
        values.extend(np.exp(-0.5 * np.square(distances.reshape(-1))).tolist())
    return _symmetric_row_normalized(scipy_sparse, rows, cols, values, node_count)


def _point_affinity(
    *,
    scipy_sparse: Any,
    cKDTree: Any,
    positions: np.ndarray,
    normals: np.ndarray,
    neighbors: int,
) -> Any:
    count = int(positions.shape[0])
    if count == 0:
        return scipy_sparse.csr_matrix((0, 0), dtype=np.float32)
    k = min(max(int(neighbors), 1) + 1, count)
    if k <= 1:
        return scipy_sparse.identity(count, dtype=np.float32, format="csr")
    distances, indices = cKDTree(positions).query(positions, k=k, workers=-1)
    distances = np.asarray(distances, dtype=np.float32)[:, 1:]
    indices = np.asarray(indices, dtype=np.int64)[:, 1:]
    positive = distances[np.isfinite(distances) & (distances > 1.0e-8)]
    sigma = float(np.median(positive)) * 2.0 if positive.size else 1.0
    repeated = np.repeat(np.arange(count), indices.shape[1])
    normal_dot = np.abs(np.sum(normals[repeated] * normals[indices.reshape(-1)], axis=1))
    weights = np.exp(-0.5 * np.square(distances.reshape(-1) / max(sigma, 1.0e-8)))
    weights *= np.exp(-2.0 * (1.0 - np.clip(normal_dot, 0.0, 1.0)))
    return _symmetric_row_normalized(
        scipy_sparse,
        repeated.tolist(),
        indices.reshape(-1).tolist(),
        weights.tolist(),
        count,
    )


def _symmetric_row_normalized(
    scipy_sparse: Any,
    rows: list[int],
    cols: list[int],
    values: list[float],
    size: int,
) -> Any:
    matrix = scipy_sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(size, size),
        dtype=np.float32,
    ).tocsr()
    matrix = matrix.maximum(matrix.T)
    row_sum = np.asarray(matrix.sum(axis=1)).reshape(-1)
    inverse = np.zeros_like(row_sum, dtype=np.float32)
    positive = row_sum > 1.0e-8
    inverse[positive] = 1.0 / row_sum[positive]
    return scipy_sparse.diags(inverse) @ matrix


def _aggregate_point_probabilities(
    q_point: np.ndarray,
    point_nodes: np.ndarray,
    node_count: int,
    label_count: int,
) -> np.ndarray:
    output = np.zeros((node_count, label_count), dtype=np.float32)
    if q_point.shape[0] == 0:
        return output
    np.add.at(output, point_nodes, q_point)
    counts = np.bincount(point_nodes, minlength=node_count).astype(np.float32)
    positive = counts > 0
    output[positive] /= counts[positive, None]
    return output


def _decode_nodes(
    probabilities: np.ndarray,
    region_ids: np.ndarray,
    nearest_seed_distance: np.ndarray,
    config: MultiFieldCrfConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positive = probabilities[:, 1:]
    best_column = np.argmax(positive, axis=1)
    node_index = np.arange(probabilities.shape[0])
    confidence = positive[node_index, best_column]
    if positive.shape[1] > 1:
        second = np.partition(positive, -2, axis=1)[:, -2]
    else:
        second = probabilities[:, 0]
    margin = confidence - second
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12)), axis=1)
    entropy /= max(float(np.log(probabilities.shape[1])), 1.0e-8)
    accepted = (
        (confidence >= float(config.min_confidence))
        & (margin >= float(config.min_competitor_margin))
        & (entropy <= float(config.max_normalized_entropy))
        & (nearest_seed_distance <= float(config.max_seed_distance_pixels))
    )
    labels = np.zeros(probabilities.shape[0], dtype=np.uint16)
    labels[accepted] = region_ids[best_column[accepted]]
    return labels, accepted, confidence, margin, entropy


def _segment_mean(values: np.ndarray, segments: np.ndarray, node_count: int) -> np.ndarray:
    flat_segments = segments.reshape(-1)
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1, values.shape[-1])
    count = np.maximum(np.bincount(flat_segments, minlength=node_count), 1).astype(np.float32)
    output = np.empty((node_count, flat_values.shape[1]), dtype=np.float32)
    for channel in range(flat_values.shape[1]):
        output[:, channel] = np.bincount(
            flat_segments,
            weights=flat_values[:, channel],
            minlength=node_count,
        ) / count
    return output


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norm, 1.0e-8)).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.shape[0] == 0:
        return values.copy()
    shifted = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -40.0, 0.0))
    return (exponent / np.maximum(np.sum(exponent, axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


def _validate_point_field(field: ProjectedPointField) -> None:
    count = int(np.asarray(field.point_ids).shape[0])
    expected = {
        "pixel_xy": (count, 2),
        "positions": (count, 3),
        "normals": (count, 3),
        "labels": (count,),
    }
    for name, shape in expected.items():
        if np.asarray(getattr(field, name)).shape != shape:
            raise ValueError(f"Projected point field {name} has shape {np.asarray(getattr(field, name)).shape}; expected {shape}.")


def _empty_stats(sparse_pixel_count: int, visible_point_count: int, total_pixels: int) -> dict[str, Any]:
    return {
        "sparsePixelCount": int(sparse_pixel_count),
        "densePixelCount": int(sparse_pixel_count),
        "denseCoverage": float(sparse_pixel_count / max(total_pixels, 1)),
        "coverageGain": 1.0 if sparse_pixel_count else 0.0,
        "regionCount": 0,
        "superpixelCount": 0,
        "seedSuperpixelCount": 0,
        "ambiguousSeedSuperpixelCount": 0,
        "acceptedSuperpixelCount": 0,
        "visiblePointCount": int(visible_point_count),
        "labeledVisiblePointCount": 0,
    }
