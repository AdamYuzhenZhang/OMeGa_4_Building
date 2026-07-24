"""Boundary-aware dense recovery from sparse projected region observations."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .dense_recovery_source import load_sparse_labels, sparse_source_availability, validate_sparse_source
from .sam2_video_propagation import PropagationFrame


@dataclass(frozen=True)
class ColmapDenseConfig:
    source_method_id: str
    source_run_dir: Path
    evidence_root: Path
    source_name: str = "COLMAP Tracks"
    target_superpixels: int = 10_000
    compactness: float = 8.0
    max_seed_distance_pixels: float = 160.0
    max_boundary_barrier: float = 1.25
    min_competitor_margin: float = 0.10


class ColmapDensePropagationSession:
    """Densifies a completed sparse projected-point candidate layer."""

    def __init__(self, config: ColmapDenseConfig) -> None:
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

            source_by_frame = {int(source.frame_id): source for source in run_input.sources}
            rows: list[dict[str, Any]] = []
            total = len(run_input.frames)
            for index, frame in enumerate(run_input.frames):
                sparse_labels = load_sparse_labels(
                    self.config.source_run_dir,
                    frame,
                    source_name=str(self.config.source_name),
                )
                source = source_by_frame.get(int(frame.frame_id))
                if source is not None:
                    dense_labels = np.asarray(source.labels, dtype=np.uint16).copy()
                    stats = {
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
                    dense_labels, stats = densify_sparse_colmap_labels(
                        rgb=rgb,
                        depth=depth,
                        depth_valid=depth_valid,
                        normal=normal,
                        normal_valid=normal_valid,
                        sparse_labels=sparse_labels,
                        config=self.config,
                    )
                    stats["recoveryPolicy"] = "rgb_depth_normal_superpixel_geodesic"
                sparse_coverage = float(np.count_nonzero(sparse_labels) / max(sparse_labels.size, 1))
                rows.append(
                    {
                        "frameId": int(frame.frame_id),
                        "sourceMethodId": str(self.config.source_method_id),
                        "sparseCoverage": sparse_coverage,
                        "coverage": float(stats["denseCoverage"]),
                        **stats,
                    }
                )
                if save_callback is not None:
                    save_callback(frame, dense_labels)
                if progress_callback is not None:
                    progress_callback(index + 1, total, int(frame.frame_id))
            return rows


def colmap_dense_availability(config: ColmapDenseConfig) -> tuple[bool, str]:
    root = config.evidence_root.expanduser().resolve()
    for label, path in (
        ("Depth Anything evidence", root / "depth_npz"),
        ("StableNormal evidence", root / "normal_npz"),
    ):
        if not path.is_dir() or not any(path.glob("*.npz")):
            return False, f"{label} is missing: {path}"
    return sparse_source_availability(
        config.source_run_dir,
        source_name=str(config.source_name),
    )


def densify_sparse_colmap_labels(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    normal: np.ndarray,
    normal_valid: np.ndarray,
    sparse_labels: np.ndarray,
    config: ColmapDenseConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Grow sparse labels over fine superpixels and abstain at weak assignments."""

    try:
        import cv2
        from scipy.spatial import cKDTree
        from skimage.color import rgb2lab
        from skimage.segmentation import slic
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Superpixel Geodesic requires OpenCV, SciPy, and scikit-image.") from exc

    sparse = np.asarray(sparse_labels, dtype=np.uint16)
    if sparse.ndim != 2:
        raise ValueError(f"Sparse projected labels must be a 2D array, got {sparse.shape}.")
    height, width = sparse.shape
    rgb_u8 = _resize_rgb(np.asarray(rgb), width, height)
    depth_f32, depth_ok = _resize_depth(cv2, depth, depth_valid, width, height)
    normal_f32, normal_ok = _resize_normal(cv2, normal, normal_valid, width, height)
    # RGB is always available; missing geometric evidence removes that cue from
    # boundary scoring but should not manufacture holes in an otherwise valid image.
    valid = np.ones((height, width), dtype=bool)

    sparse_pixel_count = int(np.count_nonzero(sparse))
    if sparse_pixel_count == 0:
        return sparse.copy(), {
            "sparsePixelCount": 0,
            "densePixelCount": 0,
            "denseCoverage": 0.0,
            "superpixelCount": 0,
            "seedSuperpixelCount": 0,
            "ambiguousSeedSuperpixelCount": 0,
            "assignedSuperpixelCount": 0,
        }

    requested = int(np.clip(config.target_superpixels, 500, max(500, sparse.size // 16)))
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
    adjacency = _boundary_graph(
        rgb2lab(rgb_u8.astype(np.float32) / 255.0).astype(np.float32),
        depth_f32,
        depth_ok,
        normal_f32,
        normal_ok,
        segments,
        node_count,
    )
    seed_labels, ambiguous_seed_count = _seed_superpixels(segments, sparse, node_count)
    seeded_nodes = np.flatnonzero(seed_labels > 0)
    if seeded_nodes.size == 0:
        return sparse.copy(), {
            "sparsePixelCount": sparse_pixel_count,
            "densePixelCount": sparse_pixel_count,
            "denseCoverage": float(sparse_pixel_count / max(sparse.size, 1)),
            "superpixelCount": node_count,
            "seedSuperpixelCount": 0,
            "ambiguousSeedSuperpixelCount": ambiguous_seed_count,
            "assignedSuperpixelCount": 0,
        }

    region_ids = np.unique(seed_labels[seeded_nodes]).astype(np.uint16)
    barrier_by_region = np.full((len(region_ids), node_count), np.inf, dtype=np.float32)
    distance_by_region = np.full((len(region_ids), node_count), np.inf, dtype=np.float32)
    all_seeded = seed_labels > 0
    for index, region_id in enumerate(region_ids.tolist()):
        region_seeds = np.flatnonzero(seed_labels == int(region_id))
        blocked = all_seeded & (seed_labels != int(region_id))
        barrier_by_region[index] = _minimax_distances(adjacency, region_seeds, blocked)
        tree = cKDTree(centroids[region_seeds])
        distance_by_region[index] = tree.query(centroids, k=1, workers=-1)[0].astype(np.float32)

    normalized_distance = distance_by_region / max(float(config.max_seed_distance_pixels), 1.0)
    score = barrier_by_region + 0.20 * normalized_distance
    best_index = np.argmin(score, axis=0)
    node_indices = np.arange(node_count)
    best_score = score[best_index, node_indices]
    best_barrier = barrier_by_region[best_index, node_indices]
    best_distance = distance_by_region[best_index, node_indices]
    if len(region_ids) > 1:
        second_score = np.partition(score, 1, axis=0)[1]
        margin = second_score - best_score
    else:
        margin = np.full(node_count, np.inf, dtype=np.float32)

    accepted = (
        np.isfinite(best_score)
        & (best_barrier <= float(config.max_boundary_barrier))
        & (best_distance <= float(config.max_seed_distance_pixels))
        & (margin >= float(config.min_competitor_margin))
    )
    node_labels = np.zeros(node_count, dtype=np.uint16)
    node_labels[accepted] = region_ids[best_index[accepted]]
    node_labels[seeded_nodes] = seed_labels[seeded_nodes]

    dense = node_labels[segments]
    dense[~valid] = 0
    dense[sparse > 0] = sparse[sparse > 0]
    dense_pixel_count = int(np.count_nonzero(dense))
    return dense.astype(np.uint16, copy=False), {
        "sparsePixelCount": sparse_pixel_count,
        "densePixelCount": dense_pixel_count,
        "denseCoverage": float(dense_pixel_count / max(dense.size, 1)),
        "coverageGain": float(dense_pixel_count / max(sparse_pixel_count, 1)),
        "superpixelCount": node_count,
        "seedSuperpixelCount": int(seeded_nodes.size),
        "ambiguousSeedSuperpixelCount": int(ambiguous_seed_count),
        "assignedSuperpixelCount": int(np.count_nonzero(node_labels)),
        "maxSeedDistancePixels": float(config.max_seed_distance_pixels),
        "maxBoundaryBarrier": float(config.max_boundary_barrier),
        "minCompetitorMargin": float(config.min_competitor_margin),
    }


def _load_frame_evidence(
    frame: PropagationFrame,
    evidence_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(frame.image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    stem = f"{int(frame.frame_id):06d}.npz"
    depth_path = evidence_root / "depth_npz" / stem
    normal_path = evidence_root / "normal_npz" / stem
    if not depth_path.is_file() or not normal_path.is_file():
        raise FileNotFoundError(
            f"Superpixel Geodesic needs Depth Anything and StableNormal evidence for frame {frame.frame_id}: "
            f"{depth_path}, {normal_path}"
        )
    with np.load(depth_path) as payload:
        depth = np.asarray(payload["depth_m"], dtype=np.float32)
        depth_valid = np.asarray(payload["valid"], dtype=bool)
    with np.load(normal_path) as payload:
        normal = np.asarray(payload["normal"], dtype=np.float32)
        normal_valid = np.asarray(payload["valid_mask"], dtype=bool)
    return rgb, depth, depth_valid, normal, normal_valid


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return rgb.astype(np.uint8, copy=False)
    return np.asarray(
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _resize_depth(cv2: Any, depth: np.ndarray, valid: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != (height, width):
        values = cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    mask &= np.isfinite(values) & (values > 0)
    finite = values[mask]
    fill = float(np.median(finite)) if finite.size else 1.0
    values = np.log(np.maximum(np.where(mask, values, fill), 1.0e-6)).astype(np.float32)
    return values, mask


def _resize_normal(cv2: Any, normal: np.ndarray, valid: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(normal, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if values.shape[:2] != (height, width):
        values = cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    mask &= np.isfinite(values).all(axis=-1) & (norm[..., 0] > 1.0e-6)
    values = values / np.maximum(norm, 1.0e-6)
    values[~mask] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return values.astype(np.float32, copy=False), mask


def _segment_centroids(segments: np.ndarray, node_count: int) -> np.ndarray:
    flat = segments.reshape(-1)
    yy, xx = np.indices(segments.shape, dtype=np.float32)
    count = np.maximum(np.bincount(flat, minlength=node_count), 1)
    x = np.bincount(flat, weights=xx.reshape(-1), minlength=node_count) / count
    y = np.bincount(flat, weights=yy.reshape(-1), minlength=node_count) / count
    return np.column_stack((x, y)).astype(np.float32)


def _boundary_graph(
    lab: np.ndarray,
    depth: np.ndarray,
    depth_valid: np.ndarray,
    normal: np.ndarray,
    normal_valid: np.ndarray,
    segments: np.ndarray,
    node_count: int,
) -> list[list[tuple[int, float]]]:
    keys: list[np.ndarray] = []
    color_values: list[np.ndarray] = []
    depth_values: list[np.ndarray] = []
    normal_values: list[np.ndarray] = []

    def collect(a: np.ndarray, b: np.ndarray, slice_a: tuple[slice, slice], slice_b: tuple[slice, slice]) -> None:
        changed = a != b
        if not np.any(changed):
            return
        lo = np.minimum(a[changed], b[changed]).astype(np.int64)
        hi = np.maximum(a[changed], b[changed]).astype(np.int64)
        keys.append(lo * node_count + hi)
        lab_a, lab_b = lab[slice_a][changed], lab[slice_b][changed]
        color_values.append(np.linalg.norm(lab_a - lab_b, axis=-1).astype(np.float32) / 100.0)
        depth_ok = depth_valid[slice_a][changed] & depth_valid[slice_b][changed]
        depth_delta = np.abs(depth[slice_a][changed] - depth[slice_b][changed]).astype(np.float32)
        depth_values.append(np.where(depth_ok, depth_delta, 0.0))
        normal_ok = normal_valid[slice_a][changed] & normal_valid[slice_b][changed]
        dot = np.sum(normal[slice_a][changed] * normal[slice_b][changed], axis=-1)
        normal_delta = np.arccos(np.clip(dot, -1.0, 1.0)).astype(np.float32) / np.pi
        normal_values.append(np.where(normal_ok, normal_delta, 0.0))

    collect(segments[:, :-1], segments[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None)))
    collect(segments[:-1, :], segments[1:, :], (slice(None, -1), slice(None)), (slice(1, None), slice(None)))
    if not keys:
        return [[] for _ in range(node_count)]

    key = np.concatenate(keys)
    color = np.concatenate(color_values)
    depth_delta = np.concatenate(depth_values)
    normal_delta = np.concatenate(normal_values)
    order = np.argsort(key, kind="stable")
    key = key[order]
    starts = np.concatenate(([0], np.flatnonzero(np.diff(key)) + 1))
    count = np.diff(np.append(starts, key.size)).astype(np.float32)

    def aggregate(values: np.ndarray) -> np.ndarray:
        ordered = values[order]
        mean = np.add.reduceat(ordered, starts) / np.maximum(count, 1.0)
        maximum = np.maximum.reduceat(ordered, starts)
        mixed = 0.70 * mean + 0.30 * maximum
        positive = mixed[np.isfinite(mixed) & (mixed > 1.0e-8)]
        scale = float(np.percentile(positive, 70.0)) if positive.size else 1.0
        return np.clip(mixed / max(scale, 1.0e-6), 0.0, 3.0).astype(np.float32)

    color_cost = aggregate(color)
    depth_cost = aggregate(depth_delta)
    normal_cost = aggregate(normal_delta)
    cost = 0.02 + 0.45 * color_cost + 0.30 * depth_cost + 0.25 * normal_cost
    unique_key = key[starts]
    left = unique_key // node_count
    right = unique_key % node_count
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    for u, v, value in zip(left.tolist(), right.tolist(), cost.tolist()):
        adjacency[int(u)].append((int(v), float(value)))
        adjacency[int(v)].append((int(u), float(value)))
    return adjacency


def _seed_superpixels(segments: np.ndarray, sparse: np.ndarray, node_count: int) -> tuple[np.ndarray, int]:
    positive = sparse > 0
    nodes = segments[positive].astype(np.int64)
    labels = sparse[positive].astype(np.int64)
    pairs, counts = np.unique(np.column_stack((nodes, labels)), axis=0, return_counts=True)
    votes: dict[int, list[tuple[int, int]]] = {}
    for (node, label), count in zip(pairs.tolist(), counts.tolist()):
        votes.setdefault(int(node), []).append((int(label), int(count)))

    seed_labels = np.zeros(node_count, dtype=np.uint16)
    ambiguous = 0
    for node, rows in votes.items():
        rows.sort(key=lambda item: (-item[1], item[0]))
        total = sum(count for _, count in rows)
        winner, winner_count = rows[0]
        if winner_count / max(total, 1) >= 0.75 and (len(rows) == 1 or winner_count > rows[1][1]):
            seed_labels[node] = np.uint16(winner)
        else:
            ambiguous += 1
    return seed_labels, ambiguous


def _minimax_distances(
    adjacency: list[list[tuple[int, float]]],
    seeds: np.ndarray,
    blocked: np.ndarray,
) -> np.ndarray:
    distance = np.full(len(adjacency), np.inf, dtype=np.float32)
    queue: list[tuple[float, int]] = []
    for raw_node in seeds.tolist():
        node = int(raw_node)
        distance[node] = 0.0
        heapq.heappush(queue, (0.0, node))
    while queue:
        current, node = heapq.heappop(queue)
        if current > float(distance[node]) + 1.0e-8:
            continue
        for neighbor, edge_cost in adjacency[node]:
            if blocked[neighbor]:
                continue
            candidate = max(current, float(edge_cost))
            if candidate + 1.0e-8 < float(distance[neighbor]):
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return distance
