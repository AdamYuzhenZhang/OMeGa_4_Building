"""Cached V2-Anchor correspondence and PCCS point-cycle utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PairMap:
    source_grid: tuple[int, int]
    target_grid: tuple[int, int]
    target_patch_by_source: np.ndarray
    cosine_by_source: np.ndarray


@dataclass(frozen=True)
class PointMatch:
    source_xy: np.ndarray
    target_xy: np.ndarray
    cosine: float
    foreground_patch_count: int


def compute_pair_map(source_features: np.ndarray, target_features: np.ndarray, device) -> PairMap:
    """Compute V2-Anchor's per-source-patch cosine argmax without retaining H."""
    import torch

    source = torch.from_numpy(np.asarray(source_features).copy()).to(device=device, dtype=torch.float32)
    target = torch.from_numpy(np.asarray(target_features).copy()).to(device=device, dtype=torch.float32)
    source_grid = (int(source.shape[1]), int(source.shape[2]))
    target_grid = (int(target.shape[1]), int(target.shape[2]))
    source_flat = source.reshape(source.shape[0], -1).transpose(0, 1).contiguous()
    target_flat = target.reshape(target.shape[0], -1).transpose(0, 1).contiguous()
    indices: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    chunk = 512
    with torch.inference_mode():
        for start in range(0, source_flat.shape[0], chunk):
            similarity = source_flat[start : start + chunk] @ target_flat.transpose(0, 1)
            value, index = similarity.max(dim=1)
            indices.append(index.cpu())
            values.append(value.cpu())
    del source, target, source_flat, target_flat
    return PairMap(
        source_grid=source_grid,
        target_grid=target_grid,
        target_patch_by_source=torch.cat(indices).numpy().astype(np.int32),
        cosine_by_source=torch.cat(values).numpy().astype(np.float32),
    )


def region_correspondence_score(
    pair: PairMap,
    source_mask: np.ndarray,
    *,
    foreground_threshold: float = 0.6,
) -> float:
    fractions = mask_patch_fractions(source_mask, pair.source_grid)
    selected = fractions.reshape(-1) > float(foreground_threshold)
    if not np.any(selected):
        return float("-inf")
    values = pair.cosine_by_source[selected]
    return float(np.quantile(values, 0.75))


def match_region(
    pair: PairMap,
    source_mask: np.ndarray,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    image_size: int = 768,
    patch_size: int = 16,
    foreground_threshold: float = 0.6,
    stratify_distance: float = 150.0,
    outlier_removal_ratio: float = 0.25,
    seed: int = 0,
) -> PointMatch | None:
    fractions = mask_patch_fractions(source_mask, pair.source_grid)
    selected = np.flatnonzero(fractions.reshape(-1) > float(foreground_threshold))
    if selected.size == 0:
        return None

    source_points = _patch_centers(selected, pair.source_grid, patch_size)
    target_indices = pair.target_patch_by_source[selected]
    target_points = _patch_centers(target_indices, pair.target_grid, patch_size)
    source_scale = float(source_hw[0]) / float(image_size)
    target_scale = float(target_hw[0]) / float(image_size)
    source_points *= source_scale
    target_points *= target_scale

    keep = _stratify_points(source_points, float(stratify_distance) ** 2)
    source_points = source_points[keep]
    target_points = target_points[keep]
    cosines = pair.cosine_by_source[selected][keep]
    if source_points.shape[0] == 0:
        return None

    if source_points.shape[0] > 3:
        distances = np.linalg.norm(source_points[:, None] - source_points[None, :], axis=2)
        average = distances.mean(axis=1)
        remove_count = int(source_points.shape[0] * float(outlier_removal_ratio))
        if remove_count > 0:
            inliers = np.argsort(average)[:-remove_count]
            source_points = source_points[inliers]
            target_points = target_points[inliers]
            cosines = cosines[inliers]
    if source_points.shape[0] == 0:
        return None

    # The release randomly reduces each object's stratified set to one point.
    # A pair/object seed makes that official sampling rule reproducible.
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    chosen = int(rng.integers(0, source_points.shape[0]))
    return PointMatch(
        source_xy=source_points[chosen].astype(np.float32),
        target_xy=target_points[chosen].astype(np.float32),
        cosine=float(cosines[chosen]),
        foreground_patch_count=int(selected.size),
    )


def cycle_error(
    reverse_pair: PairMap,
    target_mask: np.ndarray,
    reference_source_xy: np.ndarray,
    *,
    target_hw: tuple[int, int],
    source_hw: tuple[int, int],
    image_size: int = 768,
    patch_size: int = 16,
    foreground_threshold: float = 0.6,
    stratify_distance: float = 150.0,
    outlier_removal_ratio: float = 0.25,
    seed: int = 0,
) -> float:
    match = match_region(
        reverse_pair,
        target_mask,
        source_hw=target_hw,
        target_hw=source_hw,
        image_size=image_size,
        patch_size=patch_size,
        foreground_threshold=foreground_threshold,
        stratify_distance=stratify_distance,
        outlier_removal_ratio=outlier_removal_ratio,
        seed=seed,
    )
    if match is None:
        return float("inf")
    diagonal = float(np.hypot(source_hw[0], source_hw[1]))
    distance = float(np.linalg.norm(match.target_xy - np.asarray(reference_source_xy, dtype=np.float32)))
    return distance / max(diagonal, 1.0)


def mask_patch_fractions(mask: np.ndarray, grid: tuple[int, int], patch_size: int = 16) -> np.ndarray:
    """Match the release's resize followed by fixed 16x16 patch averaging."""
    values = (np.asarray(mask, dtype=bool).astype(np.uint8) * 255)
    target_height = int(grid[0]) * int(patch_size)
    target_width = int(grid[1]) * int(patch_size)
    resized = np.asarray(
        Image.fromarray(values, mode="L").resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.float32,
    ) / 255.0
    return resized.reshape(grid[0], patch_size, grid[1], patch_size).mean(axis=(1, 3))


def _patch_centers(indices: np.ndarray, grid: tuple[int, int], patch_size: int) -> np.ndarray:
    rows = indices // int(grid[1])
    columns = indices % int(grid[1])
    return np.stack((columns + 0.5, rows + 0.5), axis=1).astype(np.float32) * float(patch_size)


def _stratify_points(points_xy: np.ndarray, squared_threshold: float) -> np.ndarray:
    count = int(points_xy.shape[0])
    if count <= 1:
        return np.arange(count, dtype=np.int64)
    squared_norm = np.sum(points_xy * points_xy, axis=1)
    distances = squared_norm[:, None] + squared_norm[None, :] - 2.0 * points_xy @ points_xy.T
    maximum = float(squared_threshold) + 1.0
    np.fill_diagonal(distances, maximum)
    active = np.ones(count, dtype=bool)
    while True:
        neighbor = distances <= float(squared_threshold)
        counts = neighbor.sum(axis=1)
        if not np.any(counts):
            break
        index = int(np.argmax(counts))
        active[index] = False
        distances[index, :] = maximum
        distances[:, index] = maximum
    return np.flatnonzero(active)
