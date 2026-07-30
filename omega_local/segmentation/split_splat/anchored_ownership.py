"""Complete persistent-region ownership for a shared Gaussian scene."""

from __future__ import annotations

from typing import Any

import numpy as np


PROVENANCE_UNKNOWN = np.uint8(0)
PROVENANCE_PROPAGATED = np.uint8(1)
PROVENANCE_MANUAL = np.uint8(2)
PROVENANCE_GEOMETRY = np.uint8(3)

_COMPLETION_NEIGHBORS = 4
_COMPLETION_CHUNK_SIZE = 100_000
_MIN_SEED_WEIGHT = 0.05
_MIN_DISTANCE = 1.0e-5


def assign_point_labels(
    manual_votes: np.ndarray,
    propagated_votes: np.ndarray,
    region_ids: np.ndarray,
    *,
    manual_frame_weight: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Assign every directly observed Gaussian; uncertainty never rejects it."""
    if manual_votes.shape != propagated_votes.shape:
        raise ValueError("Manual and propagated point-vote arrays must align.")
    if manual_votes.ndim != 2 or manual_votes.shape[1] != region_ids.size:
        raise ValueError("Point votes and region IDs do not align.")
    if region_ids.size == 0:
        raise ValueError("At least one persistent region is required.")

    manual = manual_votes.astype(np.float32)
    propagated = propagated_votes.astype(np.float32)
    combined = manual * float(manual_frame_weight) + propagated
    manual_total = manual.sum(axis=1)
    combined_total = combined.sum(axis=1)

    manual_top = manual.max(axis=1)
    manual_winners = manual == manual_top[:, None]
    manual_ties = (
        np.count_nonzero(manual_winners, axis=1) > 1
    ) & (manual_top > 0)

    # Completed manual frames define identity. Propagated evidence only breaks
    # ties between equally supported manual labels.
    manual_candidates = np.where(manual_winners, combined, -1.0)
    manual_choice = np.argmax(manual_candidates, axis=1)
    propagated_choice = np.argmax(propagated, axis=1)
    manual_observed = manual_total > 0
    chosen = np.where(manual_observed, manual_choice, propagated_choice)

    selected = combined[np.arange(combined.shape[0]), chosen]
    second = (
        np.partition(combined, -2, axis=1)[:, -2]
        if combined.shape[1] > 1
        else np.zeros_like(selected)
    )
    confidence = np.divide(
        selected,
        combined_total,
        out=np.zeros_like(selected),
        where=combined_total > 0,
    )
    margin = np.divide(
        selected - second,
        np.maximum(selected, 1.0),
        out=np.zeros_like(selected),
        where=selected > 0,
    )

    observed = combined_total > 0
    labels = np.zeros(manual.shape[0], dtype=np.uint16)
    labels[observed] = region_ids[chosen[observed]]
    provenance = np.full(
        manual.shape[0],
        PROVENANCE_UNKNOWN,
        dtype=np.uint8,
    )
    provenance[observed & ~manual_observed] = PROVENANCE_PROPAGATED
    provenance[manual_observed] = PROVENANCE_MANUAL

    selected_ties = (
        np.count_nonzero(combined == selected[:, None], axis=1) > 1
    ) & observed
    return labels, confidence.astype(np.float32), provenance, {
        "directlyObservedPointCount": int(np.count_nonzero(observed)),
        "unobservedPointCount": int(np.count_nonzero(~observed)),
        "manualTiePointCount": int(np.count_nonzero(manual_ties)),
        "combinedTiePointCount": int(np.count_nonzero(selected_ties)),
        "lowConfidenceDirectPointCount": int(
            np.count_nonzero(observed & (confidence < 0.55))
        ),
        "lowMarginDirectPointCount": int(
            np.count_nonzero(observed & (margin < 0.10))
        ),
    }


def complete_point_labels(
    points: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    provenance: np.ndarray,
    region_ids: np.ndarray,
    *,
    neighbors: int = _COMPLETION_NEIGHBORS,
    chunk_size: int = _COMPLETION_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill no-evidence Gaussians by local consensus of direct 3D labels."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint16).copy()
    confidence = np.asarray(confidence, dtype=np.float32).copy()
    provenance = np.asarray(provenance, dtype=np.uint8).copy()
    region_ids = np.asarray(region_ids, dtype=np.uint16)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Gaussian means must have shape (N, 3).")
    if any(values.shape != (points.shape[0],) for values in (
        labels,
        confidence,
        provenance,
    )):
        raise ValueError("Labels, confidence, and provenance must align to points.")

    seed_ids = np.flatnonzero(labels > 0)
    if seed_ids.size == 0:
        raise ValueError("Cannot complete Gaussian ownership without direct labels.")
    unknown_ids = np.flatnonzero(labels == 0)
    seeded_region_ids = np.unique(labels[seed_ids])
    missing_seed_ids = sorted(
        set(int(value) for value in region_ids).difference(
            int(value) for value in seeded_region_ids
        )
    )
    if unknown_ids.size == 0:
        return labels, confidence, provenance, {
            "policy": "local_weighted_knn",
            "completedPointCount": 0,
            "regionIdsWithoutDirectSeeds": missing_seed_ids,
            "neighborCount": 0,
            "distanceScale": 0.0,
            "nearestDistanceQuantiles": {},
        }

    tree = cKDTree(points[seed_ids].astype(np.float64, copy=False))
    neighbor_count = min(max(int(neighbors), 1), int(seed_ids.size))
    chunk_size = max(int(chunk_size), 1)
    distance_scale = _estimate_distance_scale(tree, points[seed_ids])
    region_columns = {
        int(region_id): index
        for index, region_id in enumerate(region_ids)
    }
    nearest_distances: list[np.ndarray] = []

    for start in range(0, unknown_ids.size, chunk_size):
        target_ids = unknown_ids[start : start + chunk_size]
        distances, neighbor_rows = tree.query(
            points[target_ids].astype(np.float64, copy=False),
            k=neighbor_count,
            workers=-1,
        )
        if neighbor_count == 1:
            distances = distances[:, None]
            neighbor_rows = neighbor_rows[:, None]
        neighbor_ids = seed_ids[np.asarray(neighbor_rows, dtype=np.int64)]
        neighbor_labels = labels[neighbor_ids]
        neighbor_confidence = np.maximum(
            confidence[neighbor_ids],
            _MIN_SEED_WEIGHT,
        )
        weights = neighbor_confidence / np.maximum(
            np.asarray(distances, dtype=np.float32),
            _MIN_DISTANCE,
        )

        scores = np.zeros(
            (target_ids.size, region_ids.size),
            dtype=np.float32,
        )
        rows = np.repeat(np.arange(target_ids.size), neighbor_count)
        columns = np.fromiter(
            (
                region_columns[int(value)]
                for value in neighbor_labels.reshape(-1)
            ),
            dtype=np.int64,
            count=neighbor_labels.size,
        )
        np.add.at(scores, (rows, columns), weights.reshape(-1))
        chosen = np.argmax(scores, axis=1)
        selected = scores[np.arange(target_ids.size), chosen]
        total = scores.sum(axis=1)
        agreement = np.divide(
            selected,
            total,
            out=np.zeros_like(selected),
            where=total > 0,
        )
        nearest = np.asarray(distances[:, 0], dtype=np.float32)
        locality = np.exp(
            -nearest / max(distance_scale * 4.0, _MIN_DISTANCE)
        ).astype(np.float32)

        labels[target_ids] = region_ids[chosen]
        confidence[target_ids] = agreement * locality
        provenance[target_ids] = PROVENANCE_GEOMETRY
        nearest_distances.append(nearest)

    distances = np.concatenate(nearest_distances)
    quantiles = {
        str(value): float(np.quantile(distances, value))
        for value in (0.5, 0.9, 0.95, 0.99, 1.0)
    }
    return labels, confidence, provenance, {
        "policy": "local_weighted_knn",
        "completedPointCount": int(unknown_ids.size),
        "regionIdsWithoutDirectSeeds": missing_seed_ids,
        "neighborCount": int(neighbor_count),
        "distanceScale": float(distance_scale),
        "nearestDistanceQuantiles": quantiles,
    }


def _estimate_distance_scale(tree: Any, seed_points: np.ndarray) -> float:
    if seed_points.shape[0] < 2:
        return 1.0
    sample_count = min(int(seed_points.shape[0]), 50_000)
    if sample_count == seed_points.shape[0]:
        sample = seed_points
    else:
        sample_ids = np.linspace(
            0,
            seed_points.shape[0] - 1,
            sample_count,
            dtype=np.int64,
        )
        sample = seed_points[sample_ids]
    distances, _ = tree.query(
        sample.astype(np.float64, copy=False),
        k=2,
        workers=-1,
    )
    positive = np.asarray(distances[:, 1], dtype=np.float32)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size == 0:
        return 1.0
    return max(float(np.quantile(positive, 0.95)), _MIN_DISTANCE)
