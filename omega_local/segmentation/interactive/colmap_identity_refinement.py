"""Conservative persistent-ID correction for propagated masks using COLMAP tracks."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import cv2
import numpy as np

from .colmap_track_propagation import (
    ColmapTrackConfig,
    _label_points_with_confidence_from_anchors,
    _match_colmap_images,
    _read_pixel_transforms,
    _read_reconstruction,
    _scaled_observations,
    _valid_point_mask,
    colmap_track_availability,
)
from .dense_recovery_source import (
    load_sparse_labels,
    sparse_source_availability,
    validate_sparse_source,
)
from .sam2_video_propagation import PropagationFrame

if TYPE_CHECKING:
    from .propagation_backends import PropagationRunInput


@dataclass(frozen=True)
class ColmapIdentityRefinementConfig:
    colmap: ColmapTrackConfig
    source_method_id: str
    source_run_dir: Path
    source_name: str = "SAM2 Video"
    min_component_track_count: int = 3
    min_target_track_count: int = 3
    min_total_vote_weight: float = 1.5
    min_target_fraction: float = 0.72
    min_target_margin: float = 0.28
    max_pixels_per_target_track: int = 5000
    track_length_saturation: int = 8


class ColmapIdentityRefinementSession:
    """Keeps source mask pixels fixed and changes only confident component IDs."""

    def __init__(self, config: ColmapIdentityRefinementConfig) -> None:
        self.config = config
        self._run_lock = threading.Lock()

    def refine_from_source(
        self,
        *,
        run_input: "PropagationRunInput",
        progress_callback: Callable[[int, int, int], None] | None = None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None = None,
    ) -> list[dict[str, Any]]:
        validate_sparse_source(
            self.config.source_run_dir,
            source_name=self.config.source_name,
            expected_fingerprint=run_input.fingerprint,
        )

        with self._run_lock:
            reconstruction = _read_reconstruction(self.config.colmap.model_path)
            frames = list(run_input.frames)
            image_by_frame_id = _match_colmap_images(reconstruction, frames)
            pixel_transforms = _read_pixel_transforms(self.config.colmap.pixel_transform_summary)
            valid_points = _valid_point_mask(
                reconstruction,
                min_track_length=int(self.config.colmap.min_track_length),
                max_reprojection_error=float(self.config.colmap.max_reprojection_error),
            )
            point_labels, anchor_confidence, vote_summary = (
                _label_points_with_confidence_from_anchors(
                    list(run_input.sources),
                    frames,
                    image_by_frame_id,
                    valid_points,
                    pixel_transforms,
                )
            )
            if not np.any(point_labels > 0):
                raise ValueError(
                    "No reliable COLMAP tracks intersect the complete persistent-region anchors."
                )
            point_weights = _track_weights(
                reconstruction,
                anchor_confidence,
                max_reprojection_error=float(self.config.colmap.max_reprojection_error),
                track_length_saturation=int(self.config.track_length_saturation),
            )

            rows: list[dict[str, Any]] = []
            total = len(frames)
            for index, frame in enumerate(frames):
                source_labels = load_sparse_labels(
                    self.config.source_run_dir,
                    frame,
                    source_name=self.config.source_name,
                )
                if int(frame.frame_id) in run_input.anchor_labels_by_frame:
                    refined = run_input.anchor_labels_by_frame[int(frame.frame_id)].copy()
                    diagnostics = _empty_frame_diagnostics(source_labels)
                    diagnostics["anchorPreserved"] = True
                else:
                    image = image_by_frame_id[int(frame.frame_id)]
                    point_ids, pixel_xy = _scaled_observations(
                        image,
                        frame,
                        valid_points,
                        pixel_transforms=pixel_transforms,
                    )
                    refined, diagnostics = _relabel_components(
                        source_labels,
                        pixel_xy,
                        point_labels[point_ids],
                        point_weights[point_ids],
                        self.config,
                    )
                    if not np.array_equal(refined > 0, source_labels > 0):
                        raise RuntimeError(
                            f"Identity refinement changed the mask footprint for frame {frame.frame_id}."
                        )
                    diagnostics["observedTrackCount"] = int(point_ids.size)
                    diagnostics["labeledObservedTrackCount"] = int(
                        np.count_nonzero(point_labels[point_ids] > 0)
                    )

                if save_callback is not None:
                    save_callback(frame, refined)
                rows.append(
                    {
                        "frameId": int(frame.frame_id),
                        **diagnostics,
                    }
                )
                if progress_callback is not None:
                    progress_callback(index + 1, total, int(frame.frame_id))

            aggregate_keys = (
                "componentCount",
                "supportedComponentCount",
                "agreementComponentCount",
                "relabeledComponentCount",
                "relabeledPixelCount",
                "lowSupportComponentCount",
                "ambiguousComponentCount",
                "insufficientDensityComponentCount",
            )
            run_summary = {
                key: int(sum(int(row.get(key, 0)) for row in rows))
                for key in aggregate_keys
            }
            run_summary["changedFrameCount"] = int(
                sum(int(row.get("relabeledComponentCount", 0)) > 0 for row in rows)
            )
            rows.append(
                {
                    "runSummary": run_summary,
                    "trackVoteSummary": vote_summary,
                }
            )
            return rows


def colmap_identity_refinement_availability(
    config: ColmapIdentityRefinementConfig,
) -> tuple[bool, str]:
    available, reason = colmap_track_availability(config.colmap)
    if not available:
        return available, reason
    return sparse_source_availability(
        config.source_run_dir,
        source_name=config.source_name,
    )


def _track_weights(
    reconstruction,
    anchor_confidence: np.ndarray,
    *,
    max_reprojection_error: float,
    track_length_saturation: int,
) -> np.ndarray:
    if track_length_saturation < 2:
        raise ValueError("Track-length saturation must be at least 2 observations.")
    weights = np.zeros(anchor_confidence.shape[0], dtype=np.float32)
    denominator = math.log1p(float(track_length_saturation))
    for point_id, point in reconstruction.points3D.items():
        pid = int(point_id)
        if pid >= weights.shape[0] or anchor_confidence[pid] <= 0:
            continue
        error = max(0.0, float(point.error))
        error_weight = math.exp(-0.5 * (error / max(max_reprojection_error, 1e-6)) ** 2)
        length = max(1, int(point.track.length()))
        length_weight = min(1.0, math.log1p(float(length)) / denominator)
        weights[pid] = np.float32(
            float(anchor_confidence[pid]) * error_weight * (0.5 + 0.5 * length_weight)
        )
    return weights


def _relabel_components(
    source_labels: np.ndarray,
    point_pixels: np.ndarray,
    point_region_ids: np.ndarray,
    point_weights: np.ndarray,
    config: ColmapIdentityRefinementConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Relabel whole connected components only when track consensus is decisive."""
    source = np.asarray(source_labels, dtype=np.uint16)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2D source label map, got {source.shape}.")
    pixels = np.asarray(point_pixels, dtype=np.int32).reshape(-1, 2)
    region_ids = np.asarray(point_region_ids, dtype=np.uint16).reshape(-1)
    weights = np.asarray(point_weights, dtype=np.float32).reshape(-1)
    if pixels.shape[0] != region_ids.size or region_ids.size != weights.size:
        raise ValueError("COLMAP observation pixels, labels, and weights must have equal lengths.")

    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < source.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < source.shape[0])
        & (region_ids > 0)
        & np.isfinite(weights)
        & (weights > 0)
    )
    pixels = pixels[inside]
    region_ids = region_ids[inside]
    weights = weights[inside]

    output = source.copy()
    diagnostics = _empty_frame_diagnostics(source)
    changes: list[dict[str, Any]] = []
    for source_region_id in np.unique(source).tolist():
        source_region_id = int(source_region_id)
        if source_region_id <= 0:
            continue
        component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
            (source == source_region_id).astype(np.uint8),
            connectivity=8,
        )
        for component_id in range(1, int(component_count)):
            diagnostics["componentCount"] += 1
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if pixels.size == 0:
                diagnostics["lowSupportComponentCount"] += 1
                continue
            member = component_map[pixels[:, 1], pixels[:, 0]] == component_id
            component_regions = region_ids[member]
            component_weights = weights[member]
            support_count = int(component_regions.size)
            total_weight = float(np.sum(component_weights, dtype=np.float64))
            if (
                support_count < int(config.min_component_track_count)
                or total_weight < float(config.min_total_vote_weight)
            ):
                diagnostics["lowSupportComponentCount"] += 1
                continue

            candidates, inverse = np.unique(component_regions, return_inverse=True)
            vote_weights = np.bincount(
                inverse,
                weights=component_weights.astype(np.float64),
                minlength=candidates.size,
            )
            vote_counts = np.bincount(inverse, minlength=candidates.size)
            order = np.argsort(vote_weights)[::-1]
            top_index = int(order[0])
            top_region = int(candidates[top_index])
            top_weight = float(vote_weights[top_index])
            second_weight = float(vote_weights[int(order[1])]) if order.size > 1 else 0.0
            fraction = top_weight / max(total_weight, 1e-8)
            margin = (top_weight - second_weight) / max(total_weight, 1e-8)
            top_count = int(vote_counts[top_index])
            required_target_count = max(
                int(config.min_target_track_count),
                int(math.ceil(area / max(int(config.max_pixels_per_target_track), 1))),
            )
            diagnostics["supportedComponentCount"] += 1

            if top_region == source_region_id:
                diagnostics["agreementComponentCount"] += 1
                continue
            if (
                top_count < required_target_count
                or fraction < float(config.min_target_fraction)
                or margin < float(config.min_target_margin)
            ):
                if top_count < required_target_count:
                    diagnostics["insufficientDensityComponentCount"] += 1
                diagnostics["ambiguousComponentCount"] += 1
                continue

            component_mask = component_map == component_id
            output[component_mask] = np.uint16(top_region)
            diagnostics["relabeledComponentCount"] += 1
            diagnostics["relabeledPixelCount"] += area
            changes.append(
                {
                    "sourceRegionId": source_region_id,
                    "targetRegionId": top_region,
                    "areaPixels": area,
                    "supportTrackCount": support_count,
                    "targetTrackCount": top_count,
                    "requiredTargetTrackCount": required_target_count,
                    "targetFraction": fraction,
                    "targetMargin": margin,
                    "voteWeight": total_weight,
                }
            )

    diagnostics["changes"] = changes
    diagnostics["sourceCoverage"] = float(np.count_nonzero(source) / max(source.size, 1))
    diagnostics["outputCoverage"] = float(np.count_nonzero(output) / max(output.size, 1))
    return output, diagnostics


def _empty_frame_diagnostics(source_labels: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source_labels)
    return {
        "anchorPreserved": False,
        "componentCount": 0,
        "supportedComponentCount": 0,
        "agreementComponentCount": 0,
        "relabeledComponentCount": 0,
        "relabeledPixelCount": 0,
        "lowSupportComponentCount": 0,
        "ambiguousComponentCount": 0,
        "insufficientDensityComponentCount": 0,
        "sourceCoverage": float(np.count_nonzero(source) / max(source.size, 1)),
        "outputCoverage": float(np.count_nonzero(source) / max(source.size, 1)),
        "changes": [],
    }
