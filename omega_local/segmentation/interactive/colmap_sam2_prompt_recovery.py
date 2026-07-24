"""SAM2 dense recovery from persistent-region projected point prompts."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from .dense_recovery_source import load_sparse_labels, sparse_source_availability, validate_sparse_source
from .sam2_session import Sam2Config, Sam2Session
from .sam2_video_propagation import PropagationFrame


class Sam2PromptPredictor(Protocol):
    def predict_candidates(
        self,
        *,
        frame_id: int,
        image_rgb: np.ndarray,
        points_xy: np.ndarray,
        point_labels: np.ndarray,
        multimask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class ColmapSam2PromptConfig:
    source_method_id: str
    source_run_dir: Path
    sam2: Sam2Config
    source_name: str = "COLMAP Tracks"
    positive_prompt_count: int = 16
    negative_prompt_count: int = 16
    prompt_clearance_pixels: float = 6.0
    min_positive_prompts: int = 3
    min_positive_recall: float = 0.75
    min_negative_rejection: float = 0.75
    min_sam_score: float = 0.0
    max_mask_fraction: float = 0.90


class ColmapSam2PromptRecoverySession:
    """Turns sparse projected region observations into disjoint SAM2 masks."""

    def __init__(
        self,
        config: ColmapSam2PromptConfig,
        *,
        predictor: Sam2PromptPredictor | None = None,
    ) -> None:
        self.config = config
        self.predictor = predictor if predictor is not None else Sam2Session(config.sam2)
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
                sparse = load_sparse_labels(
                    self.config.source_run_dir,
                    frame,
                    source_name=str(self.config.source_name),
                )
                source = source_by_frame.get(int(frame.frame_id))
                if source is not None:
                    dense = np.asarray(source.labels, dtype=np.uint16).copy()
                    stats = {
                        "recoveryPolicy": "manual_anchor",
                        "regionCount": int(np.count_nonzero(np.unique(dense) > 0)),
                        "attemptedRegionCount": 0,
                        "acceptedRegionCount": 0,
                        "rejectedRegionCount": 0,
                    }
                else:
                    image = np.array(Image.open(frame.image_path).convert("RGB"), dtype=np.uint8, copy=True)
                    dense, stats = recover_with_sam2_point_prompts(
                        frame_id=int(frame.frame_id),
                        image_rgb=image,
                        sparse_labels=sparse,
                        predictor=self.predictor,
                        config=self.config,
                    )
                    stats["recoveryPolicy"] = (
                        f"sampro3d_inspired_{self.config.source_method_id}_points_to_sam2"
                    )

                sparse_pixels = int(np.count_nonzero(sparse))
                dense_pixels = int(np.count_nonzero(dense))
                rows.append(
                    {
                        "frameId": int(frame.frame_id),
                        "sourceMethodId": str(self.config.source_method_id),
                        "sparsePixelCount": sparse_pixels,
                        "densePixelCount": dense_pixels,
                        "sparseCoverage": float(sparse_pixels / max(sparse.size, 1)),
                        "coverage": float(dense_pixels / max(dense.size, 1)),
                        "coverageGain": float(dense_pixels / max(sparse_pixels, 1)),
                        **stats,
                    }
                )
                if save_callback is not None:
                    save_callback(frame, dense)
                if progress_callback is not None:
                    progress_callback(index + 1, total, int(frame.frame_id))
            return rows


def colmap_sam2_prompt_availability(config: ColmapSam2PromptConfig) -> tuple[bool, str]:
    for label, path in (
        ("SAM2 root", config.sam2.root),
        ("SAM2 build script", config.sam2.root / "sam2" / "build_sam.py"),
        ("SAM2 checkpoint", config.sam2.checkpoint),
    ):
        if not path.exists():
            return False, f"{label} is missing: {path}"
    return sparse_source_availability(config.source_run_dir, source_name=str(config.source_name))


def recover_with_sam2_point_prompts(
    *,
    frame_id: int,
    image_rgb: np.ndarray,
    sparse_labels: np.ndarray,
    predictor: Sam2PromptPredictor,
    config: ColmapSam2PromptConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover one disjoint label map from sparse positive and negative prompts."""

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("SAM2 Point Prompts requires OpenCV.") from exc

    sparse = np.asarray(sparse_labels, dtype=np.uint16)
    if sparse.ndim != 2:
        raise ValueError(f"Sparse projected labels must be a 2D array, got {sparse.shape}.")
    height, width = sparse.shape
    image = _resize_rgb(np.asarray(image_rgb), width, height)
    region_ids = np.unique(sparse)
    region_ids = region_ids[region_ids > 0].astype(np.uint16, copy=False)
    if region_ids.size == 0:
        return sparse.copy(), _empty_stats()

    candidates: list[dict[str, Any]] = []
    rejected = 0
    for region_id in region_ids.tolist():
        positive, negative = _region_prompts(
            cv2,
            sparse,
            int(region_id),
            positive_count=int(config.positive_prompt_count),
            negative_count=int(config.negative_prompt_count),
            clearance=float(config.prompt_clearance_pixels),
        )
        if positive.shape[0] < int(config.min_positive_prompts):
            rejected += 1
            continue
        points = np.concatenate((positive, negative), axis=0).astype(np.float32, copy=False)
        point_labels = np.concatenate(
            (
                np.ones(positive.shape[0], dtype=np.int32),
                np.zeros(negative.shape[0], dtype=np.int32),
            )
        )
        logits, sam_scores = predictor.predict_candidates(
            frame_id=int(frame_id),
            image_rgb=image,
            points_xy=points,
            point_labels=point_labels,
            multimask=True,
        )
        candidate = _select_candidate(
            logits,
            sam_scores,
            positive=positive,
            negative=negative,
            output_shape=(height, width),
            config=config,
        )
        if candidate is None:
            rejected += 1
            continue
        candidates.append({"regionId": int(region_id), **candidate})

    dense = _assemble_candidates(candidates, output_shape=(height, width))
    # Exact track observations are hard constraints and always override SAM2.
    dense[sparse > 0] = sparse[sparse > 0]
    stats = {
        "regionCount": int(region_ids.size),
        "attemptedRegionCount": int(region_ids.size),
        "acceptedRegionCount": int(len(candidates)),
        "rejectedRegionCount": int(rejected),
        "positivePromptCount": int(sum(row["positivePromptCount"] for row in candidates)),
        "negativePromptCount": int(sum(row["negativePromptCount"] for row in candidates)),
        "meanSamScore": float(np.mean([row["samScore"] for row in candidates])) if candidates else 0.0,
        "meanPromptAgreement": (
            float(np.mean([row["promptAgreement"] for row in candidates])) if candidates else 0.0
        ),
    }
    return dense.astype(np.uint16, copy=False), stats


def _region_prompts(
    cv2: Any,
    sparse: np.ndarray,
    region_id: int,
    *,
    positive_count: int,
    negative_count: int,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray]:
    support = sparse == int(region_id)
    competing = (sparse > 0) & ~support
    kernel = np.ones((3, 3), dtype=np.uint8)
    core = cv2.erode(support.astype(np.uint8), kernel, iterations=1) > 0
    if not np.any(core):
        core = support

    if np.any(competing) and clearance > 0:
        distance_from_competitor = cv2.distanceTransform((~competing).astype(np.uint8), cv2.DIST_L2, 3)
        cleared = core & (distance_from_competitor >= float(clearance))
        if np.count_nonzero(cleared) >= min(max(int(positive_count), 1), 3):
            core = cleared
    positive = _sample_coreset(_mask_points(core), max(int(positive_count), 1))

    negative = np.zeros((0, 2), dtype=np.float32)
    if np.any(competing) and int(negative_count) > 0:
        distance_from_support = cv2.distanceTransform((~support).astype(np.uint8), cv2.DIST_L2, 3)
        ys, xs = np.nonzero(competing)
        distances = distance_from_support[ys, xs]
        pool_size = min(xs.size, max(int(negative_count) * 128, int(negative_count)))
        nearest = np.argpartition(distances, pool_size - 1)[:pool_size] if pool_size < xs.size else np.arange(xs.size)
        pool = np.column_stack((xs[nearest], ys[nearest])).astype(np.float32, copy=False)
        negative = _sample_coreset(pool, int(negative_count))
    return positive, negative


def _select_candidate(
    logits: np.ndarray,
    sam_scores: np.ndarray,
    *,
    positive: np.ndarray,
    negative: np.ndarray,
    output_shape: tuple[int, int],
    config: ColmapSam2PromptConfig,
) -> dict[str, Any] | None:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    if values.ndim != 3:
        raise ValueError(f"Unexpected SAM2 candidate shape: {values.shape}")
    scores = np.asarray(sam_scores, dtype=np.float32).reshape(-1)
    best: dict[str, Any] | None = None
    for index, raw_logit in enumerate(values):
        logit = _resize_logit(raw_logit, output_shape)
        mask = logit > 0.0
        area_fraction = float(np.count_nonzero(mask) / max(mask.size, 1))
        pos_recall = _point_fraction(mask, positive, expected=True)
        neg_rejection = _point_fraction(mask, negative, expected=False)
        sam_score = float(scores[index]) if index < scores.size else 0.0
        if (
            pos_recall < float(config.min_positive_recall)
            or neg_rejection < float(config.min_negative_rejection)
            or sam_score < float(config.min_sam_score)
            or area_fraction > float(config.max_mask_fraction)
        ):
            continue
        prompt_agreement = 0.70 * pos_recall + 0.30 * neg_rejection
        quality = 0.60 * float(np.clip(sam_score, 0.0, 1.0)) + 0.40 * prompt_agreement
        row = {
            "logit": logit,
            "mask": mask,
            "quality": float(quality),
            "samScore": sam_score,
            "promptAgreement": float(prompt_agreement),
            "positiveRecall": float(pos_recall),
            "negativeRejection": float(neg_rejection),
            "positivePromptCount": int(positive.shape[0]),
            "negativePromptCount": int(negative.shape[0]),
            "areaFraction": area_fraction,
        }
        if best is None or float(row["quality"]) > float(best["quality"]):
            best = row
    return best


def _assemble_candidates(candidates: list[dict[str, Any]], *, output_shape: tuple[int, int]) -> np.ndarray:
    labels = np.zeros(output_shape, dtype=np.uint16)
    winner = np.full(output_shape, -np.inf, dtype=np.float32)
    for row in candidates:
        mask = np.asarray(row["mask"], dtype=bool)
        # Preserve SAM2's spatial confidence while using prompt agreement only
        # as a small cross-region calibration term.
        score = np.asarray(row["logit"], dtype=np.float32) + 0.20 * (float(row["quality"]) - 0.5)
        update = mask & (score > winner)
        labels[update] = np.uint16(int(row["regionId"]))
        winner[update] = score[update]
    return labels


def _sample_coreset(points_xy: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] <= int(count):
        return points
    if points.shape[0] > 20_000:
        keep = np.linspace(0, points.shape[0] - 1, 20_000, dtype=np.int64)
        points = points[keep]
    centroid = points.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))
    chosen = [first]
    min_distance = np.sum((points - points[first]) ** 2, axis=1)
    for _ in range(1, min(int(count), points.shape[0])):
        index = int(np.argmax(min_distance))
        chosen.append(index)
        distance = np.sum((points - points[index]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
    return points[np.asarray(chosen, dtype=np.int64)]


def _mask_points(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return np.column_stack((xs, ys)).astype(np.float32, copy=False)


def _point_fraction(mask: np.ndarray, points_xy: np.ndarray, *, expected: bool) -> float:
    if points_xy.shape[0] == 0:
        return 1.0
    x = np.clip(np.rint(points_xy[:, 0]).astype(np.int64), 0, mask.shape[1] - 1)
    y = np.clip(np.rint(points_xy[:, 1]).astype(np.int64), 0, mask.shape[0] - 1)
    values = mask[y, x]
    return float(np.mean(values if expected else ~values))


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if rgb.shape[:2] == (height, width):
        return rgb.astype(np.uint8, copy=False)
    return np.asarray(
        Image.fromarray(rgb.astype(np.uint8), mode="RGB").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def _resize_logit(logit: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    height, width = output_shape
    values = np.asarray(logit, dtype=np.float32)
    if values.shape == output_shape:
        return values
    return np.asarray(
        Image.fromarray(values, mode="F").resize((width, height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )


def _empty_stats() -> dict[str, Any]:
    return {
        "regionCount": 0,
        "attemptedRegionCount": 0,
        "acceptedRegionCount": 0,
        "rejectedRegionCount": 0,
        "positivePromptCount": 0,
        "negativePromptCount": 0,
        "meanSamScore": 0.0,
        "meanPromptAgreement": 0.0,
    }
