"""Keyframe selection for the interactive segmentation editor.

The selector is intentionally practical: it combines image reliability,
segmentation-boundary usefulness, proposal quality, view novelty, and a
pose-based view graph. It produces suggested user-edit anchors, not hard
segmentation decisions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class KeyframeFrame:
    frame_id: int
    image_name: str
    image_path: Path
    pose_world_from_camera: np.ndarray
    width: int = 0
    height: int = 0
    proposal_label_path: Path | None = None
    depth_path: Path | None = None
    depth_coverage: float = 0.0
    proposal_coverage: float = 0.0
    proposal_label_count: int = 0


@dataclass(frozen=True)
class KeyframeConfig:
    reliability_weight: float = 0.35
    boundary_weight: float = 0.30
    proposal_weight: float = 0.20
    rare_weight: float = 0.15
    coverage_gain_weight: float = 1.00
    intrinsic_gain_weight: float = 0.25
    redundancy_weight: float = 0.25
    target_keyframes: int = 0
    min_keyframes: int = 12
    max_keyframes: int = 32
    min_keyframe_gap: int = 5
    max_keyframe_gap: int = 40
    resize_long_edge: int = 320
    rotation_sigma_deg: float = 35.0
    translation_sigma_scale: float = 6.0
    graph_top_k: int = 12

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "KeyframeConfig":
        payload = payload or {}
        return cls(
            reliability_weight=float(payload.get("reliabilityWeight", cls.reliability_weight)),
            boundary_weight=float(payload.get("boundaryWeight", cls.boundary_weight)),
            proposal_weight=float(payload.get("proposalWeight", cls.proposal_weight)),
            rare_weight=float(payload.get("rareWeight", cls.rare_weight)),
            coverage_gain_weight=float(payload.get("coverageGainWeight", cls.coverage_gain_weight)),
            intrinsic_gain_weight=float(payload.get("intrinsicGainWeight", cls.intrinsic_gain_weight)),
            redundancy_weight=float(payload.get("redundancyWeight", cls.redundancy_weight)),
            target_keyframes=int(payload.get("targetKeyframes", cls.target_keyframes)),
            min_keyframes=int(payload.get("minKeyframes", cls.min_keyframes)),
            max_keyframes=int(payload.get("maxKeyframes", cls.max_keyframes)),
            min_keyframe_gap=int(payload.get("minKeyframeGap", cls.min_keyframe_gap)),
            max_keyframe_gap=int(payload.get("maxKeyframeGap", cls.max_keyframe_gap)),
            resize_long_edge=int(payload.get("resizeLongEdge", cls.resize_long_edge)),
            rotation_sigma_deg=float(payload.get("rotationSigmaDeg", cls.rotation_sigma_deg)),
            translation_sigma_scale=float(payload.get("translationSigmaScale", cls.translation_sigma_scale)),
            graph_top_k=int(payload.get("graphTopK", cls.graph_top_k)),
        ).normalized()

    def normalized(self) -> "KeyframeConfig":
        weights = np.asarray(
            [self.reliability_weight, self.boundary_weight, self.proposal_weight, self.rare_weight],
            dtype=np.float64,
        )
        weights = np.maximum(weights, 0.0)
        total = float(weights.sum())
        if total <= 1e-9:
            weights[:] = [0.35, 0.30, 0.20, 0.15]
            total = float(weights.sum())
        weights /= total
        min_count = max(int(self.min_keyframes), 1)
        max_count = max(int(self.max_keyframes), min_count)
        return KeyframeConfig(
            reliability_weight=float(weights[0]),
            boundary_weight=float(weights[1]),
            proposal_weight=float(weights[2]),
            rare_weight=float(weights[3]),
            coverage_gain_weight=max(float(self.coverage_gain_weight), 0.0),
            intrinsic_gain_weight=max(float(self.intrinsic_gain_weight), 0.0),
            redundancy_weight=max(float(self.redundancy_weight), 0.0),
            target_keyframes=max(int(self.target_keyframes), 0),
            min_keyframes=min_count,
            max_keyframes=max_count,
            min_keyframe_gap=max(int(self.min_keyframe_gap), 0),
            max_keyframe_gap=max(int(self.max_keyframe_gap), 0),
            resize_long_edge=max(int(self.resize_long_edge), 64),
            rotation_sigma_deg=max(float(self.rotation_sigma_deg), 1.0),
            translation_sigma_scale=max(float(self.translation_sigma_scale), 0.5),
            graph_top_k=max(int(self.graph_top_k), 1),
        )

    def target_count(self, n_frames: int) -> int:
        if n_frames <= 1:
            return max(n_frames, 1)
        if self.target_keyframes > 0:
            return max(1, min(int(self.target_keyframes), int(n_frames)))
        auto = int(round(float(n_frames) / 8.0))
        auto = max(int(self.min_keyframes), min(int(self.max_keyframes), auto))
        return max(1, min(auto, int(n_frames)))

    def to_json(self) -> dict[str, Any]:
        return {
            "reliabilityWeight": float(self.reliability_weight),
            "boundaryWeight": float(self.boundary_weight),
            "proposalWeight": float(self.proposal_weight),
            "rareWeight": float(self.rare_weight),
            "coverageGainWeight": float(self.coverage_gain_weight),
            "intrinsicGainWeight": float(self.intrinsic_gain_weight),
            "redundancyWeight": float(self.redundancy_weight),
            "targetKeyframes": int(self.target_keyframes),
            "minKeyframes": int(self.min_keyframes),
            "maxKeyframes": int(self.max_keyframes),
            "minKeyframeGap": int(self.min_keyframe_gap),
            "maxKeyframeGap": int(self.max_keyframe_gap),
            "resizeLongEdge": int(self.resize_long_edge),
            "rotationSigmaDeg": float(self.rotation_sigma_deg),
            "translationSigmaScale": float(self.translation_sigma_scale),
            "graphTopK": int(self.graph_top_k),
        }


@dataclass(frozen=True)
class FrameEvidence:
    rgb_edge_density: float
    rgb_laplacian_var: float
    luminance_mean: float
    luminance_std: float
    proposal_boundary_density: float
    proposal_coverage: float
    proposal_label_count: int
    proposal_entropy: float
    proposal_tiny_fraction: float
    depth_edge_density: float
    depth_coverage: float


@dataclass(frozen=True)
class SelectionResult:
    indices: list[int]
    reasons: dict[int, str]
    order: dict[int, int]
    coverage_gain: dict[int, float]
    redundancy: dict[int, float]
    selection_score: dict[int, float]


def detect_keyframes(
    frames: list[KeyframeFrame],
    *,
    output_path: Path,
    config: KeyframeConfig | None = None,
) -> dict[str, Any]:
    config = (config or KeyframeConfig()).normalized()
    if not frames:
        raise ValueError("Cannot detect keyframes without frames.")

    evidence = [_frame_evidence(frame, config.resize_long_edge) for frame in frames]
    scores = _score_frames(frames, evidence, config)
    graph_payload = _view_graph(frames, config)
    rare_visibility = _rare_visibility(graph_payload["graph"], config.graph_top_k)
    intrinsic = (
        float(config.reliability_weight) * scores["image_reliability"]
        + float(config.boundary_weight) * scores["boundary_usefulness"]
        + float(config.proposal_weight) * scores["proposal_usefulness"]
        + float(config.rare_weight) * rare_visibility
    )
    intrinsic = np.clip(intrinsic, 0.0, 1.0).astype(np.float32, copy=False)

    overlap_support = _overlap_support(graph_payload["graph"], config.graph_top_k)
    combined_score = np.clip(0.70 * intrinsic + 0.30 * overlap_support, 0.0, 1.0)
    selection = _select_greedy(
        graph_payload["graph"],
        intrinsic,
        combined_score,
        config,
    )

    frame_rows = _frame_rows(
        frames,
        evidence,
        selection,
        scores=scores,
        rare_visibility=rare_visibility,
        overlap_support=overlap_support,
        intrinsic_score=intrinsic,
        combined_score=combined_score,
        graph=graph_payload["graph"],
        top_k=config.graph_top_k,
    )
    selected = [row for row in frame_rows if row["isKeyframe"]]
    payload = {
        "stage": "interactive_keyframe_detection",
        "timestampUtc": datetime.now(timezone.utc).isoformat(),
        "method": (
            "Segmentation-aware greedy keyframe selection. Scores combine image reliability, "
            "SAM2/RGB/depth boundary usefulness, SAM2 proposal usefulness, pose novelty, "
            "and pose-graph coverage."
        ),
        "frameCount": int(len(frames)),
        "keyframeCount": int(len(selected)),
        "targetKeyframeCount": int(config.target_count(len(frames))),
        "keyframeIds": [int(row["frameId"]) for row in selected],
        "parameters": config.to_json(),
        "viewGraph": {
            "translationSigma": float(graph_payload["translation_sigma"]),
            "rotationSigmaDeg": float(config.rotation_sigma_deg),
            "meanEdgeWeight": float(np.mean(graph_payload["graph"])),
            "medianEdgeWeight": float(np.median(graph_payload["graph"])),
        },
        "scoreSummary": _score_summary(frame_rows),
        "frames": frame_rows,
        "output": str(output_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def read_keyframes(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _frame_evidence(frame: KeyframeFrame, resize_long_edge: int) -> FrameEvidence:
    rgb = _read_small_rgb(frame.image_path, resize_long_edge)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luminance = gray.astype(np.float32) / 255.0
    lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    edges = cv2.Canny(gray, 80, 160)
    rgb_edge_density = float(np.mean(edges > 0))

    proposal_boundary = 0.0
    proposal_coverage = max(float(frame.proposal_coverage), 0.0)
    proposal_label_count = max(int(frame.proposal_label_count), 0)
    proposal_entropy = 0.0
    proposal_tiny_fraction = 1.0 if proposal_label_count <= 0 else 0.0
    labels = _read_label_map(frame.proposal_label_path)
    if labels is not None and labels.size:
        labels_small = _resize_nearest(labels, (rgb.shape[1], rgb.shape[0]))
        proposal_boundary = _label_boundary_density(labels_small)
        positive = labels_small[labels_small > 0]
        if positive.size:
            proposal_coverage = float(positive.size) / float(labels_small.size)
            unique, counts = np.unique(positive, return_counts=True)
            proposal_label_count = int(unique.size)
            probs = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
            entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
            proposal_entropy = entropy / max(math.log(float(unique.size + 1)), 1e-6)
            tiny = counts < max(24, int(round(0.0005 * float(labels_small.size))))
            proposal_tiny_fraction = float(np.mean(tiny)) if tiny.size else 0.0

    depth_edge_density = 0.0
    depth_coverage = max(float(frame.depth_coverage), 0.0)
    depth = _read_depth(frame.depth_path)
    if depth is not None and depth.size:
        depth_small = _resize_float(depth, (rgb.shape[1], rgb.shape[0]))
        valid = np.isfinite(depth_small) & (depth_small > 0)
        if np.any(valid):
            depth_coverage = float(np.mean(valid))
            depth_edge_density = _depth_edge_density(depth_small, valid)

    return FrameEvidence(
        rgb_edge_density=rgb_edge_density,
        rgb_laplacian_var=lap_var,
        luminance_mean=float(np.mean(luminance)),
        luminance_std=float(np.std(luminance)),
        proposal_boundary_density=proposal_boundary,
        proposal_coverage=proposal_coverage,
        proposal_label_count=proposal_label_count,
        proposal_entropy=proposal_entropy,
        proposal_tiny_fraction=proposal_tiny_fraction,
        depth_edge_density=depth_edge_density,
        depth_coverage=depth_coverage,
    )


def _score_frames(
    frames: list[KeyframeFrame],
    evidence: list[FrameEvidence],
    config: KeyframeConfig,
) -> dict[str, np.ndarray]:
    del frames, config
    lap = np.asarray([item.rgb_laplacian_var for item in evidence], dtype=np.float32)
    contrast = np.asarray([item.luminance_std for item in evidence], dtype=np.float32)
    exposure = np.asarray([1.0 - min(abs(item.luminance_mean - 0.50) / 0.50, 1.0) for item in evidence], dtype=np.float32)
    depth_cov = np.asarray([item.depth_coverage for item in evidence], dtype=np.float32)
    sharpness_score = _robust_normalize(lap)
    contrast_score = _robust_normalize(contrast)
    image_reliability = np.clip(
        0.45 * sharpness_score + 0.25 * exposure + 0.20 * contrast_score + 0.10 * depth_cov,
        0.0,
        1.0,
    )

    rgb_edges = _robust_normalize(np.asarray([item.rgb_edge_density for item in evidence], dtype=np.float32))
    proposal_edges = _robust_normalize(np.asarray([item.proposal_boundary_density for item in evidence], dtype=np.float32))
    depth_edges = _robust_normalize(np.asarray([item.depth_edge_density for item in evidence], dtype=np.float32))
    boundary_usefulness = np.clip(0.45 * rgb_edges + 0.35 * proposal_edges + 0.20 * depth_edges, 0.0, 1.0)

    proposal_counts = np.asarray([item.proposal_label_count for item in evidence], dtype=np.float32)
    proposal_coverage = np.asarray([item.proposal_coverage for item in evidence], dtype=np.float32)
    proposal_entropy = np.asarray([item.proposal_entropy for item in evidence], dtype=np.float32)
    tiny_fraction = np.asarray([item.proposal_tiny_fraction for item in evidence], dtype=np.float32)
    count_score = np.sqrt(np.clip(proposal_counts / 80.0, 0.0, 1.0))
    proposal_usefulness = np.clip(
        proposal_coverage * (0.45 * count_score + 0.35 * proposal_entropy + 0.20 * (1.0 - tiny_fraction)),
        0.0,
        1.0,
    )

    return {
        "sharpnessScore": sharpness_score,
        "contrastScore": contrast_score,
        "exposureScore": exposure,
        "image_reliability": image_reliability.astype(np.float32, copy=False),
        "rgbEdgeScore": rgb_edges,
        "proposalBoundaryScore": proposal_edges,
        "depthBoundaryScore": depth_edges,
        "boundary_usefulness": boundary_usefulness.astype(np.float32, copy=False),
        "proposalCountScore": count_score.astype(np.float32, copy=False),
        "proposal_usefulness": proposal_usefulness.astype(np.float32, copy=False),
    }


def _view_graph(frames: list[KeyframeFrame], config: KeyframeConfig) -> dict[str, Any]:
    n_frames = len(frames)
    centers = np.zeros((n_frames, 3), dtype=np.float64)
    rotations = np.zeros((n_frames, 3, 3), dtype=np.float64)
    for index, frame in enumerate(frames):
        pose = np.asarray(frame.pose_world_from_camera, dtype=np.float64).reshape(4, 4)
        centers[index] = pose[:3, 3]
        rotations[index] = pose[:3, :3]

    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    positive = distances[distances > 1e-8]
    if positive.size:
        nearest = np.partition(distances + np.eye(n_frames) * 1e12, kth=min(3, n_frames - 1), axis=1)
        nearest_positive = nearest[:, min(3, n_frames - 1)]
        base = float(np.median(nearest_positive[np.isfinite(nearest_positive) & (nearest_positive < 1e11)]))
        if not np.isfinite(base) or base <= 1e-8:
            base = float(np.median(positive))
    else:
        base = 1.0
    translation_sigma = max(base * float(config.translation_sigma_scale), 1e-6)
    translation_similarity = np.exp(-np.square(distances / translation_sigma))

    rotation_angles = np.zeros((n_frames, n_frames), dtype=np.float64)
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            relative = rotations[i].T @ rotations[j]
            trace = float(np.trace(relative))
            cos_theta = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
            angle = math.degrees(math.acos(cos_theta))
            rotation_angles[i, j] = angle
            rotation_angles[j, i] = angle
    rotation_similarity = np.exp(-np.square(rotation_angles / float(config.rotation_sigma_deg)))

    graph = (translation_similarity * rotation_similarity).astype(np.float32)
    graph[~np.isfinite(graph)] = 0.0
    np.fill_diagonal(graph, 1.0)
    return {
        "graph": graph,
        "translation_sigma": float(translation_sigma),
        "rotation_angles": rotation_angles.astype(np.float32),
        "distances": distances.astype(np.float32),
    }


def _rare_visibility(graph: np.ndarray, top_k: int) -> np.ndarray:
    support = _overlap_support(graph, top_k)
    rare = 1.0 - support
    return np.clip(rare, 0.0, 1.0).astype(np.float32, copy=False)


def _overlap_support(graph: np.ndarray, top_k: int) -> np.ndarray:
    graph = np.asarray(graph, dtype=np.float32)
    if graph.shape[0] <= 1:
        return np.ones(graph.shape[0], dtype=np.float32)
    k = min(max(int(top_k), 1), graph.shape[0] - 1)
    masked = graph.copy()
    np.fill_diagonal(masked, -1.0)
    top = np.partition(masked, kth=masked.shape[1] - k, axis=1)[:, -k:]
    top = np.clip(top, 0.0, 1.0)
    return np.mean(top, axis=1).astype(np.float32, copy=False)


def _select_greedy(
    graph: np.ndarray,
    intrinsic: np.ndarray,
    combined_score: np.ndarray,
    config: KeyframeConfig,
) -> SelectionResult:
    n_frames = int(graph.shape[0])
    target = config.target_count(n_frames)
    row_sums = np.sum(graph, axis=1)
    coverage_scale = float(np.percentile(row_sums, 90)) if row_sums.size else 1.0
    coverage_scale = max(coverage_scale, 1e-6)

    selected: list[int] = []
    reasons: dict[int, str] = {}
    order: dict[int, int] = {}
    coverage_gain: dict[int, float] = {}
    redundancy: dict[int, float] = {}
    selection_score: dict[int, float] = {}
    covered = np.zeros(n_frames, dtype=np.float32)

    def add(index: int, reason: str, gain: float, red: float, score: float) -> None:
        nonlocal covered
        index = int(index)
        if index in order:
            return
        selected.append(index)
        order[index] = len(selected)
        reasons[index] = reason
        coverage_gain[index] = float(gain)
        redundancy[index] = float(red)
        selection_score[index] = float(score)
        covered = np.maximum(covered, graph[index])

    if n_frames == 1:
        add(0, "only_frame", 1.0, 0.0, 1.0)
        return SelectionResult(selected, reasons, order, coverage_gain, redundancy, selection_score)

    add(0, "first", float(np.sum(graph[0]) / coverage_scale), 0.0, float(combined_score[0]))
    if target > 1:
        add(n_frames - 1, "last", float(np.sum(graph[-1]) / coverage_scale), float(graph[-1, 0]), float(combined_score[-1]))

    while len(selected) < target:
        best_index = -1
        best_tuple: tuple[float, float, float, float] | None = None
        for index in range(n_frames):
            if index in order:
                continue
            if not _allowed_by_gap(index, selected, config.min_keyframe_gap) and len(selected) < n_frames - 1:
                continue
            new = np.maximum(graph[index] - covered, 0.0)
            gain = float(np.sum(new) / coverage_scale)
            red = float(np.max(graph[index, selected])) if selected else 0.0
            score = (
                float(config.coverage_gain_weight) * gain
                + float(config.intrinsic_gain_weight) * float(intrinsic[index])
                - float(config.redundancy_weight) * red
            )
            candidate_tuple = (score, gain, float(intrinsic[index]), -red)
            if best_tuple is None or candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_index = index
        if best_index < 0:
            break
        score, gain, intrinsic_value, neg_red = best_tuple or (0.0, 0.0, 0.0, 0.0)
        reason = _dominant_reason(best_index, intrinsic, combined_score, gain, -neg_red)
        if gain >= 0.35:
            reason = "coverage"
        elif intrinsic_value >= 0.70:
            reason = "high_intrinsic"
        add(best_index, reason, gain, -neg_red, score)

    selected = _insert_gap_indices(selected, config.max_keyframe_gap, config.min_keyframe_gap, combined_score, n_frames)
    for index in list(selected):
        if index not in order:
            red = float(np.max(graph[index, list(order)])) if order else 0.0
            gain = float(np.sum(np.maximum(graph[index] - covered, 0.0)) / coverage_scale)
            add(index, "sequence_gap", gain, red, float(combined_score[index]))

    selected = sorted(set(selected))
    return SelectionResult(selected, reasons, order, coverage_gain, redundancy, selection_score)


def _allowed_by_gap(index: int, selected: list[int], min_gap: int) -> bool:
    if min_gap <= 0:
        return True
    return all(abs(int(index) - int(other)) >= int(min_gap) for other in selected)


def _dominant_reason(index: int, intrinsic: np.ndarray, combined: np.ndarray, gain: float, redundancy: float) -> str:
    del intrinsic
    if gain > 0.35:
        return "coverage"
    if redundancy < 0.25 and combined[index] > 0.45:
        return "rare_view"
    return "intrinsic"


def _insert_gap_indices(
    indices: list[int],
    max_gap: int,
    min_gap: int,
    scores: np.ndarray,
    n_frames: int,
) -> list[int]:
    if max_gap <= 0:
        return indices
    out = sorted(set(indices))
    changed = True
    while changed:
        changed = False
        next_out: list[int] = []
        for start, end in zip(out[:-1], out[1:], strict=False):
            next_out.append(start)
            if end - start > max_gap:
                lo = min(end, start + max(int(min_gap), 1))
                hi = max(start, end - max(int(min_gap), 1))
                interior = np.arange(lo, hi + 1, dtype=np.int32) if lo <= hi else np.zeros(0, dtype=np.int32)
                if interior.size:
                    best = int(interior[int(np.argmax(scores[interior]))])
                else:
                    best = int(round((start + end) * 0.5))
                next_out.append(max(0, min(n_frames - 1, best)))
                changed = True
        next_out.append(out[-1])
        out = sorted(set(next_out))
    return out


def _frame_rows(
    frames: list[KeyframeFrame],
    evidence: list[FrameEvidence],
    selection: SelectionResult,
    *,
    scores: dict[str, np.ndarray],
    rare_visibility: np.ndarray,
    overlap_support: np.ndarray,
    intrinsic_score: np.ndarray,
    combined_score: np.ndarray,
    graph: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    key_set = set(int(index) for index in selection.indices)
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        item = evidence[index]
        neighbors = _top_neighbors(index, frames, graph, top_k=top_k)
        row = {
            "index": int(index),
            "frameId": int(frame.frame_id),
            "imageName": str(frame.image_name),
            "isKeyframe": bool(index in key_set),
            "reason": selection.reasons.get(index, "intermediate" if index not in key_set else "selected"),
            "selectionOrder": int(selection.order.get(index, 0)),
            "selectionScore": float(selection.selection_score.get(index, 0.0)),
            "coverageGain": float(selection.coverage_gain.get(index, 0.0)),
            "redundancy": float(selection.redundancy.get(index, 0.0)),
            "combinedScore": float(combined_score[index]),
            "intrinsicScore": float(intrinsic_score[index]),
            "imageReliability": float(scores["image_reliability"][index]),
            "boundaryUsefulness": float(scores["boundary_usefulness"][index]),
            "proposalUsefulness": float(scores["proposal_usefulness"][index]),
            "rareVisibility": float(rare_visibility[index]),
            "overlapSupport": float(overlap_support[index]),
            "sharpnessScore": float(scores["sharpnessScore"][index]),
            "exposureScore": float(scores["exposureScore"][index]),
            "contrastScore": float(scores["contrastScore"][index]),
            "rgbEdgeDensity": float(item.rgb_edge_density),
            "proposalBoundaryDensity": float(item.proposal_boundary_density),
            "depthEdgeDensity": float(item.depth_edge_density),
            "proposalCoverage": float(item.proposal_coverage),
            "proposalLabelCount": int(item.proposal_label_count),
            "proposalEntropy": float(item.proposal_entropy),
            "proposalTinyFraction": float(item.proposal_tiny_fraction),
            "depthCoverage": float(item.depth_coverage),
            "topOverlapNeighbors": neighbors,
        }
        rows.append(row)
    return rows


def _top_neighbors(index: int, frames: list[KeyframeFrame], graph: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
    if len(frames) <= 1:
        return []
    k = min(max(int(top_k), 1), len(frames) - 1)
    weights = graph[index].copy()
    weights[index] = -1.0
    order = np.argsort(weights)[::-1][:k]
    return [
        {
            "frameId": int(frames[int(other)].frame_id),
            "index": int(other),
            "weight": float(max(weights[int(other)], 0.0)),
        }
        for other in order
        if weights[int(other)] > 0.0
    ]


def _read_small_rgb(path: Path, resize_long_edge: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = float(resize_long_edge) / max(float(max(height, width)), 1.0)
    if scale < 1.0:
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    return rgb


def _read_label_map(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            labels = np.load(path)
        else:
            with Image.open(path) as image:
                labels = np.asarray(image)
    except Exception:
        return None
    if labels.ndim == 3:
        labels = labels[..., 0]
    if labels.ndim != 2:
        return None
    return labels.astype(np.int32, copy=False)


def _read_depth(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    try:
        if path.suffix.lower() == ".npy":
            depth = np.load(path)
        else:
            with Image.open(path) as image:
                depth = np.asarray(image)
    except Exception:
        return None
    if depth.ndim == 3:
        depth = depth[..., 0]
    if depth.ndim != 2:
        return None
    depth = depth.astype(np.float32, copy=False)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def _resize_nearest(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if values.shape[1] == width and values.shape[0] == height:
        return values
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_NEAREST)


def _resize_float(values: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if values.shape[1] == width and values.shape[0] == height:
        return values.astype(np.float32, copy=False)
    return cv2.resize(values.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_AREA)


def _label_boundary_density(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    horizontal &= (labels[:, 1:] > 0) | (labels[:, :-1] > 0)
    vertical &= (labels[1:, :] > 0) | (labels[:-1, :] > 0)
    count = int(np.count_nonzero(horizontal)) + int(np.count_nonzero(vertical))
    denom = max(int(horizontal.size) + int(vertical.size), 1)
    return float(count) / float(denom)


def _depth_edge_density(depth: np.ndarray, valid: np.ndarray) -> float:
    work = depth.astype(np.float32, copy=True)
    if not np.any(valid):
        return 0.0
    median = float(np.median(work[valid]))
    work[~valid] = median
    lo, hi = np.percentile(work[valid], [5, 95])
    scale = max(float(hi - lo), 1e-6)
    normalized = np.clip((work - float(lo)) / scale, 0.0, 1.0)
    gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    finite = grad[np.isfinite(grad)]
    if finite.size == 0:
        return 0.0
    threshold = float(np.percentile(finite, 90))
    if threshold <= 1e-8:
        return 0.0
    return float(np.mean((grad > threshold) & valid))


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size == 0:
        return out
    lo = float(np.percentile(positive, 5))
    hi = float(np.percentile(positive, 90))
    if not np.isfinite(hi) or hi <= lo + 1e-8:
        hi = float(np.max(positive))
        lo = 0.0
    if hi <= lo + 1e-8:
        return out
    out = np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)
    out[~np.isfinite(out)] = 0.0
    return out


def _score_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    keys = [
        "combinedScore",
        "intrinsicScore",
        "imageReliability",
        "boundaryUsefulness",
        "proposalUsefulness",
        "rareVisibility",
        "overlapSupport",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        out[key] = {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }
    return out
