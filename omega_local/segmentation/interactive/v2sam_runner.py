"""Isolated V2-SAM source-target propagation runner.

The three experts use the official release. PCCS follows the paper's point-cycle
description because the release has no standalone selector runner.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dinov3_runner import prepare_dinov3_cache
from .v2sam_cache import DinoFeatureCache, DinoFrame
from .v2sam_correspondence import (
    PairMap,
    compute_pair_map,
    cycle_error,
    match_region,
    region_correspondence_score,
)
from .v2sam_experts import ExpertConfig, V2SamExperts


EXPERT_ORDER = ("anchor", "visual", "fusion")


@dataclass(frozen=True)
class FrameRecord:
    local_index: int
    frame_id: int
    image_path: Path
    width: int
    height: int

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.height), int(self.width)


@dataclass(frozen=True)
class SourceRecord:
    frame: FrameRecord
    labels: np.ndarray
    region_ids: tuple[int, ...]


@dataclass(frozen=True)
class Candidate:
    region_id: int
    source_frame_id: int
    expert: str
    probability: np.ndarray
    mask: np.ndarray
    cycle_error: float
    source_similarity: float
    weight: float


def _resolve_device(value: str):
    import torch

    key = str(value or "auto").strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(key)


def _load_frames(path: Path) -> list[FrameRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload) if isinstance(payload, dict) else payload
    frames = [
        FrameRecord(
            local_index=int(row["localIndex"]),
            frame_id=int(row["frameId"]),
            image_path=Path(str(row["imagePath"])).expanduser().resolve(),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        for row in rows
    ]
    frames.sort(key=lambda frame: frame.local_index)
    if [frame.local_index for frame in frames] != list(range(len(frames))):
        raise ValueError("V2-SAM manifest local indices must be contiguous and zero-based.")
    return frames


def _load_sources(frames: list[FrameRecord], anchors_dir: Path, source_indices: list[int]) -> list[SourceRecord]:
    sources: list[SourceRecord] = []
    for local_index in source_indices:
        if local_index < 0 or local_index >= len(frames):
            raise ValueError(f"Source index {local_index} is outside the staged frame sequence.")
        frame = frames[local_index]
        path = anchors_dir / f"{local_index:05d}.npy"
        labels = np.load(path).astype(np.uint16, copy=False)
        if labels.shape != frame.hw:
            raise ValueError(f"Anchor map {path} has shape {labels.shape}; expected {frame.hw}.")
        region_ids = tuple(int(value) for value in np.unique(labels).tolist() if int(value) > 0)
        if region_ids:
            sources.append(SourceRecord(frame=frame, labels=labels.copy(), region_ids=region_ids))
    if not sources:
        raise ValueError("V2-SAM received no non-empty manual anchor maps.")
    return sources


def _load_rgb(frame: FrameRecord) -> np.ndarray:
    with Image.open(frame.image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.shape[:2] != frame.hw:
        rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB").resize((frame.width, frame.height), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        ).copy()
    return rgb


def _pair_candidates(
    *,
    source: SourceRecord,
    target: FrameRecord,
    region_ids: list[int],
    forward_pair: PairMap,
    reverse_pair: PairMap,
    experts: V2SamExperts,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    mask_threshold: float,
    cycle_temperature: float,
    foreground_threshold: float,
    seed: int,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    valid_regions: list[int] = []
    source_masks: list[np.ndarray] = []
    forward_matches = []
    for region_id in region_ids:
        match = match_region(
            forward_pair,
            source.labels == region_id,
            source_hw=source.frame.hw,
            target_hw=target.hw,
            foreground_threshold=foreground_threshold,
            seed=_mix_seed(seed, source.frame.frame_id, target.frame_id, region_id, 0),
        )
        if match is None:
            continue
        valid_regions.append(region_id)
        source_masks.append(source.labels == region_id)
        forward_matches.append(match)
    if not valid_regions:
        return [], []

    predictions = experts.predict(
        source_rgb=source_rgb,
        source_masks=np.stack(source_masks),
        target_points_xy=np.stack([match.target_xy for match in forward_matches]),
    )
    candidates: list[Candidate] = []
    diagnostics: list[dict[str, Any]] = []
    for object_index, region_id in enumerate(valid_regions):
        expert_errors: dict[str, float] = {}
        for expert_index, expert_name in enumerate(EXPERT_ORDER):
            expert_output = predictions[expert_name]
            expert_errors[expert_name] = cycle_error(
                reverse_pair,
                expert_output.mask[object_index],
                forward_matches[object_index].source_xy,
                target_hw=target.hw,
                source_hw=source.frame.hw,
                foreground_threshold=foreground_threshold,
                seed=_mix_seed(seed, source.frame.frame_id, target.frame_id, region_id, expert_index + 1),
            )
        selected_expert = min(EXPERT_ORDER, key=lambda name: expert_errors[name])
        selected_error = float(expert_errors[selected_expert])
        source_similarity = float(forward_matches[object_index].cosine)
        cycle_weight = 0.0 if not np.isfinite(selected_error) else math.exp(
            -selected_error / max(float(cycle_temperature), 1e-6)
        )
        weight = max(cycle_weight * _similarity_weight(source_similarity), 1e-8)
        selected_output = predictions[selected_expert]
        selected_mask = selected_output.mask[object_index].astype(bool)
        selected_probability = selected_output.probability[object_index].astype(np.float32)
        selected_probability = np.where(
            selected_mask,
            np.maximum(selected_probability, float(mask_threshold) + 1e-4),
            np.minimum(selected_probability, float(mask_threshold) - 1e-4),
        ).astype(np.float16)
        candidates.append(
            Candidate(
                region_id=region_id,
                source_frame_id=source.frame.frame_id,
                expert=selected_expert,
                probability=selected_probability,
                mask=selected_mask,
                cycle_error=selected_error,
                source_similarity=source_similarity,
                weight=float(weight),
            )
        )
        diagnostics.append(
            {
                "regionId": int(region_id),
                "sourceFrameId": int(source.frame.frame_id),
                "targetFrameId": int(target.frame_id),
                "selectedExpert": selected_expert,
                "cycleError": None if not np.isfinite(selected_error) else selected_error,
                "expertCycleErrors": {
                    name: None if not np.isfinite(value) else float(value)
                    for name, value in expert_errors.items()
                },
                "sourceSimilarity": source_similarity,
                "fusionWeight": float(weight),
                "sourcePointXY": forward_matches[object_index].source_xy.tolist(),
                "targetPointXY": forward_matches[object_index].target_xy.tolist(),
                "sourceForegroundPatchCount": int(forward_matches[object_index].foreground_patch_count),
                "candidateAreaPixels": int(np.count_nonzero(selected_mask)),
            }
        )
    return candidates, diagnostics


def _direct_pair_output(
    candidates: list[Candidate],
    *,
    frame_shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Emit one PCCS-selected expert mask without cross-source consensus."""
    if len(candidates) != 1:
        raise RuntimeError(
            "Direct V2-SAM pair output requires exactly one source-region candidate; "
            f"received {len(candidates)}."
        )
    candidate = candidates[0]
    mask = np.asarray(candidate.mask, dtype=bool)
    probability = np.asarray(candidate.probability, dtype=np.float32)
    if mask.shape != frame_shape or probability.shape != frame_shape:
        raise ValueError(
            f"Direct V2-SAM candidate has shape {mask.shape}/{probability.shape}; "
            f"expected {frame_shape}."
        )
    labels = np.zeros(frame_shape, dtype=np.uint16)
    labels[mask] = np.uint16(candidate.region_id)
    confidence = np.where(mask, probability, 0.0).astype(np.float16)
    return labels, {
        "region_ids": np.asarray([candidate.region_id], dtype=np.uint16),
        "confidence": confidence,
        "margin": confidence.copy(),
        "uncertainty": np.clip(1.0 - confidence, 0.0, 1.0).astype(np.float16),
    }, [
        {
            "regionId": int(candidate.region_id),
            "candidateCount": 1,
            "sourceFrameIds": [int(candidate.source_frame_id)],
            "experts": [candidate.expert],
            "selectedExpert": candidate.expert,
            "cycleError": None if not np.isfinite(candidate.cycle_error) else float(candidate.cycle_error),
            "outputMode": "direct_pccs_selected_expert",
        }
    ]


def run(args: argparse.Namespace) -> None:
    import torch

    device = _resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The official V2-SAM ViT-L/Hiera-L expert stack currently requires CUDA.")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    frames = _load_frames(args.manifest)
    source = _load_sources(frames, args.anchors_dir, [int(args.source_index)])[0]
    target_index = int(args.target_index)
    if target_index < 0 or target_index >= len(frames):
        raise ValueError(f"Target index is outside the frame sequence: {target_index}")
    target = frames[target_index]
    if target.local_index == source.frame.local_index:
        raise ValueError("V2-SAM pair transfer requires different source and target frames.")
    if len(source.region_ids) != 1:
        raise ValueError(
            f"V2-SAM pair transfer requires one source region; received {source.region_ids}."
        )
    region_id = int(source.region_ids[0])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    frame_diagnostics_dir = args.diagnostics_dir / "frames"
    confidence_dir = args.diagnostics_dir / "confidence"
    frame_diagnostics_dir.mkdir(parents=True, exist_ok=True)
    confidence_dir.mkdir(parents=True, exist_ok=True)

    _atomic_npy(masks_dir / f"{source.frame.local_index:05d}.npy", source.labels)

    cache_frames = [
        DinoFrame(
            local_index=frame.local_index,
            frame_id=frame.frame_id,
            image_path=frame.image_path,
            width=frame.width,
            height=frame.height,
        )
        for frame in (source.frame, target)
    ]
    feature_cache = prepare_dinov3_cache(
        cache_frames,
        root=args.feature_cache_dir,
        repo_root=args.root,
        checkpoint=args.dino_checkpoint,
        device=device,
    )

    experts = V2SamExperts(
        ExpertConfig(
            root=args.root,
            sam_checkpoint=args.sam_checkpoint,
            visual_checkpoint=args.visual_checkpoint,
            fusion_checkpoint=args.fusion_checkpoint,
            dino_checkpoint=args.dino_checkpoint,
            device=str(device),
            object_batch_size=int(args.expert_batch_size),
        )
    )
    try:
        source_features = feature_cache.load(source.frame.frame_id)
        target_rgb = _load_rgb(target)
        target_features = feature_cache.load(target.frame_id)
        forward_pair = compute_pair_map(source_features, target_features, device)
        reverse_pair = compute_pair_map(target_features, source_features, device)
        source_similarity = region_correspondence_score(
            forward_pair,
            source.labels == region_id,
            foreground_threshold=float(args.foreground_threshold),
        )
        experts.set_target(target_rgb)
        candidates, pair_rows = _pair_candidates(
            source=source,
            target=target,
            region_ids=[region_id],
            forward_pair=forward_pair,
            reverse_pair=reverse_pair,
            experts=experts,
            source_rgb=_load_rgb(source.frame),
            target_rgb=target_rgb,
            mask_threshold=float(args.mask_threshold),
            cycle_temperature=float(args.cycle_temperature),
            foreground_threshold=float(args.foreground_threshold),
            seed=int(args.seed),
        )
        labels, confidence, region_rows = _direct_pair_output(
            candidates,
            frame_shape=target.hw,
        )
        _atomic_npz(confidence_dir / f"{target.frame_id:06d}.npz", **confidence)
        _atomic_json(
            frame_diagnostics_dir / f"{target.frame_id:06d}.json",
            {
                "schemaVersion": 1,
                "frameId": int(target.frame_id),
                "localIndex": int(target.local_index),
                "sourceFrameId": int(source.frame.frame_id),
                "regionId": region_id,
                "sourceSimilarity": (
                    None if not np.isfinite(source_similarity) else float(source_similarity)
                ),
                "pairCandidates": pair_rows,
                "regionResult": region_rows,
                "outputMode": "direct_pccs_selected_expert",
                "outputCoverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
                "updatedUtc": _now(),
            },
        )
        _atomic_npy(masks_dir / f"{target.local_index:05d}.npy", labels)
        print(f"V2SAM_FRAME 1 1 {target.frame_id} infer", flush=True)
    finally:
        experts.close()

    run_row = {
        "frameId": int(target.frame_id),
        "sourceFrameId": int(source.frame.frame_id),
        "regionId": region_id,
        "candidateCount": len(candidates),
        "coverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
    }
    _atomic_json(
        args.diagnostics_dir / "summary.json",
        {
            "schemaVersion": 1,
            "stage": "v2sam_pair_propagation",
            "paperMethod": "V2-SAM Anchor + Visual + Fusion experts with PCCS Cycle-Points",
            "adaptation": (
                "First manual region occurrence transferred to one target using the direct "
                "PCCS-selected expert mask."
            ),
            "outputMode": "direct_pccs_selected_expert",
            "expertProfile": str(args.expert_profile),
            "frameCount": len(frames),
            "processedFrameCount": 1,
            "sourceFrameId": int(source.frame.frame_id),
            "targetFrameId": int(target.frame_id),
            "regionId": region_id,
            "parameters": {
                "maskThreshold": float(args.mask_threshold),
                "foregroundThreshold": float(args.foreground_threshold),
                "cycleTemperature": float(args.cycle_temperature),
            },
            "dinoEvidenceDirectory": str(args.feature_cache_dir),
            "frames": [run_row],
            "updatedUtc": _now(),
        },
    )


def _similarity_weight(cosine: float) -> float:
    if not np.isfinite(cosine):
        return 0.0
    return float(np.clip((float(cosine) + 1.0) * 0.5, 0.0, 1.0))


def _mix_seed(*values: int) -> int:
    state = 2166136261
    for value in values:
        state ^= int(value) & 0xFFFFFFFF
        state = (state * 16777619) & 0xFFFFFFFF
    return state


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one V2-SAM persistent-region pair transfer.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-checkpoint", type=Path, required=True)
    parser.add_argument("--fusion-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--anchors-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, required=True)
    parser.add_argument("--target-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--expert-profile", default="ego2exo")
    parser.add_argument("--expert-batch-size", type=int, default=8)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--foreground-threshold", type=float, default=0.6)
    parser.add_argument("--cycle-temperature", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=71)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for name in (
        "root",
        "sam_checkpoint",
        "dino_checkpoint",
        "visual_checkpoint",
        "fusion_checkpoint",
        "manifest",
        "anchors_dir",
        "output_dir",
        "feature_cache_dir",
        "diagnostics_dir",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
