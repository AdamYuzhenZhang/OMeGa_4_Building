"""Bounded SAM2 propagation between complete persistent-region anchors.

This mode keeps all complete keyframes as conditioning masks, then runs SAM2
only inside short anchor-to-anchor intervals. Each interval gets a forward pass
from the left anchor and a backward pass from the right anchor; per-frame
candidates are fused so useful intermediate memory is kept without a full
chronological drift path.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .sam2_session import Sam2Config
from .sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
    Sam2VideoPropagationSession,
    _prepare_video_sources,
    _resize_label_map,
    _stage_video_frame,
)


@dataclass
class _Candidate:
    labels: np.ndarray
    scores: np.ndarray


class Sam2BoundedPropagationSession(Sam2VideoPropagationSession):
    """Runs SAM2 propagation in bounded ranges between complete anchors."""

    def __init__(self, config: Sam2Config, *, work_root: Path) -> None:
        super().__init__(config, work_root=work_root)

    def propagate_labeled_sources(
        self,
        *,
        frames: list[PropagationFrame],
        sources: list[LabeledPropagationSource],
        start_local_index: int,
        region_rows: list[dict[str, Any]],
        progress_callback: Callable[[int, int, int], None] | None = None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Propagate persistent labels inside short anchor-bounded intervals."""
        del region_rows  # Shared previews render directly from the saved uint16 label maps.
        if not frames:
            raise ValueError("SAM2 bounded propagation needs at least one frame.")
        if not sources:
            raise ValueError("SAM2 bounded propagation needs at least one complete keyframe region map.")
        if start_local_index < 0 or start_local_index >= len(frames):
            raise ValueError(f"SAM2 start frame index {start_local_index} is outside the propagation sequence.")

        start_frame = frames[start_local_index]
        video_size = (int(start_frame.width), int(start_frame.height))
        prepared_sources, source_labels_by_local, label_ids = _prepare_video_sources(
            frames,
            sources,
            video_size,
        )
        if not label_ids:
            raise ValueError("The complete keyframe region maps have no positive persistent regions to propagate.")

        source_locals = sorted(source_labels_by_local)
        anchor_frame_ids = sorted(int(source.frame_id) for source in prepared_sources)

        self._ensure_loaded()
        assert self.predictor is not None
        run_context = self._autocast_context()

        labels_by_local: dict[int, np.ndarray] = {}
        stats_by_local: dict[int, dict[str, Any]] = {}
        seen_frames: set[int] = set()
        saved_frames: set[int] = set()
        total_frames = int(len(frames))

        def save_frame(local_idx: int, labels: np.ndarray) -> None:
            if local_idx in saved_frames:
                return
            frame = frames[local_idx]
            out_labels = np.asarray(labels, dtype=np.uint16)
            if (int(frame.width), int(frame.height)) != video_size:
                out_labels = _resize_label_map(out_labels, (int(frame.width), int(frame.height)))
            if save_callback is not None:
                save_callback(frame, out_labels)
            saved_frames.add(local_idx)

        def mark_frame(local_idx: int) -> None:
            if local_idx in seen_frames:
                return
            seen_frames.add(local_idx)
            if progress_callback is not None:
                progress_callback(len(seen_frames), total_frames, int(frames[local_idx].frame_id))

        for local_index, labels in sorted(source_labels_by_local.items()):
            labels_by_local[int(local_index)] = labels.astype(np.uint16, copy=True)
            stats_by_local[int(local_index)] = {
                "boundedMode": "anchor",
                "intervalStart": int(local_index),
                "intervalEnd": int(local_index),
                "conflictPixels": 0,
                "agreementPixels": int(np.count_nonzero(labels > 0)),
            }
            save_frame(int(local_index), labels_by_local[int(local_index)])
            mark_frame(int(local_index))

        self.work_root.mkdir(parents=True, exist_ok=True)
        with self._run_lock:
            with run_context:
                with tempfile.TemporaryDirectory(prefix="sam2_region_bounded_", dir=self.work_root) as tmp_name:
                    video_dir = Path(tmp_name)
                    for local_index, frame in enumerate(frames):
                        _stage_video_frame(frame.image_path, video_dir / f"{local_index:05d}.jpg", video_size)

                    for interval in _bounded_intervals(source_locals, len(frames)):
                        left, right, has_left_anchor, has_right_anchor = interval
                        forward = (
                            self._run_interval(
                                video_dir=video_dir,
                                sources=prepared_sources,
                                start_index=left,
                                end_index=right,
                                reverse=False,
                            )
                            if has_left_anchor
                            else {}
                        )
                        backward = (
                            self._run_interval(
                                video_dir=video_dir,
                                sources=prepared_sources,
                                start_index=right,
                                end_index=left,
                                reverse=True,
                            )
                            if has_right_anchor and left != right
                            else {}
                        )

                        for local_index in range(left, right + 1):
                            if local_index in source_labels_by_local:
                                continue
                            fused, stats = _fuse_interval_candidates(
                                local_index=local_index,
                                left_index=left,
                                right_index=right,
                                forward=forward.get(local_index),
                                backward=backward.get(local_index),
                                shape=(video_size[1], video_size[0]),
                            )
                            labels_by_local[local_index] = fused
                            stats_by_local[local_index] = stats
                            save_frame(local_index, fused)
                            mark_frame(local_index)

        rows: list[dict[str, Any]] = []
        for local_index, frame in enumerate(frames):
            labels = labels_by_local.get(local_index)
            if labels is None:
                labels = np.zeros((video_size[1], video_size[0]), dtype=np.uint16)
                save_frame(local_index, labels)
                mark_frame(local_index)
            if (int(frame.width), int(frame.height)) != video_size:
                labels = _resize_label_map(labels, (int(frame.width), int(frame.height)))
            labels = np.asarray(labels, dtype=np.uint16)
            positive = labels > 0
            row_stats = stats_by_local.get(local_index, {})
            rows.append(
                {
                    "frameId": int(frame.frame_id),
                    "offset": int(local_index - start_local_index),
                    "isSource": bool(local_index in source_labels_by_local),
                    "isAnchor": bool(local_index in source_labels_by_local),
                    "width": int(frame.width),
                    "height": int(frame.height),
                    "areaPixels": int(np.count_nonzero(positive)),
                    "coverage": float(np.count_nonzero(positive) / max(labels.size, 1)),
                    "regionCount": int(len([label for label in np.unique(labels).tolist() if int(label) > 0])),
                    "anchorFrameIds": anchor_frame_ids,
                    "sourceRegionCount": int(len(label_ids)),
                    "boundedMode": row_stats.get("boundedMode", "empty"),
                    "intervalStartFrameId": int(frames[int(row_stats.get("intervalStart", local_index))].frame_id),
                    "intervalEndFrameId": int(frames[int(row_stats.get("intervalEnd", local_index))].frame_id),
                    "agreementPixels": int(row_stats.get("agreementPixels", 0)),
                    "conflictPixels": int(row_stats.get("conflictPixels", 0)),
                    "uncertainPixels": int(row_stats.get("uncertainPixels", 0)),
                }
            )
        return rows

    def _run_interval(
        self,
        *,
        video_dir: Path,
        sources: list[LabeledPropagationSource],
        start_index: int,
        end_index: int,
        reverse: bool,
    ) -> dict[int, _Candidate]:
        assert self.predictor is not None
        inference_state = self.predictor.init_state(
            video_path=str(video_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
            async_loading_frames=False,
        )
        try:
            for source in sources:
                labels = np.asarray(source.labels, dtype=np.uint16)
                for label_id in [int(label) for label in np.unique(labels).tolist() if int(label) > 0]:
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=int(source.local_index),
                        obj_id=int(label_id),
                        mask=np.asarray(labels == label_id, dtype=bool),
                    )

            max_track = abs(int(end_index) - int(start_index))
            candidates: dict[int, _Candidate] = {}
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=int(start_index),
                max_frame_num_to_track=int(max_track),
                reverse=bool(reverse),
            ):
                local_idx = int(frame_idx)
                if local_idx < min(start_index, end_index) or local_idx > max(start_index, end_index):
                    continue
                labels, scores = _label_map_and_score_from_logits(obj_ids, mask_logits)
                candidates[local_idx] = _Candidate(labels=labels, scores=scores)
            return candidates
        finally:
            self.predictor.reset_state(inference_state)


def _bounded_intervals(source_indices: list[int], frame_count: int) -> list[tuple[int, int, bool, bool]]:
    anchors = sorted({int(index) for index in source_indices if 0 <= int(index) < frame_count})
    if not anchors:
        return []
    intervals: list[tuple[int, int, bool, bool]] = []
    if anchors[0] > 0:
        intervals.append((0, anchors[0], False, True))
    for left, right in zip(anchors[:-1], anchors[1:]):
        if right >= left:
            intervals.append((left, right, True, True))
    if anchors[-1] < frame_count - 1:
        intervals.append((anchors[-1], frame_count - 1, True, False))
    if not intervals:
        intervals.append((anchors[0], anchors[0], True, True))
    return intervals


def _fuse_interval_candidates(
    *,
    local_index: int,
    left_index: int,
    right_index: int,
    forward: _Candidate | None,
    backward: _Candidate | None,
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    if forward is None and backward is None:
        return np.zeros(shape, dtype=np.uint16), _stats("empty", local_index, left_index, right_index)
    if backward is None:
        labels = _candidate_or_empty(forward, shape).labels.astype(np.uint16, copy=True)
        return labels, _stats("forward", local_index, left_index, right_index, agreement=int(np.count_nonzero(labels > 0)))
    if forward is None:
        labels = _candidate_or_empty(backward, shape).labels.astype(np.uint16, copy=True)
        return labels, _stats("backward", local_index, left_index, right_index, agreement=int(np.count_nonzero(labels > 0)))

    fwd = _candidate_or_empty(forward, shape)
    bwd = _candidate_or_empty(backward, shape)
    f_labels = fwd.labels.astype(np.uint16, copy=False)
    b_labels = bwd.labels.astype(np.uint16, copy=False)
    f_pos = f_labels > 0
    b_pos = b_labels > 0
    same = f_pos & b_pos & (f_labels == b_labels)
    conflict = f_pos & b_pos & (f_labels != b_labels)

    labels = np.zeros(shape, dtype=np.uint16)
    labels[same] = f_labels[same]
    labels[f_pos & ~b_pos] = f_labels[f_pos & ~b_pos]
    labels[b_pos & ~f_pos] = b_labels[b_pos & ~f_pos]

    gap = max(int(right_index) - int(left_index), 1)
    alpha = (int(local_index) - int(left_index)) / gap
    forward_weight = max(0.05, 1.0 - alpha)
    backward_weight = max(0.05, alpha)
    f_score = fwd.scores * forward_weight
    b_score = bwd.scores * backward_weight
    margin = np.abs(f_score - b_score)
    confident_conflict = conflict & (margin >= 0.20)
    uncertain_conflict = conflict & ~confident_conflict
    choose_forward = confident_conflict & (f_score >= b_score)
    choose_backward = confident_conflict & ~choose_forward
    labels[choose_forward] = f_labels[choose_forward]
    labels[choose_backward] = b_labels[choose_backward]

    stats = _stats(
        "bounded_fused",
        local_index,
        left_index,
        right_index,
        agreement=int(np.count_nonzero(same)),
        conflict=int(np.count_nonzero(conflict)),
        uncertain=int(np.count_nonzero(uncertain_conflict)),
    )
    return labels, stats


def _candidate_or_empty(candidate: _Candidate | None, shape: tuple[int, int]) -> _Candidate:
    if candidate is None:
        return _Candidate(np.zeros(shape, dtype=np.uint16), np.zeros(shape, dtype=np.float32))
    labels = np.asarray(candidate.labels, dtype=np.uint16)
    scores = np.asarray(candidate.scores, dtype=np.float32)
    if labels.shape != shape:
        labels = _resize_label_map(labels, (shape[1], shape[0]))
        scores = np.asarray(scores, dtype=np.float32)
        if scores.shape != shape:
            scores = np.zeros(shape, dtype=np.float32)
    if scores.shape != shape:
        scores = np.zeros(shape, dtype=np.float32)
    return _Candidate(labels=labels, scores=scores)


def _stats(
    mode: str,
    local_index: int,
    left_index: int,
    right_index: int,
    *,
    agreement: int = 0,
    conflict: int = 0,
    uncertain: int = 0,
) -> dict[str, Any]:
    return {
        "boundedMode": mode,
        "intervalStart": int(left_index),
        "intervalEnd": int(right_index),
        "localIndex": int(local_index),
        "agreementPixels": int(agreement),
        "conflictPixels": int(conflict),
        "uncertainPixels": int(uncertain),
    }


def _label_map_and_score_from_logits(obj_ids: list[int], mask_logits: Any) -> tuple[np.ndarray, np.ndarray]:
    logits = mask_logits.detach().cpu().numpy() if hasattr(mask_logits, "detach") else np.asarray(mask_logits)
    logits = np.asarray(logits)
    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim == 2:
        logits = logits[None, ...]
    if logits.ndim != 3:
        raise ValueError(f"Unexpected SAM2 multi-object logits shape: {logits.shape}")

    ids = np.asarray([int(obj_id) for obj_id in obj_ids], dtype=np.uint16)
    if logits.shape[0] != ids.shape[0]:
        raise ValueError(f"SAM2 returned {logits.shape[0]} masks for {ids.shape[0]} object IDs.")
    labels = np.zeros(logits.shape[1:], dtype=np.uint16)
    scores = np.zeros(logits.shape[1:], dtype=np.float32)
    if ids.size == 0:
        return labels, scores

    positive_scores = np.where(logits > 0, logits, -np.inf)
    best = np.argmax(positive_scores, axis=0)
    has_label = np.any(np.isfinite(positive_scores), axis=0)
    labels[has_label] = ids[best[has_label]]
    rows, cols = np.nonzero(has_label)
    if rows.size:
        scores[rows, cols] = positive_scores[best[rows, cols], rows, cols].astype(np.float32)
    return labels, scores
