"""SAM2-video propagation for complete persistent-region keyframe anchors."""

from __future__ import annotations

import shutil
import sys
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .paths import require_dir, require_file
from .sam2_session import Sam2Config


@dataclass(frozen=True)
class PropagationFrame:
    frame_id: int
    image_name: str
    image_path: Path
    width: int
    height: int
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    pose_world_from_camera: np.ndarray | None = None


@dataclass(frozen=True)
class LabeledPropagationSource:
    local_index: int
    frame_id: int
    labels: np.ndarray


class Sam2VideoPropagationSession:
    """Runs SAM2 video propagation from complete persistent-region maps."""

    def __init__(self, config: Sam2Config, *, work_root: Path) -> None:
        self.config = config
        self.work_root = work_root
        self.predictor: Any | None = None
        self.device: Any | None = None
        self._run_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self.predictor is not None:
            return

        sam2_root = require_dir(self.config.root, "SAM2 root")
        require_file(sam2_root / "sam2" / "build_sam.py", "SAM2 build_sam.py")
        checkpoint = require_file(self.config.checkpoint, "SAM2 checkpoint")
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))

        import torch
        from sam2.build_sam import build_sam2_video_predictor

        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        self.device = device

        if device.type == "cuda":
            if torch.cuda.get_device_properties(device).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.predictor = build_sam2_video_predictor(
            str(self.config.config),
            str(checkpoint),
            device=device,
            vos_optimized=False,
        )

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
        """Propagate persistent region labels from all complete keyframe anchors."""
        del region_rows  # Colors are rendered from the saved label maps by the shared preview layer.
        if not frames:
            raise ValueError("SAM2 propagation needs at least one frame.")
        if not sources:
            raise ValueError("SAM2 region propagation needs at least one complete keyframe region map.")
        if start_local_index < 0 or start_local_index >= len(frames):
            raise ValueError(f"SAM2 start frame index {start_local_index} is outside the propagation sequence.")

        start_frame = frames[start_local_index]
        video_size = (int(start_frame.width), int(start_frame.height))
        prepared_sources, source_labels_by_local, label_ids_sorted = _prepare_video_sources(
            frames,
            sources,
            video_size,
        )
        if not label_ids_sorted:
            raise ValueError("The complete keyframe region maps have no positive persistent regions to propagate.")
        source_locals = set(source_labels_by_local)
        anchor_frame_ids = sorted(int(source.frame_id) for source in prepared_sources)

        self._ensure_loaded()
        assert self.predictor is not None
        run_context = self._autocast_context()

        self.work_root.mkdir(parents=True, exist_ok=True)
        with self._run_lock:
            with run_context:
                with tempfile.TemporaryDirectory(prefix="sam2_region_video_", dir=self.work_root) as tmp_name:
                    video_dir = Path(tmp_name)
                    for local_index, frame in enumerate(frames):
                        _stage_video_frame(frame.image_path, video_dir / f"{local_index:05d}.jpg", video_size)

                    inference_state = self.predictor.init_state(
                        video_path=str(video_dir),
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=True,
                        async_loading_frames=False,
                    )
                    try:
                        for source in prepared_sources:
                            for label_id in [int(label) for label in np.unique(source.labels).tolist() if int(label) > 0]:
                                self.predictor.add_new_mask(
                                    inference_state=inference_state,
                                    frame_idx=int(source.local_index),
                                    obj_id=int(label_id),
                                    mask=np.asarray(source.labels == label_id, dtype=bool),
                                )

                        labels_by_local: dict[int, np.ndarray] = {}
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
                            save_frame(int(local_index), labels_by_local[int(local_index)])
                            mark_frame(int(local_index))

                        def record_frame(frame_idx: int, obj_ids: list[int], mask_logits: Any) -> None:
                            local_idx = int(frame_idx)
                            labels_by_local[local_idx] = _label_map_from_logits(obj_ids, mask_logits)
                            save_frame(local_idx, labels_by_local[local_idx])
                            mark_frame(local_idx)

                        full_span = max(len(frames), 1)
                        for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=int(start_local_index),
                            max_frame_num_to_track=full_span,
                            reverse=False,
                        ):
                            record_frame(int(frame_idx), obj_ids, mask_logits)

                        if start_local_index > 0:
                            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(
                                inference_state,
                                start_frame_idx=int(start_local_index),
                                max_frame_num_to_track=full_span,
                                reverse=True,
                            ):
                                record_frame(int(frame_idx), obj_ids, mask_logits)
                    finally:
                        self.predictor.reset_state(inference_state)

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
            rows.append(
                {
                    "frameId": int(frame.frame_id),
                    "offset": int(local_index - start_local_index),
                    "isSource": bool(local_index in source_locals),
                    "isAnchor": bool(local_index in source_locals),
                    "width": int(frame.width),
                    "height": int(frame.height),
                    "areaPixels": int(np.count_nonzero(positive)),
                    "coverage": float(np.count_nonzero(positive) / max(labels.size, 1)),
                    "regionCount": int(len([label for label in np.unique(labels).tolist() if int(label) > 0])),
                    "anchorFrameIds": anchor_frame_ids,
                    "sourceRegionCount": int(len(label_ids_sorted)),
                }
            )
        return rows

    def _autocast_context(self):
        if self.device is None or self.device.type != "cuda":
            return nullcontext()
        import torch

        return torch.autocast("cuda", dtype=torch.bfloat16)


def _label_map_from_logits(obj_ids: list[int], mask_logits: Any) -> np.ndarray:
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
    positive = logits > 0
    labels = np.zeros(logits.shape[1:], dtype=np.uint16)
    if ids.size == 0:
        return labels
    best = np.argmax(np.where(positive, logits, -np.inf), axis=0)
    has_label = np.any(positive, axis=0)
    labels[has_label] = ids[best[has_label]]
    return labels


def _resize_label_map(labels: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(labels, dtype=np.uint16))
    image = image.resize((int(size[0]), int(size[1])), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint16)


def _prepare_video_sources(
    frames: list[PropagationFrame],
    sources: list[LabeledPropagationSource],
    video_size: tuple[int, int],
) -> tuple[list[LabeledPropagationSource], dict[int, np.ndarray], list[int]]:
    """Validate anchors once and map them into the staged video's pixel grid."""
    prepared: list[LabeledPropagationSource] = []
    labels_by_local: dict[int, np.ndarray] = {}
    label_ids: set[int] = set()
    for source in sources:
        local_index = int(source.local_index)
        if local_index < 0 or local_index >= len(frames):
            raise ValueError(f"Source frame index {local_index} is outside the propagation sequence.")
        frame = frames[local_index]
        expected_shape = (int(frame.height), int(frame.width))
        labels = np.asarray(source.labels, dtype=np.uint16)
        if labels.shape != expected_shape:
            raise ValueError(
                f"Source region map shape {labels.shape} does not match frame {frame.frame_id} shape {expected_shape}."
            )
        if (int(frame.width), int(frame.height)) != video_size:
            labels = _resize_label_map(labels, video_size)
        ids = [int(label) for label in np.unique(labels).tolist() if int(label) > 0]
        if not ids:
            continue
        prepared.append(
            LabeledPropagationSource(
                local_index=local_index,
                frame_id=int(frame.frame_id),
                labels=labels,
            )
        )
        labels_by_local[local_index] = labels
        label_ids.update(ids)
    return prepared, labels_by_local, sorted(label_ids)


def _stage_video_frame(source: Path, target: Path, size: tuple[int, int]) -> None:
    """Stage JPEG frames for SAM2 video without recompressing when possible."""
    source = Path(source)
    target = Path(target)
    with Image.open(source) as image:
        if image.size == (int(size[0]), int(size[1])) and source.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                target.symlink_to(source)
            except OSError:
                shutil.copy2(source, target)
            return
        rgb = image.convert("RGB")
        if rgb.size != (int(size[0]), int(size[1])):
            rgb = rgb.resize((int(size[0]), int(size[1])), Image.Resampling.LANCZOS)
        rgb.save(target, quality=100, subsampling=0)
