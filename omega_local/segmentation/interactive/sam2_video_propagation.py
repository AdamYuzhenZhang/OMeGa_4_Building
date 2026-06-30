"""Small SAM2-video propagation wrapper for interactive proposal experiments."""

from __future__ import annotations

import base64
import io
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .paths import require_dir, require_file
from .sam2_session import Sam2Config, encode_mask_png


@dataclass(frozen=True)
class PropagationFrame:
    frame_id: int
    image_path: Path
    width: int
    height: int


class Sam2VideoPropagationSession:
    """Runs a short SAM2 video propagation from a user-edited source mask."""

    def __init__(self, config: Sam2Config, *, work_root: Path) -> None:
        self.config = config
        self.work_root = work_root
        self.predictor: Any | None = None
        self.device: Any | None = None
        self.amp_context: Any = nullcontext()

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
            self.amp_context = torch.autocast("cuda", dtype=torch.bfloat16)
            self.amp_context.__enter__()
            if torch.cuda.get_device_properties(device).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        self.predictor = build_sam2_video_predictor(
            str(self.config.config),
            str(checkpoint),
            device=device,
            vos_optimized=False,
        )

    def propagate(
        self,
        *,
        frames: list[PropagationFrame],
        source_local_index: int,
        source_mask: np.ndarray,
        max_neighbors: int,
    ) -> list[dict[str, Any]]:
        if not frames:
            raise ValueError("SAM2 propagation needs at least one frame.")
        if source_local_index < 0 or source_local_index >= len(frames):
            raise ValueError(f"Source frame index {source_local_index} is outside the propagation sequence.")
        if not np.any(source_mask):
            raise ValueError("SAM2 propagation source mask is empty.")

        self._ensure_loaded()
        assert self.predictor is not None

        source = frames[source_local_index]
        source_size = (int(source.width), int(source.height))
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != (source_size[1], source_size[0]):
            raise ValueError(
                f"Source mask shape {source_mask.shape} does not match source frame size {(source_size[1], source_size[0])}."
            )

        self.work_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="sam2_video_", dir=self.work_root) as tmp_name:
            video_dir = Path(tmp_name)
            for local_index, frame in enumerate(frames):
                with Image.open(frame.image_path) as image:
                    rgb = image.convert("RGB")
                    if rgb.size != source_size:
                        rgb = rgb.resize(source_size, Image.Resampling.LANCZOS)
                    rgb.save(video_dir / f"{local_index:05d}.jpg", quality=95)

            inference_state = self.predictor.init_state(
                video_path=str(video_dir),
                offload_video_to_cpu=True,
                offload_state_to_cpu=True,
                async_loading_frames=False,
            )
            self.predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=int(source_local_index),
                obj_id=1,
                mask=source_mask,
            )

            masks_by_local: dict[int, np.ndarray] = {}
            span = max(int(max_neighbors), 1)
            for frame_idx, _obj_ids, mask_logits in self.predictor.propagate_in_video(
                inference_state,
                start_frame_idx=int(source_local_index),
                max_frame_num_to_track=span,
                reverse=False,
            ):
                masks_by_local[int(frame_idx)] = _mask_logits_to_numpy(mask_logits)

            if source_local_index > 0:
                for frame_idx, _obj_ids, mask_logits in self.predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=int(source_local_index),
                    max_frame_num_to_track=span,
                    reverse=True,
                ):
                    masks_by_local[int(frame_idx)] = _mask_logits_to_numpy(mask_logits)

            self.predictor.reset_state(inference_state)

        rows: list[dict[str, Any]] = []
        for local_index, frame in enumerate(frames):
            mask = masks_by_local.get(local_index)
            if mask is None:
                mask = source_mask if local_index == source_local_index else np.zeros(source_mask.shape, dtype=bool)
            if (int(frame.width), int(frame.height)) != source_size:
                mask = _resize_mask(mask, (int(frame.width), int(frame.height)))
            area = int(np.count_nonzero(mask))
            rows.append(
                {
                    "frameId": int(frame.frame_id),
                    "offset": int(local_index - source_local_index),
                    "isSource": bool(local_index == source_local_index),
                    "width": int(frame.width),
                    "height": int(frame.height),
                    "areaPixels": area,
                    "coverage": float(area / max(mask.size, 1)),
                    "maskPng": encode_mask_png(mask),
                    "maskOverlayPng": _encode_propagation_overlay(mask, is_source=local_index == source_local_index),
                }
            )
        return rows


def _mask_logits_to_numpy(mask_logits: Any) -> np.ndarray:
    mask = mask_logits[0]
    if getattr(mask, "ndim", 0) == 3:
        mask = mask[0]
    mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
    return np.asarray(mask_np > 0, dtype=bool)


def _resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    image = image.resize((int(size[0]), int(size[1])), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8) > 0


def _encode_propagation_overlay(mask: np.ndarray, *, is_source: bool) -> str:
    mask = np.asarray(mask, dtype=bool)
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    if is_source:
        fill = [255, 221, 86]
    else:
        fill = [89, 236, 147]
    overlay[mask, :3] = fill
    overlay[mask, 3] = 126
    edge = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.FIND_EDGES)) > 0
    overlay[edge, :3] = [255, 255, 255]
    overlay[edge, 3] = 235
    buffer = io.BytesIO()
    Image.fromarray(overlay, mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
