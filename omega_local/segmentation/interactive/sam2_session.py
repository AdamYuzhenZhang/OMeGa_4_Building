"""Lazy SAM2 inference wrapper used by the browser editor."""

from __future__ import annotations

import base64
import io
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import require_dir, require_file


@dataclass(frozen=True)
class Sam2Config:
    root: Path
    checkpoint: Path
    config: str
    device: str


class Sam2Session:
    def __init__(self, config: Sam2Config) -> None:
        self.config = config
        self.predictor: Any | None = None
        self.device: Any | None = None
        self.amp_context: Any = nullcontext()
        self.current_frame_id: int | None = None

    def _ensure_loaded(self) -> None:
        if self.predictor is not None:
            return
        sam2_root = require_dir(self.config.root, "SAM2 root")
        require_file(sam2_root / "sam2" / "build_sam.py", "SAM2 build_sam.py")
        checkpoint = require_file(self.config.checkpoint, "SAM2 checkpoint")
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

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

        model = build_sam2(str(self.config.config), str(checkpoint), device=device)
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, frame_id: int, image_rgb: np.ndarray) -> None:
        self._ensure_loaded()
        if self.current_frame_id == int(frame_id):
            return
        assert self.predictor is not None
        self.predictor.set_image(image_rgb)
        self.current_frame_id = int(frame_id)

    def predict(
        self,
        *,
        frame_id: int,
        image_rgb: np.ndarray,
        points_xy: np.ndarray,
        point_labels: np.ndarray,
        multimask: bool,
    ) -> tuple[np.ndarray, float]:
        if points_xy.size == 0:
            raise ValueError("SAM2 needs at least one prompt point.")
        self.set_image(frame_id, image_rgb)
        assert self.predictor is not None
        masks, scores, _logits = self.predictor.predict(
            point_coords=points_xy.astype(np.float32, copy=False),
            point_labels=point_labels.astype(np.int32, copy=False),
            multimask_output=bool(multimask),
        )
        masks_arr = np.asarray(masks)
        scores_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
        if masks_arr.ndim == 2:
            mask = masks_arr.astype(bool, copy=False)
            score = float(scores_arr[0]) if scores_arr.size else 0.0
        elif masks_arr.ndim == 3:
            best = int(np.argmax(scores_arr)) if scores_arr.size else 0
            mask = masks_arr[best].astype(bool, copy=False)
            score = float(scores_arr[best]) if scores_arr.size else 0.0
        else:
            raise ValueError(f"Unexpected SAM2 mask shape: {masks_arr.shape}")
        return mask, score


def encode_mask_overlay(mask: np.ndarray) -> str:
    height, width = mask.shape
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[..., 0] = 124
    overlay[..., 1] = 199
    overlay[..., 2] = 255
    overlay[..., 3] = (mask.astype(np.uint8) * 150).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(overlay, mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def encode_mask_png(mask: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
