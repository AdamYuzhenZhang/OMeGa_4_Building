"""Official V2-SAM expert loading and pairwise inference adapter."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ExpertConfig:
    root: Path
    sam_checkpoint: Path
    visual_checkpoint: Path
    fusion_checkpoint: Path
    dino_checkpoint: Path
    device: str = "cuda"
    object_batch_size: int = 8


@dataclass(frozen=True)
class ExpertOutput:
    probability: np.ndarray
    mask: np.ndarray


class V2SamExperts:
    """Anchor, Visual, and Fusion experts from the official V2-SAM release."""

    def __init__(self, config: ExpertConfig) -> None:
        self.config = config
        root_text = str(config.root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        import torch
        from third_parts.mmdet.models.losses import CrossEntropyLoss, DiceLoss
        from third_parts.sam2.build_sam import build_sam2
        from third_parts.sam2.sam2_image_predictor import SAM2ImagePredictor
        from projects.v2sam_fusion.models import SAM2TrainRunner as FusionSam2TrainRunner
        from projects.v2sam_fusion.models import V2SAM as FusionModel
        from projects.v2sam_visual.models import SAM2TrainRunner as VisualSam2TrainRunner
        from projects.v2sam_visual.models import V2SAM as VisualModel

        self.torch = torch
        self.device = torch.device(config.device)
        loss_mask = dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction="mean",
            loss_weight=2.0,
        )
        loss_dice = dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction="mean",
            naive_dice=True,
            eps=1.0,
            loss_weight=0.5,
        )
        sam_base = str(config.sam_checkpoint.parent)
        self.visual = VisualModel(
            grounding_encoder=dict(type=VisualSam2TrainRunner, base_dir=sam_base),
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            pretrained_pth=str(config.visual_checkpoint),
            frozen_sam2_decoder=False,
            loss_sample_points=True,
            bs=1,
        ).eval().to(self.device)
        self.fusion = FusionModel(
            grounding_encoder=dict(type=FusionSam2TrainRunner, base_dir=sam_base),
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            pretrained_pth=str(config.fusion_checkpoint),
            frozen_sam2_decoder=False,
            loss_sample_points=True,
            bs=1,
            dinov3_cfg=dict(weights_path=str(config.dino_checkpoint)),
            sparse_corr_cfg={},
            external_sparse_correspondence=True,
        ).eval().to(self.device)
        anchor_model = build_sam2(
            "sam2_hiera_l.yaml",
            str(config.sam_checkpoint),
            device=str(self.device),
            mode="eval",
        )
        self.anchor = SAM2ImagePredictor(anchor_model)
        self._target_rgb: np.ndarray | None = None

    def set_target(self, target_rgb: np.ndarray) -> None:
        target = np.asarray(target_rgb, dtype=np.uint8).copy()
        self.anchor.set_image(target)
        self._target_rgb = target

    def predict(
        self,
        *,
        source_rgb: np.ndarray,
        source_masks: np.ndarray,
        target_points_xy: np.ndarray,
    ) -> dict[str, ExpertOutput]:
        if self._target_rgb is None:
            raise RuntimeError("set_target() must be called before pairwise expert prediction.")
        masks = np.asarray(source_masks, dtype=bool)
        points = np.asarray(target_points_xy, dtype=np.float32)
        if masks.ndim != 3 or points.shape != (masks.shape[0], 2):
            raise ValueError(f"Expected masks [N,H,W] and points [N,2], got {masks.shape} and {points.shape}.")
        anchor = self._predict_anchor(points)
        visual = self._predict_learned(self.visual, source_rgb, masks, points=None)
        fusion = self._predict_learned(self.fusion, source_rgb, masks, points=points)
        return {"anchor": anchor, "visual": visual, "fusion": fusion}

    def close(self) -> None:
        del self.anchor, self.visual, self.fusion
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    def _predict_anchor(self, points: np.ndarray) -> ExpertOutput:
        torch = self.torch
        outputs: list[np.ndarray] = []
        for point in points:
            masks, scores, _ = self.anchor.predict(
                point_coords=point.reshape(1, 2),
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
                return_logits=True,
                normalize_coords=True,
            )
            chosen = np.asarray(masks[int(np.argmax(scores))], dtype=np.float32)
            outputs.append(1.0 / (1.0 + np.exp(-np.clip(chosen, -30.0, 30.0))))
        if not outputs:
            height, width = self._target_rgb.shape[:2]
            empty = np.zeros((0, height, width), dtype=np.float32)
            return ExpertOutput(probability=empty, mask=empty.astype(bool))
        result = np.stack(outputs).astype(np.float32)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return ExpertOutput(probability=result, mask=result >= 0.5)

    def _predict_learned(
        self,
        model,
        source_rgb: np.ndarray,
        masks: np.ndarray,
        *,
        points: np.ndarray | None,
    ) -> ExpertOutput:
        torch = self.torch
        source_square = _square_rgb(source_rgb)
        target_square = _square_rgb(self._target_rgb)
        source_tensor = torch.from_numpy(source_square).permute(2, 0, 1).contiguous().to(self.device)
        target_tensor = torch.from_numpy(target_square).permute(2, 0, 1).contiguous().to(self.device)
        target_height, target_width = self._target_rgb.shape[:2]
        outputs: list[np.ndarray] = []
        binary_outputs: list[np.ndarray] = []
        batch_size = max(1, int(self.config.object_batch_size))
        for start in range(0, masks.shape[0], batch_size):
            stop = min(start + batch_size, masks.shape[0])
            prompt_masks = _square_masks(masks[start:stop], self.device)
            data = {
                "g_pixel_values": [target_tensor],
                "prompt_g_pixel_values": [source_tensor],
                "masks": [torch.zeros((stop - start, target_height, target_width), device=self.device)],
                "prompt_masks": [prompt_masks],
                "frames_per_batch": [1],
            }
            if points is not None:
                point_chunk = points[start:stop].copy()
                point_chunk[:, 0] *= 1024.0 / float(target_width)
                point_chunk[:, 1] *= 1024.0 / float(target_height)
                data["sparse_points"] = {
                    "point_coords": torch.from_numpy(point_chunk[:, None]).to(self.device),
                    "point_labels": torch.ones((stop - start, 1), dtype=torch.int32, device=self.device),
                }
            with torch.inference_mode():
                result = model.predict(data)[0]
            outputs.append(result["pred_probs"].detach().float().cpu().numpy())
            binary_outputs.append(result["pred_masks"].detach().bool().cpu().numpy())
            del prompt_masks, data, result
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return ExpertOutput(
            probability=np.concatenate(outputs, axis=0).astype(np.float32),
            mask=np.concatenate(binary_outputs, axis=0).astype(bool),
        )


def _square_rgb(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    resized = image.resize((1024, 1024), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).copy()


def _square_masks(masks: np.ndarray, device):
    import torch
    import torch.nn.functional as F

    values = torch.from_numpy(np.asarray(masks, dtype=np.float32))[:, None]
    values = F.interpolate(values, size=(1024, 1024), mode="nearest")[:, 0]
    return values.to(device)
