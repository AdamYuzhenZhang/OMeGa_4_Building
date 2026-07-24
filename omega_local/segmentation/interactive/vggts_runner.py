"""Isolated one-source/one-target inference for the official VGGT-S model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class Crop:
    x1: int
    y1: int
    size: int
    output_size: int

    def points_to_output(self, points: np.ndarray) -> np.ndarray:
        scale = float(self.output_size) / float(self.size)
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 0] = (result[:, 0] - float(self.x1)) * scale
        result[:, 1] = (result[:, 1] - float(self.y1)) * scale
        return result

    def to_json(self) -> dict[str, int]:
        return {
            "x1": int(self.x1),
            "y1": int(self.y1),
            "x2": int(self.x1 + self.size),
            "y2": int(self.y1 + self.size),
            "size": int(self.size),
            "outputSize": int(self.output_size),
        }


@dataclass(frozen=True)
class Letterbox:
    source_width: int
    source_height: int
    content_width: int
    content_height: int
    output_size: int

    def points_to_source(self, points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 0] /= float(self.content_width) / float(self.source_width)
        result[:, 1] /= float(self.content_height) / float(self.source_height)
        return result

    def clip_to_content(self, points: np.ndarray) -> np.ndarray:
        result = np.asarray(points, dtype=np.float32).copy()
        result[:, 0] = np.clip(result[:, 0], 0, self.content_width - 1)
        result[:, 1] = np.clip(result[:, 1], 0, self.content_height - 1)
        return result

    def to_json(self) -> dict[str, int]:
        return {
            "sourceWidth": int(self.source_width),
            "sourceHeight": int(self.source_height),
            "contentWidth": int(self.content_width),
            "contentHeight": int(self.content_height),
            "outputSize": int(self.output_size),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vggt-model-id", default="facebook/VGGT-1B")
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--source-frame-id", type=int, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--target-frame-id", type=int, required=True)
    parser.add_argument("--region-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--locator-points", type=int, default=50)
    parser.add_argument("--locator-outliers", type=int, default=10)
    parser.add_argument("--prompt-points", type=int, default=5)
    parser.add_argument("--refinement-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=71)
    return parser.parse_args()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _letterbox(
    image: np.ndarray,
    mask: np.ndarray | None,
    size: int,
) -> tuple[np.ndarray, np.ndarray | None, Letterbox]:
    height, width = image.shape[:2]
    if width >= height:
        content_width = int(size)
        content_height = max(14, round(height * size / width / 14) * 14)
    else:
        content_height = int(size)
        content_width = max(14, round(width * size / height / 14) * 14)
    content_width = min(int(size), int(content_width))
    content_height = min(int(size), int(content_height))
    resized = cv2.resize(image, (content_width, content_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:content_height, :content_width] = resized
    mask_canvas = None
    if mask is not None:
        resized_mask = cv2.resize(
            mask.astype(np.uint8),
            (content_width, content_height),
            interpolation=cv2.INTER_NEAREST,
        )
        mask_canvas = np.zeros((size, size), dtype=np.uint8)
        mask_canvas[:content_height, :content_width] = resized_mask
    meta = Letterbox(
        source_width=int(width),
        source_height=int(height),
        content_width=int(content_width),
        content_height=int(content_height),
        output_size=int(size),
    )
    return canvas, mask_canvas, meta


def _source_crop(image: np.ndarray, mask: np.ndarray, output_size: int) -> tuple[np.ndarray, np.ndarray, Crop]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise ValueError("VGGT-S source crop received an empty mask.")
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    crop_size = min(output_size, max(output_size // 2, bbox_width, bbox_height))
    center_x = int((int(xs.min()) + int(xs.max())) // 2)
    center_y = int((int(ys.min()) + int(ys.max())) // 2)
    x1 = int(np.clip(center_x - crop_size // 2, 0, output_size - crop_size))
    y1 = int(np.clip(center_y - crop_size // 2, 0, output_size - crop_size))
    crop = Crop(x1=x1, y1=y1, size=int(crop_size), output_size=int(output_size))
    rgb = cv2.resize(
        image[y1 : y1 + crop_size, x1 : x1 + crop_size],
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )
    cropped_mask = cv2.resize(
        mask[y1 : y1 + crop_size, x1 : x1 + crop_size].astype(np.uint8),
        (output_size, output_size),
        interpolation=cv2.INTER_NEAREST,
    )
    return rgb, cropped_mask, crop


def _target_crop_size(points: np.ndarray, image_shape: tuple[int, ...]) -> int:
    height, width = image_shape[:2]
    short_side = float(min(height, width))
    extent = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))
    canonical_extent = extent * 540.0 / max(short_side, 1.0)
    if canonical_extent < 50.0:
        canonical_crop = 180.0
    elif canonical_extent > 200.0:
        canonical_crop = 540.0
    else:
        canonical_crop = 2.4 * (canonical_extent - 50.0) + 180.0
    return int(np.clip(round(canonical_crop * short_side / 540.0), 1, round(short_side)))


def _target_crop(
    image: np.ndarray,
    points: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray, Crop]:
    height, width = image.shape[:2]
    crop_size = _target_crop_size(points, image.shape)
    center_x = float(points[:, 0].min() + points[:, 0].max()) * 0.5
    center_y = float(points[:, 1].min() + points[:, 1].max()) * 0.5
    x1 = int(np.clip(round(center_x - crop_size * 0.5), 0, width - crop_size))
    y1 = int(np.clip(round(center_y - crop_size * 0.5), 0, height - crop_size))
    crop = Crop(x1=x1, y1=y1, size=int(crop_size), output_size=int(output_size))
    rgb = cv2.resize(
        image[y1 : y1 + crop_size, x1 : x1 + crop_size],
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )
    return rgb, crop.points_to_output(points), crop


def _pair_tensor(source: np.ndarray, target: np.ndarray, device: torch.device) -> torch.Tensor:
    source_tensor = torch.from_numpy(source.astype(np.float32) / 255.0).permute(2, 0, 1)
    target_tensor = torch.from_numpy(target.astype(np.float32) / 255.0).permute(2, 0, 1)
    return torch.stack([source_tensor, target_tensor], dim=0).unsqueeze(0).to(device).contiguous()


def _extract_features(vggt: Any, images: torch.Tensor) -> torch.Tensor:
    aggregated_tokens, patch_start_idx = vggt.aggregator(images)
    return vggt.track_head.feature_extractor(aggregated_tokens, images, patch_start_idx)


def _track_points(vggt: Any, images: torch.Tensor, points: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_maps = _extract_features(vggt, images)
    coords, visibility, confidence = vggt.track_head.tracker(
        query_points=points,
        fmaps=feature_maps,
        iters=vggt.track_head.iters,
    )
    final_coords = coords[-1] if isinstance(coords, (list, tuple)) else coords
    target_coords = final_coords[0, 1].float().detach().cpu().numpy()
    target_visibility = visibility[0, 1].float().detach().cpu().numpy()
    target_confidence = confidence[0, 1].float().detach().cpu().numpy()
    return target_coords, target_visibility, target_confidence


def _resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("VGGT-S requested CUDA, but CUDA is unavailable.")
    return device


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def _json_values(value: np.ndarray) -> list[Any]:
    return np.asarray(value).astype(np.float32).tolist()


def main() -> int:
    args = _parse_args()
    root = args.root.expanduser().resolve()
    sys.path.insert(0, str(root / "src"))
    from model.predictor import Predictor
    from utils import cluster_points, mask_sampling, sample_feat_at_points
    from vggt.models.vggt import VGGT

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cv2.setNumThreads(1)
    device = _resolve_device(args.device)
    size = int(args.image_size)
    if size != 518:
        raise ValueError("The released VGGT-S head is configured for its official 518-pixel input.")

    source_rgb = _load_rgb(args.source_image)
    target_rgb = _load_rgb(args.target_image)
    source_labels = np.load(args.source_labels)
    if source_labels.shape != source_rgb.shape[:2]:
        raise ValueError(
            f"Source labels have shape {source_labels.shape}; expected {source_rgb.shape[:2]}."
        )
    source_mask = (source_labels == int(args.region_id)).astype(np.uint8)
    if not np.any(source_mask):
        raise ValueError(f"Region {args.region_id} is absent from the source labels.")

    source_square, source_square_mask, source_letterbox = _letterbox(source_rgb, source_mask, size)
    target_square, _, target_letterbox = _letterbox(target_rgb, None, size)
    locator_source_points = mask_sampling(
        source_square_mask[None, None],
        int(args.locator_points),
    )[0, 0]

    print(f"[VGGT-S] loading VGGT {args.vggt_model_id} on {device}", flush=True)
    vggt = VGGT.from_pretrained(
        str(args.vggt_model_id),
        enable_camera=False,
        enable_point=False,
        enable_depth=False,
    ).to(device).eval()
    print(f"[VGGT-S] loading union head {args.checkpoint}", flush=True)
    predictor = Predictor(
        img_size=size,
        sa_tokens=37,
        resume_ckpt=str(args.checkpoint),
    ).to(device).eval()

    with torch.inference_mode(), _autocast(device):
        locator_pair = _pair_tensor(source_square, target_square, device)
        locator_points_tensor = torch.from_numpy(locator_source_points).unsqueeze(0).to(device)
        tracked_square, visibility, confidence = _track_points(vggt, locator_pair, locator_points_tensor)

    finite = np.all(np.isfinite(tracked_square), axis=1)
    tracked_square = tracked_square[finite]
    visibility = visibility[finite]
    confidence = confidence[finite]
    if tracked_square.shape[0] < int(args.prompt_points):
        raise RuntimeError("VGGT produced too few finite target correspondences for VGGT-S.")
    tracked_square = target_letterbox.clip_to_content(tracked_square)
    tracked_target = target_letterbox.points_to_source(tracked_square)
    distances = np.linalg.norm(tracked_target - np.median(tracked_target, axis=0), axis=1)
    remove_count = min(
        int(args.locator_outliers),
        max(0, tracked_target.shape[0] - int(args.prompt_points)),
    )
    keep_order = np.argsort(distances)
    tracked_inliers = tracked_target[keep_order[: tracked_target.shape[0] - remove_count]]

    source_crop_rgb, source_crop_mask, source_crop = _source_crop(
        source_square,
        source_square_mask,
        size,
    )
    target_crop_rgb, target_crop_points, target_crop = _target_crop(
        target_rgb,
        tracked_inliers,
        size,
    )
    _, target_prompt_points = cluster_points(target_crop_points, int(args.prompt_points))
    source_prompt_points = mask_sampling(
        source_crop_mask[None, None],
        int(args.prompt_points),
    )[0, 0]

    with torch.inference_mode(), _autocast(device):
        final_pair = _pair_tensor(source_crop_rgb, target_crop_rgb, device)
        feature_maps = _extract_features(vggt, final_pair).float()
        source_points_tensor = torch.from_numpy(source_prompt_points).unsqueeze(0).to(device).float()
        target_points_tensor = torch.from_numpy(target_prompt_points).unsqueeze(0).to(device).float()
        source_point_features = sample_feat_at_points(
            feature_maps[:, 0],
            source_points_tensor,
            size,
            size,
        ).float()
        refinement_logits = None
        predicted_logits = None
        for _ in range(int(args.refinement_steps) + 1):
            predicted_logits, refinement_logits = predictor(
                feat_map_b=feature_maps,
                src_points_b=source_points_tensor,
                tar_points_b=target_points_tensor,
                src_mask_b=source_crop_mask[None],
                src_points_feat_b=source_point_features,
                tar_mask_b=refinement_logits,
                ret_logits=True,
            )
        crop_logits = predicted_logits[0, 0].float()
        recovered_crop = F.interpolate(
            crop_logits[None, None],
            size=(target_crop.size, target_crop.size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        recovered_mask = (recovered_crop > 0.0).detach().cpu().numpy()

    labels = np.zeros(target_rgb.shape[:2], dtype=np.uint16)
    target_slice = labels[
        target_crop.y1 : target_crop.y1 + target_crop.size,
        target_crop.x1 : target_crop.x1 + target_crop.size,
    ]
    target_slice[recovered_mask] = np.uint16(args.region_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)

    args.diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "schemaVersion": 1,
        "method": "VGGT-S official region transfer",
        "sourceFrameId": int(args.source_frame_id),
        "targetFrameId": int(args.target_frame_id),
        "regionId": int(args.region_id),
        "sourceAreaPixels": int(np.count_nonzero(source_mask)),
        "targetAreaPixels": int(np.count_nonzero(labels == int(args.region_id))),
        "targetCoverage": float(np.count_nonzero(labels) / max(labels.size, 1)),
        "imageSize": size,
        "locatorPointCount": int(args.locator_points),
        "locatorFiniteCount": int(np.count_nonzero(finite)),
        "locatorOutlierCount": int(remove_count),
        "promptPointCount": int(args.prompt_points),
        "refinementStepsAfterInitial": int(args.refinement_steps),
        "sourceLetterbox": source_letterbox.to_json(),
        "targetLetterbox": target_letterbox.to_json(),
        "sourceCrop": source_crop.to_json(),
        "targetCrop": target_crop.to_json(),
        "sourcePromptPoints": _json_values(source_prompt_points),
        "targetPromptPoints": _json_values(target_prompt_points),
        "trackedTargetPoints": _json_values(tracked_target),
        "trackedInlierPoints": _json_values(tracked_inliers),
        "visibility": _json_values(visibility),
        "confidence": _json_values(confidence),
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "vggtModelId": str(args.vggt_model_id),
        "device": str(device),
    }
    args.diagnostics.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    print(
        f"[VGGT-S] source={args.source_frame_id} target={args.target_frame_id} "
        f"area={diagnostics['targetAreaPixels']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
