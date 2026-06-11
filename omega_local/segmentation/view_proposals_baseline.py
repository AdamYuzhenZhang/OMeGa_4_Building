"""Generic per-view 2D proposal masks for downstream 3D segmentation.

This baseline is an input provider, not a cross-view segmentation method. It
stages posed OMeGa DSLR frames and runs SAM2 automatic mask generation
independently on each frame. Downstream methods such as SAI3D can consume these
integer proposal masks without depending on SAM2Object tracking or graph logic.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


OMEGA_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = OMEGA_ROOT.parent


@dataclass(frozen=True)
class ViewProposalPaths:
    baseline_dir: Path
    dataset_dir: Path
    image_dir: Path
    pose_dir: Path
    view_mask_dir: Path
    mask_dir: Path
    overlay_dir: Path
    summary_dir: Path
    log_dir: Path
    scene_name: str

    @property
    def frame_manifest(self) -> Path:
        return self.dataset_dir / "frame_manifest.jsonl"

    @property
    def dataset_summary(self) -> Path:
        return self.dataset_dir / "dataset_summary.json"

    @property
    def predictions_path(self) -> Path:
        return self.view_mask_dir / "predictions.jsonl"

    @property
    def summary(self) -> Path:
        return self.baseline_dir / "baseline_summary.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _slug_token(value: str) -> str:
    out: list[str] = []
    last = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last = False
        elif not last:
            out.append("_")
            last = True
    return "".join(out).strip("_") or "view_proposals"


def _require_file(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def resolve_paths(
    model_dir: Path,
    *,
    baseline_name: str = "view_proposals",
    scene_name: str | None = None,
    output_dir: Path | None = None,
) -> ViewProposalPaths:
    baseline_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else model_dir / "segmentation" / "baselines" / _slug_token(baseline_name)
    )
    scene_slug = _slug_token(scene_name or f"{model_dir.parent.parent.name}_{model_dir.name}")
    dataset_dir = baseline_dir / "dataset"
    return ViewProposalPaths(
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        image_dir=dataset_dir / "images" / scene_slug,
        pose_dir=dataset_dir / "poses" / scene_slug,
        view_mask_dir=baseline_dir / "view_masks",
        mask_dir=baseline_dir / "view_masks" / "masks",
        overlay_dir=baseline_dir / "view_masks" / "overlays",
        summary_dir=baseline_dir / "summaries",
        log_dir=baseline_dir / "logs",
        scene_name=scene_slug,
    )


def _infer_capture_root(model_dir: Path) -> Path:
    for candidate in [model_dir, *model_dir.parents]:
        if (candidate / "cameras" / "poses" / "final_poses.jsonl").exists():
            return candidate
    raise SystemExit(
        "Could not infer capture root from --model-dir. "
        "Pass --capture-root pointing to the scan-processing package."
    )


def _pose_from_row(row: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(row["poseWorldTCam"], dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Expected 16 pose values, got {arr.size}")
    return arr.reshape(4, 4)


def _selected_pose_rows(capture_root: Path, frame_stride: int, max_frames: int) -> list[dict[str, Any]]:
    pose_path = capture_root / "cameras" / "poses" / "final_poses.jsonl"
    rows = _read_jsonl(_require_file(pose_path, "final poses jsonl"))
    rows.sort(key=lambda row: (str(row.get("scanID", "")), int(row.get("frameID", 0))))
    rows = rows[:: max(int(frame_stride), 1)]
    if int(max_frames) > 0:
        rows = rows[: int(max_frames)]
    if not rows:
        raise SystemExit(f"No pose rows selected from {pose_path}")
    return rows


def _resolve_image_path(capture_root: Path, row: dict[str, Any]) -> Path:
    rel = row.get("imagePath") or row.get("colmapImageName")
    if rel is None:
        raise ValueError(f"Pose row has no imagePath: {row}")
    return _require_file(capture_root / str(rel), "capture RGB image")


def _resize_rgb(image_path: Path, output_width: int) -> tuple[np.ndarray, float, float]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read RGB image: {image_path}")
    height, width = image.shape[:2]
    if int(output_width) <= 0 or int(output_width) == width:
        return image, 1.0, 1.0
    out_w = int(output_width)
    out_h = max(1, int(round(height * (out_w / float(width)))))
    return cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA), out_w / float(width), out_h / float(height)


def _write_intrinsics(path: Path, fx: float, fy: float, cx: float, cy: float) -> None:
    mat = np.array(
        [
            [float(fx), 0.0, float(cx), 0.0],
            [0.0, float(fy), float(cy), 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, mat, fmt="%.10f")


def _label_color(label: int) -> np.ndarray:
    value = (int(label) * 1103515245 + 12345) & 0xFFFFFFFF
    return np.array(
        [
            45 + ((value >> 0) & 0x7F),
            65 + ((value >> 8) & 0x7F),
            85 + ((value >> 16) & 0x7F),
        ],
        dtype=np.uint8,
    )


def _label_overlay(rgb_bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    out = rgb.astype(np.float32, copy=True)
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        mask = labels == label_i
        out[mask] = (1.0 - alpha) * out[mask] + alpha * _label_color(label_i).astype(np.float32)
    return cv2.cvtColor(np.clip(np.rint(out), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def prepare(args: argparse.Namespace, paths: ViewProposalPaths) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    capture_root = args.capture_root.expanduser().resolve() if args.capture_root is not None else _infer_capture_root(model_dir)
    pose_rows = _selected_pose_rows(capture_root, int(args.frame_stride), int(args.max_frames))

    if paths.dataset_dir.exists() and any(paths.dataset_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"View proposal dataset already exists: {paths.dataset_dir}. Pass --overwrite to replace it.")
    if args.overwrite and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)

    for directory in [paths.image_dir, paths.pose_dir, paths.summary_dir, paths.log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    first_rgb, sx, sy = _resize_rgb(_resolve_image_path(capture_root, pose_rows[0]), int(args.image_width))
    out_h, out_w = first_rgb.shape[:2]
    fx = float(pose_rows[0]["fx"]) * sx
    fy = float(pose_rows[0]["fy"]) * sy
    cx = float(pose_rows[0]["cx"]) * sx
    cy = float(pose_rows[0]["cy"]) * sy
    _write_intrinsics(paths.pose_dir / "intrinsic_color.txt", fx, fy, cx, cy)

    manifest: list[dict[str, Any]] = []
    for out_index, row in enumerate(pose_rows):
        image_path = _resolve_image_path(capture_root, row)
        rgb, sx_i, sy_i = _resize_rgb(image_path, int(args.image_width))
        if rgb.shape[:2] != (out_h, out_w):
            raise SystemExit(f"Resized frame shape differs: {image_path} -> {rgb.shape[:2]} vs {(out_h, out_w)}")
        if abs(sx_i - sx) > 1e-6 or abs(sy_i - sy) > 1e-6:
            raise SystemExit("Frame resize scale differs across selected frames.")

        frame_name = f"{out_index}.jpg"
        color_path = paths.image_dir / frame_name
        pose_path = paths.pose_dir / f"{out_index}.txt"
        if not cv2.imwrite(str(color_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
            raise RuntimeError(f"Failed to write RGB image: {color_path}")
        np.savetxt(pose_path, _pose_from_row(row), fmt="%.10f")
        manifest.append(
            {
                "sourceFrameId": int(out_index),
                "scanID": str(row.get("scanID", "")),
                "frameID": int(row.get("frameID", out_index)),
                "imageName": str(row.get("imageName", image_path.name)),
                "sourceImagePath": str(image_path),
                "colorPath": str(color_path.relative_to(paths.baseline_dir)),
                "posePath": str(pose_path.relative_to(paths.baseline_dir)),
                "width": int(out_w),
                "height": int(out_h),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            }
        )
        print(f"[view-proposals prepare {out_index + 1:04d}/{len(pose_rows):04d}] {image_path.name}")

    _write_jsonl(paths.frame_manifest, manifest)
    summary = {
        "stage": "prepare",
        "timestampUtc": _now(),
        "method": "Stage posed RGB frames for per-view 2D proposal generation.",
        "modelDir": str(model_dir),
        "captureRoot": str(capture_root),
        "sceneName": paths.scene_name,
        "frameCount": int(len(manifest)),
        "frameStride": int(args.frame_stride),
        "maxFrames": int(args.max_frames),
        "imageWidth": int(out_w),
        "imageHeight": int(out_h),
        "intrinsics": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "outputs": {
            "datasetDir": str(paths.dataset_dir),
            "imageDir": str(paths.image_dir),
            "poseDir": str(paths.pose_dir),
            "frameManifest": str(paths.frame_manifest),
        },
    }
    _write_json(paths.dataset_summary, summary)
    _write_json(paths.summary_dir / "prepare_summary.json", summary)
    return summary


def _resolve_sam2_root(path: Path | None) -> Path:
    root = THIRD_PARTY_ROOT / "sam2" if path is None else path.expanduser().resolve()
    return _require_dir(root, "SAM2 root")


def _resolve_sam2_checkpoint(sam2_root: Path, checkpoint: Path | None) -> Path:
    ckpt = checkpoint.expanduser().resolve() if checkpoint is not None else sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    return _require_file(ckpt, "SAM2 checkpoint")


def _mask_records(masks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in masks:
        mask = np.asarray(item.get("segmentation"), dtype=bool)
        area = int(np.count_nonzero(mask))
        if area < int(args.min_mask_area):
            continue
        predicted_iou = float(item.get("predicted_iou", 0.0))
        stability = float(item.get("stability_score", 0.0))
        if predicted_iou < float(args.pred_iou_thresh_keep) or stability < float(args.stability_score_thresh_keep):
            continue
        score = predicted_iou * 0.7 + stability * 0.3
        records.append({"mask": mask, "area": area, "predicted_iou": predicted_iou, "stability": stability, "score": score})
    records.sort(key=lambda item: (float(item["score"]), int(item["area"])), reverse=True)
    if int(args.max_masks_per_frame) > 0:
        records = records[: int(args.max_masks_per_frame)]
    return records


def _records_to_label_map(records: list[dict[str, Any]], shape: tuple[int, int], min_remaining_area: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.zeros(shape, dtype=np.uint16)
    occupied = np.zeros(shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    next_label = 1
    for record in records:
        raw_mask = np.asarray(record["mask"], dtype=bool)
        clean = raw_mask & ~occupied
        remaining = int(np.count_nonzero(clean))
        if remaining < int(min_remaining_area):
            continue
        label = next_label
        next_label += 1
        labels[clean] = np.uint16(label)
        occupied |= clean
        rows.append(
            {
                "labelId": int(label),
                "rawAreaPixels": int(record["area"]),
                "assignedPixels": int(remaining),
                "predictedIou": float(record["predicted_iou"]),
                "stabilityScore": float(record["stability"]),
                "score": float(record["score"]),
            }
        )
    return labels, rows


def sam2_auto(args: argparse.Namespace, paths: ViewProposalPaths) -> dict[str, Any]:
    frame_rows = _read_jsonl(_require_file(paths.frame_manifest, "view proposal frame manifest"))
    if not frame_rows:
        raise SystemExit(f"No frames found in {paths.frame_manifest}")
    if paths.view_mask_dir.exists() and any(paths.view_mask_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"View proposal masks already exist: {paths.view_mask_dir}. Pass --overwrite to replace them.")
    if args.overwrite and paths.view_mask_dir.exists():
        shutil.rmtree(paths.view_mask_dir)
    for directory in [paths.mask_dir, paths.overlay_dir, paths.summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    sam2_root = _resolve_sam2_root(args.sam2_root)
    checkpoint = _resolve_sam2_checkpoint(sam2_root, args.sam2_checkpoint)
    if str(sam2_root) not in sys.path:
        sys.path.insert(0, str(sam2_root))

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    device = str(args.device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if device.startswith("cuda") and not use_cuda:
        print("[view-proposals] CUDA requested but unavailable; falling back to CPU.")
        device = "cpu"
    if use_cuda and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

    prediction_rows: list[dict[str, Any]] = []
    frame_summaries: list[dict[str, Any]] = []
    with torch.inference_mode(), autocast_context:
        model = build_sam2(str(args.sam2_config), str(checkpoint), device=device)
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=int(args.points_per_side),
            pred_iou_thresh=float(args.pred_iou_thresh),
            stability_score_thresh=float(args.stability_score_thresh),
            stability_score_offset=float(args.stability_score_offset),
            crop_n_layers=int(args.crop_n_layers),
            box_nms_thresh=float(args.box_nms_thresh),
            crop_n_points_downscale_factor=int(args.crop_n_points_downscale_factor),
            min_mask_region_area=float(args.min_mask_region_area),
            use_m2m=True,
            multimask_output=False,
        )

        for row_index, row in enumerate(frame_rows):
            frame_id = int(row["sourceFrameId"])
            image_path = paths.baseline_dir / str(row["colorPath"])
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError(f"Could not read staged RGB image: {image_path}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            generated = generator.generate(image_rgb)
            records = _mask_records(generated, args)
            label_map, label_rows = _records_to_label_map(records, image_bgr.shape[:2], int(args.min_assigned_area))

            npy_path = paths.mask_dir / f"mask_{frame_id}.npy"
            png_path = paths.mask_dir / f"{frame_id:06d}.png"
            overlay_path = paths.overlay_dir / f"{frame_id:06d}.jpg"
            np.save(npy_path, label_map)
            if not cv2.imwrite(str(png_path), label_map):
                raise RuntimeError(f"Failed to write proposal mask: {png_path}")
            if not cv2.imwrite(str(overlay_path), _label_overlay(image_bgr, label_map), [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                raise RuntimeError(f"Failed to write proposal overlay: {overlay_path}")

            coverage = float(np.count_nonzero(label_map > 0) / max(label_map.size, 1))
            prediction_rows.append(
                {
                    "sourceFrameId": int(frame_id),
                    "rawMaskCount": int(len(generated)),
                    "keptMaskCount": int(len(label_rows)),
                    "coverage": coverage,
                    "maskNpy": str(npy_path),
                    "maskPng": str(png_path),
                    "overlay": str(overlay_path),
                    "labels": label_rows,
                }
            )
            frame_summaries.append(
                {
                    "sourceFrameId": int(frame_id),
                    "rawMaskCount": int(len(generated)),
                    "keptMaskCount": int(len(label_rows)),
                    "coverage": coverage,
                }
            )
            print(
                f"[view-proposals sam2_auto {row_index + 1:04d}/{len(frame_rows):04d}] "
                f"frame={frame_id} raw={len(generated)} kept={len(label_rows)} coverage={coverage:.3f}"
            )

    _write_jsonl(paths.predictions_path, prediction_rows)
    summary = {
        "stage": "sam2_auto",
        "timestampUtc": _now(),
        "method": "SAM2 automatic masks per view, merged into mutually exclusive integer proposal labels.",
        "sceneName": paths.scene_name,
        "frameCount": int(len(frame_rows)),
        "sam2Root": str(sam2_root),
        "sam2Checkpoint": str(checkpoint),
        "sam2Config": str(args.sam2_config),
        "device": device,
        "parameters": {
            "pointsPerSide": int(args.points_per_side),
            "predIouThresh": float(args.pred_iou_thresh),
            "stabilityScoreThresh": float(args.stability_score_thresh),
            "predIouThreshKeep": float(args.pred_iou_thresh_keep),
            "stabilityScoreThreshKeep": float(args.stability_score_thresh_keep),
            "cropNLayers": int(args.crop_n_layers),
            "maxMasksPerFrame": int(args.max_masks_per_frame),
            "minMaskArea": int(args.min_mask_area),
            "minAssignedArea": int(args.min_assigned_area),
        },
        "frames": frame_summaries,
        "outputs": {
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskMasksDir": str(paths.mask_dir),
            "viewMaskOverlayDir": str(paths.overlay_dir),
            "viewMaskPredictions": str(paths.predictions_path),
        },
    }
    _write_json(paths.summary_dir / "sam2_auto_summary.json", summary)
    return summary


def _merge_stage_summary(paths: ViewProposalPaths, stage_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for path in [paths.summary_dir / "prepare_summary.json", paths.summary_dir / "sam2_auto_summary.json"]:
        if path.exists():
            summaries[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    outputs = {
        "datasetDir": str(paths.dataset_dir),
        "imageDir": str(paths.image_dir),
        "poseDir": str(paths.pose_dir),
        "frameManifest": str(paths.frame_manifest),
        "viewMaskDir": str(paths.view_mask_dir),
        "viewMaskMasksDir": str(paths.mask_dir),
        "viewMaskOverlayDir": str(paths.overlay_dir),
        "viewMaskPredictions": str(paths.predictions_path),
    }
    payload = {
        "stage": "view_proposals",
        "timestampUtc": _now(),
        "baselineName": str(args.baseline_name),
        "sceneName": paths.scene_name,
        "baselineDir": str(paths.baseline_dir),
        "activeStageResults": stage_results,
        "summaries": summaries,
        "outputs": outputs,
    }
    _write_json(paths.summary, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build generic per-view 2D proposal masks for SAI3D or other downstream methods.")
    parser.add_argument("--baseline", default="view_proposals", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("prepare", "sam2_auto", "all"), default="prepare")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, default=None)
    parser.add_argument("--baseline-name", default="view_proposals")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--image-width", type=int, default=1024, help="Resize RGB width before proposal generation. <=0 keeps original.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--sam2-root", type=Path, default=None, help="Defaults to third_party/sam2.")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--stability-score-offset", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=float, default=25.0)
    parser.add_argument("--pred-iou-thresh-keep", type=float, default=0.0)
    parser.add_argument("--stability-score-thresh-keep", type=float, default=0.0)
    parser.add_argument("--min-mask-area", type=int, default=120)
    parser.add_argument("--min-assigned-area", type=int, default=80)
    parser.add_argument("--max-masks-per-frame", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    paths = resolve_paths(
        model_dir,
        baseline_name=str(args.baseline_name),
        scene_name=args.scene_name,
        output_dir=args.output_dir,
    )
    stages = ["prepare", "sam2_auto"] if args.stage == "all" else [args.stage]
    stage_results: list[dict[str, Any]] = []
    for stage in stages:
        if stage == "prepare":
            stage_results.append(prepare(args, paths))
        elif stage == "sam2_auto":
            stage_results.append(sam2_auto(args, paths))
        else:  # pragma: no cover
            raise AssertionError(stage)
    summary = _merge_stage_summary(paths, stage_results, args)
    print("=== View Proposal Baseline ===")
    print(f"Stage: {args.stage}")
    print(f"Baseline dir: {paths.baseline_dir}")
    print(f"Summary: {paths.summary}")
    print(f"Proposal masks: {summary['outputs']['viewMaskMasksDir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
