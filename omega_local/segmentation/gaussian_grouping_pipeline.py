"""Faithful Gaussian Grouping runner for OMeGa-prepared DSLR datasets.

This is intentionally separate from the other OMeGa segmentation baselines.
It stages the existing OMeGa COLMAP dataset into Gaussian Grouping's native
layout, runs Gaussian Grouping's own DEVA/SAM pseudo-label script, then calls
the original Gaussian Grouping training and rendering scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from PIL import Image


OMEGA_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = OMEGA_ROOT.parent


@dataclass(frozen=True)
class GaussianGroupingPaths:
    model_dir: Path
    source_dataset_dir: Path
    gaussian_grouping_root: Path
    dataset_name: str
    run_dir: Path
    staged_dataset_dir: Path
    model_output_dir: Path
    summary_dir: Path
    log_dir: Path

    @property
    def summary_path(self) -> Path:
        return self.run_dir / "gaussian_grouping_summary.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    return "".join(out).strip("_") or "gaussian_grouping"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _default_dataset_name(model_dir: Path) -> str:
    capture = model_dir.parent.parent.name if len(model_dir.parents) >= 2 else model_dir.parent.name
    return _slug_token(f"{capture}_gaussian_grouping")


def _resolve_paths(args: argparse.Namespace) -> GaussianGroupingPaths:
    model_dir = args.model_dir.expanduser().resolve()
    source_dataset_dir = (
        args.source_dataset_dir.expanduser().resolve()
        if args.source_dataset_dir is not None
        else model_dir.parent / "dataset"
    )
    gaussian_grouping_root = (
        args.gaussian_grouping_root.expanduser().resolve()
        if args.gaussian_grouping_root is not None
        else THIRD_PARTY_ROOT / "gaussian-grouping"
    )
    dataset_name = _slug_token(args.dataset_name or _default_dataset_name(model_dir))
    run_name = _slug_token(args.run_name)
    run_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else model_dir / "segmentation" / "gaussian_grouping" / run_name
    )
    staged_dataset_dir = gaussian_grouping_root / "data" / dataset_name
    return GaussianGroupingPaths(
        model_dir=model_dir,
        source_dataset_dir=source_dataset_dir,
        gaussian_grouping_root=gaussian_grouping_root,
        dataset_name=dataset_name,
        run_dir=run_dir,
        staged_dataset_dir=staged_dataset_dir,
        model_output_dir=run_dir / "model",
        summary_dir=run_dir / "summaries",
        log_dir=run_dir / "logs",
    )


def _safe_remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _link_or_copy_dir(src: Path, dst: Path, *, overwrite: bool, copy: bool = False) -> None:
    _require_dir(src, f"source directory for {dst.name}")
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        _safe_remove_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst, symlinks=True)
    else:
        dst.symlink_to(src, target_is_directory=True)


def _iter_images(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    images = [path for path in image_dir.iterdir() if path.suffix.lower() in exts]
    images.sort(key=lambda p: p.name)
    return images


def _make_scaled_images(src_dir: Path, dst_dir: Path, scale: int, *, overwrite: bool) -> dict[str, Any]:
    if scale <= 1:
        _link_or_copy_dir(src_dir, dst_dir, overwrite=overwrite)
        return {"mode": "symlink", "scale": int(scale), "imageCount": len(_iter_images(src_dir))}
    src_images = _iter_images(src_dir)
    if not src_images:
        raise SystemExit(f"No RGB images found in {src_dir}")
    if dst_dir.exists() and overwrite:
        _safe_remove_path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    first_size: list[int] | None = None
    for src in src_images:
        dst = dst_dir / src.name
        if dst.exists() and not overwrite:
            written += 1
            continue
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Could not read RGB image: {src}")
        height, width = image.shape[:2]
        out_w = max(1, int(round(width / float(scale))))
        out_h = max(1, int(round(height / float(scale))))
        resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(dst), resized):
            raise SystemExit(f"Could not write scaled image: {dst}")
        if first_size is None:
            first_size = [out_w, out_h]
        written += 1
    return {
        "mode": "resize",
        "scale": int(scale),
        "imageCount": int(written),
        "firstImageSize": first_size,
    }


def _check_mask_names(staged_dataset_dir: Path) -> dict[str, Any]:
    image_dir = staged_dataset_dir / "images"
    object_dir = staged_dataset_dir / "object_mask"
    images = _iter_images(image_dir) if image_dir.exists() else []
    masks = sorted(object_dir.glob("*.png")) if object_dir.exists() else []
    mask_names = {path.stem for path in masks}
    missing = [path.stem for path in images if path.stem not in mask_names]
    first_mask_size = None
    if masks:
        with Image.open(masks[0]) as mask:
            first_mask_size = list(mask.size)
    return {
        "imageCount": len(images),
        "maskCount": len(masks),
        "missingMaskCount": len(missing),
        "missingMaskExamples": missing[:12],
        "firstMaskSize": first_mask_size,
    }


def _require_complete_object_masks(staged_dataset_dir: Path) -> dict[str, Any]:
    stats = _check_mask_names(staged_dataset_dir)
    if stats["imageCount"] <= 0:
        raise SystemExit(f"No staged RGB images found in {staged_dataset_dir / 'images'}")
    if stats["missingMaskCount"] > 0:
        examples = ", ".join(stats["missingMaskExamples"])
        raise SystemExit(
            "Gaussian Grouping object masks are incomplete: "
            f"{stats['maskCount']} masks for {stats['imageCount']} images; "
            f"missing {stats['missingMaskCount']} masks. Examples: {examples}. "
            "Rerun the prepare_pseudo_labels stage with --overwrite before training."
        )
    return stats


def stage_dataset(args: argparse.Namespace, paths: GaussianGroupingPaths) -> dict[str, Any]:
    _require_dir(paths.gaussian_grouping_root, "Gaussian Grouping repository")
    _require_dir(paths.source_dataset_dir, "OMeGa COLMAP dataset")
    image_dir = _require_dir(paths.source_dataset_dir / "images", "OMeGa dataset image directory")
    sparse_dir = _require_dir(paths.source_dataset_dir / "sparse", "OMeGa dataset sparse COLMAP directory")

    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    paths.staged_dataset_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for stale_scaled_dir in sorted(paths.staged_dataset_dir.glob("images_*")):
            _safe_remove_path(stale_scaled_dir)
        stale_object_mask_dir = paths.staged_dataset_dir / "object_mask"
        if stale_object_mask_dir.exists():
            _safe_remove_path(stale_object_mask_dir)

    _link_or_copy_dir(image_dir, paths.staged_dataset_dir / "images", overwrite=args.overwrite)
    _link_or_copy_dir(sparse_dir, paths.staged_dataset_dir / "sparse", overwrite=args.overwrite)
    scaled_summary = _make_scaled_images(
        image_dir,
        paths.staged_dataset_dir / f"images_{args.scale}",
        int(args.scale),
        overwrite=args.overwrite,
    )

    summary = {
        "stage": "stage_dataset",
        "createdAt": _now(),
        "method": "Gaussian Grouping native dataset staging from OMeGa COLMAP output",
        "inputs": {
            "modelDir": str(paths.model_dir),
            "sourceDatasetDir": str(paths.source_dataset_dir),
            "sourceImageDir": str(image_dir),
            "sourceSparseDir": str(sparse_dir),
        },
        "parameters": {"scale": int(args.scale)},
        "outputs": {
            "runDir": str(paths.run_dir),
            "gaussianGroupingRoot": str(paths.gaussian_grouping_root),
            "datasetName": paths.dataset_name,
            "stagedDatasetDir": str(paths.staged_dataset_dir),
            "stagedImagesDir": str(paths.staged_dataset_dir / "images"),
            "stagedScaledImagesDir": str(paths.staged_dataset_dir / f"images_{args.scale}"),
            "stagedSparseDir": str(paths.staged_dataset_dir / "sparse"),
            "modelOutputDir": str(paths.model_output_dir),
        },
        "stats": {
            "sourceImageCount": len(_iter_images(image_dir)),
            "scaledImages": scaled_summary,
        },
    }
    _write_json(paths.summary_dir / "stage_dataset_summary.json", summary)
    return summary


def _python_env(python: Path) -> dict[str, str]:
    env = os.environ.copy()
    python = python.expanduser().absolute()
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    # DEVA/SAM at scale 2 can fragment CUDA memory over long image sequences.
    # Respect an explicit user value, but default to PyTorch's expandable allocator.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _python_path(python: Path) -> str:
    return str(python.expanduser().absolute())


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    env: dict[str, str],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[gaussian-grouping]", " ".join(command))
    print(f"[gaussian-grouping] log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise SystemExit(f"Command failed with exit code {return_code}. See log: {log_path}")


def prepare_pseudo_labels(args: argparse.Namespace, paths: GaussianGroupingPaths) -> dict[str, Any]:
    _require_dir(paths.staged_dataset_dir / "images", "staged Gaussian Grouping images")
    _require_dir(paths.staged_dataset_dir / f"images_{args.scale}", "staged Gaussian Grouping scaled images")
    _require_file(paths.gaussian_grouping_root / "script" / "prepare_pseudo_label.sh", "Gaussian Grouping pseudo-label script")
    if args.max_num_objects is not None and int(args.max_num_objects) > 255:
        raise SystemExit(
            "--max-num-objects cannot exceed 255 for the current Gaussian Grouping baseline, "
            "because pseudo labels are saved as 8-bit grayscale masks and the training config uses 256 classes. "
            "Use 240-250 for a high-cap test, or extend the mask/class pipeline for uint16 labels."
        )

    object_mask_dir = paths.staged_dataset_dir / "object_mask"
    deva_output_dir = (
        paths.gaussian_grouping_root
        / "Tracking-Anything-with-DEVA"
        / "example"
        / "output_gaussian_dataset"
        / paths.dataset_name
    )
    if object_mask_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Pseudo-label output already exists: {object_mask_dir}. Pass --overwrite to rerun.")
        _safe_remove_path(object_mask_dir)
    if deva_output_dir.exists() and args.overwrite:
        _safe_remove_path(deva_output_dir)

    command_env = _python_env(args.python)
    command_env["GG_DEVA_CHUNK_SIZE"] = str(int(args.deva_chunk_size))
    command_env["GG_DEVA_SIZE"] = str(int(args.deva_size))
    command_env["GG_SAM_PRED_IOU_THRESHOLD"] = str(float(args.sam_pred_iou_threshold))
    command_env["GG_SUPPRESS_SMALL_OBJECTS"] = "0" if args.preserve_small_objects else "1"
    if args.sam_num_points_per_side is not None:
        command_env["GG_SAM_NUM_POINTS_PER_SIDE"] = str(int(args.sam_num_points_per_side))
    if args.sam_num_points_per_batch is not None:
        command_env["GG_SAM_NUM_POINTS_PER_BATCH"] = str(int(args.sam_num_points_per_batch))
    if args.sam_overlap_threshold is not None:
        command_env["GG_SAM_OVERLAP_THRESHOLD"] = str(float(args.sam_overlap_threshold))
    if args.detection_every is not None:
        command_env["GG_DETECTION_EVERY"] = str(int(args.detection_every))
    if args.num_voting_frames is not None:
        command_env["GG_NUM_VOTING_FRAMES"] = str(int(args.num_voting_frames))
    if args.max_num_objects is not None:
        command_env["GG_MAX_NUM_OBJECTS"] = str(int(args.max_num_objects))

    command = ["bash", "script/prepare_pseudo_label.sh", paths.dataset_name, str(int(args.scale))]
    _run_command(
        command,
        cwd=paths.gaussian_grouping_root,
        log_path=paths.log_dir / "prepare_pseudo_labels.log",
        env=command_env,
    )
    mask_stats = _require_complete_object_masks(paths.staged_dataset_dir)
    summary = {
        "stage": "prepare_pseudo_labels",
        "createdAt": _now(),
        "method": "Original Gaussian Grouping DEVA/SAM pseudo-label preparation",
        "command": command,
        "parameters": {
            "scale": int(args.scale),
            "devaChunkSize": int(args.deva_chunk_size),
            "devaSize": int(args.deva_size),
            "samPredIouThreshold": float(args.sam_pred_iou_threshold),
            "preserveSmallObjects": bool(args.preserve_small_objects),
            "samNumPointsPerSide": args.sam_num_points_per_side,
            "samNumPointsPerBatch": args.sam_num_points_per_batch,
            "samOverlapThreshold": args.sam_overlap_threshold,
            "detectionEvery": args.detection_every,
            "numVotingFrames": args.num_voting_frames,
            "maxNumObjects": args.max_num_objects,
        },
        "outputs": {
            "objectMaskDir": str(object_mask_dir),
            "devaColorMaskDir": str(deva_output_dir / "Annotations_color"),
        },
        "stats": mask_stats,
    }
    _write_json(paths.summary_dir / "prepare_pseudo_labels_summary.json", summary)
    return summary


def train(args: argparse.Namespace, paths: GaussianGroupingPaths) -> dict[str, Any]:
    object_mask_dir = paths.staged_dataset_dir / "object_mask"
    if not object_mask_dir.exists():
        raise SystemExit(
            f"Gaussian Grouping object_mask directory does not exist: {object_mask_dir}. "
            "Run the prepare_pseudo_labels stage with --overwrite before training."
        )
    mask_stats = _require_complete_object_masks(paths.staged_dataset_dir)
    _require_file(paths.gaussian_grouping_root / "train.py", "Gaussian Grouping train.py")
    config_file = (
        args.config_file.expanduser().resolve()
        if args.config_file is not None
        else paths.gaussian_grouping_root / "config" / "gaussian_dataset" / "train.json"
    )
    _require_file(config_file, "Gaussian Grouping training config")

    if paths.model_output_dir.exists() and any(paths.model_output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Gaussian Grouping model output already exists: {paths.model_output_dir}. "
                "Pass --overwrite to replace it."
            )
        _safe_remove_path(paths.model_output_dir)
    paths.model_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        _python_path(args.python),
        "train.py",
        "-s",
        str(paths.staged_dataset_dir),
        "-r",
        str(int(args.scale)),
        "-m",
        str(paths.model_output_dir),
        "--config_file",
        str(config_file),
        "--data_device",
        str(args.data_device),
    ]
    if args.iterations is not None:
        command.extend(["--iterations", str(int(args.iterations))])
    if args.save_iterations:
        command.append("--save_iterations")
        command.extend(str(int(value)) for value in args.save_iterations)
    if args.test_iterations:
        command.append("--test_iterations")
        command.extend(str(int(value)) for value in args.test_iterations)
    if args.quiet:
        command.append("--quiet")
    if args.use_wandb:
        command.append("--use_wandb")

    _run_command(
        command,
        cwd=paths.gaussian_grouping_root,
        log_path=paths.log_dir / "train.log",
        env=_python_env(args.python),
    )
    summary = {
        "stage": "train",
        "createdAt": _now(),
        "method": "Original Gaussian Grouping training",
        "command": command,
        "parameters": {
            "scale": int(args.scale),
            "iterations": args.iterations,
            "saveIterations": args.save_iterations,
            "testIterations": args.test_iterations,
            "dataDevice": args.data_device,
        },
        "outputs": {
            "modelOutputDir": str(paths.model_output_dir),
            "pointCloudDir": str(paths.model_output_dir / "point_cloud"),
            "cfgArgs": str(paths.model_output_dir / "cfg_args"),
        },
        "stats": {"objectMasks": mask_stats},
    }
    _write_json(paths.summary_dir / "train_summary.json", summary)
    return summary


def render(args: argparse.Namespace, paths: GaussianGroupingPaths) -> dict[str, Any]:
    _require_file(paths.gaussian_grouping_root / "render.py", "Gaussian Grouping render.py")
    _require_file(paths.model_output_dir / "cfg_args", "Gaussian Grouping trained cfg_args")

    command = [
        _python_path(args.python),
        "render.py",
        "-m",
        str(paths.model_output_dir),
        "--num_classes",
        str(int(args.num_classes)),
    ]
    if args.render_iteration is not None:
        command.extend(["--iteration", str(int(args.render_iteration))])
    if args.skip_train_render:
        command.append("--skip_train")
    if args.skip_test_render:
        command.append("--skip_test")
    if args.quiet:
        command.append("--quiet")

    _run_command(
        command,
        cwd=paths.gaussian_grouping_root,
        log_path=paths.log_dir / "render.log",
        env=_python_env(args.python),
    )
    summary = {
        "stage": "render",
        "createdAt": _now(),
        "method": "Original Gaussian Grouping mask rendering",
        "command": command,
        "parameters": {
            "numClasses": int(args.num_classes),
            "renderIteration": args.render_iteration,
        },
        "outputs": {"modelOutputDir": str(paths.model_output_dir)},
    }
    _write_json(paths.summary_dir / "render_summary.json", summary)
    return summary


def check(args: argparse.Namespace, paths: GaussianGroupingPaths) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    statuses["gaussianGroupingRootExists"] = paths.gaussian_grouping_root.exists()
    statuses["sourceDatasetExists"] = paths.source_dataset_dir.exists()
    statuses["sourceImagesExists"] = (paths.source_dataset_dir / "images").exists()
    statuses["sourceSparseExists"] = (paths.source_dataset_dir / "sparse" / "0").exists()
    statuses["devaRepoExists"] = (paths.gaussian_grouping_root / "Tracking-Anything-with-DEVA").exists()
    statuses["prepareScriptExists"] = (paths.gaussian_grouping_root / "script" / "prepare_pseudo_label.sh").exists()
    statuses["trainScriptExists"] = (paths.gaussian_grouping_root / "train.py").exists()
    statuses["renderScriptExists"] = (paths.gaussian_grouping_root / "render.py").exists()
    deva_saves = paths.gaussian_grouping_root / "Tracking-Anything-with-DEVA" / "saves"
    checkpoint_files = [
        "DEVA-propagation.pth",
        "groundingdino_swint_ogc.pth",
        "sam_vit_h_4b8939.pth",
        "mobile_sam.pt",
        "GroundingDINO_SwinT_OGC.py",
    ]
    statuses["devaCheckpoints"] = {
        name: {
            "exists": (deva_saves / name).exists(),
            "sizeBytes": (deva_saves / name).stat().st_size if (deva_saves / name).exists() else 0,
            "path": str(deva_saves / name),
        }
        for name in checkpoint_files
    }
    for module in [
        "torch",
        "torchvision",
        "diff_gaussian_rasterization",
        "simple_knn",
        "cv2",
        "sklearn",
        "lpips",
        "deva",
        "segment_anything",
        "groundingdino",
        "transformers",
        "huggingface_hub",
    ]:
        proc = subprocess.run(
            [_python_path(args.python), "-c", f"import {module}; print('ok')"],
            cwd=str(paths.gaussian_grouping_root),
            env=_python_env(args.python),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        statuses[f"pythonImport:{module}"] = {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip()[-800:],
        }
    summary = {
        "stage": "check",
        "createdAt": _now(),
        "method": "Gaussian Grouping runtime readiness check",
        "inputs": {
            "modelDir": str(paths.model_dir),
            "sourceDatasetDir": str(paths.source_dataset_dir),
            "gaussianGroupingRoot": str(paths.gaussian_grouping_root),
            "python": _python_path(args.python),
        },
        "outputs": {"runDir": str(paths.run_dir), "summaryDir": str(paths.summary_dir)},
        "status": statuses,
    }
    _write_json(paths.summary_dir / "check_summary.json", summary)
    return summary


def _merge_summary(paths: GaussianGroupingPaths, stage_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summary = {
        "stage": "gaussian_grouping",
        "createdAt": _now(),
        "method": "Faithful Gaussian Grouping baseline staged for OMeGa DSLR data",
        "activeStage": args.stage,
        "activeStageResults": stage_results,
        "inputs": {
            "modelDir": str(paths.model_dir),
            "sourceDatasetDir": str(paths.source_dataset_dir),
            "gaussianGroupingRoot": str(paths.gaussian_grouping_root),
        },
        "parameters": {
            "datasetName": paths.dataset_name,
            "runName": _slug_token(args.run_name),
            "scale": int(args.scale),
            "numClasses": int(args.num_classes),
        },
        "outputs": {
            "runDir": str(paths.run_dir),
            "stagedDatasetDir": str(paths.staged_dataset_dir),
            "objectMaskDir": str(paths.staged_dataset_dir / "object_mask"),
            "modelOutputDir": str(paths.model_output_dir),
            "summaryDir": str(paths.summary_dir),
            "logDir": str(paths.log_dir),
        },
    }
    _write_json(paths.summary_path, summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gaussian Grouping faithfully on an OMeGa COLMAP dataset. "
            "This script does not consume OMeGa SAM2Object/SAI3D masks."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("check", "stage_dataset", "prepare_pseudo_labels", "train", "render", "all"),
        default="check",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--gaussian-grouping-root", type=Path, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--run-name", default="deva_scale4")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--save-iterations", nargs="*", type=int, default=None)
    parser.add_argument("--test-iterations", nargs="*", type=int, default=None)
    parser.add_argument("--render-iteration", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=256)
    parser.add_argument("--deva-chunk-size", type=int, default=4)
    parser.add_argument("--deva-size", type=int, default=480)
    parser.add_argument("--sam-pred-iou-threshold", type=float, default=0.7)
    parser.add_argument("--sam-num-points-per-side", type=int, default=None)
    parser.add_argument("--sam-num-points-per-batch", type=int, default=None)
    parser.add_argument("--sam-overlap-threshold", type=float, default=None)
    parser.add_argument("--detection-every", type=int, default=None)
    parser.add_argument("--num-voting-frames", type=int, default=None)
    parser.add_argument("--max-num-objects", type=int, default=None)
    parser.add_argument(
        "--preserve-small-objects",
        action="store_true",
        help=(
            "Do not pass DEVA's --suppress_small_objects flag during pseudo-label preparation. "
            "This usually keeps more SAM component masks instead of letting large masks absorb them."
        ),
    )
    parser.add_argument(
        "--data-device",
        choices=("cuda", "cpu"),
        default="cuda",
        help=(
            "Device used by Gaussian Grouping to store loaded camera RGB/mask tensors. "
            "Use cpu for high-resolution runs when CUDA memory is tight."
        ),
    )
    parser.add_argument("--skip-train-render", action="store_true")
    parser.add_argument("--skip-test-render", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if int(args.scale) <= 0:
        raise SystemExit("--scale must be positive")
    paths = _resolve_paths(args)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.summary_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)

    stage_results: list[dict[str, Any]] = []
    stages = ["stage_dataset", "prepare_pseudo_labels", "train", "render"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if stage == "check":
            stage_results.append(check(args, paths))
        elif stage == "stage_dataset":
            stage_results.append(stage_dataset(args, paths))
        elif stage == "prepare_pseudo_labels":
            stage_results.append(prepare_pseudo_labels(args, paths))
        elif stage == "train":
            stage_results.append(train(args, paths))
        elif stage == "render":
            stage_results.append(render(args, paths))
        else:
            raise AssertionError(stage)
    summary = _merge_summary(paths, stage_results, args)
    print("=== Gaussian Grouping For OMeGa ===")
    print(f"Stage: {args.stage}")
    print(f"Run dir: {summary['outputs']['runDir']}")
    print(f"Staged dataset: {summary['outputs']['stagedDatasetDir']}")
    print(f"Model output: {summary['outputs']['modelOutputDir']}")
    print(f"Summary: {paths.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
