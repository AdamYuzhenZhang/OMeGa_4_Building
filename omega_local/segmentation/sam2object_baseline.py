"""SAM2Object-style view-mask baseline for OMeGa DSLR captures.

The upstream SAM2Object repository is ScanNet-oriented and its public
`seg_tracking.py` has hardcoded paths. This adapter keeps the method as an
external baseline, but starts from the common per-view proposal initializer
instead of restaging the DSLR capture itself. It runs the useful SAM2Object
parts for our current goal: SAM2 keyframe masks, forward and reverse view
propagation, then the SAM2Object multi-view graph consolidation that turns local
tracked masks into consistent scene object IDs.

The output is intentionally view-first:

    <model_dir>/segmentation/baselines/sam2object/
      dataset/color_images_cluster/<scene>/
      tracking/{forward,reverse,merge}/masks/
      view_masks/masks/
      graph_consistency/
      view_masks_consistent/masks/
      view_masks/predictions.jsonl
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
import trimesh

from omega_local.remesh.mesh_clean import resolve_latest_omega_mesh
from omega_local.segmentation.proposal_source import resolve_proposal_source


OMEGA_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = OMEGA_ROOT.parent


@dataclass(frozen=True)
class SAM2ObjectPaths:
    baseline_dir: Path
    dataset_dir: Path
    forward_frame_dir: Path
    reverse_frame_dir: Path
    posed_dir: Path
    tracking_dir: Path
    forward_dir: Path
    reverse_dir: Path
    merge_dir: Path
    view_mask_dir: Path
    view_mask_masks_dir: Path
    view_mask_overlay_dir: Path
    graph_dir: Path
    consistent_view_mask_dir: Path
    consistent_view_mask_masks_dir: Path
    consistent_view_mask_overlay_dir: Path
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
    def summary(self) -> Path:
        return self.baseline_dir / "baseline_summary.json"

    @property
    def scene_list(self) -> Path:
        return self.dataset_dir / "sam2object_scene_val.txt"


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    return "".join(out).strip("_") or "sam2object"


def resolve_paths(
    model_dir: Path,
    *,
    baseline_name: str = "sam2object",
    scene_name: str | None = None,
    output_dir: Path | None = None,
) -> SAM2ObjectPaths:
    baseline_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else model_dir / "segmentation" / "baselines" / _slug_token(baseline_name)
    )
    scene_slug = _slug_token(scene_name or f"{model_dir.parent.parent.name}_{model_dir.name}")
    dataset_dir = baseline_dir / "dataset"
    return SAM2ObjectPaths(
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        forward_frame_dir=dataset_dir / "color_images_cluster" / scene_slug,
        reverse_frame_dir=dataset_dir / "color_images_cluster_reverse" / scene_slug,
        posed_dir=dataset_dir / "posed_images" / scene_slug,
        tracking_dir=baseline_dir / "tracking",
        forward_dir=baseline_dir / "tracking" / "forward",
        reverse_dir=baseline_dir / "tracking" / "reverse",
        merge_dir=baseline_dir / "tracking" / "merge",
        view_mask_dir=baseline_dir / "view_masks",
        view_mask_masks_dir=baseline_dir / "view_masks" / "masks",
        view_mask_overlay_dir=baseline_dir / "view_masks" / "overlays",
        graph_dir=baseline_dir / "graph_consistency",
        consistent_view_mask_dir=baseline_dir / "view_masks_consistent",
        consistent_view_mask_masks_dir=baseline_dir / "view_masks_consistent" / "masks",
        consistent_view_mask_overlay_dir=baseline_dir / "view_masks_consistent" / "overlays",
        summary_dir=baseline_dir / "summaries",
        log_dir=baseline_dir / "logs",
        scene_name=scene_slug,
    )


def _require_file(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _resolve_sam2object_root(path: Path | None) -> Path:
    root = THIRD_PARTY_ROOT / "SAM2Object" if path is None else path.expanduser().resolve()
    _require_dir(root, "SAM2Object root")
    _require_dir(root / "segtrack", "SAM2Object segtrack directory")
    _require_file(root / "segtrack" / "sam2" / "build_sam.py", "SAM2Object vendored SAM2 build_sam.py")
    return root


def _resolve_sam2_checkpoint(args: argparse.Namespace, sam2object_root: Path) -> Path:
    if args.sam2_checkpoint is not None:
        raw = args.sam2_checkpoint.expanduser()
        return raw.resolve() if raw.is_absolute() else (sam2object_root / raw).resolve()
    candidates = [
        THIRD_PARTY_ROOT / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt",
        sam2object_root / "segtrack" / "checkpoints" / "sam2.1_hiera_large.pt",
        sam2object_root / "segtrack" / "checkpoints" / "sam2_hiera_large.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_input_mesh(model_dir: Path, mesh: Path | None, iteration: int) -> Path:
    if mesh is not None:
        raw = mesh.expanduser()
        candidates = [raw.resolve()] if raw.is_absolute() else [raw.resolve(), (model_dir / raw).resolve()]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate
        raise SystemExit(f"Input mesh does not exist. Tried: {[str(candidate) for candidate in candidates]}")
    healed = model_dir / "remesh" / "local" / "preclean_healed_mesh.ply"
    if healed.exists() and healed.is_file():
        return healed
    preclean = model_dir / "remesh" / "local" / "preclean_mesh.ply"
    if preclean.exists() and preclean.is_file():
        return preclean
    return resolve_latest_omega_mesh(model_dir, iteration=iteration)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        parts = [geom for geom in mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not parts:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(parts)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {path}: {type(mesh)!r}")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError(f"SAM2Object graph consolidation expects a triangle mesh: {path}")
    return mesh


def prepare(args: argparse.Namespace, paths: SAM2ObjectPaths) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    proposal_source = resolve_proposal_source(
        model_dir,
        proposal_source_name=str(args.proposal_source_name),
        proposal_source_dir=args.proposal_source_dir,
    )
    source_rows = proposal_source.rows(int(args.source_frame_stride), int(args.max_frames))

    if paths.dataset_dir.exists() and any(paths.dataset_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"SAM2Object dataset already exists: {paths.dataset_dir}. Pass --overwrite to replace it.")
    if args.overwrite and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)

    for directory in [paths.forward_frame_dir, paths.reverse_frame_dir, paths.posed_dir, paths.summary_dir, paths.log_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    rgbs: list[np.ndarray] = []
    out_h = out_w = 0
    for out_index, row in enumerate(source_rows):
        source_frame_id = proposal_source.frame_id(row, out_index)
        image_path = proposal_source.image_path(row)
        pose_src = proposal_source.pose_path(row)
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Could not read proposal source RGB frame: {image_path}")
        height, width = rgb.shape[:2]
        if out_index == 0:
            out_h, out_w = int(height), int(width)
        elif (height, width) != (out_h, out_w):
            raise SystemExit(
                f"Proposal source frame shape differs from the first frame: {image_path} -> {(height, width)} vs {(out_h, out_w)}"
            )
        rgbs.append(rgb)
        frame_name = f"{out_index}.jpg"
        color_path = paths.forward_frame_dir / frame_name
        posed_color_path = paths.posed_dir / frame_name
        for out_path in [color_path, posed_color_path]:
            if not cv2.imwrite(str(out_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
                raise RuntimeError(f"Failed to write RGB image: {out_path}")
        pose_path = paths.posed_dir / f"{out_index}.txt"
        shutil.copyfile(pose_src, pose_path)
        manifest.append(
            {
                "sam2objectFrameId": int(out_index),
                "sourceFrameId": int(source_frame_id),
                "scanID": str(row.get("scanID", "")),
                "frameID": int(row.get("frameID", source_frame_id)),
                "imageName": str(row.get("imageName", image_path.name)),
                "sourceImagePath": str(image_path),
                "colorPath": str(color_path.relative_to(paths.baseline_dir)),
                "reverseColorPath": str((paths.reverse_frame_dir / frame_name).relative_to(paths.baseline_dir)),
                "posePath": str(pose_path.relative_to(paths.baseline_dir)),
                "width": int(width),
                "height": int(height),
                "fx": float(row["fx"]),
                "fy": float(row["fy"]),
                "cx": float(row["cx"]),
                "cy": float(row["cy"]),
            }
        )
        print(
            f"[sam2object prepare {out_index + 1:04d}/{len(source_rows):04d}] "
            f"source={source_frame_id}"
        )

    for rev_index, rgb in enumerate(reversed(rgbs)):
        out_path = paths.reverse_frame_dir / f"{rev_index}.jpg"
        if not cv2.imwrite(str(out_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
            raise RuntimeError(f"Failed to write reverse RGB image: {out_path}")

    paths.scene_list.write_text(paths.scene_name + "\n", encoding="utf-8")
    _write_jsonl(paths.frame_manifest, manifest)
    summary = {
        "stage": "prepare",
        "timestampUtc": _now(),
        "method": "Stage SAM2Object video frames from the common 2D proposal source.",
        "modelDir": str(model_dir),
        "proposalSourceName": str(args.proposal_source_name),
        "proposalSourceDir": str(proposal_source.baseline_dir),
        "proposalSourceFrameManifest": str(proposal_source.frame_manifest),
        "sceneName": paths.scene_name,
        "frameCount": int(len(manifest)),
        "sourceFrameStride": int(args.source_frame_stride),
        "maxFrames": int(args.max_frames),
        "imageWidth": int(out_w),
        "imageHeight": int(out_h),
        "outputs": {
            "datasetDir": str(paths.dataset_dir),
            "forwardFrameDir": str(paths.forward_frame_dir),
            "reverseFrameDir": str(paths.reverse_frame_dir),
            "posedDir": str(paths.posed_dir),
            "frameManifest": str(paths.frame_manifest),
            "sceneList": str(paths.scene_list),
        },
    }
    _write_json(paths.dataset_summary, summary)
    _write_json(paths.summary_dir / "prepare_summary.json", summary)
    return summary


def _sam2object_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    entries = [str(root / "segtrack")]
    old = env.get("PYTHONPATH")
    if old:
        entries.append(old)
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def check(args: argparse.Namespace, paths: SAM2ObjectPaths, sam2object_root: Path) -> dict[str, Any]:
    checkpoint = _resolve_sam2_checkpoint(args, sam2object_root)
    code = (
        "import cv2, hydra, numpy, omegaconf, supervision, torch; "
        "import sam2; from sam2.build_sam import build_sam2; "
        "print('sam2object dependency check ok')"
    )
    ok = True
    error = ""
    import subprocess

    completed = subprocess.run(
        [args.python, "-c", code],
        cwd=str(sam2object_root / "segtrack"),
        env=_sam2object_env(sam2object_root),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    log_path = paths.log_dir / "check.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "COMMAND:\n"
        + " ".join([args.python, "-c", code])
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        ok = False
        error = f"SAM2Object dependency check failed. Log: {log_path}"
    if not checkpoint.exists():
        ok = False
        error = f"Missing SAM2 checkpoint: {checkpoint}"
    summary = {
        "stage": "check",
        "timestampUtc": _now(),
        "ok": bool(ok),
        "sam2objectRoot": str(sam2object_root),
        "sam2Checkpoint": str(checkpoint),
        "sam2Config": str(args.sam2_config),
        "log": str(log_path),
        "error": error,
    }
    _write_json(paths.summary_dir / "check_summary.json", summary)
    if not ok and not bool(args.keep_going_after_check):
        raise SystemExit(error)
    return summary


def _frame_names(video_dir: Path) -> list[str]:
    names = [p.name for p in video_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    names.sort(key=lambda name: int(Path(name).stem))
    return names


def _smooth_signal(x: np.ndarray, window_len: int = 13, window: str = "hanning") -> np.ndarray:
    if x.size < 3:
        return x.astype(np.float64, copy=True)
    window_len = min(max(int(window_len), 3), int(x.size))
    if window_len % 2 == 0:
        window_len -= 1
    if window_len < 3:
        return x.astype(np.float64, copy=True)
    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = np.ones(window_len, dtype=np.float64) if window == "flat" else getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode="same")
    return y[window_len - 1 : -window_len + 1]


def _extract_keyframes(video_dir: Path, frame_names: list[str], args: argparse.Namespace) -> list[int]:
    n_frames = len(frame_names)
    if n_frames <= 1:
        return [0]
    if args.keyframe_mode == "fixed":
        stride = max(int(args.keyframe_stride), 1)
        keyframes = list(range(0, n_frames, stride))
    else:
        diffs: list[float] = []
        prev = None
        for name in frame_names:
            image = cv2.imread(str(video_dir / name), cv2.IMREAD_COLOR)
            if image is None:
                continue
            luv = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
            if prev is not None:
                diffs.append(float(np.mean(cv2.absdiff(luv, prev))))
            prev = luv
        keyframes = []
        if diffs:
            smoothed = _smooth_signal(np.asarray(diffs, dtype=np.float64), int(args.keyframe_window))
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                    keyframes.append(i)

    keyframes.append(0)
    keyframes.append(n_frames - 1)
    keyframes = sorted(set(int(k) for k in keyframes if 0 <= int(k) < n_frames))
    keyframes = _filter_close_keyframes(keyframes, int(args.min_keyframe_gap))
    keyframes = _insert_large_gap_keyframes(keyframes, int(args.max_keyframe_gap))
    keyframes = sorted(set(keyframes))
    if keyframes[0] != 0:
        keyframes.insert(0, 0)
    if keyframes[-1] != n_frames - 1:
        keyframes.append(n_frames - 1)
    return keyframes


def _filter_close_keyframes(keyframes: list[int], threshold: int) -> list[int]:
    if threshold <= 0 or len(keyframes) <= 2:
        return keyframes
    out = [keyframes[0]]
    for value in keyframes[1:-1]:
        if value - out[-1] >= threshold:
            out.append(value)
    if keyframes[-1] != out[-1]:
        out.append(keyframes[-1])
    return out


def _insert_large_gap_keyframes(keyframes: list[int], threshold: int) -> list[int]:
    if threshold <= 0 or len(keyframes) < 2:
        return keyframes
    out: list[int] = []
    for a, b in zip(keyframes[:-1], keyframes[1:], strict=False):
        out.append(a)
        if b - a > threshold:
            out.append(int(round((a + b) * 0.5)))
    out.append(keyframes[-1])
    return out


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    inter = float(np.count_nonzero(a & b))
    union = float(np.count_nonzero(a | b))
    return inter / union if union > 0.0 else 0.0


def _generated_masks_to_records(masks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in masks:
        mask = np.asarray(item.get("segmentation"), dtype=bool)
        area = int(np.count_nonzero(mask))
        if area < int(args.min_mask_area):
            continue
        predicted_iou = float(item.get("predicted_iou", 0.0))
        stability = float(item.get("stability_score", 0.0))
        score = predicted_iou * 0.7 + stability * 0.3
        records.append({"mask": mask, "area": area, "predicted_iou": predicted_iou, "stability": stability, "score": score})
    records.sort(key=lambda item: (float(item["score"]), int(item["area"])), reverse=True)
    if int(args.max_masks_per_keyframe) > 0:
        records = records[: int(args.max_masks_per_keyframe)]
    return records


def _assign_object_ids(
    records: list[dict[str, Any]],
    previous: dict[int, np.ndarray],
    next_id: int,
    iou_threshold: float,
) -> tuple[list[tuple[int, np.ndarray]], int]:
    assigned: list[tuple[int, np.ndarray]] = []
    used_previous: set[int] = set()
    occupied: np.ndarray | None = None
    for record in records:
        mask = np.asarray(record["mask"], dtype=bool)
        if occupied is None:
            occupied = np.zeros(mask.shape, dtype=bool)
        clean = mask & ~occupied
        if np.count_nonzero(clean) < 25:
            continue
        best_id = 0
        best_iou = 0.0
        for object_id, prev_mask in previous.items():
            if object_id in used_previous:
                continue
            score = _mask_iou(clean, prev_mask)
            if score > best_iou:
                best_iou = score
                best_id = int(object_id)
        if best_id > 0 and best_iou >= float(iou_threshold):
            object_id = best_id
            used_previous.add(object_id)
        else:
            next_id += 1
            object_id = next_id
        assigned.append((object_id, clean))
        occupied |= clean
    return assigned, next_id


def _label_map_from_logits(object_ids: list[int], logits: Any) -> np.ndarray:
    import torch

    if isinstance(logits, torch.Tensor):
        arr = logits.detach().float().cpu().numpy()
    else:
        arr = np.asarray(logits, dtype=np.float32)
    if arr.ndim == 4:
        arr = arr[:, 0]
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError(f"Unexpected SAM2 logits shape: {arr.shape}")
    positive = arr > 0.0
    max_index = np.argmax(arr, axis=0)
    has_label = np.any(positive, axis=0)
    out = np.zeros(arr.shape[1:], dtype=np.uint16)
    ids = np.asarray(object_ids, dtype=np.int64)
    chosen = ids[max_index[has_label]]
    out[has_label] = np.clip(chosen, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    return out


def _masks_from_label_map(label_map: np.ndarray) -> dict[int, np.ndarray]:
    labels: dict[int, np.ndarray] = {}
    for value in np.unique(label_map):
        label = int(value)
        if label > 0:
            labels[label] = label_map == label
    return labels


def _save_label_map(path: Path, label_map: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, label_map.astype(np.uint16, copy=False))


def _run_direction(
    *,
    direction_name: str,
    video_dir: Path,
    output_dir: Path,
    keyframes: list[int],
    args: argparse.Namespace,
    video_predictor: Any,
    mask_generator: Any,
) -> dict[str, Any]:
    frame_names = _frame_names(video_dir)
    if not frame_names:
        raise SystemExit(f"No frames found for SAM2Object tracking: {video_dir}")
    masks_dir = output_dir / "masks"
    keyframe_dir = output_dir / "keyframes"
    if output_dir.exists() and bool(args.overwrite):
        shutil.rmtree(output_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    inference_state = video_predictor.init_state(
        video_path=str(video_dir),
        offload_video_to_cpu=bool(args.offload_video_to_cpu),
        async_loading_frames=True,
    )
    previous_masks: dict[int, np.ndarray] = {}
    next_object_id = 0
    saved_indices: set[int] = set()
    keyframe_rows: list[dict[str, Any]] = []

    for segment_index, start_idx in enumerate(keyframes[:-1]):
        end_idx = keyframes[segment_index + 1]
        if end_idx < start_idx:
            continue
        image_bgr = cv2.imread(str(video_dir / frame_names[start_idx]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read keyframe image: {video_dir / frame_names[start_idx]}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        generated = mask_generator.generate(image_rgb)
        records = _generated_masks_to_records(generated, args)
        assigned, next_object_id = _assign_object_ids(
            records,
            previous_masks,
            next_object_id,
            float(args.keyframe_iou_threshold),
        )
        keyframe_rows.append(
            {
                "direction": direction_name,
                "segmentIndex": int(segment_index),
                "startFrame": int(start_idx),
                "endFrame": int(end_idx),
                "rawMaskCount": int(len(generated)),
                "usedMaskCount": int(len(assigned)),
                "objectIds": [int(object_id) for object_id, _ in assigned],
            }
        )
        if not assigned:
            continue

        video_predictor.reset_state(inference_state)
        for object_id, mask in assigned:
            video_predictor.add_new_mask(inference_state, start_idx, int(object_id), mask)

        max_to_track = max(int(end_idx - start_idx + 1), 1)
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(
            inference_state,
            max_frame_num_to_track=max_to_track,
            start_frame_idx=start_idx,
        ):
            out_frame_idx = int(out_frame_idx)
            if out_frame_idx > end_idx:
                continue
            label_map = _label_map_from_logits([int(value) for value in out_obj_ids], out_mask_logits)
            _save_label_map(masks_dir / f"mask_{out_frame_idx}.npy", label_map)
            saved_indices.add(out_frame_idx)
            if out_frame_idx == end_idx:
                previous_masks = _masks_from_label_map(label_map)

    for idx in range(len(frame_names)):
        path = masks_dir / f"mask_{idx}.npy"
        if not path.exists():
            image = cv2.imread(str(video_dir / frame_names[idx]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            _save_label_map(path, np.zeros(image.shape[:2], dtype=np.uint16))

    _write_json(keyframe_dir / "keyframes.json", {"direction": direction_name, "keyframes": keyframes, "segments": keyframe_rows})
    return {
        "direction": direction_name,
        "frameCount": int(len(frame_names)),
        "keyframes": [int(value) for value in keyframes],
        "savedFrameCount": int(len(saved_indices)),
        "segments": keyframe_rows,
        "masksDir": str(masks_dir),
    }


def _merge_forward_reverse(paths: SAM2ObjectPaths, frame_count: int, args: argparse.Namespace) -> dict[str, Any]:
    masks_dir = paths.merge_dir / "masks"
    if paths.merge_dir.exists() and bool(args.overwrite):
        shutil.rmtree(paths.merge_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    next_id = 0
    reverse_id_map: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    for index in range(frame_count):
        f_path = paths.forward_dir / "masks" / f"mask_{index}.npy"
        r_path = paths.reverse_dir / "masks" / f"mask_{frame_count - 1 - index}.npy"
        forward = np.load(f_path) if f_path.exists() else None
        reverse_rev = np.load(r_path) if r_path.exists() else None
        if forward is None and reverse_rev is None:
            continue
        if forward is None:
            forward = np.zeros_like(reverse_rev, dtype=np.uint16)
        if reverse_rev is None:
            reverse = np.zeros_like(forward, dtype=np.uint16)
        else:
            reverse = reverse_rev
        merged = forward.astype(np.uint16, copy=True)
        next_id = max(next_id, int(np.max(merged)))
        added = 0
        for value in np.unique(reverse):
            reverse_id = int(value)
            if reverse_id <= 0:
                continue
            region = reverse == reverse_id
            area = int(np.count_nonzero(region))
            if area <= 0:
                continue
            hole = merged == 0
            add_region = region & hole
            add_area = int(np.count_nonzero(add_region))
            if add_area / max(area, 1) < float(args.reverse_hole_fill_ratio):
                continue
            if reverse_id not in reverse_id_map:
                next_id += 1
                reverse_id_map[reverse_id] = next_id
            merged[add_region] = np.uint16(min(reverse_id_map[reverse_id], np.iinfo(np.uint16).max))
            added += 1
        _save_label_map(masks_dir / f"mask_{index}.npy", merged)
        labels = np.unique(merged)
        rows.append(
            {
                "frameIndex": int(index),
                "labelCount": int(np.count_nonzero(labels > 0)),
                "coverage": float(np.count_nonzero(merged > 0) / max(merged.size, 1)),
                "reverseComponentsAdded": int(added),
            }
        )
    summary = {
        "stage": "merge",
        "timestampUtc": _now(),
        "frameCount": int(frame_count),
        "reverseHoleFillRatio": float(args.reverse_hole_fill_ratio),
        "masksDir": str(masks_dir),
        "frames": rows,
    }
    _write_json(paths.summary_dir / "merge_summary.json", summary)
    return summary


def _palette(labels: np.ndarray) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(47)
    table: dict[int, np.ndarray] = {}
    for label in np.unique(labels.astype(np.int64)):
        label_int = int(label)
        if label_int == 0:
            table[label_int] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            table[label_int] = rng.integers(35, 255, size=3, dtype=np.uint8)
    return table


def _overlay(rgb_bgr: np.ndarray, labels: np.ndarray, alpha: float = 0.50) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    out = rgb.astype(np.float32, copy=True)
    table = _palette(labels)
    for label, color in table.items():
        if label == 0:
            continue
        mask = labels == label
        out[mask] = (1.0 - alpha) * out[mask] + alpha * color.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _write_view_masks(paths: SAM2ObjectPaths, frame_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if paths.view_mask_dir.exists() and bool(args.overwrite):
        shutil.rmtree(paths.view_mask_dir)
    paths.view_mask_masks_dir.mkdir(parents=True, exist_ok=True)
    paths.view_mask_overlay_dir.mkdir(parents=True, exist_ok=True)
    pred_rows: list[dict[str, Any]] = []
    for row in frame_rows:
        index = int(row["sam2objectFrameId"])
        mask = np.load(paths.merge_dir / "masks" / f"mask_{index}.npy").astype(np.uint16, copy=False)
        png_path = paths.view_mask_masks_dir / f"{index:06d}.png"
        npy_path = paths.view_mask_masks_dir / f"{index:06d}.npy"
        cv2.imwrite(str(png_path), mask)
        np.save(npy_path, mask)
        rgb_path = paths.baseline_dir / str(row["colorPath"])
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        overlay_path = paths.view_mask_overlay_dir / f"{index:06d}.png"
        if rgb_bgr is not None:
            overlay = cv2.cvtColor(_overlay(rgb_bgr, mask), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(overlay_path), overlay)
        labels = np.unique(mask)
        pred_rows.append(
            {
                "sam2objectFrameId": int(index),
                "scanID": row.get("scanID", ""),
                "frameID": int(row.get("frameID", index)),
                "imageName": row.get("imageName", ""),
                "maskPng": str(png_path),
                "maskNpy": str(npy_path),
                "overlay": str(overlay_path),
                "labelCount": int(np.count_nonzero(labels > 0)),
                "coverage": float(np.count_nonzero(mask > 0) / max(mask.size, 1)),
            }
        )
    predictions = paths.view_mask_dir / "predictions.jsonl"
    _write_jsonl(predictions, pred_rows)
    summary = {
        "stage": "normalize",
        "timestampUtc": _now(),
        "frameCount": int(len(pred_rows)),
        "outputs": {
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskMasksDir": str(paths.view_mask_masks_dir),
            "viewMaskOverlayDir": str(paths.view_mask_overlay_dir),
            "predictions": str(predictions),
        },
        "coverage": {
            "min": float(min((row["coverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["coverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["coverage"] for row in pred_rows), default=0.0)),
        },
    }
    _write_json(paths.summary_dir / "normalize_summary.json", summary)
    return summary


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(int(size), dtype=np.int64)
        self.rank = np.zeros(int(size), dtype=np.uint8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        if parent != item:
            self.parent[item] = self.find(parent)
        return int(self.parent[item])

    def union(self, a: int, b: int) -> None:
        ra = self.find(int(a))
        rb = self.find(int(b))
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def labels(self, active: np.ndarray | None = None) -> np.ndarray:
        roots = np.asarray([self.find(i) for i in range(self.parent.size)], dtype=np.int64)
        if active is not None:
            roots = np.where(active, roots, -1)
        unique = np.unique(roots[roots >= 0])
        out = np.zeros(roots.shape[0], dtype=np.int32)
        for label, root in enumerate(unique, start=1):
            out[roots == root] = int(label)
        return out


def _parse_threshold_schedule(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(values) == 3 and values[2] >= 2.0 and float(values[2]).is_integer():
        return [float(v) for v in np.linspace(values[0], values[1], int(values[2]))]
    if not values:
        raise SystemExit("--graph-thres-connect must contain at least one threshold")
    return values


def _mesh_vertex_edges(faces: np.ndarray) -> np.ndarray:
    faces_i = np.asarray(faces, dtype=np.int64)
    edges = np.vstack([faces_i[:, [0, 1]], faces_i[:, [1, 2]], faces_i[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0).astype(np.int64, copy=False)


def _build_raycast_scene(mesh: trimesh.Trimesh) -> Any:
    import open3d as o3d

    legacy = o3d.geometry.TriangleMesh()
    legacy.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    legacy.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)
    return scene


def _render_depth_m(
    scene: Any,
    *,
    pose_world_t_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    chunk_rows: int,
) -> np.ndarray:
    import open3d as o3d

    pose = np.asarray(pose_world_t_cam, dtype=np.float64)
    world_to_cam = np.linalg.inv(pose)
    rot = pose[:3, :3]
    origin = pose[:3, 3].astype(np.float32)
    depth = np.zeros((height, width), dtype=np.float32)

    for y0 in range(0, height, max(int(chunk_rows), 1)):
        y1 = min(height, y0 + max(int(chunk_rows), 1))
        yy, xx = np.mgrid[y0:y1, 0:width].astype(np.float32)
        dirs_cam = np.stack(
            [
                (xx + 0.5 - float(cx)) / max(float(fx), 1e-6),
                (yy + 0.5 - float(cy)) / max(float(fy), 1e-6),
                np.ones_like(xx, dtype=np.float32),
            ],
            axis=-1,
        ).reshape(-1, 3)
        dirs_world = dirs_cam @ rot.T.astype(np.float32)
        dirs_world = dirs_world / np.maximum(np.linalg.norm(dirs_world, axis=1, keepdims=True), 1e-8)
        origins = np.broadcast_to(origin.reshape(1, 3), dirs_world.shape)
        rays = np.concatenate([origins, dirs_world.astype(np.float32)], axis=1)
        answer = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        t_hit = answer["t_hit"].numpy().reshape(-1)
        hit = np.isfinite(t_hit)
        if not np.any(hit):
            continue
        points_world = origins[hit].astype(np.float64) + dirs_world[hit].astype(np.float64) * t_hit[hit, None]
        points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
        points_cam = (world_to_cam @ points_h.T).T[:, :3]
        z = points_cam[:, 2]
        z_valid = z > 0.0
        flat = depth[y0:y1].reshape(-1)
        hit_indices = np.nonzero(hit)[0][z_valid]
        flat[hit_indices] = z[z_valid].astype(np.float32)
    return depth


def _project_vertices(
    vertices: np.ndarray,
    pose_world_t_cam: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose = np.asarray(pose_world_t_cam, dtype=np.float64)
    world_to_cam = np.linalg.inv(pose)
    points_h = np.concatenate([vertices.astype(np.float64), np.ones((vertices.shape[0], 1), dtype=np.float64)], axis=1)
    points_cam = (world_to_cam @ points_h.T).T[:, :3]
    z = points_cam[:, 2]
    denom = np.maximum(z, 1e-8)
    u = np.rint(points_cam[:, 0] * float(fx) / denom + float(cx)).astype(np.int32)
    v = np.rint(points_cam[:, 1] * float(fy) / denom + float(cy)).astype(np.int32)
    return z.astype(np.float32), u, v


def _project_mesh_to_sam2object_masks(
    *,
    paths: SAM2ObjectPaths,
    mesh: trimesh.Trimesh,
    frame_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    scene = _build_raycast_scene(mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    graph_stride = max(int(args.graph_view_stride), 1)
    graph_frame_ids = [int(row["sam2objectFrameId"]) for row in frame_rows[::graph_stride]]
    graph_col = {frame_id: col for col, frame_id in enumerate(graph_frame_ids)}
    point_labels = np.zeros((vertices.shape[0], len(graph_frame_ids)), dtype=np.uint16)
    point_seen = np.zeros((vertices.shape[0], len(graph_frame_ids)), dtype=bool)
    projection_cache: list[dict[str, Any]] = []
    frame_graph_stats: list[dict[str, Any]] = []

    for row_index, row in enumerate(frame_rows):
        frame_id = int(row["sam2objectFrameId"])
        width = int(row["width"])
        height = int(row["height"])
        pose = np.loadtxt(paths.baseline_dir / str(row["posePath"]), dtype=np.float64)
        mask_path = paths.merge_dir / "masks" / f"mask_{frame_id}.npy"
        mask = np.load(_require_file(mask_path, "SAM2Object merged mask")).astype(np.uint16, copy=False)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        depth = _render_depth_m(
            scene,
            pose_world_t_cam=pose,
            fx=float(row["fx"]),
            fy=float(row["fy"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            width=width,
            height=height,
            chunk_rows=int(args.graph_depth_chunk_rows),
        )
        z, u, v = _project_vertices(
            vertices,
            pose,
            fx=float(row["fx"]),
            fy=float(row["fy"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
        )
        bounded = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        valid_indices = np.nonzero(bounded)[0]
        sampled_depth = np.zeros(valid_indices.shape[0], dtype=np.float32)
        if valid_indices.size:
            sampled_depth = depth[v[valid_indices], u[valid_indices]]
        visible_local = sampled_depth > 0.0
        if valid_indices.size:
            visible_local &= np.isclose(
                z[valid_indices],
                sampled_depth,
                rtol=float(args.graph_visibility_rtol),
                atol=float(args.graph_visibility_atol),
            )
        visible_indices = valid_indices[visible_local]
        labels = mask[v[visible_indices], u[visible_indices]] if visible_indices.size else np.zeros(0, dtype=np.uint16)
        if frame_id in graph_col:
            col = graph_col[frame_id]
            point_seen[visible_indices, col] = True
            point_labels[visible_indices, col] = labels.astype(np.uint16, copy=False)
            frame_graph_stats.append(
                {
                    "frameId": int(frame_id),
                    "visibleVertexCount": int(visible_indices.size),
                    "positiveVertexCount": int(np.count_nonzero(labels > 0)),
                    "labelCount": int(np.max(mask)) if mask.size else 0,
                }
            )
        projection_cache.append(
            {
                "frameId": int(frame_id),
                "indices": visible_indices.astype(np.int32, copy=False),
                "u": u[visible_indices].astype(np.int32, copy=False),
                "v": v[visible_indices].astype(np.int32, copy=False),
                "labels": labels.astype(np.uint16, copy=False),
            }
        )
        print(
            f"[consolidate project {row_index + 1:04d}/{len(frame_rows):04d}] "
            f"frame={frame_id} visible={visible_indices.size} positive={int(np.count_nonzero(labels > 0))}"
        )
    return point_labels, point_seen, projection_cache, graph_frame_ids, frame_graph_stats


def _edge_affinity_scores(
    edges: np.ndarray,
    point_labels: np.ndarray,
    point_seen: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(edges.shape[0], dtype=np.float32)
    confidence = np.zeros(edges.shape[0], dtype=np.float32)
    chunk = max(int(args.graph_edge_chunk), 1)
    for start in range(0, edges.shape[0], chunk):
        stop = min(edges.shape[0], start + chunk)
        ee = edges[start:stop]
        a = ee[:, 0]
        b = ee[:, 1]
        la = point_labels[a]
        lb = point_labels[b]
        shared = point_seen[a] & point_seen[b]
        same = shared & (la == lb)
        conf = shared.sum(axis=1).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = same.sum(axis=1).astype(np.float32) / np.maximum(conf, 1.0)
        confidence[start:stop] = conf
        scores[start:stop] = score
    return scores, confidence


def _labels_from_unions(vertex_count: int, edges: np.ndarray, active: np.ndarray, mask: np.ndarray) -> np.ndarray:
    uf = _UnionFind(vertex_count)
    selected = edges[mask]
    for a, b in selected:
        uf.union(int(a), int(b))
    return uf.labels(active=active)


def _component_edge_scores(
    vertex_labels: np.ndarray,
    edges: np.ndarray,
    edge_scores: np.ndarray,
    edge_confidence: np.ndarray,
    min_shared_views: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    accum: dict[tuple[int, int], list[float]] = {}
    labels_a = vertex_labels[edges[:, 0]]
    labels_b = vertex_labels[edges[:, 1]]
    valid = (labels_a > 0) & (labels_b > 0) & (labels_a != labels_b) & (edge_confidence >= int(min_shared_views))
    for comp_a, comp_b, score, conf in zip(
        labels_a[valid],
        labels_b[valid],
        edge_scores[valid],
        edge_confidence[valid],
        strict=False,
    ):
        lo = int(min(comp_a, comp_b))
        hi = int(max(comp_a, comp_b))
        entry = accum.setdefault((lo, hi), [0.0, 0.0])
        entry[0] += float(score) * float(conf)
        entry[1] += float(conf)
    if not accum:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        )
    comp_edges: list[tuple[int, int]] = []
    scores: list[float] = []
    confidences: list[float] = []
    for pair, (weighted, conf_sum) in accum.items():
        if conf_sum <= 0.0:
            continue
        comp_edges.append(pair)
        scores.append(weighted / conf_sum)
        confidences.append(conf_sum)
    return (
        np.asarray(comp_edges, dtype=np.int64),
        np.asarray(scores, dtype=np.float32),
        np.asarray(confidences, dtype=np.float32),
    )


def _merge_components(
    vertex_labels: np.ndarray,
    comp_edges: np.ndarray,
    comp_scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    active_labels = np.unique(vertex_labels[vertex_labels > 0])
    if active_labels.size == 0:
        return vertex_labels.astype(np.int32, copy=True), 0
    remap = {int(label): idx for idx, label in enumerate(active_labels)}
    uf = _UnionFind(active_labels.size)
    selected_mask = comp_scores >= float(threshold)
    selected = comp_edges[selected_mask]
    for a, b in selected:
        if int(a) in remap and int(b) in remap:
            uf.union(remap[int(a)], remap[int(b)])
    root_to_new: dict[int, int] = {}
    label_map: dict[int, int] = {}
    next_label = 1
    for old_label in active_labels:
        root = uf.find(remap[int(old_label)])
        if root not in root_to_new:
            root_to_new[root] = next_label
            next_label += 1
        label_map[int(old_label)] = root_to_new[root]
    out = np.zeros_like(vertex_labels, dtype=np.int32)
    for old_label, new_label in label_map.items():
        out[vertex_labels == old_label] = int(new_label)
    return out, int(np.count_nonzero(selected_mask))


def _merge_small_vertex_components(
    vertex_labels: np.ndarray,
    edges: np.ndarray,
    edge_scores: np.ndarray,
    merge_threshold: int,
) -> tuple[np.ndarray, int]:
    if merge_threshold <= 0:
        return vertex_labels.astype(np.int32, copy=True), 0
    out = vertex_labels.astype(np.int32, copy=True)
    counts = np.bincount(out)
    small = [label for label in range(1, counts.size) if 0 < counts[label] < int(merge_threshold)]
    merged = 0
    labels_a = out[edges[:, 0]]
    labels_b = out[edges[:, 1]]
    for label in small:
        edge_mask = ((labels_a == label) & (labels_b != label) & (labels_b > 0)) | (
            (labels_b == label) & (labels_a != label) & (labels_a > 0)
        )
        if not np.any(edge_mask):
            out[out == label] = 0
            merged += 1
            continue
        neighbor_labels = np.where(labels_a[edge_mask] == label, labels_b[edge_mask], labels_a[edge_mask])
        neighbor_scores = edge_scores[edge_mask]
        best_label = 0
        best_score = -1.0
        for candidate in np.unique(neighbor_labels):
            score = float(np.mean(neighbor_scores[neighbor_labels == candidate]))
            if score > best_score:
                best_score = score
                best_label = int(candidate)
        out[out == label] = int(best_label)
        merged += 1
    active = np.unique(out[out > 0])
    relabeled = np.zeros_like(out, dtype=np.int32)
    for new_label, old_label in enumerate(active, start=1):
        relabeled[out == old_label] = int(new_label)
    return relabeled, merged


def _run_graph_region_growing(
    *,
    edges: np.ndarray,
    edge_scores: np.ndarray,
    edge_confidence: np.ndarray,
    point_seen: np.ndarray,
    point_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    thresholds = _parse_threshold_schedule(str(args.graph_thres_connect))
    active = np.any(point_seen, axis=1)
    min_shared = max(int(args.graph_min_shared_views), 1)
    eligible = (edge_confidence >= min_shared) & active[edges[:, 0]] & active[edges[:, 1]]
    history: list[dict[str, Any]] = []
    seed_threshold = float(args.graph_from_points_thres)
    vertex_labels = _labels_from_unions(
        point_seen.shape[0],
        edges,
        active,
        eligible & (edge_scores >= seed_threshold),
    )
    history.append(
        {
            "stage": "from_points",
            "threshold": seed_threshold,
            "componentCount": int(np.count_nonzero(np.unique(vertex_labels) > 0)),
            "activeVertexCount": int(np.count_nonzero(vertex_labels > 0)),
            "edgeCount": int(np.count_nonzero(eligible & (edge_scores >= seed_threshold))),
        }
    )

    for idx, threshold in enumerate(thresholds):
        comp_edges, comp_scores, comp_conf = _component_edge_scores(
            vertex_labels,
            edges,
            edge_scores,
            edge_confidence,
            min_shared,
        )
        before = int(np.count_nonzero(np.unique(vertex_labels) > 0))
        vertex_labels, merged_edges = _merge_components(
            vertex_labels,
            comp_edges,
            comp_scores,
            float(threshold),
        )
        after = int(np.count_nonzero(np.unique(vertex_labels) > 0))
        history.append(
            {
                "stage": "progressive_region_growing",
                "iteration": int(idx),
                "threshold": float(threshold),
                "componentCountBefore": before,
                "componentCountAfter": after,
                "candidateComponentEdges": int(comp_edges.shape[0]),
                "mergedComponentEdges": int(merged_edges),
                "meanCandidateConfidence": float(np.mean(comp_conf)) if comp_conf.size else 0.0,
            }
        )

    vertex_labels, merged_small = _merge_small_vertex_components(
        vertex_labels,
        edges[eligible],
        edge_scores[eligible],
        int(args.graph_thres_merge),
    )
    history.append(
        {
            "stage": "merge_small_segments",
            "mergeThresholdVertices": int(args.graph_thres_merge),
            "mergedOrDroppedComponentCount": int(merged_small),
            "componentCountAfter": int(np.count_nonzero(np.unique(vertex_labels) > 0)),
        }
    )
    return vertex_labels, history


def _write_labeled_mesh(path: Path, mesh: trimesh.Trimesh, vertex_labels: np.ndarray) -> None:
    colors = np.full((len(mesh.vertices), 4), 210, dtype=np.uint8)
    table = _palette(vertex_labels)
    for label, color in table.items():
        if label <= 0:
            continue
        colors[vertex_labels == label, :3] = color
        colors[vertex_labels == label, 3] = 255
    out = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
        process=False,
    )
    out.visual.vertex_colors = colors
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(path)


def _assign_local_mask_majority(
    *,
    local: np.ndarray,
    local_id: int,
    proj: dict[str, Any],
    vertex_labels: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    seen_inside = np.asarray(proj["labels"]) == int(local_id)
    if not np.any(seen_inside):
        return np.zeros(local.shape, dtype=np.uint16), {
            "localLabel": int(local_id),
            "globalLabel": 0,
            "supportVertices": 0,
            "majority": 0.0,
        }

    candidate_vertices = np.asarray(proj["indices"])[seen_inside]
    global_values = vertex_labels[candidate_vertices]
    global_values = global_values[global_values > 0]
    if global_values.size < int(args.consistent_min_vertices):
        return np.zeros(local.shape, dtype=np.uint16), {
            "localLabel": int(local_id),
            "globalLabel": 0,
            "supportVertices": int(global_values.size),
            "majority": 0.0,
        }

    labels, counts = np.unique(global_values, return_counts=True)
    best = int(labels[int(np.argmax(counts))])
    majority = float(np.max(counts) / max(int(np.sum(counts)), 1))
    out = np.zeros(local.shape, dtype=np.uint16)
    if majority < float(args.consistent_min_majority):
        best = 0
    if best > 0:
        out[local == int(local_id)] = np.uint16(min(best, np.iinfo(np.uint16).max))
    return out, {
        "localLabel": int(local_id),
        "globalLabel": int(best),
        "supportVertices": int(global_values.size),
        "majority": majority,
    }


def _write_consistent_view_masks(
    *,
    paths: SAM2ObjectPaths,
    frame_rows: list[dict[str, Any]],
    projection_cache: list[dict[str, Any]],
    vertex_labels: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if paths.consistent_view_mask_dir.exists() and bool(args.overwrite):
        shutil.rmtree(paths.consistent_view_mask_dir)
    paths.consistent_view_mask_masks_dir.mkdir(parents=True, exist_ok=True)
    paths.consistent_view_mask_overlay_dir.mkdir(parents=True, exist_ok=True)
    pred_rows: list[dict[str, Any]] = []

    for row, proj in zip(frame_rows, projection_cache, strict=False):
        index = int(row["sam2objectFrameId"])
        local = np.load(paths.merge_dir / "masks" / f"mask_{index}.npy").astype(np.uint16, copy=False)
        consistent = np.zeros(local.shape, dtype=np.uint16)
        assignments: list[dict[str, Any]] = []
        for local_label in np.unique(local):
            local_id = int(local_label)
            if local_id <= 0:
                continue
            assigned, assignment = _assign_local_mask_majority(
                local=local,
                local_id=local_id,
                proj=proj,
                vertex_labels=vertex_labels,
                args=args,
            )
            fill = assigned > 0
            consistent[fill] = assigned[fill]
            assignments.append(assignment)
        png_path = paths.consistent_view_mask_masks_dir / f"{index:06d}.png"
        npy_path = paths.consistent_view_mask_masks_dir / f"{index:06d}.npy"
        cv2.imwrite(str(png_path), consistent)
        np.save(npy_path, consistent)
        rgb_bgr = cv2.imread(str(paths.baseline_dir / str(row["colorPath"])), cv2.IMREAD_COLOR)
        overlay_path = paths.consistent_view_mask_overlay_dir / f"{index:06d}.png"
        if rgb_bgr is not None:
            cv2.imwrite(str(overlay_path), cv2.cvtColor(_overlay(rgb_bgr, consistent), cv2.COLOR_RGB2BGR))
        pred_rows.append(
            {
                "sam2objectFrameId": int(index),
                "scanID": row.get("scanID", ""),
                "frameID": int(row.get("frameID", index)),
                "imageName": row.get("imageName", ""),
                "maskPng": str(png_path),
                "maskNpy": str(npy_path),
                "overlay": str(overlay_path),
                "labelCount": int(np.count_nonzero(np.unique(consistent) > 0)),
                "coverage": float(np.count_nonzero(consistent > 0) / max(consistent.size, 1)),
                "localAssignments": assignments,
            }
        )

    predictions = paths.consistent_view_mask_dir / "predictions.jsonl"
    _write_jsonl(predictions, pred_rows)
    summary = {
        "stage": "consistent_view_masks",
        "timestampUtc": _now(),
        "frameCount": int(len(pred_rows)),
        "outputs": {
            "consistentViewMaskDir": str(paths.consistent_view_mask_dir),
            "consistentViewMaskMasksDir": str(paths.consistent_view_mask_masks_dir),
            "consistentViewMaskOverlayDir": str(paths.consistent_view_mask_overlay_dir),
            "consistentPredictions": str(predictions),
        },
        "coverage": {
            "min": float(min((row["coverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["coverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["coverage"] for row in pred_rows), default=0.0)),
        },
    }
    _write_json(paths.summary_dir / "consistent_view_masks_summary.json", summary)
    return summary


def consolidate(args: argparse.Namespace, paths: SAM2ObjectPaths) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    frame_rows = _read_jsonl(_require_file(paths.frame_manifest, "SAM2Object frame manifest"))
    _require_dir(paths.merge_dir / "masks", "SAM2Object merged mask directory")
    if paths.graph_dir.exists() and bool(args.overwrite):
        shutil.rmtree(paths.graph_dir)
    paths.graph_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = _resolve_input_mesh(model_dir, args.mesh, int(args.iteration))
    mesh = _load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    edges = _mesh_vertex_edges(faces)
    print(f"[consolidate] mesh={mesh_path}")
    print(f"[consolidate] vertices={vertices.shape[0]} faces={faces.shape[0]} edges={edges.shape[0]}")

    point_labels, point_seen, projection_cache, graph_frame_ids, frame_graph_stats = _project_mesh_to_sam2object_masks(
        paths=paths,
        mesh=mesh,
        frame_rows=frame_rows,
        args=args,
    )
    edge_scores, edge_confidence = _edge_affinity_scores(
        edges,
        point_labels,
        point_seen,
        args,
    )
    vertex_labels, graph_history = _run_graph_region_growing(
        edges=edges,
        edge_scores=edge_scores,
        edge_confidence=edge_confidence,
        point_seen=point_seen,
        point_labels=point_labels,
        args=args,
    )
    graph_npz = paths.graph_dir / "graph_data.npz"
    np.savez_compressed(
        graph_npz,
        edges=edges.astype(np.int32, copy=False),
        edge_scores=edge_scores.astype(np.float32, copy=False),
        edge_confidence=edge_confidence.astype(np.float32, copy=False),
        vertex_labels=vertex_labels.astype(np.int32, copy=False),
        vertex_seen_count=point_seen.sum(axis=1).astype(np.uint16, copy=False),
        vertex_positive_count=(point_labels > 0).sum(axis=1).astype(np.uint16, copy=False),
        graph_frame_ids=np.asarray(graph_frame_ids, dtype=np.int32),
    )
    vertex_label_path = paths.graph_dir / "vertex_labels.npy"
    np.save(vertex_label_path, vertex_labels.astype(np.int32, copy=False))
    labeled_mesh_path = paths.graph_dir / "labeled_mesh.ply"
    _write_labeled_mesh(labeled_mesh_path, mesh, vertex_labels)
    consistent_summary = _write_consistent_view_masks(
        paths=paths,
        frame_rows=frame_rows,
        projection_cache=projection_cache,
        vertex_labels=vertex_labels,
        args=args,
    )
    unique_labels = np.unique(vertex_labels)
    summary = {
        "stage": "consolidate",
        "timestampUtc": _now(),
        "method": "SAM2Object multi-view mask graph adapted to OMeGa mesh vertices",
        "modelDir": str(model_dir),
        "mesh": str(mesh_path),
        "sceneName": paths.scene_name,
        "frameCount": int(len(frame_rows)),
        "graphFrameCount": int(len(graph_frame_ids)),
        "graphFrameIds": [int(value) for value in graph_frame_ids],
        "vertexCount": int(vertices.shape[0]),
        "faceCount": int(faces.shape[0]),
        "edgeCount": int(edges.shape[0]),
        "labelCount": int(np.count_nonzero(unique_labels > 0)),
        "labeledVertexCount": int(np.count_nonzero(vertex_labels > 0)),
        "parameters": {
            "fromPointsThreshold": float(args.graph_from_points_thres),
            "thresConnect": _parse_threshold_schedule(str(args.graph_thres_connect)),
            "thresMerge": int(args.graph_thres_merge),
            "minSharedViews": int(args.graph_min_shared_views),
            "visibilityRtol": float(args.graph_visibility_rtol),
            "visibilityAtol": float(args.graph_visibility_atol),
            "consistentMinVertices": int(args.consistent_min_vertices),
            "consistentMinMajority": float(args.consistent_min_majority),
        },
        "graphHistory": graph_history,
        "graphFrameStats": frame_graph_stats,
        "consistentViewMasks": consistent_summary,
        "outputs": {
            "graphDir": str(paths.graph_dir),
            "graphData": str(graph_npz),
            "vertexLabels": str(vertex_label_path),
            "labeledMesh": str(labeled_mesh_path),
            "consistentViewMaskDir": str(paths.consistent_view_mask_dir),
            "consistentViewMaskMasksDir": str(paths.consistent_view_mask_masks_dir),
            "consistentViewMaskPredictions": str(paths.consistent_view_mask_dir / "predictions.jsonl"),
        },
    }
    _write_json(paths.summary_dir / "consolidate_summary.json", summary)
    return summary


def track(args: argparse.Namespace, paths: SAM2ObjectPaths, sam2object_root: Path) -> dict[str, Any]:
    _require_dir(paths.forward_frame_dir, "SAM2Object prepared forward frame directory")
    checkpoint = _require_file(_resolve_sam2_checkpoint(args, sam2object_root), "SAM2 checkpoint")
    frame_rows = _read_jsonl(_require_file(paths.frame_manifest, "SAM2Object frame manifest"))
    frame_names = _frame_names(paths.forward_frame_dir)
    if len(frame_names) != len(frame_rows):
        raise SystemExit(f"Frame manifest count does not match staged RGB frames: {len(frame_rows)} vs {len(frame_names)}")
    if paths.tracking_dir.exists() and any(paths.tracking_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"SAM2Object tracking output already exists: {paths.tracking_dir}. Pass --overwrite to replace it.")

    segtrack_root = sam2object_root / "segtrack"
    sys.path.insert(0, str(segtrack_root))

    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

    device = str(args.device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

    print(f"[sam2object] loading SAM2 image/video models on {device}")
    with torch.inference_mode(), autocast_context:
        video_predictor = build_sam2_video_predictor(str(args.sam2_config), str(checkpoint), device=device)
        video_predictor_rev = build_sam2_video_predictor(str(args.sam2_config), str(checkpoint), device=device)
        image_model = build_sam2(str(args.sam2_config), str(checkpoint), device=device)
        mask_generator = SAM2AutomaticMaskGenerator(
            model=image_model,
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

        keyframes = _extract_keyframes(paths.forward_frame_dir, frame_names, args)
        reverse_keyframes = sorted({len(frame_names) - 1 - value for value in keyframes})
        print(f"[sam2object] forward keyframes: {keyframes}")
        print(f"[sam2object] reverse keyframes: {reverse_keyframes}")
        forward_summary = _run_direction(
            direction_name="forward",
            video_dir=paths.forward_frame_dir,
            output_dir=paths.forward_dir,
            keyframes=keyframes,
            args=args,
            video_predictor=video_predictor,
            mask_generator=mask_generator,
        )
        reverse_summary = _run_direction(
            direction_name="reverse",
            video_dir=paths.reverse_frame_dir,
            output_dir=paths.reverse_dir,
            keyframes=reverse_keyframes,
            args=args,
            video_predictor=video_predictor_rev,
            mask_generator=mask_generator,
        )

    merge_summary = _merge_forward_reverse(paths, len(frame_names), args)
    normalize_summary = _write_view_masks(paths, frame_rows, args)
    summary = {
        "stage": "track",
        "timestampUtc": _now(),
        "sceneName": paths.scene_name,
        "frameCount": int(len(frame_names)),
        "sam2objectRoot": str(sam2object_root),
        "sam2Checkpoint": str(checkpoint),
        "sam2Config": str(args.sam2_config),
        "forward": forward_summary,
        "reverse": reverse_summary,
        "merge": merge_summary,
        "normalize": normalize_summary,
        "outputs": {
            "trackingDir": str(paths.tracking_dir),
            "forwardMaskDir": str(paths.forward_dir / "masks"),
            "reverseMaskDir": str(paths.reverse_dir / "masks"),
            "mergedMaskDir": str(paths.merge_dir / "masks"),
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskPredictions": str(paths.view_mask_dir / "predictions.jsonl"),
        },
    }
    _write_json(paths.summary_dir / "track_summary.json", summary)
    return summary


def normalize(args: argparse.Namespace, paths: SAM2ObjectPaths) -> dict[str, Any]:
    frame_rows = _read_jsonl(_require_file(paths.frame_manifest, "SAM2Object frame manifest"))
    _require_dir(paths.merge_dir / "masks", "SAM2Object merged mask directory")
    return _write_view_masks(paths, frame_rows, args)


def _merge_stage_summary(paths: SAM2ObjectPaths, stage_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for path in sorted(paths.summary_dir.glob("*_summary.json")):
        try:
            summaries.append(_read_json(path))
        except json.JSONDecodeError:
            continue
    payload = {
        "stage": "sam2object_baseline",
        "timestampUtc": _now(),
        "modelDir": str(args.model_dir.expanduser().resolve()),
        "baselineDir": str(paths.baseline_dir),
        "sceneName": paths.scene_name,
        "activeStageResults": stage_results,
        "summaries": summaries,
        "outputs": {
            "datasetDir": str(paths.dataset_dir),
            "forwardFrameDir": str(paths.forward_frame_dir),
            "reverseFrameDir": str(paths.reverse_frame_dir),
            "posedDir": str(paths.posed_dir),
            "frameManifest": str(paths.frame_manifest),
            "trackingDir": str(paths.tracking_dir),
            "forwardMaskDir": str(paths.forward_dir / "masks"),
            "reverseMaskDir": str(paths.reverse_dir / "masks"),
            "mergedMaskDir": str(paths.merge_dir / "masks"),
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskMasksDir": str(paths.view_mask_masks_dir),
            "viewMaskPredictions": str(paths.view_mask_dir / "predictions.jsonl"),
            "graphDir": str(paths.graph_dir),
            "consistentViewMaskDir": str(paths.consistent_view_mask_dir),
            "consistentViewMaskMasksDir": str(paths.consistent_view_mask_masks_dir),
            "consistentViewMaskPredictions": str(paths.consistent_view_mask_dir / "predictions.jsonl"),
        },
    }
    _write_json(paths.summary, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the SAM2Object view-mask baseline in OMeGa conventions.")
    parser.add_argument("--baseline", default="sam2object", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("check", "prepare", "track", "normalize", "consolidate", "all"), default="prepare")
    parser.add_argument("--model-dir", type=Path, required=True, help="OMeGa model/result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Optional mesh for graph consolidation. Defaults to healed/preclean/latest OMeGa mesh.")
    parser.add_argument("--iteration", type=int, default=-1, help="OMeGa mesh iteration to use when --mesh is omitted.")
    parser.add_argument("--baseline-name", default="sam2object")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--proposal-source-name", default="view_proposals_1024", help="Common 2D proposal source that provides staged RGB/pose frames.")
    parser.add_argument("--proposal-source-dir", type=Path, default=None, help="Explicit common 2D proposal source directory.")
    parser.add_argument("--sam2object-root", type=Path, default=None, help="Defaults to third_party/SAM2Object.")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all selected frames.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--keyframe-mode", choices=("fixed", "sam2object"), default="fixed")
    parser.add_argument("--keyframe-stride", type=int, default=20)
    parser.add_argument("--keyframe-window", type=int, default=20)
    parser.add_argument("--min-keyframe-gap", type=int, default=5)
    parser.add_argument("--max-keyframe-gap", type=int, default=40)
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--stability-score-offset", type=float, default=0.7)
    parser.add_argument("--crop-n-layers", type=int, default=1)
    parser.add_argument("--box-nms-thresh", type=float, default=0.7)
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=2)
    parser.add_argument("--min-mask-region-area", type=float, default=25.0)
    parser.add_argument("--min-mask-area", type=int, default=120)
    parser.add_argument("--max-masks-per-keyframe", type=int, default=80)
    parser.add_argument("--keyframe-iou-threshold", type=float, default=0.7)
    parser.add_argument("--reverse-hole-fill-ratio", type=float, default=0.7)
    parser.add_argument("--offload-video-to-cpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-view-stride", type=int, default=1, help="Use every Nth staged view in the SAM2Object graph affinity pass.")
    parser.add_argument("--graph-from-points-thres", type=float, default=0.9, help="Initial point/vertex primitive merge threshold.")
    parser.add_argument("--graph-thres-connect", default="0.9,0.3,5", help="Progressive graph thresholds, or start,end,count.")
    parser.add_argument("--graph-thres-merge", type=int, default=200, help="Try to merge 3D components with fewer vertices than this. 0 preserves all small components.")
    parser.add_argument("--graph-min-shared-views", type=int, default=2, help="Minimum common visible views for a mesh edge affinity.")
    parser.add_argument("--graph-visibility-rtol", type=float, default=0.15, help="Relative z-buffer visibility tolerance.")
    parser.add_argument("--graph-visibility-atol", type=float, default=0.02, help="Absolute z-buffer visibility tolerance in scene units.")
    parser.add_argument("--graph-depth-chunk-rows", type=int, default=64)
    parser.add_argument("--graph-edge-chunk", type=int, default=200000)
    parser.add_argument("--consistent-min-vertices", type=int, default=6, help="Minimum projected labeled vertices needed to assign a local 2D mask to a global ID.")
    parser.add_argument("--consistent-min-majority", type=float, default=0.55, help="Minimum majority vote ratio for local mask to global ID assignment.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-going-after-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    paths = resolve_paths(
        model_dir,
        baseline_name=args.baseline_name,
        scene_name=args.scene_name,
        output_dir=args.output_dir,
    )
    stage_results: list[dict[str, Any]] = []
    stages = ["prepare", "track", "consolidate"] if args.stage == "all" else [args.stage]
    sam2object_root = _resolve_sam2object_root(args.sam2object_root) if any(stage in {"check", "track"} for stage in stages) else None
    for stage in stages:
        if stage == "check":
            assert sam2object_root is not None
            stage_results.append(check(args, paths, sam2object_root))
        elif stage == "prepare":
            stage_results.append(prepare(args, paths))
        elif stage == "track":
            assert sam2object_root is not None
            stage_results.append(track(args, paths, sam2object_root))
        elif stage == "normalize":
            stage_results.append(normalize(args, paths))
        elif stage == "consolidate":
            stage_results.append(consolidate(args, paths))
        else:  # pragma: no cover
            raise AssertionError(stage)
    summary = _merge_stage_summary(paths, stage_results, args)
    print("=== SAM2Object Baseline ===")
    print(f"Stage: {args.stage}")
    print(f"Baseline dir: {paths.baseline_dir}")
    print(f"Scene: {paths.scene_name}")
    print(f"Summary: {paths.summary}")
    print(f"View masks: {summary['outputs']['viewMaskDir']}")
    print(f"Consistent view masks: {summary['outputs']['consistentViewMaskDir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
