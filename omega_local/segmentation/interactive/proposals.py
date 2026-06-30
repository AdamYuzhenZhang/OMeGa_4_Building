"""Interactive SAM2 proposal generation and storage.

The editor keeps proposal masks separate from SAI3D's automatic labels. These
proposals are draft 2D regions that users can later merge, split, relabel, or
promote into high-confidence 3D observations.
"""

from __future__ import annotations

import json
import shutil
import sys
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import EditorPaths, require_dir, require_file
from .sam2_session import Sam2Config


@dataclass(frozen=True)
class ProposalFrame:
    frame_id: int
    width: int
    height: int
    image_name: str
    image_path: Path


@dataclass(frozen=True)
class ProposalRunPaths:
    run_dir: Path
    label_map_dir: Path
    overlay_dir: Path
    metadata_dir: Path
    summary: Path
    progress: Path
    frame_index: Path
    config: Path


@dataclass(frozen=True)
class ProposalConfig:
    run_name: str
    points_per_side: int = 64
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    stability_score_offset: float = 0.7
    crop_n_layers: int = 1
    box_nms_thresh: float = 0.7
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: float = 25.0
    pred_iou_thresh_keep: float = 0.0
    stability_score_thresh_keep: float = 0.0
    min_mask_area: int = 120
    min_assigned_area: int = 80
    max_masks_per_frame: int = 120

    def to_json(self) -> dict[str, Any]:
        return {
            "runName": self.run_name,
            "pointsPerSide": int(self.points_per_side),
            "predIouThresh": float(self.pred_iou_thresh),
            "stabilityScoreThresh": float(self.stability_score_thresh),
            "stabilityScoreOffset": float(self.stability_score_offset),
            "cropNLayers": int(self.crop_n_layers),
            "boxNmsThresh": float(self.box_nms_thresh),
            "cropNPointsDownscaleFactor": int(self.crop_n_points_downscale_factor),
            "minMaskRegionArea": float(self.min_mask_region_area),
            "predIouThreshKeep": float(self.pred_iou_thresh_keep),
            "stabilityScoreThreshKeep": float(self.stability_score_thresh_keep),
            "minMaskArea": int(self.min_mask_area),
            "minAssignedArea": int(self.min_assigned_area),
            "maxMasksPerFrame": int(self.max_masks_per_frame),
        }


class ProposalManager:
    def __init__(self, paths: EditorPaths, sam2_config: Sam2Config) -> None:
        self.paths = paths
        self.sam2_config = sam2_config
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None

    def default_run_name(self, frames: list[ProposalFrame]) -> str:
        width = int(frames[0].width) if frames else 0
        return f"sam2_auto_{width}" if width > 0 else "sam2_auto"

    def run_paths(self, run_name: str) -> ProposalRunPaths:
        run_dir = self.paths.interactive_dir / "proposals" / _slug_token(run_name)
        return ProposalRunPaths(
            run_dir=run_dir,
            label_map_dir=run_dir / "label_maps",
            overlay_dir=run_dir / "overlays",
            metadata_dir=run_dir / "metadata",
            summary=run_dir / "summary.json",
            progress=run_dir / "progress.json",
            frame_index=run_dir / "frames.jsonl",
            config=run_dir / "config.json",
        )

    def editable_run_name(self, frames: list[ProposalFrame], run_name: str | None = None) -> str:
        base = _slug_token(run_name or self.default_run_name(frames))
        return f"{base}_edited"

    def editable_paths(self, frames: list[ProposalFrame], run_name: str | None = None) -> ProposalRunPaths:
        return self.run_paths(self.editable_run_name(frames, run_name))

    def ensure_editable_copy(self, frames: list[ProposalFrame], run_name: str | None = None) -> ProposalRunPaths:
        raw_name = _slug_token(run_name or self.default_run_name(frames))
        raw_paths = self.run_paths(raw_name)
        if not raw_paths.summary.exists():
            raise FileNotFoundError(f"Raw SAM2 proposal run does not exist: {raw_paths.summary}")

        edit_paths = self.editable_paths(frames, raw_name)
        if edit_paths.summary.exists():
            _repair_editable_metadata(edit_paths, frames)
            return edit_paths

        for directory in [edit_paths.label_map_dir, edit_paths.overlay_dir, edit_paths.metadata_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        _copy_existing(raw_paths.config, edit_paths.config)
        _copy_existing(raw_paths.progress, edit_paths.progress)
        _copy_existing(raw_paths.frame_index, edit_paths.frame_index)
        _copy_tree_files(raw_paths.label_map_dir, edit_paths.label_map_dir)
        _copy_tree_files(raw_paths.overlay_dir, edit_paths.overlay_dir)
        _copy_tree_files(raw_paths.metadata_dir, edit_paths.metadata_dir)

        raw_summary = json.loads(raw_paths.summary.read_text(encoding="utf-8"))
        raw_summary["stage"] = "interactive_sam2_edited_proposals"
        raw_summary["method"] = "Editable copy of SAM2 per-frame proposal label maps."
        raw_summary["rawRunName"] = raw_name
        raw_summary["rawRunDir"] = str(raw_paths.run_dir)
        raw_summary["runName"] = edit_paths.run_dir.name
        raw_summary["runDir"] = str(edit_paths.run_dir)
        raw_summary["timestampUtc"] = _now()
        raw_summary["outputs"] = {
            "labelMapDir": str(edit_paths.label_map_dir),
            "overlayDir": str(edit_paths.overlay_dir),
            "metadataDir": str(edit_paths.metadata_dir),
            "frameIndex": str(edit_paths.frame_index),
        }
        _write_json(edit_paths.summary, raw_summary)
        _write_json(edit_paths.run_dir / "source_raw_run.json", {"rawRunName": raw_name, "rawRunDir": str(raw_paths.run_dir)})
        _repair_editable_metadata(edit_paths, frames)
        return edit_paths

    def status(self, frames: list[ProposalFrame], run_name: str | None = None) -> dict[str, Any]:
        run_name = _slug_token(run_name or self.default_run_name(frames))
        with self._lock:
            if self._job is not None and self._job.get("runName") == run_name:
                return dict(self._job)

        paths = self.run_paths(run_name)
        if paths.progress.exists():
            try:
                payload = json.loads(paths.progress.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        elif paths.summary.exists():
            payload = {
                "runName": run_name,
                "runDir": str(paths.run_dir),
                "ready": True,
                "running": False,
                "failed": False,
                "frameCount": len(frames),
                "completedFrameCount": _count_existing_overlays(paths, frames),
                "message": "SAM2 proposals ready.",
            }
        else:
            payload = {
                "runName": run_name,
                "runDir": str(paths.run_dir),
                "ready": False,
                "running": False,
                "failed": False,
                "frameCount": len(frames),
                "completedFrameCount": 0,
                "message": "No SAM2 proposal run found.",
            }
        payload.setdefault("runName", run_name)
        payload.setdefault("runDir", str(paths.run_dir))
        payload.setdefault("ready", paths.summary.exists())
        payload.setdefault("running", False)
        payload.setdefault("failed", False)
        payload.setdefault("frameCount", len(frames))
        payload.setdefault("completedFrameCount", _count_existing_overlays(paths, frames))
        payload.setdefault("overlayUrlTemplate", "/api/proposals/sam2/frame/{frameId}/overlay")
        payload.setdefault("summaryPath", str(paths.summary))
        return payload

    def start(self, frames: list[ProposalFrame], payload: dict[str, Any]) -> dict[str, Any]:
        run_name = _slug_token(str(payload.get("runName") or self.default_run_name(frames)))
        overwrite = bool(payload.get("overwrite", False))
        config = ProposalConfig(
            run_name=run_name,
            points_per_side=int(payload.get("pointsPerSide", 64)),
            pred_iou_thresh=float(payload.get("predIouThresh", 0.7)),
            stability_score_thresh=float(payload.get("stabilityScoreThresh", 0.92)),
            stability_score_offset=float(payload.get("stabilityScoreOffset", 0.7)),
            crop_n_layers=int(payload.get("cropNLayers", 1)),
            box_nms_thresh=float(payload.get("boxNmsThresh", 0.7)),
            crop_n_points_downscale_factor=int(payload.get("cropNPointsDownscaleFactor", 2)),
            min_mask_region_area=float(payload.get("minMaskRegionArea", 25.0)),
            pred_iou_thresh_keep=float(payload.get("predIouThreshKeep", 0.0)),
            stability_score_thresh_keep=float(payload.get("stabilityScoreThreshKeep", 0.0)),
            min_mask_area=int(payload.get("minMaskArea", 120)),
            min_assigned_area=int(payload.get("minAssignedArea", 80)),
            max_masks_per_frame=int(payload.get("maxMasksPerFrame", 120)),
        )
        paths = self.run_paths(run_name)

        with self._lock:
            if self._job is not None and self._job.get("running"):
                return dict(self._job)
            if paths.summary.exists() and not overwrite:
                return self.status(frames, run_name)

            job = {
                "runName": run_name,
                "runDir": str(paths.run_dir),
                "ready": False,
                "running": True,
                "failed": False,
                "frameCount": len(frames),
                "completedFrameCount": 0,
                "currentFrameId": None,
                "message": "Starting SAM2 automatic proposals.",
                "overlayUrlTemplate": "/api/proposals/sam2/frame/{frameId}/overlay",
                "summaryPath": str(paths.summary),
                "updatedUtc": _now(),
            }
            self._job = job

        thread = threading.Thread(
            target=self._run_job,
            args=(frames, config, paths, overwrite),
            name=f"sam2-proposals-{run_name}",
            daemon=True,
        )
        thread.start()
        return dict(job)

    def overlay_path(self, frames: list[ProposalFrame], frame_id: int, run_name: str | None = None) -> Path:
        paths = self.ensure_editable_copy(frames, run_name)
        return paths.overlay_dir / f"{int(frame_id):06d}.png"

    def label_map_path(self, frames: list[ProposalFrame], frame_id: int, run_name: str | None = None) -> Path:
        paths = self.ensure_editable_copy(frames, run_name)
        return paths.label_map_dir / f"{int(frame_id):06d}.npy"

    def metadata_path(self, frames: list[ProposalFrame], frame_id: int, run_name: str | None = None) -> Path:
        paths = self.ensure_editable_copy(frames, run_name)
        return paths.metadata_dir / f"{int(frame_id):06d}.json"

    def save_label_map(
        self,
        frames: list[ProposalFrame],
        frame_id: int,
        labels: np.ndarray,
        *,
        edit_records: list[dict[str, Any]] | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        edit_paths = self.ensure_editable_copy(frames, run_name)
        frame = next((item for item in frames if int(item.frame_id) == int(frame_id)), None)
        if frame is None:
            raise KeyError(frame_id)
        labels = np.asarray(labels)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D proposal label map, got shape {labels.shape}.")

        npy_path = edit_paths.label_map_dir / f"{int(frame_id):06d}.npy"
        png_path = edit_paths.label_map_dir / f"{int(frame_id):06d}.png"
        overlay_path = edit_paths.overlay_dir / f"{int(frame_id):06d}.png"
        metadata_path = edit_paths.metadata_dir / f"{int(frame_id):06d}.json"

        np.save(npy_path, labels.astype(np.uint16, copy=False))
        Image.fromarray(labels.astype(np.uint16, copy=False), mode="I;16").save(png_path)
        Image.fromarray(_transparent_label_overlay(labels), mode="RGBA").save(overlay_path)

        label_rows = _label_rows_from_label_map(labels)
        payload = {
            "frameId": int(frame_id),
            "imageName": frame.image_name,
            "width": int(frame.width),
            "height": int(frame.height),
            "rawMaskCount": int(len(label_rows)),
            "keptMaskCount": int(len(label_rows)),
            "coverage": float(np.count_nonzero(labels > 0) / max(labels.size, 1)),
            "labelMapNpy": str(npy_path),
            "labelMapPng": str(png_path),
            "overlayPng": str(overlay_path),
            "metadata": str(metadata_path),
            "labels": label_rows,
            "updatedUtc": _now(),
            "editRecords": edit_records or [],
        }
        _write_json(metadata_path, payload)
        _write_jsonl(edit_paths.frame_index, _metadata_summaries(edit_paths.metadata_dir))
        _refresh_summary(edit_paths, frames)
        return payload

    def _update_job(self, paths: ProposalRunPaths, **updates: Any) -> None:
        with self._lock:
            if self._job is None:
                return
            self._job.update(updates)
            self._job["updatedUtc"] = _now()
            payload = dict(self._job)
        _write_json(paths.progress, payload)

    def _run_job(
        self,
        frames: list[ProposalFrame],
        config: ProposalConfig,
        paths: ProposalRunPaths,
        overwrite: bool,
    ) -> None:
        try:
            if overwrite and paths.run_dir.exists():
                shutil.rmtree(paths.run_dir)
            for directory in [paths.label_map_dir, paths.overlay_dir, paths.metadata_dir]:
                directory.mkdir(parents=True, exist_ok=True)
            _write_json(paths.config, config.to_json())
            _write_json(paths.run_dir / "source_summary.json", self._source_summary(frames))

            import torch

            generator, device, autocast_context = self._load_generator(config)
            frame_rows: list[dict[str, Any]] = []
            frame_summaries: list[dict[str, Any]] = []

            with torch.inference_mode(), autocast_context:
                for index, frame in enumerate(frames):
                    self._update_job(
                        paths,
                        currentFrameId=int(frame.frame_id),
                        completedFrameCount=int(index),
                        message=f"Running SAM2 on frame {frame.frame_id} ({index + 1}/{len(frames)}).",
                    )
                    image_rgb = np.asarray(Image.open(frame.image_path).convert("RGB"))
                    generated = generator.generate(image_rgb)
                    records = _mask_records(generated, config)
                    label_map, label_rows = _records_to_label_map(records, image_rgb.shape[:2], int(config.min_assigned_area))

                    npy_path = paths.label_map_dir / f"{frame.frame_id:06d}.npy"
                    png_path = paths.label_map_dir / f"{frame.frame_id:06d}.png"
                    overlay_path = paths.overlay_dir / f"{frame.frame_id:06d}.png"
                    metadata_path = paths.metadata_dir / f"{frame.frame_id:06d}.json"
                    np.save(npy_path, label_map)
                    Image.fromarray(label_map.astype(np.uint16), mode="I;16").save(png_path)
                    Image.fromarray(_transparent_label_overlay(label_map), mode="RGBA").save(overlay_path)

                    coverage = float(np.count_nonzero(label_map > 0) / max(label_map.size, 1))
                    frame_payload = {
                        "frameId": int(frame.frame_id),
                        "imageName": frame.image_name,
                        "width": int(frame.width),
                        "height": int(frame.height),
                        "rawMaskCount": int(len(generated)),
                        "keptMaskCount": int(len(label_rows)),
                        "coverage": coverage,
                        "labelMapNpy": str(npy_path),
                        "labelMapPng": str(png_path),
                        "overlayPng": str(overlay_path),
                        "metadata": str(metadata_path),
                        "labels": label_rows,
                    }
                    _write_json(metadata_path, frame_payload)
                    frame_rows.append(frame_payload)
                    frame_summaries.append(
                        {
                            "frameId": int(frame.frame_id),
                            "rawMaskCount": int(len(generated)),
                            "keptMaskCount": int(len(label_rows)),
                            "coverage": coverage,
                        }
                    )
                    _write_jsonl(paths.frame_index, frame_rows)
                    self._update_job(
                        paths,
                        completedFrameCount=int(index + 1),
                        currentFrameId=int(frame.frame_id),
                        message=(
                            f"Frame {frame.frame_id}: {len(label_rows)} proposals, "
                            f"{coverage:.1%} coverage."
                        ),
                    )

            summary = {
                "stage": "interactive_sam2_auto_proposals",
                "timestampUtc": _now(),
                "method": "SAM2 automatic per-view proposal label maps for interactive proposal editing.",
                "runName": config.run_name,
                "runDir": str(paths.run_dir),
                "source": self._source_summary(frames),
                "sam2": {
                    "root": str(self.sam2_config.root),
                    "checkpoint": str(self.sam2_config.checkpoint),
                    "config": str(self.sam2_config.config),
                    "device": str(device),
                },
                "parameters": config.to_json(),
                "frameCount": int(len(frames)),
                "frames": frame_summaries,
                "outputs": {
                    "labelMapDir": str(paths.label_map_dir),
                    "overlayDir": str(paths.overlay_dir),
                    "metadataDir": str(paths.metadata_dir),
                    "frameIndex": str(paths.frame_index),
                },
            }
            _write_json(paths.summary, summary)
            self._update_job(
                paths,
                ready=True,
                running=False,
                failed=False,
                completedFrameCount=int(len(frames)),
                currentFrameId=None,
                message="SAM2 proposals ready.",
            )
        except Exception as exc:  # pragma: no cover - surfaced through UI status
            self._update_job(
                paths,
                ready=False,
                running=False,
                failed=True,
                message=f"SAM2 proposal generation failed: {exc}",
            )

    def _source_summary(self, frames: list[ProposalFrame]) -> dict[str, Any]:
        return {
            "baselineName": self.paths.baseline_name,
            "baselineDir": str(self.paths.baseline_dir),
            "datasetDir": str(self.paths.dataset_dir),
            "frameManifest": str(self.paths.frame_manifest),
            "pointsPath": str(self.paths.points_path),
            "pointCloudRole": "Initial unsegmented point cloud sampled from the OMeGa mesh and used as SAI3D input.",
            "frameCount": int(len(frames)),
        }

    def _load_generator(self, config: ProposalConfig) -> tuple[Any, Any, Any]:
        sam2_root = require_dir(self.sam2_config.root, "SAM2 root")
        require_file(sam2_root / "sam2" / "build_sam.py", "SAM2 build_sam.py")
        checkpoint = require_file(self.sam2_config.checkpoint, "SAM2 checkpoint")
        if str(sam2_root) not in sys.path:
            sys.path.insert(0, str(sam2_root))

        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2

        if self.sam2_config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.sam2_config.device)
        use_cuda = device.type == "cuda"
        if use_cuda and torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else nullcontext()

        model = build_sam2(str(self.sam2_config.config), str(checkpoint), device=device)
        generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=int(config.points_per_side),
            pred_iou_thresh=float(config.pred_iou_thresh),
            stability_score_thresh=float(config.stability_score_thresh),
            stability_score_offset=float(config.stability_score_offset),
            crop_n_layers=int(config.crop_n_layers),
            box_nms_thresh=float(config.box_nms_thresh),
            crop_n_points_downscale_factor=int(config.crop_n_points_downscale_factor),
            min_mask_region_area=float(config.min_mask_region_area),
            use_m2m=True,
            multimask_output=False,
        )
        return generator, device, autocast_context


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
    return "".join(out).strip("_") or "sam2_auto"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _copy_existing(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if src.is_file():
            shutil.copy2(src, dst_dir / src.name)


def _label_rows_from_label_map(labels: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        ys, xs = np.nonzero(labels == label_i)
        if xs.size == 0:
            continue
        rows.append(
            {
                "labelId": label_i,
                "rawAreaPixels": int(xs.size),
                "assignedPixels": int(xs.size),
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "predictedIou": 0.0,
                "stabilityScore": 0.0,
                "score": 0.0,
            }
        )
    rows.sort(key=lambda row: int(row["labelId"]))
    return rows


def _metadata_summary(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return {
        "frameId": int(payload.get("frameId", 0)),
        "rawMaskCount": int(payload.get("rawMaskCount", payload.get("keptMaskCount", 0))),
        "keptMaskCount": int(payload.get("keptMaskCount", 0)),
        "coverage": float(payload.get("coverage", 0.0)),
    }


def _metadata_summaries(metadata_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(metadata_dir.glob("*.json")):
        row = _metadata_summary(path)
        if row is not None:
            rows.append(row)
    return rows


def _refresh_summary(paths: ProposalRunPaths, frames: list[ProposalFrame]) -> None:
    if paths.summary.exists():
        try:
            summary = json.loads(paths.summary.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    else:
        summary = {}
    frame_summaries = _metadata_summaries(paths.metadata_dir)
    summary.update(
        {
            "stage": "interactive_sam2_edited_proposals",
            "timestampUtc": _now(),
            "runName": paths.run_dir.name,
            "runDir": str(paths.run_dir),
            "frameCount": int(len(frames)),
            "frames": frame_summaries,
            "outputs": {
                "labelMapDir": str(paths.label_map_dir),
                "overlayDir": str(paths.overlay_dir),
                "metadataDir": str(paths.metadata_dir),
                "frameIndex": str(paths.frame_index),
            },
        }
    )
    _write_json(paths.summary, summary)


def _repair_editable_metadata(paths: ProposalRunPaths, frames: list[ProposalFrame]) -> None:
    frame_by_id = {int(frame.frame_id): frame for frame in frames}
    changed = False
    for metadata_path in sorted(paths.metadata_dir.glob("*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        frame_id = int(payload.get("frameId", metadata_path.stem))
        frame = frame_by_id.get(frame_id)
        if frame is not None:
            payload["imageName"] = frame.image_name
            payload["width"] = int(frame.width)
            payload["height"] = int(frame.height)
        payload["labelMapNpy"] = str(paths.label_map_dir / f"{frame_id:06d}.npy")
        payload["labelMapPng"] = str(paths.label_map_dir / f"{frame_id:06d}.png")
        payload["overlayPng"] = str(paths.overlay_dir / f"{frame_id:06d}.png")
        payload["metadata"] = str(metadata_path)
        _write_json(metadata_path, payload)
        changed = True
    if changed:
        _write_jsonl(paths.frame_index, _metadata_summaries(paths.metadata_dir))
        _refresh_summary(paths, frames)


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


def _transparent_label_overlay(labels: np.ndarray) -> np.ndarray:
    overlay = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        mask = labels == label_i
        overlay[mask, :3] = _label_color(label_i)
        overlay[mask, 3] = 138
    return overlay


def _mask_records(masks: list[dict[str, Any]], config: ProposalConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in masks:
        mask = np.asarray(item.get("segmentation"), dtype=bool)
        area = int(np.count_nonzero(mask))
        if area < int(config.min_mask_area):
            continue
        predicted_iou = float(item.get("predicted_iou", 0.0))
        stability = float(item.get("stability_score", 0.0))
        if predicted_iou < float(config.pred_iou_thresh_keep) or stability < float(config.stability_score_thresh_keep):
            continue
        bbox = item.get("bbox", [0, 0, 0, 0])
        score = predicted_iou * 0.7 + stability * 0.3
        records.append(
            {
                "mask": mask,
                "area": area,
                "bbox": [float(value) for value in bbox],
                "predicted_iou": predicted_iou,
                "stability": stability,
                "score": score,
            }
        )
    records.sort(key=lambda item: (float(item["score"]), int(item["area"])), reverse=True)
    if int(config.max_masks_per_frame) > 0:
        records = records[: int(config.max_masks_per_frame)]
    return records


def _records_to_label_map(
    records: list[dict[str, Any]],
    shape: tuple[int, int],
    min_remaining_area: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
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
                "bbox": list(record["bbox"]),
                "predictedIou": float(record["predicted_iou"]),
                "stabilityScore": float(record["stability"]),
                "score": float(record["score"]),
            }
        )
    return labels, rows


def _count_existing_overlays(paths: ProposalRunPaths, frames: list[ProposalFrame]) -> int:
    return sum(1 for frame in frames if (paths.overlay_dir / f"{int(frame.frame_id):06d}.png").exists())
