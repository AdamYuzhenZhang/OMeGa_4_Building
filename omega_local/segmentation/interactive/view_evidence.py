"""Per-frame normal/depth evidence for interactive proposal editing."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import EditorPaths, THIRD_PARTY_ROOT
from .view_evidence_images import (
    decode_normal_rgb,
    depth_edge_rgb,
    depth_rgb,
    normal_edge_rgb,
    resize_depth,
    resize_rgb,
)
from .view_evidence_models import DepthAnythingPredictor, StableNormalPredictor
from .view_evidence_store import (
    _adopt_legacy_evidence_run,
    _clear_target_outputs,
    _count_existing,
    _count_generated_kind,
    _evidence_kind_status,
    _frame_has_generated_kind,
    _now,
    _read_frame_metadata,
    _slug_token,
    _target_label,
    _target_names,
    _targets_ready,
    _write_evidence_kind_status,
    _write_json,
    _write_jsonl,
)


@dataclass(frozen=True)
class ViewEvidenceFrame:
    frame_id: int
    source_frame_id: int
    scan_id: str
    width: int
    height: int
    image_name: str
    image_path: Path
    manifest_row: dict[str, Any]


@dataclass(frozen=True)
class ViewEvidenceRunPaths:
    run_dir: Path
    normal_dir: Path
    normal_npz_dir: Path
    normal_edge_dir: Path
    depth_dir: Path
    depth_npz_dir: Path
    depth_edge_dir: Path
    metadata_dir: Path
    summary: Path
    progress: Path
    frame_index: Path
    config: Path


@dataclass(frozen=True)
class ViewEvidenceConfig:
    run_name: str
    targets: tuple[str, ...] = ("normal", "depth")
    normal_source: str = "stable_normal"
    depth_source: str = "depth_anything_v2"
    stable_normal_root: Path = THIRD_PARTY_ROOT / "StableNormal"
    stable_normal_variant: str = "turbo"
    stable_normal_processing_resolution: int = 1024
    depth_anything_model: str = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    device: str = "auto"
    overwrite: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "runName": self.run_name,
            "targets": list(self.targets),
            "normalSource": self.normal_source,
            "depthSource": self.depth_source,
            "stableNormalRoot": str(self.stable_normal_root),
            "stableNormalVariant": self.stable_normal_variant,
            "stableNormalProcessingResolution": int(self.stable_normal_processing_resolution),
            "depthAnythingModel": self.depth_anything_model,
            "device": self.device,
            "overwrite": bool(self.overwrite),
        }


class ViewEvidenceManager:
    def __init__(self, paths: EditorPaths) -> None:
        self.paths = paths
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None

    def default_run_name(self, frames: list[ViewEvidenceFrame]) -> str:
        return "view_evidence"

    def run_paths(self, run_name: str) -> ViewEvidenceRunPaths:
        # The editor owns one local evidence cache. Width/model choices are
        # recorded in config.json instead of encoded into nested run folders.
        run_dir = self.paths.interactive_dir / "view_evidence"
        return ViewEvidenceRunPaths(
            run_dir=run_dir,
            normal_dir=run_dir / "normal_rgb",
            normal_npz_dir=run_dir / "normal_npz",
            normal_edge_dir=run_dir / "normal_edges",
            depth_dir=run_dir / "depth_rgb",
            depth_npz_dir=run_dir / "depth_npz",
            depth_edge_dir=run_dir / "depth_edges",
            metadata_dir=run_dir / "metadata",
            summary=run_dir / "summary.json",
            progress=run_dir / "progress.json",
            frame_index=run_dir / "frames.jsonl",
            config=run_dir / "config.json",
        )

    def status(self, frames: list[ViewEvidenceFrame], run_name: str | None = None) -> dict[str, Any]:
        run_name = _slug_token(run_name or self.default_run_name(frames))
        active_job: dict[str, Any] | None = None
        with self._lock:
            if self._job is not None and self._job.get("runName") == run_name:
                active_job = dict(self._job)

        paths = self.run_paths(run_name)
        _adopt_legacy_evidence_run(paths, run_name)
        normal_count = _count_generated_kind(paths, frames, "normal")
        depth_count = _count_generated_kind(paths, frames, "depth")
        if active_job is not None:
            payload = active_job
        elif paths.progress.exists():
            try:
                payload = json.loads(paths.progress.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        elif paths.summary.exists():
            payload = {
                "runName": run_name,
                "runDir": str(paths.run_dir),
                "ready": bool(normal_count or depth_count),
                "running": False,
                "failed": False,
                "frameCount": len(frames),
                "completedFrameCount": _count_existing(paths.metadata_dir, frames),
                "message": "View evidence ready.",
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
                "message": "No view evidence run found.",
            }
        payload.setdefault("runName", run_name)
        payload.setdefault("runDir", str(paths.run_dir))
        normal_status = _evidence_kind_status(paths, frames, "normal", active_job)
        depth_status = _evidence_kind_status(paths, frames, "depth", active_job)
        payload["ready"] = bool(payload.get("running")) or bool(normal_status["ready"] or depth_status["ready"])
        payload.setdefault("running", False)
        payload["failed"] = bool(payload.get("running") and payload.get("failed"))
        payload.setdefault("frameCount", len(frames))
        payload.setdefault("completedFrameCount", _count_existing(paths.metadata_dir, frames))
        payload["normalFrameCount"] = normal_count
        payload["depthFrameCount"] = depth_count
        payload["normalReady"] = bool(normal_status["ready"])
        payload["depthReady"] = bool(depth_status["ready"])
        payload["normalStatus"] = normal_status
        payload["depthStatus"] = depth_status
        if not payload.get("running"):
            if normal_status["failed"] or depth_status["failed"]:
                payload["message"] = "One or more evidence jobs failed; each method can be retried independently."
            elif normal_count or depth_count:
                payload["message"] = f"Generated evidence ready: {normal_count} normal frames, {depth_count} depth frames."
            else:
                payload["message"] = "No generated StableNormal or Depth Anything V2 evidence found."
        payload.setdefault("summaryPath", str(paths.summary))
        payload.setdefault("normalUrlTemplate", "/api/view-evidence/frame/{frameId}/normal")
        payload.setdefault("depthUrlTemplate", "/api/view-evidence/frame/{frameId}/depth")
        return payload

    def start(self, frames: list[ViewEvidenceFrame], payload: dict[str, Any]) -> dict[str, Any]:
        run_name = self.default_run_name(frames)
        targets = _target_names(payload)
        config = ViewEvidenceConfig(
            run_name=run_name,
            targets=targets,
            normal_source="stable_normal",
            depth_source="depth_anything_v2",
            stable_normal_root=Path(str(payload.get("stableNormalRoot") or THIRD_PARTY_ROOT / "StableNormal")),
            stable_normal_variant=str(payload.get("stableNormalVariant", "turbo")),
            stable_normal_processing_resolution=int(payload.get("stableNormalProcessingResolution", 1024)),
            depth_anything_model=str(
                payload.get("depthAnythingModel", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf")
            ),
            device=str(payload.get("device", "auto")),
            overwrite=bool(payload.get("overwrite", False)),
        )
        paths = self.run_paths(run_name)
        _adopt_legacy_evidence_run(paths, run_name)

        with self._lock:
            if self._job is not None and self._job.get("running"):
                return dict(self._job)
        if not config.overwrite and _targets_ready(paths, frames, targets):
            return self.status(frames, run_name)

        with self._lock:
            if self._job is not None and self._job.get("running"):
                return dict(self._job)
            for directory in [
                paths.normal_dir,
                paths.normal_npz_dir,
                paths.normal_edge_dir,
                paths.depth_dir,
                paths.depth_npz_dir,
                paths.depth_edge_dir,
                paths.metadata_dir,
            ]:
                directory.mkdir(parents=True, exist_ok=True)
            if config.overwrite:
                _clear_target_outputs(paths, targets)
                for target in targets:
                    _write_evidence_kind_status(
                        paths,
                        target,
                        {
                            "target": target,
                            "ready": False,
                            "running": False,
                            "failed": False,
                            "frameCount": len(frames),
                            "completedFrameCount": 0,
                            "message": f"Cleared {_target_label((target,))}.",
                            "updatedUtc": _now(),
                        },
                    )
            self._job = {
                "runName": run_name,
                "runDir": str(paths.run_dir),
                "targets": list(targets),
                "ready": False,
                "running": True,
                "failed": False,
                "frameCount": len(frames),
                "completedFrameCount": 0,
                "currentFrameId": None,
                "message": f"Starting {_target_label(targets)} generation.",
                "updatedUtc": _now(),
            }
            _write_json(paths.progress, self._job)
            for target in targets:
                _write_evidence_kind_status(
                    paths,
                    target,
                    {
                        "target": target,
                        "ready": False,
                        "running": True,
                        "failed": False,
                        "frameCount": len(frames),
                        "completedFrameCount": 0,
                        "currentFrameId": None,
                        "message": f"Starting {_target_label((target,))}.",
                        "updatedUtc": _now(),
                    },
                )

        thread = threading.Thread(target=self._run_job, args=(frames, config, paths), daemon=True)
        thread.start()
        return self.status(frames, run_name)

    def image_path(self, frames: list[ViewEvidenceFrame], frame_id: int, kind: str, run_name: str | None = None) -> Path:
        paths = self.run_paths(_slug_token(run_name or self.default_run_name(frames)))
        _adopt_legacy_evidence_run(paths, _slug_token(run_name or self.default_run_name(frames)))
        kind = str(kind).strip().lower().replace("_", "-")
        if kind in {"normal", "normal-rgb"}:
            path = paths.normal_dir / f"{int(frame_id):06d}.png"
        elif kind in {"normal-edge", "normal-edges"}:
            path = paths.normal_edge_dir / f"{int(frame_id):06d}.png"
        elif kind in {"depth", "depth-rgb"}:
            path = paths.depth_dir / f"{int(frame_id):06d}.png"
        elif kind in {"depth-edge", "depth-edges"}:
            path = paths.depth_edge_dir / f"{int(frame_id):06d}.png"
        else:
            raise KeyError(kind)
        if not path.exists():
            raise FileNotFoundError(f"View evidence image does not exist: {path}")
        source_kind = "normal" if kind.startswith("normal") else "depth"
        if not _frame_has_generated_kind(paths, int(frame_id), source_kind):
            raise FileNotFoundError(f"Generated {source_kind} evidence does not exist for frame {frame_id}: {path}")
        return path

    def _run_job(self, frames: list[ViewEvidenceFrame], config: ViewEvidenceConfig, paths: ViewEvidenceRunPaths) -> None:
        try:
            _write_json(paths.config, config.to_json())
            _write_json(paths.run_dir / "source_summary.json", self._source_summary(frames, config))

            stable_normal = None
            depth_anything = None
            frame_rows: list[dict[str, Any]] = []
            generate_normal = "normal" in config.targets
            generate_depth = "depth" in config.targets

            for index, frame in enumerate(frames):
                self._update_job(
                    paths,
                    currentFrameId=int(frame.frame_id),
                    completedFrameCount=int(index),
                    message=f"Preparing {_target_label(config.targets)} for frame {frame.frame_id} ({index + 1}/{len(frames)}).",
                )
                image = Image.open(frame.image_path).convert("RGB")
                if image.size != (frame.width, frame.height):
                    image = image.resize((frame.width, frame.height), Image.Resampling.LANCZOS)

                metadata_path = paths.metadata_dir / f"{frame.frame_id:06d}.json"
                frame_payload = _read_frame_metadata(metadata_path, frame)

                normal_rgb: np.ndarray | None = None
                if generate_normal:
                    stable_normal = stable_normal or StableNormalPredictor(config)
                    normal_rgb, normal_meta = stable_normal.predict(image)
                    if normal_rgb is not None:
                        normal_rgb = resize_rgb(normal_rgb, (frame.width, frame.height))
                        normal = decode_normal_rgb(normal_rgb)
                        Image.fromarray(normal_rgb, mode="RGB").save(paths.normal_dir / f"{frame.frame_id:06d}.png")
                        np.savez_compressed(
                            paths.normal_npz_dir / f"{frame.frame_id:06d}.npz",
                            normal=normal.astype(np.float16, copy=False),
                        )
                        Image.fromarray(normal_edge_rgb(normal), mode="RGB").save(
                            paths.normal_edge_dir / f"{frame.frame_id:06d}.png"
                        )
                        frame_payload["normalReady"] = True
                        frame_payload["normalSource"] = normal_meta
                        frame_payload.setdefault("outputs", {})
                        frame_payload["outputs"].update(
                            {
                                "normal": str(paths.normal_dir / f"{frame.frame_id:06d}.png"),
                                "normalNpz": str(paths.normal_npz_dir / f"{frame.frame_id:06d}.npz"),
                                "normalEdges": str(paths.normal_edge_dir / f"{frame.frame_id:06d}.png"),
                            }
                        )

                depth_m: np.ndarray | None = None
                if generate_depth:
                    depth_anything = depth_anything or DepthAnythingPredictor(config)
                    depth_m, depth_meta = depth_anything.predict(image)
                    if depth_m is not None:
                        depth_m = resize_depth(depth_m, (frame.width, frame.height))
                        valid = np.isfinite(depth_m) & (depth_m > 0)
                        depth_view_rgb = depth_rgb(depth_m, valid)
                        depth_edge_view_rgb = depth_edge_rgb(depth_m, valid)
                        Image.fromarray(depth_view_rgb, mode="RGB").save(paths.depth_dir / f"{frame.frame_id:06d}.png")
                        Image.fromarray(depth_edge_view_rgb, mode="RGB").save(paths.depth_edge_dir / f"{frame.frame_id:06d}.png")
                        np.savez_compressed(
                            paths.depth_npz_dir / f"{frame.frame_id:06d}.npz",
                            depth_m=depth_m.astype(np.float16, copy=False),
                            valid=valid,
                        )
                        frame_payload["depthReady"] = True
                        frame_payload["depthSource"] = depth_meta
                        frame_payload.setdefault("outputs", {})
                        frame_payload["outputs"].update(
                            {
                                "depth": str(paths.depth_dir / f"{frame.frame_id:06d}.png"),
                                "depthNpz": str(paths.depth_npz_dir / f"{frame.frame_id:06d}.npz"),
                                "depthEdges": str(paths.depth_edge_dir / f"{frame.frame_id:06d}.png"),
                            }
                        )

                frame_payload["updatedUtc"] = _now()
                _write_json(metadata_path, frame_payload)
                frame_rows.append(frame_payload)
                self._update_job(
                    paths,
                    currentFrameId=int(frame.frame_id),
                    completedFrameCount=int(index + 1),
                    message=(
                        f"Frame {frame.frame_id}: "
                        f"normal={'generated' if normal_rgb is not None else 'kept'}, "
                        f"depth={'generated' if depth_m is not None else 'kept'}."
                    ),
                )

            frame_rows = [_read_frame_metadata(paths.metadata_dir / f"{frame.frame_id:06d}.json", frame) for frame in frames]
            _write_jsonl(paths.frame_index, frame_rows)
            normal_count = _count_generated_kind(paths, frames, "normal")
            depth_count = _count_generated_kind(paths, frames, "depth")
            summary = {
                "stage": "interactive_view_evidence",
                "timestampUtc": _now(),
                "method": "Generated per-frame normal/depth images for debugging and geometry-aware proposal editing.",
                "runName": config.run_name,
                "runDir": str(paths.run_dir),
                "source": self._source_summary(frames, config),
                "parameters": config.to_json(),
                "frameCount": int(len(frames)),
                "normalFrameCount": int(normal_count),
                "depthFrameCount": int(depth_count),
                "frames": [
                    {
                        "frameId": int(row["frameId"]),
                        "normalReady": bool(row["normalReady"]),
                        "depthReady": bool(row["depthReady"]),
                    }
                    for row in frame_rows
                ],
                "outputs": {
                    "normalDir": str(paths.normal_dir),
                    "normalNpzDir": str(paths.normal_npz_dir),
                    "normalEdgeDir": str(paths.normal_edge_dir),
                    "depthDir": str(paths.depth_dir),
                    "depthNpzDir": str(paths.depth_npz_dir),
                    "depthEdgeDir": str(paths.depth_edge_dir),
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
                currentFrameId=None,
                completedFrameCount=int(len(frames)),
                normalFrameCount=int(normal_count),
                depthFrameCount=int(depth_count),
                message=f"View evidence ready: {normal_count} normal frames, {depth_count} depth frames.",
            )
            for target in config.targets:
                count = normal_count if target == "normal" else depth_count
                _write_evidence_kind_status(
                    paths,
                    target,
                    {
                        "target": target,
                        "ready": bool(frames and count >= len(frames)),
                        "running": False,
                        "failed": False,
                        "frameCount": len(frames),
                        "completedFrameCount": int(count),
                        "message": f"{_target_label((target,))} ready: {count}/{len(frames)} frames.",
                        "updatedUtc": _now(),
                    },
                )
        except Exception as exc:  # pragma: no cover - surfaced through UI status
            for target in config.targets:
                count = _count_generated_kind(paths, frames, target)
                _write_evidence_kind_status(
                    paths,
                    target,
                    {
                        "target": target,
                        "ready": bool(frames and count >= len(frames)),
                        "running": False,
                        "failed": True,
                        "frameCount": len(frames),
                        "completedFrameCount": int(count),
                        "currentFrameId": getattr(locals().get("frame", None), "frame_id", None),
                        "message": f"{_target_label((target,))} failed: {exc}",
                        "updatedUtc": _now(),
                    },
                )
            self._update_job(
                paths,
                ready=False,
                running=False,
                failed=True,
                message=f"View evidence generation failed: {exc}",
            )

    def _update_job(self, paths: ViewEvidenceRunPaths, **updates: Any) -> None:
        with self._lock:
            if self._job is None:
                return
            self._job.update(updates)
            self._job["updatedUtc"] = _now()
            payload = dict(self._job)
        _write_json(paths.progress, payload)
        if payload.get("running"):
            for target in payload.get("targets", []):
                _write_evidence_kind_status(
                    paths,
                    str(target),
                    {
                        "target": str(target),
                        "ready": False,
                        "running": True,
                        "failed": False,
                        "frameCount": int(payload.get("frameCount", 0) or 0),
                        "completedFrameCount": int(payload.get("completedFrameCount", 0) or 0),
                        "currentFrameId": payload.get("currentFrameId"),
                        "message": str(payload.get("message", "")),
                        "updatedUtc": payload.get("updatedUtc", _now()),
                    },
                )

    def _capture_root(self) -> Path:
        if self.paths.model_dir.parent.name == "omega_stable_mesh":
            return self.paths.model_dir.parent.parent
        return self.paths.model_dir

    def _source_summary(self, frames: list[ViewEvidenceFrame], config: ViewEvidenceConfig) -> dict[str, Any]:
        return {
            "baselineName": self.paths.baseline_name,
            "baselineDir": str(self.paths.baseline_dir),
            "datasetDir": str(self.paths.dataset_dir),
            "captureRoot": str(self._capture_root()),
            "frameManifest": str(self.paths.frame_manifest),
            "frameCount": int(len(frames)),
            "targets": list(config.targets),
            "normalSource": config.normal_source,
            "depthSource": config.depth_source,
        }

