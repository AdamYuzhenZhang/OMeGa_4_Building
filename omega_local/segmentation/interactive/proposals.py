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
from .propagation_backends import (
    is_propagation_layer,
    normalize_method_id,
    propagation_layer_key,
    propagation_method_id,
)
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
    input_manifest: Path


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
        return self._paths_for_dir(run_dir)

    def propagation_root(self) -> Path:
        return self.paths.interactive_dir / "proposals" / "propagation"

    def propagation_paths(self, method_id: str) -> ProposalRunPaths:
        run_dir = self.propagation_root() / normalize_method_id(method_id)
        return self._paths_for_dir(run_dir)

    def save_propagation_registry(self, payload: dict[str, Any]) -> Path:
        path = self.propagation_root() / "registry.json"
        _write_json(path, payload)
        return path

    def reconcile_interrupted_propagation_runs(self, registry: dict[str, Any]) -> None:
        for row in registry.get("methods", []):
            method_id = str(row.get("methodId") or "").strip()
            if not method_id:
                continue
            progress_path = self.propagation_paths(method_id).progress
            if not progress_path.is_file():
                continue
            try:
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if payload.get("status") != "running":
                continue
            payload["status"] = "interrupted"
            payload["message"] = "Previous run was interrupted; run this method again."
            payload["updatedUtc"] = _now()
            _write_json(progress_path, payload)

    @staticmethod
    def _paths_for_dir(run_dir: Path) -> ProposalRunPaths:
        return ProposalRunPaths(
            run_dir=run_dir,
            label_map_dir=run_dir / "label_maps",
            overlay_dir=run_dir / "overlays",
            metadata_dir=run_dir / "metadata",
            summary=run_dir / "summary.json",
            progress=run_dir / "progress.json",
            frame_index=run_dir / "frames.jsonl",
            config=run_dir / "config.json",
            input_manifest=run_dir / "input.json",
        )

    def clear_propagation_output(self, method_id: str) -> None:
        paths = self.propagation_paths(method_id)
        if paths.run_dir.exists():
            shutil.rmtree(paths.run_dir)

    def save_propagation_config(
        self,
        method_id: str,
        *,
        config: dict[str, Any],
        run_input: dict[str, Any],
    ) -> ProposalRunPaths:
        paths = self.propagation_paths(method_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(paths.config, config)
        _write_json(paths.input_manifest, run_input)
        return paths

    def save_propagation_progress(self, method_id: str, payload: dict[str, Any]) -> Path:
        paths = self.propagation_paths(method_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(paths.progress, payload)
        return paths.progress

    def propagation_frame_summaries(self, method_id: str) -> list[dict[str, Any]]:
        return _metadata_summaries(self.propagation_paths(method_id).metadata_dir)

    def layer_paths(self, frames: list[ProposalFrame], layer: str, run_name: str | None = None) -> ProposalRunPaths:
        layer_key = normalize_proposal_layer(layer)
        if layer_key == "sam2":
            return self.run_paths(_slug_token(run_name or self.default_run_name(frames)))
        if is_propagation_layer(layer_key):
            return self.propagation_paths(propagation_method_id(layer_key))
        raise ValueError(f"Unhandled proposal layer '{layer_key}'.")

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
            if payload.get("running"):
                payload["ready"] = bool(
                    paths.summary.is_file()
                    and _count_complete_frames(paths, frames) == len(frames)
                )
                payload["running"] = False
                payload["failed"] = False
                payload["message"] = "Previous SAM2 run was interrupted; generate again to finish missing frames."
                _write_json(paths.progress, payload)
        elif paths.summary.exists():
            payload = {
                "runName": run_name,
                "runDir": str(paths.run_dir),
                "ready": True,
                "running": False,
                "failed": False,
                "frameCount": len(frames),
                "completedFrameCount": _count_complete_frames(paths, frames),
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
        payload.setdefault("completedFrameCount", _count_complete_frames(paths, frames))
        payload.setdefault("overlayUrlTemplate", "/api/proposals/sam2/frame/{frameId}/overlay")
        payload.setdefault("summaryPath", str(paths.summary))
        return payload

    def layer_status(
        self,
        frames: list[ProposalFrame],
        propagation_registry: dict[str, Any],
        run_name: str | None = None,
    ) -> dict[str, Any]:
        raw_name = _slug_token(run_name or self.default_run_name(frames))
        raw_status = self.status(frames, raw_name)
        method_rows = [
            dict(row)
            for row in propagation_registry.get("methods", [])
            if isinstance(row, dict)
        ]
        layer_specs = [
            {
                "key": "sam2",
                "label": "SAM2 Automatic",
                "description": "Original frame-local SAM2 anything proposals.",
                "kind": "automatic",
                "labelSpace": "local_proposal",
                "layerGroup": "frame_proposals",
                "methodId": None,
                "available": True,
                "availabilityMessage": "Ready",
            }
        ]
        for method in method_rows:
            layer_specs.append(
                {
                    "key": str(method.get("layerKey") or propagation_layer_key(str(method.get("methodId", "")))),
                    "label": str(method.get("displayName") or method.get("methodId") or "Propagation"),
                    "description": str(method.get("description") or "Anchor-conditioned region candidates."),
                    "kind": "propagation",
                    "labelSpace": str(method.get("labelSpace") or "persistent_region"),
                    "readOnly": bool(method.get("readOnly", False)),
                    "methodId": str(method.get("methodId") or ""),
                    "engineName": str(method.get("engineName") or ""),
                    "supportsFullRun": bool(method.get("supportsFullRun", True)),
                    "supportsRegionPair": bool(method.get("supportsRegionPair", False)),
                    "stage": str(method.get("stage") or "anchor_to_mask"),
                    "sourceMethodId": str(method.get("sourceMethodId") or ""),
                    "sourceLayerKey": str(method.get("sourceLayerKey") or ""),
                    "layerGroup": str(method.get("layerGroup") or "video_propagation"),
                    "available": bool(method.get("available", False)),
                    "availabilityMessage": str(method.get("availabilityMessage") or ""),
                }
            )

        layers = []
        anchor_updated_ns = (
            self.paths.regions_summary.stat().st_mtime_ns
            if self.paths.regions_summary.is_file()
            else 0
        )
        for spec in layer_specs:
            layer_key = normalize_proposal_layer(str(spec["key"]))
            paths = self.layer_paths(frames, layer_key, raw_name)
            frame_count = _count_complete_frames(paths, frames) if paths.summary.exists() else 0
            ready = bool(paths.summary.exists() and frame_count == len(frames))
            if bool(spec.get("supportsRegionPair", False)):
                ready = bool(paths.summary.exists() and frame_count > 0)
            updated = ""
            progress: dict[str, Any] = {}
            if paths.progress.exists():
                try:
                    progress = json.loads(paths.progress.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    progress = {}
            if paths.summary.exists():
                try:
                    summary = json.loads(paths.summary.read_text(encoding="utf-8"))
                    updated = str(summary.get("timestampUtc") or summary.get("updatedUtc") or "")
                except json.JSONDecodeError:
                    updated = ""
            progress_updated = str(progress.get("updatedUtc") or "")
            if progress_updated > updated:
                updated = progress_updated
            running = progress.get("status") == "running"
            failed = progress.get("status") == "failed"
            message = str(progress.get("message") or "")
            anchor_stale = bool(
                layer_key != "sam2"
                and spec.get("supportsFullRun", False)
                and paths.summary.is_file()
                and paths.summary.stat().st_mtime_ns < anchor_updated_ns
            )
            if layer_key == "sam2":
                ready = bool(raw_status.get("ready", ready) and frame_count == len(frames))
                running = bool(raw_status.get("running", False))
                failed = bool(raw_status.get("failed", False))
                message = str(raw_status.get("message") or message)
            elif anchor_stale:
                ready = False
                if not running:
                    message = "This result is stale because the manual anchors were updated."
            layers.append(
                {
                    "key": layer_key,
                    **{key: value for key, value in spec.items() if key != "key"},
                    "ready": ready,
                    "stale": anchor_stale,
                    "running": running,
                    "failed": failed,
                    "message": message,
                    "runName": paths.run_dir.name,
                    "runDir": str(paths.run_dir),
                    "completedFrameCount": int(frame_count),
                    "frameCount": int(len(frames)),
                    "summaryPath": str(paths.summary),
                    "updatedUtc": updated,
                    "overlayUrlTemplate": f"/api/proposals/layers/{layer_key}/frame/{{frameId}}/overlay",
                }
            )
        layer_by_key = {str(layer["key"]): layer for layer in layers}
        for layer in layers:
            if not str(layer.get("sourceLayerKey") or ""):
                continue
            source_key = normalize_proposal_layer(str(layer.get("sourceLayerKey") or ""))
            source = layer_by_key.get(source_key)
            source_ready = bool(source and source.get("ready"))
            source_updated = str(source.get("updatedUtc") or "") if source else ""
            layer["sourceReady"] = source_ready
            layer["sourceUpdatedUtc"] = source_updated
            if not source_ready:
                layer["ready"] = False
                layer["message"] = "Run the required source method first."
            elif int(layer.get("completedFrameCount") or 0) == 0:
                layer["ready"] = False
                layer["message"] = "Ready to process the saved source layer."
            elif source_updated and str(layer.get("updatedUtc") or "") < source_updated:
                layer["ready"] = False
                layer["message"] = "This result is stale because its source layer was updated."
        return {
            "ready": any(bool(layer["ready"]) for layer in layers),
            "layers": layers,
            "defaultLayer": "sam2",
            "pickOrder": [str(layer["key"]) for layer in layers],
            "defaultPropagationMethodId": str(propagation_registry.get("defaultMethodId") or ""),
            "defaultSourceRefinementMethodId": str(
                propagation_registry.get("defaultSourceRefinementMethodId") or ""
            ),
            "methods": method_rows,
            "updatedUtc": max((str(layer.get("updatedUtc") or "") for layer in layers), default=""),
        }

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
        paths = self.run_paths(_slug_token(run_name or self.default_run_name(frames)))
        return paths.overlay_dir / f"{int(frame_id):06d}.png"

    def layer_overlay_path(self, frames: list[ProposalFrame], frame_id: int, layer: str, run_name: str | None = None) -> Path:
        paths = self.layer_paths(frames, layer, run_name)
        return paths.overlay_dir / f"{int(frame_id):06d}.png"

    def layer_label_map_path(self, frames: list[ProposalFrame], frame_id: int, layer: str, run_name: str | None = None) -> Path:
        paths = self.layer_paths(frames, layer, run_name)
        return paths.label_map_dir / f"{int(frame_id):06d}.npy"

    def layer_metadata_path(self, frames: list[ProposalFrame], frame_id: int, layer: str, run_name: str | None = None) -> Path:
        paths = self.layer_paths(frames, layer, run_name)
        return paths.metadata_dir / f"{int(frame_id):06d}.json"

    def save_region_candidate_map(
        self,
        frames: list[ProposalFrame],
        frame_id: int,
        labels: np.ndarray,
        *,
        method_id: str,
        region_rows: list[dict[str, Any]],
        source_frame_ids_by_region: dict[int, list[int]] | None = None,
        result_description: str,
        input_fingerprint: str,
        stage: str = "anchor_to_mask",
    ) -> dict[str, Any]:
        method_id = normalize_method_id(method_id)
        layer_key = propagation_layer_key(method_id)
        paths = self.propagation_paths(method_id)
        frame = next((item for item in frames if int(item.frame_id) == int(frame_id)), None)
        if frame is None:
            raise KeyError(frame_id)
        for directory in [
            paths.label_map_dir,
            paths.overlay_dir,
            paths.metadata_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        labels = np.asarray(labels, dtype=np.uint16)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D propagated label map, got shape {labels.shape}.")
        expected = (int(frame.height), int(frame.width))
        if labels.shape != expected:
            raise ValueError(f"Propagated label map shape {labels.shape} does not match frame shape {expected}.")

        npy_path = paths.label_map_dir / f"{int(frame_id):06d}.npy"
        metadata_path = paths.metadata_dir / f"{int(frame_id):06d}.json"
        source_frame_ids_by_region = source_frame_ids_by_region or {}

        color_by_label = _region_color_map(region_rows)
        png_path = paths.label_map_dir / f"{int(frame_id):06d}.png"
        overlay_path = paths.overlay_dir / f"{int(frame_id):06d}.png"
        np.save(npy_path, labels)
        Image.fromarray(labels).save(png_path)
        Image.fromarray(_transparent_label_overlay(labels, color_by_label), mode="RGBA").save(overlay_path)

        label_rows = _label_rows_from_label_map(labels)
        for row in label_rows:
            region_id = int(row["labelId"])
            row["regionId"] = region_id
            row["sourceRegionId"] = region_id
            row["proposalKind"] = "propagated_region"
            row["sourceFrameIds"] = _source_frame_ids(source_frame_ids_by_region.get(region_id, []))
            row["sourceFrameId"] = int(row["sourceFrameIds"][0]) if row["sourceFrameIds"] else -1
            row["layer"] = layer_key
            row["methodId"] = method_id
            row["labelSpace"] = "persistent_region"

        all_source_frame_ids = sorted(
            {
                int(value)
                for values in source_frame_ids_by_region.values()
                for value in _source_frame_ids(values)
                if int(value) >= 0
            }
        )
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
            "layer": layer_key,
            "methodId": method_id,
            "stage": str(stage),
            "labelSpace": "persistent_region",
            "sourceFrameId": int(all_source_frame_ids[0]) if all_source_frame_ids else -1,
            "sourceFrameIds": all_source_frame_ids,
            "updatedUtc": _now(),
        }
        _write_json(metadata_path, payload)
        _write_jsonl(paths.frame_index, _metadata_summaries(paths.metadata_dir))
        summary = {
            "schemaVersion": 1,
            "stage": _candidate_stage_name(stage),
            "timestampUtc": _now(),
            "methodId": method_id,
            "method": result_description,
            "inputFingerprint": str(input_fingerprint),
            "runName": paths.run_dir.name,
            "runDir": str(paths.run_dir),
            "layer": layer_key,
            "labelSpace": "persistent_region",
            "frameCount": int(len(frames)),
            "frames": _metadata_summaries(paths.metadata_dir),
            "outputs": {
                "labelMapDir": str(paths.label_map_dir),
                "overlayDir": str(paths.overlay_dir),
                "metadataDir": str(paths.metadata_dir),
                "frameIndex": str(paths.frame_index),
                "config": str(paths.config),
                "input": str(paths.input_manifest),
            },
        }
        _write_json(paths.summary, summary)
        return payload

    def save_propagation_diagnostics(self, method_id: str, rows: list[dict[str, Any]]) -> Path:
        paths = self.propagation_paths(method_id)
        path = paths.run_dir / "diagnostics.jsonl"
        _write_jsonl(path, [dict(row) for row in rows])
        if paths.summary.is_file():
            summary = json.loads(paths.summary.read_text(encoding="utf-8"))
            outputs = summary.setdefault("outputs", {})
            outputs["diagnostics"] = str(path)
            _write_json(paths.summary, summary)
        return path

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
                    Image.fromarray(label_map.astype(np.uint16)).save(png_path)
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
            paths.progress.unlink(missing_ok=True)
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


def _candidate_stage_name(stage: str) -> str:
    names = {
        "anchor_to_mask": "interactive_anchor_to_mask_candidates",
        "dense_recovery": "interactive_dense_recovery_candidates",
        "identity_refinement": "interactive_identity_refinement_candidates",
    }
    return names.get(str(stage), f"interactive_{str(stage).strip() or 'candidate'}_candidates")


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


def normalize_proposal_layer(value: str) -> str:
    key = str(value or "sam2").strip().lower().replace("-", "_")
    if key == "sam2":
        return "sam2"
    if is_propagation_layer(key):
        return propagation_layer_key(propagation_method_id(key))
    raise ValueError(f"Unknown proposal layer '{value}'. Expected sam2 or propagation_<method_id>.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(path)


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


def _source_frame_ids(value: Any) -> list[int]:
    raw_values = value if isinstance(value, list) else [value]
    out: set[int] = set()
    for item in raw_values:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            out.add(parsed)
    return sorted(out)


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


def _transparent_label_overlay(
    labels: np.ndarray,
    color_by_label: dict[int, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    overlay = np.zeros((*labels.shape, 4), dtype=np.uint8)
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        mask = labels == label_i
        color = color_by_label.get(label_i) if color_by_label else None
        overlay[mask, :3] = np.asarray(color, dtype=np.uint8) if color else _label_color(label_i)
        overlay[mask, 3] = 138
    return overlay


def _region_color_map(region_rows: list[dict[str, Any]]) -> dict[int, tuple[int, int, int]]:
    return {
        int(row.get("id", 0)): _hsl_to_rgb(str(row.get("color", "")))
        for row in region_rows
        if int(row.get("id", 0)) > 0
    }


def _hsl_to_rgb(css: str) -> tuple[int, int, int]:
    try:
        raw = css.strip().lower()
        raw = raw.removeprefix("hsl(").removesuffix(")")
        parts = raw.replace("%", "").split()
        h = float(parts[0]) % 360.0
        s = float(parts[1]) / 100.0
        light = float(parts[2]) / 100.0
    except Exception:
        return (124, 199, 255)

    c = (1.0 - abs(2.0 * light - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = light - c / 2.0
    if h < 60:
        rgb = (c, x, 0.0)
    elif h < 120:
        rgb = (x, c, 0.0)
    elif h < 180:
        rgb = (0.0, c, x)
    elif h < 240:
        rgb = (0.0, x, c)
    elif h < 300:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    return tuple(int(round((channel + m) * 255.0)) for channel in rgb)


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


def _count_complete_frames(paths: ProposalRunPaths, frames: list[ProposalFrame]) -> int:
    return sum(
        1
        for frame in frames
        if (paths.label_map_dir / f"{int(frame.frame_id):06d}.npy").is_file()
        and (paths.metadata_dir / f"{int(frame.frame_id):06d}.json").is_file()
    )
