"""Mutable data model behind the interactive segmentation editor."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .dinov3_evidence import DinoV3EvidenceManager
from .geometry_selection import grow_normal_selection
from .keyframes import KeyframeConfig, KeyframeFrame, detect_keyframes, read_keyframes
from .mask_edits import compose_selection_mask, selected_mask_overlay
from .paths import EditorPaths, read_jsonl
from .point_cloud_sources import PointCloudSourceManager
from .region_pair_test import build_region_pair_test_input
from .propagation_backends import (
    PropagationBackendRegistry,
    PropagationFrame,
    build_propagation_input,
    discover_split_splat_layer_status,
)
from .proposals import ProposalFrame, ProposalManager, normalize_proposal_layer
from .regions import PersistentRegionManager
from .rgbd_cue_selection import cue_select_rgbd, rgbd_cue_debug
from .sam2_session import Sam2Config, Sam2Session, encode_mask_overlay, encode_mask_png
from .segmentation3d_manager import Segmentation3DManager
from .view_evidence import ViewEvidenceFrame, ViewEvidenceManager
from .v2sam_cache import DinoFrame


class EditorState:
    def __init__(
        self,
        paths: EditorPaths,
        *,
        max_points: int,
        seed: int,
        label_source: str,
        sam2_config: Sam2Config,
        sam2_session: Sam2Session,
        propagation_backends: PropagationBackendRegistry,
        dinov3_evidence: DinoV3EvidenceManager,
        segmentation3d: Segmentation3DManager,
    ) -> None:
        self.paths = paths
        self.max_points = max(int(max_points), 1)
        self.seed = int(seed)
        self.label_source = str(label_source)
        self.sam2_config = sam2_config
        self.propagation_backends = propagation_backends
        self.sam2_session = sam2_session
        self.proposals = ProposalManager(paths, sam2_config)
        propagation_registry = propagation_backends.status()
        self.proposals.save_propagation_registry(propagation_registry)
        self.proposals.reconcile_interrupted_propagation_runs(propagation_registry)
        self.regions = PersistentRegionManager(paths)
        self.view_evidence = ViewEvidenceManager(paths)
        self.dinov3_evidence = dinov3_evidence
        self.segmentation3d = segmentation3d
        self.point_cloud_sources = PointCloudSourceManager(paths)
        self._manifest_rows: list[dict[str, Any]] | None = None
        self._manifest_by_frame_id: dict[int, dict[str, Any]] | None = None
        self._points_full: np.ndarray | None = None
        self._labels_full: np.ndarray | None = None
        self._active_label_source: str | None = None
        self._label_summary: dict[str, Any] | None = None
        self._points_payload: dict[str, Any] | None = None
        self._frames: list[dict[str, Any]] | None = None
        self._frame_by_id: dict[int, dict[str, Any]] | None = None
        self._propagation_jobs: dict[str, dict[str, Any]] = {}
        self._propagation_job_lock = threading.Lock()

    def project_summary(self) -> dict[str, Any]:
        frames = self.frames()
        points = self.points_payload()
        frame_resolution = {
            "width": int(frames[0]["width"]) if frames else 0,
            "height": int(frames[0]["height"]) if frames else 0,
            "policy": "Interactive proposals, prompts, propagation outputs, StableNormal, Depth Anything, and DINOv3 PCA evidence use the staged frame grid.",
        }
        return {
            "modelDir": str(self.paths.model_dir),
            "baselineName": self.paths.baseline_name,
            "baselineDir": str(self.paths.baseline_dir),
            "frameCount": len(frames),
            "frameResolution": frame_resolution,
            "pointCount": points["pointCount"],
            "servedPointCount": points["servedPointCount"],
            "labelCount": points["labelCount"],
            "bounds": points["bounds"],
            "pointsPath": str(self.paths.points_path),
            "pointCloudSources": self.point_cloud_sources_status(),
            "labelSource": self._active_label_source or self.label_source,
            "sai3dLabelsPath": str(self.paths.point_labels) if self.paths.point_labels is not None else "",
            "sam2": {
                "root": str(self.sam2_config.root),
                "checkpoint": str(self.sam2_config.checkpoint),
                "config": str(self.sam2_config.config),
                "device": str(self.sam2_config.device),
            },
            "propagation": self.propagation_backends.status(),
            "segmentation3d": self.segmentation3d.status(),
        }

    def segmentation3d_status(self) -> dict[str, Any]:
        return self.segmentation3d.status()

    def start_segmentation3d_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.segmentation3d.start(payload)

    def segmentation3d_job_status(self, job_id: str) -> dict[str, Any]:
        return self.segmentation3d.job_status(job_id)

    def segmentation3d_result_points(self, run_id: str) -> dict[str, Any]:
        return self.segmentation3d.result_points(run_id)

    def point_cloud_sources_status(self) -> dict[str, Any]:
        return {
            "colmap": self.point_cloud_sources.status(),
            "feedforward": self.point_cloud_sources.feedforward_status(),
            "omegaFinal": self.point_cloud_sources.omega_final_status(),
        }

    def colmap_points_payload(self, *, mode: str = "raw") -> dict[str, Any]:
        return self.point_cloud_sources.colmap_payload(
            self.points_full(),
            max_points=self.max_points,
            seed=self.seed + 1009,
            mode=mode,
        )

    def feedforward_points_payload(self, *, mode: str = "raw") -> dict[str, Any]:
        return self.point_cloud_sources.feedforward_payload(
            self.points_full(),
            max_points=self.max_points,
            seed=self.seed + 2017,
            mode=mode,
        )

    def omega_final_points_payload(self, *, mode: str = "raw") -> dict[str, Any]:
        return self.point_cloud_sources.omega_final_payload(
            max_points=self.max_points,
            seed=self.seed + 3023,
            mode=mode,
        )

    def manifest_rows(self) -> list[dict[str, Any]]:
        if self._manifest_rows is None:
            rows = read_jsonl(self.paths.frame_manifest)
            self._manifest_rows = rows
            self._manifest_by_frame_id = {int(row["sai3dFrameId"]): row for row in rows}
        return self._manifest_rows

    def manifest_row(self, frame_id: int) -> dict[str, Any]:
        self.manifest_rows()
        assert self._manifest_by_frame_id is not None
        if int(frame_id) not in self._manifest_by_frame_id:
            raise KeyError(frame_id)
        return self._manifest_by_frame_id[int(frame_id)]

    def frames(self) -> list[dict[str, Any]]:
        if self._frames is None:
            frames: list[dict[str, Any]] = []
            for row in self.manifest_rows():
                frame_id = int(row["sai3dFrameId"])
                pose_path = self.paths.dataset_dir / str(row["posePath"])
                pose = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
                frames.append(
                    {
                        "id": frame_id,
                        "sourceFrameId": int(row.get("sourceFrameId", frame_id)),
                        "imageName": row.get("imageName", f"{frame_id:06d}.jpg"),
                        "width": int(row["width"]),
                        "height": int(row["height"]),
                        "fx": float(row["fx"]),
                        "fy": float(row["fy"]),
                        "cx": float(row["cx"]),
                        "cy": float(row["cy"]),
                        "poseWorldFromCamera": pose.reshape(-1).tolist(),
                        "imageUrl": f"/api/frame/{frame_id}/image",
                    }
                )
            self._frames = frames
            self._frame_by_id = {int(frame["id"]): frame for frame in frames}
        return self._frames

    def frame(self, frame_id: int) -> dict[str, Any]:
        self.frames()
        assert self._frame_by_id is not None
        if int(frame_id) not in self._frame_by_id:
            raise KeyError(frame_id)
        return self._frame_by_id[int(frame_id)]

    def image_path(self, frame_id: int) -> Path:
        row = self.manifest_row(frame_id)
        staged = self.paths.dataset_dir / str(row["colorPath"])
        if staged.exists():
            return staged
        source = Path(str(row.get("sourceImagePath", "")))
        if source.exists():
            return source
        raise FileNotFoundError(f"No RGB image found for frame {frame_id}")

    def inference_image_path(self, frame_id: int) -> Path:
        # Keep all interactive 2D products on one pixel grid. Full-resolution
        # runs should be staged as full-resolution manifests instead of mixing
        # original capture images with downsampled proposal images.
        return self.image_path(frame_id)

    def image_rgb(self, frame_id: int) -> np.ndarray:
        path = self.image_path(frame_id)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"))

    def proposal_frames(self) -> list[ProposalFrame]:
        frames: list[ProposalFrame] = []
        for frame in self.frames():
            frame_id = int(frame["id"])
            frames.append(
                ProposalFrame(
                    frame_id=frame_id,
                    width=int(frame["width"]),
                    height=int(frame["height"]),
                    image_name=str(frame["imageName"]),
                    image_path=self.image_path(frame_id),
                )
            )
        return frames

    def view_evidence_frames(self) -> list[ViewEvidenceFrame]:
        frames: list[ViewEvidenceFrame] = []
        for frame in self.frames():
            frame_id = int(frame["id"])
            row = self.manifest_row(frame_id)
            frames.append(
                ViewEvidenceFrame(
                    frame_id=frame_id,
                    source_frame_id=int(row.get("sourceFrameId", row.get("frameID", frame_id))),
                    scan_id=str(row.get("scanID", "scan_000")),
                    width=int(frame["width"]),
                    height=int(frame["height"]),
                    image_name=str(frame["imageName"]),
                    image_path=self.image_path(frame_id),
                    inference_image_path=self.inference_image_path(frame_id),
                    manifest_row=row,
                )
            )
        return frames

    def keyframe_status(self) -> dict[str, Any]:
        payload = read_keyframes(self.paths.keyframes_summary)
        if payload is None:
            return {
                "ready": False,
                "running": False,
                "failed": False,
                "frameCount": len(self.frames()),
                "keyframeCount": 0,
                "keyframeIds": [],
                "message": "No keyframe analysis found.",
                "summaryPath": str(self.paths.keyframes_summary),
            }
        payload = dict(payload)
        payload["ready"] = True
        payload.setdefault("running", False)
        payload.setdefault("failed", False)
        payload.setdefault("message", f"{int(payload.get('keyframeCount', 0))} keyframes detected.")
        payload.setdefault("summaryPath", str(self.paths.keyframes_summary))
        return payload

    def detect_keyframes(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = KeyframeConfig.from_payload(payload)
        frames = self._keyframe_frames()
        result = detect_keyframes(
            frames,
            output_path=self.paths.keyframes_summary,
            config=config,
        )
        result["ready"] = True
        result["running"] = False
        result["failed"] = False
        result["message"] = f"Detected {int(result.get('keyframeCount', 0))} keyframes."
        result["summaryPath"] = str(self.paths.keyframes_summary)
        return result

    def _keyframe_frames(self) -> list[KeyframeFrame]:
        out: list[KeyframeFrame] = []
        proposal_frames = self.proposal_frames()
        for frame in self.frames():
            frame_id = int(frame["id"])
            row = self.manifest_row(frame_id)
            proposal_label_path = self._keyframe_proposal_label_path(frame_id, row, proposal_frames)
            depth_path = self._optional_dataset_path(row.get("depthPath"))
            depth_stats = row.get("depthStats") if isinstance(row.get("depthStats"), dict) else {}
            out.append(
                KeyframeFrame(
                    frame_id=frame_id,
                    image_name=str(frame["imageName"]),
                    image_path=self.image_path(frame_id),
                    pose_world_from_camera=np.asarray(frame["poseWorldFromCamera"], dtype=np.float64).reshape(4, 4),
                    width=int(frame["width"]),
                    height=int(frame["height"]),
                    proposal_label_path=proposal_label_path,
                    depth_path=depth_path,
                    depth_coverage=float(depth_stats.get("coverage", 0.0) or 0.0),
                    proposal_coverage=float(row.get("inputCoverage", 0.0) or 0.0),
                    proposal_label_count=int(row.get("inputLabelCount", 0) or 0),
                )
            )
        return out

    def _keyframe_proposal_label_path(
        self,
        frame_id: int,
        row: dict[str, Any],
        frames: list[ProposalFrame],
    ) -> Path | None:
        run_name = self.proposals.default_run_name(frames)
        paths = self.proposals.run_paths(run_name)
        for suffix in [".npy", ".png"]:
            path = paths.label_map_dir / f"{int(frame_id):06d}{suffix}"
            if path.exists():
                return path
        return self._optional_dataset_path(row.get("maskPath"))

    def _optional_dataset_path(self, value: Any) -> Path | None:
        if not value:
            return None
        path = Path(str(value))
        if not path.is_absolute():
            path = self.paths.dataset_dir / path
        return path if path.exists() else None

    def proposal_status(self) -> dict[str, Any]:
        frames = self.proposal_frames()
        status = self.proposals.status(frames)
        status["editingRepresentation"] = "candidate proposal layers plus confirmed region label maps"
        status["layers"] = self.proposal_layers_status()
        return status

    def proposal_layers_status(self) -> dict[str, Any]:
        registry = self.propagation_backends.status()
        methods = list(registry.get("methods", []))
        known_ids = {
            str(row.get("methodId") or "")
            for row in methods
            if isinstance(row, dict)
        }
        for row in discover_split_splat_layer_status(
            self.paths.interactive_dir / "tmp"
        ):
            method_id = str(row.get("methodId") or "")
            if method_id and method_id not in known_ids:
                methods.append(row)
                known_ids.add(method_id)
        registry["methods"] = methods
        return self.proposals.layer_status(
            self.proposal_frames(),
            registry,
        )

    def region_status(self) -> dict[str, Any]:
        return self.regions.status()

    def region_frame_summary(self, frame_id: int) -> dict[str, Any]:
        self.frame(frame_id)
        return self.regions.frame_summary(frame_id)

    def set_region_frame_complete(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        self.frame(frame_id)
        return self.regions.set_frame_complete(
            frame_id=int(frame_id),
            complete=_truthy(payload.get("complete", True)),
        )

    def region_overlay_path(self, frame_id: int) -> Path:
        self.frame(frame_id)
        path = self.regions.overlay_path(frame_id)
        if not path.exists():
            raise FileNotFoundError(f"Persistent region overlay does not exist for frame {frame_id}: {path}")
        return path

    def single_region_overlay_png(self, frame_id: int, region_id: int) -> bytes:
        self.frame(frame_id)
        return self.regions.single_region_overlay_png(frame_id=int(frame_id), region_id=int(region_id))

    def start_proposal_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.proposals.start(self.proposal_frames(), payload)

    def view_evidence_status(self) -> dict[str, Any]:
        payload = self.view_evidence.status(self.view_evidence_frames())
        dino_status = self.dinov3_evidence.status(self.dinov3_frames())
        payload["dinoFrameCount"] = int(dino_status["generatedFrameCount"])
        payload["dinoReady"] = bool(dino_status["ready"])
        payload["dinoStatus"] = dino_status
        payload["dinoUrlTemplate"] = dino_status["urlTemplate"]
        payload["ready"] = bool(payload.get("ready") or dino_status["ready"])
        payload["running"] = bool(payload.get("running") or dino_status["running"])
        payload["failed"] = bool(payload.get("failed") or dino_status["failed"])
        return payload

    def start_view_evidence_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        target = str(payload.get("target", "")).strip().lower().replace("_", "-")
        if target in {"dino", "dinov3", "dino-pca", "dinov3-pca"}:
            if self._propagation_job_running():
                raise RuntimeError("Wait for the active propagation job before generating DINOv3 evidence.")
            self.dinov3_evidence.start(
                self.dinov3_frames(),
                overwrite=bool(payload.get("overwrite", False)),
            )
            return self.view_evidence_status()
        return self.view_evidence.start(self.view_evidence_frames(), payload)

    def dinov3_frames(self) -> list[DinoFrame]:
        return [
            DinoFrame(
                local_index=index,
                frame_id=int(frame["id"]),
                image_path=self.image_path(int(frame["id"])),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
            for index, frame in enumerate(self.frames())
        ]

    def view_evidence_image_path(self, frame_id: int, kind: str) -> Path:
        self.frame(frame_id)
        key = str(kind).strip().lower().replace("_", "-")
        if key in {"dino", "dinov3", "dino-pca", "dinov3-pca"}:
            return self.dinov3_evidence.image_path(int(frame_id))
        return self.view_evidence.image_path(self.view_evidence_frames(), int(frame_id), str(kind))

    def grow_evidence_selection(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self.frame(frame_id)
        mode = str(payload.get("mode", payload.get("tool", "normal-grow"))).strip().lower().replace("_", "-")
        x = float(payload.get("x", np.nan))
        y = float(payload.get("y", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("Geometry grow selection needs finite x/y seed coordinates.")
        if x < 0 or x >= int(frame["width"]) or y < 0 or y >= int(frame["height"]):
            raise ValueError("Geometry grow seed is outside the active frame.")
        strength = str(payload.get("strength", "normal"))

        frames = self.view_evidence_frames()
        if mode in {"normal-grow", "normal"}:
            evidence_path = self.view_evidence.npz_path(frames, int(frame_id), "normal")
            normal_angle_deg = payload.get("normalAngleDeg", None)
            result = grow_normal_selection(evidence_path, (x, y), strength, normal_angle_deg=normal_angle_deg)
        elif mode == "rgbd-cue-select":
            depth_path = self.view_evidence.npz_path(frames, int(frame_id), "depth")
            normal_path = self.view_evidence.npz_path(frames, int(frame_id), "normal")
            result = cue_select_rgbd(
                rgb=self.image_rgb(frame_id),
                depth_path=depth_path,
                normal_path=normal_path,
                seed_xy=(x, y),
                strength=strength,
                smoothness=payload.get("cueSmoothness", payload.get("graphBoundaryWeight", payload.get("boundaryWeight", None))),
                prompts=payload.get("prompts", None),
                n_segments=payload.get("nSegments", payload.get("superpixels", None)),
            )
        else:
            raise ValueError(f"Unknown geometry grow mode: {mode}")

        selected = np.asarray(result.mask, dtype=bool)
        expected_shape = (int(frame["height"]), int(frame["width"]))
        if selected.shape != expected_shape:
            raise ValueError(f"Geometry grow mask shape {selected.shape} does not match frame shape {expected_shape}.")
        payload_out = result.to_json()
        payload_out.update(
            {
                "frameId": int(frame_id),
                "width": int(frame["width"]),
                "height": int(frame["height"]),
                "maskPng": encode_mask_png(selected),
                "maskOverlayPng": encode_mask_overlay(selected),
            }
        )
        return payload_out

    def rgbd_cue_debug(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        frame = self.frame(frame_id)
        x = float(payload.get("x", np.nan))
        y = float(payload.get("y", np.nan))
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("RGB-D Cue Debug needs finite x/y seed coordinates.")
        if x < 0 or x >= int(frame["width"]) or y < 0 or y >= int(frame["height"]):
            raise ValueError("RGB-D Cue Debug seed is outside the active frame.")
        depth_path = self.view_evidence.npz_path(self.view_evidence_frames(), int(frame_id), "depth")
        normal_path = self.view_evidence.npz_path(self.view_evidence_frames(), int(frame_id), "normal")
        payload_out = rgbd_cue_debug(
            rgb=self.image_rgb(frame_id),
            depth_path=depth_path,
            normal_path=normal_path,
            seed_xy=(x, y),
            strength=str(payload.get("strength", "normal")),
            smoothness=payload.get("cueSmoothness", payload.get("graphBoundaryWeight", payload.get("boundaryWeight", None))),
            prompts=payload.get("prompts", None),
            n_segments=payload.get("nSegments", payload.get("superpixels", None)),
        )
        payload_out.update({"frameId": int(frame_id)})
        return payload_out

    def proposal_overlay_path(self, frame_id: int) -> Path:
        self.frame(frame_id)
        path = self.proposals.layer_overlay_path(self.proposal_frames(), int(frame_id), "sam2")
        if not path.exists():
            raise FileNotFoundError(f"SAM2 proposal overlay does not exist for frame {frame_id}: {path}")
        return path

    def proposal_layer_overlay_path(self, frame_id: int, layer: str) -> Path:
        self.frame(frame_id)
        layer_key = normalize_proposal_layer(layer)
        path = self.proposals.layer_overlay_path(self.proposal_frames(), int(frame_id), layer_key)
        if not path.exists():
            raise FileNotFoundError(f"{layer_key} proposal overlay does not exist for frame {frame_id}: {path}")
        return path

    def proposal_label_map(self, frame_id: int) -> np.ndarray:
        return self._empty_proposal_label_map(frame_id)

    def proposal_layer_label_map(self, frame_id: int, layer: str) -> np.ndarray:
        frame = self.frame(frame_id)
        layer_key = normalize_proposal_layer(layer)
        path = self.proposals.layer_label_map_path(self.proposal_frames(), int(frame_id), layer_key)
        if not path.exists():
            raise FileNotFoundError(f"{layer_key} proposal label map does not exist for frame {frame_id}: {path}")
        labels = np.load(path).astype(np.uint16, copy=False)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D {layer_key} proposal label map, got shape {labels.shape}: {path}")
        expected_shape = (int(frame["height"]), int(frame["width"]))
        if labels.shape != expected_shape:
            raise ValueError(f"{layer_key} proposal label map shape {labels.shape} does not match frame {frame_id} shape {expected_shape}: {path}")
        return labels

    def proposal_layer_frame_summary(self, frame_id: int, layer: str) -> dict[str, Any]:
        self.frame(frame_id)
        layer_key = normalize_proposal_layer(layer)
        path = self.proposals.layer_metadata_path(self.proposal_frames(), int(frame_id), layer_key)
        if not path.exists():
            raise FileNotFoundError(f"{layer_key} proposal metadata does not exist for frame {frame_id}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["layer"] = layer_key
        for row in payload.get("labels", []):
            if isinstance(row, dict):
                row["layer"] = layer_key
        return payload

    def proposal_combined_frame_summary(self, frame_id: int) -> dict[str, Any]:
        self.frame(frame_id)
        layers = []
        labels = []
        for layer_status in self.proposal_layers_status().get("layers", []):
            layer_key = normalize_proposal_layer(str(layer_status.get("key", "")))
            if not layer_status.get("ready"):
                continue
            try:
                summary = self.proposal_layer_frame_summary(frame_id, layer_key)
            except FileNotFoundError:
                continue
            layers.append(
                {
                    "key": layer_key,
                    "label": layer_status.get("label", layer_key),
                    "ready": True,
                    "coverage": float(summary.get("coverage", 0.0)),
                    "labelCount": int(len(summary.get("labels", []))),
                }
            )
            labels.extend(summary.get("labels", []))
        return {
            "frameId": int(frame_id),
            "layers": layers,
            "labels": labels,
        }

    def pick_proposal(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        x_raw = float(payload.get("x", -1))
        y_raw = float(payload.get("y", -1))
        layers = _payload_layers(payload.get("layers", []))
        if not layers:
            layers = [
                normalize_proposal_layer(str(layer))
                for layer in self.proposal_layers_status().get("pickOrder", [])
            ]
        base = None
        for layer_key in layers:
            try:
                base = self.proposal_layer_label_map(frame_id, layer_key)
                break
            except FileNotFoundError:
                continue
        if base is None:
            return {"frameId": int(frame_id), "labelId": 0, "inside": False}
        if not np.isfinite(x_raw) or not np.isfinite(y_raw):
            return {"frameId": int(frame_id), "labelId": 0, "inside": False}
        x = int(np.floor(x_raw))
        y = int(np.floor(y_raw))
        if x < 0 or x >= base.shape[1] or y < 0 or y >= base.shape[0]:
            return {"frameId": int(frame_id), "labelId": 0, "inside": False}
        label = 0
        layer = ""
        details = {}
        hits: list[dict[str, Any]] = []
        for layer_key in layers:
            try:
                labels = self.proposal_layer_label_map(frame_id, layer_key)
            except FileNotFoundError:
                continue
            value = int(labels[y, x])
            if value <= 0:
                continue
            hits.append({"layer": layer_key, "labelId": value})
            if label <= 0:
                label = value
                layer = layer_key
                details = self._proposal_details(frame_id, layer_key).get(label, {})
        return {
            "frameId": int(frame_id),
            "labelId": label,
            "layer": layer,
            "layers": layers,
            "hits": hits,
            "x": int(x),
            "y": int(y),
            "inside": True,
            "details": details,
        }

    def preview_selection(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        labels = self._empty_proposal_label_map(frame_id)
        selected_raw = self._compose_proposal_selection(frame_id, payload, labels, protect_regions=False)
        selected = self._apply_region_selection_lock(frame_id, selected_raw) if _truthy(payload.get("protectRegions")) else selected_raw
        area = int(np.count_nonzero(selected))
        raw_area = int(np.count_nonzero(selected_raw))
        return {
            "frameId": int(frame_id),
            "areaPixels": area,
            "rawAreaPixels": raw_area,
            "protectedAreaPixels": int(max(0, raw_area - area)),
            "protectRegions": bool(_truthy(payload.get("protectRegions"))),
            "coverage": float(area / max(selected.size, 1)),
            "maskOverlayPng": selected_mask_overlay(selected),
        }

    def start_region_propagation_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        trigger_frame_id = int(payload.get("frameId", -1))
        self.frame(trigger_frame_id)
        if int(self.region_status().get("completeFrameCount", 0) or 0) <= 0:
            raise ValueError("No complete persistent-region keyframes exist yet. Mark at least one edited frame complete before propagating.")
        method_id = str(payload.get("methodId") or self.propagation_backends.default_method_id)
        backend = self.propagation_backends.get(method_id)
        if not backend.info.supports_full_run:
            raise ValueError(f"{backend.info.display_name} is available only through its focused region test.")
        backend_status = backend.status()
        if not backend_status["available"]:
            raise FileNotFoundError(
                f"{backend.info.display_name} is unavailable: {backend_status['availabilityMessage']}"
            )
        return self._start_propagation_job(
            payload={**payload, "methodId": backend.info.method_id},
            backend_method_id=backend.info.method_id,
            kind=str(backend.info.stage),
            total_frames=len(self.frames()),
            message=f"Starting {backend.info.display_name} {_propagation_stage_action(backend.info.stage)}",
            operation=self.propagate_regions,
        )

    def start_region_pair_test_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_frame_id = int(payload.get("frameId", -1))
        region_id = int(payload.get("regionId", 0) or 0)
        self.frame(target_frame_id)
        if region_id <= 0:
            raise ValueError("Select a persistent region before running a focused transfer test.")
        method_id = str(payload.get("methodId") or "")
        backend = self.propagation_backends.get(method_id)
        if not backend.info.supports_region_pair:
            raise ValueError(f"{backend.info.display_name} does not support focused region-pair testing.")
        backend_status = backend.status()
        if not backend_status["available"]:
            raise FileNotFoundError(
                f"{backend.info.display_name} is unavailable: {backend_status['availabilityMessage']}"
            )
        if method_id == "v2sam_pair" and self.dinov3_evidence.status(self.dinov3_frames())["running"]:
            raise RuntimeError("Wait for DINOv3 evidence generation before running V2-SAM.")
        return self._start_propagation_job(
            payload={**payload, "methodId": method_id, "regionId": region_id},
            backend_method_id=method_id,
            kind="region_pair_test",
            total_frames=1,
            message=f"Starting {backend.info.display_name} pair test for region {region_id}",
            operation=self.test_region_pair,
        )

    def _propagation_job_running(self) -> bool:
        with self._propagation_job_lock:
            return any(job.get("status") == "running" for job in self._propagation_jobs.values())

    def _start_propagation_job(
        self,
        *,
        payload: dict[str, Any],
        backend_method_id: str,
        kind: str,
        total_frames: int,
        message: str,
        operation,
    ) -> dict[str, Any]:
        backend = self.propagation_backends.get(backend_method_id)
        job_id = uuid.uuid4().hex
        now = _now()
        with self._propagation_job_lock:
            if any(existing.get("status") == "running" for existing in self._propagation_jobs.values()):
                raise ValueError("Another propagation job is already running.")
            self._propagation_jobs[job_id] = {
                "jobId": job_id,
                "kind": str(kind),
                "methodId": backend.info.method_id,
                "displayName": backend.info.display_name,
                "proposalLayer": backend.info.layer_key,
                "status": "running",
                "frameId": int(payload.get("frameId", -1)),
                "regionId": int(payload.get("regionId", 0) or 0),
                "framesDone": 0,
                "totalFrames": int(total_frames),
                "currentFrameId": None,
                "completedFrameIds": [],
                "message": str(message),
                "createdUtc": now,
                "updatedUtc": now,
            }
        thread = threading.Thread(
            target=self._run_propagation_job,
            args=(job_id, payload, operation),
            daemon=True,
        )
        thread.start()
        return self.propagation_job_status(job_id)

    def propagation_job_status(self, job_id: str) -> dict[str, Any]:
        with self._propagation_job_lock:
            job = self._propagation_jobs.get(str(job_id))
            if job is None:
                raise KeyError(job_id)
            return dict(job)

    def _run_propagation_job(self, job_id: str, payload: dict[str, Any], operation) -> None:
        def progress(done: int, total: int, current_frame_id: int) -> None:
            snapshot: dict[str, Any] | None = None
            with self._propagation_job_lock:
                job = self._propagation_jobs.get(job_id)
                if job is None:
                    return
                completed = list(job.get("completedFrameIds", []))
                if int(current_frame_id) not in completed:
                    completed.append(int(current_frame_id))
                job["framesDone"] = int(done)
                job["totalFrames"] = int(total)
                job["currentFrameId"] = int(current_frame_id)
                job["completedFrameIds"] = completed
                display_name = str(job.get("displayName") or job.get("methodId") or "Propagation")
                verb = {
                    "dense_recovery": "recovered",
                    "identity_refinement": "verified",
                }.get(str(job.get("kind")), "propagated")
                job["message"] = f"{display_name} {verb} {int(done)} / {int(total)} frames"
                job["updatedUtc"] = _now()
                snapshot = dict(job)
            if snapshot is not None:
                self.proposals.save_propagation_progress(
                    str(snapshot["methodId"]),
                    _propagation_progress_payload(snapshot),
                )

        try:
            result = operation(payload, progress_callback=progress)
        except Exception as exc:
            snapshot = None
            with self._propagation_job_lock:
                job = self._propagation_jobs.get(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["error"] = str(exc)
                    job["message"] = str(exc)
                    job["updatedUtc"] = _now()
                    snapshot = dict(job)
            if snapshot is not None:
                self.proposals.save_propagation_progress(
                    str(snapshot["methodId"]),
                    _propagation_progress_payload(snapshot),
                )
            return

        snapshot = None
        with self._propagation_job_lock:
            job = self._propagation_jobs.get(job_id)
            if job is not None:
                job["status"] = "complete"
                processed = int(result.get("processedFrameCount", result.get("frameCount", len(result.get("frames", [])))))
                job["framesDone"] = processed
                job["totalFrames"] = processed
                job["currentFrameId"] = None
                action = _propagation_stage_action(str(job.get("kind") or ""))
                job["message"] = str(
                    result.get("message")
                    or f"{result.get('displayName', 'Region')} {action} complete"
                )
                job["result"] = result
                job["updatedUtc"] = _now()
                snapshot = dict(job)
        if snapshot is not None:
            self.proposals.save_propagation_progress(
                str(snapshot["methodId"]),
                _propagation_progress_payload(snapshot),
            )

    def propagate_regions(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        trigger_frame_id = int(payload.get("frameId", -1))
        method_id = str(payload.get("methodId") or self.propagation_backends.default_method_id)
        backend = self.propagation_backends.get(method_id)
        self.frame(trigger_frame_id)
        propagation_frames = self._propagation_frames()
        region_status = self.region_status()
        run_input = build_propagation_input(
            trigger_frame_id=trigger_frame_id,
            frames=propagation_frames,
            region_status=region_status,
            region_maps_dir=self.paths.region_maps_dir,
        )
        self.proposals.clear_propagation_output(backend.info.method_id)
        layer = backend.info.layer_key
        self.proposals.save_propagation_config(
            backend.info.method_id,
            config={
                "schemaVersion": 1,
                "methodId": backend.info.method_id,
                "displayName": backend.info.display_name,
                "engineName": backend.info.engine_name,
                "description": backend.info.description,
                "stage": backend.info.stage,
                "sourceMethodId": backend.info.source_method_id,
                "sourceLayerKey": (
                    f"propagation_{backend.info.source_method_id}"
                    if backend.info.source_method_id
                    else ""
                ),
                "proposalLayer": layer,
                "labelSpace": "persistent_region",
                "backend": dict(backend.info.runtime_config),
                "outputResolution": {
                    "width": int(propagation_frames[0].width),
                    "height": int(propagation_frames[0].height),
                },
                "inputFingerprint": run_input.fingerprint,
                "updatedUtc": _now(),
            },
            run_input=run_input.to_json(),
        )
        self.proposals.save_propagation_progress(
            backend.info.method_id,
            {
                "schemaVersion": 1,
                "jobId": "",
                "methodId": backend.info.method_id,
                "displayName": backend.info.display_name,
                "proposalLayer": layer,
                "status": "running",
                "framesDone": 0,
                "totalFrames": int(len(propagation_frames)),
                "completedFrameIds": [],
                "inputFingerprint": run_input.fingerprint,
                "updatedUtc": _now(),
            },
        )
        source_frame_ids_by_region = {
            region_id: list(frame_ids)
            for region_id, frame_ids in run_input.source_frame_ids_by_region.items()
        }
        region_rows = [dict(row) for row in run_input.region_rows]

        def save_frame(frame: PropagationFrame, labels: np.ndarray) -> None:
            validated = run_input.validate_output(frame, labels)
            self.proposals.save_region_candidate_map(
                self.proposal_frames(),
                int(frame.frame_id),
                validated,
                method_id=backend.info.method_id,
                region_rows=region_rows,
                source_frame_ids_by_region=source_frame_ids_by_region,
                result_description=backend.info.result_description,
                input_fingerprint=run_input.fingerprint,
                stage=backend.info.stage,
            )

        backend_rows = backend.run(
            run_input,
            progress_callback=progress_callback,
            save_callback=save_frame,
        )
        self.proposals.save_propagation_diagnostics(
            backend.info.method_id,
            [dict(row) for row in backend_rows],
        )
        output_paths = self.proposals.propagation_paths(backend.info.method_id)
        missing_frame_ids = [
            int(frame.frame_id)
            for frame in propagation_frames
            if not (output_paths.label_map_dir / f"{int(frame.frame_id):06d}.npy").exists()
        ]
        if missing_frame_ids:
            raise RuntimeError(
                f"{backend.info.display_name} did not produce candidate maps for frames: {missing_frame_ids[:12]}"
            )
        frame_by_id = {int(frame.frame_id): frame for frame in propagation_frames}
        anchor_frame_ids = set(run_input.anchor_frame_ids)
        rows = self.proposals.propagation_frame_summaries(backend.info.method_id)
        diagnostics_by_frame = {
            int(item.get("frameId", -1)): dict(item)
            for item in backend_rows
            if int(item.get("frameId", -1)) >= 0
        }
        for row in rows:
            frame_id = int(row.get("frameId", -1))
            frame = frame_by_id.get(frame_id)
            pixel_count = int(frame.width * frame.height) if frame is not None else 0
            row["areaPixels"] = int(round(float(row.get("coverage", 0.0)) * pixel_count))
            row["regionCount"] = int(row.get("keptMaskCount", 0))
            row["isAnchor"] = bool(frame_id in anchor_frame_ids)
            row["isSource"] = bool(frame_id in anchor_frame_ids)
            if frame_id in diagnostics_by_frame:
                row["backendDiagnostics"] = diagnostics_by_frame[frame_id]

        regions = []
        for row in region_rows:
            rid = int(row.get("id", 0))
            if rid <= 0:
                continue
            item = dict(row)
            item["sourceAreaPixels"] = int(run_input.source_area_pixels.get(rid, 0))
            item["sourceFrameIds"] = list(source_frame_ids_by_region.get(rid, []))
            regions.append(item)

        return {
            "schemaVersion": 1,
            "methodId": backend.info.method_id,
            "displayName": backend.info.display_name,
            "engineName": backend.info.engine_name,
            "stage": backend.info.stage,
            "sourceMethodId": backend.info.source_method_id,
            "sourceLayerKey": (
                f"propagation_{backend.info.source_method_id}"
                if backend.info.source_method_id
                else ""
            ),
            "proposalLayer": layer,
            "primaryPolicy": (
                "saved_sparse_source"
                if backend.info.stage == "dense_recovery"
                else (
                    "saved_mask_source"
                    if backend.info.stage == "identity_refinement"
                    else "sequence_first_frame"
                )
            ),
            "frameId": int(trigger_frame_id),
            "triggerFrameId": int(trigger_frame_id),
            "primaryFrameId": int(run_input.primary_frame_id),
            "startFrameId": int(run_input.primary_frame_id),
            "frameCount": int(len(propagation_frames)),
            "anchorFrameCount": int(len(run_input.anchor_frame_ids)),
            "anchorFrameIds": run_input.anchor_frame_ids,
            "sourceRegionCount": int(len(run_input.source_region_ids)),
            "sourceAreaPixels": int(run_input.source_area_total),
            "inputFingerprint": run_input.fingerprint,
            "regions": regions,
            "frames": rows,
        }

    def test_region_pair(self, payload: dict[str, Any], progress_callback=None) -> dict[str, Any]:
        target_frame_id = int(payload.get("frameId", -1))
        region_id = int(payload.get("regionId", 0) or 0)
        backend = self.propagation_backends.get(str(payload.get("methodId") or ""))
        output_policy = str(backend.info.pair_output_policy)
        if not output_policy:
            raise ValueError(f"{backend.info.display_name} has no focused output policy configured.")
        propagation_frames = self._propagation_frames()
        pair_input = build_region_pair_test_input(
            method_id=backend.info.method_id,
            output_policy=output_policy,
            target_frame_id=target_frame_id,
            region_id=region_id,
            frames=propagation_frames,
            region_status=self.region_status(),
            region_maps_dir=self.paths.region_maps_dir,
        )
        layer = backend.info.layer_key
        region_rows = [dict(pair_input.region_row)]
        self.proposals.save_propagation_config(
            backend.info.method_id,
            config={
                "schemaVersion": 1,
                "methodId": backend.info.method_id,
                "displayName": backend.info.display_name,
                "engineName": backend.info.engine_name,
                "description": backend.info.description,
                "mode": "single_region_pair",
                "sourcePolicy": "first_manual_occurrence",
                "outputPolicy": output_policy,
                "proposalLayer": layer,
                "labelSpace": "persistent_region",
                "backend": dict(backend.info.runtime_config),
                "outputResolution": {
                    "width": int(pair_input.target_frame.width),
                    "height": int(pair_input.target_frame.height),
                },
                "inputFingerprint": pair_input.fingerprint,
                "updatedUtc": _now(),
            },
            run_input=pair_input.to_json(),
        )

        saved: dict[str, Any] = {}

        def save_target(frame: PropagationFrame, labels: np.ndarray) -> None:
            if int(frame.frame_id) != pair_input.target_frame_id:
                return
            values = np.asarray(labels)
            expected = (int(frame.height), int(frame.width))
            if values.ndim != 2 or values.shape != expected:
                raise ValueError(
                    f"{backend.info.display_name} pair result for frame {frame.frame_id} has shape "
                    f"{values.shape}; expected {expected}."
                )
            isolated = np.zeros(expected, dtype=np.uint16)
            isolated[values == pair_input.region_id] = np.uint16(pair_input.region_id)
            saved.update(
                self.proposals.save_region_candidate_map(
                    self.proposal_frames(),
                    int(frame.frame_id),
                    isolated,
                    method_id=backend.info.method_id,
                    region_rows=region_rows,
                    source_frame_ids_by_region={pair_input.region_id: [pair_input.source_frame_id]},
                    result_description=backend.info.result_description,
                    input_fingerprint=pair_input.fingerprint,
                    stage=backend.info.stage,
                )
            )

        result = backend.run_region_pair(
            frames=propagation_frames,
            source=pair_input.source,
            target_local_index=pair_input.target_local_index,
            region_id=pair_input.region_id,
            region_rows=region_rows,
            progress_callback=progress_callback,
            save_callback=save_target,
        )
        if not saved:
            raise RuntimeError(f"{backend.info.display_name} pair testing did not save a target-frame result.")
        name = str(pair_input.region_row.get("name") or f"region_{pair_input.region_id}")
        return {
            "schemaVersion": 1,
            "methodId": backend.info.method_id,
            "displayName": backend.info.display_name,
            "engineName": backend.info.engine_name,
            "proposalLayer": layer,
            "mode": "single_region_pair",
            "sourcePolicy": "first_manual_occurrence",
            "outputPolicy": output_policy,
            "frameId": pair_input.target_frame_id,
            "targetFrameId": pair_input.target_frame_id,
            "sourceFrameId": pair_input.source_frame_id,
            "regionId": pair_input.region_id,
            "regionName": name,
            "processedFrameCount": 1,
            "frameCount": 1,
            "inputFingerprint": pair_input.fingerprint,
            "areaPixels": int(result.get("areaPixels", 0)),
            "coverage": float(result.get("coverage", 0.0)),
            "frames": [saved],
            "message": (
                f"{backend.info.display_name} tested {name}: source frame {pair_input.source_frame_id} "
                f"to target frame {pair_input.target_frame_id}"
            ),
        }

    def _propagation_frames(self) -> list[PropagationFrame]:
        return [
            PropagationFrame(
                frame_id=int(frame["id"]),
                image_name=str(frame["imageName"]),
                image_path=self.image_path(int(frame["id"])),
                width=int(frame["width"]),
                height=int(frame["height"]),
                fx=float(frame["fx"]),
                fy=float(frame["fy"]),
                cx=float(frame["cx"]),
                cy=float(frame["cy"]),
                pose_world_from_camera=np.asarray(
                    frame["poseWorldFromCamera"], dtype=np.float64
                ).reshape(4, 4),
            )
            for frame in self.frames()
        ]

    def create_region_from_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._commit_region_selection(payload, mode="new")

    def assign_selection_to_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._commit_region_selection(payload, mode="replace")

    def add_selection_to_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._commit_region_selection(payload, mode="add")

    def clear_selection_from_regions(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._commit_region_selection(payload, mode="clear")

    def clear_region_from_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_id = int(payload.get("frameId", -1))
        if frame_id < 0:
            raise ValueError("Frame clear needs a non-negative frame ID.")
        self.frame(frame_id)
        region_id = int(payload.get("regionId", 0) or 0)
        if region_id <= 0:
            raise ValueError("Frame clear needs a selected persistent region.")
        return self.regions.clear_region_from_frame(
            frame_id=frame_id,
            region_id=region_id,
        )

    def rename_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.regions.rename_region(
            region_id=int(payload.get("regionId", 0) or 0),
            name=str(payload.get("name", "") or ""),
        )

    def delete_region(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.regions.delete_region(region_id=int(payload.get("regionId", 0) or 0))

    def _commit_region_selection(self, payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
        frame_id = int(payload.get("frameId", -1))
        frame = self.frame(frame_id)
        base_labels = self._empty_proposal_label_map(frame_id)
        selected = self._compose_proposal_selection(frame_id, payload, base_labels)
        expected_shape = (int(frame["height"]), int(frame["width"]))
        if base_labels.shape != expected_shape:
            raise ValueError(f"Selection base shape {base_labels.shape} does not match frame shape {expected_shape}.")
        return self.regions.commit_selection(
            frame_id=frame_id,
            frame_shape=expected_shape,
            selected=selected,
            region_id=int(payload.get("regionId", 0) or 0),
            name=str(payload.get("name", "") or ""),
            mode=mode,
        )

    def _compose_proposal_selection(
        self,
        frame_id: int,
        payload: dict[str, Any],
        base_labels: np.ndarray | None = None,
        *,
        protect_regions: bool | None = None,
    ) -> np.ndarray:
        labels = base_labels if base_labels is not None else self.proposal_label_map(frame_id)

        def resolver(layer: str) -> np.ndarray:
            return self.proposal_layer_label_map(frame_id, normalize_proposal_layer(layer))

        selected = compose_selection_mask(labels, payload, label_map_resolver=resolver)
        should_protect = _truthy(payload.get("protectRegions")) if protect_regions is None else bool(protect_regions)
        if should_protect:
            selected = self._apply_region_selection_lock(frame_id, selected)
        return selected

    def _apply_region_selection_lock(self, frame_id: int, selected: np.ndarray) -> np.ndarray:
        selected = np.asarray(selected, dtype=bool)
        path = self.regions.region_map_path(frame_id)
        if not path.exists():
            return selected
        region_labels = np.load(path).astype(np.uint16, copy=False)
        if region_labels.shape != selected.shape:
            raise ValueError(
                f"Persistent region map shape {region_labels.shape} does not match selected mask shape {selected.shape}."
            )
        return selected & ~(region_labels > 0)

    def _empty_proposal_label_map(self, frame_id: int) -> np.ndarray:
        frame = self.frame(frame_id)
        return np.zeros((int(frame["height"]), int(frame["width"])), dtype=np.uint16)

    def _proposal_details(self, frame_id: int, layer: str = "sam2") -> dict[int, dict[str, Any]]:
        try:
            summary = self.proposal_layer_frame_summary(frame_id, layer)
        except FileNotFoundError:
            return {}
        details: dict[int, dict[str, Any]] = {}
        for row in summary.get("labels", []):
            if not isinstance(row, dict):
                continue
            label = int(row.get("labelId", 0) or 0)
            if label > 0:
                details[label] = row
        return details

    def points_full(self) -> np.ndarray:
        if self._points_full is None:
            points = np.loadtxt(self.paths.points_path, dtype=np.float32)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            points = points[:, :3].astype(np.float32, copy=False)
            if points.shape[0] == 0:
                raise ValueError(f"No points found in {self.paths.points_path}")
            self._points_full = points
        return self._points_full

    def labels_full(self) -> np.ndarray:
        if self._labels_full is None:
            point_count = int(self.points_full().shape[0])
            source = self.label_source
            if source == "auto":
                if self.paths.point_labels is not None and self.paths.point_labels.exists():
                    source = "sai3d"
                else:
                    source = "raw"

            label_path: Path | None = None
            if source == "raw":
                labels = np.zeros(point_count, dtype=np.int32)
                self._active_label_source = "raw"
            elif source == "sai3d":
                label_path = self.paths.point_labels
                if label_path is None or not label_path.exists():
                    raise FileNotFoundError(f"SAI3D point labels do not exist: {label_path}")
                labels = np.load(label_path).astype(np.int32, copy=False)
                self._active_label_source = "sai3d"
            else:
                raise ValueError(f"Unknown label source: {self.label_source}")

            labels = labels.reshape(-1).astype(np.int32, copy=False)
            if labels.shape[0] < point_count:
                labels = np.pad(labels, (0, point_count - labels.shape[0]), mode="constant", constant_values=0)
            elif labels.shape[0] > point_count:
                labels = labels[:point_count]
            self._labels_full = labels
        return self._labels_full

    def label_summary(self) -> dict[str, Any]:
        if self._label_summary is not None:
            return self._label_summary
        labels = self.labels_full()
        unique, counts = np.unique(labels, return_counts=True)
        rows = [
            {"label": int(label), "count": int(count)}
            for label, count in zip(unique.tolist(), counts.tolist(), strict=True)
            if int(label) > 0
        ]
        rows.sort(key=lambda row: int(row["label"]))
        self._label_summary = {
            "pointCount": int(labels.shape[0]),
            "labelCount": int(len(rows)),
            "labels": rows,
        }
        return self._label_summary

    def points_payload(self) -> dict[str, Any]:
        if self._points_payload is not None:
            return self._points_payload

        points = self.points_full()
        labels = self.labels_full()
        count = min(points.shape[0], labels.shape[0])
        points = points[:count]
        labels = labels[:count]

        if count > self.max_points:
            rng = np.random.default_rng(self.seed)
            keep = rng.choice(count, size=self.max_points, replace=False)
            keep.sort()
            served_points = points[keep]
            served_labels = labels[keep]
            served_indices = keep.astype(np.int64, copy=False)
        else:
            served_points = points
            served_labels = labels
            served_indices = np.arange(count, dtype=np.int64)

        bounds_min = np.min(points, axis=0)
        bounds_max = np.max(points, axis=0)
        unique_labels = np.unique(labels)
        positive = unique_labels[unique_labels > 0]
        self._points_payload = {
            "pointCount": int(count),
            "servedPointCount": int(served_points.shape[0]),
            "labelCount": int(positive.shape[0]),
            "bounds": {
                "min": bounds_min.astype(float).tolist(),
                "max": bounds_max.astype(float).tolist(),
                "center": ((bounds_min + bounds_max) * 0.5).astype(float).tolist(),
                "radius": float(np.linalg.norm(bounds_max - bounds_min) * 0.5),
            },
            "indices": served_indices.astype(int).tolist(),
            "positions": served_points.reshape(-1).astype(float).tolist(),
            "labels": served_labels.astype(int).tolist(),
        }
        return self._points_payload

    def predict_sam2(self, payload: dict[str, Any]) -> dict[str, Any]:
        frame_id = int(payload.get("frameId", -1))
        self.frame(frame_id)
        prompts = payload.get("prompts", [])
        if not isinstance(prompts, list):
            raise ValueError("prompts must be a list.")

        coords: list[list[float]] = []
        labels: list[int] = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                continue
            label = 1 if int(prompt.get("label", 1)) > 0 else 0
            if "sourceX" in prompt and "sourceY" in prompt:
                x = float(prompt["sourceX"])
                y = float(prompt["sourceY"])
            else:
                x = float(prompt.get("x", np.nan))
                y = float(prompt.get("y", np.nan))
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            coords.append([x, y])
            labels.append(label)

        if not coords:
            raise ValueError("SAM2 needs at least one valid prompt.")
        if not any(label > 0 for label in labels):
            raise ValueError("SAM2 needs at least one positive prompt.")

        image_rgb = self.image_rgb(frame_id)
        height, width = image_rgb.shape[:2]
        points_xy = np.asarray(coords, dtype=np.float32)
        points_xy[:, 0] = np.clip(points_xy[:, 0], 0, max(width - 1, 0))
        points_xy[:, 1] = np.clip(points_xy[:, 1], 0, max(height - 1, 0))
        point_labels = np.asarray(labels, dtype=np.int32)
        mask, score = self.sam2_session.predict(
            frame_id=frame_id,
            image_rgb=image_rgb,
            points_xy=points_xy,
            point_labels=point_labels,
            multimask=bool(payload.get("multimask", True)),
            mask_threshold=float(payload.get("maskThreshold", 0.0)),
        )
        if mask.shape != (height, width):
            raise ValueError(f"SAM2 returned unexpected mask shape: {mask.shape}, expected {(height, width)}.")

        ys, xs = np.nonzero(mask)
        if xs.size:
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        else:
            bbox = [0, 0, 0, 0]
        area = int(np.count_nonzero(mask))
        return {
            "frameId": int(frame_id),
            "width": int(width),
            "height": int(height),
            "score": float(score),
            "area": area,
            "coverage": float(area / max(mask.size, 1)),
            "bbox": bbox,
            "promptCount": int(points_xy.shape[0]),
            "positivePromptCount": int(np.count_nonzero(point_labels > 0)),
            "negativePromptCount": int(np.count_nonzero(point_labels == 0)),
            "maskPng": encode_mask_png(mask),
            "maskOverlayPng": encode_mask_overlay(mask),
        }

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _propagation_stage_action(stage: str) -> str:
    return {
        "dense_recovery": "dense recovery",
        "identity_refinement": "identity refinement",
    }.get(str(stage), "region propagation")


def _propagation_progress_payload(job: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "jobId",
        "kind",
        "methodId",
        "displayName",
        "proposalLayer",
        "status",
        "frameId",
        "framesDone",
        "totalFrames",
        "currentFrameId",
        "completedFrameIds",
        "message",
        "error",
        "createdUtc",
        "updatedUtc",
    )
    payload = {key: job[key] for key in keys if key in job}
    result = job.get("result")
    if isinstance(result, dict):
        payload["inputFingerprint"] = str(result.get("inputFingerprint") or "")
        payload["anchorFrameCount"] = int(result.get("anchorFrameCount", 0) or 0)
        payload["sourceRegionCount"] = int(result.get("sourceRegionCount", 0) or 0)
    return {"schemaVersion": 1, **payload}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _payload_layers(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        layer = normalize_proposal_layer(str(item))
        if layer in seen:
            continue
        seen.add(layer)
        out.append(layer)
    return out
