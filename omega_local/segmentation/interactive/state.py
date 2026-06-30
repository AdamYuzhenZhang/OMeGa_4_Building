"""Mutable data model behind the interactive segmentation editor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .keyframes import KeyframeConfig, KeyframeFrame, detect_keyframes, read_keyframes
from .mask_edits import compose_selection_mask, selected_mask_overlay
from .paths import EditorPaths, read_jsonl
from .proposals import ProposalFrame, ProposalManager
from .sam2_session import Sam2Config, Sam2Session, encode_mask_overlay, encode_mask_png
from .sam2_video_propagation import PropagationFrame, Sam2VideoPropagationSession
from .view_evidence import ViewEvidenceFrame, ViewEvidenceManager


class EditorState:
    def __init__(
        self,
        paths: EditorPaths,
        *,
        max_points: int,
        seed: int,
        label_source: str,
        sam2_config: Sam2Config,
    ) -> None:
        self.paths = paths
        self.max_points = max(int(max_points), 1)
        self.seed = int(seed)
        self.label_source = str(label_source)
        self.sam2_config = sam2_config
        self.sam2_session = Sam2Session(sam2_config)
        self.sam2_video_propagation = Sam2VideoPropagationSession(
            sam2_config,
            work_root=paths.interactive_dir / "tmp",
        )
        self.proposals = ProposalManager(paths, sam2_config)
        self.view_evidence = ViewEvidenceManager(paths)
        self._manifest_rows: list[dict[str, Any]] | None = None
        self._manifest_by_frame_id: dict[int, dict[str, Any]] | None = None
        self._points_full: np.ndarray | None = None
        self._labels_full: np.ndarray | None = None
        self._active_label_source: str | None = None
        self._label_summary: dict[str, Any] | None = None
        self._points_payload: dict[str, Any] | None = None
        self._frames: list[dict[str, Any]] | None = None
        self._frame_by_id: dict[int, dict[str, Any]] | None = None

    def project_summary(self) -> dict[str, Any]:
        frames = self.frames()
        points = self.points_payload()
        return {
            "modelDir": str(self.paths.model_dir),
            "baselineName": self.paths.baseline_name,
            "baselineDir": str(self.paths.baseline_dir),
            "frameCount": len(frames),
            "pointCount": points["pointCount"],
            "servedPointCount": points["servedPointCount"],
            "labelCount": points["labelCount"],
            "bounds": points["bounds"],
            "pointsPath": str(self.paths.points_path),
            "labelSource": self._active_label_source or self.label_source,
            "sai3dLabelsPath": str(self.paths.point_labels) if self.paths.point_labels is not None else "",
            "interactiveLabelsPath": str(self.paths.interactive_labels),
            "hasInteractiveLabels": self.paths.interactive_labels.exists(),
            "sam2": {
                "root": str(self.sam2_config.root),
                "checkpoint": str(self.sam2_config.checkpoint),
                "config": str(self.sam2_config.config),
                "device": str(self.sam2_config.device),
            },
        }

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
        for paths in [
            self.proposals.editable_paths(frames, run_name),
            self.proposals.run_paths(run_name),
        ]:
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
        if status.get("ready"):
            try:
                edit_paths = self.proposals.ensure_editable_copy(frames)
                status["editableRunName"] = edit_paths.run_dir.name
                status["editableRunDir"] = str(edit_paths.run_dir)
                status["editingRepresentation"] = "single integer label map per frame"
                if edit_paths.summary.exists():
                    try:
                        edit_summary = json.loads(edit_paths.summary.read_text(encoding="utf-8"))
                    except Exception:
                        edit_summary = {}
                    editable_updated = str(edit_summary.get("timestampUtc") or "")
                    if editable_updated:
                        status["editableUpdatedUtc"] = editable_updated
                        status["overlayUpdatedUtc"] = editable_updated
            except FileNotFoundError:
                pass
        return status

    def start_proposal_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.proposals.start(self.proposal_frames(), payload)

    def view_evidence_status(self) -> dict[str, Any]:
        return self.view_evidence.status(self.view_evidence_frames())

    def start_view_evidence_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.view_evidence.start(self.view_evidence_frames(), payload)

    def view_evidence_image_path(self, frame_id: int, kind: str) -> Path:
        self.frame(frame_id)
        return self.view_evidence.image_path(self.view_evidence_frames(), int(frame_id), str(kind))

    def proposal_overlay_path(self, frame_id: int) -> Path:
        self.frame(frame_id)
        path = self.proposals.overlay_path(self.proposal_frames(), int(frame_id))
        if not path.exists():
            raise FileNotFoundError(f"SAM2 proposal overlay does not exist for frame {frame_id}: {path}")
        return path

    def proposal_label_map(self, frame_id: int) -> np.ndarray:
        self.frame(frame_id)
        path = self.proposals.label_map_path(self.proposal_frames(), int(frame_id))
        if not path.exists():
            raise FileNotFoundError(f"SAM2 proposal label map does not exist for frame {frame_id}: {path}")
        labels = np.load(path)
        if labels.ndim != 2:
            raise ValueError(f"Expected 2D proposal label map, got shape {labels.shape}: {path}")
        return labels

    def proposal_frame_summary(self, frame_id: int) -> dict[str, Any]:
        self.frame(frame_id)
        path = self.proposals.metadata_path(self.proposal_frames(), int(frame_id))
        if not path.exists():
            raise FileNotFoundError(f"SAM2 proposal metadata does not exist for frame {frame_id}: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def pick_proposal(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        labels = self.proposal_label_map(frame_id)
        x = int(round(float(payload.get("x", -1))))
        y = int(round(float(payload.get("y", -1))))
        if x < 0 or x >= labels.shape[1] or y < 0 or y >= labels.shape[0]:
            return {"frameId": int(frame_id), "labelId": 0, "inside": False}
        label = int(labels[y, x])
        details = self._proposal_details(frame_id).get(label, {}) if label > 0 else {}
        return {
            "frameId": int(frame_id),
            "labelId": label,
            "inside": True,
            "details": details,
        }

    def preview_selection(self, frame_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        labels = self.proposal_label_map(frame_id)
        selected = compose_selection_mask(labels, payload)
        area = int(np.count_nonzero(selected))
        return {
            "frameId": int(frame_id),
            "areaPixels": area,
            "coverage": float(area / max(selected.size, 1)),
            "maskOverlayPng": selected_mask_overlay(selected),
        }

    def propagate_selection(self, payload: dict[str, Any]) -> dict[str, Any]:
        source_frame_id = int(payload.get("frameId", -1))
        self.frame(source_frame_id)
        labels = self.proposal_label_map(source_frame_id)
        selected = compose_selection_mask(labels, payload)
        selected_area = int(np.count_nonzero(selected))
        if selected_area <= 0:
            raise ValueError("Propagation source selection is empty.")

        neighbor_count = int(payload.get("neighborCount", 2))
        neighbor_count = max(1, min(neighbor_count, 8))
        frames = self.frames()
        source_index = next((idx for idx, frame in enumerate(frames) if int(frame["id"]) == source_frame_id), -1)
        if source_index < 0:
            raise KeyError(source_frame_id)

        start = max(0, source_index - neighbor_count)
        end = min(len(frames), source_index + neighbor_count + 1)
        sequence = frames[start:end]
        propagation_frames = [
            PropagationFrame(
                frame_id=int(frame["id"]),
                image_path=self.image_path(int(frame["id"])),
                width=int(frame["width"]),
                height=int(frame["height"]),
            )
            for frame in sequence
        ]
        rows = self.sam2_video_propagation.propagate(
            frames=propagation_frames,
            source_local_index=source_index - start,
            source_mask=selected,
            max_neighbors=neighbor_count,
        )
        return {
            "method": "sam2_video_mask_prompt",
            "frameId": int(source_frame_id),
            "neighborCount": int(neighbor_count),
            "sourceAreaPixels": int(selected_area),
            "sourceCoverage": float(selected_area / max(selected.size, 1)),
            "frames": rows,
        }

    def save_proposal_edits(self, payload: dict[str, Any]) -> dict[str, Any]:
        edits = payload.get("edits", [])
        if not isinstance(edits, list):
            raise ValueError("edits must be a list.")

        by_frame: dict[int, list[dict[str, Any]]] = {}
        for item in edits:
            if not isinstance(item, dict):
                continue
            frame_id = int(item.get("frameId", -1))
            if frame_id < 0:
                continue
            by_frame.setdefault(frame_id, []).append(item)

        saved_frames: list[dict[str, Any]] = []
        for frame_id, frame_edits in sorted(by_frame.items()):
            labels = self.proposal_label_map(frame_id)
            records: list[dict[str, Any]] = []
            for edit in frame_edits:
                labels, target_id, selected, info = self._apply_proposal_label_edit(labels, edit)
                records.append(
                    {
                        "targetLabelId": int(target_id),
                        "areaPixels": int(np.count_nonzero(selected)),
                        "selectionOpCount": int(len(edit.get("selectionOps", []))) if isinstance(edit.get("selectionOps", []), list) else 0,
                        "createdNewTarget": bool(info.get("createdNewTarget", False)),
                        "completeSourceLabelIds": info.get("completeSourceLabelIds", []),
                        "partialSourceLabelIds": info.get("partialSourceLabelIds", []),
                    }
                )
            saved = self.proposals.save_label_map(self.proposal_frames(), frame_id, labels, edit_records=records)
            saved_frames.append(
                {
                    "frameId": int(frame_id),
                    "labelCount": int(len(saved.get("labels", []))),
                    "coverage": float(saved.get("coverage", 0.0)),
                    "metadata": str(saved.get("metadata", "")),
                    "editRecords": records,
                }
            )
        return {
            "saved": True,
            "frameCount": int(len(saved_frames)),
            "frames": saved_frames,
        }

    def _apply_proposal_label_edit(self, labels: np.ndarray, payload: dict[str, Any]) -> tuple[np.ndarray, int, np.ndarray, dict[str, Any]]:
        selected = compose_selection_mask(labels, payload)
        if not np.any(selected):
            raise ValueError("Updated proposal selection is empty.")

        preferred_target_id = int(
            payload.get("preferredTargetLabelId", payload.get("targetLabelId", payload.get("maskId", 0))) or 0
        )
        target_id, info = self._proposal_edit_target_id(labels, selected, preferred_target_id)
        updated = np.asarray(labels).copy()
        updated[selected] = np.asarray(target_id, dtype=updated.dtype)
        return updated, target_id, selected, info

    def _proposal_edit_target_id(self, labels: np.ndarray, selected: np.ndarray, preferred_target_id: int = 0) -> tuple[int, dict[str, Any]]:
        selected_labels = np.unique(labels[selected])
        selected_labels = selected_labels[selected_labels > 0].astype(np.int64, copy=False)

        complete_ids: list[int] = []
        partial_ids: list[int] = []
        for label in selected_labels.tolist():
            region = labels == int(label)
            if np.all(selected[region]):
                complete_ids.append(int(label))
            else:
                partial_ids.append(int(label))

        if preferred_target_id > 0 and preferred_target_id in complete_ids:
            target_id = int(preferred_target_id)
            created_new = False
        elif complete_ids:
            target_id = int(sorted(complete_ids)[0])
            created_new = False
        else:
            target_id = int(np.max(labels)) + 1
            created_new = True
            if target_id > np.iinfo(np.uint16).max:
                raise ValueError("Cannot allocate a new proposal ID; uint16 label map is full.")

        return target_id, {
            "createdNewTarget": created_new,
            "preferredTargetLabelId": int(preferred_target_id),
            "completeSourceLabelIds": sorted(complete_ids),
            "partialSourceLabelIds": sorted(partial_ids),
        }

    def _proposal_details(self, frame_id: int) -> dict[int, dict[str, Any]]:
        try:
            summary = self.proposal_frame_summary(frame_id)
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
                if self.paths.interactive_labels.exists():
                    source = "saved"
                elif self.paths.point_labels is not None and self.paths.point_labels.exists():
                    source = "sai3d"
                else:
                    source = "raw"

            label_path: Path | None = None
            if source == "raw":
                labels = np.zeros(point_count, dtype=np.int32)
                self._active_label_source = "raw"
            elif source == "saved":
                label_path = self.paths.interactive_labels
                if not label_path.exists():
                    raise FileNotFoundError(f"Saved interactive labels do not exist: {label_path}")
                labels = np.load(label_path).astype(np.int32, copy=False)
                self._active_label_source = "saved"
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

    def save_interactive_labels(self, payload: dict[str, Any]) -> dict[str, Any]:
        edits = payload.get("edits", [])
        if not isinstance(edits, list):
            raise ValueError("edits must be a list.")
        labels = self.labels_full().astype(np.int32, copy=True)
        applied: list[dict[str, Any]] = []

        for edit in edits:
            if not isinstance(edit, dict):
                continue
            kind = str(edit.get("type", ""))
            if kind == "assign_points":
                target = int(edit.get("label", 0))
                if target <= 0:
                    continue
                indices = np.asarray(edit.get("indices", []), dtype=np.int64)
                if indices.size == 0:
                    continue
                valid = indices[(indices >= 0) & (indices < labels.shape[0])]
                if valid.size == 0:
                    continue
                labels[valid] = np.int32(target)
                applied.append(
                    {
                        "type": "assign_points",
                        "label": int(target),
                        "count": int(valid.size),
                    }
                )
            elif kind == "merge_labels":
                source_labels = np.asarray(edit.get("labels", []), dtype=np.int32)
                source_labels = source_labels[source_labels > 0]
                target = int(edit.get("target", 0))
                if target <= 0 or source_labels.size == 0:
                    continue
                mask = np.isin(labels, source_labels)
                count = int(np.count_nonzero(mask))
                if count == 0:
                    continue
                labels[mask] = np.int32(target)
                applied.append(
                    {
                        "type": "merge_labels",
                        "labels": sorted({int(value) for value in source_labels.tolist()}),
                        "target": int(target),
                        "count": count,
                    }
                )

        if not applied:
            return {
                "saved": False,
                "message": "No valid edits to save.",
                "appliedEditCount": 0,
                "labelsPath": str(self.paths.interactive_labels),
            }

        self.paths.interactive_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.paths.interactive_labels, labels)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.paths.interactive_edits.open("a", encoding="utf-8") as handle:
            for record in applied:
                handle.write(json.dumps({"timestamp": timestamp, **record}, separators=(",", ":")) + "\n")

        self._labels_full = labels
        self._label_summary = None
        self._points_payload = None
        unique_labels = np.unique(labels)
        positive = unique_labels[unique_labels > 0]
        return {
            "saved": True,
            "appliedEditCount": int(len(applied)),
            "labelsPath": str(self.paths.interactive_labels),
            "editLogPath": str(self.paths.interactive_edits),
            "labelCount": int(positive.shape[0]),
            "pointCount": int(labels.shape[0]),
        }
