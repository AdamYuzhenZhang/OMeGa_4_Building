"""Mutable data model behind the interactive segmentation editor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .paths import EditorPaths, read_jsonl
from .sam2_session import Sam2Config, Sam2Session, encode_mask_overlay


class EditorState:
    def __init__(self, paths: EditorPaths, *, max_points: int, seed: int, sam2_config: Sam2Config) -> None:
        self.paths = paths
        self.max_points = max(int(max_points), 1)
        self.seed = int(seed)
        self.sam2_config = sam2_config
        self.sam2_session = Sam2Session(sam2_config)
        self._manifest_rows: list[dict[str, Any]] | None = None
        self._manifest_by_frame_id: dict[int, dict[str, Any]] | None = None
        self._labels_full: np.ndarray | None = None
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

    def labels_full(self) -> np.ndarray:
        if self._labels_full is None:
            label_path = self.paths.interactive_labels if self.paths.interactive_labels.exists() else self.paths.point_labels
            self._labels_full = np.load(label_path).astype(np.int32, copy=False)
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

        points = np.loadtxt(self.paths.points_path, dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        points = points[:, :3].astype(np.float32, copy=False)
        labels = self.labels_full()
        count = min(points.shape[0], labels.shape[0])
        points = points[:count]
        labels = labels[:count]
        if count == 0:
            raise ValueError(f"No points found in {self.paths.points_path}")

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
