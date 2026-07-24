"""Standalone job and status access for V2-SAM-compatible DINOv3 evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .v2sam_cache import DinoFeatureCache, DinoFrame


@dataclass(frozen=True)
class DinoV3EvidenceConfig:
    root: Path
    repo_root: Path
    python: Path
    checkpoint: Path
    omega_root: Path
    device: str = "auto"
    image_size: int = 768
    patch_size: int = 16


class DinoV3EvidenceManager:
    def __init__(self, config: DinoV3EvidenceConfig) -> None:
        self.config = config
        self.root = Path(config.root)
        self.features_dir = self.root / "features"
        self.pca_dir = self.root / "pca"
        self.progress_path = self.root / "progress.json"
        self.summary_path = self.root / "summary.json"
        self._lock = threading.Lock()
        self._job: dict[str, Any] | None = None

    def status(self, frames: list[DinoFrame]) -> dict[str, Any]:
        expected = [int(frame.frame_id) for frame in frames]
        feature_count = sum(self._feature_path(frame_id).is_file() for frame_id in expected)
        pca_count = sum(self._pca_path(frame_id).is_file() for frame_id in expected)
        progress = _read_json(self.progress_path)
        summary = _read_json(self.summary_path)
        with self._lock:
            active_job = dict(self._job) if self._job is not None else None

        try:
            cache = DinoFeatureCache(
                self.root,
                model=None,
                device="cpu",
                checkpoint=self.config.checkpoint,
                image_size=int(self.config.image_size),
                patch_size=int(self.config.patch_size),
            )
            ready = cache.ready(frames)
        except OSError:
            ready = False
        running = bool(active_job and active_job.get("running"))
        failed = bool(active_job and active_job.get("failed"))
        if ready:
            message = "DINOv3 features and PCA views are ready."
        elif running:
            message = f"Extracting DINOv3 features: {feature_count}/{len(expected)} frames."
        elif failed:
            message = str(active_job.get("message") or "DINOv3 evidence generation failed.")
        elif progress.get("status") == "running":
            message = f"Previous DINOv3 run was interrupted; resume to finish {feature_count}/{len(expected)} features."
        elif feature_count:
            message = f"DINOv3 cache is incomplete: {feature_count}/{len(expected)} features, {pca_count} PCA views."
        else:
            message = "DINOv3 evidence has not been generated."
        updated = str(
            (active_job or {}).get("updatedUtc")
            or summary.get("updatedUtc")
            or progress.get("updatedUtc")
            or ""
        )
        return {
            "target": "dinov3",
            "ready": bool(ready),
            "running": running,
            "failed": failed,
            "frameCount": len(expected),
            "completedFrameCount": int(feature_count),
            "generatedFrameCount": int(pca_count),
            "featureFrameCount": int(feature_count),
            "pcaFrameCount": int(pca_count),
            "currentFrameId": progress.get("currentFrameId") if running else None,
            "imageSize": int(self.config.image_size),
            "patchSize": int(self.config.patch_size),
            "model": "dinov3_vitl16",
            "message": message,
            "updatedUtc": updated,
            "root": str(self.root),
            "summaryPath": str(self.summary_path),
            "urlTemplate": "/api/view-evidence/frame/{frameId}/dinov3",
        }

    def start(self, frames: list[DinoFrame], *, overwrite: bool = False) -> dict[str, Any]:
        self._validate()
        with self._lock:
            running = bool(self._job is not None and self._job.get("running"))
        if running:
            return self.status(frames)
        if not overwrite and self.status(frames)["ready"]:
            return self.status(frames)
        if overwrite:
            self._clear_cache()
        self.root.mkdir(parents=True, exist_ok=True)
        manifest_path = self.root / "input.json"
        _write_json(
            manifest_path,
            {
                "schemaVersion": 1,
                "frames": [
                    {
                        "localIndex": int(frame.local_index),
                        "frameId": int(frame.frame_id),
                        "imagePath": str(frame.image_path),
                        "width": int(frame.width),
                        "height": int(frame.height),
                    }
                    for frame in frames
                ],
            },
        )
        with self._lock:
            self._job = {
                "running": True,
                "failed": False,
                "message": "Starting DINOv3 evidence generation.",
                "updatedUtc": _now(),
            }
        thread = threading.Thread(target=self._run, args=(frames, manifest_path), daemon=True)
        thread.start()
        return self.status(frames)

    def image_path(self, frame_id: int) -> Path:
        path = self._pca_path(int(frame_id))
        if not path.is_file():
            raise FileNotFoundError(f"DINOv3 PCA evidence does not exist for frame {frame_id}: {path}")
        return path

    def _run(self, frames: list[DinoFrame], manifest_path: Path) -> None:
        log_path = self.root / "backend.log"
        command = [
            str(self.config.python),
            "-m",
            "omega_local.segmentation.interactive.dinov3_runner",
            "--root",
            str(self.root),
            "--repo-root",
            str(self.config.repo_root),
            "--checkpoint",
            str(self.config.checkpoint),
            "--manifest",
            str(manifest_path),
            "--device",
            str(self.config.device),
            "--image-size",
            str(self.config.image_size),
            "--patch-size",
            str(self.config.patch_size),
        ]
        environment = dict(os.environ)
        existing_python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(self.config.omega_root), existing_python_path) if value
        )
        environment["PYTHONUNBUFFERED"] = "1"
        environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=self.config.repo_root,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"DINOv3 evidence generation failed (exit {completed.returncode}).\n{_tail(log_path)}")
            ready = self.status(frames)["ready"]
            if not ready:
                raise RuntimeError("DINOv3 runner finished without a complete feature/PCA cache.")
        except Exception as exc:
            with self._lock:
                self._job = {
                    "running": False,
                    "failed": True,
                    "message": str(exc),
                    "updatedUtc": _now(),
                }
            return
        with self._lock:
            self._job = {
                "running": False,
                "failed": False,
                "message": "DINOv3 evidence ready.",
                "updatedUtc": _now(),
            }

    def _validate(self) -> None:
        requirements = (
            (self.config.repo_root, "V2-SAM root", "dir"),
            (self.config.python, "V2-SAM Python", "file"),
            (self.config.checkpoint, "DINOv3 checkpoint", "file"),
            (self.config.omega_root, "OMeGa fork root", "dir"),
        )
        for raw_path, label, kind in requirements:
            path = Path(raw_path).expanduser().resolve()
            valid = path.is_dir() if kind == "dir" else path.is_file()
            if not valid:
                raise FileNotFoundError(f"{label} is missing: {path}")

    def _clear_cache(self) -> None:
        if not self.root.exists():
            return
        for child in self.root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def _feature_path(self, frame_id: int) -> Path:
        return self.features_dir / f"{int(frame_id):06d}.npy"

    def _pca_path(self, frame_id: int) -> Path:
        return self.pca_dir / f"{int(frame_id):06d}.png"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"DINOv3 log unavailable: {path}"
    return "\n".join(lines[-line_count:])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
