"""Persistent DINOv3 patch features used by the V2-SAM propagation backend."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class DinoFrame:
    local_index: int
    frame_id: int
    image_path: Path
    width: int
    height: int


class DinoFeatureCache:
    """Resumable cache matching V2-SAM's DINOv3 ViT-L/16 preprocessing."""

    schema_version = 1

    def __init__(
        self,
        root: Path,
        *,
        model,
        device,
        checkpoint: Path,
        image_size: int = 768,
        patch_size: int = 16,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.device = device
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.cache_signature = self._cache_signature()
        self.features_dir = self.root / "features"
        self.metadata_dir = self.root / "metadata"
        self.pca_dir = self.root / "pca"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.pca_dir.mkdir(parents=True, exist_ok=True)

    def feature_path(self, frame_id: int) -> Path:
        return self.features_dir / f"{int(frame_id):06d}.npy"

    def metadata_path(self, frame_id: int) -> Path:
        return self.metadata_dir / f"{int(frame_id):06d}.json"

    def pca_path(self, frame_id: int) -> Path:
        return self.pca_dir / f"{int(frame_id):06d}.png"

    def prepare(
        self,
        frames: list[DinoFrame],
        *,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self._write_config(frames)
        completed = 0
        for frame in frames:
            if not self._is_current(frame):
                features = self._extract(frame.image_path)
                self._save(frame, features)
            completed += 1
            self._write_progress(frames, completed, frame.frame_id, complete=False)
            if progress_callback is not None:
                progress_callback(completed, len(frames), int(frame.frame_id))
        self._write_progress(frames, completed, None, complete=True)

    def load(self, frame_id: int) -> np.ndarray:
        path = self.feature_path(frame_id)
        if not path.exists():
            raise FileNotFoundError(f"DINOv3 feature cache is missing frame {frame_id}: {path}")
        features = np.load(path, mmap_mode="r")
        if features.ndim != 3:
            raise ValueError(f"Expected DINOv3 features [D,H,W], got {features.shape}: {path}")
        return features

    def features_ready(self, frames: list[DinoFrame]) -> bool:
        return bool(frames) and all(self._is_current(frame) for frame in frames)

    def pca_ready(self, frames: list[DinoFrame]) -> bool:
        if not self.features_ready(frames):
            return False
        basis_path = self.root / "pca_basis.npz"
        summary_path = self.root / "summary.json"
        if not basis_path.is_file() or not summary_path.is_file():
            return False
        if not all(self.pca_path(frame.frame_id).is_file() for frame in frames):
            return False
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return False
        return bool(summary.get("featureFingerprint") == self._feature_fingerprint(frames))

    def ready(self, frames: list[DinoFrame]) -> bool:
        return self.pca_ready(frames)

    def build_pca_visualizations(self, frames: list[DinoFrame], *, max_samples: int = 65536) -> None:
        if not frames:
            return
        fingerprint = self._feature_fingerprint(frames)
        basis_path = self.root / "pca_basis.npz"
        summary_path = self.root / "summary.json"
        if basis_path.exists() and summary_path.exists() and all(self.pca_path(frame.frame_id).exists() for frame in frames):
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary.get("featureFingerprint") == fingerprint:
                    return
            except (OSError, ValueError, TypeError):
                pass

        import torch

        samples: list[np.ndarray] = []
        per_frame = max(32, int(max_samples) // max(len(frames), 1))
        for frame in frames:
            feature = np.asarray(self.load(frame.frame_id), dtype=np.float32)
            flat = np.moveaxis(feature, 0, -1).reshape(-1, feature.shape[0])
            if flat.shape[0] > per_frame:
                positions = np.linspace(0, flat.shape[0] - 1, per_frame, dtype=np.int64)
                flat = flat[positions]
            samples.append(flat)
        sample = np.concatenate(samples, axis=0)
        if sample.shape[0] > max_samples:
            positions = np.linspace(0, sample.shape[0] - 1, max_samples, dtype=np.int64)
            sample = sample[positions]

        sample_t = torch.from_numpy(sample).to(self.device, dtype=torch.float32)
        mean_t = sample_t.mean(dim=0)
        _, _, basis_t = torch.pca_lowrank(sample_t - mean_t, q=3, center=False, niter=4)
        for column in range(basis_t.shape[1]):
            vector = basis_t[:, column]
            pivot = torch.argmax(torch.abs(vector))
            if vector[pivot] < 0:
                basis_t[:, column].neg_()
        projected_sample = (sample_t - mean_t) @ basis_t
        low_t = torch.quantile(projected_sample, 0.01, dim=0)
        high_t = torch.quantile(projected_sample, 0.99, dim=0)
        mean = mean_t.cpu().numpy().astype(np.float32)
        basis = basis_t.cpu().numpy().astype(np.float32)
        low = low_t.cpu().numpy().astype(np.float32)
        high = high_t.cpu().numpy().astype(np.float32)
        del sample_t, mean_t, basis_t, projected_sample, low_t, high_t

        _atomic_npz(basis_path, mean=mean, basis=basis, low=low, high=high)
        denominator = np.maximum(high - low, 1e-6)
        for frame in frames:
            feature = np.asarray(self.load(frame.frame_id), dtype=np.float32)
            grid = np.moveaxis(feature, 0, -1)
            rgb = np.clip(((grid - mean) @ basis - low) / denominator, 0.0, 1.0)
            rgb_u8 = np.rint(rgb * 255.0).astype(np.uint8)
            image = Image.fromarray(rgb_u8, mode="RGB").resize(
                (int(frame.width), int(frame.height)),
                Image.Resampling.BILINEAR,
            )
            _atomic_image(self.pca_path(frame.frame_id), image)

        _atomic_json(
            summary_path,
            {
                "schemaVersion": self.schema_version,
                "stage": "interactive_dinov3_evidence",
                "model": "dinov3_vitl16",
                "imageSize": self.image_size,
                "patchSize": self.patch_size,
                "frameCount": len(frames),
                "featureFingerprint": fingerprint,
                "featureDirectory": str(self.features_dir),
                "pcaDirectory": str(self.pca_dir),
                "pcaBasis": str(basis_path),
                "visualization": "Dataset-consistent PCA of normalized final-layer DINOv3 patch features.",
                "updatedUtc": _now(),
            },
        )

    def _extract(self, image_path: Path) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if self.model is None:
            raise RuntimeError("DINOv3 model is required to extract missing feature frames.")

        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            grid_h = self.image_size // self.patch_size
            grid_w = int((width * self.image_size) / (height * self.patch_size))
            target_size = (grid_w * self.patch_size, grid_h * self.patch_size)
            resized = image.resize(target_size, Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            feature = self.model.get_intermediate_layers(
                tensor,
                n=[23],
                reshape=True,
                norm=True,
            )[-1].squeeze(0)
            feature = F.normalize(feature.float(), p=2, dim=0)
        return feature.cpu().numpy().astype(np.float16)

    def _save(self, frame: DinoFrame, features: np.ndarray) -> None:
        feature_path = self.feature_path(frame.frame_id)
        _atomic_npy(feature_path, np.asarray(features, dtype=np.float16))
        stat = frame.image_path.stat()
        _atomic_json(
            self.metadata_path(frame.frame_id),
            {
                "schemaVersion": self.schema_version,
                "frameId": int(frame.frame_id),
                "localIndex": int(frame.local_index),
                "imagePath": str(frame.image_path),
                "imageBytes": int(stat.st_size),
                "imageMtimeNs": int(stat.st_mtime_ns),
                "width": int(frame.width),
                "height": int(frame.height),
                "featureShape": list(features.shape),
                "featureDtype": "float16",
                "featurePath": str(feature_path),
                "cacheSignature": self.cache_signature,
                "updatedUtc": _now(),
            },
        )

    def _is_current(self, frame: DinoFrame) -> bool:
        feature_path = self.feature_path(frame.frame_id)
        metadata_path = self.metadata_path(frame.frame_id)
        if not feature_path.exists() or not metadata_path.exists():
            return False
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if payload.get("cacheSignature") != self.cache_signature:
                return False
            stat = frame.image_path.stat()
            if int(payload.get("imageBytes", -1)) != int(stat.st_size):
                return False
            if int(payload.get("imageMtimeNs", -1)) != int(stat.st_mtime_ns):
                return False
            feature = np.load(feature_path, mmap_mode="r")
            return feature.ndim == 3 and feature.shape[0] == 1024
        except (OSError, ValueError, TypeError):
            return False

    def _write_config(self, frames: list[DinoFrame]) -> None:
        _atomic_json(
            self.root / "config.json",
            {
                "schemaVersion": self.schema_version,
                "stage": "interactive_dinov3_evidence",
                "model": "dinov3_vitl16",
                "checkpoint": str(self.checkpoint),
                "imageSize": self.image_size,
                "patchSize": self.patch_size,
                "frameCount": len(frames),
                "featureDtype": "float16",
                "featureNormalization": "L2 channel normalization per patch",
                "preprocessing": "V2-SAM aspect-preserving resize with height 768 and dimensions divisible by 16",
                "cacheSignature": self.cache_signature,
                "updatedUtc": _now(),
            },
        )

    def _write_progress(
        self,
        frames: list[DinoFrame],
        completed: int,
        current_frame_id: int | None,
        *,
        complete: bool,
    ) -> None:
        _atomic_json(
            self.root / "progress.json",
            {
                "schemaVersion": self.schema_version,
                "status": "complete" if complete else "running",
                "framesDone": int(completed),
                "totalFrames": int(len(frames)),
                "currentFrameId": None if current_frame_id is None else int(current_frame_id),
                "updatedUtc": _now(),
            },
        )

    def _feature_fingerprint(self, frames: list[DinoFrame]) -> str:
        digest = hashlib.sha256()
        digest.update(f"dinov3_vitl16:{self.image_size}:{self.patch_size}\0".encode("ascii"))
        for frame in frames:
            path = self.feature_path(frame.frame_id)
            stat = path.stat()
            digest.update(f"{frame.frame_id}:{stat.st_size}:{stat.st_mtime_ns}\0".encode("ascii"))
        return digest.hexdigest()

    def _cache_signature(self) -> str:
        stat = self.checkpoint.stat()
        payload = (
            f"dinov3_vitl16:{self.image_size}:{self.patch_size}:"
            f"{self.checkpoint}:{stat.st_size}:{stat.st_mtime_ns}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(path)


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    image.save(temporary)
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
