"""Parent-process adapter for focused VGGT-S region transfer."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
)


@dataclass(frozen=True)
class VggtSConfig:
    root: Path
    python: Path
    checkpoint: Path
    diagnostics_dir: Path
    vggt_model_id: str = "facebook/VGGT-1B"
    device: str = "auto"
    image_size: int = 518
    locator_points: int = 50
    locator_outliers: int = 10
    prompt_points: int = 5
    refinement_steps: int = 1
    seed: int = 71


class VggtSPropagationSession:
    def __init__(self, config: VggtSConfig, *, work_root: Path, omega_root: Path) -> None:
        self.config = config
        self.work_root = Path(work_root)
        self.omega_root = Path(omega_root)
        self._run_lock = threading.Lock()

    def propagate_labeled_sources(self, **_: Any) -> list[dict[str, Any]]:
        raise ValueError("VGGT-S is exposed as a focused source-region-target test.")

    def propagate_single_region(
        self,
        *,
        frames: list[PropagationFrame],
        source: LabeledPropagationSource,
        target_local_index: int,
        region_id: int,
        region_rows: list[dict[str, Any]],
        progress_callback: Callable[[int, int, int], None] | None = None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None = None,
    ) -> dict[str, Any]:
        del region_rows  # Pair previews render from the saved map and current persistent-region metadata.
        self._validate(frames, source, target_local_index, region_id)
        target_local_index = int(target_local_index)
        target = frames[target_local_index]
        source_frame = frames[int(source.local_index)]
        self.work_root.mkdir(parents=True, exist_ok=True)
        self.config.diagnostics_dir.mkdir(parents=True, exist_ok=True)

        with self._run_lock:
            with tempfile.TemporaryDirectory(prefix="vggts_pair_", dir=self.work_root) as tmp_name:
                run_dir = Path(tmp_name)
                source_mask_path = run_dir / "source_labels.npy"
                output_path = run_dir / "target_labels.npy"
                diagnostics_path = self.config.diagnostics_dir / (
                    f"source_{int(source.frame_id):06d}_target_{int(target.frame_id):06d}"
                    f"_region_{int(region_id):04d}.json"
                )
                np.save(source_mask_path, source.labels.astype(np.uint16, copy=False))
                command = self._command(
                    source_frame=source_frame,
                    target_frame=target,
                    source_mask_path=source_mask_path,
                    output_path=output_path,
                    diagnostics_path=diagnostics_path,
                    region_id=int(region_id),
                )
                environment = dict(os.environ)
                existing_python_path = environment.get("PYTHONPATH", "")
                environment["PYTHONPATH"] = os.pathsep.join(
                    value for value in (str(self.omega_root), existing_python_path) if value
                )
                environment["PYTHONUNBUFFERED"] = "1"
                environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                log_path = self.config.diagnostics_dir / "backend.log"
                with log_path.open("w", encoding="utf-8") as log_handle:
                    process = subprocess.run(
                        command,
                        cwd=self.config.root,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if process.returncode != 0:
                    raise RuntimeError(
                        f"VGGT-S region transfer failed (exit {process.returncode}).\n{_tail(log_path)}"
                    )
                if not output_path.exists():
                    raise RuntimeError(f"VGGT-S did not write its target mask.\n{_tail(log_path)}")
                labels = np.load(output_path).astype(np.uint16, copy=False)

        expected = (int(target.height), int(target.width))
        if labels.shape != expected:
            raise ValueError(f"VGGT-S target frame {target.frame_id} has output shape {labels.shape}; expected {expected}.")
        invalid = (labels != 0) & (labels != int(region_id))
        if np.any(invalid):
            raise ValueError("VGGT-S returned labels outside the requested persistent region.")
        if save_callback is not None:
            save_callback(target, labels)
        if progress_callback is not None:
            progress_callback(1, 1, int(target.frame_id))

        positive = int(np.count_nonzero(labels == int(region_id)))
        return {
            "frameId": int(target.frame_id),
            "sourceFrameId": int(source.frame_id),
            "regionId": int(region_id),
            "width": int(target.width),
            "height": int(target.height),
            "areaPixels": positive,
            "coverage": float(positive / max(labels.size, 1)),
            "diagnosticsPath": str(diagnostics_path),
        }

    def _command(
        self,
        *,
        source_frame: PropagationFrame,
        target_frame: PropagationFrame,
        source_mask_path: Path,
        output_path: Path,
        diagnostics_path: Path,
        region_id: int,
    ) -> list[str]:
        config = self.config
        return [
            str(config.python),
            "-m",
            "omega_local.segmentation.interactive.vggts_runner",
            "--root",
            str(config.root),
            "--checkpoint",
            str(config.checkpoint),
            "--vggt-model-id",
            str(config.vggt_model_id),
            "--source-image",
            str(source_frame.image_path),
            "--source-labels",
            str(source_mask_path),
            "--source-frame-id",
            str(source_frame.frame_id),
            "--target-image",
            str(target_frame.image_path),
            "--target-frame-id",
            str(target_frame.frame_id),
            "--region-id",
            str(region_id),
            "--output",
            str(output_path),
            "--diagnostics",
            str(diagnostics_path),
            "--device",
            str(config.device),
            "--image-size",
            str(config.image_size),
            "--locator-points",
            str(config.locator_points),
            "--locator-outliers",
            str(config.locator_outliers),
            "--prompt-points",
            str(config.prompt_points),
            "--refinement-steps",
            str(config.refinement_steps),
            "--seed",
            str(config.seed),
        ]

    def _validate(
        self,
        frames: list[PropagationFrame],
        source: LabeledPropagationSource,
        target_local_index: int,
        region_id: int,
    ) -> None:
        if not frames:
            raise ValueError("VGGT-S needs staged source and target frames.")
        if int(self.config.image_size) < 1 or int(self.config.image_size) % 14 != 0:
            raise ValueError("VGGT-S image size must be a positive multiple of the model's 14-pixel patch size.")
        if int(self.config.locator_points) < 1 or int(self.config.locator_outliers) < 0:
            raise ValueError("VGGT-S locator counts must be non-negative with at least one sampled point.")
        retained_locator_points = int(self.config.locator_points) - int(self.config.locator_outliers)
        if int(self.config.prompt_points) < 1 or int(self.config.prompt_points) > retained_locator_points:
            raise ValueError("VGGT-S prompt count must fit within the retained locator points.")
        if int(self.config.refinement_steps) < 0:
            raise ValueError("VGGT-S refinement steps cannot be negative.")
        target_local_index = int(target_local_index)
        if target_local_index < 0 or target_local_index >= len(frames):
            raise ValueError(f"VGGT-S target index is outside the frame sequence: {target_local_index}")
        if target_local_index == int(source.local_index):
            raise ValueError("VGGT-S pair testing needs different source and target frames.")
        source_ids = {int(value) for value in np.unique(source.labels) if int(value) > 0}
        if source_ids != {int(region_id)}:
            raise ValueError(f"VGGT-S source must contain only region {region_id}; found {sorted(source_ids)}.")
        requirements = (
            (self.config.root, "VGGT-S root", "dir"),
            (self.config.root / "src" / "model" / "predictor.py", "VGGT-S predictor", "file"),
            (self.config.python, "VGGT-S Python", "file"),
            (self.config.checkpoint, "VGGT-S checkpoint", "file"),
        )
        for raw_path, label, kind in requirements:
            path = Path(raw_path).expanduser().resolve()
            valid = path.is_dir() if kind == "dir" else path.is_file()
            if not valid:
                raise FileNotFoundError(f"{label} is missing: {path}")


def _tail(path: Path, line_count: int = 50) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"VGGT-S log unavailable: {path}"
    return "\n".join(lines[-line_count:])
