"""Parent-process adapter for an isolated V2-SAM region-pair transfer."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
)


@dataclass(frozen=True)
class V2SamConfig:
    root: Path
    python: Path
    sam_checkpoint: Path
    dino_checkpoint: Path
    visual_checkpoint: Path
    fusion_checkpoint: Path
    feature_cache_dir: Path
    diagnostics_dir: Path
    expert_profile: str = "ego2exo"
    device: str = "auto"
    expert_batch_size: int = 8
    foreground_threshold: float = 0.6
    seed: int = 71


class V2SamPropagationSession:
    def __init__(self, config: V2SamConfig, *, work_root: Path, omega_root: Path) -> None:
        self.config = config
        self.work_root = Path(work_root)
        self.omega_root = Path(omega_root)
        self._run_lock = threading.Lock()

    def propagate_labeled_sources(
        self, **_: Any
    ) -> list[dict[str, Any]]:
        raise ValueError("V2-SAM is exposed as a focused source-region-target test.")

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
        del region_rows  # The pair-test layer renders the saved label map with live region colors.
        self._validate(frames, [source])
        target_local_index = int(target_local_index)
        if target_local_index < 0 or target_local_index >= len(frames):
            raise ValueError(f"V2-SAM target index is outside the frame sequence: {target_local_index}")
        if target_local_index == int(source.local_index):
            raise ValueError("V2-SAM pair testing needs different source and target frames.")
        source_ids = {int(value) for value in np.unique(source.labels) if int(value) > 0}
        if source_ids != {int(region_id)}:
            raise ValueError(
                f"V2-SAM pair source must contain only region {region_id}; found {sorted(source_ids)}."
            )
        labels_by_local = self._execute(
            frames=frames,
            source=source,
            target_local_index=target_local_index,
            progress_callback=progress_callback,
            save_callback=save_callback,
        )
        frame = frames[target_local_index]
        labels = labels_by_local[target_local_index]
        positive = int(np.count_nonzero(labels == int(region_id)))
        return {
            "frameId": int(frame.frame_id),
            "sourceFrameId": int(source.frame_id),
            "regionId": int(region_id),
            "width": int(frame.width),
            "height": int(frame.height),
            "areaPixels": positive,
            "coverage": float(positive / max(labels.size, 1)),
        }

    def _execute(
        self,
        *,
        frames: list[PropagationFrame],
        source: LabeledPropagationSource,
        target_local_index: int,
        progress_callback: Callable[[int, int, int], None] | None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None,
    ) -> dict[int, np.ndarray]:
        expected = {int(target_local_index)}
        self.work_root.mkdir(parents=True, exist_ok=True)
        labels_by_local: dict[int, np.ndarray] = {}
        with self._run_lock:
            with tempfile.TemporaryDirectory(prefix="v2sam_pair_", dir=self.work_root) as tmp_name:
                run_dir = Path(tmp_name)
                anchors_dir = run_dir / "anchors"
                output_dir = run_dir / "output"
                anchors_dir.mkdir()
                output_dir.mkdir()
                manifest_path = run_dir / "frames.json"
                manifest_path.write_text(
                    json.dumps(
                        {
                            "schemaVersion": 1,
                            "frames": [
                                {
                                    "localIndex": index,
                                    "frameId": int(frame.frame_id),
                                    "imagePath": str(frame.image_path),
                                    "width": int(frame.width),
                                    "height": int(frame.height),
                                }
                                for index, frame in enumerate(frames)
                            ],
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                np.save(
                    anchors_dir / f"{int(source.local_index):05d}.npy",
                    source.labels.astype(np.uint16),
                )

                command = self._command(
                    manifest_path=manifest_path,
                    anchors_dir=anchors_dir,
                    output_dir=output_dir,
                    source_index=int(source.local_index),
                    target_index=int(target_local_index),
                )
                environment = dict(os.environ)
                existing_python_path = environment.get("PYTHONPATH", "")
                environment["PYTHONPATH"] = os.pathsep.join(
                    value for value in (str(self.omega_root), existing_python_path) if value
                )
                environment["PYTHONUNBUFFERED"] = "1"
                environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                log_path = self.config.diagnostics_dir / "backend.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                completed: set[int] = set()
                with log_path.open("w", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=self.config.root,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        while process.poll() is None:
                            _collect_outputs(
                                output_dir / "masks",
                                frames,
                                expected,
                                labels_by_local,
                                completed,
                                save_callback,
                                progress_callback,
                            )
                            time.sleep(0.25)
                        _collect_outputs(
                            output_dir / "masks",
                            frames,
                            expected,
                            labels_by_local,
                            completed,
                            save_callback,
                            progress_callback,
                        )
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                if process.returncode != 0:
                    raise RuntimeError(f"V2-SAM propagation failed (exit {process.returncode}).\n{_tail(log_path)}")
                missing = sorted(expected - set(labels_by_local))
                if missing:
                    raise RuntimeError(f"V2-SAM did not produce {len(missing)} requested frame maps.\n{_tail(log_path)}")
        return labels_by_local

    def _command(
        self,
        *,
        manifest_path: Path,
        anchors_dir: Path,
        output_dir: Path,
        source_index: int,
        target_index: int,
    ) -> list[str]:
        config = self.config
        command = [
            str(config.python),
            "-m",
            "omega_local.segmentation.interactive.v2sam_runner",
            "--root",
            str(config.root),
            "--sam-checkpoint",
            str(config.sam_checkpoint),
            "--dino-checkpoint",
            str(config.dino_checkpoint),
            "--visual-checkpoint",
            str(config.visual_checkpoint),
            "--fusion-checkpoint",
            str(config.fusion_checkpoint),
            "--manifest",
            str(manifest_path),
            "--anchors-dir",
            str(anchors_dir),
            "--source-index",
            str(source_index),
            "--target-index",
            str(target_index),
            "--output-dir",
            str(output_dir),
            "--feature-cache-dir",
            str(config.feature_cache_dir),
            "--diagnostics-dir",
            str(config.diagnostics_dir),
            "--expert-profile",
            str(config.expert_profile),
            "--device",
            str(config.device),
            "--expert-batch-size",
            str(config.expert_batch_size),
            "--foreground-threshold",
            str(config.foreground_threshold),
            "--seed",
            str(config.seed),
        ]
        return command

    def _validate(self, frames: list[PropagationFrame], sources: list[LabeledPropagationSource]) -> None:
        if not frames or not sources:
            raise ValueError("V2-SAM needs staged frames and at least one completed manual anchor.")
        if int(self.config.expert_batch_size) < 1:
            raise ValueError("V2-SAM expert batch size must be positive.")
        if not 0.0 < float(self.config.foreground_threshold) < 1.0:
            raise ValueError("V2-SAM foreground threshold must be in (0, 1).")
        requirements = (
            (self.config.root, "V2-SAM root", "dir"),
            (self.config.python, "V2-SAM Python", "file"),
            (self.config.sam_checkpoint, "V2-SAM SAM2 checkpoint", "file"),
            (self.config.dino_checkpoint, "V2-SAM DINOv3 checkpoint", "file"),
            (self.config.visual_checkpoint, "V2-SAM Visual checkpoint", "file"),
            (self.config.fusion_checkpoint, "V2-SAM Fusion checkpoint", "file"),
        )
        for raw_path, label, kind in requirements:
            path = Path(raw_path).expanduser().resolve()
            valid = path.is_dir() if kind == "dir" else path.is_file()
            if not valid:
                raise FileNotFoundError(f"{label} is missing: {path}")


def _collect_outputs(
    masks_dir: Path,
    frames: list[PropagationFrame],
    expected_local_indices: set[int],
    labels_by_local: dict[int, np.ndarray],
    completed: set[int],
    save_callback: Callable[[PropagationFrame, np.ndarray], None] | None,
    progress_callback: Callable[[int, int, int], None] | None,
) -> None:
    for local_index in sorted(expected_local_indices):
        frame = frames[local_index]
        if local_index in completed:
            continue
        path = masks_dir / f"{local_index:05d}.npy"
        if not path.exists():
            continue
        try:
            labels = np.load(path).astype(np.uint16, copy=False)
        except (OSError, ValueError):
            continue
        if labels.shape != (int(frame.height), int(frame.width)):
            raise ValueError(f"V2-SAM frame {frame.frame_id} has output shape {labels.shape}.")
        labels_by_local[local_index] = labels.copy()
        if save_callback is not None:
            save_callback(frame, labels)
        completed.add(local_index)
        if progress_callback is not None:
            progress_callback(len(completed), len(expected_local_indices), int(frame.frame_id))


def _tail(path: Path, line_count: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"V2-SAM log unavailable: {path}"
    return "\n".join(lines[-line_count:])
