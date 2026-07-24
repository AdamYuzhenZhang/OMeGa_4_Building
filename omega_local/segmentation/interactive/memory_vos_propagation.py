"""Adapters for XMem++ and Cutie persistent-memory video propagation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from .paths import require_dir, require_file
from .sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
    _resize_label_map,
    _stage_video_frame,
)


@dataclass(frozen=True)
class MemoryVOSConfig:
    backend: str
    root: Path
    checkpoint: Path
    internal_size: int = 480
    device: str = "auto"


class MemoryVOSPropagationSession:
    """Runs one native memory-VOS backend in a disposable child process."""

    def __init__(self, config: MemoryVOSConfig, *, work_root: Path) -> None:
        backend = str(config.backend).strip().lower()
        if backend not in {"xmem", "cutie"}:
            raise ValueError(f"Unknown memory VOS backend: {config.backend}")
        self.config = MemoryVOSConfig(
            backend=backend,
            root=Path(config.root),
            checkpoint=Path(config.checkpoint),
            internal_size=int(config.internal_size),
            device=str(config.device),
        )
        self.work_root = Path(work_root)
        self._run_lock = threading.Lock()

    def propagate_labeled_sources(
        self,
        *,
        frames: list[PropagationFrame],
        sources: list[LabeledPropagationSource],
        start_local_index: int,
        region_rows: list[dict[str, Any]],
        progress_callback: Callable[[int, int, int], None] | None = None,
        save_callback: Callable[[PropagationFrame, np.ndarray], None] | None = None,
    ) -> list[dict[str, Any]]:
        del start_local_index, region_rows  # Native backends traverse the full sequence; previews use saved maps.
        root, checkpoint = self._validate_installation()
        prepared_sources, stable_ids = _prepare_sources(frames, sources)
        if len(stable_ids) > 255:
            raise ValueError(
                f"{self.config.backend} supports at most 255 simultaneous propagated regions; "
                f"the completed frames contain {len(stable_ids)}."
            )

        dense_by_stable = {stable_id: dense_id for dense_id, stable_id in enumerate(stable_ids, start=1)}
        stable_by_dense = np.zeros(256, dtype=np.uint16)
        for stable_id, dense_id in dense_by_stable.items():
            stable_by_dense[dense_id] = np.uint16(stable_id)
        source_by_local = {int(source.local_index): source.labels for source in prepared_sources}
        source_locals = set(source_by_local)
        anchor_frame_ids = sorted(int(source.frame_id) for source in prepared_sources)

        self.work_root.mkdir(parents=True, exist_ok=True)
        labels_by_local: dict[int, np.ndarray] = {}
        with self._run_lock:
            with tempfile.TemporaryDirectory(
                prefix=f"{self.config.backend}_region_video_",
                dir=self.work_root,
            ) as tmp_name:
                run_dir = Path(tmp_name)
                frames_dir = run_dir / "frames"
                masks_dir = run_dir / "anchors"
                output_dir = run_dir / "output"
                frames_dir.mkdir()
                masks_dir.mkdir()
                output_dir.mkdir()
                stage_size = (int(frames[0].width), int(frames[0].height))
                for local_index, frame in enumerate(frames):
                    _stage_video_frame(frame.image_path, frames_dir / f"{local_index:05d}.jpg", stage_size)
                for source in prepared_sources:
                    dense = _stable_to_dense(source.labels, dense_by_stable)
                    _save_palette_mask(dense, masks_dir / f"{int(source.local_index):05d}.png")

                log_path = run_dir / "backend.log"
                runner_device, device_environment = _subprocess_device_settings(self.config.device)
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("memory_vos_runner.py")),
                    "--backend",
                    self.config.backend,
                    "--root",
                    str(root),
                    "--checkpoint",
                    str(checkpoint),
                    "--frames-dir",
                    str(frames_dir),
                    "--masks-dir",
                    str(masks_dir),
                    "--output-dir",
                    str(output_dir),
                    "--source-indices",
                    ",".join(str(value) for value in sorted(source_locals)),
                    "--object-count",
                    str(len(stable_ids)),
                    "--internal-size",
                    str(self.config.internal_size),
                    "--device",
                    runner_device,
                ]
                environment = dict(os.environ)
                environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
                environment["PYTHONUNBUFFERED"] = "1"
                environment.update(device_environment)

                completed: set[int] = set()
                with log_path.open("w", encoding="utf-8") as log_handle:
                    process = subprocess.Popen(
                        command,
                        cwd=root,
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    try:
                        while process.poll() is None:
                            _collect_outputs(
                                output_dir / "masks",
                                frames,
                                source_by_local,
                                stable_by_dense,
                                labels_by_local,
                                completed,
                                save_callback,
                                progress_callback,
                            )
                            time.sleep(0.2)
                        _collect_outputs(
                            output_dir / "masks",
                            frames,
                            source_by_local,
                            stable_by_dense,
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
                    raise RuntimeError(
                        f"{self.config.backend.upper()} propagation failed (exit {process.returncode}).\n"
                        f"{_tail_text(log_path)}"
                    )
                missing = sorted(set(range(len(frames))) - set(labels_by_local))
                if missing:
                    raise RuntimeError(
                        f"{self.config.backend.upper()} did not produce {len(missing)} frame masks.\n"
                        f"{_tail_text(log_path)}"
                    )

        rows: list[dict[str, Any]] = []
        for local_index, frame in enumerate(frames):
            labels = labels_by_local[local_index]
            positive_count = int(np.count_nonzero(labels > 0))
            rows.append(
                {
                    "frameId": int(frame.frame_id),
                    "offset": int(local_index),
                    "isSource": bool(local_index in source_locals),
                    "isAnchor": bool(local_index in source_locals),
                    "width": int(frame.width),
                    "height": int(frame.height),
                    "areaPixels": positive_count,
                    "coverage": float(positive_count / max(labels.size, 1)),
                    "regionCount": int(np.count_nonzero(np.unique(labels) > 0)),
                    "anchorFrameIds": anchor_frame_ids,
                    "sourceRegionCount": int(len(stable_ids)),
                }
            )
        return rows

    def _validate_installation(self) -> tuple[Path, Path]:
        label = "XMem++" if self.config.backend == "xmem" else "Cutie"
        root = require_dir(self.config.root.expanduser().resolve(), f"{label} root")
        checkpoint = require_file(self.config.checkpoint.expanduser().resolve(), f"{label} checkpoint")
        if self.config.backend == "xmem":
            require_file(root / "inference" / "run_on_video.py", "XMem++ run_on_video.py")
        else:
            require_file(root / "cutie" / "inference" / "inference_core.py", "Cutie inference_core.py")
            require_file(root / "cutie" / "config" / "video_config.yaml", "Cutie video_config.yaml")
        if self.config.internal_size == 0 or self.config.internal_size < -1:
            raise ValueError("Memory VOS internal size must be -1 or a positive integer.")
        return root, checkpoint


def _prepare_sources(
    frames: list[PropagationFrame],
    sources: list[LabeledPropagationSource],
) -> tuple[list[LabeledPropagationSource], list[int]]:
    if not frames:
        raise ValueError("Memory VOS propagation needs at least one frame.")
    if not sources:
        raise ValueError("Memory VOS propagation needs at least one completed keyframe.")
    target_size = (int(frames[0].width), int(frames[0].height))
    prepared: list[LabeledPropagationSource] = []
    stable_ids: set[int] = set()
    for source in sources:
        local_index = int(source.local_index)
        if local_index < 0 or local_index >= len(frames):
            raise ValueError(f"Anchor index {local_index} is outside the staged sequence.")
        frame = frames[local_index]
        labels = np.asarray(source.labels, dtype=np.uint16)
        expected_shape = (int(frame.height), int(frame.width))
        if labels.shape != expected_shape:
            raise ValueError(
                f"Anchor region map shape {labels.shape} does not match frame {frame.frame_id} shape {expected_shape}."
            )
        if (int(frame.width), int(frame.height)) != target_size:
            labels = _resize_label_map(labels, target_size)
        ids = [int(value) for value in np.unique(labels).tolist() if int(value) > 0]
        if not ids:
            continue
        stable_ids.update(ids)
        prepared.append(
            LabeledPropagationSource(
                local_index=local_index,
                frame_id=int(frame.frame_id),
                labels=labels,
            )
        )
    if not prepared or not stable_ids:
        raise ValueError("Completed keyframe maps contain no persistent region pixels.")
    return prepared, sorted(stable_ids)


def _stable_to_dense(labels: np.ndarray, dense_by_stable: dict[int, int]) -> np.ndarray:
    dense = np.zeros(labels.shape, dtype=np.uint8)
    for stable_id, dense_id in dense_by_stable.items():
        dense[labels == stable_id] = np.uint8(dense_id)
    return dense


def _subprocess_device_settings(requested: str) -> tuple[str, dict[str, str]]:
    """Map a physical CUDA index to logical cuda:0 inside the isolated runner."""
    value = str(requested or "auto").strip()
    key = value.lower()
    if key == "cpu":
        return "cpu", {"CUDA_VISIBLE_DEVICES": ""}
    if key.startswith("cuda:"):
        try:
            index = int(key.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {requested!r}") from exc
        if index < 0:
            raise ValueError(f"Invalid CUDA device: {requested!r}")
        return "cuda", {"CUDA_VISIBLE_DEVICES": str(index)}
    return value, {}


def _save_palette_mask(labels: np.ndarray, path: Path) -> None:
    image = Image.fromarray(np.asarray(labels, dtype=np.uint8), mode="P")
    palette: list[int] = []
    for value in range(256):
        palette.extend((value, 0, 0))
    image.putpalette(palette)
    image.save(path)


def _read_dense_output(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    return np.asarray(array, dtype=np.uint8)


def _collect_outputs(
    output_dir: Path,
    frames: list[PropagationFrame],
    source_by_local: dict[int, np.ndarray],
    stable_by_dense: np.ndarray,
    labels_by_local: dict[int, np.ndarray],
    completed: set[int],
    save_callback: Callable[[PropagationFrame, np.ndarray], None] | None,
    progress_callback: Callable[[int, int, int], None] | None,
) -> None:
    for local_index, frame in enumerate(frames):
        if local_index in completed:
            continue
        path = output_dir / f"{local_index:05d}.png"
        if not path.exists():
            continue
        try:
            dense = _read_dense_output(path)
        except (OSError, ValueError):
            continue
        labels = stable_by_dense[dense]
        if local_index in source_by_local:
            labels = np.asarray(source_by_local[local_index], dtype=np.uint16).copy()
        target_size = (int(frame.width), int(frame.height))
        if labels.shape != (target_size[1], target_size[0]):
            labels = _resize_label_map(labels, target_size)
        labels = np.asarray(labels, dtype=np.uint16)
        labels_by_local[local_index] = labels
        if save_callback is not None:
            save_callback(frame, labels)
        completed.add(local_index)
        if progress_callback is not None:
            progress_callback(len(completed), len(frames), int(frame.frame_id))


def _tail_text(path: Path, line_count: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"Backend log unavailable: {path}"
    return "\n".join(lines[-line_count:])
