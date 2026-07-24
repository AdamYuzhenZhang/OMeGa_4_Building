"""Shared access to a completed sparse candidate layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .sam2_video_propagation import PropagationFrame


def sparse_source_availability(
    source_run_dir: Path,
    *,
    source_name: str,
) -> tuple[bool, str]:
    source_dir = source_run_dir.expanduser().resolve()
    for label, path in (
        ("input manifest", source_dir / "input.json"),
        ("summary", source_dir / "summary.json"),
        ("label maps", source_dir / "label_maps"),
    ):
        if not path.exists():
            return False, f"Run {source_name} first; its {label} is missing: {path}"
    return True, "Ready"


def validate_sparse_source(
    source_run_dir: Path,
    *,
    source_name: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    path = source_run_dir.expanduser().resolve() / "input.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read {source_name} input manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{source_name} input manifest must be a JSON object: {path}")
    if str(payload.get("fingerprint") or "") != str(expected_fingerprint):
        raise ValueError(
            f"{source_name} is stale relative to the current complete frames. "
            f"Run {source_name} again before dense recovery."
        )
    return payload


def load_sparse_labels(source_run_dir: Path, frame: PropagationFrame, *, source_name: str) -> np.ndarray:
    path = source_run_dir.expanduser().resolve() / "label_maps" / f"{int(frame.frame_id):06d}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"{source_name} has no sparse label map for frame {frame.frame_id}: {path}")
    labels = np.load(path)
    expected = (int(frame.height), int(frame.width))
    if labels.ndim != 2 or labels.shape != expected:
        raise ValueError(
            f"{source_name} label map for frame {frame.frame_id} has shape {labels.shape}; "
            f"expected {expected}."
        )
    return labels.astype(np.uint16, copy=False)
