"""Path discovery and small IO helpers for the interactive editor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OMEGA_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = OMEGA_ROOT.parents[1]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class EditorPaths:
    model_dir: Path
    baseline_name: str
    baseline_dir: Path
    dataset_dir: Path
    frame_manifest: Path
    point_labels: Path | None
    points_path: Path
    interactive_dir: Path
    interactive_labels: Path
    interactive_edits: Path
    keyframes_dir: Path
    keyframes_summary: Path


def slug_token(value: str) -> str:
    out: list[str] = []
    last = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last = False
        elif not last:
            out.append("_")
            last = True
    return "".join(out).strip("_") or "interactive"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def resolve_paths(model_dir: Path, baseline_name: str) -> EditorPaths:
    model_dir = model_dir.expanduser().resolve()
    baseline_name = slug_token(baseline_name)
    baseline_dir = model_dir / "segmentation" / "baselines" / baseline_name
    if not baseline_dir.exists():
        parent = baseline_dir.parent
        available = sorted(path.name for path in parent.iterdir() if path.is_dir()) if parent.exists() else []
        raise FileNotFoundError(
            f"SAI3D baseline does not exist: {baseline_dir}. "
            f"Available baselines: {', '.join(available) if available else 'none'}"
        )

    dataset_dir = baseline_dir / "dataset"
    frame_manifest = dataset_dir / "frame_manifest.jsonl"
    point_labels_candidate = baseline_dir / "mesh_labels" / "point_labels.npy"
    point_labels = point_labels_candidate if point_labels_candidate.exists() else None
    scan_dirs = sorted((dataset_dir / "scans").glob("*"))
    points_candidates = [scan_dir / "points.pts" for scan_dir in scan_dirs]
    points_path = next((path for path in points_candidates if path.exists()), Path())

    missing = [
        ("frame manifest", frame_manifest),
        ("points.pts", points_path),
    ]
    absent = [f"{label}: {path}" for label, path in missing if not path.exists()]
    if absent:
        raise FileNotFoundError("Interactive editor is missing required SAI3D input data:\n" + "\n".join(absent))

    interactive_dir = baseline_dir / "interactive"
    keyframes_dir = interactive_dir / "keyframes"
    return EditorPaths(
        model_dir=model_dir,
        baseline_name=baseline_name,
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        frame_manifest=frame_manifest,
        point_labels=point_labels,
        points_path=points_path,
        interactive_dir=interactive_dir,
        interactive_labels=interactive_dir / "interactive_labels.npy",
        interactive_edits=interactive_dir / "interactive_edits.jsonl",
        keyframes_dir=keyframes_dir,
        keyframes_summary=keyframes_dir / "keyframes.json",
    )
