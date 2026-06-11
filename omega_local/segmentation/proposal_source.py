"""Shared resolver for per-view 2D proposal sources.

Downstream segmentation methods should consume this small contract instead of
knowing which method produced the masks. A proposal source lives under

    <model_dir>/segmentation/baselines/<proposal_source_name>/

and exposes a `baseline_summary.json` with `outputs.frameManifest` and
`outputs.viewMaskMasksDir`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    return "".join(out).strip("_") or "view_proposals"


def require_file(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def require_dir(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _resolve_output_path(value: Any, fallback: Path, base_dir: Path) -> Path:
    raw = Path(str(value)) if value is not None else fallback
    return raw.expanduser().resolve() if raw.is_absolute() else (base_dir / raw).resolve()


@dataclass(frozen=True)
class ProposalSource:
    name: str
    baseline_dir: Path
    summary_path: Path
    frame_manifest: Path
    mask_dir: Path
    image_dir: Path | None = None
    pose_dir: Path | None = None
    overlay_dir: Path | None = None
    predictions_path: Path | None = None

    def rows(self, frame_stride: int = 1, max_frames: int = 0) -> list[dict[str, Any]]:
        selected = read_jsonl(require_file(self.frame_manifest, "proposal source frame manifest"))
        selected = selected[:: max(int(frame_stride), 1)]
        if int(max_frames) > 0:
            selected = selected[: int(max_frames)]
        if not selected:
            raise SystemExit(f"No proposal source frames selected from {self.frame_manifest}")
        return selected

    def frame_id(self, row: dict[str, Any], fallback: int) -> int:
        return int(row.get("sourceFrameId", row.get("frameId", fallback)))

    def image_path(self, row: dict[str, Any]) -> Path:
        return require_file(self.baseline_dir / str(row["colorPath"]), "proposal source RGB frame")

    def pose_path(self, row: dict[str, Any]) -> Path:
        return require_file(self.baseline_dir / str(row["posePath"]), "proposal source camera pose")

    def mask_path(self, row: dict[str, Any], fallback: int) -> Path:
        frame_id = self.frame_id(row, fallback)
        candidates = [
            self.mask_dir / f"mask_{frame_id}.npy",
            self.mask_dir / f"{frame_id:06d}.npy",
            self.mask_dir / f"{frame_id:06d}.png",
            self.mask_dir / f"mask_{frame_id}.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise SystemExit(f"Could not find proposal mask for frame {frame_id} in {self.mask_dir}")


def resolve_proposal_source(
    model_dir: Path,
    *,
    proposal_source_name: str = "view_proposals_1024",
    proposal_source_dir: Path | None = None,
) -> ProposalSource:
    baseline_dir = (
        proposal_source_dir.expanduser().resolve()
        if proposal_source_dir is not None
        else model_dir / "segmentation" / "baselines" / slug_token(proposal_source_name)
    )
    summary_path = require_file(baseline_dir / "baseline_summary.json", "proposal source summary")
    summary = read_json(summary_path)
    outputs = summary.get("outputs", {})

    frame_manifest = _resolve_output_path(
        outputs.get("frameManifest"),
        baseline_dir / "dataset" / "frame_manifest.jsonl",
        baseline_dir,
    )
    mask_dir = _resolve_output_path(
        outputs.get("viewMaskMasksDir"),
        baseline_dir / "view_masks" / "masks",
        baseline_dir,
    )
    image_dir = _resolve_output_path(outputs.get("imageDir"), baseline_dir / "dataset" / "images", baseline_dir)
    pose_dir = _resolve_output_path(outputs.get("poseDir"), baseline_dir / "dataset" / "poses", baseline_dir)
    overlay_dir = _resolve_output_path(outputs.get("viewMaskOverlayDir"), baseline_dir / "view_masks" / "overlays", baseline_dir)
    predictions_path = _resolve_output_path(
        outputs.get("viewMaskPredictions"),
        baseline_dir / "view_masks" / "predictions.jsonl",
        baseline_dir,
    )

    return ProposalSource(
        name=slug_token(proposal_source_name),
        baseline_dir=require_dir(baseline_dir, "proposal source baseline directory"),
        summary_path=summary_path,
        frame_manifest=require_file(frame_manifest, "proposal source frame manifest"),
        mask_dir=require_dir(mask_dir, "proposal source mask directory"),
        image_dir=image_dir if image_dir.exists() else None,
        pose_dir=pose_dir if pose_dir.exists() else None,
        overlay_dir=overlay_dir if overlay_dir.exists() else None,
        predictions_path=predictions_path if predictions_path.exists() else None,
    )
