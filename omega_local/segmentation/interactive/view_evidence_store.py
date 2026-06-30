"""Cache/status helpers for interactive view evidence."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _target_names(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("targets", payload.get("target", "all"))
    values = raw if isinstance(raw, list) else [raw]
    out: list[str] = []
    for value in values:
        item = str(value).strip().lower().replace("_", "-")
        if item in {"all", "both", "view-evidence"}:
            for target in ["normal", "depth"]:
                if target not in out:
                    out.append(target)
        elif item in {"normal", "normals", "stable-normal", "stablenormal"}:
            if "normal" not in out:
                out.append("normal")
        elif item in {"depth", "depths", "depth-anything", "depth-anything-v2"}:
            if "depth" not in out:
                out.append("depth")
        elif item in {"", "none"}:
            continue
        else:
            raise ValueError(f"Unknown view evidence target: {value}")
    if not out:
        raise ValueError("No view evidence target requested.")
    return tuple(out)


def _target_label(targets: tuple[str, ...]) -> str:
    if set(targets) == {"normal", "depth"}:
        return "normal/depth evidence"
    if targets == ("normal",):
        return "StableNormal evidence"
    if targets == ("depth",):
        return "Depth Anything V2 evidence"
    return "/".join(targets)


def _targets_ready(paths: ViewEvidenceRunPaths, frames: list[ViewEvidenceFrame], targets: tuple[str, ...]) -> bool:
    if "normal" in targets and _count_generated_kind(paths, frames, "normal") < len(frames):
        return False
    if "depth" in targets and _count_generated_kind(paths, frames, "depth") < len(frames):
        return False
    return bool(frames)


def _clear_target_outputs(paths: ViewEvidenceRunPaths, targets: tuple[str, ...]) -> None:
    dirs: list[Path] = []
    if "normal" in targets:
        dirs.extend([paths.normal_dir, paths.normal_npz_dir, paths.normal_edge_dir])
    if "depth" in targets:
        dirs.extend([paths.depth_dir, paths.depth_npz_dir, paths.depth_edge_dir])
    for directory in dirs:
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)


def _read_frame_metadata(path: Path, frame: ViewEvidenceFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload.setdefault("frameId", int(frame.frame_id))
    payload.setdefault("sourceFrameId", int(frame.source_frame_id))
    payload.setdefault("scanID", frame.scan_id)
    payload.setdefault("imageName", frame.image_name)
    payload.setdefault("width", int(frame.width))
    payload.setdefault("height", int(frame.height))
    payload.setdefault("normalReady", False)
    payload.setdefault("depthReady", False)
    payload.setdefault("normalSource", {})
    payload.setdefault("depthSource", {})
    payload.setdefault("outputs", {})
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_token(value: str) -> str:
    out: list[str] = []
    last = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last = False
        elif not last:
            out.append("_")
            last = True
    return "".join(out).strip("_") or "view_evidence"


def _adopt_legacy_evidence_run(paths: ViewEvidenceRunPaths, run_name: str) -> None:
    """Move one pre-canonical evidence cache into the editor-owned folder."""

    if paths.summary.exists() or not paths.run_dir.exists():
        return
    try:
        candidates = [
            child
            for child in paths.run_dir.iterdir()
            if child.is_dir() and child.name.startswith("view_evidence_") and (child / "summary.json").exists()
        ]
    except OSError:
        return
    if len(candidates) != 1:
        return

    legacy_dir = candidates[0]
    try:
        legacy_items = list(legacy_dir.iterdir())
    except OSError:
        return
    for item in legacy_items:
        if (paths.run_dir / item.name).exists():
            return

    for item in legacy_items:
        item.rename(paths.run_dir / item.name)
    try:
        legacy_dir.rmdir()
    except OSError:
        pass

    _patch_adopted_json(paths.summary, run_name, paths)
    _patch_adopted_json(paths.progress, run_name, paths)
    _patch_adopted_json(paths.config, run_name, paths)


def _patch_adopted_json(path: Path, run_name: str, paths: ViewEvidenceRunPaths) -> None:
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if not isinstance(payload, dict):
        return
    payload["runName"] = run_name
    payload["runDir"] = str(paths.run_dir)
    if path == paths.summary:
        payload["outputs"] = {
            "normalDir": str(paths.normal_dir),
            "normalNpzDir": str(paths.normal_npz_dir),
            "normalEdgeDir": str(paths.normal_edge_dir),
            "depthDir": str(paths.depth_dir),
            "depthNpzDir": str(paths.depth_npz_dir),
            "depthEdgeDir": str(paths.depth_edge_dir),
            "metadataDir": str(paths.metadata_dir),
            "frameIndex": str(paths.frame_index),
        }
    if path == paths.progress:
        payload["summaryPath"] = str(paths.summary)
        payload.setdefault("ready", paths.summary.exists())
        payload.setdefault("running", False)
    _write_json(path, payload)


def _count_existing(metadata_dir: Path, frames: list[ViewEvidenceFrame]) -> int:
    return sum(1 for frame in frames if (metadata_dir / f"{int(frame.frame_id):06d}.json").exists())


def _evidence_kind_status(
    paths: ViewEvidenceRunPaths,
    frames: list[ViewEvidenceFrame],
    kind: str,
    active_job: dict[str, Any] | None,
) -> dict[str, Any]:
    count = _count_generated_kind(paths, frames, kind)
    total = len(frames)
    payload = _read_evidence_kind_status(paths, kind)
    running = bool(
        active_job
        and active_job.get("running")
        and kind in {str(target) for target in active_job.get("targets", [])}
    )
    ready = bool(total and count >= total)
    failed = bool(payload.get("failed", False)) and not running and not ready
    if running:
        completed = int(active_job.get("completedFrameCount", count) or 0)
        message = str(active_job.get("message", f"Generating {_target_label((kind,))}."))
        updated = active_job.get("updatedUtc", _now())
        current_frame_id = active_job.get("currentFrameId")
    else:
        completed = count
        message = str(payload.get("message") or _default_kind_message(kind, count, total, ready, failed))
        updated = payload.get("updatedUtc")
        current_frame_id = payload.get("currentFrameId")
    return {
        "target": kind,
        "ready": ready,
        "running": running,
        "failed": failed,
        "frameCount": total,
        "completedFrameCount": completed,
        "generatedFrameCount": count,
        "currentFrameId": current_frame_id,
        "message": message,
        "updatedUtc": updated,
    }


def _default_kind_message(kind: str, count: int, total: int, ready: bool, failed: bool) -> str:
    label = _target_label((kind,))
    if failed:
        return f"{label} failed. Retry this evidence type."
    if ready:
        return f"{label} ready: {count}/{total} frames."
    if count > 0:
        return f"{label} incomplete: {count}/{total} frames."
    return f"{label} not generated."


def _target_status_path(paths: ViewEvidenceRunPaths, kind: str) -> Path:
    if kind == "normal":
        return paths.run_dir / "normal_status.json"
    if kind == "depth":
        return paths.run_dir / "depth_status.json"
    raise ValueError(f"Unknown view evidence target: {kind}")


def _read_evidence_kind_status(paths: ViewEvidenceRunPaths, kind: str) -> dict[str, Any]:
    path = _target_status_path(paths, kind)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_evidence_kind_status(paths: ViewEvidenceRunPaths, kind: str, payload: dict[str, Any]) -> None:
    _write_json(_target_status_path(paths, kind), payload)


def _count_generated_kind(paths: ViewEvidenceRunPaths, frames: list[ViewEvidenceFrame], kind: str) -> int:
    directory = paths.normal_dir if kind == "normal" else paths.depth_dir
    return sum(
        1
        for frame in frames
        if (directory / f"{int(frame.frame_id):06d}.png").exists()
        and _frame_has_generated_kind(paths, int(frame.frame_id), kind)
    )


def _frame_has_generated_kind(paths: ViewEvidenceRunPaths, frame_id: int, kind: str) -> bool:
    metadata_path = paths.metadata_dir / f"{int(frame_id):06d}.json"
    if not metadata_path.exists():
        return False
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, dict):
        return False
    source = payload.get("normalSource") if kind == "normal" else payload.get("depthSource")
    if not isinstance(source, dict):
        return False
    method = str(source.get("method", "")).strip().lower()
    if kind == "normal":
        return method == "stable_normal_runtime"
    return method == "depth_anything_v2_transformers"
