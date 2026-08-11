"""Discover Gaussian Flats jobs launched outside the interactive editor."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omega_local.reconstruction.backends.gaussian_flats.visualization import (
    ensure_visualization_artifacts,
)


_ITERATION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_STAGES = (
    ("prepare", "Prepare", Path("stages/prepare.json"), None),
    ("train", "Train Hybrid", Path("stages/train.json"), Path("logs/train.log")),
    ("render", "Render", Path("stages/render.json"), Path("logs/render_0.log")),
    ("mesh", "Extract Mesh", Path("stages/mesh.json"), Path("logs/mesh_0.log")),
)


def discover_gaussian_flats_runs(interactive_dir: Path) -> list[dict[str, Any]]:
    root = interactive_dir / "reconstruction" / "runs" / "gaussian_flats"
    rows = []
    for run_dir in sorted(root.glob("*")):
        manifest = _read_json(run_dir / "run.json")
        if manifest:
            rows.append(_run_status(run_dir, manifest))
    rows.sort(key=lambda row: str(row.get("updatedUtc") or ""), reverse=True)
    return rows


def _run_status(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    iterations = max(int(manifest.get("iterations", 30_000)), 1)
    stages = []
    first_incomplete = None
    active_log = None
    latest_mtime = 0.0
    for index, (stage_id, display_name, summary_rel, log_rel) in enumerate(_STAGES):
        complete = _read_json(run_dir / summary_rel).get("status") == "complete"
        if first_incomplete is None and not complete:
            first_incomplete = index
            active_log = run_dir / log_rel if log_rel is not None else None
        candidates = [run_dir / summary_rel]
        if log_rel is not None:
            candidates.append(run_dir / log_rel)
        latest_mtime = max(
            latest_mtime,
            *(path.stat().st_mtime for path in candidates if path.exists()),
        )
        stages.append(
            {
                "stageId": stage_id,
                "displayName": display_name,
                "complete": complete,
                "status": "complete" if complete else "pending",
            }
        )
    if first_incomplete is None:
        first_incomplete = len(stages)

    current_iteration = 0
    train_log = run_dir / "logs" / "train.log"
    if train_log.is_file():
        matches = _ITERATION_RE.findall(_read_text_tail(train_log))
        if matches:
            current_iteration = min(int(matches[-1][0]), iterations)
    if stages[1]["complete"]:
        current_iteration = iterations
    train_fraction = current_iteration / iterations

    complete = all(stage["complete"] for stage in stages)
    active_stage = first_incomplete if first_incomplete < len(stages) else -1
    running = bool(
        not complete
        and active_log is not None
        and active_log.is_file()
        and time.time() - active_log.stat().st_mtime < 180.0
    )
    if active_stage >= 0:
        stages[active_stage]["status"] = "running" if running else "pending"
    stage_fraction = train_fraction if active_stage == 1 else 0.0
    overall = (
        1.0
        if complete
        else min(
            (sum(stage["complete"] for stage in stages) + stage_fraction)
            / len(stages),
            1.0,
        )
    )

    viewer_ready = False
    artifact_error = ""
    final_ply = (
        run_dir
        / "02_model"
        / "point_cloud"
        / f"iteration_{iterations}"
        / "point_cloud.ply"
    )
    if stages[1]["complete"] and final_ply.is_file():
        try:
            artifacts = ensure_visualization_artifacts(run_dir)
            viewer_ready = bool(
                artifacts.get("planeMaskArtifacts")
                or artifacts.get("meshArtifacts")
            )
        except (OSError, ValueError, ImportError) as exc:
            artifact_error = str(exc)

    if active_stage == 1 and current_iteration:
        message = f"Train Hybrid · {current_iteration}/{iterations}"
    elif active_stage >= 0:
        message = stages[active_stage]["displayName"]
    else:
        message = "Complete"
    if artifact_error:
        message = f"Artifacts need attention: {artifact_error}"

    updated = datetime.fromtimestamp(
        latest_mtime or time.time(), tz=timezone.utc
    ).isoformat()
    return {
        "runId": str(manifest.get("runId") or run_dir.name),
        "savedRunId": f"gaussian_flats_{run_dir.name}",
        "displayName": f"Gaussian Flats · {manifest.get('planarRegion') or 'Planar Region'}",
        "runDir": str(run_dir),
        "planarRegion": str(manifest.get("planarRegion") or ""),
        "status": "complete" if complete else "running" if running else "paused",
        "running": running,
        "failed": False,
        "complete": complete,
        "viewerReady": viewer_ready,
        "message": message,
        "detailMessage": artifact_error,
        "updatedUtc": updated,
        "stageIndex": active_stage,
        "stageCount": len(stages),
        "overallProgress": round(100.0 * overall, 2),
        "stages": stages,
        "currentIteration": current_iteration,
        "currentIterationTotal": iterations,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_text_tail(path: Path, byte_count: int = 262_144) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - max(int(byte_count), 1)))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
