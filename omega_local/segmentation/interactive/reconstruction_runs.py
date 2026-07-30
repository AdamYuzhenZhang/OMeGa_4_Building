"""Read-only discovery of externally launched reconstruction pipelines."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_STAGES = (
    ("prepare", "Prepare", Path("01_input/stage.json")),
    ("split", "Split", Path("02_split/clean_masks/stage.json")),
    (
        "splat_prepare",
        "Region Data",
        Path("03_splat/01_region_datasets/stage.json"),
    ),
    (
        "splat_train",
        "Train Regions",
        Path("03_splat/02_region_models/stage.json"),
    ),
    (
        "splat_compose",
        "Compose",
        Path("03_splat/03_outputs/stage.json"),
    ),
)
_ITERATION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def discover_mapanything_runs(interactive_dir: Path) -> list[dict[str, Any]]:
    root = (
        interactive_dir
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
    )
    rows = []
    for run_dir in sorted(root.glob("*")):
        manifest = _read_json(run_dir / "run.json")
        if not manifest:
            continue
        rows.append(_mapanything_status(run_dir, manifest))
    rows.sort(key=lambda row: str(row.get("updatedUtc") or ""), reverse=True)
    return rows


def live_mapanything_experiment(
    interactive_dir: Path,
    run_id: str,
) -> dict[str, Any] | None:
    normalized = str(run_id).strip()
    if not normalized.endswith("_splat"):
        return None
    base_run_id = normalized[: -len("_splat")]
    pipeline = next(
        (
            row
            for row in discover_mapanything_runs(interactive_dir)
            if str(row["runId"]) == base_run_id
        ),
        None,
    )
    if pipeline is None or not pipeline["completedRegions"]:
        return None
    artifacts = {
        f"rgb_region_{row['regionId']}": {
            "displayName": f"Region {row['regionId']} RGB",
            "path": row["pointCloud"],
            "format": "3dgs_ply",
            "pointCount": row["gaussianCount"],
            "instanceId": row["regionId"],
            "artifactGroup": "rgb_objects",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
        for row in pipeline["completedRegions"]
    }
    return {
        "schemaVersion": 1,
        "runId": normalized,
        "methodId": "mapanything_region_3dgs",
        "experimentFamily": "mapanything_region_3dgs",
        "artifactRole": "reconstruction",
        "baseRunId": base_run_id,
        "reconstructionVariant": "mapanything_point_initialized_3dgs",
        "inputId": pipeline["propagationMethod"],
        "displayName": f"MapAnything Region 3DGS: {base_run_id}",
        "labelSpace": "persistent_region",
        "geometrySource": "omega_mapanything_initializer",
        "manualFrameWeight": pipeline["manualFrameWeight"],
        "ready": True,
        "timestampUtc": pipeline["updatedUtc"],
        "canonicalRunDir": pipeline["runDir"],
        "pointSummary": {},
        "gaussianArtifacts": artifacts,
    }


def _mapanything_status(
    run_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    progress = _read_json(run_dir / "progress.json")
    stage_rows = []
    first_incomplete = None
    for index, (stage_id, display_name, relative) in enumerate(_STAGES):
        summary = _read_json(run_dir / relative)
        complete = summary.get("status") == "complete"
        if first_incomplete is None and not complete:
            first_incomplete = index
        stage_rows.append(
            {
                "stageId": stage_id,
                "displayName": display_name,
                "complete": complete,
                "status": "complete" if complete else "pending",
            }
        )
    if first_incomplete is None:
        first_incomplete = len(stage_rows)

    prepared = _read_json(run_dir / _STAGES[2][2])
    prepared_regions = [
        int(row["regionId"])
        for row in prepared.get("instances", [])
        if int(row.get("regionId", 0)) > 0
    ]
    iterations = max(int(manifest.get("instanceIterations", 10_000)), 1)
    completed_regions = _completed_regions(
        run_dir / "03_splat/02_region_models",
        prepared_regions,
        iterations,
    )
    completed_ids = {int(row["regionId"]) for row in completed_regions}
    remaining_ids = [
        region_id
        for region_id in prepared_regions
        if region_id not in completed_ids
    ]
    active_region_id = remaining_ids[0] if remaining_ids else 0

    message = str(progress.get("message") or "")
    current_iteration = 0
    if "Training progress:" in message:
        match = _ITERATION_RE.search(message)
        if match:
            current_iteration = min(int(match.group(1)), iterations)
    region_fraction = current_iteration / iterations if active_region_id else 0.0
    training_fraction = (
        (len(completed_regions) + region_fraction) / len(prepared_regions)
        if prepared_regions
        else 0.0
    )
    if stage_rows[3]["complete"]:
        training_fraction = 1.0

    progress_status = str(progress.get("status") or "")
    complete = all(row["complete"] for row in stage_rows)
    failed = progress_status == "failed"
    running = progress_status == "running" and not complete and not failed
    active_stage = first_incomplete if first_incomplete < len(stage_rows) else -1
    if active_stage >= 0:
        stage_rows[active_stage]["status"] = (
            "failed" if failed else "running" if running else "pending"
        )

    stage_fraction = 0.0
    if active_stage == 3:
        stage_fraction = training_fraction
    completed_stage_count = sum(row["complete"] for row in stage_rows)
    overall_fraction = min(
        (completed_stage_count + stage_fraction) / len(stage_rows),
        1.0,
    )
    if complete:
        overall_fraction = 1.0

    if active_stage == 3 and active_region_id:
        display_message = (
            f"Region {active_region_id} · "
            f"{len(completed_regions)}/{len(prepared_regions)} complete · "
            f"{round(100 * region_fraction)}%"
        )
    elif active_stage >= 0:
        display_message = stage_rows[active_stage]["displayName"]
    else:
        display_message = "Complete"
    if failed:
        display_message = message or "Pipeline failed"

    return {
        "runId": str(manifest.get("runId") or run_dir.name),
        "displayName": _display_name(
            str(manifest.get("runId") or run_dir.name)
        ),
        "runDir": str(run_dir),
        "propagationMethod": str(
            manifest.get("propagationMethod") or ""
        ),
        "manualFrameWeight": int(manifest.get("manualFrameWeight", 1)),
        "instanceIterations": iterations,
        "status": (
            "complete"
            if complete
            else "failed"
            if failed
            else "running"
            if running
            else "paused"
        ),
        "running": running,
        "failed": failed,
        "complete": complete,
        "message": display_message,
        "detailMessage": message,
        "updatedUtc": str(
            progress.get("updatedUtc")
            or manifest.get("updatedUtc")
            or datetime.now(timezone.utc).isoformat()
        ),
        "stageIndex": active_stage,
        "stageCount": len(stage_rows),
        "overallProgress": round(100.0 * overall_fraction, 2),
        "stages": stage_rows,
        "regionCount": len(prepared_regions),
        "trainedRegionCount": len(completed_regions),
        "currentRegionId": active_region_id,
        "currentIteration": current_iteration,
        "currentIterationTotal": iterations,
        "regionProgress": round(100.0 * region_fraction, 2),
        "discardedRegionCount": int(
            prepared.get("discardedInstanceCount", 0)
        ),
        "completedRegions": completed_regions,
    }


def _completed_regions(
    models_dir: Path,
    region_ids: list[int],
    iterations: int,
) -> list[dict[str, Any]]:
    rows = []
    for region_id in region_ids:
        point_cloud = (
            models_dir
            / str(region_id)
            / "point_cloud"
            / f"iteration_{iterations}"
            / "point_cloud.ply"
        )
        if not point_cloud.is_file():
            continue
        rows.append(
            {
                "regionId": region_id,
                "pointCloud": str(point_cloud),
                "gaussianCount": _ply_vertex_count(point_cloud),
            }
        )
    return rows


def _ply_vertex_count(path: Path) -> int:
    try:
        with path.open("rb") as stream:
            for _ in range(100):
                line = stream.readline()
                if not line:
                    break
                text = line.decode("ascii", errors="ignore").strip()
                if text.startswith("element vertex "):
                    return int(text.rsplit(" ", 1)[-1])
                if text == "end_header":
                    break
    except (OSError, ValueError):
        return 0
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _display_name(run_id: str) -> str:
    return " ".join(
        word.capitalize()
        for word in run_id.split("_")
        if word
    )
