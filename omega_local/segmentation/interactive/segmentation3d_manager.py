"""Background-job manager for editor-launched 3D segmentation experiments."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .segmentation3d_contract import Segmentation3DRegistry, Segmentation3DRunRequest


_POINT_BUDGETS = {200_000, 400_000, 800_000, 0}
_SUPERPOINT_TARGETS = {4_000, 8_000, 12_000}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class Segmentation3DManager:
    def __init__(self, root: Path, registry: Segmentation3DRegistry, *, max_points: int, seed: int) -> None:
        self.root = root
        self.registry = registry
        self.max_points = max(int(max_points), 1)
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def status(self) -> dict[str, Any]:
        registry = self.registry.status()
        methods = registry["methods"]
        inputs: dict[str, list[dict[str, Any]]] = {}
        for method in methods:
            backend = self.registry.get(str(method["methodId"]))
            inputs[str(method["methodId"])] = [row.to_json() for row in backend.input_status()]
        with self._lock:
            jobs = [dict(job) for job in self._jobs.values()]
        return {
            **registry,
            "defaultInputId": "sam2_auto",
            "defaultPointBudget": 400_000,
            "defaultSuperpointTarget": 8_000,
            "pointBudgetOptions": sorted(_POINT_BUDGETS),
            "superpointTargetOptions": sorted(_SUPERPOINT_TARGETS),
            "inputs": inputs,
            "runs": self._saved_runs(),
            "jobs": jobs,
        }

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        method_id = str(payload.get("methodId") or self.registry.default_method_id).strip().lower()
        input_id = str(payload.get("inputId") or "sam2_auto").strip().lower()
        point_budget = int(payload.get("pointBudget", 400_000))
        superpoint_target = int(payload.get("superpointTarget", 8_000))
        if point_budget not in _POINT_BUDGETS:
            raise ValueError(f"Unsupported point budget: {point_budget}")
        if superpoint_target not in _SUPERPOINT_TARGETS:
            raise ValueError(f"Unsupported superpoint target: {superpoint_target}")

        backend = self.registry.get(method_id)
        if not backend.info.available:
            raise FileNotFoundError(backend.info.availability_message)
        input_row = next((row for row in backend.input_status() if row.input_id == input_id), None)
        if input_row is None:
            raise ValueError(f"Unknown {method_id} input: {input_id}")
        if not input_row.implemented or not input_row.ready:
            raise ValueError(input_row.message)

        source_id = ""
        manual_frame_weight = 1
        if input_row.source_options:
            source_id = str(payload.get("sourceId") or input_row.default_source_id).strip().lower()
            source_row = next(
                (row for row in input_row.source_options if str(row.get("sourceId")) == source_id),
                None,
            )
            if source_row is None:
                raise ValueError(f"Unknown {input_id} source: {source_id}")
            if not bool(source_row.get("ready")):
                raise ValueError(str(source_row.get("message") or f"{source_id} is not ready"))
            manual_frame_weight = int(
                payload.get("manualFrameWeight", input_row.default_manual_weight)
            )
            if manual_frame_weight not in input_row.manual_weight_options:
                raise ValueError(f"Unsupported manual-frame weight: {manual_frame_weight}")

        run_id = f"{method_id}_{input_id}"
        if source_id:
            run_id = f"{run_id}_{source_id}_w{manual_frame_weight}"
        geometry_source_id = str(backend.info.geometry_source_id or "").strip().lower()
        run_dir = self.root / "runs" / run_id
        (run_dir / "experiment.json").unlink(missing_ok=True)
        job_id = uuid.uuid4().hex
        job = {
            "jobId": job_id,
            "runId": run_id,
            "methodId": method_id,
            "inputId": input_id,
            "sourceId": source_id,
            "manualFrameWeight": manual_frame_weight,
            "geometrySourceId": geometry_source_id,
            "status": "running",
            "running": True,
            "ready": False,
            "failed": False,
            "stage": "queued",
            "stageIndex": 0,
            "stageCount": 3,
            "message": "Queued 3D segmentation experiment",
            "startedUtc": _now(),
        }
        with self._lock:
            if any(existing.get("running") for existing in self._jobs.values()):
                raise RuntimeError("Another 3D segmentation experiment is already running.")
            self._jobs[job_id] = job
        thread = threading.Thread(
            target=self._run,
            args=(
                job_id,
                backend,
                Segmentation3DRunRequest(
                    method_id,
                    input_id,
                    source_id,
                    manual_frame_weight,
                    point_budget,
                    superpoint_target,
                    run_id,
                    run_dir,
                ),
            ),
            daemon=True,
            name=f"segmentation3d-{run_id}",
        )
        thread.start()
        return dict(job)

    def job_status(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def result_points(self, run_id: str) -> dict[str, Any]:
        result_path = self.root / "runs" / str(run_id) / "experiment.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"3D segmentation result does not exist: {run_id}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        points = np.loadtxt(Path(result["pointsPath"]), dtype=np.float32)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        points = points[:, :3]
        labels = np.load(Path(result["pointLabelsPath"])).reshape(-1).astype(np.int32, copy=False)
        count = min(points.shape[0], labels.shape[0])
        points = points[:count]
        labels = labels[:count]
        if count > self.max_points:
            rng = np.random.default_rng(self.seed + 4099)
            keep = np.sort(rng.choice(count, size=self.max_points, replace=False))
            served_points = points[keep]
            served_labels = labels[keep]
        else:
            served_points = points
            served_labels = labels
        colors = _label_colors(served_labels)
        return {
            "runId": str(run_id),
            "pointCount": int(count),
            "servedPointCount": int(served_points.shape[0]),
            "labelCount": int(np.count_nonzero(np.unique(labels) > 0)),
            "positions": served_points.astype(float).reshape(-1).tolist(),
            "colors": colors.reshape(-1).tolist(),
        }

    def _run(self, job_id: str, backend, request: Segmentation3DRunRequest) -> None:
        def progress(update: dict[str, Any]) -> None:
            with self._lock:
                self._jobs[job_id].update(update)
                snapshot = dict(self._jobs[job_id])
            _write_json(request.run_dir / "job.json", snapshot)

        try:
            result = backend.run(request, progress)
        except Exception as exc:  # noqa: BLE001 - surfaced through job API
            with self._lock:
                self._jobs[job_id].update(
                    {
                        "status": "failed",
                        "running": False,
                        "ready": False,
                        "failed": True,
                        "message": str(exc),
                        "finishedUtc": _now(),
                    }
                )
                snapshot = dict(self._jobs[job_id])
            _write_json(request.run_dir / "job.json", snapshot)
            return
        with self._lock:
            self._jobs[job_id].update(
                {
                    "status": "complete",
                    "running": False,
                    "ready": True,
                    "failed": False,
                    "message": "3D segmentation result ready",
                    "finishedUtc": _now(),
                    "result": result,
                }
            )
            snapshot = dict(self._jobs[job_id])
        _write_json(request.run_dir / "job.json", snapshot)

    def _saved_runs(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted((self.root / "runs").glob("*/experiment.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            rows.append(
                {
                    "runId": str(payload.get("runId") or path.parent.name),
                    "methodId": str(payload.get("methodId") or ""),
                    "inputId": str(payload.get("inputId") or ""),
                    "sourceId": str(payload.get("sourceId") or ""),
                    "manualFrameWeight": int(payload.get("manualFrameWeight", 1)),
                    "pointBudget": int(payload.get("pointBudget", 0)),
                    "superpointTarget": int(payload.get("superpointTarget", 0)),
                    "geometrySource": str(
                        payload.get("geometrySource") or "omega_final_clean_hybrid"
                    ),
                    "ready": True,
                    "runDir": str(path.parent),
                    "timestampUtc": str(payload.get("timestampUtc") or ""),
                }
            )
        rows.sort(key=lambda row: str(row.get("timestampUtc") or ""), reverse=True)
        return rows


def _label_colors(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    colors = np.full((labels.shape[0], 3), 210, dtype=np.uint8)
    positive = labels > 0
    if not np.any(positive):
        return colors
    values = labels[positive].astype(np.uint64)
    colors[positive, 0] = ((values * 47 + 71) % 205 + 35).astype(np.uint8)
    colors[positive, 1] = ((values * 89 + 29) % 205 + 35).astype(np.uint8)
    colors[positive, 2] = ((values * 137 + 11) % 205 + 35).astype(np.uint8)
    return colors
