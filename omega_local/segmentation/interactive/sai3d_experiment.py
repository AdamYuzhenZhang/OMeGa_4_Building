"""SAI3D experiment backend for the interactive segmentation editor."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .omega_mesh_point_cloud import (
    OmegaMeshHybridPointCloudConfig,
    ensure_omega_mesh_hybrid_point_cloud,
)
from .paths import EditorPaths, OMEGA_ROOT, read_jsonl
from .segmentation3d_contract import (
    ProgressCallback,
    Segmentation3DInputInfo,
    Segmentation3DMethodInfo,
    Segmentation3DRunRequest,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class SAI3DExperimentConfig:
    paths: EditorPaths
    sai3d_root: Path
    python: Path = Path(sys.executable)
    workers: int = 20
    seed: int = 71


class SAI3DExperimentBackend:
    method_id = "sai3d"
    automatic_input_id = "sam2_auto"
    weighted_input_id = "propagated_weighted"
    manual_weight_options = (2, 4, 8)
    default_manual_weight = 4

    def __init__(self, config: SAI3DExperimentConfig) -> None:
        self.config = config

    @property
    def info(self) -> Segmentation3DMethodInfo:
        root = self.config.sai3d_root.expanduser().resolve()
        available = (root / "sai3d_base.py").is_file() and (root / "helpers" / "sai3d_utils.py").is_file()
        return Segmentation3DMethodInfo(
            method_id=self.method_id,
            display_name="SAI3D",
            description="Progressive 3D superpoint grouping from multi-view 2D mask agreement.",
            available=available,
            availability_message="Ready" if available else f"SAI3D source is incomplete: {root}",
            geometry_source_id="omega_final_clean_hybrid",
            geometry_source_name="Clean Hybrid",
        )

    def input_status(self) -> list[Segmentation3DInputInfo]:
        sam2_dir = self._sam2_run_dir()
        manifest_rows = read_jsonl(self.config.paths.frame_manifest)
        frame_ids = [
            int(row.get("sai3dFrameId", row.get("sourceFrameId", index)))
            for index, row in enumerate(manifest_rows)
        ]
        ready_count, auto_ready = _label_map_status(sam2_dir, frame_ids)
        complete_frame_ids = self._complete_frame_ids()
        propagation_sources = tuple(self._propagation_sources(frame_ids))
        ready_sources = [row for row in propagation_sources if bool(row.get("ready"))]
        return [
            Segmentation3DInputInfo(
                input_id=self.automatic_input_id,
                display_name="SAM2 Automatic",
                description="Original SAI3D input: independent SAM2 automatic proposals in every view.",
                implemented=True,
                ready=auto_ready,
                message=(
                    f"{ready_count}/{len(frame_ids)} frame masks ready"
                    if auto_ready
                    else f"Generate SAM2 proposals first ({ready_count}/{len(frame_ids)} ready)"
                ),
            ),
            Segmentation3DInputInfo(
                input_id=self.weighted_input_id,
                display_name="Propagated + Manual Anchors",
                description=(
                    "A selected persistent-region propagation, with completed manual keyframes "
                    "substituted and repeated as stronger SAI3D observations."
                ),
                implemented=True,
                ready=bool(ready_sources and complete_frame_ids),
                message=(
                    f"{len(ready_sources)} propagation source(s); "
                    f"{len(complete_frame_ids)} completed manual frame(s)"
                    if ready_sources and complete_frame_ids
                    else "Run a full propagation and mark at least one edited frame complete."
                ),
                source_options=propagation_sources,
                default_source_id="sam2_video",
                manual_weight_options=self.manual_weight_options,
                default_manual_weight=self.default_manual_weight,
            ),
        ]

    def run(
        self,
        request: Segmentation3DRunRequest,
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        if not self.info.available:
            raise FileNotFoundError(self.info.availability_message)
        if request.input_id not in {self.automatic_input_id, self.weighted_input_id}:
            raise ValueError(f"Unknown SAI3D input: {request.input_id}")
        input_info = next(row for row in self.input_status() if row.input_id == request.input_id)
        if not input_info.ready:
            raise FileNotFoundError(input_info.message)
        if request.input_id == self.weighted_input_id:
            source = next(
                (row for row in input_info.source_options if row.get("sourceId") == request.source_id),
                None,
            )
            if source is None or not bool(source.get("ready")):
                message = source.get("message") if source else f"Unknown propagation source: {request.source_id}"
                raise FileNotFoundError(str(message))

        progress({"stage": "geometry", "message": "Preparing cleaned hybrid strongest-mesh points", "stageIndex": 0, "stageCount": 3})
        geometry_summary = ensure_omega_mesh_hybrid_point_cloud(
            OmegaMeshHybridPointCloudConfig(
                mesh_path=self.config.paths.omega_final_mesh_source,
                points_path=self.config.paths.omega_final_hybrid_points,
                cache_path=self.config.paths.omega_final_hybrid_cache,
                summary_path=self.config.paths.omega_final_hybrid_summary,
                surface_cache_path=self.config.paths.omega_final_surface_cache,
                surface_summary_path=self.config.paths.omega_final_surface_summary,
                seed=self.config.seed,
            )
        )
        proposal_source = (
            self._stage_automatic_input()
            if request.input_id == self.automatic_input_id
            else self._stage_weighted_input(request)
        )

        request.run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = request.run_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        common = self._command_args(request, proposal_source)
        stages = (
            ("prepare", "Staging SAI3D dataset", 1),
            ("segment", "Solving SAI3D 3D labels", 2),
        )
        commands: list[list[str]] = []
        for stage, message, stage_index in stages:
            progress(
                {
                    "stage": stage,
                    "message": message,
                    "stageIndex": stage_index,
                    "stageCount": 3,
                }
            )
            command = [*common, "--stage", stage]
            commands.append(command)
            log_path = log_dir / f"{stage}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    cwd=OMEGA_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if process.returncode != 0:
                tail = _tail(log_path)
                raise RuntimeError(f"SAI3D {stage} failed (exit {process.returncode}).\n{tail}")

        summary_path = request.run_dir / "baseline_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"SAI3D completed without a summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        point_labels = request.run_dir / "mesh_labels" / "point_labels.npy"
        points = next(iter(sorted((request.run_dir / "dataset" / "scans").glob("*/points.pts"))), None)
        if points is None or not point_labels.is_file():
            raise RuntimeError("SAI3D completed without point coordinates or point labels.")

        result = {
            "schemaVersion": 1,
            "timestampUtc": _now(),
            "runId": request.run_id,
            "methodId": request.method_id,
            "inputId": request.input_id,
            "sourceId": request.source_id,
            "manualFrameWeight": int(request.manual_frame_weight),
            "pointBudget": int(request.point_budget),
            "superpointTarget": int(request.superpoint_target),
            "geometrySource": "omega_final_clean_hybrid",
            "geometrySummary": geometry_summary,
            "runDir": str(request.run_dir),
            "summaryPath": str(summary_path),
            "pointsPath": str(points),
            "pointLabelsPath": str(point_labels),
            "labeledPointsPath": str(request.run_dir / "mesh_labels" / "labeled_points.ply"),
            "commands": commands,
            "summary": summary,
        }
        _write_json(request.run_dir / "experiment.json", result)
        progress({"stage": "complete", "message": "SAI3D 3D labels ready", "stageIndex": 3, "stageCount": 3})
        return result

    def _sam2_run_dir(self) -> Path:
        rows = read_jsonl(self.config.paths.frame_manifest)
        width = int(rows[0]["width"]) if rows else 0
        return self.config.paths.interactive_dir / "proposals" / f"sam2_auto_{width}"

    def _stage_automatic_input(self) -> Path:
        source_dir = self.config.paths.segmentation3d_dir / "inputs" / self.automatic_input_id
        mask_dir = self._sam2_run_dir() / "label_maps"
        rows = self._input_rows()
        manifest_path = source_dir / "frames.jsonl"
        _write_jsonl(manifest_path, rows)
        _write_json(
            source_dir / "baseline_summary.json",
            {
                "schemaVersion": 1,
                "stage": "interactive_3d_segmentation_input",
                "timestampUtc": _now(),
                "inputId": self.automatic_input_id,
                "frameCount": len(rows),
                "outputs": {
                    "frameManifest": str(manifest_path),
                    "viewMaskMasksDir": str(mask_dir),
                },
            },
        )
        return source_dir

    def _stage_weighted_input(self, request: Segmentation3DRunRequest) -> Path:
        source_dir = (
            self.config.paths.segmentation3d_dir
            / "inputs"
            / f"{self.weighted_input_id}_{request.source_id}_w{request.manual_frame_weight}"
        )
        mask_dir = source_dir / "label_maps"
        if mask_dir.exists():
            shutil.rmtree(mask_dir)
        mask_dir.mkdir(parents=True, exist_ok=True)

        propagation_dir = (
            self.config.paths.interactive_dir
            / "proposals"
            / "propagation"
            / request.source_id
            / "label_maps"
        )
        complete_frame_ids = set(self._complete_frame_ids())
        rows = self._input_rows()
        manual_count = 0
        for row in rows:
            frame_id = int(row["frameId"])
            propagated_path = propagation_dir / f"{frame_id:06d}.npy"
            manual_path = self.config.paths.region_maps_dir / f"{frame_id:06d}.npy"
            is_manual = frame_id in complete_frame_ids and manual_path.is_file()
            source_path = manual_path if is_manual else propagated_path
            if not source_path.is_file():
                raise FileNotFoundError(f"Missing SAI3D input mask for frame {frame_id}: {source_path}")
            target_path = mask_dir / f"{frame_id:06d}.npy"
            target_path.symlink_to(source_path.resolve())
            row["viewWeight"] = int(request.manual_frame_weight if is_manual else 1)
            row["isManualAnchor"] = bool(is_manual)
            row["maskSource"] = "manual_region" if is_manual else request.source_id
            manual_count += int(is_manual)

        manifest_path = source_dir / "frames.jsonl"
        _write_jsonl(manifest_path, rows)
        _write_json(
            source_dir / "baseline_summary.json",
            {
                "schemaVersion": 1,
                "stage": "interactive_3d_segmentation_input",
                "timestampUtc": _now(),
                "inputId": self.weighted_input_id,
                "sourceId": request.source_id,
                "manualFrameWeight": int(request.manual_frame_weight),
                "frameCount": len(rows),
                "manualFrameCount": int(manual_count),
                "weightingPolicy": "integer_view_replication_in_sai3d_affinity",
                "outputs": {
                    "frameManifest": str(manifest_path),
                    "viewMaskMasksDir": str(mask_dir),
                },
            },
        )
        return source_dir

    def _input_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for fallback, source in enumerate(read_jsonl(self.config.paths.frame_manifest)):
            frame_id = int(source.get("sai3dFrameId", source.get("sourceFrameId", fallback)))
            color_path = (self.config.paths.dataset_dir / str(source["colorPath"])).resolve()
            pose_path = (self.config.paths.dataset_dir / str(source["posePath"])).resolve()
            rows.append(
                {
                    "sourceFrameId": frame_id,
                    "frameId": frame_id,
                    "imageName": str(source.get("imageName", f"{frame_id:06d}.jpg")),
                    "colorPath": str(color_path),
                    "posePath": str(pose_path),
                    "width": int(source["width"]),
                    "height": int(source["height"]),
                    "fx": float(source["fx"]),
                    "fy": float(source["fy"]),
                    "cx": float(source["cx"]),
                    "cy": float(source["cy"]),
                    "viewWeight": 1,
                    "isManualAnchor": False,
                }
            )
        return rows

    def _complete_frame_ids(self) -> list[int]:
        path = self.config.paths.regions_summary
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        states = payload.get("frameStates", {})
        return sorted(
            int(frame_id)
            for frame_id, row in states.items()
            if str(frame_id).isdigit()
            and bool(row.get("complete"))
            and (self.config.paths.region_maps_dir / f"{int(frame_id):06d}.npy").is_file()
        )

    def _propagation_sources(self, frame_ids: list[int]) -> list[dict[str, Any]]:
        root = self.config.paths.interactive_dir / "proposals" / "propagation"
        registry_path = root / "registry.json"
        anchor_updated_ns = (
            self.config.paths.regions_summary.stat().st_mtime_ns
            if self.config.paths.regions_summary.is_file()
            else 0
        )
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            registry = {"methods": []}
        sources: list[dict[str, Any]] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        for method in registry.get("methods", []):
            if not bool(method.get("supportsFullRun")):
                continue
            source_id = str(method.get("methodId") or "").strip().lower()
            if not source_id:
                continue
            run_dir = root / source_id
            ready_count, ready = _label_map_status(run_dir, frame_ids)
            stale = bool(
                ready
                and anchor_updated_ns
                and (run_dir / "summary.json").stat().st_mtime_ns < anchor_updated_ns
            )
            ready = bool(ready and not stale)
            row = {
                "sourceId": source_id,
                "displayName": str(method.get("displayName") or source_id),
                "ready": bool(ready),
                "message": (
                    "Manual anchors changed; rerun this propagation."
                    if stale
                    else f"{ready_count}/{len(frame_ids)} masks ready"
                    if ready
                    else f"Run this propagation first ({ready_count}/{len(frame_ids)} masks)"
                ),
                "_sourceMethodId": str(method.get("sourceMethodId") or ""),
                "_updatedNs": (
                    (run_dir / "summary.json").stat().st_mtime_ns
                    if (run_dir / "summary.json").is_file()
                    else 0
                ),
            }
            sources.append(row)
            source_by_id[source_id] = row

        for row in sources:
            source_method_id = str(row.get("_sourceMethodId") or "")
            updated_ns = int(row.get("_updatedNs", 0))
            if not source_method_id:
                continue
            source = source_by_id.get(source_method_id)
            source_ready = bool(source and source.get("ready"))
            source_updated_ns = int(source.get("_updatedNs", 0)) if source else 0
            if not source_ready:
                row["ready"] = False
                row["message"] = "Required source propagation is not current."
            elif updated_ns < source_updated_ns:
                row["ready"] = False
                row["message"] = "Source propagation changed; rerun this recovery."
        for row in sources:
            row.pop("_sourceMethodId", None)
            row.pop("_updatedNs", None)
        return sources

    def _command_args(self, request: Segmentation3DRunRequest, proposal_source: Path) -> list[str]:
        return [
            str(self.config.python),
            str(OMEGA_ROOT / "scripts" / "run_omega_segmentation_baselines.py"),
            "--baseline",
            "sai3d",
            "--model-dir",
            str(self.config.paths.model_dir),
            "--output-dir",
            str(request.run_dir),
            "--baseline-name",
            request.run_id,
            "--scene-name",
            f"{self.config.paths.model_dir.name}_omega_final",
            "--proposal-source-name",
            request.input_id,
            "--proposal-source-dir",
            str(proposal_source),
            "--sai3d-root",
            str(self.config.sai3d_root),
            "--mesh",
            str(self.config.paths.omega_final_mesh_source),
            "--point-source",
            "point_cloud",
            "--point-cloud",
            str(self.config.paths.omega_final_hybrid_points),
            "--point-cloud-max-points",
            str(request.point_budget),
            "--point-cloud-sample-seed",
            str(self.config.seed),
            "--point-visibility-source",
            "point_zbuffer",
            "--point-zbuffer-depth-band",
            "0.02",
            "--superpoint-mode",
            "voxel",
            "--superpoint-target-count",
            str(request.superpoint_target),
            "--sai3d-view-stride",
            "1",
            "--thres-connect",
            "0.9,0.5,5",
            "--thres-merge",
            "200",
            "--max-neighbor-distance",
            "2",
            "--similar-metric",
            "2-norm",
            "--k-graph",
            "8",
            "--sai3d-workers",
            str(self.config.workers),
            "--overwrite",
        ]


def _tail(path: Path, line_count: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"Could not read log: {path}"
    return "\n".join(lines[-line_count:])


def _label_map_status(run_dir: Path, frame_ids: list[int]) -> tuple[int, bool]:
    label_dir = run_dir / "label_maps"
    ready_count = sum(
        1 for frame_id in frame_ids if (label_dir / f"{int(frame_id):06d}.npy").is_file()
    )
    return ready_count, bool(
        frame_ids
        and ready_count == len(frame_ids)
        and (run_dir / "summary.json").is_file()
    )
