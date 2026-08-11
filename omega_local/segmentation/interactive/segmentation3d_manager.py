"""Background-job manager for editor-launched 3D segmentation experiments."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from omega_local.segmentation.color_palette import categorical_rgb

from .gaussian_flats_runs import discover_gaussian_flats_runs
from .reconstruction_runs import (
    discover_mapanything_runs,
    discover_objectgs_runs,
    live_mapanything_experiment,
)
from .segmentation3d_contract import Segmentation3DRegistry, Segmentation3DRunRequest
from .static_semantic_runs import (
    discover_static_semantic_3dgs_runs,
    live_static_semantic_3dgs_experiment,
)


_POINT_BUDGETS = {200_000, 400_000, 800_000, 0}
_SUPERPOINT_TARGETS = {4_000, 8_000, 12_000}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _persistent_region_metadata(interactive_root: Path) -> dict[int, dict[str, Any]]:
    path = interactive_root / "regions" / "regions.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    rows = payload.get("regions", []) if isinstance(payload, dict) else []
    metadata: dict[int, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            region_id = int(row.get("id", 0))
        except (TypeError, ValueError):
            continue
        if region_id <= 0:
            continue
        color = str(row.get("color") or "").strip().lstrip("#")
        rgb: list[int] = []
        if len(color) == 6:
            try:
                rgb = [int(color[index : index + 2], 16) for index in (0, 2, 4)]
            except ValueError:
                rgb = []
        metadata[region_id] = {
            "name": str(row.get("name") or "").strip(),
            "color": rgb,
        }
    return metadata


class Segmentation3DManager:
    def __init__(self, root: Path, registry: Segmentation3DRegistry, *, max_points: int, seed: int) -> None:
        self.root = root
        self.registry = registry
        self.max_points = max(int(max_points), 1)
        self.seed = int(seed)
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._cleanup_completed_job_files()

    def _cleanup_completed_job_files(self) -> None:
        for experiment in (self.root / "runs").glob("*/experiment.json"):
            (experiment.parent / "job.json").unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        registry = self.registry.status()
        pipeline_runs = discover_mapanything_runs(self.root.parent)
        objectgs_runs = discover_objectgs_runs(self.root.parent)
        gaussian_flats_runs = discover_gaussian_flats_runs(self.root.parent)
        static_semantic_runs = discover_static_semantic_3dgs_runs(
            self.root.parent
        )
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
            "runs": self._saved_runs(pipeline_runs) + static_semantic_runs,
            "reconstructionPipelines": pipeline_runs,
            "objectgsPipelines": objectgs_runs,
            "gaussianFlatsPipelines": gaussian_flats_runs,
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
            if run_dir.exists():
                shutil.rmtree(run_dir)
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
        result = self._run_payload(run_id)
        cache_path = Path(str(result.get("pointsCachePath") or ""))
        point_colors = None
        if cache_path.is_file():
            with np.load(cache_path) as cache:
                points = np.asarray(cache["points"], dtype=np.float32)
                labels = np.asarray(cache["labels"], dtype=np.int32).reshape(-1)
                if "colors" in cache.files:
                    cached_colors = np.asarray(cache["colors"], dtype=np.uint8)
                    if cached_colors.shape == (points.shape[0], 3):
                        point_colors = cached_colors
        else:
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
            served_colors = point_colors[keep] if point_colors is not None else None
        else:
            served_points = points
            served_labels = labels
            served_colors = point_colors
        colors = (
            served_colors
            if served_colors is not None
            else _label_colors(served_labels)
        )
        region_metadata = _persistent_region_metadata(self.root.parent)
        unique_labels, label_counts = np.unique(
            served_labels,
            return_counts=True,
        )
        regions = []
        for label, point_count in zip(
            unique_labels.tolist(),
            label_counts.tolist(),
            strict=True,
        ):
            metadata = region_metadata.get(int(label), {})
            color = list(metadata.get("color") or [])
            if len(color) != 3:
                color = _label_colors(
                    np.asarray([label], dtype=np.int32)
                )[0].tolist()
            regions.append(
                {
                    "id": int(label),
                    "name": str(
                        metadata.get("name")
                        or (f"Region {label}" if label > 0 else "Unknown")
                    ),
                    "color": color,
                    "pointCount": int(point_count),
                }
            )
        return {
            "runId": str(run_id),
            "pointCount": int(count),
            "servedPointCount": int(served_points.shape[0]),
            "labelCount": int(np.count_nonzero(np.unique(labels) > 0)),
            "positions": served_points.astype(float).reshape(-1).tolist(),
            "colors": colors.reshape(-1).tolist(),
            "labels": served_labels.astype(int).tolist(),
            "regions": regions,
        }

    def objectgs_rgb_render(self, run_id: str, frame_id: int) -> Path:
        payload = self._run_payload(run_id)
        if str(payload.get("experimentFamily") or "") != "mapanything_objectgs":
            raise FileNotFoundError(f"Run is not an ObjectGS result: {run_id}")
        canonical_value = str(payload.get("canonicalRunDir") or "").strip()
        if not canonical_value:
            raise FileNotFoundError(
                f"ObjectGS run has no canonical artifact directory: {run_id}"
            )
        canonical = Path(
            canonical_value
        ).expanduser().resolve()
        render_root = Path(
            str(
                payload.get("rgbRendersPath")
                or canonical / "04_outputs" / "rgb_renders"
            )
        ).expanduser().resolve()
        frame_map = Path(
            str(
                payload.get("frameMapPath")
                or canonical / "01_dataset" / "frame_map.jsonl"
            )
        ).expanduser().resolve()
        allowed_root = self.root.parent.resolve()
        try:
            render_root.relative_to(allowed_root)
            frame_map.relative_to(allowed_root)
        except ValueError as exc:
            raise FileNotFoundError(
                "ObjectGS render artifacts are outside the interactive dataset."
            ) from exc
        if not frame_map.is_file() or not render_root.is_dir():
            raise FileNotFoundError(
                f"ObjectGS RGB renders are not ready for {run_id}."
            )
        render_index = None
        for index, line in enumerate(
            frame_map.read_text(encoding="utf-8").splitlines()
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("frameId", -1)) == int(frame_id):
                render_index = index
                break
        if render_index is None:
            raise FileNotFoundError(
                f"Frame {frame_id} is absent from ObjectGS run {run_id}."
            )
        path = render_root / f"{render_index:05d}.png"
        if not path.is_file():
            raise FileNotFoundError(f"ObjectGS RGB render is missing: {path}")
        return path

    def gaussian_artifact(self, run_id: str, variant_id: str) -> Path:
        result = self._run_payload(run_id)
        artifacts = result.get("gaussianArtifacts")
        if not isinstance(artifacts, dict):
            raise FileNotFoundError(
                f"3D segmentation run has no Gaussian artifacts: {run_id}"
            )
        variant = artifacts.get(str(variant_id))
        if not isinstance(variant, dict):
            raise FileNotFoundError(
                f"Gaussian variant does not exist for {run_id}: {variant_id}"
            )
        path = Path(str(variant.get("path") or "")).expanduser().resolve()
        try:
            path.relative_to(self.root.parent.resolve())
        except ValueError as exc:
            raise FileNotFoundError(
                f"Gaussian artifact is outside the interactive dataset: {path}"
            ) from exc
        if path.suffix.lower() not in {".ply", ".sog"} or not path.is_file():
            raise FileNotFoundError(f"Gaussian artifact does not exist: {path}")
        return path

    def plane_mask_artifact(
        self,
        run_id: str,
        variant_id: str,
        frame_id: int,
    ) -> Path:
        result = self._run_payload(run_id)
        artifacts = result.get("planeMaskArtifacts")
        if not isinstance(artifacts, dict):
            raise FileNotFoundError(
                f"3D segmentation run has no plane-mask artifacts: {run_id}"
            )
        artifact = artifacts.get(str(variant_id))
        if not isinstance(artifact, dict):
            raise FileNotFoundError(
                f"Plane-mask variant does not exist for {run_id}: {variant_id}"
            )
        root = Path(str(artifact.get("overlayDir") or "")).expanduser().resolve()
        try:
            root.relative_to(self.root.parent.resolve())
        except ValueError as exc:
            raise FileNotFoundError(
                f"Plane-mask artifact is outside the interactive dataset: {root}"
            ) from exc
        path = root / f"{int(frame_id):06d}.png"
        if not path.is_file():
            raise FileNotFoundError(
                f"Plane-mask overlay does not exist for frame {frame_id}: {path}"
            )
        return path

    def gaussian_scene(self, run_id: str, variant_id: str) -> dict[str, Any]:
        payload = self._run_payload(run_id)
        artifacts = payload.get("gaussianArtifacts")
        if not isinstance(artifacts, dict):
            raise FileNotFoundError(
                f"3D segmentation run has no Gaussian artifacts: {run_id}"
            )
        target = artifacts.get(str(variant_id))
        if not isinstance(target, dict):
            raise FileNotFoundError(
                f"Gaussian variant does not exist for {run_id}: {variant_id}"
            )

        group = str(target.get("artifactGroup") or "")
        target_count = int(target.get("pointCount", 0))
        label_space = str(payload.get("labelSpace") or "")
        persistent_regions = (
            _persistent_region_metadata(self.root.parent)
            if label_space == "persistent_region" else {}
        )

        def grouped(group_name: str) -> list[tuple[str, dict[str, Any]]]:
            return [
                (str(key), value)
                for key, value in artifacts.items()
                if isinstance(value, dict)
                and str(value.get("artifactGroup") or "") == group_name
            ]

        def exact_group(group_name: str) -> list[tuple[str, dict[str, Any]]]:
            rows = grouped(group_name)
            count = sum(int(row[1].get("pointCount", 0)) for row in rows)
            return rows if rows and count == target_count else []

        parts: list[tuple[str, dict[str, Any]]] = []
        source_color_mode = (
            "rgb" if bool(target.get("radiometricAppearance")) else "regions"
        )
        supports_regions = source_color_mode == "regions"
        partition_group = ""
        if group == "composed_scene":
            for candidate_group in ("composed_rgb_objects", "rgb_objects"):
                candidate_parts = exact_group(candidate_group)
                if candidate_parts:
                    parts = candidate_parts
                    source_color_mode = "rgb"
                    supports_regions = True
                    partition_group = candidate_group
                    break
            if not parts:
                label_parts = exact_group("composed_objects")
                if label_parts:
                    parts = label_parts
                    source_color_mode = "regions"
                    supports_regions = True
                    partition_group = "composed_objects"
        if not parts:
            parts = [(str(variant_id), target)]
        color_modes = []
        if source_color_mode == "rgb":
            color_modes.append({"modeId": "rgb", "displayName": "RGB"})
        if supports_regions:
            color_modes.append({"modeId": "regions", "displayName": "Regions"})
        preferred_mode = (
            "rgb" if bool(target.get("radiometricAppearance")) else "regions"
        )
        default_color_mode = (
            preferred_mode
            if any(row["modeId"] == preferred_mode for row in color_modes)
            else color_modes[0]["modeId"]
        )
        parts.sort(
            key=lambda row: (
                int(row[1].get("instanceId", 0)),
                row[0],
            )
        )
        logical_parts = parts
        logical_by_id = {key: artifact for key, artifact in logical_parts}
        render_parts: list[tuple[str, dict[str, Any]]] = []
        for key, artifact in logical_parts:
            chunk_ids = [str(value) for value in artifact.get("viewerChunkIds", [])]
            chunks = [
                (chunk_id, artifacts.get(chunk_id))
                for chunk_id in chunk_ids
                if isinstance(artifacts.get(chunk_id), dict)
            ]
            if chunk_ids and len(chunks) == len(chunk_ids):
                render_parts.extend(
                    (chunk_id, dict(chunk) | {"viewerParentId": key})
                    for chunk_id, chunk in chunks
                )
            else:
                render_parts.append((key, artifact))

        def part_metadata(
            key: str,
            artifact: dict[str, Any],
            *,
            include_content: bool,
        ) -> dict[str, Any]:
            control_id = str(artifact.get("viewerParentId") or key)
            control = logical_by_id.get(control_id, artifact)
            instance_id = int(control.get("instanceId", 0))
            region_metadata = persistent_regions.get(instance_id, {})
            region_color = list(region_metadata.get("color") or [])
            if len(region_color) != 3 and supports_regions:
                region_color = (
                    categorical_rgb(np.asarray([instance_id]))[0].tolist()
                    if instance_id > 0
                    else [128, 128, 128]
                )
            row = {
                "partId": key,
                "variantId": key,
                "controlPartId": control_id,
                "displayName": str(control.get("displayName") or control_id),
                "pointCount": int(artifact.get("pointCount", 0)),
                "instanceId": instance_id,
                "colorSpace": str(control.get("colorSpace") or ""),
                "sourceColorMode": source_color_mode,
                "regionColor": region_color,
                "regionName": str(region_metadata.get("name") or ""),
            }
            if include_content:
                path = self.gaussian_artifact(run_id, key)
                stat = path.stat()
                row["contentVersion"] = f"{stat.st_size:x}-{stat.st_mtime_ns:x}"
                row["contentExtension"] = path.suffix.lower()
            return row

        controls = [
            part_metadata(key, artifact, include_content=False)
            for key, artifact in logical_parts
        ]
        scene_parts = [
            part_metadata(key, artifact, include_content=True)
            for key, artifact in render_parts
        ]
        partition_point_count = sum(int(part["pointCount"]) for part in scene_parts)
        partition_complete = bool(partition_group) and partition_point_count == target_count
        return {
            "schemaVersion": 2,
            "runId": str(run_id),
            "sceneId": str(variant_id),
            "displayName": str(target.get("displayName") or variant_id),
            "labelSpace": label_space,
            "multipart": len(controls) > 1,
            "sourceColorMode": source_color_mode,
            "colorModes": color_modes,
            "defaultColorMode": default_color_mode,
            "partCount": len(scene_parts),
            "controlCount": len(controls),
            "pointCount": partition_point_count,
            "partitionGroup": partition_group,
            "partitionComplete": partition_complete,
            "unpartitionedPointCount": 0 if partition_complete else target_count,
            "controls": controls,
            "parts": scene_parts,
        }

    def mesh_scene(self, run_id: str, variant_id: str) -> dict[str, Any]:
        payload = self._run_payload(run_id)
        artifacts = payload.get("meshArtifacts")
        if not isinstance(artifacts, dict):
            raise FileNotFoundError(
                f"3D segmentation run has no mesh artifacts: {run_id}"
            )
        target = artifacts.get(str(variant_id))
        if not isinstance(target, dict):
            raise FileNotFoundError(
                f"Mesh variant does not exist for {run_id}: {variant_id}"
            )
        manifest_path = self._dataset_artifact_path(
            target.get("manifestPath"),
            suffix=".json",
            label="Mesh manifest",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        region_metadata = _persistent_region_metadata(self.root.parent)
        parts = []
        for row in manifest.get("regions", []):
            if not isinstance(row, dict) or not str(row.get("glb") or ""):
                continue
            instance_id = int(row.get("persistentRegionId", 0))
            glb = self._dataset_artifact_path(
                row.get("glb"),
                suffix=".glb",
                label="Object mesh",
            )
            metadata = region_metadata.get(instance_id, {})
            color = list(metadata.get("color") or [])
            if len(color) != 3:
                color = categorical_rgb(np.asarray([instance_id]))[0].tolist()
            parts.append(
                {
                    "partId": f"region_{instance_id:06d}",
                    "instanceId": instance_id,
                    "displayName": str(
                        row.get("name") or f"Region {instance_id}"
                    ),
                    "regionName": str(
                        metadata.get("name") or row.get("name") or ""
                    ),
                    "regionColor": color,
                    "vertexCount": int(row.get("vertexCount", 0)),
                    "triangleCount": int(row.get("triangleCount", 0)),
                    "contentVersion": (
                        f"{glb.stat().st_size:x}-{glb.stat().st_mtime_ns:x}"
                    ),
                }
            )
        parts.sort(key=lambda row: (int(row["instanceId"]), row["partId"]))
        if not parts:
            raise FileNotFoundError(
                f"Mesh scene contains no GLB parts: {manifest_path}"
            )
        return {
            "schemaVersion": 1,
            "runId": str(run_id),
            "sceneId": str(variant_id),
            "displayName": str(target.get("displayName") or variant_id),
            "labelSpace": str(payload.get("labelSpace") or ""),
            "colorModes": [
                {"modeId": "rgb", "displayName": "RGB"},
                {"modeId": "regions", "displayName": "Regions"},
            ],
            "defaultColorMode": "rgb",
            "partCount": len(parts),
            "vertexCount": sum(int(row["vertexCount"]) for row in parts),
            "triangleCount": sum(int(row["triangleCount"]) for row in parts),
            "parts": parts,
        }

    def mesh_artifact(
        self,
        run_id: str,
        variant_id: str,
        part_id: str,
    ) -> Path:
        scene = self.mesh_scene(run_id, variant_id)
        normalized = str(part_id).strip()
        part = next(
            (row for row in scene["parts"] if str(row["partId"]) == normalized),
            None,
        )
        if part is None:
            raise FileNotFoundError(f"Mesh part does not exist: {part_id}")
        payload = self._run_payload(run_id)
        target = payload["meshArtifacts"][str(variant_id)]
        manifest_path = self._dataset_artifact_path(
            target.get("manifestPath"),
            suffix=".json",
            label="Mesh manifest",
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        instance_id = int(part["instanceId"])
        row = next(
            (
                item
                for item in manifest.get("regions", [])
                if isinstance(item, dict)
                and int(item.get("persistentRegionId", 0)) == instance_id
            ),
            None,
        )
        if row is None:
            raise FileNotFoundError(f"Mesh part source is missing: {part_id}")
        return self._dataset_artifact_path(
            row.get("glb"),
            suffix=".glb",
            label="Object mesh",
        )

    def _dataset_artifact_path(
        self,
        value: Any,
        *,
        suffix: str,
        label: str,
    ) -> Path:
        if not str(value or "").strip():
            raise FileNotFoundError(f"{label} path is missing.")
        path = Path(str(value or "")).expanduser().resolve()
        try:
            path.relative_to(self.root.parent.resolve())
        except ValueError as exc:
            raise FileNotFoundError(
                f"{label} is outside the interactive dataset: {path}"
            ) from exc
        if path.suffix.lower() != suffix or not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return path

    def _run_payload(self, run_id: str) -> dict[str, Any]:
        normalized = str(run_id).strip()
        if not normalized or Path(normalized).name != normalized:
            raise FileNotFoundError(f"Invalid 3D segmentation run ID: {run_id}")
        result_path = self.root / "runs" / normalized / "experiment.json"
        if result_path.is_file():
            return json.loads(result_path.read_text(encoding="utf-8"))
        live = live_mapanything_experiment(self.root.parent, normalized)
        if live is not None:
            return live
        static_semantic = live_static_semantic_3dgs_experiment(
            self.root.parent,
            normalized,
        )
        if static_semantic is not None:
            return static_semantic
        raise FileNotFoundError(
            f"3D segmentation result does not exist: {run_id}"
        )

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
        (request.run_dir / "job.json").unlink(missing_ok=True)

    def _saved_runs(
        self,
        pipeline_runs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted((self.root / "runs").glob("*/experiment.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            gaussian_artifacts = []
            mesh_artifacts = []
            plane_mask_artifacts = []
            raw_artifacts = payload.get("gaussianArtifacts")
            point_summary = (
                payload.get("pointSummary")
                if isinstance(payload.get("pointSummary"), dict)
                else {}
            )
            if isinstance(raw_artifacts, dict):
                for variant_id, artifact in raw_artifacts.items():
                    if not isinstance(artifact, dict):
                        continue
                    if str(artifact.get("artifactGroup") or "") == "viewer_chunks":
                        continue
                    artifact_path = Path(str(artifact.get("path") or ""))
                    if not artifact_path.is_file():
                        continue
                    point_count = int(artifact.get("pointCount", 0))
                    label_count = int(artifact.get("labelCount", 0))
                    labeled_point_count = artifact.get("labeledPointCount")
                    if labeled_point_count is None:
                        labeled_point_count = point_summary.get(
                            "labeledGlobalPointCount"
                        )
                    if (
                        labeled_point_count is None
                        and str(variant_id) in {"instance_ids", "segments"}
                        and label_count > 0
                    ):
                        labeled_point_count = point_count
                    gaussian_artifacts.append(
                        {
                            "variantId": str(variant_id),
                            "displayName": str(
                                artifact.get("displayName") or variant_id
                            ),
                            "format": str(artifact.get("format") or "3dgs_ply"),
                            "pointCount": point_count,
                            "labelCount": label_count,
                            "labeledPointCount": int(labeled_point_count or 0),
                            "colorSpace": str(
                                artifact.get("colorSpace")
                                or (
                                    "split_splat_instance"
                                    if str(variant_id) == "segments"
                                    else ""
                                )
                            ),
                            "unlabeledColor": str(
                                artifact.get("unlabeledColor")
                                or (
                                    "neutral_gray"
                                    if str(variant_id) == "segments"
                                    else ""
                                )
                            ),
                            "artifactGroup": str(
                                artifact.get("artifactGroup") or ""
                            ),
                            "instanceId": int(artifact.get("instanceId", 0)),
                            "radiometricAppearance": bool(
                                artifact.get("radiometricAppearance")
                            ),
                            "contentVersion": (
                                f"{artifact_path.stat().st_size:x}-"
                                f"{artifact_path.stat().st_mtime_ns:x}"
                            ),
                        }
                    )
            raw_mesh_artifacts = payload.get("meshArtifacts")
            if isinstance(raw_mesh_artifacts, dict):
                for variant_id, artifact in raw_mesh_artifacts.items():
                    if not isinstance(artifact, dict):
                        continue
                    manifest_path = Path(str(artifact.get("manifestPath") or ""))
                    if not manifest_path.is_file():
                        continue
                    mesh_artifacts.append(
                        {
                            "variantId": str(variant_id),
                            "displayName": str(
                                artifact.get("displayName") or variant_id
                            ),
                            "partCount": int(artifact.get("partCount", 0)),
                            "vertexCount": int(artifact.get("vertexCount", 0)),
                            "triangleCount": int(artifact.get("triangleCount", 0)),
                        }
                    )
            raw_plane_masks = payload.get("planeMaskArtifacts")
            if isinstance(raw_plane_masks, dict):
                for variant_id, artifact in raw_plane_masks.items():
                    if not isinstance(artifact, dict):
                        continue
                    overlay_dir = Path(str(artifact.get("overlayDir") or ""))
                    if not overlay_dir.is_dir():
                        continue
                    plane_mask_artifacts.append(
                        {
                            "variantId": str(variant_id),
                            "displayName": str(
                                artifact.get("displayName") or variant_id
                            ),
                            "frameCount": int(artifact.get("frameCount", 0)),
                            "nonemptyFrameCount": int(
                                artifact.get("nonemptyFrameCount", 0)
                            ),
                            "regionId": int(artifact.get("regionId", 0)),
                            "regionName": str(artifact.get("regionName") or ""),
                            "contentVersion": f"{overlay_dir.stat().st_mtime_ns:x}",
                        }
                    )
            experiment_family = str(payload.get("experimentFamily") or "")
            canonical_run_dir = str(payload.get("canonicalRunDir") or "")
            base_run_id = str(payload.get("baseRunId") or "")
            objectgs_run_id = str(payload.get("objectgsRunId") or "")
            if experiment_family == "mapanything_objectgs" and not objectgs_run_id:
                objectgs_run_id = str(
                    payload.get("runId") or path.parent.name
                )
                prefix = f"{base_run_id}_"
                if base_run_id and objectgs_run_id.startswith(prefix):
                    objectgs_run_id = objectgs_run_id[len(prefix) :]
                if objectgs_run_id.endswith("_anchors"):
                    objectgs_run_id = objectgs_run_id[: -len("_anchors")]
            rgb_render_value = str(payload.get("rgbRendersPath") or "")
            if not rgb_render_value and canonical_run_dir:
                rgb_render_value = str(
                    Path(canonical_run_dir) / "04_outputs" / "rgb_renders"
                )
            rgb_render_root = (
                Path(rgb_render_value) if rgb_render_value else None
            )
            mask_layer_key = str(payload.get("maskLayerKey") or "")
            if (
                experiment_family == "mapanything_objectgs"
                and not mask_layer_key
                and base_run_id
                and objectgs_run_id
            ):
                mask_layer_key = (
                    "propagation_"
                    f"{base_run_id}_{objectgs_run_id}_semantic"
                )
            rows.append(
                {
                    "runId": str(payload.get("runId") or path.parent.name),
                    "methodId": str(payload.get("methodId") or ""),
                    "experimentFamily": experiment_family,
                    "artifactRole": str(payload.get("artifactRole") or ""),
                    "baseRunId": base_run_id,
                    "reconstructionVariant": str(
                        payload.get("reconstructionVariant") or ""
                    ),
                    "displayName": str(payload.get("displayName") or ""),
                    "inputId": str(payload.get("inputId") or ""),
                    "sourceId": str(payload.get("sourceId") or ""),
                    "labelSpace": str(payload.get("labelSpace") or ""),
                    "maskRefinement": str(payload.get("maskRefinement") or ""),
                    "canonicalRunDir": canonical_run_dir,
                    "objectgsRunId": objectgs_run_id,
                    "rgbRendersReady": bool(
                        experiment_family == "mapanything_objectgs"
                        and rgb_render_root is not None
                        and rgb_render_root.is_dir()
                    ),
                    "maskLayerKey": mask_layer_key,
                    "manualFrameWeight": int(payload.get("manualFrameWeight", 1)),
                    "pointBudget": int(payload.get("pointBudget", 0)),
                    "superpointTarget": int(payload.get("superpointTarget", 0)),
                    "pointCount": int(
                        point_summary.get("globalPointCount")
                        or point_summary.get("labeledPointCount")
                        or 0
                    ),
                    "geometrySource": str(
                        payload.get("geometrySource") or "omega_final_clean_hybrid"
                    ),
                    "ready": True,
                    "runDir": str(path.parent),
                    "timestampUtc": str(payload.get("timestampUtc") or ""),
                    "gaussianArtifacts": gaussian_artifacts,
                    "meshArtifacts": mesh_artifacts,
                    "planeMaskArtifacts": plane_mask_artifacts,
                }
            )
        existing_run_ids = {str(row["runId"]) for row in rows}
        discovered = (
            pipeline_runs
            if pipeline_runs is not None
            else discover_mapanything_runs(self.root.parent)
        )
        for pipeline in discovered:
            run_id = f"{pipeline['runId']}_splat"
            if run_id in existing_run_ids or not pipeline["completedRegions"]:
                continue
            gaussian_artifacts = [
                {
                    "variantId": f"rgb_region_{item['regionId']}",
                    "displayName": f"Region {item['regionId']} RGB",
                    "format": "3dgs_ply",
                    "pointCount": int(item["gaussianCount"]),
                    "labelCount": 0,
                    "labeledPointCount": 0,
                    "colorSpace": "rgb_sh",
                    "unlabeledColor": "",
                    "artifactGroup": "rgb_objects",
                    "instanceId": int(item["regionId"]),
                }
                for item in pipeline["completedRegions"]
            ]
            rows.append(
                {
                    "runId": run_id,
                    "methodId": "mapanything_region_3dgs",
                    "experimentFamily": "mapanything_region_3dgs",
                    "artifactRole": "reconstruction",
                    "baseRunId": str(pipeline["runId"]),
                    "reconstructionVariant": (
                        "mapanything_point_initialized_3dgs"
                    ),
                    "displayName": (
                        f"MapAnything Region 3DGS: {pipeline['runId']}"
                    ),
                    "inputId": str(pipeline["propagationMethod"]),
                    "sourceId": "",
                    "labelSpace": "persistent_region",
                    "maskRefinement": "none",
                    "canonicalRunDir": str(pipeline["runDir"]),
                    "manualFrameWeight": int(
                        pipeline["manualFrameWeight"]
                    ),
                    "pointBudget": 0,
                    "superpointTarget": 0,
                    "geometrySource": "omega_mapanything_initializer",
                    "ready": True,
                    "runDir": str(pipeline["runDir"]),
                    "timestampUtc": str(pipeline["updatedUtc"]),
                    "gaussianArtifacts": gaussian_artifacts,
                }
            )
        rows.sort(key=lambda row: str(row.get("timestampUtc") or ""), reverse=True)
        return rows


def _label_colors(labels: np.ndarray) -> np.ndarray:
    return categorical_rgb(labels, background=(210, 210, 210))
