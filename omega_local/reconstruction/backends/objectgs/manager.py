"""Resumable ObjectGS backend consuming a completed MapAnything split."""

from __future__ import annotations

import json
import shutil
import sys
from typing import Any

from omega_local.segmentation.interactive.paths import resolve_paths
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_complete_stage,
    read_json,
)
from omega_local.stage_graph import downstream_stages

from .artifacts import export_objectgs
from .contract import (
    SCHEMA_VERSION,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    ObjectGSConfig,
    ObjectGSPaths,
)
from .prepare import prepare_objectgs_dataset
from .runner import train_objectgs
from .mesh import export_object_meshes


class ObjectGSManager:
    """Keep joint ObjectGS outputs parallel to per-region MapAnything splats."""

    def __init__(self, config: ObjectGSConfig) -> None:
        self.config = config.normalized()
        self.editor_paths = resolve_paths(
            self.config.model_dir,
            self.config.editor_baseline_name,
        )
        source_run_dir = (
            self.editor_paths.interactive_dir
            / "experiments"
            / "mapanything_region_3dgs"
            / "runs"
            / self.config.mapanything_run_id
        ).resolve()
        run_dir = (
            source_run_dir
            / "04_shared_objectgs"
            / "runs"
            / self.config.run_id
        ).resolve()
        self.paths = ObjectGSPaths(
            source_run_dir=source_run_dir,
            run_dir=run_dir,
        )

    def status(self) -> dict[str, Any]:
        stages = []
        for stage in STAGE_ORDER:
            summary = {}
            summary_path = self.paths.stage_summary(stage)
            if summary_path.is_file():
                try:
                    summary = read_json(summary_path)
                except (OSError, json.JSONDecodeError):
                    summary = {}
            stages.append(
                {
                    "stage": stage,
                    "complete": summary.get("status") == "complete",
                    "status": str(
                        summary.get("status")
                        or (
                            "invalid"
                            if summary_path.is_file()
                            else "not_ready"
                        )
                    ),
                    "summaryPath": str(summary_path),
                }
            )
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "sourceMapAnythingRun": str(self.paths.source_run_dir),
            "stages": stages,
        }

    def run_stage(
        self,
        stage: str,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        stage = str(stage).strip().lower()
        if stage == "all":
            return {
                item: self.run_stage(
                    item,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                for item in STAGE_ORDER
            }
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown ObjectGS stage: {stage}")

        self._initialize_run(stage, overwrite=overwrite)
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            return {**cached, "cacheHit": True}
        if summary_path.is_file() and not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} has a non-complete summary at "
                f"{summary_path}. Inspect it or use --overwrite-stage."
            )
        if not dry_run:
            self._require_prerequisites(stage)
        if overwrite and not dry_run:
            self._clear_stage_tree(stage)

        self._announce(f"{stage}: starting for {self.config.run_id}")
        try:
            result = getattr(self, f"_run_{stage}")(dry_run=dry_run)
            if not dry_run:
                result = {**result, "status": "complete"}
                atomic_write_json(summary_path, result)
        except Exception as exc:
            self._announce(f"{stage}: failed: {exc}", status="failed")
            raise
        self._announce(
            f"{stage}: {'planned' if dry_run else 'complete'}",
            status="planned" if dry_run else "complete",
        )
        return result

    def _run_prepare(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "prepare",
                "status": "planned",
                "sourceRunDir": str(self.paths.source_run_dir),
                "datasetDir": str(self.paths.dataset_dir),
            }
        return prepare_objectgs_dataset(
            self.config,
            self.paths,
            self.editor_paths,
            progress=self._announce,
        )

    def _run_train(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_runtime()
        return train_objectgs(
            self.config,
            self.paths,
            progress=self._stream,
            dry_run=dry_run,
        )

    def _run_export(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "export",
                "status": "planned",
                "outputsDir": str(self.paths.outputs_dir),
            }
        return export_objectgs(
            self.config,
            self.paths,
            self.editor_paths,
            progress=self._announce,
        )

    def _run_mesh(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_runtime()
        return export_object_meshes(
            self.config,
            self.paths,
            self.editor_paths,
            progress=self._stream,
            dry_run=dry_run,
        )

    def _initialize_run(self, stage: str, *, overwrite: bool) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "method": "ObjectGS joint shared-scene reconstruction",
            "runId": self.config.run_id,
            "mapanythingRunId": self.config.mapanything_run_id,
            "runDir": str(self.paths.run_dir),
            "sourceRunDir": str(self.paths.source_run_dir),
            "modelDir": str(self.config.model_dir),
            "editorBaselineName": self.config.editor_baseline_name,
            "objectgsRoot": str(self.config.objectgs_root),
            "python": str(self.config.python),
            "iterations": self.config.iterations,
            "semanticLossWeight": self.config.semantic_loss_weight,
            "initializerStride": self.config.initializer_stride,
            "voxelSize": self.config.voxel_size,
            "geometryCompletedAsUnknown": (
                self.config.geometry_completed_as_unknown
            ),
            "meshVoxelSize": self.config.mesh_voxel_size,
            "meshResolution": self.config.mesh_resolution,
            "meshClusters": self.config.mesh_clusters,
            "meshMaxTriangles": self.config.mesh_max_triangles,
            "stageOrder": list(STAGE_ORDER),
            "updatedUtc": now_utc(),
        }
        if self.paths.run_manifest.is_file():
            existing = read_json(self.paths.run_manifest)
            upstream_keys = (
                "mapanythingRunId",
                "iterations",
                "semanticLossWeight",
                "initializerStride",
                "voxelSize",
                "geometryCompletedAsUnknown",
            )
            mesh_keys = (
                "meshVoxelSize",
                "meshResolution",
                "meshClusters",
                "meshMaxTriangles",
            )
            upstream_changed = [
                key
                for key in upstream_keys
                if key in existing and existing.get(key) != manifest.get(key)
            ]
            mesh_changed = [
                key
                for key in mesh_keys
                if key in existing and existing.get(key) != manifest.get(key)
            ]
            changed = [*upstream_changed, *mesh_changed]
            if changed and not overwrite:
                raise RuntimeError(
                    f"ObjectGS run {self.config.run_id!r} changed settings: "
                    f"{', '.join(changed)}. Use --overwrite-stage from "
                    "'prepare', or choose a new --run-id."
                )
            if upstream_changed and stage != "prepare":
                raise RuntimeError(
                    "Changed ObjectGS inputs must be overwritten from "
                    f"'prepare', not {stage!r}."
                )
            if mesh_changed and stage not in {"prepare", "mesh"}:
                raise RuntimeError(
                    "Changed mesh settings must be overwritten from "
                    f"'mesh', not {stage!r}."
                )
        atomic_write_json(self.paths.run_manifest, manifest)

    def _require_prerequisites(self, stage: str) -> None:
        missing = [
            dependency
            for dependency in STAGE_DEPENDENCIES[stage]
            if read_complete_stage(
                self.paths.stage_summary(dependency)
            ) is None
        ]
        if missing:
            raise RuntimeError(
                "Run earlier ObjectGS stages first: " + ", ".join(missing)
            )

    def _require_runtime(self) -> None:
        if not self.config.python.is_file():
            raise FileNotFoundError(
                f"ObjectGS Python does not exist: {self.config.python}"
            )
        if not (self.config.objectgs_root / "train.py").is_file():
            raise FileNotFoundError(
                f"ObjectGS source does not exist: {self.config.objectgs_root}"
            )

    def _clear_stage_tree(self, stage: str) -> None:
        affected = (
            stage,
            *downstream_stages(
                stage,
                order=STAGE_ORDER,
                dependencies=STAGE_DEPENDENCIES,
            ),
        )
        for item in reversed(affected):
            directory = self.paths.stage_summary(item).parent
            if directory.exists():
                shutil.rmtree(directory)

    def _announce(
        self,
        message: str,
        *,
        status: str = "running",
    ) -> None:
        print(f"[objectgs] {message}", file=sys.stderr, flush=True)
        atomic_write_json(
            self.paths.progress,
            {
                "status": status,
                "message": message,
                "updatedUtc": now_utc(),
            },
        )

    def _stream(self, message: str) -> None:
        atomic_write_json(
            self.paths.progress,
            {
                "status": "running",
                "message": message,
                "updatedUtc": now_utc(),
            },
        )
