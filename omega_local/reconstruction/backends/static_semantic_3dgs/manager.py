"""Resumable manager for comparable hard- and soft-identity 3DGS runs."""

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

from .contract import (
    SCHEMA_VERSION,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    StaticSemantic3DGSConfig,
    StaticSemantic3DGSPaths,
)
from .export import export_static_semantic_3dgs
from .prepare import prepare_static_semantic_dataset
from .runner import train_static_semantic_3dgs, validate_runtime


class StaticSemantic3DGSManager:
    """Keep both joint static methods parallel to the MapAnything splat run."""

    def __init__(self, config: StaticSemantic3DGSConfig) -> None:
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
        family_dir = source_run_dir / "04_static_semantic_3dgs"
        run_dir = (
            family_dir
            / "runs"
            / self.config.method
            / self.config.run_id
        ).resolve()
        self.paths = StaticSemantic3DGSPaths(
            source_run_dir=source_run_dir,
            family_dir=family_dir,
            run_dir=run_dir,
        )

    def status(self) -> dict[str, Any]:
        stages = []
        for stage in STAGE_ORDER:
            summary_path = self.paths.stage_summary(stage)
            summary: dict[str, Any] = {}
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
                        or ("invalid" if summary_path.is_file() else "not_ready")
                    ),
                    "summaryPath": str(summary_path),
                }
            )
        return {
            "schemaVersion": SCHEMA_VERSION,
            "method": self.config.method,
            "displayName": self.config.display_name,
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "sourceMapAnythingRun": str(self.paths.source_run_dir),
            "sharedDataset": str(self.paths.shared_dir),
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
            raise ValueError(f"Unknown {self.config.display_name} stage: {stage}")

        self._initialize_run(stage, overwrite=overwrite)
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            return {**cached, "cacheHit": True}
        if summary_path.is_file() and not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} has a non-complete summary at {summary_path}. "
                "Inspect it or use --overwrite-stage."
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
                "sharedDataset": str(self.paths.shared_dir),
                "datasetDir": str(self.paths.dataset_dir),
            }
        return prepare_static_semantic_dataset(
            self.config,
            self.paths,
            self.editor_paths,
            progress=self._announce,
        )

    def _run_train(self, *, dry_run: bool) -> dict[str, Any]:
        if not dry_run:
            runtime = validate_runtime(self.config)
            self._announce(
                f"runtime ready: {runtime['rasterizerSignature']}"
            )
        return train_static_semantic_3dgs(
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
        return export_static_semantic_3dgs(
            self.config,
            self.paths,
            progress=self._announce,
        )

    def _initialize_run(self, stage: str, *, overwrite: bool) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "method": self.config.method,
            "displayName": self.config.display_name,
            "runId": self.config.run_id,
            "mapanythingRunId": self.config.mapanything_run_id,
            "runDir": str(self.paths.run_dir),
            "sourceRunDir": str(self.paths.source_run_dir),
            "sharedDataset": str(self.paths.shared_dir),
            "modelDir": str(self.config.model_dir),
            "editorBaselineName": self.config.editor_baseline_name,
            "methodRoot": str(self.config.method_root),
            "python": str(self.config.python),
            "iterations": self.config.iterations,
            "resolution": self.config.resolution,
            "densifyUntilIteration": self.config.densify_until_iter,
            "numSampleObjects": self.config.num_sample_objects,
            "partialMaskIoU": self.config.partial_mask_iou,
            "reg3dInterval": self.config.reg3d_interval,
            "reg3dK": self.config.reg3d_k,
            "reg3dLambda": self.config.reg3d_lambda,
            "reg3dMaxPoints": self.config.reg3d_max_points,
            "reg3dSampleSize": self.config.reg3d_sample_size,
            "nativeExtensionsDir": (
                str(self.config.native_extensions_dir)
                if self.config.native_extensions_dir is not None
                else None
            ),
            "stageOrder": list(STAGE_ORDER),
            "updatedUtc": now_utc(),
        }
        if self.paths.run_manifest.is_file():
            existing = read_json(self.paths.run_manifest)
            immutable = (
                "method",
                "mapanythingRunId",
                "iterations",
                "resolution",
                "densifyUntilIteration",
                "numSampleObjects",
                "partialMaskIoU",
                "reg3dInterval",
                "reg3dK",
                "reg3dLambda",
                "reg3dMaxPoints",
                "reg3dSampleSize",
            )
            changed = [
                key
                for key in immutable
                if key in existing and existing.get(key) != manifest.get(key)
            ]
            if changed and not overwrite:
                raise RuntimeError(
                    f"Run {self.config.run_id!r} changed settings: "
                    f"{', '.join(changed)}. Use --overwrite-stage or choose "
                    "a new --run-id."
                )
            if changed and stage != "prepare":
                raise RuntimeError(
                    "Changed static semantic 3DGS settings must be overwritten "
                    f"from 'prepare', not {stage!r}."
                )
        atomic_write_json(self.paths.run_manifest, manifest)

    def _require_prerequisites(self, stage: str) -> None:
        missing = [
            dependency
            for dependency in STAGE_DEPENDENCIES[stage]
            if read_complete_stage(self.paths.stage_summary(dependency)) is None
        ]
        if missing:
            raise RuntimeError(
                f"Run earlier {self.config.display_name} stages first: "
                + ", ".join(missing)
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

    def _announce(self, message: str, *, status: str = "running") -> None:
        print(
            f"[{self.config.method.replace('_', '-')}] {message}",
            file=sys.stderr,
            flush=True,
        )
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
