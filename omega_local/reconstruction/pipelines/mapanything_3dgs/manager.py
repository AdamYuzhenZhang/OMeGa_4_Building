"""Resumable stage manager for MapAnything-carried region 3DGS."""

from __future__ import annotations

import json
import shutil
import sys
from typing import Any

from omega_local.stage_graph import downstream_stages
from omega_local.segmentation.interactive.paths import resolve_paths
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_complete_stage,
    read_json,
)

from .artifacts import register_splat_artifacts, register_split_artifacts
from .contract import (
    SCHEMA_VERSION,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    MapAnything3DGSConfig,
    MapAnything3DGSPaths,
)
from .dataset import prepare_input_dataset
from .splat import (
    compose_regions,
    prepare_region_datasets,
    region_storage_cleanup,
    train_regions,
)
from .split import run_split


class MapAnything3DGSManager:
    """Run the custom pipeline without touching Split&Splat experiment roots."""

    def __init__(self, config: MapAnything3DGSConfig) -> None:
        self.config = config.normalized()
        self.editor_paths = resolve_paths(
            self.config.model_dir,
            self.config.editor_baseline_name,
            feedforward_point_cloud=self.config.point_cloud,
        )
        point_cloud = (
            self.config.point_cloud
            if self.config.point_cloud is not None
            else self.editor_paths.feedforward_source
        )
        run_dir = (
            self.editor_paths.interactive_dir
            / "experiments"
            / "mapanything_region_3dgs"
            / "runs"
            / self.config.run_id
        ).resolve()
        self.paths = MapAnything3DGSPaths(
            run_dir=run_dir,
            point_cloud=point_cloud.resolve(),
        )

    def status(self) -> dict[str, Any]:
        stages = []
        for index, stage in enumerate(STAGE_ORDER):
            summary_path = self.paths.stage_summary(stage)
            payload = {}
            if summary_path.is_file():
                try:
                    payload = read_json(summary_path)
                except (OSError, json.JSONDecodeError):
                    payload = {}
            stages.append(
                {
                    "stage": stage,
                    "index": index,
                    "complete": payload.get("status") == "complete",
                    "status": str(
                        payload.get("status")
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
            "pointCloud": str(self.paths.point_cloud),
            "propagationMethod": self.config.propagation_method,
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
        groups = {
            "all_split": ("prepare", "split"),
            "all_splat": (
                "splat_prepare",
                "splat_train",
                "splat_compose",
            ),
            "all": STAGE_ORDER,
        }
        if stage in groups:
            return {
                item: self.run_stage(
                    item,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                for item in groups[stage]
            }
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown MapAnything stage: {stage}")

        self._initialize_run(stage, overwrite=overwrite)
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            self._refresh_registration(stage, cached)
            if stage == "splat_train" and not dry_run:
                cached = self._cleanup_training(cached)
            return {**cached, "cacheHit": True}
        if summary_path.is_file() and not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} has a non-complete summary at {summary_path}. "
                "Inspect it or rerun with --overwrite-stage."
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
                if stage == "splat_train":
                    result = self._cleanup_training(result)
        except Exception as exc:
            self._announce(
                f"{stage}: failed: {exc}",
                status="failed",
            )
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
                "datasetDir": str(self.paths.dataset_dir),
                "pointCloud": str(self.paths.point_cloud),
            }
        return prepare_input_dataset(
            self.config,
            self.editor_paths,
            self.paths,
            progress=self._announce,
        )

    def _run_split(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "split",
                "status": "planned",
                "pointCloud": str(self.paths.point_cloud),
                "propagationMethod": self.config.propagation_method,
                "manualFrameWeight": self.config.manual_frame_weight,
            }
        return run_split(
            self.config,
            self.editor_paths,
            self.paths,
            progress=self._announce,
        )

    def _run_splat_prepare(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "splat_prepare",
                "status": "planned",
                "datasetsDir": str(self.paths.region_datasets),
            }
        return prepare_region_datasets(
            self.config,
            self.paths,
            progress=self._announce,
        )

    def _run_splat_train(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_trainer()
        return train_regions(
            self.config,
            self.paths,
            progress=self._announce,
            command_progress=self._stream,
            dry_run=dry_run,
        )

    def _run_splat_compose(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "splat_compose",
                "status": "planned",
                "output": str(self.paths.composed_scene),
            }
        return compose_regions(
            self.config,
            self.paths,
            self.editor_paths,
        )

    def _initialize_run(self, stage: str, *, overwrite: bool) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "method": "MapAnything persistent-region Split and 3DGS",
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "modelDir": str(self.config.model_dir),
            "editorBaselineName": self.config.editor_baseline_name,
            "pointCloud": str(self.paths.point_cloud),
            "propagationMethod": self.config.propagation_method,
            "manualFrameWeight": self.config.manual_frame_weight,
            "instanceIterations": self.config.instance_iterations,
            "minimumRegionPoints": self.config.minimum_region_points,
            "minimumPositiveViews": self.config.minimum_positive_views,
            "maxNumSplats": self.config.max_num_splats,
            "maskLossWeight": self.config.mask_loss_weight,
            "floaterPruning": self.config.floater_pruning,
            "floaterOpacityThreshold": (
                self.config.floater_opacity_threshold
            ),
            "floaterMinVisibleViews": (
                self.config.floater_min_visible_views
            ),
            "floaterMaskDilationPixels": (
                self.config.floater_mask_dilation_pixels
            ),
            "splitSplatRoot": str(self.config.split_splat_root),
            "stageOrder": list(STAGE_ORDER),
            "updatedUtc": now_utc(),
        }
        if self.paths.run_manifest.is_file():
            existing = read_json(self.paths.run_manifest)
            first_affected_stage = {
                "runId": "prepare",
                "pointCloud": "prepare",
                "propagationMethod": "split",
                "manualFrameWeight": "split",
                "instanceIterations": "splat_train",
                "minimumRegionPoints": "splat_prepare",
                "minimumPositiveViews": "splat_prepare",
                "maxNumSplats": "splat_train",
                "maskLossWeight": "splat_train",
                "floaterPruning": "splat_train",
                "floaterOpacityThreshold": "splat_train",
                "floaterMinVisibleViews": "splat_train",
                "floaterMaskDilationPixels": "splat_train",
            }
            changed = [
                key
                for key in first_affected_stage
                if key in existing and existing.get(key) != manifest.get(key)
            ]
            if changed:
                stage_index = STAGE_ORDER.index(stage)
                unsafe = [
                    key
                    for key in changed
                    if (
                        not overwrite
                        or stage_index
                        > STAGE_ORDER.index(first_affected_stage[key])
                    )
                ]
                if unsafe:
                    required = min(
                        (first_affected_stage[key] for key in unsafe),
                        key=STAGE_ORDER.index,
                    )
                    changed_text = ", ".join(changed)
                    raise RuntimeError(
                        f"Run ID {self.config.run_id!r} has changed settings: "
                        f"{changed_text}. Rerun from {required!r} with "
                        "--overwrite-stage, or choose a new --run-id."
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
                "Run earlier MapAnything stages first: " + ", ".join(missing)
            )

    def _require_trainer(self) -> None:
        train = self.config.split_splat_root / "train.py"
        if not train.is_file():
            raise FileNotFoundError(
                f"Masked 3DGS trainer does not exist: {train}"
            )

    def _clear_stage(self, stage: str) -> None:
        directory = self.paths.stage_summary(stage).parent
        if directory.exists():
            shutil.rmtree(directory)

    def _clear_stage_tree(self, stage: str) -> None:
        """Clear a stage and every cached result that depends on it."""
        affected = (
            stage,
            *downstream_stages(
                stage,
                order=STAGE_ORDER,
                dependencies=STAGE_DEPENDENCIES,
            ),
        )
        for item in reversed(affected):
            self._clear_stage(item)

    def _refresh_registration(
        self,
        stage: str,
        summary: dict[str, Any],
    ) -> None:
        if stage == "split":
            register_split_artifacts(
                self.config,
                self.paths,
                self.editor_paths,
                summary,
            )
        elif stage == "splat_compose":
            register_splat_artifacts(
                self.config,
                self.paths,
                self.editor_paths,
                summary,
            )

    def _cleanup_training(self, result: dict[str, Any]) -> dict[str, Any]:
        cleanup = region_storage_cleanup(
            self.paths,
            final_iteration=self.config.instance_iterations,
        )
        refreshed = {**result, "storageCleanup": cleanup}
        atomic_write_json(
            self.paths.stage_summary("splat_train"),
            refreshed,
        )
        return refreshed

    def _announce(
        self,
        message: str,
        *,
        status: str = "running",
    ) -> None:
        print(
            f"[mapanything-3dgs] {message}",
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
