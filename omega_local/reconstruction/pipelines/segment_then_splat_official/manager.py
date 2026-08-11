"""Resumable orchestration for the complete released paper pipeline."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from omega_local.segmentation.interactive.paths import resolve_paths
from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
    read_complete_stage,
    read_json,
)
from omega_local.stage_graph import downstream_stages

from .autoseg import run_official_autoseg
from .contract import (
    SCHEMA_VERSION,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    OfficialSegmentThenSplatConfig,
    OfficialSegmentThenSplatPaths,
)
from .initialize import initialize_paper_objects
from .prepare import prepare_paper_dataset
from .train_export import export_paper_model, train_paper_model


class OfficialSegmentThenSplatManager:
    def __init__(self, config: OfficialSegmentThenSplatConfig) -> None:
        self.config = config.normalized()
        self.editor = resolve_paths(
            self.config.model_dir,
            self.config.editor_baseline_name,
        )
        run_dir = (
            self.editor.interactive_dir
            / "experiments"
            / "segment_then_splat_paper"
            / "runs"
            / self.config.run_id
        ).resolve()
        self.paths = OfficialSegmentThenSplatPaths(run_dir=run_dir)

    def status(self) -> dict[str, Any]:
        stages = []
        for stage in STAGE_ORDER:
            path = self.paths.stage_summary(stage)
            payload = {}
            if path.is_file():
                try:
                    payload = read_json(path)
                except (OSError, json.JSONDecodeError):
                    payload = {}
            stages.append(
                {
                    "stage": stage,
                    "status": str(payload.get("status") or ("invalid" if path.is_file() else "not_ready")),
                    "complete": payload.get("status") == "complete",
                    "summaryPath": str(path),
                }
            )
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
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
                name: self.run_stage(name, overwrite=overwrite, dry_run=dry_run)
                for name in STAGE_ORDER
            }
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown paper stage: {stage}")
        self._initialize_run(stage, overwrite=overwrite)
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            return {**cached, "cacheHit": True}
        if not dry_run:
            self._require_prerequisites(stage)
        if overwrite and not dry_run:
            self._clear_stage_tree(stage)
        self._announce(f"{stage}: starting")
        try:
            result = getattr(self, f"_run_{stage}")(dry_run=dry_run)
            if not dry_run and result.get("status") != "complete":
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
            return {"stage": "prepare", "status": "planned", "sceneDir": str(self.paths.scene_dir)}
        return prepare_paper_dataset(
            self.config,
            self.paths,
            self.editor,
            progress=self._stream,
        )

    def _run_autoseg(self, *, dry_run: bool) -> dict[str, Any]:
        return run_official_autoseg(
            self.config,
            self.paths,
            progress=self._stream,
            dry_run=dry_run,
        )

    def _run_initialize(self, *, dry_run: bool) -> dict[str, Any]:
        return initialize_paper_objects(
            self.config,
            self.paths,
            self.editor,
            progress=self._stream,
            dry_run=dry_run,
        )

    def _run_train(self, *, dry_run: bool) -> dict[str, Any]:
        return train_paper_model(
            self.config,
            self.paths,
            progress=self._stream,
            dry_run=dry_run,
        )

    def _run_export(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {"stage": "export", "status": "planned", "output": str(self.paths.outputs_stage)}
        return export_paper_model(
            self.config,
            self.paths,
            progress=self._stream,
        )

    def _initialize_run(self, stage: str, *, overwrite: bool) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schemaVersion": SCHEMA_VERSION,
            "pipeline": "segment_then_splat_paper",
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "modelDir": str(self.config.model_dir),
            "editorBaselineName": self.config.editor_baseline_name,
            "sourceRoot": str(self.config.source_root),
            "python": str(self.config.python),
            "imageLongSide": int(self.config.image_long_side),
            "detectStride": int(self.config.detect_stride),
            "batchSize": int(self.config.batch_size),
            "iterations": int(self.config.iterations),
            "densifyUntilIteration": int(self.config.densify_until_iter),
            "numSampleObjects": int(self.config.num_sample_objects),
            "partialMaskIoU": float(self.config.partial_mask_iou),
            "stageOrder": list(STAGE_ORDER),
            "updatedUtc": now_utc(),
        }
        if self.paths.run_manifest.is_file() and not overwrite:
            old = read_json(self.paths.run_manifest)
            immutable = (
                "imageLongSide", "detectStride", "iterations",
                "densifyUntilIteration", "numSampleObjects", "partialMaskIoU",
            )
            changed = [key for key in immutable if old.get(key) != payload.get(key)]
            if changed:
                raise RuntimeError(
                    "Paper run settings changed: " + ", ".join(changed) +
                    ". Use a new --run-id or overwrite from prepare."
                )
        atomic_write_json(self.paths.run_manifest, payload)

    def _require_prerequisites(self, stage: str) -> None:
        missing = [
            name for name in STAGE_DEPENDENCIES[stage]
            if read_complete_stage(self.paths.stage_summary(name)) is None
        ]
        if missing:
            raise RuntimeError("Run earlier paper stages first: " + ", ".join(missing))

    def _clear_stage_tree(self, stage: str) -> None:
        affected = (
            stage,
            *downstream_stages(stage, order=STAGE_ORDER, dependencies=STAGE_DEPENDENCIES),
        )
        for name in reversed(affected):
            directory = self.paths.stage_summary(name).parent
            if directory.exists():
                shutil.rmtree(directory)

    def _announce(self, message: str, *, status: str = "running") -> None:
        print(f"[segment-then-splat-paper] {message}", file=sys.stderr, flush=True)
        atomic_write_json(
            self.paths.progress,
            {
                "status": status,
                "message": message,
                "updatedUtc": now_utc(),
            },
        )

    def _stream(self, message: str) -> None:
        self._announce(message)

