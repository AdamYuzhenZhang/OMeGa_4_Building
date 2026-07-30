"""Stage manager for the paper-faithful Split&Splat Split baseline."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from omega_local.stage_graph import downstream_stages
from omega_local.segmentation.interactive.paths import resolve_paths

from .adapter import prepare_dataset
from .anchored_split import run_anchored_split
from .artifacts import (
    finalize_official_proposals,
    finalize_split,
    register_editor_artifacts,
    register_global_point_cloud,
    register_proposal_layer,
    register_splat_refined_mask_layer,
)
from .contract import (
    RUN_SCHEMA_VERSION,
    SPLAT_STAGE_ORDER,
    SPLIT_STAGE_ORDER,
    STAGE_DEPENDENCIES,
    STAGE_INDEX,
    STAGE_ORDER,
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    default_shared_id,
    now_utc,
    read_complete_stage,
    read_json,
)
from .depth_alignment import stage_split_depth
from .proposal_sources import import_propagation_proposals
from .splat import (
    compose_instances,
    prepare_splat,
    refine_instance_masks,
    register_splat_artifacts,
    train_initial_instances,
    train_refined_instances,
)
from .storage import (
    cleanup_completed_composition,
    cleanup_completed_models,
    cleanup_report_changed,
)
from .upstream import (
    ensure_upstream_runtime,
    global_gs_command,
    official_proposals_command,
    prepare_official_proposal_workspace,
    prepare_split_workspace,
    run_command,
    split_command,
    upstream_availability,
    write_compatibility_report,
)


class SplitSplatExperimentManager:
    """Run Split&Splat stages without mutating editor or upstream source data."""

    def __init__(self, config: SplitSplatRunConfig) -> None:
        self.config = config.normalized()
        self.editor_paths = resolve_paths(
            self.config.model_dir,
            self.config.editor_baseline_name,
        )
        shared_id = self.config.shared_id or default_shared_id(self.config)
        self.paths = SplitSplatRunPaths.for_editor(
            editor_interactive_dir=self.editor_paths.interactive_dir,
            run_id=self.config.run_id,
            shared_id=shared_id,
            iterations=self.config.iterations,
        )
        self._last_progress_update = 0.0

    def status(self) -> dict[str, Any]:
        stages = []
        for stage in STAGE_ORDER:
            summary_path = self.paths.stage_summary(stage)
            payload: dict[str, Any] = {}
            if summary_path.is_file():
                try:
                    payload = read_json(summary_path)
                except (OSError, json.JSONDecodeError):
                    payload = {}
            stages.append(
                {
                    "stage": stage,
                    "index": STAGE_INDEX[stage],
                    "scope": (
                        "shared_dataset"
                        if stage in {"prepare", "global_gs"}
                        else "segmentation_run"
                    ),
                    "complete": payload.get("status") == "complete",
                    "status": str(
                        payload.get("status")
                        or ("invalid" if summary_path.is_file() else "not_ready")
                    ),
                    "summaryPath": str(summary_path),
                    "message": str(payload.get("message") or ""),
                }
            )
        return {
            "schemaVersion": RUN_SCHEMA_VERSION,
            "runId": self.paths.run_id,
            "runDir": str(self.paths.run_dir),
            "sharedId": self.paths.shared_id,
            "sharedDir": str(self.paths.shared_dir),
            "proposalSource": self.config.proposal_source,
            "splitMethod": self.config.split_method,
            "propagationMethod": (
                self.config.propagation_method
                if self.config.proposal_source == "propagation"
                else None
            ),
            "upstream": upstream_availability(self.config),
            "stages": stages,
        }

    def run_stage(
        self,
        stage: str,
        *,
        overwrite: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> dict[str, Any]:
        stage = str(stage).strip().lower()
        if stage == "all_split":
            result = {}
            for item in SPLIT_STAGE_ORDER:
                result[item] = self.run_stage(
                    item,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    verbose=verbose,
                )
            return result
        if stage == "all_splat":
            result = {}
            for item in SPLAT_STAGE_ORDER:
                result[item] = self.run_stage(
                    item,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    verbose=verbose,
                )
            return result
        if stage == "all":
            result = {}
            for item in STAGE_ORDER:
                result[item] = self.run_stage(
                    item,
                    overwrite=overwrite,
                    dry_run=dry_run,
                    verbose=verbose,
                )
            return result
        if stage not in STAGE_INDEX:
            raise ValueError(f"Unknown Split&Splat stage: {stage}")

        self._initialize_run()
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            if stage == "global_gs" and not dry_run:
                register_global_point_cloud(self.paths, self.editor_paths)
            elif stage == "proposals" and not dry_run:
                register_proposal_layer(
                    self.paths,
                    self.editor_paths,
                    cached,
                )
            elif stage == "split" and not dry_run:
                register_editor_artifacts(
                    self.paths,
                    self.editor_paths,
                    cached,
                )
            elif stage == "splat_masks" and not dry_run:
                register_splat_refined_mask_layer(
                    self.paths,
                    self.editor_paths,
                    cached,
                    progress=lambda message: self._report_progress(
                        "splat_masks",
                        message,
                    ),
                )
            elif stage == "splat_compose" and not dry_run:
                register_splat_artifacts(
                    self.paths,
                    self.editor_paths,
                    cached,
                )
                cached = read_complete_stage(summary_path) or cached
            if not dry_run:
                cached = self._cleanup_stage_storage(stage, cached)
            self._announce(
                f"{stage}: cache hit ({summary_path})"
            )
            return {
                **cached,
                "cacheHit": True,
            }
        if summary_path.is_file() and not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} has a non-complete summary at {summary_path}. "
                "Inspect the recorded status or rerun with --overwrite-stage."
            )
        if not dry_run:
            self._require_prerequisites(stage)
        if overwrite and not dry_run:
            self._clear_stage_tree(stage)

        self._announce(
            f"{stage}: starting for run {self.paths.run_id}"
        )
        self._write_progress(stage, "running", f"Running Split&Splat stage: {stage}")
        try:
            result = getattr(self, f"_run_{stage}")(
                dry_run=dry_run,
                verbose=verbose,
                overwrite=overwrite,
            )
        except KeyboardInterrupt:
            self._write_progress(stage, "cancelled", "Cancelled by user.")
            self._announce(f"{stage}: cancelled")
            raise
        except Exception as exc:
            self._write_progress(stage, "failed", str(exc))
            self._announce(f"{stage}: failed: {exc}")
            raise
        status = "planned" if dry_run else "complete"
        if not dry_run:
            result = {**result, "status": status}
            atomic_write_json(summary_path, result)
            if stage == "proposals":
                register_proposal_layer(
                    self.paths,
                    self.editor_paths,
                    result,
                )
            elif stage == "splat_masks":
                register_splat_refined_mask_layer(
                    self.paths,
                    self.editor_paths,
                    result,
                    progress=lambda message: self._report_progress(
                        "splat_masks",
                        message,
                    ),
                )
            result = self._cleanup_stage_storage(stage, result)
        self._write_progress(stage, status, f"Split&Splat stage {status}: {stage}")
        self._announce(f"{stage}: {status}")
        return result

    def _run_prepare(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose
        if dry_run:
            return {
                "stage": "prepare",
                "status": "planned",
                "datasetDir": str(self.paths.dataset_dir),
            }
        return prepare_dataset(
            self.config,
            self.editor_paths,
            self.paths,
            overwrite=overwrite,
            progress=lambda message: self._report_progress("prepare", message),
        )

    def _run_proposals(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        if self.config.proposal_source == "propagation":
            if dry_run:
                return {
                    "stage": "proposals",
                    "status": "planned",
                    "proposalSource": "propagation",
                    "sourceMethodId": self.config.propagation_method,
                }
            return import_propagation_proposals(
                self.config,
                self.editor_paths,
                self.paths,
                progress=lambda message: self._report_progress(
                    "proposals",
                    message,
                ),
            )
        if self.config.proposal_source != "official_auto":
            raise ValueError(
                f"Unknown Split&Splat proposal source: {self.config.proposal_source}"
            )

        self._require_upstream()
        command = official_proposals_command(self.config, self.paths)
        if dry_run:
            return {
                "stage": "proposals",
                "status": "planned",
                "proposalSource": "official_auto",
                "command": {
                    "command": command,
                    "cwd": str(self.paths.proposals_dir / "work"),
                },
            }
        work = prepare_official_proposal_workspace(self.paths)
        command_result = run_command(
            command,
            cwd=work,
            config=self.config,
            log_path=self.paths.stage_log("proposals"),
            progress=lambda line: self._stream_message("proposals", line),
            dry_run=dry_run,
        )
        return {
            **finalize_official_proposals(
                self.paths,
                progress=lambda message: self._report_progress(
                    "proposals",
                    message,
                ),
            ),
            "command": command_result,
        }

    def _run_global_gs(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        self._require_upstream()
        self.paths.global_gs_dir.mkdir(parents=True, exist_ok=True)
        command = global_gs_command(self.config, self.paths)
        command_result = run_command(
            command,
            cwd=self.config.split_splat_root,
            config=self.config,
            log_path=self.paths.stage_log("global_gs"),
            progress=lambda line: self._stream_message("global_gs", line),
            dry_run=dry_run,
        )
        if dry_run:
            return {"stage": "global_gs", "status": "planned", "command": command_result}
        if not self.paths.global_point_cloud.is_file():
            raise FileNotFoundError(
                "Global Split&Splat training finished without its expected point cloud: "
                f"{self.paths.global_point_cloud}"
            )
        point_artifact = register_global_point_cloud(self.paths, self.editor_paths)
        self._report_progress(
            "global_gs",
            "Registered the shared global Gaussian means for Step 6.",
        )
        return {
            "schemaVersion": 1,
            "stage": "global_gs",
            "timestampUtc": now_utc(),
            "method": "Released depth-regularized Split&Splat global 3DGS",
            "iterations": self.config.iterations,
            "modelDir": str(self.paths.global_gs_model),
            "pointCloud": str(self.paths.global_point_cloud),
            "pointArtifact": point_artifact,
            "command": command_result,
        }

    def _run_split(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del overwrite
        if self.config.split_method == "anchored_3d":
            if dry_run:
                return {
                    "stage": "split",
                    "status": "planned",
                    "splitMethod": "anchored_3d",
                    "proposalSource": self.config.proposal_source,
                    "propagationMethod": self.config.propagation_method,
                    "manualFrameWeight": self.config.manual_frame_weight,
                }
            return run_anchored_split(
                self.config,
                self.editor_paths,
                self.paths,
                progress=lambda message: self._report_progress(
                    "split",
                    message,
                ),
            )

        self._require_upstream()
        released = split_command(self.config, self.paths, verbose=verbose)
        command = [
            str(self.config.python),
            "-m",
            "omega_local.segmentation.split_splat.launch",
            *released[1:],
        ]
        if dry_run:
            return {
                "stage": "split",
                "status": "planned",
                "command": {
                    "command": command,
                    "cwd": str(self.paths.split_work_dir),
                },
            }
        work = prepare_split_workspace(self.paths)
        depth_alignment = stage_split_depth(
            self.config,
            self.paths,
            work,
            progress=lambda message: self._report_progress("split", message),
        )
        command_result = run_command(
            command,
            cwd=work,
            config=self.config,
            log_path=self.paths.stage_log("split"),
            progress=lambda line: self._stream_message("split", line),
        )
        return {
            **finalize_split(
                self.paths,
                self.editor_paths,
                progress=lambda message: self._report_progress(
                    "split",
                    message,
                ),
            ),
            "depthAlignment": depth_alignment,
            "command": command_result,
        }

    def _run_splat_prepare(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        if dry_run:
            return {
                "stage": "splat_prepare",
                "status": "planned",
                "instancesDir": str(self.paths.splat_instances_dir),
            }
        return prepare_splat(
            self.config,
            self.paths,
            progress=lambda message: self._report_progress(
                "splat_prepare",
                message,
            ),
        )

    def _run_splat_initial(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        if dry_run:
            return {
                "stage": "splat_initial",
                "status": "planned",
                "iterations": self.config.instance_iterations,
            }
        return train_initial_instances(
            self.config,
            self.paths,
            progress=lambda message: self._report_progress(
                "splat_initial",
                message,
            ),
            command_progress=lambda line: self._stream_message(
                "splat_initial",
                line,
            ),
        )

    def _run_splat_masks(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        if dry_run:
            return {
                "stage": "splat_masks",
                "status": "planned",
                "method": "Released Gaussian reprojection and SAM2 refinement",
            }
        return refine_instance_masks(
            self.config,
            self.paths,
            progress=lambda message: self._report_progress(
                "splat_masks",
                message,
            ),
            command_progress=lambda line: self._stream_message(
                "splat_masks",
                line,
            ),
        )

    def _run_splat_refined(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        if dry_run:
            return {
                "stage": "splat_refined",
                "status": "planned",
                "iterations": self.config.instance_iterations,
            }
        return train_refined_instances(
            self.config,
            self.paths,
            progress=lambda message: self._report_progress(
                "splat_refined",
                message,
            ),
            command_progress=lambda line: self._stream_message(
                "splat_refined",
                line,
            ),
        )

    def _run_splat_compose(
        self,
        *,
        dry_run: bool,
        verbose: bool,
        overwrite: bool,
    ) -> dict[str, Any]:
        del verbose, overwrite
        if dry_run:
            return {
                "stage": "splat_compose",
                "status": "planned",
                "iterationsPerMerge": self.config.composition_iterations,
                "maskLossWeights": list(
                    self.config.composition_mask_weights
                ),
            }
        return compose_instances(
            self.config,
            self.paths,
            self.editor_paths,
            progress=lambda message: self._report_progress(
                "splat_compose",
                message,
            ),
            command_progress=lambda line: self._stream_message(
                "splat_compose",
                line,
            ),
        )

    def _initialize_run(self) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.shared_dir.mkdir(parents=True, exist_ok=True)
        shared_definition = {
            "schemaVersion": RUN_SCHEMA_VERSION,
            "sharedId": self.paths.shared_id,
            "ownership": "dataset_geometry",
            "modelDir": str(self.config.model_dir),
            "editorBaselineName": self.config.editor_baseline_name,
            "depthSource": self.config.depth_source,
            "depthDir": str(self.config.depth_dir) if self.config.depth_dir else None,
            "iterations": self.config.iterations,
        }
        if self.paths.shared_manifest.is_file():
            existing = read_json(self.paths.shared_manifest)
            mismatched = [
                key
                for key, value in shared_definition.items()
                if key != "schemaVersion" and existing.get(key) != value
            ]
            if mismatched:
                raise RuntimeError(
                    f"Shared cache ID '{self.paths.shared_id}' already belongs to "
                    "different geometry inputs. Choose another --shared-id. "
                    f"Mismatched fields: {', '.join(mismatched)}"
                )
        else:
            atomic_write_json(
                self.paths.shared_manifest,
                {**shared_definition, "createdUtc": now_utc()},
            )
        if self.paths.shared_manifest.is_file():
            existing_shared = read_json(self.paths.shared_manifest)
            atomic_write_json(
                self.paths.shared_manifest,
                {
                    **existing_shared,
                    **shared_definition,
                    "createdUtc": existing_shared.get("createdUtc", now_utc()),
                },
            )
        manifest = {
            "schemaVersion": RUN_SCHEMA_VERSION,
            "runId": self.paths.run_id,
            "sharedId": self.paths.shared_id,
            "sharedDir": str(self.paths.shared_dir),
            "method": (
                "Split&Splat official baseline"
                if self.config.proposal_source == "official_auto"
                else (
                    "OMeGa anchored 3D split"
                    if self.config.split_method == "anchored_3d"
                    else "Split&Splat proposal-source substitution"
                )
            ),
            "experiment": (
                "C"
                if self.config.split_method == "anchored_3d"
                else (
                    "A"
                    if self.config.proposal_source == "official_auto"
                    else "B"
                )
            ),
            "proposalSource": self.config.proposal_source,
            "splitMethod": self.config.split_method,
            "propagationMethod": (
                self.config.propagation_method
                if self.config.proposal_source == "propagation"
                else None
            ),
            "modelDir": str(self.config.model_dir),
            "editorBaselineName": self.config.editor_baseline_name,
            "splitSplatRoot": str(self.config.split_splat_root),
            "python": str(self.config.python),
            "depthSource": self.config.depth_source,
            "depthDir": str(self.config.depth_dir) if self.config.depth_dir else None,
            "iterations": self.config.iterations,
            "instanceIterations": self.config.instance_iterations,
            "compositionIterations": self.config.composition_iterations,
            "compositionMaskWeights": list(
                self.config.composition_mask_weights
            ),
            "manualFrameWeight": self.config.manual_frame_weight,
            "stageOrder": list(STAGE_ORDER),
            "updatedUtc": now_utc(),
        }
        if self.paths.run_manifest.is_file():
            existing_run = read_json(self.paths.run_manifest)
            immutable_fields = (
                "runId",
                "sharedId",
                "proposalSource",
                "splitMethod",
                "propagationMethod",
                "manualFrameWeight",
                "instanceIterations",
                "compositionIterations",
                "compositionMaskWeights",
            )
            mismatched = [
                key
                for key in immutable_fields
                if key in existing_run
                and existing_run.get(key) != manifest.get(key)
            ]
            if mismatched:
                raise RuntimeError(
                    f"Run ID '{self.paths.run_id}' already belongs to another "
                    "Split&Splat experiment. Choose a new --run-id. "
                    f"Mismatched fields: {', '.join(mismatched)}"
                )
        atomic_write_json(self.paths.run_manifest, manifest)
        write_compatibility_report(self.config, self.paths)

    def _require_prerequisites(self, stage: str) -> None:
        missing = [
            item
            for item in STAGE_DEPENDENCIES[stage]
            if read_complete_stage(self.paths.stage_summary(item)) is None
        ]
        if missing:
            raise RuntimeError(
                f"Run earlier Split&Splat stages first: {', '.join(missing)}"
            )
        if stage == "split" and not self.paths.global_point_cloud.is_file():
            raise FileNotFoundError(
                f"Global Gaussian point cloud does not exist: {self.paths.global_point_cloud}"
            )

    def _require_upstream(self) -> None:
        ensure_upstream_runtime(self.config)
        status = upstream_availability(self.config)
        if not status["available"]:
            raise FileNotFoundError(
                "Split&Splat runtime is incomplete. Missing: "
                + ", ".join(status["missing"])
            )

    def _clear_stage(self, stage: str) -> None:
        stage_dir = {
            "prepare": self.paths.input_dir,
            "proposals": self.paths.proposals_dir,
            "global_gs": self.paths.global_gs_dir,
            "split": self.paths.point_labels_dir,
            "splat_prepare": self.paths.splat_instances_dir,
            "splat_initial": self.paths.splat_initial_models,
            "splat_masks": self.paths.splat_mask_refinement,
            "splat_refined": self.paths.splat_refined_models,
            "splat_compose": self.paths.splat_outputs,
        }[stage]
        if stage == "split":
            for path in (self.paths.point_labels_dir, self.paths.consistent_masks_dir):
                if path.exists():
                    shutil.rmtree(path)
        elif stage == "splat_compose":
            for path in (self.paths.splat_composition, self.paths.splat_outputs):
                if path.exists():
                    shutil.rmtree(path)
        elif stage_dir.exists():
            shutil.rmtree(stage_dir)

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

    def _cleanup_stage_storage(
        self,
        stage: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        cleanup: dict[str, Any] | None = None
        if stage == "global_gs":
            cleanup = cleanup_completed_models(
                self.paths.global_gs_model,
                final_iteration=self.config.iterations,
            )
        elif stage == "splat_initial":
            cleanup = cleanup_completed_models(
                self.paths.splat_initial_models,
                final_iteration=self.config.instance_iterations,
            )
        elif stage == "splat_refined":
            cleanup = cleanup_completed_models(
                self.paths.splat_refined_models,
                final_iteration=self.config.instance_iterations,
            )
        elif stage == "splat_compose":
            cleanup = cleanup_completed_composition(self.paths, result)
        if cleanup is None:
            return result
        if (
            not cleanup_report_changed(cleanup)
            and not cleanup.get("skipped")
            and isinstance(result.get("storageCleanup"), dict)
        ):
            return result
        refreshed = {**result, "storageCleanup": cleanup}
        atomic_write_json(self.paths.stage_summary(stage), refreshed)
        return refreshed

    def _write_progress(self, stage: str, status: str, message: str) -> None:
        atomic_write_json(
            self.paths.progress,
            {
                "schemaVersion": 1,
                "runId": self.paths.run_id,
                "stage": stage,
                "stageIndex": STAGE_INDEX[stage],
                "stageCount": len(STAGE_ORDER),
                "status": status,
                "running": status == "running",
                "failed": status == "failed",
                "message": message,
                "updatedUtc": now_utc(),
            },
        )

    def _stream_message(self, stage: str, line: str) -> None:
        if not line:
            return
        current = time.monotonic()
        if current - self._last_progress_update < 0.5:
            return
        self._last_progress_update = current
        self._write_progress(stage, "running", line[-500:])

    def _report_progress(self, stage: str, message: str) -> None:
        self._announce(message)
        self._write_progress(stage, "running", message[-500:])

    @staticmethod
    def _announce(message: str) -> None:
        print(f"[split-splat] {message}", file=sys.stderr, flush=True)
