"""Stage manager for the isolated anchored 3DGS reconstruction variant."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from omega_local.reconstruction.gaussian_pruning import (
    prune_composed_gaussian_floaters,
)
from omega_local.stage_graph import downstream_stages
from omega_local.segmentation.interactive.paths import resolve_paths
from omega_local.segmentation.split_splat.contract import (
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    read_complete_stage,
    read_json,
)
from omega_local.segmentation.split_splat.splat import (
    compose_instances,
    register_splat_artifacts,
)
from omega_local.segmentation.split_splat.storage import (
    cleanup_completed_composition,
    cleanup_completed_models,
    cleanup_report_changed,
)
from omega_local.segmentation.split_splat.upstream import (
    ensure_upstream_runtime,
    upstream_availability,
)

from .artifacts import register_anchored_artifacts
from .contract import (
    SCHEMA_VERSION,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    AnchoredSplatConfig,
    AnchoredSplatPaths,
)
from .prepare import prepare_regions
from .refine import link_unrefined_models, refine_masks
from .train import train_initial_regions, train_refined_regions


class AnchoredSplatManager:
    """Run anchored reconstruction without modifying source Split runs."""

    def __init__(self, config: AnchoredSplatConfig) -> None:
        self.config = config.normalized()
        self.editor_paths = resolve_paths(
            self.config.model_dir,
            self.config.editor_baseline_name,
        )
        experiment_root = (
            self.editor_paths.interactive_dir / "experiments" / "split_splat"
        )
        source_manifest_path = (
            experiment_root / "runs" / self.config.source_run_id / "run.json"
        )
        if not source_manifest_path.is_file():
            raise FileNotFoundError(
                "Anchored source Split run does not exist: "
                f"{source_manifest_path}"
            )
        self.source_manifest = read_json(source_manifest_path)
        self._validate_source_manifest(source_manifest_path)
        self.source_paths = SplitSplatRunPaths.for_editor(
            editor_interactive_dir=self.editor_paths.interactive_dir,
            run_id=self.config.source_run_id,
            shared_id=str(self.source_manifest["sharedId"]),
            iterations=int(self.source_manifest["iterations"]),
        )
        run_dir = (
            self.editor_paths.interactive_dir
            / "reconstruction"
            / "runs"
            / "anchored_3dgs"
            / self.config.run_id
        ).resolve()
        self.paths = AnchoredSplatPaths(
            run_dir=run_dir,
            source=self.source_paths,
            run_id=self.config.run_id,
        )
        self.runtime_config = self.config.released_runtime(
            shared_id=self.source_paths.shared_id,
            iterations=self.source_paths.iterations,
            depth_source=str(self.source_manifest["depthSource"]),
            propagation_method=str(self.source_manifest["propagationMethod"]),
        )
        self._assert_isolated_output()

    def status(self) -> dict[str, Any]:
        stages = []
        for index, stage in enumerate(STAGE_ORDER):
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
                    "index": index,
                    "complete": bool(
                        payload.get("status") == "complete"
                    ),
                    "status": str(
                        payload.get("status")
                        or ("invalid" if summary_path.is_file() else "not_ready")
                    ),
                    "summaryPath": str(summary_path),
                    "message": str(payload.get("message") or ""),
                }
            )
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "sourceRunId": self.config.source_run_id,
            "sourceRunDir": str(self.source_paths.run_dir),
            "maskRefinement": self.config.mask_refinement,
            "sam2RefinementEnabled": self.config.mask_refinement == "paper_sam2",
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
            raise ValueError(f"Unknown anchored Splat stage: {stage}")
        self._initialize_run()
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            if stage == "compose":
                register_splat_artifacts(
                    self.paths,
                    self.editor_paths,
                    cached,
                )
                register_anchored_artifacts(
                    self.config,
                    self.paths,
                    self.editor_paths,
                )
                cached = read_complete_stage(summary_path) or cached
            if not dry_run:
                cached = self._cleanup_stage_storage(stage, cached)
            return {**cached, "cacheHit": True}
        if summary_path.is_file() and not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} has a non-complete summary at {summary_path}. "
                "Inspect the recorded status or rerun with --overwrite-stage."
            )
        if not dry_run:
            self._require_prerequisites(stage)
        if overwrite and not dry_run:
            self._clear_stage_tree(stage)

        self._announce(f"{stage}: starting for anchored run {self.config.run_id}")
        result = getattr(self, f"_run_{stage}")(dry_run=dry_run)
        if not dry_run:
            result = {**result, "status": "complete"}
            atomic_write_json(summary_path, result)
            result = self._cleanup_stage_storage(stage, result)
        self._announce(f"{stage}: {'planned' if dry_run else 'complete'}")
        return result

    def _run_prepare(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "prepare",
                "status": "planned",
                "sourceSplit": str(self.source_paths.stage_summary("split")),
                "instancesDir": str(self.paths.splat_instances_dir),
            }
        return prepare_regions(
            self.config,
            self.runtime_config,
            self.paths,
            progress=self._announce,
        )

    def _run_initial(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_upstream()
        return train_initial_regions(
            self.config,
            self.runtime_config,
            self.paths,
            progress=self._announce,
            command_progress=self._stream,
            dry_run=dry_run,
        )

    def _run_refine_masks(self, *, dry_run: bool) -> dict[str, Any]:
        if self.config.mask_refinement == "paper_sam2":
            self._require_upstream()
        return refine_masks(
            self.config,
            self.runtime_config,
            self.paths,
            progress=self._announce,
            command_progress=self._stream,
            dry_run=dry_run,
        )

    def _run_refined(self, *, dry_run: bool) -> dict[str, Any]:
        if self.config.mask_refinement == "none":
            if dry_run:
                return {
                    "stage": "refined",
                    "status": "planned",
                    "method": "reuse initial models",
                    "retrained": False,
                }
            return link_unrefined_models(self.config, self.paths)
        self._require_upstream()
        return train_refined_regions(
            self.config,
            self.runtime_config,
            self.paths,
            progress=self._announce,
            command_progress=self._stream,
            dry_run=dry_run,
        )

    def _run_compose(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_upstream()
        if dry_run:
            return {
                "stage": "compose",
                "status": "planned",
                "method": "Released Split&Splat collision-driven composition",
                "iterationsPerMerge": self.config.composition_iterations,
                "maskLossWeights": list(self.config.composition_mask_weights),
            }
        result = {
            **compose_instances(
                self.runtime_config,
                self.paths,
                self.editor_paths,
                progress=self._announce,
                command_progress=self._stream,
                post_composition_filter=(
                    self._prune_composed_scene
                    if self.config.floater_pruning
                    else None
                ),
                foreground_balanced=True,
            ),
            "stage": "compose",
            "anchoredVariant": True,
            "maskRefinementPolicy": self.config.mask_refinement,
        }
        return {
            **result,
            "editorArtifact": register_anchored_artifacts(
                self.config,
                self.paths,
                self.editor_paths,
            ),
        }

    def _prune_composed_scene(
        self,
        point_cloud: Path,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return prune_composed_gaussian_floaters(
            point_cloud,
            labels,
            frame_map=self.paths.frame_map,
            colmap_images=(
                self.paths.dataset_dir / "sparse" / "0" / "images.txt"
            ),
            summary_path=(
                self.paths.splat_outputs
                / "post_composition_floater_pruning.json"
            ),
            opacity_threshold=self.config.floater_opacity_threshold,
        )

    def _initialize_run(self) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "method": "Anchored per-region 3DGS reconstruction",
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "sourceRunId": self.config.source_run_id,
            "sourceRunDir": str(self.source_paths.run_dir),
            "sourceSplitSummary": str(self.source_paths.stage_summary("split")),
            "sourceGlobalGaussians": str(self.source_paths.global_point_cloud),
            "sharedId": self.source_paths.shared_id,
            "globalIterations": self.source_paths.iterations,
            "instanceIterations": self.config.instance_iterations,
            "compositionIterations": self.config.composition_iterations,
            "compositionMaskWeights": list(
                self.config.composition_mask_weights
            ),
            "compositionSupervision": "foreground_balanced",
            "minimumGaussians": self.config.minimum_gaussians,
            "minimumPositiveViews": self.config.minimum_positive_views,
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
            "maskRefinement": self.config.mask_refinement,
            "initializerTopologyPolicy": (
                "preserve_during_optimization_then_semantic_prune"
            ),
            "sam2RefinementEnabled": (
                self.config.mask_refinement == "paper_sam2"
            ),
            "stageOrder": list(STAGE_ORDER),
            "updatedUtc": now_utc(),
        }
        if self.paths.run_manifest.is_file():
            existing = read_json(self.paths.run_manifest)
            immutable = (
                "runId",
                "sourceRunId",
                "sharedId",
                "globalIterations",
                "instanceIterations",
                "compositionIterations",
                "compositionMaskWeights",
                "compositionSupervision",
                "minimumGaussians",
                "minimumPositiveViews",
                "maskLossWeight",
                "floaterPruning",
                "floaterOpacityThreshold",
                "floaterMinVisibleViews",
                "floaterMaskDilationPixels",
                "maskRefinement",
            )
            mismatched = [
                key
                for key in immutable
                if key in existing and existing.get(key) != manifest.get(key)
            ]
            if mismatched:
                raise RuntimeError(
                    f"Anchored run ID '{self.config.run_id}' already has other "
                    "settings. Choose a new --run-id. Mismatched fields: "
                    f"{', '.join(mismatched)}"
                )
        atomic_write_json(self.paths.run_manifest, manifest)

    def _require_prerequisites(self, stage: str) -> None:
        if read_complete_stage(self.source_paths.stage_summary("split")) is None:
            raise FileNotFoundError(
                "The source anchored Split stage is incomplete: "
                f"{self.source_paths.stage_summary('split')}"
            )
        missing = [
            item
            for item in STAGE_DEPENDENCIES[stage]
            if read_complete_stage(self.paths.stage_summary(item)) is None
        ]
        if missing:
            raise RuntimeError(
                "Run earlier anchored Splat stages first: "
                f"{', '.join(missing)}"
            )

    def _clear_stage(self, stage: str) -> None:
        if stage == "compose":
            for directory in (
                self.paths.splat_composition,
                self.paths.splat_outputs,
            ):
                if directory.exists():
                    shutil.rmtree(directory)
            return
        summary = self.paths.stage_summary(stage)
        directory = summary.parent
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

    def _cleanup_stage_storage(
        self,
        stage: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        cleanup: dict[str, Any] | None = None
        if stage == "initial":
            cleanup = cleanup_completed_models(
                self.paths.splat_initial_models,
                final_iteration=self.config.instance_iterations,
            )
        elif stage == "refined":
            cleanup = cleanup_completed_models(
                self.paths.splat_refined_models,
                final_iteration=self.config.instance_iterations,
            )
        elif stage == "compose":
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

    def _require_upstream(self) -> None:
        ensure_upstream_runtime(self.runtime_config)
        availability = upstream_availability(self.runtime_config)
        if not availability["available"]:
            raise FileNotFoundError(
                "Split&Splat runtime is incomplete. Missing: "
                + ", ".join(availability["missing"])
            )

    def _assert_isolated_output(self) -> None:
        output = self.paths.run_dir
        source = self.source_paths.run_dir.resolve()
        baseline_root = (
            self.editor_paths.interactive_dir
            / "experiments"
            / "split_splat"
        ).resolve()
        if output == source or output.is_relative_to(source):
            raise RuntimeError("Anchored output cannot be inside its source run.")
        if output == baseline_root or output.is_relative_to(baseline_root):
            raise RuntimeError(
                "Anchored reconstruction must not write into Split&Splat "
                "baseline experiment directories."
            )

    def _validate_source_manifest(self, path: Path) -> None:
        required = (
            "sharedId",
            "iterations",
            "depthSource",
            "propagationMethod",
        )
        missing = [key for key in required if self.source_manifest.get(key) is None]
        if missing:
            raise ValueError(
                f"Anchored source manifest is missing {', '.join(missing)}: {path}"
            )
        if self.source_manifest.get("splitMethod") != "anchored_3d":
            raise ValueError(
                "Anchored reconstruction requires a source run with "
                f"splitMethod='anchored_3d': {path}"
            )
        if self.source_manifest.get("proposalSource") != "propagation":
            raise ValueError(
                "Anchored reconstruction requires persistent-region propagation "
                f"proposals: {path}"
            )

    @staticmethod
    def _announce(message: str) -> None:
        print(f"[anchored-splat] {message}", file=sys.stderr, flush=True)

    @staticmethod
    def _stream(message: str) -> None:
        del message
