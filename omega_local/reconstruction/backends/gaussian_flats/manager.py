"""Stage manager for the released 3D Gaussian Flats pipeline."""

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
    replace_symlink,
)

from .contract import (
    SCHEMA_VERSION,
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    GaussianFlatsConfig,
    GaussianFlatsPaths,
)
from .prepare import prepare_inputs
from .runner import mesh_commands, render_commands, run_command, train_command
from .visualization import ensure_visualization_artifacts


class GaussianFlatsManager:
    def __init__(self, config: GaussianFlatsConfig) -> None:
        self.config = config.normalized()
        self.editor_paths = resolve_paths(
            self.config.model_dir, self.config.editor_baseline_name
        )
        self.paths = GaussianFlatsPaths(
            (
                self.editor_paths.interactive_dir
                / "reconstruction"
                / "runs"
                / "gaussian_flats"
                / self.config.run_id
            ).resolve()
        )

    def status(self) -> dict[str, Any]:
        stages = []
        for stage in STAGE_ORDER:
            path = self.paths.stage_summary(stage)
            payload: dict[str, Any] = {}
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    payload = {}
            stages.append(
                {
                    "stage": stage,
                    "status": payload.get("status", "not_ready"),
                    "complete": payload.get("status") == "complete",
                    "summaryPath": str(path),
                }
            )
        return {
            "schemaVersion": SCHEMA_VERSION,
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "planarRegion": self.config.planar_region,
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
                item: self.run_stage(item, overwrite=overwrite, dry_run=dry_run)
                for item in STAGE_ORDER
            }
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown Gaussian Flats stage: {stage}")
        self._initialize_run()
        summary_path = self.paths.stage_summary(stage)
        cached = read_complete_stage(summary_path)
        if cached is not None and not overwrite:
            if stage in {"train", "render", "mesh"} and not dry_run:
                ensure_visualization_artifacts(self.paths.run_dir)
            return {**cached, "cacheHit": True}
        if summary_path.is_file() and not overwrite:
            raise RuntimeError(
                f"Stage {stage!r} has an incomplete summary. Rerun with "
                "--overwrite-stage after inspecting it."
            )
        self._require_prerequisites(stage, dry_run=dry_run)
        if overwrite and not dry_run:
            self._clear_stage(stage)
        self._announce(f"{stage}: {'planning' if dry_run else 'starting'}")
        result = getattr(self, f"_run_{stage}")(dry_run=dry_run)
        if not dry_run:
            result = {**result, "status": "complete"}
            atomic_write_json(summary_path, result)
        self._announce(f"{stage}: {'planned' if dry_run else 'complete'}")
        return result

    def _run_prepare(self, *, dry_run: bool) -> dict[str, Any]:
        if dry_run:
            return {
                "stage": "prepare",
                "status": "planned",
                "scene": str(self.paths.scene_dir),
                "planeMasks": str(self.paths.plane_masks_dir),
                "planarRegion": self.config.planar_region,
            }
        return prepare_inputs(
            self.config, self.paths, self.editor_paths, progress=self._announce
        )

    def _run_train(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_runtime(require_python=not dry_run)
        final_ply = (
            self.paths.model_dir
            / "point_cloud"
            / f"iteration_{self.config.iterations}"
            / "point_cloud.ply"
        )
        if final_ply.is_file() and not dry_run:
            ensure_visualization_artifacts(self.paths.run_dir)
            return {
                "schemaVersion": 1,
                "stage": "train",
                "status": "complete",
                "recoveredExistingOutput": True,
                "hybridPointCloud": str(final_ply),
                "iteration": self.config.iterations,
            }
        if not dry_run:
            self._clear_stage("train")
            (self.paths.logs_dir / "train.log").unlink(missing_ok=True)
            (self.paths.logs_dir / "train.json").unlink(missing_ok=True)
        command = train_command(self.config, self.paths)
        result = run_command(
            command,
            cwd=self.config.gaussian_flats_root,
            log_path=self.paths.logs_dir / "train.log",
            progress=self._stream,
            dry_run=dry_run,
        )
        if not dry_run and not final_ply.is_file():
            raise FileNotFoundError(f"Final hybrid Gaussian model is missing: {final_ply}")
        if not dry_run:
            self._cleanup_training_progress()
            ensure_visualization_artifacts(self.paths.run_dir)
        return {
            "schemaVersion": 1,
            "stage": "train",
            "status": "planned" if dry_run else "complete",
            "command": result["command"],
            "hybridPointCloud": str(final_ply),
            "iteration": self.config.iterations,
        }

    def _run_render(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_runtime(require_python=not dry_run)
        commands = render_commands(self.config, self.paths)
        results = [
            run_command(
                command,
                cwd=self.config.gaussian_flats_root,
                log_path=self.paths.logs_dir / f"render_{index}.log",
                progress=self._stream,
                dry_run=dry_run,
            )
            for index, command in enumerate(commands)
        ]
        return {
            "schemaVersion": 1,
            "stage": "render",
            "status": "planned" if dry_run else "complete",
            "commands": [item["command"] for item in results],
            "trainRenders": str(
                self.paths.model_dir
                / "train"
                / f"ours_{self.config.iterations}"
                / "renders"
            ),
            "planarRenders": str(
                self.paths.model_dir
                / "train"
                / f"ours_{self.config.iterations}"
                / "renders_planes"
            ),
        }

    def _run_mesh(self, *, dry_run: bool) -> dict[str, Any]:
        self._require_runtime(require_python=not dry_run)
        commands = mesh_commands(self.config, self.paths)
        results = [
            run_command(
                command,
                cwd=self.config.gaussian_flats_root,
                log_path=self.paths.logs_dir / f"mesh_{index}.log",
                progress=self._stream,
                dry_run=dry_run,
            )
            for index, command in enumerate(commands)
        ]
        artifacts = self._register_outputs(dry_run=dry_run)
        if not dry_run:
            self._cleanup_transient_run_files()
        return {
            "schemaVersion": 1,
            "stage": "mesh",
            "status": "planned" if dry_run else "complete",
            "commands": [item["command"] for item in results],
            "artifacts": artifacts,
        }

    def _initialize_run(self) -> None:
        self.paths.run_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.stage_summary("prepare").parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schemaVersion": SCHEMA_VERSION,
            "timestampUtc": now_utc(),
            "method": "3D Gaussian Flats released hybrid reconstruction",
            "paper": "https://arxiv.org/abs/2509.16423",
            "source": "https://github.com/theialab/3dgs-flats",
            "runId": self.config.run_id,
            "runDir": str(self.paths.run_dir),
            "modelDir": str(self.config.model_dir),
            "editorBaseline": self.config.editor_baseline_name,
            "propagationMethod": self.config.propagation_method,
            "planarRegion": self.config.planar_region,
            "iterations": self.config.iterations,
            "capMax": self.config.cap_max,
            "resolution": self.config.resolution,
            "meshVoxelSize": self.config.mesh_voxel_size,
            "paperSettings": {
                "planeFitIteration": self.config.plane_fit_iter,
                "planeFitMinPoints": self.config.plane_fit_min_points,
                "planeSigmaResidual": self.config.plane_sigma_res,
                "planeSigmaDistance": self.config.plane_sigma_dist,
                "planarMaskLossWeight": self.config.planar_mask_loss_weight,
                "depthTvLossWeight": self.config.depthtv_loss_weight,
                "scaleRegularization": self.config.scale_reg,
                "opacityRegularization": self.config.opacity_reg,
            },
        }
        atomic_write_json(self.paths.run_manifest, manifest)

    def _require_prerequisites(self, stage: str, *, dry_run: bool) -> None:
        if dry_run:
            return
        missing = [
            dependency
            for dependency in STAGE_DEPENDENCIES[stage]
            if read_complete_stage(self.paths.stage_summary(dependency)) is None
        ]
        if missing:
            raise RuntimeError("Run prerequisite stages first: " + ", ".join(missing))

    def _require_runtime(self, *, require_python: bool = True) -> None:
        if not (self.config.gaussian_flats_root / "train_planar.py").is_file():
            raise FileNotFoundError(
                f"3D Gaussian Flats source is missing: {self.config.gaussian_flats_root}"
            )
        if require_python and not self.config.python.is_file():
            raise FileNotFoundError(
                f"3D Gaussian Flats Python is missing: {self.config.python}. "
                "Run scripts/setup_gaussian_flats_env.sh first."
            )

    def _clear_stage(self, stage: str) -> None:
        start = STAGE_ORDER.index(stage)
        for item in STAGE_ORDER[start:]:
            self.paths.stage_summary(item).unlink(missing_ok=True)
        if stage == "prepare":
            shutil.rmtree(self.paths.input_dir, ignore_errors=True)
        if stage in {"prepare", "train"}:
            shutil.rmtree(self.paths.model_dir, ignore_errors=True)
        if stage in {"prepare", "train", "render", "mesh"}:
            shutil.rmtree(self.paths.outputs_dir, ignore_errors=True)

    def _cleanup_training_progress(self) -> None:
        for checkpoint in self.paths.model_dir.glob("chkpnt*.pth"):
            checkpoint.unlink(missing_ok=True)
        final_name = f"iteration_{self.config.iterations}"
        point_cloud_root = self.paths.model_dir / "point_cloud"
        for iteration_dir in point_cloud_root.glob("iteration_*"):
            if iteration_dir.name != final_name:
                shutil.rmtree(iteration_dir, ignore_errors=True)
        for event_log in self.paths.model_dir.glob("events.out.tfevents.*"):
            event_log.unlink(missing_ok=True)

    def _cleanup_transient_run_files(self) -> None:
        self._cleanup_training_progress()
        self.paths.progress.unlink(missing_ok=True)
        shutil.rmtree(self.paths.logs_dir, ignore_errors=True)

    def _register_outputs(self, *, dry_run: bool) -> dict[str, str]:
        iteration_dir = (
            self.paths.model_dir
            / "point_cloud"
            / f"iteration_{self.config.iterations}"
        )
        sources = {
            "hybridGaussians": iteration_dir / "point_cloud.ply",
            "planarGaussians": iteration_dir / "point_cloud_planar.ply",
            "planes": iteration_dir / "planes.json",
            "planeToMask": self.paths.model_dir
            / f"plane_to_mask_id_{self.config.iterations}.json",
            "fullMesh": self.paths.model_dir / "fuse_post.ply",
            "planarMesh": self.paths.model_dir / "planar_mesh.obj",
        }
        if dry_run:
            return {key: str(path) for key, path in sources.items()}
        self.paths.outputs_dir.mkdir(parents=True, exist_ok=True)
        registered: dict[str, str] = {}
        for key, source in sources.items():
            if source.exists():
                destination = self.paths.outputs_dir / source.name
                replace_symlink(destination, source)
                registered[key] = str(destination)
        ensure_visualization_artifacts(self.paths.run_dir)
        registered["visualizationManifest"] = str(
            self.paths.outputs_dir / "visualization.json"
        )
        atomic_write_json(
            self.paths.outputs_dir / "manifest.json",
            {"schemaVersion": 1, "timestampUtc": now_utc(), "artifacts": registered},
        )
        return registered

    @staticmethod
    def _announce(message: str) -> None:
        print(f"[gaussian-flats] {message}", file=sys.stderr, flush=True)

    @staticmethod
    def _stream(message: str) -> None:
        del message
