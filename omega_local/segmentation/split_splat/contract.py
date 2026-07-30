"""Run and artifact contracts for the staged Split&Splat baseline."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SPLIT_STAGE_ORDER = ("prepare", "global_gs", "proposals", "split")
SPLAT_STAGE_ORDER = (
    "splat_prepare",
    "splat_initial",
    "splat_masks",
    "splat_refined",
    "splat_compose",
)
STAGE_ORDER = (*SPLIT_STAGE_ORDER, *SPLAT_STAGE_ORDER)
STAGE_INDEX = {name: index for index, name in enumerate(STAGE_ORDER)}
STAGE_DEPENDENCIES = {
    "prepare": (),
    "global_gs": ("prepare",),
    "proposals": ("prepare",),
    "split": ("prepare", "global_gs", "proposals"),
    "splat_prepare": ("prepare", "split"),
    "splat_initial": ("splat_prepare",),
    "splat_masks": ("splat_initial",),
    "splat_refined": ("splat_masks",),
    "splat_compose": ("splat_refined",),
}
RUN_SCHEMA_VERSION = 5
PROPOSAL_SOURCES = ("official_auto", "propagation")
SPLIT_METHODS = ("released", "anchored_3d")
DEPTH_SOURCES = ("murre", "editor_depth")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip()).strip("_").lower()
    return normalized or "split_splat_official"


def path_slug(value: str, *, label: str) -> str:
    normalized = slug(value)
    if len(os.fsencode(normalized)) > 96:
        raise ValueError(
            f"{label} is too long for a safe filesystem component "
            f"({len(os.fsencode(normalized))} bytes; maximum 96)."
        )
    return normalized


def positive_int(value: int, *, label: str) -> int:
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{label} must be positive; got {value}.")
    return normalized


def composition_weights(values: tuple[float, ...]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in values)
    if not normalized:
        raise ValueError("At least one composition mask weight is required.")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError(
            "Composition mask weights must be finite values in [0, 1]."
        )
    return normalized


def _temporary_sibling(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".split_splat_",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    temporary = _temporary_sibling(path)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_complete_stage(path: Path) -> dict[str, Any] | None:
    """Return a stage summary only when it explicitly records completion."""
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if payload.get("status") == "complete" else None


def replace_symlink(destination: Path, source: Path) -> None:
    """Replace one staged path with a relative symlink."""
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.is_file():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    destination.symlink_to(
        os.path.relpath(source, start=destination.parent),
        target_is_directory=source.is_dir(),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = _temporary_sibling(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint_files(paths: list[Path], *, extra: dict[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.expanduser().resolve() for item in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        if not path.exists():
            digest.update(b"\0missing")
            continue
        stat = path.stat()
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}".encode("ascii"))
    if extra:
        digest.update(json.dumps(extra, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class SplitSplatRunConfig:
    model_dir: Path
    editor_baseline_name: str
    split_splat_root: Path
    python: Path
    run_id: str = "split_splat_official"
    shared_id: str | None = None
    proposal_source: str = "official_auto"
    propagation_method: str = "sam2_video"
    split_method: str = "released"
    manual_frame_weight: int = 8
    depth_source: str = "murre"
    depth_dir: Path | None = None
    iterations: int = 30_000
    instance_iterations: int = 1_000
    composition_iterations: int = 1_000
    composition_mask_weights: tuple[float, ...] = (0.05, 0.15, 0.25)

    def normalized(self) -> "SplitSplatRunConfig":
        proposal_source = str(self.proposal_source).strip().lower()
        if proposal_source not in PROPOSAL_SOURCES:
            raise ValueError(
                f"Unknown proposal source {proposal_source!r}; "
                f"choose one of {', '.join(PROPOSAL_SOURCES)}."
            )
        split_method = str(self.split_method).strip().lower()
        if split_method not in SPLIT_METHODS:
            raise ValueError(
                f"Unknown split method {split_method!r}; "
                f"choose one of {', '.join(SPLIT_METHODS)}."
            )
        if split_method == "anchored_3d" and proposal_source != "propagation":
            raise ValueError(
                "anchored_3d requires persistent-region propagation proposals."
            )
        depth_source = str(self.depth_source).strip().lower()
        if depth_source not in DEPTH_SOURCES:
            raise ValueError(
                f"Unknown depth source {depth_source!r}; "
                f"choose one of {', '.join(DEPTH_SOURCES)}."
            )
        return SplitSplatRunConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name,
                label="Editor baseline name",
            ),
            split_splat_root=self.split_splat_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            run_id=path_slug(self.run_id, label="Run ID"),
            shared_id=(
                path_slug(self.shared_id, label="Shared cache ID")
                if self.shared_id
                else None
            ),
            proposal_source=proposal_source,
            propagation_method=path_slug(
                self.propagation_method,
                label="Propagation method",
            ),
            split_method=split_method,
            manual_frame_weight=positive_int(
                self.manual_frame_weight,
                label="Manual frame weight",
            ),
            depth_source=depth_source,
            depth_dir=(
                self.depth_dir.expanduser().resolve()
                if self.depth_dir is not None
                else None
            ),
            iterations=positive_int(self.iterations, label="Global iterations"),
            instance_iterations=positive_int(
                self.instance_iterations,
                label="Instance iterations",
            ),
            composition_iterations=positive_int(
                self.composition_iterations,
                label="Composition iterations",
            ),
            composition_mask_weights=composition_weights(
                self.composition_mask_weights
            ),
        )


@dataclass(frozen=True)
class SplitSplatRunPaths:
    experiment_root: Path
    run_id: str
    shared_id: str
    iterations: int

    @property
    def run_dir(self) -> Path:
        return self.experiment_root / "runs" / self.run_id

    @property
    def shared_dir(self) -> Path:
        return self.experiment_root / "_shared" / self.shared_id

    @property
    def shared_manifest(self) -> Path:
        return self.shared_dir / "shared.json"

    @property
    def run_manifest(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def compatibility_report(self) -> Path:
        return self.run_dir / "compatibility_report.json"

    @property
    def progress(self) -> Path:
        return self.run_dir / "progress.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def run_split_dir(self) -> Path:
        return self.run_dir / "split"

    @property
    def input_dir(self) -> Path:
        return self.shared_dir / "01_input"

    @property
    def dataset_dir(self) -> Path:
        return self.input_dir / "dataset"

    @property
    def frame_map(self) -> Path:
        return self.input_dir / "frame_map.jsonl"

    @property
    def input_manifest(self) -> Path:
        return self.input_dir / "manifest.json"

    @property
    def camera_validation(self) -> Path:
        return self.input_dir / "camera_validation.json"

    @property
    def proposals_dir(self) -> Path:
        return self.run_split_dir / "01_proposals"

    @property
    def proposal_binary_masks(self) -> Path:
        return self.proposals_dir / "binary_masks"

    @property
    def proposal_label_maps(self) -> Path:
        return self.proposals_dir / "label_maps"

    @property
    def proposal_overlays(self) -> Path:
        return self.proposals_dir / "overlays"

    @property
    def global_gs_dir(self) -> Path:
        return self.shared_dir / "02_global_gs"

    @property
    def global_gs_model(self) -> Path:
        return self.global_gs_dir / "model"

    @property
    def global_point_cloud(self) -> Path:
        return (
            self.global_gs_model
            / "point_cloud"
            / f"iteration_{self.iterations}"
            / "point_cloud.ply"
        )

    @property
    def global_points_cache(self) -> Path:
        return self.global_gs_dir / "global_points.npz"

    @property
    def global_viewer_gaussians(self) -> Path:
        return self.global_gs_dir / "viewer_gaussians.ply"

    @property
    def shared_splat_support_dir(self) -> Path:
        return self.shared_dir / "03_splat_support"

    @property
    def point_labels_dir(self) -> Path:
        return self.run_split_dir / "02_point_labels"

    @property
    def split_work_dir(self) -> Path:
        return self.point_labels_dir / "work"

    @property
    def split_raw_output(self) -> Path:
        return self.point_labels_dir / "upstream_instances"

    @property
    def point_labels(self) -> Path:
        return self.point_labels_dir / "point_labels.npy"

    @property
    def segmented_gaussians(self) -> Path:
        return self.point_labels_dir / "segmented_gaussians.ply"

    @property
    def point_label_confidence(self) -> Path:
        return self.point_labels_dir / "point_label_confidence.npy"

    @property
    def point_label_provenance(self) -> Path:
        return self.point_labels_dir / "point_label_provenance.npy"

    @property
    def projected_support_dir(self) -> Path:
        return self.point_labels_dir / "projected_support"

    @property
    def projected_support_label_maps(self) -> Path:
        return self.projected_support_dir / "label_maps"

    @property
    def projected_support_overlays(self) -> Path:
        return self.projected_support_dir / "overlays"

    @property
    def consistent_masks_dir(self) -> Path:
        return self.run_split_dir / "03_consistent_masks"

    @property
    def consistent_label_maps(self) -> Path:
        return self.consistent_masks_dir / "label_maps"

    @property
    def consistent_overlays(self) -> Path:
        return self.consistent_masks_dir / "overlays"

    @property
    def split_summary(self) -> Path:
        return self.consistent_masks_dir / "summary.json"

    @property
    def training_view_weights(self) -> Path:
        return self.consistent_masks_dir / "training_view_weights.json"

    @property
    def run_splat_dir(self) -> Path:
        return self.run_dir / "splat"

    @property
    def splat_workspace(self) -> Path:
        return self.run_splat_dir / "work"

    @property
    def splat_scene_dir(self) -> Path:
        return self.splat_workspace / "data" / self.run_id

    @property
    def splat_instances_dir(self) -> Path:
        return self.run_splat_dir / "01_instance_datasets"

    @property
    def splat_initial_models(self) -> Path:
        return self.run_splat_dir / "02_initial_models"

    @property
    def splat_mask_refinement(self) -> Path:
        return self.run_splat_dir / "03_mask_refinement"

    @property
    def splat_mask_preview(self) -> Path:
        return self.splat_mask_refinement / "editor_preview"

    @property
    def splat_mask_label_maps(self) -> Path:
        return self.splat_mask_preview / "label_maps"

    @property
    def splat_mask_overlays(self) -> Path:
        return self.splat_mask_preview / "overlays"

    @property
    def splat_mask_preview_summary(self) -> Path:
        return self.splat_mask_preview / "summary.json"

    @property
    def splat_refined_models(self) -> Path:
        return self.run_splat_dir / "04_refined_models"

    @property
    def splat_composition(self) -> Path:
        return self.run_splat_dir / "05_composition"

    @property
    def splat_outputs(self) -> Path:
        return self.run_splat_dir / "06_outputs"

    def stage_summary(self, stage: str) -> Path:
        stage_dir = {
            "prepare": self.input_dir,
            "proposals": self.proposals_dir,
            "global_gs": self.global_gs_dir,
            "split": self.consistent_masks_dir,
            "splat_prepare": self.splat_instances_dir,
            "splat_initial": self.splat_initial_models,
            "splat_masks": self.splat_mask_refinement,
            "splat_refined": self.splat_refined_models,
            "splat_compose": self.splat_outputs,
        }[stage]
        return stage_dir / "stage.json"

    def stage_log(self, stage: str) -> Path:
        if stage in {"prepare", "global_gs"}:
            return self.shared_dir / "logs" / f"{stage}.log"
        return self.logs_dir / f"{stage}.log"

    @classmethod
    def for_editor(
        cls,
        *,
        editor_interactive_dir: Path,
        run_id: str,
        shared_id: str,
        iterations: int,
    ) -> "SplitSplatRunPaths":
        return cls(
            experiment_root=(
                editor_interactive_dir.expanduser().resolve()
                / "experiments"
                / "split_splat"
            ),
            run_id=path_slug(run_id, label="Run ID"),
            shared_id=path_slug(shared_id, label="Shared cache ID"),
            iterations=max(int(iterations), 1),
        )


def default_shared_id(config: SplitSplatRunConfig) -> str:
    """Key geometry artifacts independently from the run's mask source."""
    payload = {
        "modelDir": str(config.model_dir),
        "editorBaselineName": config.editor_baseline_name,
        "depthSource": config.depth_source,
        "depthDir": str(config.depth_dir) if config.depth_dir else None,
        "iterations": int(config.iterations),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return f"dataset_{digest}"
