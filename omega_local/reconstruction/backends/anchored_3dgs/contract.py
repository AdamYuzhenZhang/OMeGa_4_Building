"""Contracts for the isolated anchored 3DGS reconstruction backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omega_local.reconstruction.gaussian_pruning import (
    DEFAULT_MASK_DILATION_PIXELS,
    DEFAULT_MIN_VISIBLE_VIEWS,
    DEFAULT_OPACITY_THRESHOLD,
)
from omega_local.segmentation.split_splat.contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    composition_weights,
    path_slug,
    positive_int,
)


STAGE_ORDER = ("prepare", "initial", "refine_masks", "refined", "compose")
STAGE_DEPENDENCIES = {
    "prepare": (),
    "initial": ("prepare",),
    "refine_masks": ("initial",),
    "refined": ("refine_masks",),
    "compose": ("refined",),
}
MASK_REFINEMENT_POLICIES = ("none", "paper_sam2")
SCHEMA_VERSION = 1


def _positive_float(value: float, *, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {value}.")
    return result


def _opacity_threshold(value: float) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError(
            f"Floater opacity threshold must be in [0, 1), got {value}."
        )
    return result


def _nonnegative_int(value: int, *, label: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} cannot be negative, got {value}.")
    return result


@dataclass(frozen=True)
class AnchoredSplatConfig:
    model_dir: Path
    editor_baseline_name: str
    split_splat_root: Path
    python: Path
    source_run_id: str = "split_splat_anchored_sam2_video"
    run_id: str = "anchored_splat_no_refine"
    mask_refinement: str = "none"
    instance_iterations: int = 1_000
    composition_iterations: int = 1_000
    composition_mask_weights: tuple[float, ...] = (0.05, 0.15, 0.25)
    minimum_gaussians: int = 8
    minimum_positive_views: int = 2
    mask_loss_weight: float = 1.0
    floater_pruning: bool = True
    floater_opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD
    floater_min_visible_views: int = DEFAULT_MIN_VISIBLE_VIEWS
    floater_mask_dilation_pixels: int = DEFAULT_MASK_DILATION_PIXELS

    def normalized(self) -> "AnchoredSplatConfig":
        policy = str(self.mask_refinement).strip().lower()
        if policy not in MASK_REFINEMENT_POLICIES:
            raise ValueError(
                f"Unknown mask refinement {policy!r}; choose one of "
                f"{', '.join(MASK_REFINEMENT_POLICIES)}."
            )
        return AnchoredSplatConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name,
                label="Editor baseline name",
            ),
            split_splat_root=self.split_splat_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            source_run_id=path_slug(
                self.source_run_id,
                label="Source Split run ID",
            ),
            run_id=path_slug(self.run_id, label="Anchored Splat run ID"),
            mask_refinement=policy,
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
            minimum_gaussians=positive_int(
                self.minimum_gaussians,
                label="Minimum Gaussians",
            ),
            minimum_positive_views=positive_int(
                self.minimum_positive_views,
                label="Minimum positive views",
            ),
            mask_loss_weight=_positive_float(
                self.mask_loss_weight,
                label="Mask loss weight",
            ),
            floater_pruning=bool(self.floater_pruning),
            floater_opacity_threshold=_opacity_threshold(
                self.floater_opacity_threshold
            ),
            floater_min_visible_views=positive_int(
                self.floater_min_visible_views,
                label="Floater minimum visible views",
            ),
            floater_mask_dilation_pixels=_nonnegative_int(
                self.floater_mask_dilation_pixels,
                label="Floater mask dilation pixels",
            ),
        )

    def released_runtime(
        self,
        *,
        shared_id: str,
        iterations: int,
        depth_source: str,
        propagation_method: str,
    ) -> SplitSplatRunConfig:
        """Build the upstream runtime configuration without changing baselines."""
        return SplitSplatRunConfig(
            model_dir=self.model_dir,
            editor_baseline_name=self.editor_baseline_name,
            split_splat_root=self.split_splat_root,
            python=self.python,
            run_id=self.run_id,
            shared_id=shared_id,
            proposal_source="propagation",
            propagation_method=propagation_method,
            split_method="anchored_3d",
            depth_source=depth_source,
            iterations=iterations,
            instance_iterations=self.instance_iterations,
            composition_iterations=self.composition_iterations,
            composition_mask_weights=self.composition_mask_weights,
        ).normalized()


@dataclass(frozen=True)
class AnchoredSplatPaths:
    run_dir: Path
    source: SplitSplatRunPaths
    run_id: str

    @property
    def run_manifest(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def progress(self) -> Path:
        return self.run_dir / "progress.json"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def splat_instances_dir(self) -> Path:
        return self.run_dir / "01_region_datasets"

    @property
    def splat_initial_models(self) -> Path:
        return self.run_dir / "02_initial_models"

    @property
    def initial_renders(self) -> Path:
        return self.run_dir / "03_initial_renders"

    @property
    def splat_mask_refinement(self) -> Path:
        return self.run_dir / "04_mask_refinement"

    @property
    def splat_refined_models(self) -> Path:
        return self.run_dir / "05_refined_models"

    @property
    def splat_composition(self) -> Path:
        return self.run_dir / "06_composition"

    @property
    def splat_outputs(self) -> Path:
        return self.run_dir / "07_outputs"

    @property
    def splat_workspace(self) -> Path:
        return self.run_dir / "work"

    @property
    def splat_scene_dir(self) -> Path:
        return self.splat_workspace / "data" / self.run_id

    @property
    def dataset_dir(self) -> Path:
        return self.source.dataset_dir

    @property
    def frame_map(self) -> Path:
        return self.source.frame_map

    @property
    def global_gs_dir(self) -> Path:
        return self.source.global_gs_dir

    @property
    def global_point_cloud(self) -> Path:
        return self.source.global_point_cloud

    @property
    def point_labels(self) -> Path:
        return self.source.point_labels

    @property
    def split_raw_output(self) -> Path:
        return self.source.split_raw_output

    @property
    def shared_splat_support_dir(self) -> Path:
        return self.source.shared_splat_support_dir

    @property
    def training_view_weights(self) -> Path:
        return self.source.training_view_weights

    def stage_summary(self, stage: str) -> Path:
        if stage == "split":
            return self.source.stage_summary("split")
        aliases = {
            "splat_prepare": "prepare",
            "splat_initial": "initial",
            "splat_masks": "refine_masks",
            "splat_refined": "refined",
            "splat_compose": "compose",
        }
        stage = aliases.get(stage, stage)
        directory = {
            "prepare": self.splat_instances_dir,
            "initial": self.splat_initial_models,
            "refine_masks": self.splat_mask_refinement,
            "refined": self.splat_refined_models,
            "compose": self.splat_outputs,
        }[stage]
        return directory / "stage.json"
