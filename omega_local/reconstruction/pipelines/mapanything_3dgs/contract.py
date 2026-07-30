"""Run contract for MapAnything-carried region reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omega_local.reconstruction.gaussian_pruning import (
    DEFAULT_MASK_DILATION_PIXELS,
    DEFAULT_MIN_VISIBLE_VIEWS,
    DEFAULT_OPACITY_THRESHOLD,
)
from omega_local.segmentation.split_splat.contract import path_slug, positive_int


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


STAGE_ORDER = (
    "prepare",
    "split",
    "splat_prepare",
    "splat_train",
    "splat_compose",
)
STAGE_DEPENDENCIES = {
    "prepare": (),
    "split": ("prepare",),
    "splat_prepare": ("split",),
    "splat_train": ("splat_prepare",),
    "splat_compose": ("splat_train",),
}


@dataclass(frozen=True)
class MapAnything3DGSConfig:
    model_dir: Path
    editor_baseline_name: str
    split_splat_root: Path
    python: Path
    run_id: str = "mapanything_sam2_video"
    propagation_method: str = "sam2_video"
    point_cloud: Path | None = None
    manual_frame_weight: int = 8
    instance_iterations: int = 10_000
    minimum_region_points: int = 32
    minimum_positive_views: int = 2
    max_num_splats: int = 100_000
    mask_loss_weight: float = 1.0
    floater_pruning: bool = True
    floater_opacity_threshold: float = DEFAULT_OPACITY_THRESHOLD
    floater_min_visible_views: int = DEFAULT_MIN_VISIBLE_VIEWS
    floater_mask_dilation_pixels: int = DEFAULT_MASK_DILATION_PIXELS

    def normalized(self) -> "MapAnything3DGSConfig":
        return MapAnything3DGSConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name,
                label="Editor baseline name",
            ),
            split_splat_root=self.split_splat_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            run_id=path_slug(self.run_id, label="Run ID"),
            propagation_method=path_slug(
                self.propagation_method,
                label="Propagation method",
            ),
            point_cloud=(
                self.point_cloud.expanduser().resolve()
                if self.point_cloud is not None
                else None
            ),
            manual_frame_weight=positive_int(
                self.manual_frame_weight,
                label="Manual frame weight",
            ),
            instance_iterations=positive_int(
                self.instance_iterations,
                label="Instance iterations",
            ),
            minimum_region_points=positive_int(
                self.minimum_region_points,
                label="Minimum region points",
            ),
            minimum_positive_views=positive_int(
                self.minimum_positive_views,
                label="Minimum positive views",
            ),
            max_num_splats=positive_int(
                self.max_num_splats,
                label="Maximum splats per region",
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


@dataclass(frozen=True)
class MapAnything3DGSPaths:
    run_dir: Path
    point_cloud: Path

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
    def input_dir(self) -> Path:
        return self.run_dir / "01_input"

    @property
    def dataset_dir(self) -> Path:
        return self.input_dir / "dataset"

    @property
    def frame_map(self) -> Path:
        return self.input_dir / "frame_map.jsonl"

    @property
    def split_dir(self) -> Path:
        return self.run_dir / "02_split"

    @property
    def point_labels_dir(self) -> Path:
        return self.split_dir / "point_labels"

    @property
    def point_labels(self) -> Path:
        return self.point_labels_dir / "labels.npy"

    @property
    def point_confidence(self) -> Path:
        return self.point_labels_dir / "confidence.npy"

    @property
    def point_provenance(self) -> Path:
        return self.point_labels_dir / "provenance.npy"

    @property
    def point_cache(self) -> Path:
        return self.point_labels_dir / "segmented_points.npz"

    @property
    def segmented_points(self) -> Path:
        return self.point_labels_dir / "segmented_points.ply"

    @property
    def projected_support_dir(self) -> Path:
        return self.split_dir / "projected_support"

    @property
    def projected_support_label_maps(self) -> Path:
        return self.projected_support_dir / "label_maps"

    @property
    def projected_support_overlays(self) -> Path:
        return self.projected_support_dir / "overlays"

    @property
    def clean_masks_dir(self) -> Path:
        return self.split_dir / "clean_masks"

    @property
    def clean_label_maps(self) -> Path:
        return self.clean_masks_dir / "label_maps"

    @property
    def clean_overlays(self) -> Path:
        return self.clean_masks_dir / "overlays"

    @property
    def split_instances_dir(self) -> Path:
        return self.split_dir / "instances"

    @property
    def training_view_weights(self) -> Path:
        return self.clean_masks_dir / "training_view_weights.json"

    @property
    def splat_dir(self) -> Path:
        return self.run_dir / "03_splat"

    @property
    def region_datasets(self) -> Path:
        return self.splat_dir / "01_region_datasets"

    @property
    def region_models(self) -> Path:
        return self.splat_dir / "02_region_models"

    @property
    def splat_outputs(self) -> Path:
        return self.splat_dir / "03_outputs"

    @property
    def composed_scene(self) -> Path:
        return self.splat_outputs / "composed_scene.ply"

    @property
    def viewer_scene(self) -> Path:
        return self.splat_outputs / "viewer" / "composed_scene.ply"

    @property
    def viewer_regions(self) -> Path:
        return self.splat_outputs / "viewer" / "regions"

    @property
    def appearance_diagnostics(self) -> Path:
        return self.splat_outputs / "appearance_diagnostics.json"

    @property
    def composed_labels(self) -> Path:
        return self.splat_outputs / "composed_scene_region_labels.npy"

    @property
    def composed_region_ids(self) -> Path:
        return self.splat_outputs / "composed_scene_region_ids.ply"

    def stage_summary(self, stage: str) -> Path:
        directories = {
            "prepare": self.input_dir,
            "split": self.clean_masks_dir,
            "splat_prepare": self.region_datasets,
            "splat_train": self.region_models,
            "splat_compose": self.splat_outputs,
        }
        return directories[stage] / "stage.json"
