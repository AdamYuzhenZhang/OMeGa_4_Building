"""Run contract for the isolated 3D Gaussian Flats bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omega_local.segmentation.split_splat.contract import path_slug, positive_int


STAGE_ORDER = ("prepare", "train", "render", "mesh")
STAGE_DEPENDENCIES = {
    "prepare": (),
    "train": ("prepare",),
    "render": ("train",),
    "mesh": ("render",),
}
SCHEMA_VERSION = 1


def _positive_float(value: float, *, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {value}.")
    return result


@dataclass(frozen=True)
class GaussianFlatsConfig:
    model_dir: Path
    editor_baseline_name: str
    gaussian_flats_root: Path
    python: Path
    run_id: str = "door_planar_official"
    propagation_method: str = "sam2_video"
    planar_region: str = "Door"
    iterations: int = 30_000
    cap_max: int = 1_000_000
    resolution: int = 1
    plane_fit_iter: int = 3_500
    plane_fit_min_points: int = 100
    plane_sigma_res: float = 0.01
    plane_sigma_dist: float = 0.3
    planar_mask_loss_weight: float = 0.1
    depthtv_loss_weight: float = 0.1
    scale_reg: float = 0.01
    opacity_reg: float = 0.01
    planar_grid_resolution: float = 0.02
    planar_tile_size: float = 5.0
    mesh_voxel_size: float = 0.02

    def normalized(self) -> "GaussianFlatsConfig":
        region = str(self.planar_region).strip()
        if not region:
            raise ValueError("Planar region cannot be empty.")
        return GaussianFlatsConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name, label="Editor baseline name"
            ),
            gaussian_flats_root=self.gaussian_flats_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            run_id=path_slug(self.run_id, label="Gaussian Flats run ID"),
            propagation_method=path_slug(
                self.propagation_method, label="Propagation method"
            ),
            planar_region=region,
            iterations=positive_int(self.iterations, label="Iterations"),
            cap_max=positive_int(self.cap_max, label="Gaussian cap"),
            resolution=positive_int(self.resolution, label="Resolution scale"),
            plane_fit_iter=positive_int(
                self.plane_fit_iter, label="Plane fitting iteration"
            ),
            plane_fit_min_points=positive_int(
                self.plane_fit_min_points, label="Minimum plane points"
            ),
            plane_sigma_res=_positive_float(
                self.plane_sigma_res, label="Plane residual sigma"
            ),
            plane_sigma_dist=_positive_float(
                self.plane_sigma_dist, label="Plane distance sigma"
            ),
            planar_mask_loss_weight=_positive_float(
                self.planar_mask_loss_weight, label="Planar mask loss weight"
            ),
            depthtv_loss_weight=_positive_float(
                self.depthtv_loss_weight, label="Depth TV loss weight"
            ),
            scale_reg=_positive_float(self.scale_reg, label="Scale regularization"),
            opacity_reg=_positive_float(
                self.opacity_reg, label="Opacity regularization"
            ),
            planar_grid_resolution=_positive_float(
                self.planar_grid_resolution, label="Planar mesh grid resolution"
            ),
            planar_tile_size=_positive_float(
                self.planar_tile_size, label="Planar mesh tile size"
            ),
            mesh_voxel_size=_positive_float(
                self.mesh_voxel_size, label="TSDF mesh voxel size"
            ),
        )


@dataclass(frozen=True)
class GaussianFlatsPaths:
    run_dir: Path

    @property
    def run_manifest(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def progress(self) -> Path:
        return self.run_dir / "progress.json"

    @property
    def input_dir(self) -> Path:
        return self.run_dir / "01_input"

    @property
    def scene_dir(self) -> Path:
        return self.input_dir / "scene"

    @property
    def plane_masks_dir(self) -> Path:
        return self.input_dir / "plane_masks"

    @property
    def semantic_labels(self) -> Path:
        return self.input_dir / "semantic_label_maps"

    @property
    def frame_map(self) -> Path:
        return self.input_dir / "frames.jsonl"

    @property
    def plane_manifest(self) -> Path:
        return self.input_dir / "planes.json"

    @property
    def model_dir(self) -> Path:
        return self.run_dir / "02_model"

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "03_outputs"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def stage_summary(self, stage: str) -> Path:
        return self.run_dir / "stages" / f"{stage}.json"
