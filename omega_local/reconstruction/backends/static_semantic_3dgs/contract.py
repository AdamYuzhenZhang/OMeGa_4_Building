"""Run contract shared by hard- and soft-identity static 3DGS methods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omega_local.segmentation.split_splat.contract import path_slug, positive_int


SCHEMA_VERSION = 1
METHODS = ("segment_then_splat", "gaussian_grouping")
STAGE_ORDER = ("prepare", "train", "export")
STAGE_DEPENDENCIES = {
    "prepare": (),
    "train": ("prepare",),
    "export": ("train",),
}

_DEFAULT_ITERATIONS = {
    "segment_then_splat": 40_000,
    "gaussian_grouping": 30_000,
}
_DEFAULT_DENSIFY_UNTIL = {
    "segment_then_splat": 20_000,
    "gaussian_grouping": 10_000,
}


def _method(value: str) -> str:
    result = str(value).strip().lower().replace("-", "_")
    if result not in METHODS:
        raise ValueError(
            f"Unknown static semantic 3DGS method {value!r}; choose one of "
            f"{', '.join(METHODS)}."
        )
    return result


def _unit_interval(value: float, *, label: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1], got {value}.")
    return result


def _positive_float(value: float, *, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {value}.")
    return result


@dataclass(frozen=True)
class StaticSemantic3DGSConfig:
    model_dir: Path
    editor_baseline_name: str
    method: str
    method_root: Path
    python: Path
    mapanything_run_id: str = "mapanything_sam2_video"
    run_id: str = "joint_user_masks"
    iterations: int | None = None
    resolution: int = 1
    densify_until_iter: int | None = None
    num_sample_objects: int = 3
    partial_mask_iou: float = 0.3
    reg3d_interval: int = 5
    reg3d_k: int = 5
    reg3d_lambda: float = 2.0
    reg3d_max_points: int = 200_000
    reg3d_sample_size: int = 1_000
    native_extensions_dir: Path | None = None

    def normalized(self) -> "StaticSemantic3DGSConfig":
        method = _method(self.method)
        iterations = (
            _DEFAULT_ITERATIONS[method]
            if self.iterations is None
            else positive_int(self.iterations, label="Iterations")
        )
        densify_until = (
            _DEFAULT_DENSIFY_UNTIL[method]
            if self.densify_until_iter is None
            else positive_int(
                self.densify_until_iter,
                label="Densify-until iteration",
            )
        )
        if densify_until > iterations:
            raise ValueError(
                "Densify-until iteration cannot exceed total iterations: "
                f"{densify_until} > {iterations}."
            )
        resolution = positive_int(self.resolution, label="Resolution scale")
        if method == "gaussian_grouping" and resolution != 1:
            raise ValueError(
                "Gaussian Grouping's released loader does not resize indexed "
                "object masks with RGB; use --resolution 1."
            )
        return StaticSemantic3DGSConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name,
                label="Editor baseline name",
            ),
            method=method,
            method_root=self.method_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            mapanything_run_id=path_slug(
                self.mapanything_run_id,
                label="MapAnything run ID",
            ),
            run_id=path_slug(self.run_id, label="Run ID"),
            iterations=iterations,
            resolution=resolution,
            densify_until_iter=densify_until,
            num_sample_objects=positive_int(
                self.num_sample_objects,
                label="Sampled objects per iteration",
            ),
            partial_mask_iou=_unit_interval(
                self.partial_mask_iou,
                label="Partial-mask IoU",
            ),
            reg3d_interval=positive_int(
                self.reg3d_interval,
                label="3D regularization interval",
            ),
            reg3d_k=positive_int(
                self.reg3d_k,
                label="3D regularization neighbors",
            ),
            reg3d_lambda=_positive_float(
                self.reg3d_lambda,
                label="3D regularization weight",
            ),
            reg3d_max_points=positive_int(
                self.reg3d_max_points,
                label="3D regularization maximum points",
            ),
            reg3d_sample_size=positive_int(
                self.reg3d_sample_size,
                label="3D regularization sample size",
            ),
            native_extensions_dir=(
                self.native_extensions_dir.expanduser().resolve()
                if self.native_extensions_dir is not None
                else None
            ),
        )

    @property
    def display_name(self) -> str:
        return {
            "segment_then_splat": "Segment then Splat",
            "gaussian_grouping": "Gaussian Grouping",
        }[self.method]


@dataclass(frozen=True)
class StaticSemantic3DGSPaths:
    source_run_dir: Path
    family_dir: Path
    run_dir: Path

    @property
    def shared_dir(self) -> Path:
        return self.family_dir / "00_shared_dataset"

    @property
    def shared_images(self) -> Path:
        return self.shared_dir / "images"

    @property
    def shared_masks(self) -> Path:
        return self.shared_dir / "masks"

    @property
    def shared_mask_pngs(self) -> Path:
        return self.shared_dir / "masks_png"

    @property
    def shared_sparse(self) -> Path:
        return self.shared_dir / "sparse" / "0"

    @property
    def shared_initializer(self) -> Path:
        return self.shared_sparse / "points3D.ply"

    @property
    def shared_initializer_labels(self) -> Path:
        return self.shared_dir / "initializer_region_ids.npy"

    @property
    def shared_frame_map(self) -> Path:
        return self.shared_dir / "frame_map.jsonl"

    @property
    def shared_region_map(self) -> Path:
        return self.shared_dir / "region_id_map.json"

    @property
    def shared_training_weights(self) -> Path:
        return self.shared_dir / "training_view_weights.json"

    @property
    def shared_summary(self) -> Path:
        return self.shared_dir / "stage.json"

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
    def dataset_dir(self) -> Path:
        return self.run_dir / "01_dataset"

    @property
    def model_output(self) -> Path:
        return self.run_dir / "02_model"

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "03_outputs"

    @property
    def scene_ply(self) -> Path:
        return self.outputs_dir / "scene.ply"

    @property
    def gaussian_region_ids(self) -> Path:
        return self.outputs_dir / "gaussian_region_ids.npy"

    @property
    def regions_manifest(self) -> Path:
        return self.outputs_dir / "regions.json"

    @property
    def object_splats_dir(self) -> Path:
        return self.outputs_dir / "objects"

    @property
    def output_manifest(self) -> Path:
        return self.outputs_dir / "manifest.json"

    def stage_summary(self, stage: str) -> Path:
        return {
            "prepare": self.dataset_dir,
            "train": self.model_output,
            "export": self.outputs_dir,
        }[stage] / "stage.json"
