"""Run contract for joint ObjectGS reconstruction from a MapAnything split."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omega_local.segmentation.split_splat.contract import path_slug, positive_int


SCHEMA_VERSION = 1
STAGE_ORDER = ("prepare", "train", "export", "mesh")
STAGE_DEPENDENCIES = {
    "prepare": (),
    "train": ("prepare",),
    "export": ("train",),
    "mesh": ("export",),
}


def _positive_float(value: float, *, label: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {value}.")
    return result


def _nonnegative_float(value: float, *, label: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{label} cannot be negative, got {value}.")
    return result


@dataclass(frozen=True)
class ObjectGSConfig:
    model_dir: Path
    editor_baseline_name: str
    objectgs_root: Path
    python: Path
    mapanything_run_id: str = "mapanything_sam2_video"
    run_id: str = "objectgs_joint"
    iterations: int = 30_000
    semantic_loss_weight: float = 0.1
    initializer_stride: int = 1
    voxel_size: float = 0.001
    geometry_completed_as_unknown: bool = True
    mesh_voxel_size: float = 0.01
    mesh_resolution: int = 512
    mesh_clusters: int = 10
    mesh_max_triangles: int = 1_000_000

    def normalized(self) -> "ObjectGSConfig":
        return ObjectGSConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name,
                label="Editor baseline name",
            ),
            objectgs_root=self.objectgs_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            mapanything_run_id=path_slug(
                self.mapanything_run_id,
                label="MapAnything run ID",
            ),
            run_id=path_slug(self.run_id, label="ObjectGS run ID"),
            iterations=positive_int(self.iterations, label="Iterations"),
            semantic_loss_weight=_positive_float(
                self.semantic_loss_weight,
                label="Semantic loss weight",
            ),
            initializer_stride=positive_int(
                self.initializer_stride,
                label="Initializer stride",
            ),
            voxel_size=_nonnegative_float(
                self.voxel_size,
                label="Voxel size",
            ),
            geometry_completed_as_unknown=bool(
                self.geometry_completed_as_unknown
            ),
            mesh_voxel_size=_nonnegative_float(
                self.mesh_voxel_size,
                label="Mesh voxel size",
            ),
            mesh_resolution=positive_int(
                self.mesh_resolution,
                label="Mesh resolution",
            ),
            mesh_clusters=positive_int(
                self.mesh_clusters,
                label="Mesh clusters",
            ),
            mesh_max_triangles=positive_int(
                self.mesh_max_triangles,
                label="Mesh maximum triangles",
            ),
        )


@dataclass(frozen=True)
class ObjectGSPaths:
    source_run_dir: Path
    run_dir: Path

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
    def frame_map(self) -> Path:
        return self.dataset_dir / "frame_map.jsonl"

    @property
    def region_map(self) -> Path:
        return self.dataset_dir / "region_id_map.json"

    @property
    def labeled_initializer(self) -> Path:
        return self.dataset_dir / "sparse" / "0" / "points3D.ply"

    @property
    def config_dir(self) -> Path:
        return self.run_dir / "02_config"

    @property
    def train_config(self) -> Path:
        return self.config_dir / "config.yaml"

    @property
    def model_parent(self) -> Path:
        return self.run_dir / "03_model"

    @property
    def model_pointer(self) -> Path:
        return self.model_parent / "model.json"

    @property
    def outputs_dir(self) -> Path:
        return self.run_dir / "04_outputs"

    @property
    def anchor_cache(self) -> Path:
        return self.outputs_dir / "anchors" / "anchors.npz"

    @property
    def colored_anchors(self) -> Path:
        return self.outputs_dir / "anchors" / "anchors_regions.ply"

    @property
    def predicted_label_maps(self) -> Path:
        return self.outputs_dir / "predicted_masks" / "label_maps"

    @property
    def predicted_overlays(self) -> Path:
        return self.outputs_dir / "predicted_masks" / "overlays"

    @property
    def rgb_renders(self) -> Path:
        return self.outputs_dir / "rgb_renders"

    @property
    def meshes_dir(self) -> Path:
        return self.run_dir / "05_meshes"

    @property
    def mesh_manifest(self) -> Path:
        return self.meshes_dir / "mesh_manifest.json"

    def stage_summary(self, stage: str) -> Path:
        directories = {
            "prepare": self.dataset_dir,
            "train": self.model_parent,
            "export": self.outputs_dir,
            "mesh": self.meshes_dir,
        }
        return directories[stage] / "stage.json"
