"""Configuration and paths for the released Segment then Splat pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from omega_local.segmentation.split_splat.contract import path_slug, positive_int


SCHEMA_VERSION = 1
LEVELS = ("large", "middle", "small")
STAGE_ORDER = ("prepare", "autoseg", "initialize", "train", "export")
STAGE_DEPENDENCIES = {
    "prepare": (),
    "autoseg": ("prepare",),
    "initialize": ("autoseg",),
    "train": ("initialize",),
    "export": ("train",),
}


@dataclass(frozen=True)
class OfficialSegmentThenSplatConfig:
    model_dir: Path
    editor_baseline_name: str
    source_root: Path
    python: Path
    sam1_checkpoint: Path
    sam2_checkpoint: Path
    native_extensions_dir: Path
    run_id: str = "official_auto"
    image_long_side: int = 1024
    detect_stride: int = 10
    batch_size: int = 20
    iterations: int = 40_000
    densify_until_iter: int = 20_000
    num_sample_objects: int = 3
    partial_mask_iou: float = 0.3

    def normalized(self) -> "OfficialSegmentThenSplatConfig":
        iterations = positive_int(self.iterations, label="Iterations")
        densify = positive_int(
            self.densify_until_iter,
            label="Densify-until iteration",
        )
        if densify > iterations:
            raise ValueError("Densify-until iteration cannot exceed iterations.")
        partial = float(self.partial_mask_iou)
        if not 0.0 <= partial <= 1.0:
            raise ValueError("Partial-mask IoU must be in [0, 1].")
        return OfficialSegmentThenSplatConfig(
            model_dir=self.model_dir.expanduser().resolve(),
            editor_baseline_name=path_slug(
                self.editor_baseline_name,
                label="Editor baseline name",
            ),
            source_root=self.source_root.expanduser().resolve(),
            python=self.python.expanduser().absolute(),
            sam1_checkpoint=self.sam1_checkpoint.expanduser().resolve(),
            sam2_checkpoint=self.sam2_checkpoint.expanduser().resolve(),
            native_extensions_dir=self.native_extensions_dir.expanduser().resolve(),
            run_id=path_slug(self.run_id, label="Run ID"),
            image_long_side=positive_int(
                self.image_long_side,
                label="Image long side",
            ),
            detect_stride=positive_int(self.detect_stride, label="Detect stride"),
            batch_size=positive_int(self.batch_size, label="Batch size"),
            iterations=iterations,
            densify_until_iter=densify,
            num_sample_objects=positive_int(
                self.num_sample_objects,
                label="Sampled objects per iteration",
            ),
            partial_mask_iou=partial,
        )


@dataclass(frozen=True)
class OfficialSegmentThenSplatPaths:
    run_dir: Path

    @property
    def run_manifest(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def progress(self) -> Path:
        return self.run_dir / "progress.json"

    @property
    def dataset_stage(self) -> Path:
        return self.run_dir / "01_dataset"

    @property
    def scene_dir(self) -> Path:
        return self.dataset_stage / "scene"

    @property
    def images_dir(self) -> Path:
        return self.scene_dir / "images"

    @property
    def sparse_dir(self) -> Path:
        return self.scene_dir / "sparse" / "0"

    @property
    def frame_map(self) -> Path:
        return self.dataset_stage / "frame_map.jsonl"

    @property
    def autoseg_stage(self) -> Path:
        return self.run_dir / "02_autoseg"

    @property
    def autoseg_output(self) -> Path:
        return self.autoseg_stage / "output"

    @property
    def autoseg_runtime(self) -> Path:
        return self.autoseg_stage / "runtime"

    @property
    def initialization_stage(self) -> Path:
        return self.run_dir / "03_initialization"

    @property
    def label_maps_root(self) -> Path:
        return self.initialization_stage / "label_maps"

    @property
    def overlays_root(self) -> Path:
        return self.initialization_stage / "overlays"

    @property
    def model_stage(self) -> Path:
        return self.run_dir / "04_model"

    @property
    def model_output(self) -> Path:
        return self.model_stage / "model"

    @property
    def outputs_stage(self) -> Path:
        return self.run_dir / "05_outputs"

    @property
    def scene_ply(self) -> Path:
        return self.outputs_stage / "scene.ply"

    @property
    def object_splats(self) -> Path:
        return self.outputs_stage / "objects"

    @property
    def output_manifest(self) -> Path:
        return self.outputs_stage / "manifest.json"

    def stage_summary(self, stage: str) -> Path:
        roots = {
            "prepare": self.dataset_stage,
            "autoseg": self.autoseg_stage,
            "initialize": self.initialization_stage,
            "train": self.model_stage,
            "export": self.outputs_stage,
        }
        return roots[stage] / "stage.json"

