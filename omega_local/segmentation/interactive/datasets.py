"""Dataset profiles and request-scoped state selection for the editor."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .colmap_dataset import PreparedColmapDataset, prepare_colmap_editor_dataset
from .paths import slug_token


DATASET_COOKIE = "omega_editor_dataset"


@dataclass(frozen=True)
class EditorDatasetSpec:
    dataset_id: str
    name: str
    kind: str
    model_dir: Path
    baseline_name: str
    default_rotate_frames: bool
    colmap_model: Path | None = None
    colmap_point_cloud: Path | None = None
    feedforward_point_cloud: Path | None = None
    feedforward_init_mesh: Path | None = None
    omega_final_mesh: Path | None = None

    @property
    def output_dir(self) -> Path:
        return (
            self.model_dir
            / "segmentation"
            / "baselines"
            / self.baseline_name
            / "interactive"
        ).resolve()

    def summary(self, *, active: bool) -> dict[str, Any]:
        return {
            "datasetId": self.dataset_id,
            "name": self.name,
            "kind": self.kind,
            "modelDir": str(self.model_dir),
            "baselineName": self.baseline_name,
            "outputDir": str(self.output_dir),
            "defaultRotateFrames": bool(self.default_rotate_frames),
            "active": bool(active),
        }


class EditorDatasetRegistry:
    def __init__(
        self,
        entries: Iterable[tuple[EditorDatasetSpec, Any]],
        *,
        default_dataset_id: str,
    ) -> None:
        rows = list(entries)
        self.specs = {spec.dataset_id: spec for spec, _state in rows}
        self.states = {spec.dataset_id: state for spec, state in rows}
        if not self.states:
            raise ValueError("The editor dataset registry is empty.")
        if default_dataset_id not in self.states:
            raise ValueError(f"Default editor dataset is not registered: {default_dataset_id}")
        output_roots = [spec.output_dir for spec, _state in rows]
        duplicates = sorted(
            str(path) for path in set(output_roots) if output_roots.count(path) > 1
        )
        if duplicates:
            raise ValueError(
                "Editor datasets must use independent output directories: "
                + ", ".join(duplicates)
            )
        self.default_dataset_id = default_dataset_id
        self._active_id: ContextVar[str] = ContextVar(
            "omega_editor_dataset_id",
            default=default_dataset_id,
        )

    def normalize_id(self, value: str | None) -> str:
        candidate = str(value or "").strip()
        return candidate if candidate in self.states else self.default_dataset_id

    def activate(self, value: str | None) -> Token[str]:
        return self._active_id.set(self.normalize_id(value))

    def reset(self, token: Token[str]) -> None:
        self._active_id.reset(token)

    @property
    def active_id(self) -> str:
        return self._active_id.get()

    @property
    def active_state(self) -> Any:
        return self.states[self.active_id]

    def summary(self) -> dict[str, Any]:
        active_id = self.active_id
        return {
            "schemaVersion": 1,
            "activeDatasetId": active_id,
            "defaultDatasetId": self.default_dataset_id,
            "datasets": [
                self.specs[dataset_id].summary(active=dataset_id == active_id)
                for dataset_id in self.specs
            ],
        }


class ActiveEditorState:
    """Delegate route calls to the dataset selected for the current request."""

    def __init__(self, registry: EditorDatasetRegistry) -> None:
        object.__setattr__(self, "_registry", registry)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._registry.active_state, name)


def load_dataset_specs(path: Path) -> tuple[list[EditorDatasetSpec], str]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Dataset registry must contain a JSON object: {path}")
    rows = payload.get("datasets")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Dataset registry contains no datasets: {path}")

    specs = [_parse_dataset_row(row, config_dir=path.parent) for row in rows]
    ids = [spec.dataset_id for spec in specs]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Dataset registry IDs must be unique: {ids}")
    default_id = str(payload.get("defaultDatasetId") or ids[0])
    if default_id not in set(ids):
        raise ValueError(f"Unknown defaultDatasetId {default_id!r} in {path}")
    return specs, default_id


def single_dataset_spec(
    *,
    model_dir: Path,
    baseline_name: str,
    feedforward_point_cloud: Path | None,
    feedforward_init_mesh: Path | None,
    omega_final_mesh: Path | None,
) -> EditorDatasetSpec:
    model_dir = model_dir.expanduser().resolve()
    return EditorDatasetSpec(
        dataset_id=slug_token(model_dir.name),
        name=model_dir.name,
        kind="omega",
        model_dir=model_dir,
        baseline_name=slug_token(baseline_name),
        default_rotate_frames=True,
        feedforward_point_cloud=feedforward_point_cloud,
        feedforward_init_mesh=feedforward_init_mesh,
        omega_final_mesh=omega_final_mesh,
    )


def _parse_dataset_row(row: Any, *, config_dir: Path) -> EditorDatasetSpec:
    if not isinstance(row, dict):
        raise ValueError(f"Dataset registry rows must be objects, got: {type(row).__name__}")
    dataset_id = slug_token(str(row.get("id") or row.get("name") or ""))
    name = str(row.get("name") or dataset_id)
    kind = str(row.get("kind") or "omega").strip().lower()
    baseline_name = slug_token(str(row.get("baselineName") or "sai3d_area_samples_1024_dense"))
    default_rotate_frames = bool(row.get("defaultRotateFrames", kind == "omega"))

    if kind == "omega":
        model_dir = _config_path(row.get("modelDir"), config_dir=config_dir, required=True)
        return EditorDatasetSpec(
            dataset_id=dataset_id,
            name=name,
            kind=kind,
            model_dir=model_dir,
            baseline_name=baseline_name,
            default_rotate_frames=default_rotate_frames,
            feedforward_point_cloud=_config_path(row.get("feedforwardPointCloud"), config_dir=config_dir),
            feedforward_init_mesh=_config_path(row.get("feedforwardInitMesh"), config_dir=config_dir),
            omega_final_mesh=_config_path(row.get("omegaFinalMesh"), config_dir=config_dir),
        )
    if kind != "colmap":
        raise ValueError(f"Unsupported editor dataset kind {kind!r} for {dataset_id}")

    root_dir = _config_path(row.get("rootDir"), config_dir=config_dir, required=True)
    prepared: PreparedColmapDataset = prepare_colmap_editor_dataset(
        root_dir,
        baseline_name=baseline_name,
        square_images_only=bool(row.get("squareImagesOnly", False)),
    )
    unavailable = prepared.baseline_dir / "dataset" / "unavailable"
    return EditorDatasetSpec(
        dataset_id=dataset_id,
        name=name,
        kind=kind,
        model_dir=prepared.root_dir,
        baseline_name=prepared.baseline_name,
        default_rotate_frames=default_rotate_frames,
        colmap_model=prepared.colmap_model,
        colmap_point_cloud=prepared.point_cloud,
        feedforward_point_cloud=unavailable / "feedforward_points.ply",
        feedforward_init_mesh=unavailable / "init_mesh.ply",
        omega_final_mesh=unavailable / "omega_final_mesh.ply",
    )


def _config_path(
    value: Any,
    *,
    config_dir: Path,
    required: bool = False,
) -> Path | None:
    if value in (None, ""):
        if required:
            raise ValueError("Dataset registry path is required.")
        return None
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()
