"""Registry and shared contracts for candidate generation and dense recovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

from .colmap_dense_propagation import (
    ColmapDenseConfig,
    ColmapDensePropagationSession,
    colmap_dense_availability,
)
from .colmap_identity_refinement import (
    ColmapIdentityRefinementConfig,
    ColmapIdentityRefinementSession,
    colmap_identity_refinement_availability,
)
from .colmap_sam2_prompt_recovery import (
    ColmapSam2PromptConfig,
    ColmapSam2PromptRecoverySession,
    colmap_sam2_prompt_availability,
)
from .colmap_track_propagation import (
    ColmapTrackConfig,
    ColmapTrackPropagationSession,
    colmap_track_availability,
)
from .feedforward_point_propagation import (
    FeedForwardPointConfig,
    FeedForwardPointPropagationSession,
    feedforward_point_availability,
)
from .memory_vos_propagation import MemoryVOSConfig, MemoryVOSPropagationSession
from .multifield_crf_recovery import (
    MultiFieldCrfConfig,
    MultiFieldCrfRecoverySession,
    multifield_crf_availability,
)
from .sam2_bounded_propagation import Sam2BoundedPropagationSession
from .sam2_session import Sam2Config, Sam2Session
from .sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
    Sam2VideoPropagationSession,
)
from .v2sam_propagation import V2SamConfig, V2SamPropagationSession
from .vggts_propagation import VggtSConfig, VggtSPropagationSession


ProgressCallback = Callable[[int, int, int], None]
SaveCallback = Callable[[PropagationFrame, np.ndarray], None]


class PropagationSession(Protocol):
    def propagate_labeled_sources(
        self,
        *,
        frames: list[PropagationFrame],
        sources: list[LabeledPropagationSource],
        start_local_index: int,
        region_rows: list[dict[str, Any]],
        progress_callback: ProgressCallback | None = None,
        save_callback: SaveCallback | None = None,
    ) -> list[dict[str, Any]]: ...


class DenseRecoverySession(Protocol):
    def recover_from_source(
        self,
        *,
        run_input: "PropagationRunInput",
        progress_callback: ProgressCallback | None = None,
        save_callback: SaveCallback | None = None,
    ) -> list[dict[str, Any]]: ...


class IdentityRefinementSession(Protocol):
    def refine_from_source(
        self,
        *,
        run_input: "PropagationRunInput",
        progress_callback: ProgressCallback | None = None,
        save_callback: SaveCallback | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class PropagationBackendInfo:
    method_id: str
    display_name: str
    engine_name: str
    description: str
    result_description: str
    runtime_config: dict[str, Any]
    supports_full_run: bool = True
    supports_region_pair: bool = False
    pair_output_policy: str = ""
    stage: str = "anchor_to_mask"
    source_method_id: str = ""
    layer_group: str = "video_propagation"
    label_space: str = "persistent_region"
    read_only: bool = False

    @property
    def layer_key(self) -> str:
        return propagation_layer_key(self.method_id)


@dataclass(frozen=True)
class PropagationBackend:
    info: PropagationBackendInfo
    session: PropagationSession | DenseRecoverySession | IdentityRefinementSession
    availability_check: Callable[[], tuple[bool, str]]

    def status(self) -> dict[str, Any]:
        available, reason = self.availability_check()
        return {
            "methodId": self.info.method_id,
            "layerKey": self.info.layer_key,
            "displayName": self.info.display_name,
            "engineName": self.info.engine_name,
            "description": self.info.description,
            "labelSpace": str(self.info.label_space),
            "readOnly": bool(self.info.read_only),
            "supportsFullRun": bool(self.info.supports_full_run),
            "supportsRegionPair": bool(self.info.supports_region_pair),
            "pairOutputPolicy": str(self.info.pair_output_policy),
            "stage": str(self.info.stage),
            "sourceMethodId": str(self.info.source_method_id),
            "sourceLayerKey": (
                propagation_layer_key(self.info.source_method_id)
                if self.info.source_method_id
                else ""
            ),
            "layerGroup": str(self.info.layer_group),
            "available": bool(available),
            "availabilityMessage": reason,
        }

    def run(
        self,
        run_input: "PropagationRunInput",
        *,
        progress_callback: ProgressCallback | None,
        save_callback: SaveCallback,
    ) -> list[dict[str, Any]]:
        if not self.info.supports_full_run:
            raise ValueError(f"{self.info.display_name} is available only as a focused region-pair test.")
        available, reason = self.availability_check()
        if not available:
            raise FileNotFoundError(f"{self.info.display_name} is unavailable: {reason}")
        if self.info.stage == "dense_recovery":
            runner = getattr(self.session, "recover_from_source", None)
            if not callable(runner):
                raise TypeError(f"{self.info.display_name} has no dense-recovery implementation.")
            return runner(
                run_input=run_input,
                progress_callback=progress_callback,
                save_callback=save_callback,
            )
        if self.info.stage == "identity_refinement":
            runner = getattr(self.session, "refine_from_source", None)
            if not callable(runner):
                raise TypeError(f"{self.info.display_name} has no identity-refinement implementation.")
            return runner(
                run_input=run_input,
                progress_callback=progress_callback,
                save_callback=save_callback,
            )
        return self.session.propagate_labeled_sources(
            frames=list(run_input.frames),
            sources=list(run_input.sources),
            start_local_index=int(run_input.primary_index),
            region_rows=[dict(row) for row in run_input.region_rows],
            progress_callback=progress_callback,
            save_callback=save_callback,
        )

    def run_region_pair(
        self,
        *,
        frames: list[PropagationFrame],
        source: LabeledPropagationSource,
        target_local_index: int,
        region_id: int,
        region_rows: list[dict[str, Any]],
        progress_callback: ProgressCallback | None,
        save_callback: SaveCallback,
    ) -> dict[str, Any]:
        if not self.info.supports_region_pair:
            raise ValueError(f"{self.info.display_name} does not support focused region-pair tests.")
        available, reason = self.availability_check()
        if not available:
            raise FileNotFoundError(f"{self.info.display_name} is unavailable: {reason}")
        runner = getattr(self.session, "propagate_single_region", None)
        if not callable(runner):
            raise TypeError(f"{self.info.display_name} has no region-pair inference implementation.")
        return runner(
            frames=list(frames),
            source=source,
            target_local_index=int(target_local_index),
            region_id=int(region_id),
            region_rows=[dict(row) for row in region_rows],
            progress_callback=progress_callback,
            save_callback=save_callback,
        )


class PropagationBackendRegistry:
    def __init__(self, backends: list[PropagationBackend]) -> None:
        if not backends:
            raise ValueError("At least one propagation backend must be registered.")
        self._backends: dict[str, PropagationBackend] = {}
        self._order: list[str] = []
        for backend in backends:
            method_id = normalize_method_id(backend.info.method_id)
            if method_id != backend.info.method_id:
                raise ValueError(f"Propagation method ID must be normalized: {backend.info.method_id!r}")
            if method_id in self._backends:
                raise ValueError(f"Duplicate propagation method ID: {method_id}")
            self._backends[method_id] = backend
            self._order.append(method_id)
        for method_id in self._order:
            info = self._backends[method_id].info
            if info.stage not in {"anchor_to_mask", "dense_recovery", "identity_refinement"}:
                raise ValueError(f"Unsupported propagation stage for {method_id}: {info.stage}")
            if info.source_method_id:
                source_id = normalize_method_id(info.source_method_id)
                if source_id == method_id:
                    raise ValueError(f"Propagation method cannot depend on itself: {method_id}")
                if source_id not in self._backends:
                    raise ValueError(
                        f"Propagation method {method_id} requires unregistered source {source_id}."
                    )
                if not self._backends[source_id].info.supports_full_run:
                    raise ValueError(
                        f"Propagation method {method_id} requires partial-only source {source_id}."
                    )

    @property
    def default_method_id(self) -> str:
        return next(
            method_id
            for method_id in self._order
            if (
                self._backends[method_id].info.stage == "anchor_to_mask"
                and self._backends[method_id].info.supports_full_run
            )
        )

    @property
    def default_source_refinement_method_id(self) -> str:
        return next(
            (
                method_id
                for method_id in self._order
                if self._backends[method_id].info.stage
                in {"dense_recovery", "identity_refinement"}
            ),
            "",
        )

    def get(self, method_id: str) -> PropagationBackend:
        key = normalize_method_id(method_id)
        backend = self._backends.get(key)
        if backend is None:
            expected = ", ".join(self._order)
            raise ValueError(f"Unknown propagation method '{method_id}'. Expected one of: {expected}.")
        return backend

    def methods(self) -> list[PropagationBackend]:
        return [self._backends[method_id] for method_id in self._order]

    def status(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "defaultMethodId": self.default_method_id,
            "defaultSourceRefinementMethodId": self.default_source_refinement_method_id,
            "methods": [backend.status() for backend in self.methods()],
        }


@dataclass(frozen=True)
class PropagationRunInput:
    trigger_frame_id: int
    frames: tuple[PropagationFrame, ...]
    sources: tuple[LabeledPropagationSource, ...]
    region_rows: tuple[dict[str, Any], ...]
    source_frame_ids_by_region: dict[int, tuple[int, ...]]
    source_area_pixels: dict[int, int]
    known_region_ids: frozenset[int]
    anchor_labels_by_frame: dict[int, np.ndarray]
    anchor_map_paths: dict[int, Path]
    primary_index: int
    fingerprint: str

    @property
    def primary_frame_id(self) -> int:
        return int(self.frames[self.primary_index].frame_id)

    @property
    def anchor_frame_ids(self) -> list[int]:
        return sorted(int(source.frame_id) for source in self.sources)

    @property
    def source_region_ids(self) -> list[int]:
        return sorted(self.source_frame_ids_by_region)

    @property
    def source_area_total(self) -> int:
        return int(sum(self.source_area_pixels.values()))

    def validate_output(self, frame: PropagationFrame, labels: np.ndarray) -> np.ndarray:
        values = np.asarray(labels)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D region candidate map for frame {frame.frame_id}, got {values.shape}.")
        expected_shape = (int(frame.height), int(frame.width))
        if values.shape != expected_shape:
            raise ValueError(
                f"Region candidate map shape {values.shape} does not match frame {frame.frame_id} shape {expected_shape}."
            )
        if np.issubdtype(values.dtype, np.floating):
            if not np.all(np.isfinite(values)) or not np.array_equal(values, np.rint(values)):
                raise ValueError(f"Region candidate map for frame {frame.frame_id} contains non-integral labels.")
        values_i64 = values.astype(np.int64, copy=False)
        if np.any(values_i64 < 0) or np.any(values_i64 > np.iinfo(np.uint16).max):
            raise ValueError(f"Region candidate map for frame {frame.frame_id} contains labels outside uint16 range.")
        positive_ids = {int(value) for value in np.unique(values_i64).tolist() if int(value) > 0}
        unknown = positive_ids - set(self.known_region_ids)
        if unknown:
            raise ValueError(
                f"Region candidate map for frame {frame.frame_id} contains unknown persistent IDs: {sorted(unknown)}."
            )
        anchor = self.anchor_labels_by_frame.get(int(frame.frame_id))
        if anchor is not None:
            return anchor.astype(np.uint16, copy=True)
        return values_i64.astype(np.uint16, copy=True)

    def to_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "labelSpace": "persistent_region",
            "triggerFrameId": int(self.trigger_frame_id),
            "primaryFrameId": int(self.primary_frame_id),
            "primaryPolicy": "sequence_first_frame",
            "frameCount": int(len(self.frames)),
            "anchorFrameIds": self.anchor_frame_ids,
            "anchorMaps": [
                {
                    "frameId": int(frame_id),
                    "path": str(path),
                }
                for frame_id, path in sorted(self.anchor_map_paths.items())
            ],
            "sourceRegionIds": self.source_region_ids,
            "sourceFrameIdsByRegion": {
                str(region_id): list(frame_ids)
                for region_id, frame_ids in sorted(self.source_frame_ids_by_region.items())
            },
            "sourceAreaPixelsByRegion": {
                str(region_id): int(area)
                for region_id, area in sorted(self.source_area_pixels.items())
            },
            "sourceAreaPixels": self.source_area_total,
            "fingerprint": self.fingerprint,
        }


def build_propagation_input(
    *,
    trigger_frame_id: int,
    frames: list[PropagationFrame],
    region_status: dict[str, Any],
    region_maps_dir: Path,
) -> PropagationRunInput:
    if not frames:
        raise ValueError("Propagation needs at least one staged frame.")
    frame_index_by_id = {int(frame.frame_id): index for index, frame in enumerate(frames)}
    if int(trigger_frame_id) not in frame_index_by_id:
        raise KeyError(trigger_frame_id)

    region_rows_all = [dict(row) for row in region_status.get("regions", []) if isinstance(row, dict)]
    known_region_ids = {
        int(row.get("id", 0))
        for row in region_rows_all
        if 0 < int(row.get("id", 0)) <= np.iinfo(np.uint16).max
    }
    if not known_region_ids:
        raise ValueError("No persistent regions exist yet.")

    complete_frame_ids = sorted(
        {
            int(frame_id)
            for frame_id in region_status.get("completeFrames", [])
            if int(frame_id) in frame_index_by_id
        },
        key=frame_index_by_id.__getitem__,
    )
    if not complete_frame_ids:
        raise ValueError("No complete persistent-region keyframes exist yet. Mark at least one edited frame complete before propagating.")

    sources: list[LabeledPropagationSource] = []
    anchor_labels_by_frame: dict[int, np.ndarray] = {}
    anchor_map_paths: dict[int, Path] = {}
    source_frame_ids_by_region: dict[int, set[int]] = {}
    source_area_pixels: dict[int, int] = {}
    digest = hashlib.sha256()
    digest.update(b"omega-interactive-propagation-input-v1\0")
    for frame in frames:
        image_path = frame.image_path.expanduser().resolve()
        stat = image_path.stat()
        digest.update(
            (
                f"frame:{int(frame.frame_id)}:{int(frame.width)}x{int(frame.height)}:"
                f"{image_path}:{int(stat.st_size)}:{int(stat.st_mtime_ns)}\0"
            ).encode("utf-8")
        )
    for region_id in sorted(known_region_ids):
        digest.update(f"region:{region_id}\0".encode("ascii"))

    for frame_id in complete_frame_ids:
        frame = frames[frame_index_by_id[frame_id]]
        path = region_maps_dir / f"{frame_id:06d}.npy"
        if not path.exists():
            continue
        labels = np.load(path)
        if labels.ndim != 2:
            raise ValueError(f"Expected a 2D persistent region map for frame {frame_id}: {path}")
        expected_shape = (int(frame.height), int(frame.width))
        if labels.shape != expected_shape:
            raise ValueError(f"Persistent region map shape {labels.shape} does not match frame {frame_id} shape {expected_shape}.")
        labels_i64 = labels.astype(np.int64, copy=False)
        positive_ids = {int(value) for value in np.unique(labels_i64).tolist() if int(value) > 0}
        unknown = positive_ids - known_region_ids
        if unknown:
            raise ValueError(f"Persistent region map for frame {frame_id} contains deleted or unknown IDs: {sorted(unknown)}.")
        if not positive_ids:
            continue

        labels_u16 = labels_i64.astype(np.uint16, copy=True)
        local_index = frame_index_by_id[frame_id]
        sources.append(LabeledPropagationSource(local_index=local_index, frame_id=frame_id, labels=labels_u16))
        anchor_labels_by_frame[frame_id] = labels_u16
        anchor_map_paths[frame_id] = path.resolve()
        digest.update(str(frame_id).encode("ascii"))
        digest.update(b"\0")
        digest.update(labels_u16.tobytes(order="C"))
        for region_id in sorted(positive_ids):
            source_frame_ids_by_region.setdefault(region_id, set()).add(frame_id)
            source_area_pixels[region_id] = source_area_pixels.get(region_id, 0) + int(
                np.count_nonzero(labels_u16 == region_id)
            )

    if not sources:
        raise ValueError("Complete frames contain no positive persistent-region pixels to propagate.")

    source_region_ids = set(source_frame_ids_by_region)
    region_rows = tuple(row for row in region_rows_all if int(row.get("id", 0)) in source_region_ids)
    return PropagationRunInput(
        trigger_frame_id=int(trigger_frame_id),
        frames=tuple(frames),
        sources=tuple(sources),
        region_rows=region_rows,
        source_frame_ids_by_region={
            region_id: tuple(sorted(frame_ids))
            for region_id, frame_ids in sorted(source_frame_ids_by_region.items())
        },
        source_area_pixels=dict(sorted(source_area_pixels.items())),
        known_region_ids=frozenset(known_region_ids),
        anchor_labels_by_frame=anchor_labels_by_frame,
        anchor_map_paths=anchor_map_paths,
        primary_index=0,
        fingerprint=digest.hexdigest(),
    )


def build_default_propagation_registry(
    *,
    colmap_track_config: ColmapTrackConfig,
    colmap_identity_config: ColmapIdentityRefinementConfig,
    colmap_dense_config: ColmapDenseConfig,
    multifield_crf_config: MultiFieldCrfConfig,
    feedforward_point_config: FeedForwardPointConfig,
    feedforward_dense_config: ColmapDenseConfig,
    feedforward_multifield_crf_config: MultiFieldCrfConfig,
    omega_final_point_config: FeedForwardPointConfig,
    omega_final_dense_config: ColmapDenseConfig,
    omega_final_multifield_crf_config: MultiFieldCrfConfig,
    sam2_config: Sam2Config,
    sam2_image_session: Sam2Session,
    xmem_config: MemoryVOSConfig,
    cutie_config: MemoryVOSConfig,
    v2sam_config: V2SamConfig,
    vggts_config: VggtSConfig,
    work_root: Path,
    omega_root: Path,
) -> PropagationBackendRegistry:
    sam2_requirements = (
        (sam2_config.root, "SAM2 root", "dir"),
        (sam2_config.root / "sam2" / "build_sam.py", "SAM2 build_sam.py", "file"),
        (sam2_config.checkpoint, "SAM2 checkpoint", "file"),
    )
    xmem_requirements = (
        (xmem_config.root, "XMem++ root", "dir"),
        (xmem_config.root / "inference" / "run_on_video.py", "XMem++ runner", "file"),
        (xmem_config.checkpoint, "XMem++ checkpoint", "file"),
    )
    cutie_requirements = (
        (cutie_config.root, "Cutie root", "dir"),
        (cutie_config.root / "cutie" / "inference" / "inference_core.py", "Cutie inference core", "file"),
        (cutie_config.root / "cutie" / "config" / "video_config.yaml", "Cutie video config", "file"),
        (cutie_config.checkpoint, "Cutie checkpoint", "file"),
    )
    v2sam_requirements = (
        (v2sam_config.root, "V2-SAM root", "dir"),
        (v2sam_config.root / "projects" / "v2sam_fusion" / "models" / "v2sam.py", "V2-SAM Fusion implementation", "file"),
        (v2sam_config.root / "projects" / "v2sam_visual" / "models" / "v2sam.py", "V2-SAM Visual implementation", "file"),
        (v2sam_config.python, "V2-SAM Python", "file"),
        (v2sam_config.sam_checkpoint, "V2-SAM SAM2 checkpoint", "file"),
        (v2sam_config.dino_checkpoint, "V2-SAM DINOv3 checkpoint", "file"),
        (v2sam_config.visual_checkpoint, "V2-SAM Visual checkpoint", "file"),
        (v2sam_config.fusion_checkpoint, "V2-SAM Fusion checkpoint", "file"),
    )
    vggts_requirements = (
        (vggts_config.root, "VGGT-S root", "dir"),
        (vggts_config.root / "src" / "model" / "predictor.py", "VGGT-S predictor", "file"),
        (vggts_config.python, "VGGT-S Python", "file"),
        (vggts_config.checkpoint, "VGGT-S checkpoint", "file"),
    )

    sam2_runtime = {
        "root": str(sam2_config.root),
        "checkpoint": str(sam2_config.checkpoint),
        "config": str(sam2_config.config),
        "device": str(sam2_config.device),
        "maskDecision": "logit_greater_than_zero",
        "anchorPolicy": "all_complete_frames_as_conditioning_masks",
    }
    sam2_bounded_runtime = {
        **sam2_runtime,
        "intervalPolicy": "adjacent_complete_anchor_intervals",
        "passes": ["forward_from_left_anchor", "backward_from_right_anchor"],
        "conflictLogitMargin": 0.20,
        "conflictPolicy": "distance_weighted_logit_or_unknown",
    }
    xmem_runtime = {
        "root": str(xmem_config.root),
        "checkpoint": str(xmem_config.checkpoint),
        "internalShortEdge": int(xmem_config.internal_size),
        "device": str(xmem_config.device),
        "anchorMemory": "permanent",
        "memoryEveryFrames": 10,
        "parameterSource": "official_inference_defaults",
    }
    cutie_runtime = {
        "root": str(cutie_config.root),
        "checkpoint": str(cutie_config.checkpoint),
        "internalShortEdge": int(cutie_config.internal_size),
        "device": str(cutie_config.device),
        "anchorMemory": "permanent",
        "memoryEveryFrames": 10,
        "parameterSource": "official_video_config_defaults",
    }
    v2sam_runtime = {
        "root": str(v2sam_config.root),
        "python": str(v2sam_config.python),
        "expertProfile": str(v2sam_config.expert_profile),
        "samCheckpoint": str(v2sam_config.sam_checkpoint),
        "dinoCheckpoint": str(v2sam_config.dino_checkpoint),
        "visualCheckpoint": str(v2sam_config.visual_checkpoint),
        "fusionCheckpoint": str(v2sam_config.fusion_checkpoint),
        "dinoEvidenceDirectory": str(v2sam_config.feature_cache_dir),
        "diagnosticsDirectory": str(v2sam_config.diagnostics_dir),
        "foregroundThreshold": float(v2sam_config.foreground_threshold),
        "sourcePolicy": "earliest_manual_occurrence",
        "outputPolicy": "direct_pccs_selected_expert",
        "device": str(v2sam_config.device),
    }
    vggts_runtime = {
        "root": str(vggts_config.root),
        "python": str(vggts_config.python),
        "checkpoint": str(vggts_config.checkpoint),
        "vggtModelId": str(vggts_config.vggt_model_id),
        "diagnosticsDirectory": str(vggts_config.diagnostics_dir),
        "imageSize": int(vggts_config.image_size),
        "locatorPoints": int(vggts_config.locator_points),
        "locatorOutliers": int(vggts_config.locator_outliers),
        "promptPoints": int(vggts_config.prompt_points),
        "refinementStepsAfterInitial": int(vggts_config.refinement_steps),
        "sourcePolicy": "earliest_manual_occurrence",
        "parameterSource": "official_hybrid_configuration",
        "device": str(vggts_config.device),
    }
    colmap_track_runtime = {
        "modelPath": str(colmap_track_config.model_path),
        "pixelTransformSummary": (
            str(colmap_track_config.pixel_transform_summary)
            if colmap_track_config.pixel_transform_summary is not None
            else None
        ),
        "minTrackLength": int(colmap_track_config.min_track_length),
        "maxReprojectionErrorPixels": float(colmap_track_config.max_reprojection_error),
        "rasterRadiusPixels": int(colmap_track_config.raster_radius),
        "visibilityPolicy": "recorded_colmap_observations_only",
        "denseRecovery": False,
        "segmentedPointCloud": {
            "cachePath": (
                str(colmap_track_config.segmented_cache_path)
                if colmap_track_config.segmented_cache_path is not None
                else None
            ),
            "plyPath": (
                str(colmap_track_config.segmented_ply_path)
                if colmap_track_config.segmented_ply_path is not None
                else None
            ),
        },
    }
    colmap_identity_runtime = {
        "sourceMethodId": str(colmap_identity_config.source_method_id),
        "sourceRunDirectory": str(colmap_identity_config.source_run_dir),
        "modelPath": str(colmap_identity_config.colmap.model_path),
        "visibilityPolicy": "recorded_colmap_observations_only",
        "maskGeometryPolicy": "preserve_source_component_pixels",
        "decisionPolicy": "confident_component_relabel_or_keep_source",
        "minComponentTrackCount": int(colmap_identity_config.min_component_track_count),
        "minTargetTrackCount": int(colmap_identity_config.min_target_track_count),
        "minTotalVoteWeight": float(colmap_identity_config.min_total_vote_weight),
        "minTargetFraction": float(colmap_identity_config.min_target_fraction),
        "minTargetMargin": float(colmap_identity_config.min_target_margin),
        "maxPixelsPerTargetTrack": int(colmap_identity_config.max_pixels_per_target_track),
        "paperBasis": [
            "MV3DIS 3D-guided multi-view mask matching",
            "MaskClustering view-consensus mask graph",
        ],
    }
    feedforward_point_runtime = {
        "sourceName": str(feedforward_point_config.source_name),
        "pointCloudPath": str(feedforward_point_config.points_path),
        "initMeshPath": (
            str(feedforward_point_config.init_mesh_path)
            if feedforward_point_config.init_mesh_path is not None
            else None
        ),
        "visibilityPolicy": "calibrated_projection_with_per_pixel_point_zbuffer",
        "visibilityDepthBandMeters": float(feedforward_point_config.visibility_depth_band_m),
        "visibilityDepthBandRelative": float(feedforward_point_config.visibility_depth_band_relative),
        "anchorFusion": "majority_vote_ties_unknown",
        "rasterRadiusPixels": int(feedforward_point_config.raster_radius),
        "denseRecovery": False,
        "segmentedPointCloud": {
            "cachePath": (
                str(feedforward_point_config.segmented_cache_path)
                if feedforward_point_config.segmented_cache_path is not None
                else None
            ),
            "plyPath": (
                str(feedforward_point_config.segmented_ply_path)
                if feedforward_point_config.segmented_ply_path is not None
                else None
            ),
        },
    }
    colmap_dense_runtime = {
        "sourceMethodId": str(colmap_dense_config.source_method_id),
        "sourceRunDirectory": str(colmap_dense_config.source_run_dir),
        "evidenceDirectory": str(colmap_dense_config.evidence_root),
        "targetSuperpixels": int(colmap_dense_config.target_superpixels),
        "compactness": float(colmap_dense_config.compactness),
        "maxSeedDistancePixels": float(colmap_dense_config.max_seed_distance_pixels),
        "maxBoundaryBarrier": float(colmap_dense_config.max_boundary_barrier),
        "minCompetitorMargin": float(colmap_dense_config.min_competitor_margin),
        "boundaryCues": ["rgb", "depth_anything_v2", "stable_normal"],
        "unknownPolicy": "abstain_on_distance_barrier_or_competitor_margin",
    }
    multifield_crf_runtime = {
        "sourceMethodId": str(multifield_crf_config.source_method_id),
        "sourceRunDirectory": str(multifield_crf_config.source_run_dir),
        "evidenceDirectory": str(multifield_crf_config.evidence_root),
        "pointSource": "colmap_recorded_observations",
        "segmentedPointCloud": str(multifield_crf_config.point_field.segmented_points_path),
        "targetSuperpixels": int(multifield_crf_config.target_superpixels),
        "compactness": float(multifield_crf_config.compactness),
        "meanFieldIterations": int(multifield_crf_config.mean_field_iterations),
        "pixelPairwiseWeight": float(multifield_crf_config.pixel_pairwise_weight),
        "pointPairwiseWeight": float(multifield_crf_config.point_pairwise_weight),
        "crossFieldWeight": float(multifield_crf_config.cross_field_weight),
        "damping": float(multifield_crf_config.damping),
        "maxSeedDistancePixels": float(multifield_crf_config.max_seed_distance_pixels),
        "minConfidence": float(multifield_crf_config.min_confidence),
        "minCompetitorMargin": float(multifield_crf_config.min_competitor_margin),
        "maxNormalizedEntropy": float(multifield_crf_config.max_normalized_entropy),
        "pixelNeighbors": int(multifield_crf_config.pixel_neighbors),
        "pointNeighbors": int(multifield_crf_config.point_neighbors),
        "unknownPolicy": "confidence_margin_entropy_and_support_abstention",
        "paper": "Xie et al., Semantic Instance Annotation of Street Scenes by 3D to 2D Label Transfer, CVPR 2016",
        "adaptation": (
            "Fine superpixels and local/bilateral k-NN graphs approximate the unreleased dense CRF; "
            "persistent keyframe labels replace manual street-scene primitives."
        ),
    }
    colmap_sam2_prompt_config = ColmapSam2PromptConfig(
        source_method_id=str(colmap_dense_config.source_method_id),
        source_run_dir=colmap_dense_config.source_run_dir,
        sam2=sam2_config,
    )
    colmap_sam2_prompt_runtime = {
        "sourceMethodId": str(colmap_sam2_prompt_config.source_method_id),
        "sourceRunDirectory": str(colmap_sam2_prompt_config.source_run_dir),
        "positivePromptCount": int(colmap_sam2_prompt_config.positive_prompt_count),
        "negativePromptCount": int(colmap_sam2_prompt_config.negative_prompt_count),
        "promptClearancePixels": float(colmap_sam2_prompt_config.prompt_clearance_pixels),
        "minPositivePrompts": int(colmap_sam2_prompt_config.min_positive_prompts),
        "minPositiveRecall": float(colmap_sam2_prompt_config.min_positive_recall),
        "minNegativeRejection": float(colmap_sam2_prompt_config.min_negative_rejection),
        "minSamScore": float(colmap_sam2_prompt_config.min_sam_score),
        "maxMaskFraction": float(colmap_sam2_prompt_config.max_mask_fraction),
        "candidatePolicy": "multimask_prompt_agreement",
        "overlapPolicy": "per_pixel_sam2_logit_argmax",
        "unknownPolicy": "pixels_outside_all_accepted_sam2_masks",
    }
    feedforward_sam2_prompt_config = ColmapSam2PromptConfig(
        source_method_id=str(feedforward_dense_config.source_method_id),
        source_run_dir=feedforward_dense_config.source_run_dir,
        sam2=sam2_config,
        source_name=str(feedforward_dense_config.source_name),
    )
    omega_final_sam2_prompt_config = ColmapSam2PromptConfig(
        source_method_id=str(omega_final_dense_config.source_method_id),
        source_run_dir=omega_final_dense_config.source_run_dir,
        sam2=sam2_config,
        source_name=str(omega_final_dense_config.source_name),
    )
    feedforward_dense_runtime = {
        **colmap_dense_runtime,
        "sourceMethodId": str(feedforward_dense_config.source_method_id),
        "sourceRunDirectory": str(feedforward_dense_config.source_run_dir),
        "sourceName": str(feedforward_dense_config.source_name),
        "targetSuperpixels": int(feedforward_dense_config.target_superpixels),
        "compactness": float(feedforward_dense_config.compactness),
        "maxSeedDistancePixels": float(feedforward_dense_config.max_seed_distance_pixels),
        "maxBoundaryBarrier": float(feedforward_dense_config.max_boundary_barrier),
        "minCompetitorMargin": float(feedforward_dense_config.min_competitor_margin),
    }
    feedforward_multifield_runtime = {
        **multifield_crf_runtime,
        "sourceMethodId": str(feedforward_multifield_crf_config.source_method_id),
        "sourceRunDirectory": str(feedforward_multifield_crf_config.source_run_dir),
        "pointSource": "feedforward_projected_zbuffer_points",
        "segmentedPointCloud": str(feedforward_multifield_crf_config.point_field.segmented_points_path),
    }
    feedforward_sam2_prompt_runtime = {
        **colmap_sam2_prompt_runtime,
        "sourceMethodId": str(feedforward_sam2_prompt_config.source_method_id),
        "sourceRunDirectory": str(feedforward_sam2_prompt_config.source_run_dir),
    }
    omega_final_point_runtime = {
        **feedforward_point_runtime,
        "sourceName": str(omega_final_point_config.source_name),
        "pointCloudPath": str(omega_final_point_config.points_path),
        "initMeshPath": None,
        "geometrySourcePath": (
            str(omega_final_point_config.mesh_point_config.mesh_path)
            if omega_final_point_config.mesh_point_config is not None
            else None
        ),
        "segmentedPointCloud": {
            "cachePath": (
                str(omega_final_point_config.segmented_cache_path)
                if omega_final_point_config.segmented_cache_path is not None
                else None
            ),
            "plyPath": (
                str(omega_final_point_config.segmented_ply_path)
                if omega_final_point_config.segmented_ply_path is not None
                else None
            ),
        },
    }
    omega_final_dense_runtime = {
        **colmap_dense_runtime,
        "sourceMethodId": str(omega_final_dense_config.source_method_id),
        "sourceRunDirectory": str(omega_final_dense_config.source_run_dir),
        "sourceName": str(omega_final_dense_config.source_name),
        "targetSuperpixels": int(omega_final_dense_config.target_superpixels),
        "compactness": float(omega_final_dense_config.compactness),
        "maxSeedDistancePixels": float(omega_final_dense_config.max_seed_distance_pixels),
        "maxBoundaryBarrier": float(omega_final_dense_config.max_boundary_barrier),
        "minCompetitorMargin": float(omega_final_dense_config.min_competitor_margin),
    }
    omega_final_multifield_runtime = {
        **multifield_crf_runtime,
        "sourceMethodId": str(omega_final_multifield_crf_config.source_method_id),
        "sourceRunDirectory": str(omega_final_multifield_crf_config.source_run_dir),
        "pointSource": "omega_final_projected_zbuffer_points",
        "segmentedPointCloud": str(omega_final_multifield_crf_config.point_field.segmented_points_path),
    }
    omega_final_sam2_prompt_runtime = {
        **colmap_sam2_prompt_runtime,
        "sourceMethodId": str(omega_final_sam2_prompt_config.source_method_id),
        "sourceRunDirectory": str(omega_final_sam2_prompt_config.source_run_dir),
    }

    split_splat_backends = _discover_split_splat_backends(work_root)

    return PropagationBackendRegistry(
        [
            *split_splat_backends,
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="sam2_video",
                    display_name="SAM2 Video",
                    engine_name="SAM2 Video",
                    description="Long ordered SAM2 propagation conditioned by every complete frame.",
                    result_description="Persistent region candidates from long SAM2 video propagation.",
                    runtime_config=sam2_runtime,
                ),
                session=Sam2VideoPropagationSession(sam2_config, work_root=work_root),
                availability_check=lambda: _check_requirements(sam2_requirements),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="sam2_bounded",
                    display_name="SAM2 Bounded",
                    engine_name="SAM2 Video",
                    description="Forward/backward SAM2 propagation inside intervals bounded by complete frames.",
                    result_description="Persistent region candidates from bounded anchor-to-anchor SAM2 propagation.",
                    runtime_config=sam2_bounded_runtime,
                ),
                session=Sam2BoundedPropagationSession(sam2_config, work_root=work_root),
                availability_check=lambda: _check_requirements(sam2_requirements),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="xmem",
                    display_name="XMem++",
                    engine_name="XMem++",
                    description="Memory-VOS propagation with every complete frame committed to permanent memory.",
                    result_description="Persistent region candidates from XMem++ permanent-memory propagation.",
                    runtime_config=xmem_runtime,
                ),
                session=MemoryVOSPropagationSession(xmem_config, work_root=work_root),
                availability_check=lambda: _check_requirements(xmem_requirements),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="cutie",
                    display_name="Cutie",
                    engine_name="Cutie",
                    description="Object-memory VOS propagation conditioned by every complete frame.",
                    result_description="Persistent region candidates from Cutie object-memory propagation.",
                    runtime_config=cutie_runtime,
                ),
                session=MemoryVOSPropagationSession(cutie_config, work_root=work_root),
                availability_check=lambda: _check_requirements(cutie_requirements),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="colmap_tracks",
                    display_name="COLMAP Tracks",
                    engine_name="Sparse COLMAP Track Consensus",
                    description=(
                        "Transfers persistent IDs through exact SfM feature observations and renders "
                        "only sparse point pixels; no dense mask recovery."
                    ),
                    result_description=(
                        "Sparse persistent-region observations transferred through COLMAP feature tracks."
                    ),
                    runtime_config=colmap_track_runtime,
                    layer_group="sparse_3d_transfer",
                ),
                session=ColmapTrackPropagationSession(colmap_track_config),
                availability_check=lambda: colmap_track_availability(colmap_track_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="colmap_verify_video",
                    display_name="COLMAP-Verified Video",
                    engine_name="COLMAP Track Identity Consensus",
                    description=(
                        "Preserves SAM2 Video mask shapes and changes only component region IDs when "
                        "exact COLMAP tracks give decisive multi-anchor evidence."
                    ),
                    result_description=(
                        "SAM2 Video candidates with conservative COLMAP track identity correction."
                    ),
                    runtime_config=colmap_identity_runtime,
                    stage="identity_refinement",
                    source_method_id=colmap_identity_config.source_method_id,
                    layer_group="identity_refinement",
                ),
                session=ColmapIdentityRefinementSession(colmap_identity_config),
                availability_check=lambda: colmap_identity_refinement_availability(
                    colmap_identity_config
                ),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="feedforward_points",
                    display_name="OMeGa Initializer Points",
                    engine_name="Projected Feed-Forward Point Consensus",
                    description=(
                        "Projects the aligned OMeGa initializer cloud into every complete frame, "
                        "labels visible points by persistent regions, and leaves vote ties unknown."
                    ),
                    result_description=(
                        "Sparse persistent-region observations transferred through the OMeGa initializer cloud."
                    ),
                    runtime_config=feedforward_point_runtime,
                    layer_group="sparse_3d_transfer",
                ),
                session=FeedForwardPointPropagationSession(feedforward_point_config),
                availability_check=lambda: feedforward_point_availability(feedforward_point_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="omega_final_points",
                    display_name="OMeGa Clean Hybrid",
                    engine_name="Projected Clean Hybrid Point Consensus",
                    description=(
                        "Labels cleaned detail-adaptive vertices and large-face completion samples "
                        "from the strongest OMeGa mesh; conflicting anchor votes remain unknown."
                    ),
                    result_description=(
                        "Persistent-region observations transferred through the clean hybrid OMeGa cloud."
                    ),
                    runtime_config=omega_final_point_runtime,
                    layer_group="sparse_3d_transfer",
                ),
                session=FeedForwardPointPropagationSession(omega_final_point_config),
                availability_check=lambda: feedforward_point_availability(omega_final_point_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="colmap_dense_superpixels",
                    display_name="COLMAP Superpixels",
                    engine_name="Sparse COLMAP + Boundary-Aware Superpixels",
                    description=(
                        "Densifies exact COLMAP region observations over fine superpixels using "
                        "RGB, Depth Anything, and StableNormal boundaries, while retaining unknown pixels."
                    ),
                    result_description=(
                        "Boundary-aware dense persistent-region candidates grown from sparse COLMAP tracks."
                    ),
                    runtime_config=colmap_dense_runtime,
                    stage="dense_recovery",
                    source_method_id="colmap_tracks",
                    layer_group="dense_colmap",
                ),
                session=ColmapDensePropagationSession(colmap_dense_config),
                availability_check=lambda: colmap_dense_availability(colmap_dense_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="colmap_dense_multifield_crf",
                    display_name="COLMAP 2D/3D CRF",
                    engine_name="Xie16 Multi-Field CRF Adaptation",
                    description=(
                        "Jointly regularizes fine image superpixels and visible labeled COLMAP points "
                        "with 2D, 3D, and projected cross-field Potts terms."
                    ),
                    result_description=(
                        "Dense persistent-region candidates inferred jointly over image and COLMAP point fields."
                    ),
                    runtime_config=multifield_crf_runtime,
                    stage="dense_recovery",
                    source_method_id="colmap_tracks",
                    layer_group="dense_colmap",
                ),
                session=MultiFieldCrfRecoverySession(multifield_crf_config),
                availability_check=lambda: multifield_crf_availability(multifield_crf_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="colmap_dense_sam2_prompts",
                    display_name="COLMAP SAM2 Prompts",
                    engine_name="Sparse COLMAP + SAM2 Image Predictor",
                    description=(
                        "Cleans and samples persistent-region COLMAP observations as positive SAM2 "
                        "prompts, uses competing regions as negatives, and resolves mask overlap from logits."
                    ),
                    result_description=(
                        "SAMPro3D-inspired dense persistent-region candidates prompted by sparse COLMAP tracks."
                    ),
                    runtime_config=colmap_sam2_prompt_runtime,
                    stage="dense_recovery",
                    source_method_id="colmap_tracks",
                    layer_group="dense_colmap",
                ),
                session=ColmapSam2PromptRecoverySession(
                    colmap_sam2_prompt_config,
                    predictor=sam2_image_session,
                ),
                availability_check=lambda: colmap_sam2_prompt_availability(colmap_sam2_prompt_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="feedforward_dense_superpixels",
                    display_name="Init Superpixels",
                    engine_name="OMeGa Initializer Points + Boundary-Aware Superpixels",
                    description=(
                        "Densifies projected initializer-point labels over fine RGB, depth, and normal superpixels."
                    ),
                    result_description=(
                        "Boundary-aware dense candidates grown from the feed-forward initializer point layer."
                    ),
                    runtime_config=feedforward_dense_runtime,
                    stage="dense_recovery",
                    source_method_id="feedforward_points",
                    layer_group="dense_omega_init",
                ),
                session=ColmapDensePropagationSession(feedforward_dense_config),
                availability_check=lambda: colmap_dense_availability(feedforward_dense_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="feedforward_dense_multifield_crf",
                    display_name="Init 2D/3D CRF",
                    engine_name="Xie16 Multi-Field CRF Adaptation",
                    description=(
                        "Jointly regularizes image superpixels and visible labeled initializer points "
                        "with 2D, 3D, and projected cross-field Potts terms."
                    ),
                    result_description=(
                        "Dense candidates inferred jointly over image and initializer point fields."
                    ),
                    runtime_config=feedforward_multifield_runtime,
                    stage="dense_recovery",
                    source_method_id="feedforward_points",
                    layer_group="dense_omega_init",
                ),
                session=MultiFieldCrfRecoverySession(feedforward_multifield_crf_config),
                availability_check=lambda: multifield_crf_availability(feedforward_multifield_crf_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="feedforward_dense_sam2_prompts",
                    display_name="Init SAM2 Prompts",
                    engine_name="OMeGa Initializer Points + SAM2 Image Predictor",
                    description=(
                        "Uses projected initializer points as positive and competing negative SAM2 prompts, "
                        "then resolves accepted-mask overlap from logits."
                    ),
                    result_description=(
                        "SAMPro3D-inspired dense candidates prompted by feed-forward initializer points."
                    ),
                    runtime_config=feedforward_sam2_prompt_runtime,
                    stage="dense_recovery",
                    source_method_id="feedforward_points",
                    layer_group="dense_omega_init",
                ),
                session=ColmapSam2PromptRecoverySession(
                    feedforward_sam2_prompt_config,
                    predictor=sam2_image_session,
                ),
                availability_check=lambda: colmap_sam2_prompt_availability(feedforward_sam2_prompt_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="omega_final_dense_superpixels",
                    display_name="Final Superpixels",
                    engine_name="OMeGa Final Points + Boundary-Aware Superpixels",
                    description=(
                        "Densifies projected final-mesh vertex labels over fine RGB, depth, and normal superpixels."
                    ),
                    result_description=(
                        "Boundary-aware dense candidates grown from final optimized mesh vertices."
                    ),
                    runtime_config=omega_final_dense_runtime,
                    stage="dense_recovery",
                    source_method_id="omega_final_points",
                    layer_group="dense_omega_final",
                ),
                session=ColmapDensePropagationSession(omega_final_dense_config),
                availability_check=lambda: colmap_dense_availability(omega_final_dense_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="omega_final_dense_multifield_crf",
                    display_name="Final 2D/3D CRF",
                    engine_name="Xie16 Multi-Field CRF Adaptation",
                    description=(
                        "Jointly regularizes image superpixels and visible labeled final-mesh vertices "
                        "with 2D, 3D, and projected cross-field Potts terms."
                    ),
                    result_description=(
                        "Dense candidates inferred jointly over image and final optimized mesh-vertex fields."
                    ),
                    runtime_config=omega_final_multifield_runtime,
                    stage="dense_recovery",
                    source_method_id="omega_final_points",
                    layer_group="dense_omega_final",
                ),
                session=MultiFieldCrfRecoverySession(omega_final_multifield_crf_config),
                availability_check=lambda: multifield_crf_availability(omega_final_multifield_crf_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="omega_final_dense_sam2_prompts",
                    display_name="Final SAM2 Prompts",
                    engine_name="OMeGa Final Points + SAM2 Image Predictor",
                    description=(
                        "Uses projected final-mesh vertices as positive and competing negative SAM2 prompts, "
                        "then resolves accepted-mask overlap from logits."
                    ),
                    result_description=(
                        "SAMPro3D-inspired dense candidates prompted by final optimized mesh vertices."
                    ),
                    runtime_config=omega_final_sam2_prompt_runtime,
                    stage="dense_recovery",
                    source_method_id="omega_final_points",
                    layer_group="dense_omega_final",
                ),
                session=ColmapSam2PromptRecoverySession(
                    omega_final_sam2_prompt_config,
                    predictor=sam2_image_session,
                ),
                availability_check=lambda: colmap_sam2_prompt_availability(omega_final_sam2_prompt_config),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="v2sam_pair",
                    display_name="V2-SAM Pair",
                    engine_name="V2-SAM Pairwise Multi-Expert",
                    description=(
                        "Focused transfer from a region's first manual occurrence using Anchor, Visual, "
                        "Fusion, and PCCS, with no cross-source merge."
                    ),
                    result_description=(
                        "A direct PCCS-selected V2-SAM candidate from the region's first manual occurrence."
                    ),
                    runtime_config=v2sam_runtime,
                    supports_full_run=False,
                    supports_region_pair=True,
                    pair_output_policy="direct_pccs_selected_expert",
                    layer_group="pairwise_transfer",
                ),
                session=V2SamPropagationSession(
                    v2sam_config,
                    work_root=work_root,
                    omega_root=omega_root,
                ),
                availability_check=lambda: _check_requirements(v2sam_requirements),
            ),
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id="vggts_pair",
                    display_name="VGGT-S Pair",
                    engine_name="VGGT-S Union Segmentation Head",
                    description=(
                        "Focused transfer from a region's first manual occurrence using VGGT "
                        "correspondences, pair features, full source-mask conditioning, and refinement."
                    ),
                    result_description=(
                        "A refined VGGT-S target candidate from the region's first manual occurrence."
                    ),
                    runtime_config=vggts_runtime,
                    supports_full_run=False,
                    supports_region_pair=True,
                    pair_output_policy="union_head_refined_mask",
                    layer_group="pairwise_transfer",
                ),
                session=VggtSPropagationSession(
                    vggts_config,
                    work_root=work_root,
                    omega_root=omega_root,
                ),
                availability_check=lambda: _check_requirements(vggts_requirements),
            ),
        ]
    )


def _discover_split_splat_backends(work_root: Path) -> list[PropagationBackend]:
    propagation_root = work_root.parent / "proposals" / "propagation"
    backends: list[PropagationBackend] = []
    for summary_path in sorted(propagation_root.glob("*/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        label_space = str(payload.get("labelSpace") or "")
        if label_space not in {
            "split_splat_instance",
            "frame_local_proposal",
            "persistent_region",
        }:
            continue
        # Ordinary editor propagation also uses persistent-region IDs. Only
        # canonical staged Split experiments belong in this discovery path.
        if label_space == "persistent_region" and not payload.get(
            "canonicalRunDir"
        ):
            continue
        is_proposal_preview = label_space == "frame_local_proposal"
        method_id = normalize_method_id(
            str(payload.get("methodId") or summary_path.parent.name)
        )
        canonical_dir = Path(str(payload.get("canonicalRunDir") or ""))
        default_description = (
            "Read-only frame-local proposals supplied to a staged Split&Splat run."
            if is_proposal_preview
            else (
                "Read-only globally consistent masks produced by a staged "
                "Split&Splat Split experiment."
            )
        )
        backends.append(
            PropagationBackend(
                info=PropagationBackendInfo(
                    method_id=method_id,
                    display_name=str(
                        payload.get("displayName")
                        or f"Split&Splat {method_id}"
                    ),
                    engine_name=str(payload.get("engineName") or "Split&Splat"),
                    description=str(
                        payload.get("description") or default_description
                    ),
                    result_description=(
                        "Split&Splat input proposal preview."
                        if is_proposal_preview
                        else (
                            "Anchored persistent-region candidates."
                            if label_space == "persistent_region"
                            else "Split&Splat run-local instance candidates."
                        )
                    ),
                    runtime_config={
                        "canonicalRunDirectory": str(canonical_dir),
                        "parameterSource": "staged_split_splat_experiment",
                    },
                    supports_full_run=False,
                    supports_region_pair=False,
                    stage="anchor_to_mask",
                    layer_group=str(
                        payload.get("layerGroup")
                        or (
                            "frame_proposals"
                            if is_proposal_preview
                            else "refined_masks"
                        )
                    ),
                    label_space=label_space,
                    read_only=True,
                ),
                session=object(),
                availability_check=lambda path=summary_path: (
                    (True, "Ready")
                    if path.is_file()
                    else (False, "Split&Splat layer summary is missing.")
                ),
            )
        )
    return backends


def discover_split_splat_layer_status(work_root: Path) -> list[dict[str, Any]]:
    """Discover read-only Split&Splat layers created after editor startup."""
    return [
        backend.status()
        for backend in _discover_split_splat_backends(work_root)
    ]


def normalize_method_id(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if not key or any(not (char.isalnum() or char == "_") for char in key):
        raise ValueError(f"Invalid propagation method ID: {value!r}")
    return key


def propagation_layer_key(method_id: str) -> str:
    return f"propagation_{normalize_method_id(method_id)}"


def is_propagation_layer(layer_key: str) -> bool:
    return str(layer_key).startswith("propagation_") and len(str(layer_key)) > len("propagation_")


def propagation_method_id(layer_key: str) -> str:
    key = str(layer_key).strip().lower().replace("-", "_")
    if not is_propagation_layer(key):
        raise ValueError(f"Layer '{layer_key}' is not an anchor-to-mask propagation layer.")
    return normalize_method_id(key[len("propagation_") :])


def _check_requirements(requirements: tuple[tuple[Path, str, str], ...]) -> tuple[bool, str]:
    for raw_path, label, kind in requirements:
        path = raw_path.expanduser().resolve()
        valid = path.is_dir() if kind == "dir" else path.is_file()
        if not valid:
            return False, f"{label} is missing: {path}"
    return True, "Ready"
