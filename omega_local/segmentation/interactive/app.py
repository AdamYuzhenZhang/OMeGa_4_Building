"""Web entry point for the OMeGa interactive segmentation editor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .colmap_dense_propagation import ColmapDenseConfig
from .colmap_identity_refinement import ColmapIdentityRefinementConfig
from .colmap_point_field import ColmapPointFieldConfig
from .colmap_track_propagation import ColmapTrackConfig
from .dinov3_evidence import DinoV3EvidenceConfig, DinoV3EvidenceManager
from .feedforward_point_field import FeedForwardPointFieldConfig
from .feedforward_point_propagation import FeedForwardPointConfig
from .gaussian_viewer import viewer_settings, viewer_status
from .memory_vos_propagation import MemoryVOSConfig
from .multifield_crf_recovery import MultiFieldCrfConfig
from .omega_mesh_point_cloud import OmegaMeshHybridPointCloudConfig
from .paths import OMEGA_ROOT, PROJECT_ROOT, STATIC_DIR, THIRD_PARTY_ROOT, require_dir, resolve_paths
from .propagation_backends import build_default_propagation_registry
from .sam2_session import Sam2Config, Sam2Session
from .sai3d_experiment import SAI3DExperimentBackend, SAI3DExperimentConfig
from .segmentation3d_contract import Segmentation3DRegistry
from .segmentation3d_manager import Segmentation3DManager
from .state import EditorState
from .v2sam_propagation import V2SamConfig
from .vggts_propagation import VggtSConfig


def _json_response(payload: dict[str, Any] | list[Any]) -> Response:
    return Response(
        content=json.dumps(payload, separators=(",", ":")),
        media_type="application/json",
    )


def create_app(
    *,
    model_dir: Path,
    baseline_name: str,
    max_points: int,
    seed: int,
    label_source: str,
    sam2_root: Path,
    sam2_checkpoint: Path | None,
    sam2_config: str,
    sam2_device: str,
    xmem_root: Path,
    xmem_checkpoint: Path | None,
    xmem_size: int,
    cutie_root: Path,
    cutie_checkpoint: Path | None,
    cutie_size: int,
    memory_vos_device: str,
    v2sam_root: Path,
    v2sam_python: Path,
    v2sam_profile: str,
    v2sam_visual_checkpoint: Path | None,
    v2sam_fusion_checkpoint: Path | None,
    v2sam_expert_batch_size: int,
    v2sam_device: str,
    vggts_root: Path,
    vggts_python: Path,
    vggts_checkpoint: Path | None,
    vggts_device: str,
    feedforward_point_cloud: Path | None,
    feedforward_init_mesh: Path | None,
    omega_final_mesh: Path | None,
    sai3d_root: Path,
) -> FastAPI:
    paths = resolve_paths(
        model_dir,
        baseline_name,
        feedforward_point_cloud=feedforward_point_cloud,
        feedforward_init_mesh=feedforward_init_mesh,
        omega_final_mesh=omega_final_mesh,
    )
    sam2_root = require_dir(sam2_root, "SAM2 root")
    checkpoint = (
        sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
        if sam2_checkpoint is None
        else sam2_checkpoint.expanduser().resolve()
    )
    sam2_runtime_config = Sam2Config(
        root=sam2_root,
        checkpoint=checkpoint,
        config=str(sam2_config),
        device=str(sam2_device),
    )
    xmem_runtime_config = MemoryVOSConfig(
        backend="xmem",
        root=xmem_root.expanduser().resolve(),
        checkpoint=(
            xmem_root.expanduser().resolve() / "saves" / "XMem.pth"
            if xmem_checkpoint is None
            else xmem_checkpoint.expanduser().resolve()
        ),
        internal_size=int(xmem_size),
        device=str(memory_vos_device),
    )
    cutie_runtime_config = MemoryVOSConfig(
        backend="cutie",
        root=cutie_root.expanduser().resolve(),
        checkpoint=(
            cutie_root.expanduser().resolve() / "weights" / "cutie-base-mega.pth"
            if cutie_checkpoint is None
            else cutie_checkpoint.expanduser().resolve()
        ),
        internal_size=int(cutie_size),
        device=str(memory_vos_device),
    )
    v2sam_root = v2sam_root.expanduser().resolve()
    profile = str(v2sam_profile).strip().lower()
    v2sam_runtime_config = V2SamConfig(
        root=v2sam_root,
        # Keep the venv launcher path. Resolving the symlink can bypass that
        # environment's pyvenv.cfg and invoke the system interpreter instead.
        python=v2sam_python.expanduser().absolute(),
        sam_checkpoint=v2sam_root / "weights" / "sam2" / "sam2_hiera_large.pt",
        dino_checkpoint=v2sam_root / "weights" / "dinov3" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        visual_checkpoint=(
            v2sam_root / "weights" / f"visual_{profile}_full.pth"
            if v2sam_visual_checkpoint is None
            else v2sam_visual_checkpoint.expanduser().resolve()
        ),
        fusion_checkpoint=(
            v2sam_root / "weights" / f"fusion_{profile}_full.pth"
            if v2sam_fusion_checkpoint is None
            else v2sam_fusion_checkpoint.expanduser().resolve()
        ),
        feature_cache_dir=paths.interactive_dir / "view_evidence" / "dinov3_vitl16_768",
        diagnostics_dir=(
            paths.interactive_dir
            / "proposals"
            / "propagation"
            / "v2sam_pair"
            / "v2sam_diagnostics"
        ),
        expert_profile=profile,
        device=str(v2sam_device),
        expert_batch_size=int(v2sam_expert_batch_size),
        seed=int(seed),
    )
    vggts_root = vggts_root.expanduser().resolve()
    vggts_runtime_config = VggtSConfig(
        root=vggts_root,
        # Preserve a venv's Python launcher path; resolving its symlink would
        # bypass pyvenv.cfg and execute the system interpreter instead.
        python=vggts_python.expanduser().absolute(),
        checkpoint=(
            vggts_root / "official_ckpts" / "main_exp.pth"
            if vggts_checkpoint is None
            else vggts_checkpoint.expanduser().resolve()
        ),
        diagnostics_dir=(
            paths.interactive_dir
            / "proposals"
            / "propagation"
            / "vggts_pair"
            / "vggts_diagnostics"
        ),
        device=str(vggts_device),
        seed=int(seed),
    )
    colmap_track_config = ColmapTrackConfig(
        model_path=paths.colmap_track_model,
        pixel_transform_summary=(
            paths.colmap_pixel_transform_summary
            if paths.colmap_pixel_transform_summary.is_file()
            else None
        ),
        aligned_points_path=paths.colmap_sparse_source,
        segmented_cache_path=paths.colmap_sparse_segmented_cache,
        segmented_ply_path=paths.colmap_sparse_segmented_ply,
        segmented_summary_path=paths.colmap_sparse_segmented_summary,
    )
    feedforward_point_config = FeedForwardPointConfig(
        points_path=paths.feedforward_source,
        source_name="OMeGa Initializer Points",
        init_mesh_path=paths.feedforward_init_mesh,
        segmented_cache_path=paths.feedforward_segmented_cache,
        segmented_ply_path=paths.feedforward_segmented_ply,
        segmented_summary_path=paths.feedforward_segmented_summary,
    )
    feedforward_run_dir = (
        paths.interactive_dir / "proposals" / "propagation" / "feedforward_points"
    )
    omega_final_point_config = FeedForwardPointConfig(
        points_path=paths.omega_final_hybrid_points,
        source_name="OMeGa Clean Hybrid Points",
        mesh_point_config=OmegaMeshHybridPointCloudConfig(
            mesh_path=paths.omega_final_mesh_source,
            points_path=paths.omega_final_hybrid_points,
            cache_path=paths.omega_final_hybrid_cache,
            summary_path=paths.omega_final_hybrid_summary,
            surface_cache_path=paths.omega_final_surface_cache,
            surface_summary_path=paths.omega_final_surface_summary,
        ),
        segmented_cache_path=paths.omega_final_segmented_cache,
        segmented_ply_path=paths.omega_final_segmented_ply,
        segmented_summary_path=paths.omega_final_segmented_summary,
    )
    omega_final_run_dir = (
        paths.interactive_dir / "proposals" / "propagation" / "omega_final_points"
    )
    sam2_image_session = Sam2Session(sam2_runtime_config)
    propagation_backends = build_default_propagation_registry(
        colmap_track_config=colmap_track_config,
        colmap_identity_config=ColmapIdentityRefinementConfig(
            colmap=colmap_track_config,
            source_method_id="sam2_video",
            source_run_dir=(
                paths.interactive_dir
                / "proposals"
                / "propagation"
                / "sam2_video"
            ),
        ),
        colmap_dense_config=ColmapDenseConfig(
            source_method_id="colmap_tracks",
            source_run_dir=(
                paths.interactive_dir
                / "proposals"
                / "propagation"
                / "colmap_tracks"
            ),
            evidence_root=paths.interactive_dir / "view_evidence",
        ),
        multifield_crf_config=MultiFieldCrfConfig(
            source_method_id="colmap_tracks",
            source_run_dir=(
                paths.interactive_dir
                / "proposals"
                / "propagation"
                / "colmap_tracks"
            ),
            evidence_root=paths.interactive_dir / "view_evidence",
            point_field=ColmapPointFieldConfig(
                model_path=paths.colmap_track_model,
                segmented_points_path=paths.colmap_sparse_segmented_cache,
                pixel_transform_summary=(
                    paths.colmap_pixel_transform_summary
                    if paths.colmap_pixel_transform_summary.is_file()
                    else None
                ),
            ),
        ),
        feedforward_point_config=feedforward_point_config,
        feedforward_dense_config=ColmapDenseConfig(
            source_method_id="feedforward_points",
            source_run_dir=feedforward_run_dir,
            evidence_root=paths.interactive_dir / "view_evidence",
            source_name="OMeGa Initializer Points",
        ),
        feedforward_multifield_crf_config=MultiFieldCrfConfig(
            source_method_id="feedforward_points",
            source_run_dir=feedforward_run_dir,
            evidence_root=paths.interactive_dir / "view_evidence",
            source_name="OMeGa Initializer Points",
            point_visibility_policy="calibrated_projection_with_per_pixel_point_zbuffer",
            point_field=FeedForwardPointFieldConfig(
                points_path=paths.feedforward_source,
                segmented_points_path=paths.feedforward_segmented_cache,
                visibility_depth_band_m=float(feedforward_point_config.visibility_depth_band_m),
                visibility_depth_band_relative=float(feedforward_point_config.visibility_depth_band_relative),
            ),
        ),
        omega_final_point_config=omega_final_point_config,
        omega_final_dense_config=ColmapDenseConfig(
            source_method_id="omega_final_points",
            source_run_dir=omega_final_run_dir,
            evidence_root=paths.interactive_dir / "view_evidence",
            source_name="OMeGa Clean Hybrid Points",
        ),
        omega_final_multifield_crf_config=MultiFieldCrfConfig(
            source_method_id="omega_final_points",
            source_run_dir=omega_final_run_dir,
            evidence_root=paths.interactive_dir / "view_evidence",
            source_name="OMeGa Clean Hybrid Points",
            point_visibility_policy="calibrated_projection_with_per_pixel_point_zbuffer",
            point_field=FeedForwardPointFieldConfig(
                points_path=paths.omega_final_hybrid_points,
                segmented_points_path=paths.omega_final_segmented_cache,
                source_name="OMeGa Clean Hybrid Points",
                visibility_depth_band_m=float(omega_final_point_config.visibility_depth_band_m),
                visibility_depth_band_relative=float(omega_final_point_config.visibility_depth_band_relative),
            ),
        ),
        sam2_config=sam2_runtime_config,
        sam2_image_session=sam2_image_session,
        xmem_config=xmem_runtime_config,
        cutie_config=cutie_runtime_config,
        v2sam_config=v2sam_runtime_config,
        vggts_config=vggts_runtime_config,
        work_root=paths.interactive_dir / "tmp",
        omega_root=OMEGA_ROOT,
    )
    dinov3_evidence = DinoV3EvidenceManager(
        DinoV3EvidenceConfig(
            root=v2sam_runtime_config.feature_cache_dir,
            repo_root=v2sam_runtime_config.root,
            python=v2sam_runtime_config.python,
            checkpoint=v2sam_runtime_config.dino_checkpoint,
            omega_root=OMEGA_ROOT,
            device=v2sam_runtime_config.device,
        )
    )
    segmentation3d = Segmentation3DManager(
        paths.segmentation3d_dir,
        Segmentation3DRegistry(
            [
                SAI3DExperimentBackend(
                    SAI3DExperimentConfig(
                        paths=paths,
                        sai3d_root=sai3d_root.expanduser().resolve(),
                        python=Path(sys.executable),
                        seed=int(seed),
                    )
                )
            ]
        ),
        max_points=int(max_points),
        seed=int(seed),
    )
    state = EditorState(
        paths,
        max_points=max_points,
        seed=seed,
        label_source=label_source,
        sam2_config=sam2_runtime_config,
        sam2_session=sam2_image_session,
        propagation_backends=propagation_backends,
        dinov3_evidence=dinov3_evidence,
        segmentation3d=segmentation3d,
    )

    app = FastAPI(title="OMeGa Interactive Segmentation Editor")
    app.state.editor_state = state
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    supersplat_public = THIRD_PARTY_ROOT / "supersplat-viewer" / "public"
    if (supersplat_public / "index.html").is_file():
        app.mount(
            "/vendor/supersplat",
            StaticFiles(directory=supersplat_public),
            name="supersplat",
        )

    @app.middleware("http")
    async def no_cache_editor_assets(request, call_next):
        response = await call_next(request)
        if (
            request.url.path == "/"
            or request.url.path.startswith("/static/")
            or request.url.path.startswith("/api/3d-segmentation/viewer/")
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/project")
    def project() -> Response:
        return _json_response(state.project_summary())

    @app.get("/api/points")
    def points() -> Response:
        try:
            return _json_response(state.points_payload())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/point-clouds/status")
    def point_cloud_sources_status() -> Response:
        return _json_response(state.point_cloud_sources_status())

    @app.get("/api/point-clouds/colmap")
    def colmap_points(mode: str = "raw") -> Response:
        try:
            return _json_response(state.colmap_points_payload(mode=mode))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/point-clouds/feedforward")
    def feedforward_points(mode: str = "raw") -> Response:
        try:
            return _json_response(state.feedforward_points_payload(mode=mode))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/point-clouds/omegaFinal")
    def omega_final_points(mode: str = "raw") -> Response:
        try:
            return _json_response(state.omega_final_points_payload(mode=mode))
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/labels/summary")
    def label_summary() -> Response:
        return _json_response(state.label_summary())

    @app.get("/api/frames")
    def frames() -> Response:
        return _json_response(state.frames())

    @app.get("/api/proposals/sam2/status")
    def proposal_status() -> Response:
        return _json_response(state.proposal_status())

    @app.get("/api/proposals/layers/status")
    def proposal_layers_status() -> Response:
        return _json_response(state.proposal_layers_status())

    @app.get("/api/regions/status")
    def region_status() -> Response:
        return _json_response(state.region_status())

    @app.get("/api/3d-segmentation/status")
    def segmentation3d_status() -> Response:
        return _json_response(state.segmentation3d_status())

    @app.post("/api/3d-segmentation/start")
    def segmentation3d_start(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.start_segmentation3d_job(payload))
        except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/3d-segmentation/jobs/{job_id}")
    def segmentation3d_job(job_id: str) -> Response:
        try:
            return _json_response(state.segmentation3d_job_status(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"3D segmentation job not found: {job_id}") from exc

    @app.get("/api/3d-segmentation/runs/{run_id}/points")
    def segmentation3d_result_points(run_id: str) -> Response:
        try:
            return _json_response(state.segmentation3d_result_points(run_id))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/3d-segmentation/viewer/status")
    def segmentation3d_viewer_status() -> Response:
        return _json_response(viewer_status(supersplat_public))

    @app.get("/api/3d-segmentation/viewer/settings.json")
    def segmentation3d_viewer_settings() -> Response:
        return _json_response(viewer_settings())

    @app.get(
        "/api/3d-segmentation/runs/{run_id}/gaussians/{variant_id}.ply"
    )
    def segmentation3d_gaussian_artifact(
        run_id: str,
        variant_id: str,
    ) -> FileResponse:
        try:
            path = state.segmentation3d.gaussian_artifact(run_id, variant_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="application/octet-stream",
            filename=f"{run_id}_{variant_id}.ply",
            headers={
                "Cache-Control": "public, max-age=31536000, immutable",
            },
        )

    @app.get(
        "/api/3d-segmentation/runs/{run_id}/gaussian-scenes/{variant_id}.json"
    )
    def segmentation3d_gaussian_scene(
        run_id: str,
        variant_id: str,
    ) -> Response:
        try:
            return _json_response(
                state.segmentation3d.gaussian_scene(run_id, variant_id)
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/view-evidence/status")
    def view_evidence_status() -> Response:
        return _json_response(state.view_evidence_status())

    @app.post("/api/view-evidence/run")
    def view_evidence_run(payload: dict[str, Any] | None = Body(default=None)) -> Response:
        try:
            return _json_response(state.start_view_evidence_run(payload or {}))
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/keyframes/status")
    def keyframe_status() -> Response:
        return _json_response(state.keyframe_status())

    @app.post("/api/keyframes/detect")
    def keyframe_detect(payload: dict[str, Any] | None = Body(default=None)) -> Response:
        try:
            return _json_response(state.detect_keyframes(payload or {}))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/run")
    def proposal_run(payload: dict[str, Any] | None = Body(default=None)) -> Response:
        try:
            return _json_response(state.start_proposal_run(payload or {}))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/sam2/predict")
    def sam2_predict(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.predict_sam2(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/proposals/sam2/frame/{frame_id}/overlay")
    def proposal_overlay(frame_id: int) -> FileResponse:
        try:
            path = state.proposal_overlay_path(frame_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/proposals/layers/{layer}/frame/{frame_id}/overlay")
    def proposal_layer_overlay(layer: str, frame_id: int) -> FileResponse:
        try:
            path = state.proposal_layer_overlay_path(frame_id, layer)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/regions/frame/{frame_id}/summary")
    def region_frame_summary(frame_id: int) -> Response:
        try:
            return _json_response(state.region_frame_summary(frame_id))
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/regions/frame/{frame_id}/complete")
    def region_frame_complete(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.set_region_frame_complete(frame_id, payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/regions/frame/{frame_id}/overlay")
    def region_overlay(frame_id: int) -> FileResponse:
        try:
            path = state.region_overlay_path(frame_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/regions/frame/{frame_id}/overlay/{region_id}")
    def single_region_overlay(frame_id: int, region_id: int) -> Response:
        try:
            return Response(content=state.single_region_overlay_png(frame_id, region_id), media_type="image/png")
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/view-evidence/frame/{frame_id}/{kind}")
    def view_evidence_image(frame_id: int, kind: str) -> FileResponse:
        try:
            path = state.view_evidence_image_path(frame_id, kind)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.post("/api/view-evidence/frame/{frame_id}/grow-selection")
    def view_evidence_grow_selection(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.grow_evidence_selection(frame_id, payload))
        except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/view-evidence/frame/{frame_id}/rgbd-cue-debug")
    def view_evidence_rgbd_cue_debug(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.rgbd_cue_debug(frame_id, payload))
        except (KeyError, FileNotFoundError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/proposals/sam2/frame/{frame_id}/summary")
    def proposal_frame_summary(frame_id: int) -> Response:
        try:
            return _json_response(state.proposal_combined_frame_summary(frame_id))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/proposals/layers/{layer}/frame/{frame_id}/summary")
    def proposal_layer_frame_summary(layer: str, frame_id: int) -> Response:
        try:
            return _json_response(state.proposal_layer_frame_summary(frame_id, layer))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/frame/{frame_id}/pick")
    def proposal_pick(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.pick_proposal(frame_id, payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/create-from-selection")
    def region_create_from_selection(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.create_region_from_selection(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/assign-selection")
    def region_assign_selection(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.assign_selection_to_region(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/add-selection")
    def region_add_selection(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.add_selection_to_region(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/clear-selection")
    def region_clear_selection(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.clear_selection_from_regions(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/clear-frame")
    def region_clear_frame(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.clear_region_from_frame(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/rename")
    def region_rename(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.rename_region(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/regions/delete")
    def region_delete(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.delete_region(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/frame/{frame_id}/selection-preview")
    def proposal_selection_preview(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.preview_selection(frame_id, payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/propagation/{method_id}/start")
    def propagation_start(method_id: str, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.start_region_propagation_job({**payload, "methodId": method_id}))
        except (KeyError, FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/propagation/{method_id}/region-test/start")
    def region_pair_test_start(method_id: str, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.start_region_pair_test_job({**payload, "methodId": method_id}))
        except (KeyError, FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/propagation/jobs/{job_id}")
    def propagation_job(job_id: str) -> Response:
        try:
            return _json_response(state.propagation_job_status(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Propagation job not found: {job_id}") from exc

    @app.get("/api/frame/{frame_id}/image")
    def frame_image(frame_id: int) -> FileResponse:
        try:
            path = state.image_path(frame_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path)

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the OMeGa interactive segmentation editor.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="sai3d_area_samples_1024_dense")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--max-points", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument(
        "--label-source",
        choices=("raw", "sai3d", "auto"),
        default="raw",
        help=(
            "Initial labels to load. raw renders the sampled SAI3D input points as "
            "unsegmented label 0; sai3d loads mesh_labels/point_labels.npy; auto "
            "tries SAI3D labels and then raw points."
        ),
    )
    parser.add_argument("--sam2-root", type=Path, default=THIRD_PARTY_ROOT / "sam2")
    parser.add_argument("--sai3d-root", type=Path, default=THIRD_PARTY_ROOT / "SAI3D_DT")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--xmem-root", type=Path, default=THIRD_PARTY_ROOT / "XMem2")
    parser.add_argument("--xmem-checkpoint", type=Path, default=None)
    parser.add_argument("--xmem-size", type=int, default=480, help="XMem++ internal shorter-edge resolution; -1 keeps input size.")
    parser.add_argument("--cutie-root", type=Path, default=THIRD_PARTY_ROOT / "Cutie")
    parser.add_argument("--cutie-checkpoint", type=Path, default=None)
    parser.add_argument("--cutie-size", type=int, default=480, help="Cutie internal shorter-edge resolution; -1 keeps input size.")
    parser.add_argument("--memory-vos-device", default="auto", help="Device for XMem++ and Cutie: auto, cpu, cuda, etc.")
    parser.add_argument("--v2sam-root", type=Path, default=THIRD_PARTY_ROOT / "V2-SAM")
    parser.add_argument(
        "--v2sam-python",
        type=Path,
        default=Path.home() / "miniconda3" / "envs" / "v2sam-dt" / "bin" / "python",
    )
    parser.add_argument("--v2sam-profile", choices=("ego2exo", "exo2ego"), default="ego2exo")
    parser.add_argument("--v2sam-visual-checkpoint", type=Path, default=None)
    parser.add_argument("--v2sam-fusion-checkpoint", type=Path, default=None)
    parser.add_argument("--v2sam-expert-batch-size", type=int, default=8)
    parser.add_argument("--v2sam-device", default="auto")
    parser.add_argument("--vggts-root", type=Path, default=THIRD_PARTY_ROOT / "VGGT-S")
    parser.add_argument("--vggts-python", type=Path, default=PROJECT_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--vggts-checkpoint", type=Path, default=None)
    parser.add_argument("--vggts-device", default="auto")
    parser.add_argument(
        "--feedforward-point-cloud",
        type=Path,
        default=None,
        help=(
            "Aligned feed-forward initializer PLY. Defaults to the exact OMeGa copy at "
            "<omega-run>/dataset/vfm_sparse/0/points3D.ply."
        ),
    )
    parser.add_argument(
        "--feedforward-init-mesh",
        type=Path,
        default=None,
        help="Optional matching initialization mesh. Defaults to <omega-run>/init_mesh.ply.",
    )
    parser.add_argument(
        "--omega-final-mesh",
        type=Path,
        default=None,
        help=(
            "Final optimized OMeGa triangle mesh. Defaults to the strongest sibling run's "
            "plys/mesh_29999_rank0.ply."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = build_parser().parse_args(argv)
    app = create_app(
        model_dir=args.model_dir,
        baseline_name=args.baseline_name,
        max_points=int(args.max_points),
        seed=int(args.seed),
        label_source=str(args.label_source),
        sam2_root=args.sam2_root,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=str(args.sam2_config),
        sam2_device=str(args.sam2_device),
        xmem_root=args.xmem_root,
        xmem_checkpoint=args.xmem_checkpoint,
        xmem_size=int(args.xmem_size),
        cutie_root=args.cutie_root,
        cutie_checkpoint=args.cutie_checkpoint,
        cutie_size=int(args.cutie_size),
        memory_vos_device=str(args.memory_vos_device),
        v2sam_root=args.v2sam_root,
        v2sam_python=args.v2sam_python,
        v2sam_profile=str(args.v2sam_profile),
        v2sam_visual_checkpoint=args.v2sam_visual_checkpoint,
        v2sam_fusion_checkpoint=args.v2sam_fusion_checkpoint,
        v2sam_expert_batch_size=int(args.v2sam_expert_batch_size),
        v2sam_device=str(args.v2sam_device),
        vggts_root=args.vggts_root,
        vggts_python=args.vggts_python,
        vggts_checkpoint=args.vggts_checkpoint,
        vggts_device=str(args.vggts_device),
        feedforward_point_cloud=args.feedforward_point_cloud,
        feedforward_init_mesh=args.feedforward_init_mesh,
        omega_final_mesh=args.omega_final_mesh,
        sai3d_root=args.sai3d_root,
    )
    print(f"Interactive editor: http://{args.host}:{args.port}")
    print(f"Model: {args.model_dir}")
    print(f"Baseline: {args.baseline_name}")
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
