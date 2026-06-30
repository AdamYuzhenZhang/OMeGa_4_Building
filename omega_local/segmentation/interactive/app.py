"""Web entry point for the OMeGa interactive segmentation editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .paths import STATIC_DIR, THIRD_PARTY_ROOT, require_dir, resolve_paths
from .sam2_session import Sam2Config
from .state import EditorState


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
) -> FastAPI:
    paths = resolve_paths(model_dir, baseline_name)
    sam2_root = require_dir(sam2_root, "SAM2 root")
    checkpoint = (
        sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
        if sam2_checkpoint is None
        else sam2_checkpoint.expanduser().resolve()
    )
    state = EditorState(
        paths,
        max_points=max_points,
        seed=seed,
        label_source=label_source,
        sam2_config=Sam2Config(
            root=sam2_root,
            checkpoint=checkpoint,
            config=str(sam2_config),
            device=str(sam2_device),
        ),
    )

    app = FastAPI(title="OMeGa Interactive Segmentation Editor")
    app.state.editor_state = state
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.middleware("http")
    async def no_cache_editor_assets(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
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

    @app.get("/api/labels/summary")
    def label_summary() -> Response:
        return _json_response(state.label_summary())

    @app.get("/api/frames")
    def frames() -> Response:
        return _json_response(state.frames())

    @app.get("/api/proposals/sam2/status")
    def proposal_status() -> Response:
        return _json_response(state.proposal_status())

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

    @app.post("/api/labels/save")
    def save_labels(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.save_interactive_labels(payload))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/proposals/sam2/frame/{frame_id}/overlay")
    def proposal_overlay(frame_id: int) -> FileResponse:
        try:
            path = state.proposal_overlay_path(frame_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/view-evidence/frame/{frame_id}/{kind}")
    def view_evidence_image(frame_id: int, kind: str) -> FileResponse:
        try:
            path = state.view_evidence_image_path(frame_id, kind)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(path, media_type="image/png")

    @app.get("/api/proposals/sam2/frame/{frame_id}/summary")
    def proposal_frame_summary(frame_id: int) -> Response:
        try:
            return _json_response(state.proposal_frame_summary(frame_id))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/frame/{frame_id}/pick")
    def proposal_pick(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.pick_proposal(frame_id, payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/save-edits")
    def proposal_save_edits(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.save_proposal_edits(payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/frame/{frame_id}/selection-preview")
    def proposal_selection_preview(frame_id: int, payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.preview_selection(frame_id, payload))
        except (KeyError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/proposals/sam2/propagate-selection")
    def proposal_propagate_selection(payload: dict[str, Any] = Body(...)) -> Response:
        try:
            return _json_response(state.propagate_selection(payload))
        except (KeyError, FileNotFoundError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        choices=("raw", "saved", "sai3d", "auto"),
        default="raw",
        help=(
            "Initial labels to load. raw renders the sampled SAI3D input points as "
            "unsegmented label 0; saved loads interactive_labels.npy; sai3d loads "
            "mesh_labels/point_labels.npy; auto tries saved, then sai3d, then raw."
        ),
    )
    parser.add_argument("--sam2-root", type=Path, default=THIRD_PARTY_ROOT / "sam2")
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
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
    )
    print(f"Interactive editor: http://{args.host}:{args.port}")
    print(f"Model: {args.model_dir}")
    print(f"Baseline: {args.baseline_name}")
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
