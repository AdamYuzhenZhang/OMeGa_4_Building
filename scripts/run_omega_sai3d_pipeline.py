#!/usr/bin/env python3
"""Run the full SAI3D segmentation pipeline in OMeGa conventions."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent.parent
DEFAULT_SAM2_ROOT = PROJECT_ROOT / "third_party" / "sam2"
DEFAULT_SAI3D_ROOT = PROJECT_ROOT / "third_party" / "SAI3D_DT"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "scan_processing_outputs"


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("\n[omega-sai3d-pipeline] " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _maybe_path(path: Path | None) -> str | None:
    return str(path.expanduser().resolve()) if path is not None else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-command SAI3D pipeline: optional 1024px SAM2 proposals, SAI3D "
            "observe/segment/proposal-gated view masks, and debug visualizations."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, default=None, help="Capture package root, needed for proposals and visualization.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--stage", choices=("all", "proposals", "sai3d", "visualize"), default="all")
    parser.add_argument("--baseline-name", default="sai3d_area_samples_1024_dense")
    parser.add_argument("--proposal-name", default="view_proposals_1024")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--clean-baseline", action="store_true", help="Delete the SAI3D baseline directory before running SAI3D stages.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite staged SAI3D outputs and view masks.")

    parser.add_argument("--regenerate-proposals", action="store_true", help="Rebuild the common proposal source even if it already exists.")
    parser.add_argument("--proposal-image-width", type=int, default=1024)
    parser.add_argument("--proposal-frame-stride", type=int, default=1)
    parser.add_argument("--proposal-max-frames", type=int, default=0)
    parser.add_argument("--proposal-points-per-side", type=int, default=64)
    parser.add_argument("--proposal-max-masks-per-frame", type=int, default=120)
    parser.add_argument("--proposal-pred-iou-thresh", type=float, default=0.7)
    parser.add_argument("--proposal-stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--proposal-min-mask-area", type=int, default=120)
    parser.add_argument("--proposal-min-assigned-area", type=int, default=80)

    parser.add_argument("--sam2-root", type=Path, default=DEFAULT_SAM2_ROOT)
    parser.add_argument("--sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sai3d-root", type=Path, default=DEFAULT_SAI3D_ROOT)

    parser.add_argument("--point-source", choices=("mesh_vertices", "mesh_area_samples", "point_cloud"), default="mesh_area_samples")
    parser.add_argument("--point-sample-count", type=int, default=160000)
    parser.add_argument("--point-sample-seed", type=int, default=71)
    parser.add_argument("--point-cloud", type=Path, default=None)
    parser.add_argument("--point-cloud-max-points", type=int, default=0)
    parser.add_argument("--point-cloud-sample-seed", type=int, default=71)
    parser.add_argument("--point-visibility-source", choices=("auto", "rendered_depth", "point_zbuffer"), default="auto")
    parser.add_argument("--point-zbuffer-depth-band", type=float, default=0.02)
    parser.add_argument("--superpoint-mode", choices=("voxel", "vertex"), default="voxel")
    parser.add_argument("--superpoint-target-count", type=int, default=30000)
    parser.add_argument("--superpoint-voxel-size", type=float, default=0.0)
    parser.add_argument("--graph-max-edge-length", type=float, default=0.0)
    parser.add_argument("--sai3d-view-stride", type=int, default=1)
    parser.add_argument("--thres-connect", default="0.9,0.5,5")
    parser.add_argument("--thres-merge", type=int, default=200)
    parser.add_argument("--max-neighbor-distance", type=int, default=2)
    parser.add_argument("--similar-metric", choices=("2-norm", "1-norm", "inf-norm", "Hellinger"), default="2-norm")
    parser.add_argument("--k-graph", type=int, default=8)
    parser.add_argument("--sai3d-workers", type=int, default=20)

    parser.add_argument("--view-mask-refine-mode", choices=("geometry", "sam2", "split_splat"), default="split_splat")
    parser.add_argument("--view-mask-sam2-prompt-count", type=int, default=10)
    parser.add_argument("--view-mask-prompt-erode-px", type=int, default=5)
    parser.add_argument("--view-mask-sam2-min-iou", type=float, default=0.28)
    parser.add_argument("--view-mask-sam2-min-geometry-recall", type=float, default=0.35)
    parser.add_argument("--view-mask-sam2-max-area-ratio", type=float, default=3.0)
    parser.add_argument("--view-mask-split-splat-min-support-pixels", type=int, default=16)
    parser.add_argument("--view-mask-split-splat-min-majority", type=float, default=0.58)
    parser.add_argument("--view-mask-split-splat-min-support-ratio", type=float, default=0.002)
    parser.add_argument("--view-mask-split-splat-min-point-recall", type=float, default=0.20)
    parser.add_argument("--view-mask-split-splat-max-proposal-area-ratio", type=float, default=8.0)

    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--visualization-frame-stride", type=int, default=4)
    parser.add_argument("--visualization-max-frames", type=int, default=32)
    parser.add_argument("--visualization-panel-cell-width", type=int, default=420)
    parser.add_argument("--visualization-contact-cols", type=int, default=1)
    parser.add_argument("--scan-processing-root", type=Path, default=PROJECT_ROOT / "scan_processing")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _proposal_summary(model_dir: Path, proposal_name: str) -> Path:
    return model_dir / "segmentation" / "baselines" / proposal_name / "baseline_summary.json"


def _baseline_dir(model_dir: Path, baseline_name: str) -> Path:
    return model_dir / "segmentation" / "baselines" / baseline_name


def _common_sai3d_args(args: argparse.Namespace) -> list[str]:
    out = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "run_omega_segmentation_baselines.py"),
        "--baseline",
        "sai3d",
        "--model-dir",
        str(args.model_dir),
        "--baseline-name",
        str(args.baseline_name),
        "--proposal-source-name",
        str(args.proposal_name),
        "--source-frame-stride",
        "1",
        "--max-frames",
        "0",
        "--sai3d-root",
        str(args.sai3d_root),
        "--point-source",
        str(args.point_source),
        "--point-sample-count",
        str(args.point_sample_count),
        "--point-sample-seed",
        str(args.point_sample_seed),
        "--point-cloud-max-points",
        str(args.point_cloud_max_points),
        "--point-cloud-sample-seed",
        str(args.point_cloud_sample_seed),
        "--point-visibility-source",
        str(args.point_visibility_source),
        "--point-zbuffer-depth-band",
        str(args.point_zbuffer_depth_band),
        "--superpoint-mode",
        str(args.superpoint_mode),
        "--superpoint-target-count",
        str(args.superpoint_target_count),
        "--superpoint-voxel-size",
        str(args.superpoint_voxel_size),
        "--graph-max-edge-length",
        str(args.graph_max_edge_length),
        "--sai3d-view-stride",
        str(args.sai3d_view_stride),
        "--thres-connect",
        str(args.thres_connect),
        "--thres-merge",
        str(args.thres_merge),
        "--max-neighbor-distance",
        str(args.max_neighbor_distance),
        "--similar-metric",
        str(args.similar_metric),
        "--k-graph",
        str(args.k_graph),
        "--sai3d-workers",
        str(args.sai3d_workers),
    ]
    if args.scene_name:
        out.extend(["--scene-name", str(args.scene_name)])
    if args.point_cloud is not None:
        out.extend(["--point-cloud", str(args.point_cloud)])
    if args.overwrite:
        out.append("--overwrite")
    return out


def run_proposals(args: argparse.Namespace) -> None:
    if args.capture_root is None:
        raise SystemExit("--capture-root is required when running proposal generation.")
    summary = _proposal_summary(args.model_dir, str(args.proposal_name))
    if summary.exists() and not args.regenerate_proposals:
        print(f"[omega-sai3d-pipeline] Reusing existing proposals: {summary}", flush=True)
        return
    cmd = [
        str(args.python),
        str(REPO_ROOT / "scripts" / "run_omega_segmentation_baselines.py"),
        "--baseline",
        "view_proposals",
        "--stage",
        "all",
        "--model-dir",
        str(args.model_dir),
        "--baseline-name",
        str(args.proposal_name),
        "--capture-root",
        str(args.capture_root),
        "--image-width",
        str(args.proposal_image_width),
        "--frame-stride",
        str(args.proposal_frame_stride),
        "--max-frames",
        str(args.proposal_max_frames),
        "--sam2-root",
        str(args.sam2_root),
        "--sam2-checkpoint",
        str(args.sam2_checkpoint),
        "--sam2-config",
        str(args.sam2_config),
        "--device",
        str(args.device),
        "--points-per-side",
        str(args.proposal_points_per_side),
        "--pred-iou-thresh",
        str(args.proposal_pred_iou_thresh),
        "--stability-score-thresh",
        str(args.proposal_stability_score_thresh),
        "--min-mask-area",
        str(args.proposal_min_mask_area),
        "--min-assigned-area",
        str(args.proposal_min_assigned_area),
        "--max-masks-per-frame",
        str(args.proposal_max_masks_per_frame),
        "--overwrite",
    ]
    _run(cmd, dry_run=bool(args.dry_run))


def run_sai3d(args: argparse.Namespace) -> None:
    baseline_dir = _baseline_dir(args.model_dir, str(args.baseline_name))
    if args.clean_baseline and baseline_dir.exists():
        print(f"[omega-sai3d-pipeline] Removing old SAI3D baseline: {baseline_dir}", flush=True)
        if not args.dry_run:
            shutil.rmtree(baseline_dir)

    common = _common_sai3d_args(args)
    for stage in ("prepare", "observe", "segment"):
        _run(common + ["--stage", stage], dry_run=bool(args.dry_run))

    view_cmd = common + [
        "--stage",
        "view_masks",
        "--view-mask-refine-mode",
        str(args.view_mask_refine_mode),
        "--view-mask-sam2-root",
        str(args.sam2_root),
        "--view-mask-sam2-checkpoint",
        str(args.sam2_checkpoint),
        "--view-mask-sam2-config",
        str(args.sam2_config),
        "--view-mask-sam2-prompt-count",
        str(args.view_mask_sam2_prompt_count),
        "--view-mask-prompt-erode-px",
        str(args.view_mask_prompt_erode_px),
        "--view-mask-sam2-min-iou",
        str(args.view_mask_sam2_min_iou),
        "--view-mask-sam2-min-geometry-recall",
        str(args.view_mask_sam2_min_geometry_recall),
        "--view-mask-sam2-max-area-ratio",
        str(args.view_mask_sam2_max_area_ratio),
        "--view-mask-split-splat-min-support-pixels",
        str(args.view_mask_split_splat_min_support_pixels),
        "--view-mask-split-splat-min-majority",
        str(args.view_mask_split_splat_min_majority),
        "--view-mask-split-splat-min-support-ratio",
        str(args.view_mask_split_splat_min_support_ratio),
        "--view-mask-split-splat-min-point-recall",
        str(args.view_mask_split_splat_min_point_recall),
        "--view-mask-split-splat-max-proposal-area-ratio",
        str(args.view_mask_split_splat_max_proposal_area_ratio),
    ]
    _run(view_cmd, dry_run=bool(args.dry_run))


def run_visualization(args: argparse.Namespace) -> None:
    if args.capture_root is None:
        raise SystemExit("--capture-root is required when running visualization.")
    if args.skip_visualization:
        return
    base = f"{args.model_dir.name}_{args.baseline_name}"
    observe_dir = args.capture_root / "visualizations" / "omega_segmentation" / "sai3d_observe" / base
    final_dir = args.capture_root / "visualizations" / "omega_segmentation" / "sai3d" / base
    common = [
        str(args.capture_root),
        "--out-root",
        str(args.out_root),
        "--model-dir",
        str(args.model_dir),
        "--baseline-name",
        str(args.baseline_name),
        "--frame-stride",
        str(args.visualization_frame_stride),
        "--max-frames",
        str(args.visualization_max_frames),
        "--panel-cell-width",
        str(args.visualization_panel_cell_width),
        "--contact-cols",
        str(args.visualization_contact_cols),
        "--overwrite",
    ]
    _run(
        [
            str(args.python),
            str(args.scan_processing_root / "VisSegOmega05_visualize_sai3d_observe.py"),
            *common,
            "--output-dir",
            str(observe_dir),
        ],
        dry_run=bool(args.dry_run),
    )
    _run(
        [
            str(args.python),
            str(args.scan_processing_root / "VisSegOmega04_visualize_sai3d_baseline.py"),
            *common,
            "--output-dir",
            str(final_dir),
        ],
        dry_run=bool(args.dry_run),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.model_dir = args.model_dir.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    args.python = args.python.expanduser()
    if not args.python.is_absolute():
        args.python = (Path.cwd() / args.python).resolve()
    args.capture_root = Path(_maybe_path(args.capture_root)) if args.capture_root is not None else None
    args.sam2_root = args.sam2_root.expanduser().resolve()
    args.sai3d_root = args.sai3d_root.expanduser().resolve()
    args.scan_processing_root = args.scan_processing_root.expanduser().resolve()
    if args.sam2_checkpoint is None:
        args.sam2_checkpoint = args.sam2_root / "checkpoints" / "sam2.1_hiera_large.pt"
    else:
        args.sam2_checkpoint = args.sam2_checkpoint.expanduser().resolve()

    print("=== OMeGa SAI3D Pipeline ===")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Model dir: {args.model_dir}")
    print(f"Proposal source: {args.proposal_name}")
    print(f"SAI3D baseline: {args.baseline_name}")
    print(f"Point source: {args.point_source}, samples={args.point_sample_count}, superpoint target={args.superpoint_target_count}")

    if args.stage in {"all", "proposals"}:
        run_proposals(args)
    if args.stage in {"all", "sai3d"}:
        run_sai3d(args)
    if args.stage in {"all", "visualize"}:
        run_visualization(args)

    print("[omega-sai3d-pipeline] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
