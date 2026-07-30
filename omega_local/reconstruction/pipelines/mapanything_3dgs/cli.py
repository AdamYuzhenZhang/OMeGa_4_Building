"""CLI for the MapAnything persistent-region 3DGS pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import STAGE_ORDER, MapAnything3DGSConfig
from .manager import MapAnything3DGSManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split persistent regions on the OMeGa MapAnything initializer, "
            "then train independent point-initialized 3DGS region models."
        )
    )
    parser.add_argument(
        "--stage",
        choices=(
            "status",
            *STAGE_ORDER,
            "all_split",
            "all_splat",
            "all",
        ),
        default="status",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--editor-baseline-name",
        default="sai3d_area_samples_1024_dense",
    )
    parser.add_argument(
        "--split-splat-root",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "Split_and_Splat",
        help="Released trainer used only as the masked 3DGS optimization backend.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--run-id",
        default="mapanything_sam2_video",
    )
    parser.add_argument("--propagation-method", default="sam2_video")
    parser.add_argument(
        "--point-cloud",
        type=Path,
        help=(
            "Optional aligned MapAnything PLY. By default the exact OMeGa "
            "initializer at dataset/vfm_sparse/0/points3D.ply is used."
        ),
    )
    parser.add_argument("--manual-frame-weight", type=int, default=8)
    parser.add_argument("--instance-iterations", type=int, default=10_000)
    parser.add_argument("--minimum-region-points", type=int, default=32)
    parser.add_argument("--minimum-positive-views", type=int, default=2)
    parser.add_argument("--max-num-splats", type=int, default=100_000)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--disable-floater-pruning", action="store_true")
    parser.add_argument(
        "--floater-opacity-threshold", type=float, default=0.005
    )
    parser.add_argument(
        "--floater-min-visible-views", type=int, default=3
    )
    parser.add_argument(
        "--floater-mask-dilation-pixels", type=int, default=1
    )
    parser.add_argument("--overwrite-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = MapAnything3DGSManager(
        MapAnything3DGSConfig(
            model_dir=args.model_dir,
            editor_baseline_name=args.editor_baseline_name,
            split_splat_root=args.split_splat_root,
            python=args.python,
            run_id=args.run_id,
            propagation_method=args.propagation_method,
            point_cloud=args.point_cloud,
            manual_frame_weight=args.manual_frame_weight,
            instance_iterations=args.instance_iterations,
            minimum_region_points=args.minimum_region_points,
            minimum_positive_views=args.minimum_positive_views,
            max_num_splats=args.max_num_splats,
            mask_loss_weight=args.mask_loss_weight,
            floater_pruning=not args.disable_floater_pruning,
            floater_opacity_threshold=args.floater_opacity_threshold,
            floater_min_visible_views=args.floater_min_visible_views,
            floater_mask_dilation_pixels=(
                args.floater_mask_dilation_pixels
            ),
        )
    )
    result = (
        manager.status()
        if args.stage == "status"
        else manager.run_stage(
            args.stage,
            overwrite=bool(args.overwrite_stage),
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
