"""CLI for anchored per-region 3DGS reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import (
    MASK_REFINEMENT_POLICIES,
    STAGE_ORDER,
    AnchoredSplatConfig,
)
from .manager import AnchoredSplatManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct persistent regions from an anchored Split result while "
            "keeping the released Split&Splat baselines untouched."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("status", *STAGE_ORDER, "all"),
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
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--source-run-id",
        default="split_splat_anchored_sam2_video",
    )
    parser.add_argument(
        "--run-id",
        default="anchored_splat_no_refine",
    )
    parser.add_argument(
        "--mask-refinement",
        choices=MASK_REFINEMENT_POLICIES,
        default="none",
        help=(
            "none preserves anchored masks exactly (default); paper_sam2 runs "
            "the released refiner as an isolated diagnostic."
        ),
    )
    parser.add_argument("--instance-iterations", type=int, default=1_000)
    parser.add_argument("--composition-iterations", type=int, default=1_000)
    parser.add_argument(
        "--composition-mask-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.25],
    )
    parser.add_argument("--minimum-gaussians", type=int, default=8)
    parser.add_argument("--minimum-positive-views", type=int, default=2)
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
    manager = AnchoredSplatManager(
        AnchoredSplatConfig(
            model_dir=args.model_dir,
            editor_baseline_name=args.editor_baseline_name,
            split_splat_root=args.split_splat_root,
            python=args.python,
            source_run_id=args.source_run_id,
            run_id=args.run_id,
            mask_refinement=args.mask_refinement,
            instance_iterations=args.instance_iterations,
            composition_iterations=args.composition_iterations,
            composition_mask_weights=tuple(args.composition_mask_weights),
            minimum_gaussians=args.minimum_gaussians,
            minimum_positive_views=args.minimum_positive_views,
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
