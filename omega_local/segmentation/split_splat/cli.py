"""Command-line entry point for staged Split&Splat experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import (
    PROPOSAL_SOURCES,
    SPLIT_METHODS,
    STAGE_ORDER,
    SplitSplatRunConfig,
)
from .manager import SplitSplatExperimentManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the staged Split&Splat baseline or a controlled proposal-source "
            "substitution on an OMeGa editor dataset."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("status", *STAGE_ORDER, "all_split", "all_splat", "all"),
        default="status",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--editor-baseline-name",
        default="sai3d_area_samples_1024_dense",
        help="SAI3D baseline whose frame manifest defines the editor raster/camera grid.",
    )
    parser.add_argument(
        "--split-splat-root",
        type=Path,
        default=Path(__file__).resolve().parents[4] / "Split_and_Splat",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-id", default="split_splat_official")
    parser.add_argument(
        "--proposal-source",
        choices=PROPOSAL_SOURCES,
        default="official_auto",
        help=(
            "official_auto runs the released four-grid SAM2 generator; propagation "
            "imports an existing exclusive editor propagation layer."
        ),
    )
    parser.add_argument(
        "--propagation-method",
        default="sam2_video",
        help="Editor propagation method ID used when --proposal-source propagation.",
    )
    parser.add_argument(
        "--split-method",
        choices=SPLIT_METHODS,
        default="released",
        help=(
            "released runs the paper Split stage; anchored_3d preserves persistent "
            "region IDs and mask shapes while using the global 3DGS for relabeling."
        ),
    )
    parser.add_argument(
        "--manual-frame-weight",
        type=int,
        default=8,
        help=(
            "Relative evidence and future Splat supervision weight recorded for "
            "completed manual frames in anchored_3d runs."
        ),
    )
    parser.add_argument(
        "--shared-id",
        default=None,
        help="Optional explicit dataset-geometry cache ID. Normally derived automatically.",
    )
    parser.add_argument(
        "--depth-source",
        choices=("murre", "editor_depth"),
        default="murre",
        help="Murre is the paper baseline; editor_depth is a documented Depth Anything V2 ablation.",
    )
    parser.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Directory containing metric .npy/.npz depth maps keyed by editor or COLMAP frame stem.",
    )
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument(
        "--instance-iterations",
        type=int,
        default=1_000,
        help=(
            "Per-instance iterations for both Splat reconstruction passes. "
            "The released ScanNet scripts and paper use 1,000."
        ),
    )
    parser.add_argument(
        "--composition-iterations",
        type=int,
        default=1_000,
        help="Iterations after each parallel composition merge; the paper uses 1,000.",
    )
    parser.add_argument(
        "--composition-mask-weights",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.25],
        help=(
            "Mask-loss schedule across composition rounds. The default follows "
            "the paper's +0.1 progression capped at 0.25."
        ),
    )
    parser.add_argument("--overwrite-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = SplitSplatExperimentManager(
        SplitSplatRunConfig(
            model_dir=args.model_dir,
            editor_baseline_name=args.editor_baseline_name,
            split_splat_root=args.split_splat_root,
            python=args.python,
            run_id=args.run_id,
            shared_id=args.shared_id,
            proposal_source=args.proposal_source,
            propagation_method=args.propagation_method,
            split_method=args.split_method,
            manual_frame_weight=args.manual_frame_weight,
            depth_source=args.depth_source,
            depth_dir=args.depth_dir,
            iterations=args.iterations,
            instance_iterations=args.instance_iterations,
            composition_iterations=args.composition_iterations,
            composition_mask_weights=tuple(args.composition_mask_weights),
        )
    )
    result = (
        manager.status()
        if args.stage == "status"
        else manager.run_stage(
            args.stage,
            overwrite=bool(args.overwrite_stage),
            dry_run=bool(args.dry_run),
            verbose=bool(args.verbose),
        )
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
