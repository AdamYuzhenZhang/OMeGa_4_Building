"""Command-line interface for one released static semantic 3DGS backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import StaticSemantic3DGSConfig
from .manager import StaticSemantic3DGSManager


_OMEGA_ROOT = Path(__file__).resolve().parents[4]
_PROJECT_ROOT = _OMEGA_ROOT.parent.parent
_DEFAULT_ROOTS = {
    "segment_then_splat": _OMEGA_ROOT.parent / "Segment-then-Splat",
    "gaussian_grouping": _OMEGA_ROOT.parent / "gaussian-grouping",
}


def build_parser(method: str) -> argparse.ArgumentParser:
    display = {
        "segment_then_splat": "Segment then Splat",
        "gaussian_grouping": "Gaussian Grouping",
    }[method]
    parser = argparse.ArgumentParser(
        description=(
            f"Run {display} on the persistent masks and point initializer "
            "from one completed MapAnything region split."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("status", "prepare", "train", "export", "all"),
        default="status",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--editor-baseline-name",
        default="sai3d_area_samples_1024_dense",
    )
    parser.add_argument(
        "--mapanything-run-id",
        default="mapanything_sam2_video",
    )
    parser.add_argument("--run-id", default="joint_user_masks")
    parser.add_argument(
        "--method-root",
        type=Path,
        default=_DEFAULT_ROOTS[method],
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--densify-until-iter", type=int, default=None)
    parser.add_argument("--num-sample-objects", type=int, default=3)
    parser.add_argument("--partial-mask-iou", type=float, default=0.3)
    parser.add_argument("--reg3d-interval", type=int, default=5)
    parser.add_argument("--reg3d-k", type=int, default=5)
    parser.add_argument("--reg3d-lambda", type=float, default=2.0)
    parser.add_argument("--reg3d-max-points", type=int, default=200_000)
    parser.add_argument("--reg3d-sample-size", type=int, default=1_000)
    parser.add_argument(
        "--native-extensions-dir",
        type=Path,
        default=(
            _PROJECT_ROOT / ".native_extensions" / "segment_then_splat"
            if method == "segment_then_splat"
            else None
        ),
    )
    parser.add_argument("--overwrite-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(
    argv: list[str] | None = None,
    *,
    method: str,
) -> int:
    args = build_parser(method).parse_args(argv)
    config = StaticSemantic3DGSConfig(
        model_dir=args.model_dir,
        editor_baseline_name=args.editor_baseline_name,
        method=method,
        method_root=args.method_root,
        python=args.python,
        mapanything_run_id=args.mapanything_run_id,
        run_id=args.run_id,
        iterations=args.iterations,
        resolution=args.resolution,
        densify_until_iter=args.densify_until_iter,
        num_sample_objects=args.num_sample_objects,
        partial_mask_iou=args.partial_mask_iou,
        reg3d_interval=args.reg3d_interval,
        reg3d_k=args.reg3d_k,
        reg3d_lambda=args.reg3d_lambda,
        reg3d_max_points=args.reg3d_max_points,
        reg3d_sample_size=args.reg3d_sample_size,
        native_extensions_dir=args.native_extensions_dir,
    )
    manager = StaticSemantic3DGSManager(config)
    result = (
        manager.status()
        if args.stage == "status"
        else manager.run_stage(
            args.stage,
            overwrite=args.overwrite_stage,
            dry_run=args.dry_run,
        )
    )
    print(json.dumps(result, indent=2))
    return 0
