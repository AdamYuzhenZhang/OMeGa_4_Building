"""CLI for the complete released Segment then Splat pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import OfficialSegmentThenSplatConfig
from .manager import OfficialSegmentThenSplatManager


_OMEGA_ROOT = Path(__file__).resolve().parents[4]
_PROJECT_ROOT = _OMEGA_ROOT.parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paper's SAM1 proposals, SAM2 tracking, COLMAP object "
            "initialization, and joint multilevel static 3DGS training."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("status", "prepare", "autoseg", "initialize", "train", "export", "all"),
        default="status",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--editor-baseline-name", default="sai3d_area_samples_1024_dense")
    parser.add_argument("--run-id", default="official_auto")
    parser.add_argument(
        "--segment-then-splat-root",
        type=Path,
        default=_OMEGA_ROOT.parent / "Segment-then-Splat",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--sam1-checkpoint",
        type=Path,
        default=_OMEGA_ROOT.parent / "PlanarGS" / "ckpt" / "sam_vit_h_4b8939.pth",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=Path,
        default=_OMEGA_ROOT.parent / "V2-SAM" / "weights" / "sam2" / "sam2_hiera_large.pt",
    )
    parser.add_argument(
        "--native-extensions-dir",
        type=Path,
        default=_PROJECT_ROOT / ".native_extensions" / "segment_then_splat",
    )
    parser.add_argument("--image-long-side", type=int, default=1024)
    parser.add_argument("--detect-stride", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=40_000)
    parser.add_argument("--densify-until-iter", type=int, default=20_000)
    parser.add_argument("--num-sample-objects", type=int, default=3)
    parser.add_argument("--partial-mask-iou", type=float, default=0.3)
    parser.add_argument("--overwrite-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = OfficialSegmentThenSplatConfig(
        model_dir=args.model_dir,
        editor_baseline_name=args.editor_baseline_name,
        source_root=args.segment_then_splat_root,
        python=args.python,
        sam1_checkpoint=args.sam1_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        native_extensions_dir=args.native_extensions_dir,
        run_id=args.run_id,
        image_long_side=args.image_long_side,
        detect_stride=args.detect_stride,
        batch_size=args.batch_size,
        iterations=args.iterations,
        densify_until_iter=args.densify_until_iter,
        num_sample_objects=args.num_sample_objects,
        partial_mask_iou=args.partial_mask_iou,
    )
    manager = OfficialSegmentThenSplatManager(config)
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

