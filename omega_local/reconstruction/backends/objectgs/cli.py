"""CLI for ObjectGS joint reconstruction from a MapAnything split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import STAGE_ORDER, ObjectGSConfig
from .manager import ObjectGSManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Jointly reconstruct a MapAnything-labeled scene with released "
            "ObjectGS, parallel to independent per-region splat training."
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
        "--objectgs-root",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "ObjectGS",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--mapanything-run-id",
        default="mapanything_sam2_video",
    )
    parser.add_argument("--run-id", default="objectgs_joint")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument(
        "--semantic-loss-weight",
        type=float,
        default=0.1,
        help="Released 3DOVS/Replica ObjectGS semantic CE weight.",
    )
    parser.add_argument("--initializer-stride", type=int, default=1)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.001,
        help="Released ObjectGS anchor voxel size; use 0 for its auto mode.",
    )
    parser.add_argument(
        "--keep-geometry-completed-labels",
        action="store_true",
        help=(
            "Keep locally completed MapAnything labels instead of exposing "
            "those uncertain initializer points as ObjectGS class 0."
        ),
    )
    parser.add_argument(
        "--mesh-voxel-size",
        type=float,
        default=0.01,
        help=(
            "World-space bounded-TSDF voxel size; 0 uses "
            "depth_trunc / mesh_resolution instead."
        ),
    )
    parser.add_argument(
        "--mesh-resolution",
        type=int,
        default=512,
        help=(
            "Bounded-TSDF depth-range resolution. ObjectGS documents 2048, "
            "but 512 is the resource-safe default for captured scenes."
        ),
    )
    parser.add_argument(
        "--mesh-clusters",
        type=int,
        default=10,
        help="Largest connected components retained per object mesh.",
    )
    parser.add_argument(
        "--mesh-max-triangles",
        type=int,
        default=1_000_000,
        help="QEM cap applied after the released component cleanup.",
    )
    parser.add_argument("--overwrite-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = ObjectGSManager(
        ObjectGSConfig(
            model_dir=args.model_dir,
            editor_baseline_name=args.editor_baseline_name,
            objectgs_root=args.objectgs_root,
            python=args.python,
            mapanything_run_id=args.mapanything_run_id,
            run_id=args.run_id,
            iterations=args.iterations,
            semantic_loss_weight=args.semantic_loss_weight,
            initializer_stride=args.initializer_stride,
            voxel_size=args.voxel_size,
            geometry_completed_as_unknown=(
                not args.keep_geometry_completed_labels
            ),
            mesh_voxel_size=args.mesh_voxel_size,
            mesh_resolution=args.mesh_resolution,
            mesh_clusters=args.mesh_clusters,
            mesh_max_triangles=args.mesh_max_triangles,
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
