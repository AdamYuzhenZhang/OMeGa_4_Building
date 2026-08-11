"""CLI for the released 3D Gaussian Flats bridge."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contract import STAGE_ORDER, GaussianFlatsConfig
from .manager import GaussianFlatsManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run released 3D Gaussian Flats with persistent-region plane masks."
        )
    )
    parser.add_argument("--stage", choices=("status", *STAGE_ORDER, "all"), default="status")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--editor-baseline-name", default="sai3d_area_samples_1024_dense"
    )
    parser.add_argument(
        "--gaussian-flats-root",
        type=Path,
        default=Path(__file__).resolve().parents[5] / "3dgs-flats",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(__file__).resolve().parents[6]
        / ".venv_gaussian_flats"
        / "bin"
        / "python",
    )
    parser.add_argument("--run-id", default="door_planar_official")
    parser.add_argument("--propagation-method", default="sam2_video")
    parser.add_argument("--planar-region", default="Door")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--cap-max", type=int, default=1_000_000)
    parser.add_argument("--resolution", type=int, default=1)
    parser.add_argument("--plane-fit-iter", type=int, default=3_500)
    parser.add_argument("--plane-fit-min-points", type=int, default=100)
    parser.add_argument("--plane-sigma-res", type=float, default=0.01)
    parser.add_argument("--plane-sigma-dist", type=float, default=0.3)
    parser.add_argument("--planar-mask-loss-weight", type=float, default=0.1)
    parser.add_argument("--depthtv-loss-weight", type=float, default=0.1)
    parser.add_argument("--scale-reg", type=float, default=0.01)
    parser.add_argument("--opacity-reg", type=float, default=0.01)
    parser.add_argument("--planar-grid-resolution", type=float, default=0.02)
    parser.add_argument("--planar-tile-size", type=float, default=5.0)
    parser.add_argument("--mesh-voxel-size", type=float, default=0.02)
    parser.add_argument("--overwrite-stage", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = GaussianFlatsManager(
        GaussianFlatsConfig(
            model_dir=args.model_dir,
            editor_baseline_name=args.editor_baseline_name,
            gaussian_flats_root=args.gaussian_flats_root,
            python=args.python,
            run_id=args.run_id,
            propagation_method=args.propagation_method,
            planar_region=args.planar_region,
            iterations=args.iterations,
            cap_max=args.cap_max,
            resolution=args.resolution,
            plane_fit_iter=args.plane_fit_iter,
            plane_fit_min_points=args.plane_fit_min_points,
            plane_sigma_res=args.plane_sigma_res,
            plane_sigma_dist=args.plane_sigma_dist,
            planar_mask_loss_weight=args.planar_mask_loss_weight,
            depthtv_loss_weight=args.depthtv_loss_weight,
            scale_reg=args.scale_reg,
            opacity_reg=args.opacity_reg,
            planar_grid_resolution=args.planar_grid_resolution,
            planar_tile_size=args.planar_tile_size,
            mesh_voxel_size=args.mesh_voxel_size,
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
