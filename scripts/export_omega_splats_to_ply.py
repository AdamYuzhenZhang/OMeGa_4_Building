#!/usr/bin/env python3
"""Export an OMeGa mesh-splat checkpoint as a static Gaussian-splat PLY.

OMeGa stores splats in a mesh-attached form: each splat keeps barycentric
coordinates on a triangle plus learned in-plane scale/rotation. During rendering,
``update_gs`` converts those parameters into world-space Gaussian centers,
in-plane scales, and quaternions. This script reuses that same conversion, then
writes either a standard 3DGS-style PLY for browser viewers such as SuperSplat
Viewer, or a native 2DGS-style PLY with only two scale axes for OMeGa/2DGS tools.

For runs using this repo's optional ``--app_opt`` path, appearance is
augmented by a small MLP plus per-image embeddings. A static PLY cannot
represent that exactly, so this exporter bakes the learned base color by
default. The geometry export still uses
the final mesh-attached splat positions, scales, rotations, and opacities.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData, PlyElement


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPO_ROOT / "examples"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXAMPLES_ROOT))

from examples.simple_trainer_meshgs import Config, update_gs  # noqa: E402
from utils import rgb_to_sh  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an OMeGa training checkpoint to a static splat PLY."
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help="OMeGa result directory containing cfg.json and ckpts/.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to the latest ckpts/ckpt_*_rank0.pt.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PLY. Defaults to <result_dir>/web_exports/omega_final_<format>.ply.",
    )
    parser.add_argument(
        "--format",
        choices=("3dgs", "2dgs"),
        default="3dgs",
        help="3dgs writes a SuperSplat-style 3-axis approximation. 2dgs writes OMeGa/2DGS-style two-axis surfel scales.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for baking mesh-attached splats. CPU is slower but avoids GPU memory pressure.",
    )
    parser.add_argument(
        "--thickness-m",
        type=float,
        default=0.003,
        help="Normal-direction Gaussian thickness for 3DGS viewers. OMeGa renders surfel-like 2D splats, so this is only a static-viewer approximation.",
    )
    parser.add_argument(
        "--scale-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to all exported Gaussian scales.",
    )
    parser.add_argument(
        "--opacity-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied after sigmoid opacity, then converted back to logit.",
    )
    parser.add_argument(
        "--sh-degree",
        type=int,
        default=0,
        choices=(0, 1, 2, 3),
        help="Maximum SH degree to export. 0 exports only DC color, which is smallest and best for app-opt runs.",
    )
    parser.add_argument(
        "--color-mode",
        choices=("base",),
        default="base",
        help="Color baking mode. base uses sigmoid(colors) for app-opt runs or existing SH DC for SH runs.",
    )
    parser.add_argument(
        "--max-splats",
        type=int,
        default=0,
        help="Optional cap for quick viewer tests. 0 exports all splats.",
    )
    parser.add_argument(
        "--binary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write binary little-endian PLY. Use --no-binary for ASCII debugging only.",
    )
    return parser.parse_args()


def latest_checkpoint(result_dir: Path) -> Path:
    ckpt_dir = result_dir / "ckpts"
    candidates = sorted(ckpt_dir.glob("ckpt_*_rank0.pt"))
    if not candidates:
        raise SystemExit(f"No rank0 checkpoints found in {ckpt_dir}")

    def step(path: Path) -> int:
        name = path.name
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return max(candidates, key=step)


def load_config(result_dir: Path) -> Config:
    cfg_path = result_dir / "cfg.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing OMeGa cfg.json: {cfg_path}")
    cfg_data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return Config(**cfg_data)


def to_device_dict(payload: dict[str, object], device: torch.device) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.to(device)
    return tensors


def select_splats(payload: dict[str, torch.Tensor], max_splats: int) -> dict[str, torch.Tensor]:
    if max_splats <= 0:
        return payload
    selected: dict[str, torch.Tensor] = {}
    n = None
    for value in payload.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            n = int(value.shape[0])
            break
    if n is None:
        return payload
    keep = min(int(max_splats), n)
    for key, value in payload.items():
        selected[key] = value[:keep] if value.ndim > 0 and value.shape[0] == n else value
    return selected


def logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.clip(values, eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def sh_rest_count(sh_degree: int) -> int:
    return max(0, ((int(sh_degree) + 1) ** 2 - 1) * 3)


def make_attribute_names(sh_degree: int, scale_dims: int) -> list[str]:
    names = ["x", "y", "z", "nx", "ny", "nz"]
    names.extend(f"f_dc_{i}" for i in range(3))
    names.extend(f"f_rest_{i}" for i in range(sh_rest_count(sh_degree)))
    names.append("opacity")
    names.extend(f"scale_{i}" for i in range(scale_dims))
    names.extend(f"rot_{i}" for i in range(4))
    return names


def tensor_np(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def bake_colors(
    *,
    splats: dict[str, torch.Tensor],
    sh_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PLY f_dc and f_rest arrays.

    Non-app-opt OMeGa runs already store SH coefficients. App-opt runs store a
    learned base RGB logit plus feature/MLP terms; static PLY export cannot carry
    the MLP, so we bake just the base RGB to SH DC and leave higher SH terms zero.
    """

    rest_count = sh_rest_count(sh_degree)
    if "sh0" in splats:
        f_dc = tensor_np(splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous())
        if sh_degree > 0 and "shN" in splats:
            full_rest = tensor_np(splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous())
            f_rest = np.zeros((f_dc.shape[0], rest_count), dtype=np.float32)
            f_rest[:, : min(rest_count, full_rest.shape[1])] = full_rest[:, : min(rest_count, full_rest.shape[1])]
        else:
            f_rest = np.zeros((f_dc.shape[0], rest_count), dtype=np.float32)
        return f_dc, f_rest

    if "colors" not in splats:
        raise SystemExit("Checkpoint has neither sh0/shN nor app-opt colors; cannot bake PLY color.")

    rgb = torch.sigmoid(splats["colors"].detach()).clamp(0.0, 1.0)
    f_dc = tensor_np(rgb_to_sh(rgb))
    f_rest = np.zeros((f_dc.shape[0], rest_count), dtype=np.float32)
    return f_dc, f_rest


def write_gaussian_ply(
    *,
    out_path: Path,
    means: np.ndarray,
    f_dc: np.ndarray,
    f_rest: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    quats: np.ndarray,
    binary: bool,
) -> None:
    normals = np.zeros_like(means, dtype=np.float32)
    attributes = [means, normals, f_dc]
    if f_rest.shape[1] > 0:
        attributes.append(f_rest)
    attributes.extend([opacities[:, None], scales, quats])
    merged = np.concatenate(attributes, axis=1).astype(np.float32, copy=False)

    names = make_attribute_names(
        sh_degree=int(round(math.sqrt(max(f_rest.shape[1] // 3, 0) + 1) - 1)),
        scale_dims=int(scales.shape[1]),
    )
    dtype = [(name, "f4") for name in names]
    if len(dtype) != merged.shape[1]:
        raise RuntimeError(f"Internal attribute mismatch: {len(dtype)} names for {merged.shape[1]} columns")

    elements = np.empty(means.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, merged))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")], text=not binary).write(out_path)


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_checkpoint(result_dir)
    out_path = (
        args.out.expanduser().resolve()
        if args.out
        else result_dir / "web_exports" / f"omega_final_{args.format}.ply"
    )
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if float(args.thickness_m) <= 0.0:
        raise SystemExit("--thickness-m must be positive.")
    if float(args.scale_multiplier) <= 0.0:
        raise SystemExit("--scale-multiplier must be positive.")
    if float(args.opacity_multiplier) <= 0.0:
        raise SystemExit("--opacity-multiplier must be positive.")

    cfg = load_config(result_dir)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "mesh_params" not in ckpt or "splats" not in ckpt:
        raise SystemExit(f"Checkpoint is not an OMeGa training-resume checkpoint: {checkpoint}")

    mesh_params = to_device_dict(ckpt["mesh_params"], device)
    splats = to_device_dict(ckpt["splats"], device)
    splats = select_splats(splats, int(args.max_splats))

    # Reuse OMeGa's own mesh-attached splat baking. This computes:
    #   mean = A + u * (B - A) + v * (C - A)
    #   scale_xy = triangle-local scale * learned relative scale
    #   quat = rotation built from triangle tangent axes plus learned in-plane rot
    update_gs(cfg, mesh_params, splats, optimizers={}, world_size=1)

    means = tensor_np(splats["means"])
    scales = tensor_np(splats["scales"])
    quats = tensor_np(torch.nn.functional.normalize(splats["quats"], dim=-1))
    opacities_prob = torch.sigmoid(splats["opacities"].detach()).cpu().numpy().astype(np.float32)
    opacities_prob = np.clip(opacities_prob * float(args.opacity_multiplier), 1e-6, 1.0 - 1e-6)
    opacities = logit(opacities_prob).astype(np.float32)
    f_dc, f_rest = bake_colors(splats=splats, sh_degree=int(args.sh_degree))

    scales = scales.copy()
    scales[:, :2] += math.log(float(args.scale_multiplier))
    if args.format == "3dgs":
        # OMeGa is surfel-like. Its 2DGS renderer consumes the two in-plane axes
        # through ray-splat intersection and sets the third scale to exp(0)=1.
        # Generic 3DGS viewers interpret all three axes as a volumetric ellipsoid,
        # so use a small physical thickness for the normal axis.
        scales[:, 2] = math.log(float(args.thickness_m) * float(args.scale_multiplier))
        format_note = "3DGS PLY approximation from OMeGa mesh-attached surfel splats"
    else:
        # Native OMeGa/2DGS convention: keep only the two in-plane log-scales.
        # This follows examples/utils.py::save_ply(is_2dgs=True). It is not a
        # standard SuperSplat input format because generic 3DGS viewers expect
        # scale_0, scale_1, and scale_2.
        scales = scales[:, :2]
        format_note = "2DGS PLY from OMeGa mesh-attached surfel splats"

    write_gaussian_ply(
        out_path=out_path,
        means=means,
        f_dc=f_dc,
        f_rest=f_rest,
        opacities=opacities,
        scales=scales,
        quats=quats,
        binary=bool(args.binary),
    )

    summary = {
        "resultDir": str(result_dir),
        "checkpoint": str(checkpoint),
        "out": str(out_path),
        "numSplats": int(means.shape[0]),
        "format": format_note,
        "exportFormat": args.format,
        "colorMode": args.color_mode,
        "shDegree": int(args.sh_degree),
        "thicknessM": float(args.thickness_m) if args.format == "3dgs" else None,
        "scaleMultiplier": float(args.scale_multiplier),
        "opacityMultiplier": float(args.opacity_multiplier),
        "binary": bool(args.binary),
        "notes": [
            "World-space means/scales/quaternions are baked via OMeGa update_gs.",
            "For app_opt checkpoints, colors are static base colors; this repo's optional appearance MLP is not represented in PLY.",
            "format=3dgs adds a chosen normal thickness for generic 3DGS viewers.",
            "format=2dgs writes only two in-plane scale fields, matching OMeGa/2DGS-style surfel PLY conventions.",
        ],
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[done] Exported {means.shape[0]} splats to {out_path}")
    print(f"[done] Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
