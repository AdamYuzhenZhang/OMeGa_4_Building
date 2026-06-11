#!/usr/bin/env python3
"""Build Phase 3 remesh policy fields and planar proxies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from omega_local.remesh.planar_proxies import PlanarProxyConfig, extract_planar_proxies  # noqa: E402
from omega_local.remesh.policy import RemeshPolicyConfig, compute_remesh_policy  # noqa: E402
from omega_local.remesh.weights import finalize_policy_weights  # noqa: E402


def _resolve(path: Path | None, default: Path, base: Path) -> Path:
    if path is None or str(path).strip() in {"", "."}:
        return default
    raw = path.expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [raw.resolve(), (base / raw).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _resolve_existing_file(path: Path | None, default: Path, base: Path, description: str) -> Path:
    resolved = _resolve(path, default, base)
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    if resolved.is_dir():
        raise SystemExit(
            f"{description} points to a directory, not a file: {resolved}\n"
            "If this came from an environment variable, re-export it or omit the flag to use the default."
        )
    return resolved


def _resolve_output_dir(path: Path | None, default: Path) -> Path:
    if path is None or str(path).strip() in {"", "."}:
        return default
    return path.expanduser().resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Phase 3: remesh policy fields plus planar proxies.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/preclean_mesh.ply.")
    parser.add_argument("--evidence-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/evidence.npz.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/local.")
    parser.add_argument("--policy-name", default="policy")
    parser.add_argument("--proxies-name", default="proxies")
    parser.add_argument("--planarity-rings", type=int, default=2)
    parser.add_argument("--support-min", type=float, default=0.08)
    parser.add_argument("--planar-score-threshold", type=float, default=0.35)
    parser.add_argument("--detail-max-for-planar", type=float, default=0.75)
    parser.add_argument("--unexplained-max-for-planar", type=float, default=0.55)
    parser.add_argument("--min-tau-a-degrees", type=float, default=5.0)
    parser.add_argument("--boundary-score-threshold", type=float, default=0.50)
    parser.add_argument("--min-proxy-faces", type=int, default=24)
    parser.add_argument("--seed-planar-score", type=float, default=0.35)
    parser.add_argument("--max-seed-detail", type=float, default=0.75)
    parser.add_argument("--max-seed-unexplained", type=float, default=0.50)
    parser.add_argument("--max-connect-boundary-score", type=float, default=0.45)
    parser.add_argument("--fit-trim-quantile", type=float, default=0.90)
    parser.add_argument("--fit-iterations", type=int, default=3)
    parser.add_argument("--max-plane-rmse", type=float, default=0.0)
    parser.add_argument("--no-debug-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = _resolve_output_dir(args.output_dir, model_dir / "remesh" / "local")
    mesh_path = _resolve_existing_file(args.mesh, model_dir / "remesh" / "local" / "preclean_mesh.ply", model_dir, "Mesh")
    evidence_npz = _resolve_existing_file(args.evidence_npz, model_dir / "remesh" / "local" / "evidence.npz", model_dir, "Evidence npz")
    policy_result = compute_remesh_policy(
        RemeshPolicyConfig(
            mesh_path=mesh_path,
            evidence_npz=evidence_npz,
            output_dir=output_dir,
            policy_name=str(args.policy_name),
            planarity_rings=int(args.planarity_rings),
            support_min=float(args.support_min),
            planar_score_threshold=float(args.planar_score_threshold),
            detail_max_for_planar=float(args.detail_max_for_planar),
            unexplained_max_for_planar=float(args.unexplained_max_for_planar),
            min_tau_a_degrees=float(args.min_tau_a_degrees),
            boundary_score_threshold=float(args.boundary_score_threshold),
            write_debug_meshes=not bool(args.no_debug_meshes),
            overwrite=bool(args.overwrite),
        )
    )
    proxy_result = extract_planar_proxies(
        PlanarProxyConfig(
            mesh_path=mesh_path,
            policy_npz=policy_result.policy_npz,
            output_dir=output_dir,
            proxies_name=str(args.proxies_name),
            min_proxy_faces=int(args.min_proxy_faces),
            seed_planar_score=float(args.seed_planar_score),
            max_seed_detail=float(args.max_seed_detail),
            max_seed_unexplained=float(args.max_seed_unexplained),
            max_connect_boundary_score=float(args.max_connect_boundary_score),
            fit_trim_quantile=float(args.fit_trim_quantile),
            fit_iterations=int(args.fit_iterations),
            max_plane_rmse=float(args.max_plane_rmse),
            write_debug_meshes=not bool(args.no_debug_meshes),
            overwrite=bool(args.overwrite),
        )
    )
    finalize_policy_weights(
        mesh_path=mesh_path,
        policy_npz=policy_result.policy_npz,
        proxies_npz=proxy_result.proxies_npz,
        output_dir=output_dir,
        write_debug_meshes=not bool(args.no_debug_meshes),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
