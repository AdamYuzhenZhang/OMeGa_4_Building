"""Command-line orchestration for the redesigned OMeGa local remesh pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from omega_local.remesh.evidence import MeshEvidenceConfig, compute_mesh_face_evidence, resolve_dataset_dir
from omega_local.remesh.local_qem import OperationProposalConfig, compute_operation_proposals
from omega_local.remesh.mesh_clean import MeshPreprocessConfig, preprocess_mesh, resolve_latest_omega_mesh
from omega_local.remesh.mesh_heal import MeshHealingConfig, compute_mesh_healing
from omega_local.remesh.planar_patch_remesh import PlanarPatchRemeshConfig, compute_planar_patch_remesh
from omega_local.remesh.planar_proxies import PlanarProxyConfig, extract_planar_proxies
from omega_local.remesh.policy import RemeshPolicyConfig, compute_remesh_policy
from omega_local.remesh.weighted_local_remesh import WeightedRemeshConfig, compute_weighted_remesh
from omega_local.remesh.weights import finalize_policy_weights


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run stages of the redesigned OMeGa local remesh pipeline. "
            "Implemented stages: preprocessing, optional mesh healing, view-normal evidence extraction, policy/proxy extraction, "
            "dry-run local operation proposals, conservative weighted remeshing, and planar patch remeshing."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("preprocess", "heal_preprocess", "evidence", "policy", "proposals", "remesh", "patch_remesh"),
        default="preprocess",
        help="Pipeline stage to run.",
    )
    parser.add_argument("--model-dir", type=Path, default=None, help="OMeGa model/result directory.")
    parser.add_argument("--mesh", "--input-mesh", dest="mesh", type=Path, default=None, help="Mesh to process.")
    parser.add_argument("--iteration", type=int, default=-1, help="Mesh iteration to use when --mesh is omitted.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/local.")
    parser.add_argument("--output-name", default="preclean_mesh.ply")
    parser.add_argument("--summary-name", default="preclean_summary.json")
    parser.add_argument("--diagnostics-name", default="preclean_diagnostics.npz")
    parser.add_argument("--heal-output-name", default="preclean_healed_mesh.ply")
    parser.add_argument("--heal-summary-name", default="preclean_healed_summary.json")
    parser.add_argument("--heal-diagnostics-name", default="preclean_healed_diagnostics.npz")
    parser.add_argument("--cgal-binary", type=Path, default=None)
    parser.add_argument("--heal-repair-degenerate-faces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heal-repair-almost-degenerate-faces", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--heal-duplicate-nonmanifold-vertices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heal-stitch-borders", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heal-fill-holes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--heal-max-hole-edges", type=int, default=48)
    parser.add_argument("--heal-max-hole-diameter-factor", type=float, default=6.5)
    parser.add_argument("--heal-component-area-factor", type=float, default=0.0)
    parser.add_argument("--heal-repair-self-intersections", action="store_true")
    parser.add_argument("--heal-detect-self-intersections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Prepared OMeGa dataset dir for the evidence stage.")
    parser.add_argument("--view-buffer-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/view_buffers.")
    parser.add_argument("--evidence-name", default="evidence")
    parser.add_argument("--evidence-npz", type=Path, default=None, help="Evidence npz for the policy stage.")
    parser.add_argument("--policy-name", default="policy")
    parser.add_argument("--policy-npz", type=Path, default=None, help="Policy npz for the proposals stage.")
    parser.add_argument("--proxies-name", default="proxies")
    parser.add_argument("--proxies-npz", type=Path, default=None, help="Proxies npz for the proposals stage.")
    parser.add_argument("--proxies-json", type=Path, default=None, help="Proxies json for the proposals stage.")
    parser.add_argument("--proposals-name", default="operation_proposals")
    parser.add_argument("--remesh-output-name", default="mesh_weighted_remesh.ply")
    parser.add_argument("--remesh-diagnostics-name", default="weighted_remesh_diagnostics")
    parser.add_argument("--remesh-operations-name", default="weighted_remesh_operations")
    parser.add_argument("--patch-output-name", default="mesh_planar_patch_remesh.ply")
    parser.add_argument("--patch-diagnostics-name", default="planar_patch_remesh_diagnostics")
    parser.add_argument("--patch-operations-name", default="planar_patch_remesh_operations")
    parser.add_argument("--patch-planar-threshold", type=float, default=0.22)
    parser.add_argument("--patch-feature-threshold", type=float, default=0.50)
    parser.add_argument("--patch-boundary-stop-threshold", type=float, default=0.35)
    parser.add_argument("--patch-min-faces", type=int, default=12)
    parser.add_argument("--patch-normal-degrees", type=float, default=12.0)
    parser.add_argument("--patch-distance-factor", type=float, default=4.0)
    parser.add_argument("--patch-trim-distance-factor", type=float, default=2.5)
    parser.add_argument("--patch-planar-target-factor", type=float, default=7.0)
    parser.add_argument("--patch-max-passes", type=int, default=12)
    parser.add_argument("--patch-max-collapses-per-pass", type=int, default=150000)
    parser.add_argument("--patch-collapse-threshold", type=float, default=0.02)
    parser.add_argument("--patch-max-quality-error", type=float, default=2.0)
    parser.add_argument("--max-remesh-passes", type=int, default=8)
    parser.add_argument("--max-collapses-per-pass", type=int, default=100000)
    parser.add_argument("--max-flips-per-pass", type=int, default=50000)
    parser.add_argument("--collapse-threshold", type=float, default=0.01)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.10)
    parser.add_argument("--max-normal-error", type=float, default=2.0)
    parser.add_argument("--max-quality-error", type=float, default=1.5)
    parser.add_argument("--max-projection-distance-factor", type=float, default=8.0)
    parser.add_argument("--plane-residual-factor", type=float, default=2.0)
    parser.add_argument("--alpha-planar-transition-collapse", type=float, default=3.00)
    parser.add_argument("--alpha-feature-edge-collapse", type=float, default=3.00)
    parser.add_argument("--alpha-detail-collapse", type=float, default=1.75)
    parser.add_argument("--detail-mesh-target-factor", type=float, default=2.25)
    parser.add_argument("--detail-simplify-min-weight", type=float, default=0.25)
    parser.add_argument("--pre-project-before-collapse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-edge-collapse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-edge-min-planar", type=float, default=0.35)
    parser.add_argument("--feature-edge-max-uncertain", type=float, default=1.10)
    parser.add_argument("--planar-boundary-softening", type=float, default=0.85)
    parser.add_argument("--planar-transition-min-weight", type=float, default=0.35)
    parser.add_argument("--planar-projection-strength", type=float, default=1.50)
    parser.add_argument("--planar-project-iterations", type=int, default=3)
    parser.add_argument("--planar-project-min-weight", type=float, default=0.25)
    parser.add_argument("--planar-project-max-distance-factor", type=float, default=8.0)
    parser.add_argument("--detail-relax-iterations", type=int, default=4)
    parser.add_argument("--detail-relax-strength", type=float, default=0.45)
    parser.add_argument("--detail-relax-min-weight", type=float, default=0.18)
    parser.add_argument("--detail-relax-max-distance-factor", type=float, default=1.25)
    parser.add_argument("--flip-min-quality-improvement", type=float, default=0.002)
    parser.add_argument("--data-factor", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--max-buffer-width", type=int, default=1600)
    parser.add_argument("--near-plane", type=float, default=None)
    parser.add_argument("--edge-thickness", type=int, default=1)
    parser.add_argument("--target-edge-min-px", type=float, default=2.0)
    parser.add_argument("--target-edge-max-px", type=float, default=96.0)
    parser.add_argument("--high-gradient-quantile", type=float, default=0.85)
    parser.add_argument("--coverage-threshold", type=float, default=0.15)
    parser.add_argument("--support-view-k0", type=float, default=3.0)
    parser.add_argument("--min-offset-radius-px", type=float, default=2.0)
    parser.add_argument("--max-offset-radius-px", type=float, default=16.0)
    parser.add_argument("--view-buffers", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--proposal-support-threshold", type=float, default=0.08)
    parser.add_argument("--proposal-planar-threshold", type=float, default=0.35)
    parser.add_argument("--proposal-boundary-threshold", type=float, default=0.50)
    parser.add_argument("--proposal-detail-collapse-threshold", type=float, default=0.35)
    parser.add_argument("--proposal-missing-geometry-protect-threshold", type=float, default=0.70)
    parser.add_argument("--alpha-collapse", type=float, default=0.75)
    parser.add_argument("--alpha-planar-collapse", type=float, default=5.00)
    parser.add_argument("--alpha-split", type=float, default=1.50)
    parser.add_argument("--q-min-hard", type=float, default=0.03)
    parser.add_argument("--q-min-soft", type=float, default=0.12)
    parser.add_argument("--max-detailed-collapses", type=int, default=50000)
    parser.add_argument("--max-detailed-splits", type=int, default=50000)
    parser.add_argument("--max-detailed-flips", type=int, default=50000)
    parser.add_argument("--max-jsonl-rows-per-type", type=int, default=20000)
    parser.add_argument(
        "--enable-split-proposals",
        action="store_true",
        help="Enable accepted dry-run split proposals. Default keeps split pressure diagnostic for post-hoc simplification.",
    )
    parser.add_argument("--planar-area-length-factor", type=float, default=0.20)
    parser.add_argument("--planar-boundary-length-factor", type=float, default=6.0)
    parser.add_argument("--scene-max-length-factor", type=float, default=12.0)
    parser.add_argument("--proxy-min-length-factor", type=float, default=1.5)
    parser.add_argument("--proxy-confidence-floor", type=float, default=0.05)
    parser.add_argument("--lambda-proxy", type=float, default=1.0)
    parser.add_argument(
        "--area-epsilon",
        type=float,
        default=0.0,
        help="Area threshold in square world units. Default 0 removes only exactly zero-area triangles.",
    )
    parser.add_argument(
        "--skip-nonmanifold-vertex-split",
        action="store_true",
        help="Disable bow-tie vertex fan splitting.",
    )
    parser.add_argument("--no-debug-meshes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve_input_mesh(args: argparse.Namespace, model_dir: Path | None) -> Path:
    if args.mesh is None:
        if model_dir is None:
            raise SystemExit("--model-dir is required when --mesh is omitted.")
        if getattr(args, "stage", "") == "heal_preprocess":
            preclean = model_dir / "remesh" / "local" / "preclean_mesh.ply"
            if preclean.exists():
                return preclean
        if getattr(args, "stage", "") in {"evidence", "policy", "proposals", "remesh", "patch_remesh"}:
            preclean = model_dir / "remesh" / "local" / "preclean_mesh.ply"
            if preclean.exists():
                return preclean
        return resolve_latest_omega_mesh(model_dir, iteration=int(args.iteration))
    raw = Path(args.mesh).expanduser()
    candidates = [raw.resolve()] if raw.is_absolute() else [raw.resolve()]
    if not raw.is_absolute() and model_dir is not None:
        candidates.append((model_dir / raw).resolve())
    for resolved in candidates:
        if resolved.exists():
            return resolved
    raise SystemExit(f"Input mesh does not exist. Tried: {[str(candidate) for candidate in candidates]}")


def _optional_path(path: Path | None) -> Path | None:
    if path is None or str(path).strip() in {"", "."}:
        return None
    return path


def _resolve_existing_file(path: Path | None, default: Path, base: Path, description: str) -> Path:
    raw = _optional_path(path)
    resolved = default if raw is None else (raw.expanduser().resolve() if raw.expanduser().is_absolute() else (base / raw).resolve())
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    if resolved.is_dir():
        raise SystemExit(
            f"{description} points to a directory, not a file: {resolved}\n"
            "If this came from an environment variable, re-export it or omit the flag to use the default."
        )
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir is not None else None
    input_mesh = _resolve_input_mesh(args, model_dir)
    output_arg = _optional_path(args.output_dir)
    if output_arg is not None:
        output_dir = output_arg.expanduser().resolve()
    elif model_dir is not None:
        output_dir = model_dir / "remesh" / "local"
    else:
        output_dir = input_mesh.parent / "remesh" / "local"

    if args.stage == "preprocess":
        preprocess_mesh(
            MeshPreprocessConfig(
                input_mesh=input_mesh,
                output_dir=output_dir,
                output_name=str(args.output_name),
                summary_name=str(args.summary_name),
                diagnostics_name=str(args.diagnostics_name),
                area_epsilon=float(args.area_epsilon),
                split_nonmanifold_vertices=not bool(args.skip_nonmanifold_vertex_split),
                write_debug_meshes=not bool(args.no_debug_meshes),
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    if args.stage == "heal_preprocess":
        compute_mesh_healing(
            MeshHealingConfig(
                input_mesh=input_mesh,
                output_dir=output_dir,
                output_name=str(args.heal_output_name),
                summary_name=str(args.heal_summary_name),
                diagnostics_name=str(args.heal_diagnostics_name),
                cgal_binary=args.cgal_binary,
                repair_degenerate_faces=bool(args.heal_repair_degenerate_faces),
                repair_almost_degenerate_faces=bool(args.heal_repair_almost_degenerate_faces),
                duplicate_nonmanifold_vertices=bool(args.heal_duplicate_nonmanifold_vertices),
                stitch_borders=bool(args.heal_stitch_borders),
                fill_holes=bool(args.heal_fill_holes),
                max_hole_edges=int(args.heal_max_hole_edges),
                max_hole_diameter_factor=float(args.heal_max_hole_diameter_factor),
                component_area_factor=float(args.heal_component_area_factor),
                repair_self_intersections=bool(args.heal_repair_self_intersections),
                detect_self_intersections=bool(args.heal_detect_self_intersections),
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    if args.stage == "evidence":
        if model_dir is None:
            raise SystemExit("--model-dir is required for --stage evidence.")
        cfg_path = model_dir / "cfg.json"
        model_cfg = {}
        if cfg_path.exists():
            import json

            model_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        data_dir = resolve_dataset_dir(model_dir, args.data_dir)
        view_buffer_dir = (
            args.view_buffer_dir.expanduser().resolve()
            if args.view_buffer_dir is not None
            else model_dir / "remesh" / "view_buffers"
        )
        compute_mesh_face_evidence(
            MeshEvidenceConfig(
                mesh_path=input_mesh,
                data_dir=data_dir,
                out_dir=output_dir,
                view_buffer_dir=view_buffer_dir,
                evidence_name=str(args.evidence_name),
                data_factor=int(args.data_factor if args.data_factor is not None else model_cfg.get("data_factor", 1)),
                test_every=int(args.test_every if args.test_every is not None else model_cfg.get("test_every", 8)),
                frame_stride=int(args.frame_stride),
                max_frames=int(args.max_frames),
                max_buffer_width=int(args.max_buffer_width),
                near_plane=float(args.near_plane if args.near_plane is not None else model_cfg.get("near_plane", 0.2)),
                edge_thickness=int(args.edge_thickness),
                target_edge_min_px=float(args.target_edge_min_px),
                target_edge_max_px=float(args.target_edge_max_px),
                high_gradient_quantile=float(args.high_gradient_quantile),
                coverage_threshold=float(args.coverage_threshold),
                support_view_k0=float(args.support_view_k0),
                min_offset_radius_px=float(args.min_offset_radius_px),
                max_offset_radius_px=float(args.max_offset_radius_px),
                write_view_buffers=bool(args.view_buffers),
                write_debug_meshes=not bool(args.no_debug_meshes),
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    if args.stage == "policy":
        if model_dir is None:
            raise SystemExit("--model-dir is required for --stage policy.")
        evidence_npz = _resolve_existing_file(
            args.evidence_npz,
            model_dir / "remesh" / "local" / "evidence.npz",
            model_dir,
            "Evidence npz",
        )
        policy_result = compute_remesh_policy(
            RemeshPolicyConfig(
                mesh_path=input_mesh,
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
                mesh_path=input_mesh,
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
            mesh_path=input_mesh,
            policy_npz=policy_result.policy_npz,
            proxies_npz=proxy_result.proxies_npz,
            output_dir=output_dir,
            write_debug_meshes=not bool(args.no_debug_meshes),
        )
        return 0

    if args.stage == "proposals":
        if model_dir is None:
            raise SystemExit("--model-dir is required for --stage proposals.")
        policy_npz = _resolve_existing_file(
            args.policy_npz,
            model_dir / "remesh" / "local" / "policy.npz",
            model_dir,
            "Policy npz",
        )
        proxies_npz = _resolve_existing_file(
            args.proxies_npz,
            model_dir / "remesh" / "local" / "proxies.npz",
            model_dir,
            "Proxies npz",
        )
        proxies_json_arg = _optional_path(args.proxies_json)
        proxies_json = None
        if proxies_json_arg is not None:
            proxies_json = _resolve_existing_file(proxies_json_arg, model_dir / "remesh" / "local" / "proxies.json", model_dir, "Proxies json")
        elif (model_dir / "remesh" / "local" / "proxies.json").exists():
            proxies_json = model_dir / "remesh" / "local" / "proxies.json"
        compute_operation_proposals(
            OperationProposalConfig(
                mesh_path=input_mesh,
                policy_npz=policy_npz,
                proxies_npz=proxies_npz,
                proxies_json=proxies_json,
                output_dir=output_dir,
                proposals_name=str(args.proposals_name),
                support_threshold=float(args.proposal_support_threshold),
                planar_threshold=float(args.proposal_planar_threshold),
                boundary_threshold=float(args.proposal_boundary_threshold),
                detail_collapse_threshold=float(args.proposal_detail_collapse_threshold),
                missing_geometry_protect_threshold=float(args.proposal_missing_geometry_protect_threshold),
                alpha_collapse=float(args.alpha_collapse),
                alpha_planar_collapse=float(args.alpha_planar_collapse),
                alpha_split=float(args.alpha_split),
                q_min_hard=float(args.q_min_hard),
                q_min_soft=float(args.q_min_soft),
                max_detailed_collapses=int(args.max_detailed_collapses),
                max_detailed_splits=int(args.max_detailed_splits),
                max_detailed_flips=int(args.max_detailed_flips),
                max_jsonl_rows_per_type=int(args.max_jsonl_rows_per_type),
                enable_split_proposals=bool(args.enable_split_proposals),
                planar_area_length_factor=float(args.planar_area_length_factor),
                planar_boundary_length_factor=float(args.planar_boundary_length_factor),
                scene_max_length_factor=float(args.scene_max_length_factor),
                proxy_min_length_factor=float(args.proxy_min_length_factor),
                proxy_confidence_floor=float(args.proxy_confidence_floor),
                lambda_proxy=float(args.lambda_proxy),
                write_debug_meshes=not bool(args.no_debug_meshes),
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    if args.stage == "remesh":
        if model_dir is None:
            raise SystemExit("--model-dir is required for --stage remesh.")
        policy_npz = _resolve_existing_file(
            args.policy_npz,
            model_dir / "remesh" / "local" / "policy.npz",
            model_dir,
            "Policy npz",
        )
        proxies_npz = _resolve_existing_file(
            args.proxies_npz,
            model_dir / "remesh" / "local" / "proxies.npz",
            model_dir,
            "Proxies npz",
        )
        proxies_json_arg = _optional_path(args.proxies_json)
        proxies_json = None
        if proxies_json_arg is not None:
            proxies_json = _resolve_existing_file(proxies_json_arg, model_dir / "remesh" / "local" / "proxies.json", model_dir, "Proxies json")
        elif (model_dir / "remesh" / "local" / "proxies.json").exists():
            proxies_json = model_dir / "remesh" / "local" / "proxies.json"
        compute_weighted_remesh(
            WeightedRemeshConfig(
                mesh_path=input_mesh,
                policy_npz=policy_npz,
                proxies_npz=proxies_npz,
                proxies_json=proxies_json,
                output_dir=output_dir,
                output_name=str(args.remesh_output_name),
                diagnostics_name=str(args.remesh_diagnostics_name),
                operations_name=str(args.remesh_operations_name),
                max_passes=int(args.max_remesh_passes),
                max_collapses_per_pass=int(args.max_collapses_per_pass),
                max_flips_per_pass=int(args.max_flips_per_pass),
                max_jsonl_rows=int(args.max_jsonl_rows_per_type),
                collapse_threshold=float(args.collapse_threshold),
                support_threshold=float(args.proposal_support_threshold),
                boundary_threshold=float(args.proposal_boundary_threshold),
                uncertainty_threshold=float(args.uncertainty_threshold),
                alpha_collapse=float(args.alpha_collapse),
                alpha_planar_collapse=float(args.alpha_planar_collapse),
                alpha_planar_transition_collapse=float(args.alpha_planar_transition_collapse),
                alpha_feature_edge_collapse=float(args.alpha_feature_edge_collapse),
                alpha_detail_collapse=float(args.alpha_detail_collapse),
                detail_mesh_target_factor=float(args.detail_mesh_target_factor),
                detail_simplify_min_weight=float(args.detail_simplify_min_weight),
                q_min_hard=float(args.q_min_hard),
                q_min_soft=float(args.q_min_soft),
                max_normal_error=float(args.max_normal_error),
                max_quality_error=float(args.max_quality_error),
                max_projection_distance_factor=float(args.max_projection_distance_factor),
                plane_residual_factor=float(args.plane_residual_factor),
                pre_project_before_collapse=bool(args.pre_project_before_collapse),
                feature_edge_collapse=bool(args.feature_edge_collapse),
                feature_edge_min_planar=float(args.feature_edge_min_planar),
                feature_edge_max_uncertain=float(args.feature_edge_max_uncertain),
                planar_boundary_softening=float(args.planar_boundary_softening),
                planar_transition_min_weight=float(args.planar_transition_min_weight),
                planar_projection_strength=float(args.planar_projection_strength),
                planar_project_iterations=int(args.planar_project_iterations),
                planar_project_min_weight=float(args.planar_project_min_weight),
                planar_project_max_distance_factor=float(args.planar_project_max_distance_factor),
                detail_relax_iterations=int(args.detail_relax_iterations),
                detail_relax_strength=float(args.detail_relax_strength),
                detail_relax_min_weight=float(args.detail_relax_min_weight),
                detail_relax_max_distance_factor=float(args.detail_relax_max_distance_factor),
                flip_min_quality_improvement=float(args.flip_min_quality_improvement),
                planar_area_length_factor=float(args.planar_area_length_factor),
                planar_boundary_length_factor=float(args.planar_boundary_length_factor),
                scene_max_length_factor=float(args.scene_max_length_factor),
                proxy_min_length_factor=float(args.proxy_min_length_factor),
                proxy_confidence_floor=float(args.proxy_confidence_floor),
                lambda_proxy=float(args.lambda_proxy),
                write_debug_meshes=not bool(args.no_debug_meshes),
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    if args.stage == "patch_remesh":
        if model_dir is None:
            raise SystemExit("--model-dir is required for --stage patch_remesh.")
        policy_npz = _resolve_existing_file(
            args.policy_npz,
            model_dir / "remesh" / "local" / "policy.npz",
            model_dir,
            "Policy npz",
        )
        proxies_npz = _resolve_existing_file(
            args.proxies_npz,
            model_dir / "remesh" / "local" / "proxies.npz",
            model_dir,
            "Proxies npz",
        )
        compute_planar_patch_remesh(
            PlanarPatchRemeshConfig(
                mesh_path=input_mesh,
                policy_npz=policy_npz,
                proxies_npz=proxies_npz,
                output_dir=output_dir,
                output_name=str(args.patch_output_name),
                diagnostics_name=str(args.patch_diagnostics_name),
                operations_name=str(args.patch_operations_name),
                planar_threshold=float(args.patch_planar_threshold),
                feature_threshold=float(args.patch_feature_threshold),
                boundary_stop_threshold=float(args.patch_boundary_stop_threshold),
                min_patch_faces=int(args.patch_min_faces),
                patch_normal_degrees=float(args.patch_normal_degrees),
                patch_distance_factor=float(args.patch_distance_factor),
                patch_trim_distance_factor=float(args.patch_trim_distance_factor),
                planar_target_factor=float(args.patch_planar_target_factor),
                max_passes=int(args.patch_max_passes),
                max_collapses_per_pass=int(args.patch_max_collapses_per_pass),
                collapse_threshold=float(args.patch_collapse_threshold),
                q_min_hard=float(args.q_min_hard),
                q_min_soft=float(args.q_min_soft),
                max_quality_error=float(args.patch_max_quality_error),
                max_jsonl_rows=int(args.max_jsonl_rows_per_type),
                write_debug_meshes=not bool(args.no_debug_meshes),
                overwrite=bool(args.overwrite),
            )
        )
        return 0

    raise SystemExit(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
