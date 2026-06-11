"""Optional Phase 1 mesh healing for OMeGa local remeshing.

The default Phase 1 preclean pass in :mod:`mesh_clean` is intentionally minimal.
This module adds a separate, opt-in healing stage for experiments that need
small-hole filling, exact border stitching, non-manifold vertex duplication, and
self-intersection diagnostics or repair.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import trimesh

try:
    from omega_local.remesh.mesh_clean import load_mesh_arrays, mesh_stats, resolve_latest_omega_mesh
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from omega_local.remesh.mesh_clean import load_mesh_arrays, mesh_stats, resolve_latest_omega_mesh


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class MeshHealingConfig:
    input_mesh: Path
    output_dir: Path
    output_name: str = "preclean_healed_mesh.ply"
    summary_name: str = "preclean_healed_summary.json"
    diagnostics_name: str = "preclean_healed_diagnostics.npz"
    cgal_binary: Path | None = None
    repair_degenerate_faces: bool = True
    repair_almost_degenerate_faces: bool = False
    duplicate_nonmanifold_vertices: bool = True
    stitch_borders: bool = True
    fill_holes: bool = True
    max_hole_edges: int = 48
    max_hole_diameter_factor: float = 6.5
    component_area_factor: float = 0.0
    repair_self_intersections: bool = False
    detect_self_intersections: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class MeshHealingResult:
    input_mesh: Path
    output_mesh: Path
    summary_json: Path
    diagnostics_npz: Path
    input_stats: dict[str, Any]
    output_stats: dict[str, Any]
    summary: dict[str, Any]


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_cgal_binary() -> Path:
    return _repo_root() / "tools" / "cgal_mesh_heal"


def _build_command(
    config: MeshHealingConfig,
    cgal_input_mesh: Path,
    cgal_output_mesh: Path,
    cgal_summary: Path,
    filled_patch_mesh: Path,
    hole_report_jsonl: Path,
    max_hole_diameter: float,
    component_area_threshold: float,
) -> list[str]:
    binary = config.cgal_binary.expanduser().resolve() if config.cgal_binary is not None else _default_cgal_binary()
    if not binary.exists():
        raise FileNotFoundError(
            "CGAL mesh healing helper was not found:\n"
            f"  {binary}\n\n"
            "Build it once with:\n"
            f"  bash {_repo_root() / 'scripts' / 'build_cgal_mesh_heal.sh'}"
        )

    cmd = [
        str(binary),
        str(cgal_input_mesh),
        str(cgal_output_mesh),
        str(cgal_summary),
        "--filled-patch-mesh",
        str(filled_patch_mesh),
        "--hole-report-jsonl",
        str(hole_report_jsonl),
        "--max-hole-edges",
        str(int(config.max_hole_edges)),
        "--max-hole-diameter",
        f"{float(max_hole_diameter):.17g}",
        "--component-area-threshold",
        f"{float(component_area_threshold):.17g}",
    ]
    if not config.repair_degenerate_faces:
        cmd.append("--no-repair-degenerate-faces")
    if config.repair_almost_degenerate_faces:
        cmd.append("--repair-almost-degenerate-faces")
    if not config.duplicate_nonmanifold_vertices:
        cmd.append("--no-duplicate-nonmanifold-vertices")
    if not config.stitch_borders:
        cmd.append("--no-stitch-borders")
    if not config.fill_holes:
        cmd.append("--no-fill-holes")
    if config.repair_self_intersections:
        cmd.append("--repair-self-intersections")
    if not config.detect_self_intersections:
        cmd.append("--no-detect-self-intersections")
    return cmd


def _export_triangle_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    mesh.export(path)


def _write_compatible_diagnostics(
    diagnostics_npz: Path,
    *,
    input_face_count: int,
    output_face_count: int,
    output_vertex_count: int,
) -> None:
    diagnostics_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        diagnostics_npz,
        original_face_status=np.zeros((int(input_face_count),), dtype=np.int16),
        preclean_face_changed_by_split=np.zeros((int(output_face_count),), dtype=bool),
        preclean_face_source=np.full((int(output_face_count),), -1, dtype=np.int64),
        preclean_vertex_source=np.full((int(output_vertex_count),), -1, dtype=np.int64),
        preclean_vertex_created_by_split=np.zeros((int(output_vertex_count),), dtype=bool),
        splitExistingVertices=np.zeros((0,), dtype=np.int64),
        splitSourceVertices=np.zeros((0,), dtype=np.int64),
        splitFanCounts=np.zeros((0,), dtype=np.int64),
        splitIncidentFaceCounts=np.zeros((0,), dtype=np.int64),
        splitCreatedVertices=np.zeros((0,), dtype=np.int64),
        splitCreatedVertexSources=np.zeros((0,), dtype=np.int64),
        healingOutputHasFaceCorrespondence=np.asarray([False], dtype=bool),
    )


def _healing_sidecar_paths(output_dir: Path, output_name: str, summary_name: str) -> tuple[Path, Path, Path]:
    output_stem = Path(output_name).stem
    summary_stem = Path(summary_name).stem
    if output_stem == "preclean_healed_mesh" and summary_stem == "preclean_healed_summary":
        return (
            output_dir / "cgal_mesh_heal_summary.json",
            output_dir / "preclean_healed_holes.jsonl",
            output_dir / "debug_meshes" / "healing" / "preclean_healed_filled_patches.ply",
        )
    return (
        output_dir / f"{summary_stem}_cgal.json",
        output_dir / f"{output_stem}_holes.jsonl",
        output_dir / "debug_meshes" / "healing" / f"{output_stem}_filled_patches.ply",
    )


def compute_mesh_healing(config: MeshHealingConfig, progress: ProgressFn = _progress_default) -> MeshHealingResult:
    input_mesh = config.input_mesh.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_mesh = output_dir / config.output_name
    summary_json = output_dir / config.summary_name
    diagnostics_npz = output_dir / config.diagnostics_name
    cgal_summary_json, hole_report_jsonl, filled_patch_mesh = _healing_sidecar_paths(
        output_dir,
        str(config.output_name),
        str(config.summary_name),
    )
    debug_dir = filled_patch_mesh.parent

    for path, label in ((output_mesh, "Output mesh"), (summary_json, "Summary"), (diagnostics_npz, "Diagnostics")):
        if path.exists() and not config.overwrite:
            raise FileExistsError(f"{label} already exists: {path}. Re-run with --overwrite.")

    progress(f"[heal] Loading input mesh: {input_mesh}")
    input_vertices, input_faces = load_mesh_arrays(input_mesh)
    input_stats_obj = mesh_stats(input_vertices, input_faces)
    input_stats = asdict(input_stats_obj)
    median_edge = float(input_stats_obj.edge_length_quantiles[2]) if input_stats_obj.edge_length_quantiles else 0.0
    max_hole_diameter = max(0.0, float(config.max_hole_diameter_factor)) * max(median_edge, 0.0)
    component_area_threshold = max(0.0, float(config.component_area_factor)) * max(float(input_stats_obj.surface_area), 0.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = output_dir / ".cgal_mesh_heal_tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    cgal_input_mesh = temp_dir / "input.off"
    cgal_output_mesh = temp_dir / "healed.off"
    progress(f"[heal] Exporting CGAL-compatible OFF input: {cgal_input_mesh}")
    _export_triangle_mesh(cgal_input_mesh, input_vertices, input_faces)

    debug_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(
        config,
        cgal_input_mesh,
        cgal_output_mesh,
        cgal_summary_json,
        filled_patch_mesh,
        hole_report_jsonl,
        max_hole_diameter,
        component_area_threshold,
    )
    progress("[heal] Running CGAL Polygon Mesh Processing backend")
    progress("[heal] " + " ".join(cmd))
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "CGAL mesh healing failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    if not cgal_output_mesh.exists():
        raise FileNotFoundError(f"CGAL backend finished but did not write output mesh: {cgal_output_mesh}")
    if not cgal_summary_json.exists():
        raise FileNotFoundError(f"CGAL backend finished but did not write summary: {cgal_summary_json}")

    progress(f"[heal] Loading healed CGAL output: {cgal_output_mesh}")
    output_vertices, output_faces = load_mesh_arrays(cgal_output_mesh)
    progress(f"[heal] Writing healed mesh: {output_mesh}")
    _export_triangle_mesh(output_mesh, output_vertices, output_faces)
    output_stats = asdict(mesh_stats(output_vertices, output_faces))
    cgal_summary = _read_json(cgal_summary_json)

    progress(f"[heal] Writing visualization-compatible diagnostics: {diagnostics_npz}")
    _write_compatible_diagnostics(
        diagnostics_npz,
        input_face_count=len(input_faces),
        output_face_count=len(output_faces),
        output_vertex_count=len(output_vertices),
    )

    summary = {
        "stageName": "omega_local.remesh.mesh_heal.compute_mesh_healing",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "inputMesh": str(input_mesh),
        "outputMesh": str(output_mesh),
        "backend": "cgal_polygon_mesh_processing",
        "operationOrder": [
            "repair degenerate faces",
            "optionally repair almost-degenerate needle/cap triangles",
            "duplicate non-manifold vertices",
            "stitch exactly matching border halfedges",
            "fill selected small boundary cycles",
            "optionally remove tiny connected components",
            "optionally repair self-intersections",
        ],
        "conservativeDefaults": [
            "hole filling is limited by boundary edge count and diameter",
            "tiny connected component removal is disabled unless component_area_factor > 0",
            "almost-degenerate triangle repair is disabled by default",
            "self-intersection repair is disabled by default",
        ],
        "config": {
            "repairDegenerateFaces": bool(config.repair_degenerate_faces),
            "repairAlmostDegenerateFaces": bool(config.repair_almost_degenerate_faces),
            "duplicateNonmanifoldVertices": bool(config.duplicate_nonmanifold_vertices),
            "stitchBorders": bool(config.stitch_borders),
            "fillHoles": bool(config.fill_holes),
            "maxHoleEdges": int(config.max_hole_edges),
            "maxHoleDiameterFactor": float(config.max_hole_diameter_factor),
            "resolvedMaxHoleDiameter": float(max_hole_diameter),
            "componentAreaFactor": float(config.component_area_factor),
            "resolvedComponentAreaThreshold": float(component_area_threshold),
            "repairSelfIntersections": bool(config.repair_self_intersections),
            "detectSelfIntersections": bool(config.detect_self_intersections),
            "cgalBinary": str((config.cgal_binary or _default_cgal_binary()).expanduser().resolve()),
            "cgalInputFormat": "off",
            "cgalTemporaryDirectory": str(temp_dir),
        },
        "inputStats": input_stats,
        "outputStats": output_stats,
        "cgalSummary": cgal_summary,
        "outputs": {
            "healedMesh": str(output_mesh),
            "summaryJson": str(summary_json),
            "diagnosticsNpz": str(diagnostics_npz),
            "cgalSummaryJson": str(cgal_summary_json),
            "holeReportJsonl": str(hole_report_jsonl),
            "filledPatchMesh": str(filled_patch_mesh) if filled_patch_mesh.exists() else "",
        },
    }
    progress(f"[heal] Writing summary: {summary_json}")
    _write_json(summary_json, summary)
    progress(
        "[heal] Done: "
        f"V {input_stats['vertices']}->{output_stats['vertices']}, "
        f"F {input_stats['faces']}->{output_stats['faces']}, "
        f"boundary edges {input_stats['boundary_edges']}->{output_stats['boundary_edges']}, "
        f"components {input_stats['connected_face_components']}->{output_stats['connected_face_components']}"
    )
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    return MeshHealingResult(
        input_mesh=input_mesh,
        output_mesh=output_mesh,
        summary_json=summary_json,
        diagnostics_npz=diagnostics_npz,
        input_stats=input_stats,
        output_stats=output_stats,
        summary=summary,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run optional CGAL-backed healing on an OMeGa preclean mesh.")
    parser.add_argument("--model-dir", type=Path, default=None, help="OMeGa model/result directory.")
    parser.add_argument("--mesh", "--input-mesh", dest="mesh", type=Path, default=None, help="Mesh to heal.")
    parser.add_argument("--iteration", type=int, default=-1, help="Mesh iteration to use when --mesh is omitted.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/local.")
    parser.add_argument("--output-name", default="preclean_healed_mesh.ply")
    parser.add_argument("--summary-name", default="preclean_healed_summary.json")
    parser.add_argument("--diagnostics-name", default="preclean_healed_diagnostics.npz")
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
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.model_dir is None and args.mesh is None:
        raise SystemExit("Pass --model-dir, --mesh, or both.")
    model_dir = args.model_dir.expanduser().resolve() if args.model_dir is not None else None
    if args.mesh is None:
        if model_dir is None:
            raise SystemExit("--model-dir is required when --mesh is omitted.")
        default_preclean = model_dir / "remesh" / "local" / "preclean_mesh.ply"
        input_mesh = default_preclean if default_preclean.exists() else resolve_latest_omega_mesh(model_dir, iteration=int(args.iteration))
    else:
        raw = args.mesh.expanduser()
        input_mesh = raw.resolve() if raw.is_absolute() else ((model_dir / raw).resolve() if model_dir is not None else raw.resolve())
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif model_dir is not None:
        output_dir = model_dir / "remesh" / "local"
    else:
        output_dir = input_mesh.parent / "remesh" / "local"

    compute_mesh_healing(
        MeshHealingConfig(
            input_mesh=input_mesh,
            output_dir=output_dir,
            output_name=str(args.output_name),
            summary_name=str(args.summary_name),
            diagnostics_name=str(args.diagnostics_name),
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


if __name__ == "__main__":
    raise SystemExit(main())
