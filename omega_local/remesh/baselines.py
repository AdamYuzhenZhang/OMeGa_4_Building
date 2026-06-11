"""Comparison baselines for OMeGa remeshing experiments.

These baselines are intentionally separate from the view-informed remesher.
They answer a simpler question: how far do standard mesh-only methods get on
the same OMeGa mesh before we add our custom evidence and plane/crease logic?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import open3d as o3d
import pymeshlab
import trimesh

from omega_local.remesh.instant_field import OmegaInstantFieldConfig, compute_omega_instant_field
from omega_local.remesh.geogram_density import OmegaGeogramDensityConfig, compute_omega_geogram_density
from omega_local.remesh.local_ops import face_geometry, load_mesh_arrays, triangle_quality_from_vertices
from omega_local.remesh.mesh_clean import resolve_latest_omega_mesh
from omega_local.remesh.policy import build_mesh_topology


ProgressFn = Callable[[str], None]


SUPPORTED_METHODS = (
    "open3d_qem",
    "meshlab_qem",
    "meshlab_planar_qem",
    "meshlab_isotropic",
    "meshlab_feature_isotropic",
    "meshlab_isotropic_qem",
    "meshlab_feature_isotropic_qem",
    "instant_meshes_quad",
    "instant_meshes_tri",
    "instant_meshes_dominant",
    "instant_meshes_quad_guided",
    "instant_meshes_tri_guided",
    "instant_meshes_dominant_guided",
    "geogram_anisotropic",
    "geogram_anisotropic_gradation",
    "geogram_omega_density",
    "geogram_omega_density_gradation",
)

METHOD_PRESETS = {
    "qem": (
        "open3d_qem",
        "meshlab_qem",
        "meshlab_planar_qem",
    ),
    "isotropic": (
        "meshlab_isotropic",
        "meshlab_feature_isotropic",
        "meshlab_isotropic_qem",
        "meshlab_feature_isotropic_qem",
    ),
    "instant_original": (
        "instant_meshes_quad",
        "instant_meshes_tri",
        "instant_meshes_dominant",
    ),
    "geogram": (
        "geogram_anisotropic",
        "geogram_anisotropic_gradation",
        "geogram_omega_density",
        "geogram_omega_density_gradation",
    ),
    "key": (
        "open3d_qem",
        "meshlab_qem",
        "meshlab_planar_qem",
        "meshlab_isotropic_qem",
        "meshlab_feature_isotropic_qem",
        "instant_meshes_quad",
        "instant_meshes_tri",
        "instant_meshes_dominant",
        "geogram_anisotropic",
        "geogram_anisotropic_gradation",
    ),
    "geogram_guided": (
        "geogram_anisotropic",
        "geogram_anisotropic_gradation",
        "geogram_omega_density",
        "geogram_omega_density_gradation",
    ),
    "geogram_auto": (
        "geogram_omega_density",
        "geogram_omega_density_gradation",
    ),
    "instant_guided": (
        "instant_meshes_quad",
        "instant_meshes_quad_guided",
        "instant_meshes_tri",
        "instant_meshes_tri_guided",
        "instant_meshes_dominant",
        "instant_meshes_dominant_guided",
    ),
    "all": SUPPORTED_METHODS,
}

PRESET_EXPERIMENT_NAMES = {
    "qem": "qem_compare",
    "isotropic": "isotropic_compare",
    "instant_original": "instant_original_compare",
    "geogram": "geogram_compare",
    "key": "key_compare",
    "geogram_guided": "geogram_omega_density_compare",
    "geogram_auto": "geogram_auto_density_compare",
    "instant_guided": "instant_guided_compare",
    "all": "all_methods_compare",
}


@dataclass(frozen=True)
class BaselineConfig:
    model_dir: Path
    mesh_path: Path | None = None
    output_dir: Path | None = None
    experiment_name: str = ""
    methods: tuple[str, ...] = METHOD_PRESETS["key"]
    target_face_ratios: tuple[float, ...] = (0.20, 0.40, 0.60)
    target_faces: tuple[int, ...] = ()
    iteration: int = -1
    isotropic_iterations: int = 5
    isotropic_feature_degrees: float = 35.0
    isotropic_target_scale: float = 1.0
    isotropic_target_edge_length: float = 0.0
    isotropic_target_percent: float = 0.0
    isotropic_max_surface_dist_percent: float = 1.0
    feature_isotropic_degrees: float = 20.0
    feature_isotropic_smooth: bool = False
    instant_meshes_bin: Path | None = None
    instant_meshes_crease_degrees: float = 30.0
    instant_meshes_smooth_iterations: int = 2
    instant_meshes_threads: int = 0
    instant_meshes_deterministic: bool = True
    instant_meshes_align_boundaries: bool = True
    instant_meshes_dump_fields: bool = True
    instant_meshes_feature_constraints: bool = False
    instant_meshes_feature_constraint_weight: float = 0.9
    instant_meshes_feature_constraint_angle: float = 0.0
    instant_guided_feature_constraints: bool = False
    instant_field_path: Path | None = None
    instant_field_weight_scale: float = 0.7
    instant_field_frame_stride: int = 1
    instant_field_max_frames: int = 0
    instant_field_pixel_stride: int = 2
    instant_field_min_direction_confidence: float = 0.15
    instant_field_min_gradient_normalized: float = 0.35
    instant_field_gradient_mode: str = "point"
    instant_field_weight_quantile: float = 0.90
    instant_field_min_vertex_weight: float = 0.02
    instant_guided_no_subdivide: bool = False
    geogram_bin: Path | None = None
    geogram_anisotropy: float = 1.0
    geogram_gradation: float = 1.0
    geogram_normal_smooth_iterations: int = 3
    geogram_lfs_samples: int = 10000
    geogram_lloyd_iterations: int = 5
    geogram_newton_iterations: int = 30
    geogram_newton_m: int = 7
    geogram_threads: int = 0
    geogram_preserve_components: bool = True
    geogram_preserve_holes: bool = True
    geogram_postprocess_degree3: bool = False
    geogram_density_path: Path | None = None
    geogram_density_min: float = 0.25
    geogram_density_max: float = 4.0
    geogram_density_signal_gamma: float = 1.25
    geogram_density_detail_influence: float = 1.0
    geogram_density_boundary_influence: float = 0.5
    geogram_density_uncertain_influence: float = 0.0
    geogram_auto_from_normal_error_degrees: float = 0.0
    geogram_auto_target_points: bool = False
    geogram_auto_h_min_factor: float = 0.50
    geogram_auto_h_max_factor: float = 6.0
    geogram_auto_target_point_scale: float = 1.0
    geogram_use_fused_normals: bool = True
    geogram_fused_normal_blend: float = 0.70
    geogram_fused_normal_min_weight: float = 8.0
    geogram_density_runtime_scale: float = 1.0
    geogram_density_runtime_min: float = 1e-6
    geogram_density_runtime_max: float = 0.0
    geogram_density_planar_gradation_iterations: int = 12
    geogram_density_planar_gradation_strength: float = 0.45
    geogram_density_planar_gradation_planar_power: float = 2.0
    geogram_density_planar_gradation_min_edge_weight: float = 0.05
    geogram_density_planar_gradation_max_density_ratio: float = 1.75
    geogram_density_planar_gradation_normal_lift_scale: float = 0.0
    geogram_density_planar_gradation_metric_tau_factor: float = 2.0
    geogram_density_planar_gradation_min_normal_gate: float = 0.05
    geogram_density_planar_gradation_use_proxy_boundaries: bool = True
    qem_quality_threshold: float = 0.3
    qem_boundary_weight: float = 1.0
    planar_quadric_weight: float = 0.01
    preserve_boundary: bool = True
    preserve_topology: bool = True
    preserve_normal: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class BaselineResult:
    summary_json: Path
    manifest_jsonl: Path
    output_dir: Path
    method_count: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh object from {path}: {type(mesh)!r}")
    return mesh


def _mesh_stats(path: Path) -> dict[str, Any]:
    vertices, faces = load_mesh_arrays(path)
    topology = build_mesh_topology(faces)
    _centers, _normals, areas, mean_edge = face_geometry(vertices, faces)
    quality = triangle_quality_from_vertices(vertices[faces])
    edge_lengths = np.linalg.norm(
        vertices[topology.edge_vertices[:, 1]] - vertices[topology.edge_vertices[:, 0]],
        axis=1,
    )
    bbox = np.ptp(vertices, axis=0)
    return {
        "mesh": str(path),
        "vertices": int(vertices.shape[0]),
        "faces": int(faces.shape[0]),
        "edges": int(topology.edge_vertices.shape[0]),
        "boundaryEdges": int(np.count_nonzero(topology.boundary_edges)),
        "nonmanifoldEdges": int(topology.nonmanifold_edge_count),
        "bbox": [float(v) for v in bbox],
        "bboxDiag": float(np.linalg.norm(bbox)),
        "area": float(np.sum(areas)),
        "meanEdge": _summary(mean_edge),
        "edgeLength": _summary(edge_lengths),
        "triangleQuality": _summary(quality),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0, "min": 0.0, "median": 0.0, "mean": 0.0, "q90": 0.0, "max": 0.0}
    q = np.quantile(arr, [0.5, 0.9])
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "median": float(q[0]),
        "mean": float(np.mean(arr)),
        "q90": float(q[1]),
        "max": float(np.max(arr)),
    }


INSTANT_MESHES_METHODS = {
    "instant_meshes_quad",
    "instant_meshes_tri",
    "instant_meshes_dominant",
    "instant_meshes_quad_guided",
    "instant_meshes_tri_guided",
    "instant_meshes_dominant_guided",
}
GUIDED_INSTANT_MESHES_METHODS = {
    "instant_meshes_quad_guided",
    "instant_meshes_tri_guided",
    "instant_meshes_dominant_guided",
}
GEOGRAM_OMEGA_DENSITY_METHODS = {
    "geogram_omega_density",
    "geogram_omega_density_gradation",
}


def _needs_face_target(method: str) -> bool:
    return method not in {"copy_source", "meshlab_isotropic", "meshlab_feature_isotropic"}


def _target_faces(config: BaselineConfig, input_faces: int, methods: tuple[str, ...]) -> list[int]:
    if bool(config.geogram_auto_target_points):
        method_set = set(methods)
        if method_set and method_set.issubset(GEOGRAM_OMEGA_DENSITY_METHODS):
            return [0]
        raise ValueError("--geogram-auto-target-points currently requires only geogram_omega_density methods.")
    targets = [int(v) for v in config.target_faces if int(v) > 0]
    for ratio in config.target_face_ratios:
        if ratio <= 0.0:
            continue
        if ratio >= 1.0:
            targets.append(int(round(ratio)))
        else:
            targets.append(int(round(float(input_faces) * float(ratio))))
    clean = sorted({max(4, min(int(input_faces), int(v))) for v in targets})
    if not clean:
        explicit_isotropic = float(config.isotropic_target_percent) > 0.0 or float(config.isotropic_target_edge_length) > 0.0
        if explicit_isotropic and not any(_needs_face_target(method) for method in methods):
            return [0]
        if any(_needs_face_target(method) for method in methods):
            raise ValueError(
                "No target face count was provided, but one or more selected methods require it. "
                "Use --target-face-ratios, --target-faces, or select only isotropic-only methods with an explicit target edge length/percent."
            )
        clean = [max(4, int(round(0.20 * float(input_faces))))]
    return clean


def _float_token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _slug_token(value: str) -> str:
    out = []
    last_underscore = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last_underscore = False
        elif not last_underscore:
            out.append("_")
            last_underscore = True
    slug = "".join(out).strip("_")
    return slug or "baseline_compare"


def _experiment_name_from_methods(methods: tuple[str, ...]) -> str:
    method_tuple = tuple(dict.fromkeys(str(method) for method in methods if str(method).strip()))
    for preset, preset_methods in METHOD_PRESETS.items():
        if method_tuple == tuple(preset_methods):
            return PRESET_EXPERIMENT_NAMES.get(preset, f"{preset}_compare")
    prefix = _slug_token("_".join(method_tuple[:4]))
    if len(method_tuple) > 4:
        digest = hashlib.sha1(",".join(method_tuple).encode("utf-8")).hexdigest()[:8]
        prefix = f"{prefix}_plus{len(method_tuple) - 4}_{digest}"
    return f"{prefix}_compare"


def _resolve_baseline_output_dir(config: BaselineConfig, model_dir: Path, methods: tuple[str, ...]) -> Path:
    if config.output_dir is None:
        experiment_name = _slug_token(config.experiment_name) if config.experiment_name else _experiment_name_from_methods(methods)
        return model_dir / "remesh" / "baselines" / experiment_name

    output_dir = config.output_dir.expanduser().resolve()
    workspace_root = Path(__file__).resolve().parents[4]
    forbidden = {
        workspace_root.resolve(),
        model_dir.resolve(),
        (model_dir / "remesh").resolve(),
    }
    if output_dir.resolve() in forbidden:
        raise ValueError(
            "Refusing to write baseline outputs into a broad project/model directory: "
            f"{output_dir}. Use --experiment-name or pass an explicit experiment folder such as "
            "<model-dir>/remesh/baselines/my_compare."
        )
    return output_dir


def _safe_name(method: str, target: int, config: BaselineConfig) -> str:
    if int(target) <= 0 and method == "meshlab_isotropic":
        if float(config.isotropic_target_percent) > 0.0:
            return f"{method}_pct{_float_token(float(config.isotropic_target_percent))}"
        if float(config.isotropic_target_edge_length) > 0.0:
            return f"{method}_L{_float_token(float(config.isotropic_target_edge_length))}"
    if int(target) <= 0 and method == "meshlab_feature_isotropic":
        if float(config.isotropic_target_percent) > 0.0:
            return f"{method}_pct{_float_token(float(config.isotropic_target_percent))}"
        if float(config.isotropic_target_edge_length) > 0.0:
            return f"{method}_L{_float_token(float(config.isotropic_target_edge_length))}"
    return f"{method}_f{int(target):08d}"


def _copy_input_mesh(input_mesh: Path, output_mesh: Path) -> None:
    mesh = _load_trimesh(input_mesh)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)


def _resolve_instant_meshes_bin(config: BaselineConfig) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if config.instant_meshes_bin is not None:
        candidates.append(config.instant_meshes_bin)
    env_value = os.environ.get("INSTANT_MESHES_BIN", "").strip()
    if env_value:
        candidates.append(env_value)
    third_party_dir = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            third_party_dir / "instant-meshes" / "build-codex" / "InstantMeshes",
            third_party_dir / "instant-meshes" / "build-codex" / "Instant Meshes",
            third_party_dir / "instant-meshes" / "build" / "InstantMeshes",
            third_party_dir / "instant-meshes" / "build" / "Instant Meshes",
        ]
    )
    for name in ("Instant Meshes", "InstantMeshes", "instant-meshes", "instant_meshes"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for raw in candidates:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Instant Meshes binary not found. Pass --instant-meshes-bin or set INSTANT_MESHES_BIN "
        "to the built/precompiled 'Instant Meshes' executable."
    )


def _resolve_geogram_bin(config: BaselineConfig) -> Path:
    candidates: list[str | os.PathLike[str]] = []
    if config.geogram_bin is not None:
        candidates.append(config.geogram_bin)
    env_value = os.environ.get("GEOGRAM_VORPALITE_BIN", "").strip()
    if env_value:
        candidates.append(env_value)
    third_party_dir = Path(__file__).resolve().parents[3]
    candidates.extend(
        [
            third_party_dir / "geogram" / "build-codex-nogfx" / "bin" / "vorpalite",
            third_party_dir / "geogram" / "build" / "Linux64-gcc-dynamic-Release" / "bin" / "vorpalite",
            third_party_dir / "geogram" / "build" / "Linux64-gcc-dynamic-Debug" / "bin" / "vorpalite",
        ]
    )
    found = shutil.which("vorpalite")
    if found:
        candidates.append(found)
    for raw in candidates:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = path.resolve()
        if path.exists() and path.is_file():
            return path
    raise FileNotFoundError(
        "Geogram vorpalite binary not found. Build third_party/geogram with "
        "`cmake -S third_party/geogram -B third_party/geogram/build-codex-nogfx "
        "-DCMAKE_BUILD_TYPE=Release -DVORPALINE_PLATFORM=Linux64-gcc-dynamic "
        "-DGEOGRAM_WITH_GRAPHICS=OFF -DGEOGRAM_WITH_EXPLORAGRAM=OFF -DGEOGRAM_WITH_LUA=OFF` "
        "then `cmake --build third_party/geogram/build-codex-nogfx -j`, or pass --geogram-bin."
    )


def _obj_face_stats(path: Path) -> dict[str, Any]:
    if path.suffix.lower() != ".obj" or not path.exists():
        return {}
    arity_counts: dict[int, int] = {}
    face_count = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("f "):
                continue
            arity = len(stripped.split()) - 1
            if arity <= 0:
                continue
            face_count += 1
            arity_counts[arity] = arity_counts.get(arity, 0) + 1
    return {
        "rawFaceCount": int(face_count),
        "rawFaceArityCounts": {str(key): int(value) for key, value in sorted(arity_counts.items())},
    }


def _export_triangulated(input_mesh: Path, output_mesh: Path) -> None:
    mesh = _load_trimesh(input_mesh)
    if np.asarray(mesh.faces).size == 0:
        raise ValueError(f"Instant Meshes output has no faces: {input_mesh}")
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_mesh)


def _run_instant_meshes(
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int,
    config: BaselineConfig,
    *,
    rosy: int,
    posy: int,
    dominant: bool,
    omega_field: Path | None = None,
    mesh_feature_constraints: bool = False,
) -> dict[str, Any]:
    binary = _resolve_instant_meshes_bin(config)
    raw_output = output_mesh.with_name("mesh_raw.obj")
    stdout_path = output_mesh.with_name("instant_meshes_stdout.txt")
    stderr_path = output_mesh.with_name("instant_meshes_stderr.txt")
    cmd = [
        str(binary),
        "--output",
        str(raw_output),
        "--faces",
        str(int(target_faces)),
        "--rosy",
        str(int(rosy)),
        "--posy",
        str(int(posy)),
        "--crease",
        str(float(config.instant_meshes_crease_degrees)),
        "--smooth",
        str(int(config.instant_meshes_smooth_iterations)),
    ]
    if bool(config.instant_meshes_deterministic):
        cmd.append("--deterministic")
    if bool(config.instant_meshes_align_boundaries):
        cmd.append("--boundaries")
    if int(config.instant_meshes_threads) > 0:
        cmd.extend(["--threads", str(int(config.instant_meshes_threads))])
    if bool(dominant):
        cmd.append("--dominant")
    use_mesh_feature_constraints = bool(config.instant_meshes_feature_constraints) or bool(mesh_feature_constraints)
    if use_mesh_feature_constraints:
        cmd.append("--mesh-feature-constraints")
        cmd.extend(["--mesh-feature-constraint-weight", str(float(config.instant_meshes_feature_constraint_weight))])
        if float(config.instant_meshes_feature_constraint_angle) > 0:
            cmd.extend(["--mesh-feature-constraint-angle", str(float(config.instant_meshes_feature_constraint_angle))])
    field_dump_dir: Path | None = None
    if bool(config.instant_meshes_dump_fields) or omega_field is not None:
        field_dump_dir = output_mesh.parent / "instant_fields"
        cmd.extend(["--omega-dump-fields", str(field_dump_dir)])
    if omega_field is not None:
        cmd.extend(
            [
                "--omega-field",
                str(omega_field),
                "--omega-orientation-weight",
                str(float(config.instant_field_weight_scale)),
            ]
        )
        if bool(config.instant_guided_no_subdivide):
            cmd.append("--omega-no-subdivide")
    cmd.append(str(input_mesh))
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    for stale_path in (raw_output, output_mesh, stdout_path, stderr_path):
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
    result = subprocess.run(
        cmd,
        cwd=str(output_mesh.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Instant Meshes failed with code {result.returncode}: {message[:1000]}")
    if not raw_output.exists() or raw_output.stat().st_size == 0:
        raise RuntimeError(f"Instant Meshes did not create an output mesh: {raw_output}")
    _export_triangulated(raw_output, output_mesh)
    return {
        "rawMesh": str(raw_output),
        "instantMeshes": {
            "binary": str(binary),
            "command": cmd,
            "rawMesh": str(raw_output),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "rosy": int(rosy),
            "posy": int(posy),
            "dominant": bool(dominant),
            "creaseDegrees": float(config.instant_meshes_crease_degrees),
            "smoothIterations": int(config.instant_meshes_smooth_iterations),
            "deterministic": bool(config.instant_meshes_deterministic),
            "alignBoundaries": bool(config.instant_meshes_align_boundaries),
            "threads": int(config.instant_meshes_threads),
            "meshFeatureConstraints": bool(use_mesh_feature_constraints),
            "meshFeatureConstraintSource": "global" if bool(config.instant_meshes_feature_constraints) else ("guided" if bool(mesh_feature_constraints) else ""),
            "meshFeatureConstraintWeight": float(config.instant_meshes_feature_constraint_weight),
            "meshFeatureConstraintAngle": float(config.instant_meshes_feature_constraint_angle),
            "dumpFields": bool(config.instant_meshes_dump_fields) or omega_field is not None,
            "fieldDumpDir": str(field_dump_dir) if field_dump_dir is not None else "",
            "omegaField": str(omega_field) if omega_field is not None else "",
            "omegaOrientationWeight": float(config.instant_field_weight_scale) if omega_field is not None else 0.0,
            "omegaNoSubdivide": bool(config.instant_guided_no_subdivide) if omega_field is not None else False,
            **_obj_face_stats(raw_output),
        },
    }


def _run_open3d_qem(input_mesh: Path, output_mesh: Path, target_faces: int) -> None:
    mesh = o3d.io.read_triangle_mesh(str(input_mesh))
    if mesh.is_empty():
        raise ValueError(f"Open3D could not read mesh: {input_mesh}")
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=int(target_faces))
    simplified.remove_degenerate_triangles()
    simplified.remove_duplicated_triangles()
    simplified.remove_duplicated_vertices()
    simplified.remove_unreferenced_vertices()
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(output_mesh), simplified, write_ascii=False)


def _geogram_target_points(target_faces: int) -> int:
    # Geogram targets vertices, while this runner's comparison axis is faces.
    # For a mostly triangular surface, F is approximately 2V away from borders.
    return max(4, int(round(float(max(int(target_faces), 4)) * 0.5)))


def _density_summary_path(density_txt: Path) -> Path:
    return density_txt.with_name(f"{density_txt.stem}_summary.json")


def _density_normal_path(density_txt: Path) -> Path | None:
    summary_path = _density_summary_path(density_txt)
    if summary_path.exists() and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            normal_raw = str(summary.get("outputs", {}).get("normalTxt", "")).strip()
            if normal_raw:
                path = Path(normal_raw).expanduser()
                path = path if path.is_absolute() else (summary_path.parent / path)
                if path.exists() and path.is_file():
                    return path.resolve()
        except Exception:
            pass
    candidate = density_txt.with_name(f"{density_txt.stem}_normals.txt")
    return candidate.resolve() if candidate.exists() and candidate.is_file() else None


def _density_auto_target_points(density_txt: Path) -> int:
    summary_path = _density_summary_path(density_txt)
    if not summary_path.exists() or not summary_path.is_file():
        return 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    auto = summary.get("autoSizing", {}) if isinstance(summary.get("autoSizing", {}), dict) else {}
    if not bool(auto.get("enabled", False)):
        return 0
    return max(0, int(auto.get("estimatedTargetPoints", 0) or 0))


def _run_geogram_anisotropic(
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int,
    config: BaselineConfig,
    *,
    use_gradation: bool,
    omega_density: Path | None = None,
    omega_density_mode: str = "replace",
    omega_density_kind: str = "",
    omega_normal: Path | None = None,
    target_points_override: int = 0,
) -> dict[str, Any]:
    binary = _resolve_geogram_bin(config)
    stdout_path = output_mesh.with_name("geogram_stdout.txt")
    stderr_path = output_mesh.with_name("geogram_stderr.txt")
    target_points = int(target_points_override) if int(target_points_override) > 0 else _geogram_target_points(target_faces)
    gradation = float(config.geogram_gradation) if use_gradation else 0.0
    cmd = [
        str(binary),
        str(input_mesh),
        str(output_mesh),
        "profile=scan",
        "pre=true",
        "remesh=true",
        "post=true",
        f"remesh:nb_pts={int(target_points)}",
        f"remesh:anisotropy={float(config.geogram_anisotropy)}",
        f"remesh:gradation={float(gradation)}",
        f"remesh:lfs_samples={int(config.geogram_lfs_samples)}",
        f"pre:Nsmooth_iter={int(config.geogram_normal_smooth_iterations)}",
        f"opt:nb_Lloyd_iter={int(config.geogram_lloyd_iterations)}",
        f"opt:nb_Newton_iter={int(config.geogram_newton_iterations)}",
        f"opt:Newton_m={int(config.geogram_newton_m)}",
        "post:compute_normals=false",
    ]
    if bool(config.geogram_preserve_components):
        cmd.extend(["pre:min_comp_area=0%", "post:min_comp_area=0%"])
    if bool(config.geogram_preserve_holes):
        cmd.extend(["pre:max_hole_area=0%", "post:max_hole_area=0%"])
    if not bool(config.geogram_postprocess_degree3):
        cmd.append("post:max_deg3_dist=0%")
    if int(config.geogram_threads) > 0:
        cmd.append(f"sys:max_threads={int(config.geogram_threads)}")
    if omega_density is not None:
        # The OMeGa density sidecar is keyed by the preclean mesh vertices.
        # Disable Vorpalite repairs/fills that can change vertex indexing before
        # the sidecar is applied; normal lifting still runs in preprocess().
        cmd.extend(["pre:repair=false", "pre:max_hole_area=0%", "pre:min_comp_area=0%"])
        cmd.extend(
            [
                f"omega:density_file={omega_density}",
                f"omega:density_mode={omega_density_mode}",
                f"omega:density_scale={float(config.geogram_density_runtime_scale)}",
                f"omega:density_min={float(config.geogram_density_runtime_min)}",
                f"omega:density_max={float(config.geogram_density_runtime_max)}",
            ]
        )
    if omega_normal is not None and omega_normal.exists() and omega_normal.is_file():
        cmd.extend(
            [
                f"omega:normal_file={omega_normal}",
                f"omega:normal_blend={float(config.geogram_fused_normal_blend)}",
            ]
        )

    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    for stale_path in (output_mesh, stdout_path, stderr_path):
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
    result = subprocess.run(
        cmd,
        cwd=str(output_mesh.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"Geogram vorpalite failed with code {result.returncode}: {message[:1000]}")
    if not output_mesh.exists() or output_mesh.stat().st_size == 0:
        raise RuntimeError(f"Geogram vorpalite did not create an output mesh: {output_mesh}")
    return {
        "geogram": {
            "binary": str(binary),
            "command": cmd,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "targetPoints": int(target_points),
            "targetFacesApproximation": (
                "targetPoints = autoSizing.estimatedTargetPoints"
                if int(target_points_override) > 0
                else "targetPoints = round(targetFaces * 0.5)"
            ),
            "autoTargetPoints": bool(int(target_points_override) > 0),
            "anisotropy": float(config.geogram_anisotropy),
            "gradation": float(gradation),
            "usesLfsGradation": bool(use_gradation),
            "normalSmoothIterations": int(config.geogram_normal_smooth_iterations),
            "lfsSamples": int(config.geogram_lfs_samples),
            "lloydIterations": int(config.geogram_lloyd_iterations),
            "newtonIterations": int(config.geogram_newton_iterations),
            "newtonM": int(config.geogram_newton_m),
            "threads": int(config.geogram_threads),
            "preserveComponents": bool(config.geogram_preserve_components),
            "preserveHoles": bool(config.geogram_preserve_holes),
            "postprocessDegree3": bool(config.geogram_postprocess_degree3),
            "omegaDensity": str(omega_density) if omega_density is not None else "",
            "omegaDensityMode": str(omega_density_mode) if omega_density is not None else "",
            "omegaDensityKind": str(omega_density_kind) if omega_density is not None else "",
            "omegaNormal": str(omega_normal) if omega_normal is not None else "",
            "omegaNormalBlend": float(config.geogram_fused_normal_blend) if omega_normal is not None else 0.0,
            "omegaDensityRuntimeScale": float(config.geogram_density_runtime_scale) if omega_density is not None else 0.0,
            "omegaDensityRuntimeMin": float(config.geogram_density_runtime_min) if omega_density is not None else 0.0,
            "omegaDensityRuntimeMax": float(config.geogram_density_runtime_max) if omega_density is not None else 0.0,
            "omegaDensityPreserveInputVertexIndexing": bool(omega_density is not None),
        },
    }


def _load_meshlab(input_mesh: Path) -> pymeshlab.MeshSet:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(input_mesh))
    return ms


def _save_meshlab(ms: pymeshlab.MeshSet, output_mesh: Path) -> None:
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    ms.save_current_mesh(str(output_mesh), save_face_color=False, save_vertex_color=False)


def _run_meshlab_qem(input_mesh: Path, output_mesh: Path, target_faces: int, config: BaselineConfig, *, planar: bool) -> None:
    ms = _load_meshlab(input_mesh)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        qualitythr=float(config.qem_quality_threshold),
        preserveboundary=bool(config.preserve_boundary),
        boundaryweight=float(config.qem_boundary_weight),
        preservenormal=bool(config.preserve_normal),
        preservetopology=bool(config.preserve_topology),
        optimalplacement=True,
        planarquadric=bool(planar),
        planarweight=float(config.planar_quadric_weight if planar else 0.001),
        qualityweight=False,
        autoclean=True,
        selected=False,
    )
    _save_meshlab(ms, output_mesh)


def _isotropic_target_percent(input_mesh: Path, target_faces: int, config: BaselineConfig) -> float:
    vertices, faces = load_mesh_arrays(input_mesh)
    bbox_diag = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1.0e-12)
    if float(config.isotropic_target_percent) > 0.0:
        return max(1.0e-4, float(config.isotropic_target_percent))
    if float(config.isotropic_target_edge_length) > 0.0:
        return max(1.0e-4, 100.0 * float(config.isotropic_target_edge_length) / bbox_diag)
    _centers, _normals, areas, _mean_edge = face_geometry(vertices, faces)
    total_area = max(float(np.sum(areas)), 1.0e-12)
    target_len = math.sqrt((4.0 * total_area) / (math.sqrt(3.0) * max(float(target_faces), 1.0)))
    return max(1.0e-4, 100.0 * float(config.isotropic_target_scale) * target_len / bbox_diag)


def _run_meshlab_isotropic(
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int,
    config: BaselineConfig,
    *,
    feature_sensitive: bool = False,
) -> pymeshlab.MeshSet:
    ms = _load_meshlab(input_mesh)
    target_percent = _isotropic_target_percent(input_mesh, target_faces, config)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=int(config.isotropic_iterations),
        adaptive=True,
        selectedonly=False,
        targetlen=pymeshlab.PercentageValue(float(target_percent)),
        featuredeg=float(config.feature_isotropic_degrees if feature_sensitive else config.isotropic_feature_degrees),
        checksurfdist=True,
        maxsurfdist=pymeshlab.PercentageValue(float(config.isotropic_max_surface_dist_percent)),
        splitflag=True,
        collapseflag=True,
        swapflag=True,
        smoothflag=bool(config.feature_isotropic_smooth if feature_sensitive else True),
        reprojectflag=True,
    )
    _save_meshlab(ms, output_mesh)
    return ms


def _run_meshlab_isotropic_qem(
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int,
    config: BaselineConfig,
    *,
    feature_sensitive: bool = False,
) -> None:
    ms = _run_meshlab_isotropic(input_mesh, output_mesh, target_faces, config, feature_sensitive=feature_sensitive)
    current_faces = int(ms.current_mesh().face_number())
    if current_faces > int(target_faces):
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target_faces),
            qualitythr=float(config.qem_quality_threshold),
            preserveboundary=bool(config.preserve_boundary),
            boundaryweight=float(config.qem_boundary_weight),
            preservenormal=bool(config.preserve_normal),
            preservetopology=bool(config.preserve_topology),
            optimalplacement=True,
            planarquadric=True,
            planarweight=float(config.planar_quadric_weight),
            qualityweight=False,
            autoclean=True,
            selected=False,
        )
    _save_meshlab(ms, output_mesh)


def _run_method(
    method: str,
    input_mesh: Path,
    output_mesh: Path,
    target_faces: int,
    config: BaselineConfig,
    *,
    instant_field_path: Path | None = None,
    geogram_density_paths: dict[str, Path] | None = None,
    geogram_normal_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    geogram_density_paths = geogram_density_paths or {}
    geogram_normal_paths = geogram_normal_paths or {}
    if method == "copy_source":
        _copy_input_mesh(input_mesh, output_mesh)
        return {}
    elif method == "open3d_qem":
        _run_open3d_qem(input_mesh, output_mesh, target_faces)
        return {}
    elif method == "meshlab_qem":
        _run_meshlab_qem(input_mesh, output_mesh, target_faces, config, planar=False)
        return {}
    elif method == "meshlab_planar_qem":
        _run_meshlab_qem(input_mesh, output_mesh, target_faces, config, planar=True)
        return {}
    elif method == "meshlab_isotropic":
        _run_meshlab_isotropic(input_mesh, output_mesh, target_faces, config)
        return {}
    elif method == "meshlab_feature_isotropic":
        _run_meshlab_isotropic(input_mesh, output_mesh, target_faces, config, feature_sensitive=True)
        return {}
    elif method == "meshlab_isotropic_qem":
        _run_meshlab_isotropic_qem(input_mesh, output_mesh, target_faces, config)
        return {}
    elif method == "meshlab_feature_isotropic_qem":
        _run_meshlab_isotropic_qem(input_mesh, output_mesh, target_faces, config, feature_sensitive=True)
        return {}
    elif method == "geogram_anisotropic":
        return _run_geogram_anisotropic(input_mesh, output_mesh, target_faces, config, use_gradation=False)
    elif method == "geogram_anisotropic_gradation":
        return _run_geogram_anisotropic(input_mesh, output_mesh, target_faces, config, use_gradation=True)
    elif method == "geogram_omega_density":
        geogram_density_path = geogram_density_paths.get("base")
        if geogram_density_path is None:
            raise ValueError("OMeGa-density Geogram requires an omega_geogram_density.txt sidecar.")
        target_points = _density_auto_target_points(geogram_density_path) if bool(config.geogram_auto_target_points) else 0
        return _run_geogram_anisotropic(
            input_mesh,
            output_mesh,
            target_faces,
            config,
            use_gradation=False,
            omega_density=geogram_density_path,
            omega_density_mode="replace",
            omega_density_kind="base",
            omega_normal=geogram_normal_paths.get("base"),
            target_points_override=int(target_points),
        )
    elif method == "geogram_omega_density_gradation":
        geogram_density_path = geogram_density_paths.get("planar_gradation")
        if geogram_density_path is None:
            raise ValueError("OMeGa-density gradation Geogram requires an omega_geogram_density_planar_gradation.txt sidecar.")
        target_points = _density_auto_target_points(geogram_density_path) if bool(config.geogram_auto_target_points) else 0
        return _run_geogram_anisotropic(
            input_mesh,
            output_mesh,
            target_faces,
            config,
            use_gradation=False,
            omega_density=geogram_density_path,
            omega_density_mode="replace",
            omega_density_kind="planar_gradation",
            omega_normal=geogram_normal_paths.get("planar_gradation"),
            target_points_override=int(target_points),
        )
    elif method == "instant_meshes_quad":
        return _run_instant_meshes(input_mesh, output_mesh, target_faces, config, rosy=4, posy=4, dominant=False)
    elif method == "instant_meshes_tri":
        return _run_instant_meshes(input_mesh, output_mesh, target_faces, config, rosy=6, posy=6, dominant=False)
    elif method == "instant_meshes_dominant":
        return _run_instant_meshes(input_mesh, output_mesh, target_faces, config, rosy=4, posy=4, dominant=True)
    elif method == "instant_meshes_quad_guided":
        if instant_field_path is None:
            raise ValueError("Guided Instant Meshes requires an OMeGa field sidecar.")
        return _run_instant_meshes(
            input_mesh,
            output_mesh,
            target_faces,
            config,
            rosy=4,
            posy=4,
            dominant=False,
            omega_field=instant_field_path,
            mesh_feature_constraints=bool(config.instant_guided_feature_constraints),
        )
    elif method == "instant_meshes_tri_guided":
        if instant_field_path is None:
            raise ValueError("Guided Instant Meshes requires an OMeGa field sidecar.")
        return _run_instant_meshes(
            input_mesh,
            output_mesh,
            target_faces,
            config,
            rosy=6,
            posy=6,
            dominant=False,
            omega_field=instant_field_path,
            mesh_feature_constraints=bool(config.instant_guided_feature_constraints),
        )
    elif method == "instant_meshes_dominant_guided":
        if instant_field_path is None:
            raise ValueError("Guided Instant Meshes requires an OMeGa field sidecar.")
        return _run_instant_meshes(
            input_mesh,
            output_mesh,
            target_faces,
            config,
            rosy=4,
            posy=4,
            dominant=True,
            omega_field=instant_field_path,
            mesh_feature_constraints=bool(config.instant_guided_feature_constraints),
        )
    else:
        raise ValueError(f"Unsupported baseline method: {method}")


def _resolve_or_export_instant_field(
    *,
    config: BaselineConfig,
    input_mesh: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    progress: ProgressFn,
) -> Path | None:
    if not any(method in GUIDED_INSTANT_MESHES_METHODS for method in methods):
        return None
    if config.instant_field_path is not None:
        field_path = config.instant_field_path.expanduser().resolve()
        if not field_path.exists() or field_path.is_dir():
            raise FileNotFoundError(f"Requested Instant Meshes OMeGa field does not exist: {field_path}")
        progress(f"[baselines] Guided Instant field: {field_path}")
        return field_path

    field_dir = output_dir / "omega_instant_field"
    field_path = field_dir / "omega_instant_field.txt"
    if field_path.exists() and not bool(config.overwrite):
        progress(f"[baselines] Reusing guided Instant field: {field_path}")
        return field_path

    progress("[baselines] Exporting guided Instant field from Phase 2 normal evidence")
    result = compute_omega_instant_field(
        OmegaInstantFieldConfig(
            model_dir=config.model_dir,
            mesh_path=input_mesh,
            output_dir=field_dir,
            frame_stride=int(config.instant_field_frame_stride),
            max_frames=int(config.instant_field_max_frames),
            pixel_stride=int(config.instant_field_pixel_stride),
            min_direction_confidence=float(config.instant_field_min_direction_confidence),
            min_gradient_normalized=float(config.instant_field_min_gradient_normalized),
            gradient_mode=str(config.instant_field_gradient_mode),
            weight_quantile=float(config.instant_field_weight_quantile),
            min_vertex_weight=float(config.instant_field_min_vertex_weight),
            overwrite=bool(config.overwrite),
        ),
        progress=progress,
    )
    return result.field_txt


def _resolve_or_export_geogram_density(
    *,
    config: BaselineConfig,
    input_mesh: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    progress: ProgressFn,
) -> dict[str, tuple[Path | None, Path | None, Path | None, Path | None]]:
    if not any(method in GEOGRAM_OMEGA_DENSITY_METHODS for method in methods):
        return {}
    if config.geogram_density_path is not None:
        density_txt = config.geogram_density_path.expanduser().resolve()
        if not density_txt.exists() or density_txt.is_dir():
            raise FileNotFoundError(f"Requested OMeGa Geogram density sidecar does not exist: {density_txt}")
        density_npz = density_txt.with_suffix(".npz")
        density_summary = density_txt.with_name(f"{density_txt.stem}_summary.json")
        progress(f"[baselines] Guided Geogram density: {density_txt}")
        sidecar = (
            density_txt,
            density_npz if density_npz.exists() and density_npz.is_file() else None,
            density_summary if density_summary.exists() and density_summary.is_file() else None,
            _density_normal_path(density_txt),
        )
        return {"base": sidecar, "planar_gradation": sidecar}

    def export_one(kind: str, *, iterations: int) -> tuple[Path, Path, Path, Path | None]:
        density_name = "omega_geogram_density" if kind == "base" else "omega_geogram_density_planar_gradation"
        density_dir = output_dir / density_name
        density_txt = density_dir / f"{density_name}.txt"
        density_npz = density_dir / f"{density_name}.npz"
        density_summary = density_dir / f"{density_name}_summary.json"
        if density_txt.exists() and density_npz.exists() and density_summary.exists() and not bool(config.overwrite):
            progress(f"[baselines] Reusing guided Geogram density ({kind}): {density_txt}")
            return density_txt, density_npz, density_summary, _density_normal_path(density_txt)
        if any(path.exists() for path in (density_txt, density_npz, density_summary)) and not bool(config.overwrite):
            raise FileExistsError(
                f"Partial guided Geogram density outputs exist in {density_dir}. "
                "Re-run with --overwrite or pass --geogram-density-path."
            )

        progress(f"[baselines] Exporting guided Geogram density ({kind}) from Phase 3 weights")
        normal_lift_scale = float(config.geogram_density_planar_gradation_normal_lift_scale)
        if normal_lift_scale <= 0.0:
            normal_lift_scale = float(config.geogram_anisotropy)
        result = compute_omega_geogram_density(
            OmegaGeogramDensityConfig(
                model_dir=config.model_dir,
                mesh_path=input_mesh,
                output_dir=density_dir,
                density_name=density_name,
                min_density=float(config.geogram_density_min),
                max_density=float(config.geogram_density_max),
                signal_gamma=float(config.geogram_density_signal_gamma),
                detail_influence=float(config.geogram_density_detail_influence),
                boundary_influence=float(config.geogram_density_boundary_influence),
                uncertain_influence=float(config.geogram_density_uncertain_influence),
                auto_normal_error_degrees=float(config.geogram_auto_from_normal_error_degrees),
                auto_h_min_factor=float(config.geogram_auto_h_min_factor),
                auto_h_max_factor=float(config.geogram_auto_h_max_factor),
                auto_target_point_scale=float(config.geogram_auto_target_point_scale),
                use_fused_normals=bool(config.geogram_use_fused_normals),
                fused_normal_blend=float(config.geogram_fused_normal_blend),
                fused_normal_min_weight=float(config.geogram_fused_normal_min_weight),
                planar_gradation_iterations=int(iterations),
                planar_gradation_strength=float(config.geogram_density_planar_gradation_strength),
                planar_gradation_planar_power=float(config.geogram_density_planar_gradation_planar_power),
                planar_gradation_min_edge_weight=float(config.geogram_density_planar_gradation_min_edge_weight),
                planar_gradation_max_density_ratio=float(config.geogram_density_planar_gradation_max_density_ratio),
                planar_gradation_normal_lift_scale=normal_lift_scale,
                planar_gradation_metric_tau_factor=float(config.geogram_density_planar_gradation_metric_tau_factor),
                planar_gradation_min_normal_gate=float(config.geogram_density_planar_gradation_min_normal_gate),
                planar_gradation_use_proxy_boundaries=bool(config.geogram_density_planar_gradation_use_proxy_boundaries),
                overwrite=bool(config.overwrite),
            ),
            progress=progress,
        )
        return result.density_txt, result.density_npz, result.summary_json, result.normal_txt

    sidecars: dict[str, tuple[Path | None, Path | None, Path | None, Path | None]] = {}
    if "geogram_omega_density" in methods:
        sidecars["base"] = export_one("base", iterations=0)
    if "geogram_omega_density_gradation" in methods:
        sidecars["planar_gradation"] = export_one(
            "planar_gradation",
            iterations=int(config.geogram_density_planar_gradation_iterations),
        )
    return sidecars


def compute_remesh_baselines(config: BaselineConfig, progress: ProgressFn | None = None) -> BaselineResult:
    progress = progress or _progress_default
    model_dir = config.model_dir.expanduser().resolve()
    input_mesh = (
        config.mesh_path.expanduser().resolve()
        if config.mesh_path is not None
        else (model_dir / "remesh" / "local" / "preclean_mesh.ply")
    )
    if not input_mesh.exists():
        input_mesh = resolve_latest_omega_mesh(model_dir, iteration=int(config.iteration))
    methods = tuple(dict.fromkeys(str(method) for method in config.methods))
    output_dir = (
        _resolve_baseline_output_dir(config, model_dir, methods)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stats = _mesh_stats(input_mesh)
    targets = _target_faces(config, int(input_stats["faces"]), methods)
    instant_field_path = _resolve_or_export_instant_field(
        config=config,
        input_mesh=input_mesh,
        output_dir=output_dir,
        methods=methods,
        progress=progress,
    )
    geogram_density_sidecars = _resolve_or_export_geogram_density(
        config=config,
        input_mesh=input_mesh,
        output_dir=output_dir,
        methods=methods,
        progress=progress,
    )
    geogram_density_paths = {
        key: sidecar[0]
        for key, sidecar in geogram_density_sidecars.items()
        if sidecar[0] is not None
    }
    geogram_normal_paths = {
        key: sidecar[3]
        for key, sidecar in geogram_density_sidecars.items()
        if len(sidecar) > 3 and sidecar[3] is not None
    }
    rows: list[dict[str, Any]] = []

    progress(f"[baselines] Input mesh: {input_mesh}")
    progress(f"[baselines] Methods: {', '.join(methods)}")
    target_label = ", ".join("edge-length-only" if int(v) <= 0 else str(v) for v in targets)
    progress(f"[baselines] Targets: {target_label}")
    for target in targets:
        for method in methods:
            run_name = _safe_name(method, target, config)
            run_dir = output_dir / run_name
            mesh_out = run_dir / "mesh.ply"
            summary_out = run_dir / "summary.json"
            if run_dir.exists() and any(run_dir.iterdir()) and not bool(config.overwrite):
                raise FileExistsError(f"Baseline output exists: {run_dir}. Re-run with --overwrite.")
            run_dir.mkdir(parents=True, exist_ok=True)
            progress(f"[baselines] {method} target={target:,}")
            error: str | None = None
            method_extra: dict[str, Any] = {}
            try:
                method_extra = _run_method(
                    method,
                    input_mesh,
                    mesh_out,
                    int(target),
                    config,
                    instant_field_path=instant_field_path,
                    geogram_density_paths=geogram_density_paths,
                    geogram_normal_paths=geogram_normal_paths,
                )
                output_stats = _mesh_stats(mesh_out)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                output_stats = {}
                if mesh_out.exists():
                    mesh_out.unlink()
            row = {
                "name": run_name,
                "method": method,
                "targetFaces": int(target),
                "targetMode": "isotropic_edge_length_only" if int(target) <= 0 else "face_count",
                "isotropicTargetPercent": float(_isotropic_target_percent(input_mesh, int(target), config)) if "isotropic" in method else 0.0,
                "mesh": str(mesh_out) if mesh_out.exists() else "",
                "summary": str(summary_out),
                "error": error,
                "input": input_stats,
                "output": output_stats,
                **method_extra,
            }
            _write_json(summary_out, row)
            rows.append(row)
            if error is None:
                progress(f"[baselines] wrote {mesh_out} faces={output_stats.get('faces', 0):,}")
            else:
                progress(f"[baselines] failed {method}: {error}")

    manifest_jsonl = output_dir / "baseline_manifest.jsonl"
    summary_json = output_dir / "baseline_summary.json"
    _write_jsonl(manifest_jsonl, rows)
    summary = {
        "stageName": "omega_remesh_baselines",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model_dir": str(model_dir),
            "mesh_path": str(input_mesh),
            "output_dir": str(output_dir),
            "experimentName": str(output_dir.name),
            "canonicalOutputRoot": str(model_dir / "remesh" / "baselines"),
            "methods": list(methods),
            "targetFaceRatios": [float(v) for v in config.target_face_ratios],
            "targetFaces": [int(v) for v in config.target_faces],
            "iteration": int(config.iteration),
            "isotropicIterations": int(config.isotropic_iterations),
            "isotropicFeatureDegrees": float(config.isotropic_feature_degrees),
            "isotropicTargetScale": float(config.isotropic_target_scale),
            "isotropicTargetEdgeLength": float(config.isotropic_target_edge_length),
            "isotropicTargetPercent": float(config.isotropic_target_percent),
            "isotropicMaxSurfaceDistPercent": float(config.isotropic_max_surface_dist_percent),
            "featureIsotropicDegrees": float(config.feature_isotropic_degrees),
            "featureIsotropicSmooth": bool(config.feature_isotropic_smooth),
            "instantMeshesBin": str(config.instant_meshes_bin) if config.instant_meshes_bin is not None else "",
            "instantMeshesCreaseDegrees": float(config.instant_meshes_crease_degrees),
            "instantMeshesSmoothIterations": int(config.instant_meshes_smooth_iterations),
            "instantMeshesThreads": int(config.instant_meshes_threads),
            "instantMeshesDeterministic": bool(config.instant_meshes_deterministic),
            "instantMeshesAlignBoundaries": bool(config.instant_meshes_align_boundaries),
            "instantMeshesDumpFields": bool(config.instant_meshes_dump_fields),
            "instantMeshesFeatureConstraints": bool(config.instant_meshes_feature_constraints),
            "instantMeshesFeatureConstraintWeight": float(config.instant_meshes_feature_constraint_weight),
            "instantMeshesFeatureConstraintAngle": float(config.instant_meshes_feature_constraint_angle),
            "instantGuidedFeatureConstraints": bool(config.instant_guided_feature_constraints),
            "instantFieldPath": str(config.instant_field_path) if config.instant_field_path is not None else "",
            "instantFieldResolvedPath": str(instant_field_path) if instant_field_path is not None else "",
            "instantFieldWeightScale": float(config.instant_field_weight_scale),
            "instantFieldFrameStride": int(config.instant_field_frame_stride),
            "instantFieldMaxFrames": int(config.instant_field_max_frames),
            "instantFieldPixelStride": int(config.instant_field_pixel_stride),
            "instantFieldMinDirectionConfidence": float(config.instant_field_min_direction_confidence),
            "instantFieldMinGradientNormalized": float(config.instant_field_min_gradient_normalized),
            "instantFieldGradientMode": str(config.instant_field_gradient_mode),
            "instantFieldWeightQuantile": float(config.instant_field_weight_quantile),
            "instantFieldMinVertexWeight": float(config.instant_field_min_vertex_weight),
            "instantGuidedNoSubdivide": bool(config.instant_guided_no_subdivide),
            "geogramBin": str(config.geogram_bin) if config.geogram_bin is not None else "",
            "geogramAnisotropy": float(config.geogram_anisotropy),
            "geogramGradation": float(config.geogram_gradation),
            "geogramNormalSmoothIterations": int(config.geogram_normal_smooth_iterations),
            "geogramLfsSamples": int(config.geogram_lfs_samples),
            "geogramLloydIterations": int(config.geogram_lloyd_iterations),
            "geogramNewtonIterations": int(config.geogram_newton_iterations),
            "geogramNewtonM": int(config.geogram_newton_m),
            "geogramThreads": int(config.geogram_threads),
            "geogramPreserveComponents": bool(config.geogram_preserve_components),
            "geogramPreserveHoles": bool(config.geogram_preserve_holes),
            "geogramPostprocessDegree3": bool(config.geogram_postprocess_degree3),
            "geogramDensityPath": str(config.geogram_density_path) if config.geogram_density_path is not None else "",
            "geogramDensityResolvedPath": str(geogram_density_sidecars.get("base", ("", "", "", ""))[0] or ""),
            "geogramDensityNpz": str(geogram_density_sidecars.get("base", ("", "", "", ""))[1] or ""),
            "geogramDensitySummary": str(geogram_density_sidecars.get("base", ("", "", "", ""))[2] or ""),
            "geogramDensityNormalPath": str(geogram_density_sidecars.get("base", ("", "", "", ""))[3] or ""),
            "geogramDensityPlanarGradationResolvedPath": str(geogram_density_sidecars.get("planar_gradation", ("", "", "", ""))[0] or ""),
            "geogramDensityPlanarGradationNpz": str(geogram_density_sidecars.get("planar_gradation", ("", "", "", ""))[1] or ""),
            "geogramDensityPlanarGradationSummary": str(geogram_density_sidecars.get("planar_gradation", ("", "", "", ""))[2] or ""),
            "geogramDensityPlanarGradationNormalPath": str(geogram_density_sidecars.get("planar_gradation", ("", "", "", ""))[3] or ""),
            "geogramDensityMin": float(config.geogram_density_min),
            "geogramDensityMax": float(config.geogram_density_max),
            "geogramDensitySignalGamma": float(config.geogram_density_signal_gamma),
            "geogramDensityDetailInfluence": float(config.geogram_density_detail_influence),
            "geogramDensityBoundaryInfluence": float(config.geogram_density_boundary_influence),
            "geogramDensityUncertainInfluence": float(config.geogram_density_uncertain_influence),
            "geogramAutoFromNormalErrorDegrees": float(config.geogram_auto_from_normal_error_degrees),
            "geogramAutoTargetPoints": bool(config.geogram_auto_target_points),
            "geogramAutoHMinFactor": float(config.geogram_auto_h_min_factor),
            "geogramAutoHMaxFactor": float(config.geogram_auto_h_max_factor),
            "geogramAutoTargetPointScale": float(config.geogram_auto_target_point_scale),
            "geogramUseFusedNormals": bool(config.geogram_use_fused_normals),
            "geogramFusedNormalBlend": float(config.geogram_fused_normal_blend),
            "geogramFusedNormalMinWeight": float(config.geogram_fused_normal_min_weight),
            "geogramDensityRuntimeScale": float(config.geogram_density_runtime_scale),
            "geogramDensityRuntimeMin": float(config.geogram_density_runtime_min),
            "geogramDensityRuntimeMax": float(config.geogram_density_runtime_max),
            "geogramDensityPlanarGradationIterations": int(config.geogram_density_planar_gradation_iterations),
            "geogramDensityPlanarGradationStrength": float(config.geogram_density_planar_gradation_strength),
            "geogramDensityPlanarGradationPlanarPower": float(config.geogram_density_planar_gradation_planar_power),
            "geogramDensityPlanarGradationMinEdgeWeight": float(config.geogram_density_planar_gradation_min_edge_weight),
            "geogramDensityPlanarGradationMaxDensityRatio": float(config.geogram_density_planar_gradation_max_density_ratio),
            "geogramDensityPlanarGradationNormalLiftScale": float(config.geogram_density_planar_gradation_normal_lift_scale),
            "geogramDensityPlanarGradationMetricTauFactor": float(config.geogram_density_planar_gradation_metric_tau_factor),
            "geogramDensityPlanarGradationMinNormalGate": float(config.geogram_density_planar_gradation_min_normal_gate),
            "geogramDensityPlanarGradationUseProxyBoundaries": bool(config.geogram_density_planar_gradation_use_proxy_boundaries),
            "qemQualityThreshold": float(config.qem_quality_threshold),
            "qemBoundaryWeight": float(config.qem_boundary_weight),
            "planarQuadricWeight": float(config.planar_quadric_weight),
            "preserveBoundary": bool(config.preserve_boundary),
            "preserveTopology": bool(config.preserve_topology),
            "preserveNormal": bool(config.preserve_normal),
            "overwrite": bool(config.overwrite),
        },
        "input": input_stats,
        "targets": targets,
        "rows": rows,
        "outputs": {
            "manifestJsonl": str(manifest_jsonl),
            "summaryJson": str(summary_json),
            "outputDir": str(output_dir),
        },
        "notes": [
            "open3d_qem and meshlab_qem are mesh-only QEM simplification baselines.",
            "meshlab_planar_qem adds MeshLab planar quadrics, which is often better on architectural planes.",
            "meshlab_isotropic is a pure explicit isotropic remeshing baseline.",
            "meshlab_feature_isotropic uses a lower feature angle threshold and disables smoothing by default to better preserve sharp structures.",
            "meshlab_isotropic_qem runs explicit isotropic remeshing before planar QEM; it is a practical edge-quality/structure baseline, not the exact anisotropic polygonal remeshing paper.",
            "meshlab_feature_isotropic_qem applies planar QEM after the feature-sensitive isotropic pass.",
            "instant_meshes_quad, instant_meshes_tri, and instant_meshes_dominant call Wenzel Jakob's Instant Meshes in batch mode; raw OBJ output is kept, and mesh.ply is a triangulated copy for our projection visualizer.",
            "instant_meshes_*_guided uses the same Instant Meshes solver with optional OMeGa soft orientation constraints exported from view-normal evidence.",
            "geogram_anisotropic calls Geogram/Vorpalite CVT remeshing with normal lifting; target face count is converted to an approximate target vertex count.",
            "geogram_anisotropic_gradation additionally enables Geogram's LFS-based gradation weights, testing adaptive density on top of anisotropic triangle shape.",
            "geogram_omega_density replaces Geogram's vertex weight field with OMeGa Phase 3 density: high planar weight lowers density, and detail/boundary evidence raises density.",
            "geogram_omega_density_gradation uses a separate OMeGa density sidecar with bounded planar-only log-density smoothing; it does not multiply by Geogram's LFS field.",
        ],
    }
    _write_json(summary_json, summary)
    return BaselineResult(summary_json=summary_json, manifest_jsonl=manifest_jsonl, output_dir=output_dir, method_count=len(rows))


def _parse_methods(raw: str, preset: str) -> tuple[str, ...]:
    requested = tuple(value.strip() for value in str(raw).split(",") if value.strip())
    if requested:
        values = requested
    else:
        preset_key = str(preset).strip()
        if preset_key not in METHOD_PRESETS:
            raise ValueError(f"Unknown method preset '{preset}'. Choose one of: {', '.join(sorted(METHOD_PRESETS))}.")
        values = METHOD_PRESETS[preset_key]
    unsupported = [value for value in values if value not in SUPPORTED_METHODS]
    if unsupported:
        raise ValueError(f"Unsupported baseline methods: {', '.join(unsupported)}")
    return tuple(dict.fromkeys(values))


def _parse_float_list(raw: str) -> tuple[float, ...]:
    if not str(raw).strip():
        return ()
    return tuple(float(value.strip()) for value in str(raw).split(",") if value.strip())


def _parse_int_list(raw: str) -> tuple[int, ...]:
    if not str(raw).strip():
        return ()
    return tuple(int(value.strip()) for value in str(raw).split(",") if value.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mesh-only remeshing baselines for an OMeGa mesh.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/preclean_mesh.ply, then latest OMeGa mesh.")
    parser.add_argument(
        "--experiment-name",
        default="",
        help=(
            "Experiment folder name under <model-dir>/remesh/baselines/. "
            "Defaults from --method-preset, for example qem_compare or geogram_compare."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional full override. Prefer --experiment-name for conventional outputs.",
    )
    parser.add_argument(
        "--method-preset",
        choices=tuple(METHOD_PRESETS),
        default="key",
        help=(
            "Method preset to use when --methods is empty. Use qem, isotropic, "
            "instant_guided, and geogram for organized baseline sweeps."
        ),
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Comma-separated methods: open3d_qem, meshlab_qem, meshlab_planar_qem, "
            "meshlab_isotropic, meshlab_feature_isotropic, meshlab_isotropic_qem, "
            "meshlab_feature_isotropic_qem, instant_meshes_quad, instant_meshes_tri, "
            "instant_meshes_dominant, instant_meshes_quad_guided, instant_meshes_tri_guided, "
            "instant_meshes_dominant_guided, geogram_anisotropic, "
            "geogram_anisotropic_gradation, geogram_omega_density, "
            "geogram_omega_density_gradation. Overrides --method-preset."
        ),
    )
    parser.add_argument("--target-face-ratios", default="0.20,0.40,0.60", help="Comma-separated ratios relative to input face count.")
    parser.add_argument("--target-faces", default="", help="Comma-separated absolute target face counts.")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--isotropic-iterations", type=int, default=5)
    parser.add_argument("--isotropic-feature-degrees", type=float, default=35.0)
    parser.add_argument("--isotropic-target-scale", type=float, default=1.0)
    parser.add_argument(
        "--isotropic-target-edge-length",
        type=float,
        default=0.0,
        help="World-unit target edge length for MeshLab isotropic remeshing. Overrides derived length when > 0.",
    )
    parser.add_argument(
        "--isotropic-target-percent",
        type=float,
        default=0.0,
        help="MeshLab targetlen percentage of bounding-box diagonal. Overrides --isotropic-target-edge-length when > 0.",
    )
    parser.add_argument("--isotropic-max-surface-dist-percent", type=float, default=1.0)
    parser.add_argument("--feature-isotropic-degrees", type=float, default=20.0)
    parser.add_argument("--feature-isotropic-smooth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--instant-meshes-bin", type=Path, default=None, help="Path to the built/precompiled Instant Meshes executable.")
    parser.add_argument("--instant-meshes-crease-degrees", type=float, default=30.0)
    parser.add_argument("--instant-meshes-smooth-iterations", type=int, default=2)
    parser.add_argument("--instant-meshes-threads", type=int, default=0, help="0 lets Instant Meshes choose automatically.")
    parser.add_argument("--instant-meshes-deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instant-meshes-align-boundaries", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--instant-meshes-dump-fields", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--instant-meshes-feature-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass high-dihedral mesh feature edges to Instant Meshes as interior orientation constraints.",
    )
    parser.add_argument("--instant-meshes-feature-constraint-weight", type=float, default=0.9)
    parser.add_argument(
        "--instant-meshes-feature-constraint-angle",
        type=float,
        default=0.0,
        help="Dihedral threshold in degrees for feature constraints. 0 reuses --instant-meshes-crease-degrees.",
    )
    parser.add_argument(
        "--instant-guided-feature-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply mesh feature constraints only to instant_meshes_*_guided rows.",
    )
    parser.add_argument("--instant-field-path", type=Path, default=None, help="Optional precomputed omega_instant_field.txt sidecar.")
    parser.add_argument(
        "--instant-field-weight-scale",
        type=float,
        default=0.7,
        help="Scale the exported guide confidence before it becomes Instant Meshes CQw blend weight.",
    )
    parser.add_argument("--instant-field-frame-stride", type=int, default=1)
    parser.add_argument("--instant-field-max-frames", type=int, default=0, help="0 uses every selected evidence frame.")
    parser.add_argument("--instant-field-pixel-stride", type=int, default=2)
    parser.add_argument("--instant-field-min-direction-confidence", type=float, default=0.15)
    parser.add_argument("--instant-field-min-gradient-normalized", type=float, default=0.35)
    parser.add_argument("--instant-field-gradient-mode", choices=("point", "local", "combined"), default="point")
    parser.add_argument("--instant-field-weight-quantile", type=float, default=0.90)
    parser.add_argument("--instant-field-min-vertex-weight", type=float, default=0.02)
    parser.add_argument(
        "--instant-guided-no-subdivide",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug option: skip Instant Meshes internal subdivision for guided rows.",
    )
    parser.add_argument("--geogram-bin", type=Path, default=None, help="Path to Geogram's vorpalite executable.")
    parser.add_argument(
        "--geogram-anisotropy",
        type=float,
        default=1.0,
        help="Geogram remesh:anisotropy value. Vorpalite internally multiplies this by 0.02 for normal lifting.",
    )
    parser.add_argument(
        "--geogram-gradation",
        type=float,
        default=1.0,
        help="Geogram remesh:gradation value used only by geogram_anisotropic_gradation.",
    )
    parser.add_argument("--geogram-normal-smooth-iterations", type=int, default=3)
    parser.add_argument("--geogram-lfs-samples", type=int, default=10000)
    parser.add_argument("--geogram-lloyd-iterations", type=int, default=5)
    parser.add_argument("--geogram-newton-iterations", type=int, default=30)
    parser.add_argument("--geogram-newton-m", type=int, default=7)
    parser.add_argument("--geogram-threads", type=int, default=0, help="0 lets Geogram choose automatically.")
    parser.add_argument(
        "--geogram-preserve-components",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Vorpalite's small-component removal by setting pre/post min component area to 0%%.",
    )
    parser.add_argument(
        "--geogram-preserve-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Vorpalite's hole filling by setting pre/post max hole area to 0%%.",
    )
    parser.add_argument(
        "--geogram-postprocess-degree3",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow Vorpalite's postprocessing to remove degree-3 vertices near each other.",
    )
    parser.add_argument(
        "--geogram-density-path",
        type=Path,
        default=None,
        help="Optional precomputed omega_geogram_density.txt sidecar for geogram_omega_density methods.",
    )
    parser.add_argument(
        "--geogram-density-min",
        type=float,
        default=0.25,
        help="Minimum exported OMeGa density. High-planar vertices approach this value.",
    )
    parser.add_argument(
        "--geogram-density-max",
        type=float,
        default=4.0,
        help="Maximum exported OMeGa density. Low-planar/detail vertices approach this value.",
    )
    parser.add_argument(
        "--geogram-density-signal-gamma",
        type=float,
        default=1.25,
        help="Gamma on the normalized OMeGa density signal before mapping to density.",
    )
    parser.add_argument("--geogram-density-detail-influence", type=float, default=1.0)
    parser.add_argument("--geogram-density-boundary-influence", type=float, default=0.5)
    parser.add_argument(
        "--geogram-density-uncertain-influence",
        type=float,
        default=0.0,
        help="Uncertainty contribution to density. Default 0 keeps uncertainty from driving density.",
    )
    parser.add_argument(
        "--geogram-auto-from-normal-error-degrees",
        type=float,
        default=0.0,
        help="Enable h(x)-driven OMeGa density from StableNormal evidence using this tolerated normal error.",
    )
    parser.add_argument(
        "--geogram-auto-target-points",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the h(x)-estimated point count for OMeGa-density Geogram methods.",
    )
    parser.add_argument("--geogram-auto-h-min-factor", type=float, default=0.50)
    parser.add_argument("--geogram-auto-h-max-factor", type=float, default=6.0)
    parser.add_argument("--geogram-auto-target-point-scale", type=float, default=1.0)
    parser.add_argument("--geogram-use-fused-normals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--geogram-fused-normal-blend", type=float, default=0.70)
    parser.add_argument("--geogram-fused-normal-min-weight", type=float, default=8.0)
    parser.add_argument(
        "--geogram-density-runtime-scale",
        type=float,
        default=1.0,
        help="Vorpalite-side scale applied after reading the OMeGa density sidecar.",
    )
    parser.add_argument("--geogram-density-runtime-min", type=float, default=1e-6)
    parser.add_argument(
        "--geogram-density-runtime-max",
        type=float,
        default=0.0,
        help="Vorpalite-side max clamp. 0 disables the upper clamp.",
    )
    parser.add_argument(
        "--geogram-density-planar-gradation-iterations",
        type=int,
        default=12,
        help="Iterations for the OMeGa planar-only density gradation sidecar.",
    )
    parser.add_argument("--geogram-density-planar-gradation-strength", type=float, default=0.45)
    parser.add_argument("--geogram-density-planar-gradation-planar-power", type=float, default=2.0)
    parser.add_argument("--geogram-density-planar-gradation-min-edge-weight", type=float, default=0.05)
    parser.add_argument(
        "--geogram-density-planar-gradation-max-density-ratio",
        type=float,
        default=1.75,
        help="Per-vertex clamp for planar gradation. 1.75 means density can change by at most 1.75x.",
    )
    parser.add_argument(
        "--geogram-density-planar-gradation-normal-lift-scale",
        type=float,
        default=0.0,
        help="Normal-lift scale used by planar gradation. 0 inherits --geogram-anisotropy.",
    )
    parser.add_argument(
        "--geogram-density-planar-gradation-metric-tau-factor",
        type=float,
        default=2.0,
        help="Allowed lifted-normal displacement measured in local edge lengths.",
    )
    parser.add_argument("--geogram-density-planar-gradation-min-normal-gate", type=float, default=0.05)
    parser.add_argument(
        "--geogram-density-planar-gradation-use-proxy-boundaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop planar density gradation across proxy boundaries and different proxy ids.",
    )
    parser.add_argument("--qem-quality-threshold", type=float, default=0.3)
    parser.add_argument("--qem-boundary-weight", type=float, default=1.0)
    parser.add_argument("--planar-quadric-weight", type=float, default=0.01)
    parser.add_argument("--preserve-boundary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-topology", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-normal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        methods = _parse_methods(args.methods, args.method_preset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    compute_remesh_baselines(
        BaselineConfig(
            model_dir=args.model_dir,
            mesh_path=args.mesh,
            output_dir=args.output_dir,
            experiment_name=str(args.experiment_name),
            methods=methods,
            target_face_ratios=_parse_float_list(args.target_face_ratios),
            target_faces=_parse_int_list(args.target_faces),
            iteration=int(args.iteration),
            isotropic_iterations=int(args.isotropic_iterations),
            isotropic_feature_degrees=float(args.isotropic_feature_degrees),
            isotropic_target_scale=float(args.isotropic_target_scale),
            isotropic_target_edge_length=float(args.isotropic_target_edge_length),
            isotropic_target_percent=float(args.isotropic_target_percent),
            isotropic_max_surface_dist_percent=float(args.isotropic_max_surface_dist_percent),
            feature_isotropic_degrees=float(args.feature_isotropic_degrees),
            feature_isotropic_smooth=bool(args.feature_isotropic_smooth),
            instant_meshes_bin=args.instant_meshes_bin,
            instant_meshes_crease_degrees=float(args.instant_meshes_crease_degrees),
            instant_meshes_smooth_iterations=int(args.instant_meshes_smooth_iterations),
            instant_meshes_threads=int(args.instant_meshes_threads),
            instant_meshes_deterministic=bool(args.instant_meshes_deterministic),
            instant_meshes_align_boundaries=bool(args.instant_meshes_align_boundaries),
            instant_meshes_dump_fields=bool(args.instant_meshes_dump_fields),
            instant_meshes_feature_constraints=bool(args.instant_meshes_feature_constraints),
            instant_meshes_feature_constraint_weight=float(args.instant_meshes_feature_constraint_weight),
            instant_meshes_feature_constraint_angle=float(args.instant_meshes_feature_constraint_angle),
            instant_guided_feature_constraints=bool(args.instant_guided_feature_constraints),
            instant_field_path=args.instant_field_path,
            instant_field_weight_scale=float(args.instant_field_weight_scale),
            instant_field_frame_stride=int(args.instant_field_frame_stride),
            instant_field_max_frames=int(args.instant_field_max_frames),
            instant_field_pixel_stride=int(args.instant_field_pixel_stride),
            instant_field_min_direction_confidence=float(args.instant_field_min_direction_confidence),
            instant_field_min_gradient_normalized=float(args.instant_field_min_gradient_normalized),
            instant_field_gradient_mode=str(args.instant_field_gradient_mode),
            instant_field_weight_quantile=float(args.instant_field_weight_quantile),
            instant_field_min_vertex_weight=float(args.instant_field_min_vertex_weight),
            instant_guided_no_subdivide=bool(args.instant_guided_no_subdivide),
            geogram_bin=args.geogram_bin,
            geogram_anisotropy=float(args.geogram_anisotropy),
            geogram_gradation=float(args.geogram_gradation),
            geogram_normal_smooth_iterations=int(args.geogram_normal_smooth_iterations),
            geogram_lfs_samples=int(args.geogram_lfs_samples),
            geogram_lloyd_iterations=int(args.geogram_lloyd_iterations),
            geogram_newton_iterations=int(args.geogram_newton_iterations),
            geogram_newton_m=int(args.geogram_newton_m),
            geogram_threads=int(args.geogram_threads),
            geogram_preserve_components=bool(args.geogram_preserve_components),
            geogram_preserve_holes=bool(args.geogram_preserve_holes),
            geogram_postprocess_degree3=bool(args.geogram_postprocess_degree3),
            geogram_density_path=args.geogram_density_path,
            geogram_density_min=float(args.geogram_density_min),
            geogram_density_max=float(args.geogram_density_max),
            geogram_density_signal_gamma=float(args.geogram_density_signal_gamma),
            geogram_density_detail_influence=float(args.geogram_density_detail_influence),
            geogram_density_boundary_influence=float(args.geogram_density_boundary_influence),
            geogram_density_uncertain_influence=float(args.geogram_density_uncertain_influence),
            geogram_auto_from_normal_error_degrees=float(args.geogram_auto_from_normal_error_degrees),
            geogram_auto_target_points=bool(args.geogram_auto_target_points),
            geogram_auto_h_min_factor=float(args.geogram_auto_h_min_factor),
            geogram_auto_h_max_factor=float(args.geogram_auto_h_max_factor),
            geogram_auto_target_point_scale=float(args.geogram_auto_target_point_scale),
            geogram_use_fused_normals=bool(args.geogram_use_fused_normals),
            geogram_fused_normal_blend=float(args.geogram_fused_normal_blend),
            geogram_fused_normal_min_weight=float(args.geogram_fused_normal_min_weight),
            geogram_density_runtime_scale=float(args.geogram_density_runtime_scale),
            geogram_density_runtime_min=float(args.geogram_density_runtime_min),
            geogram_density_runtime_max=float(args.geogram_density_runtime_max),
            geogram_density_planar_gradation_iterations=int(args.geogram_density_planar_gradation_iterations),
            geogram_density_planar_gradation_strength=float(args.geogram_density_planar_gradation_strength),
            geogram_density_planar_gradation_planar_power=float(args.geogram_density_planar_gradation_planar_power),
            geogram_density_planar_gradation_min_edge_weight=float(args.geogram_density_planar_gradation_min_edge_weight),
            geogram_density_planar_gradation_max_density_ratio=float(args.geogram_density_planar_gradation_max_density_ratio),
            geogram_density_planar_gradation_normal_lift_scale=float(args.geogram_density_planar_gradation_normal_lift_scale),
            geogram_density_planar_gradation_metric_tau_factor=float(args.geogram_density_planar_gradation_metric_tau_factor),
            geogram_density_planar_gradation_min_normal_gate=float(args.geogram_density_planar_gradation_min_normal_gate),
            geogram_density_planar_gradation_use_proxy_boundaries=bool(args.geogram_density_planar_gradation_use_proxy_boundaries),
            qem_quality_threshold=float(args.qem_quality_threshold),
            qem_boundary_weight=float(args.qem_boundary_weight),
            planar_quadric_weight=float(args.planar_quadric_weight),
            preserve_boundary=bool(args.preserve_boundary),
            preserve_topology=bool(args.preserve_topology),
            preserve_normal=bool(args.preserve_normal),
            overwrite=bool(args.overwrite),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
