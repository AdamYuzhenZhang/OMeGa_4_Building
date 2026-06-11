"""Stage-2 minimal remesh policy from projected normal evidence.

Stage 1 projects OMeGa-selected normal-map cues onto mesh faces. Stage 2 now
keeps only the policy we actually need:

- plane-like faces: simplify aggressively and allow planar cleanup;
- detail / non-planar faces: simplify conservatively and preserve triangles.

RGB, Guide01 scores, target edge-length fields, and hard semantic region labels
are not used by the remesh policy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from . import evidence as evidence_utils


ProgressFn = Callable[[str], None]


REGION_TYPE_NAMES = {
    0: "uncertain",
    1: "planar",
    2: "detail",
    3: "boundary",
    4: "mixed",
}
REGION_TYPE_CODES = {name: code for code, name in REGION_TYPE_NAMES.items()}


@dataclass
class MeshRegionScoreConfig:
    mesh_path: Path
    evidence_npz: Path
    out_dir: Path
    scores_name: str = "mesh_region_scores"
    planar_threshold: float = 0.60
    detail_threshold: float = 0.45
    connect_boundary_threshold: float = 0.45
    score_smooth_iterations: int = 0
    region_detail_quantile: float = 0.85
    region_boundary_quantile: float = 0.90
    curvature_low_quantile: float = 0.25
    curvature_high_quantile: float = 0.90
    variation_low_quantile: float = 0.50
    variation_high_quantile: float = 0.95
    variation_detail_weight: float = 0.35
    planarity_variation_penalty: float = 0.35
    mesh_crease_detail_weight: float = 0.35
    min_region_faces: int = 8
    min_normal_samples: int = 2
    target_view_support: float = 0.05
    mesh_crease_low_deg: float = 25.0
    mesh_crease_high_deg: float = 80.0
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass
class MeshRegionScoreResult:
    scores_npz: Path
    regions_json: Path
    summary_json: Path
    debug_meshes: dict[str, Path]
    face_count: int
    region_count: int


@dataclass
class FaceTopology:
    adjacent_pairs: np.ndarray
    pair_edge_vertices: np.ndarray
    boundary_faces: np.ndarray
    nonmanifold_faces: np.ndarray
    boundary_edge_count: int
    nonmanifold_edge_group_count: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path.expanduser().resolve()
    cwd_candidate = (Path.cwd() / path).expanduser().resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    candidate = (base / path).expanduser().resolve()
    if candidate.exists():
        return candidate
    repo_candidate = (evidence_utils._repo_root() / path).expanduser().resolve()  # noqa: SLF001
    if repo_candidate.exists():
        return repo_candidate
    return candidate


def _summary_path_for_evidence(evidence_npz: Path) -> Path:
    return evidence_npz.with_name(f"{evidence_npz.stem}_summary.json")


def resolve_evidence_npz(model_dir: Path, requested: Path | None = None) -> Path:
    model_dir = model_dir.resolve()
    if requested is not None:
        path = _resolve_path(requested, model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Requested evidence npz does not exist: {path}")
        return path
    default = model_dir / "remesh" / "evidence" / "mesh_face_evidence.npz"
    if default.exists():
        return default
    raise FileNotFoundError(f"Missing evidence npz: {default}. Run run_omega_mesh_evidence.py first.")


def resolve_mesh_for_scores(model_dir: Path, requested: Path | None, evidence_npz: Path, iteration: int) -> Path:
    if requested is not None:
        path = _resolve_path(requested, model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Requested mesh does not exist: {path}")
        return path
    summary_path = _summary_path_for_evidence(evidence_npz)
    if summary_path.exists():
        summary = _read_json(summary_path)
        mesh_path = summary.get("mesh", {}).get("path")
        if mesh_path:
            path = Path(str(mesh_path)).expanduser().resolve()
            if path.exists():
                return path
    return evidence_utils.resolve_latest_omega_mesh(model_dir, None, int(iteration))


def _load_evidence(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    evidence = {key: data[key] for key in data.files}
    required = [
        "mesh_face_normals",
        "face_area",
        "view_count",
        "normal_count",
        "mean_valid_support",
        "mean_normal_variation",
        "max_normal_variation",
        "mean_normal_gradient",
        "max_normal_gradient",
        "mean_normal_curvature",
        "max_normal_curvature",
    ]
    missing = [key for key in required if key not in evidence]
    if missing:
        raise KeyError(
            "Evidence file is missing required Stage-1 normal arrays: "
            f"{missing}. Re-run scripts/run_omega_mesh_evidence.py before Stage 2."
        )
    return evidence


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors, dtype=np.float64))


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    raw = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw, axis=1)
    normals = np.divide(raw, np.maximum(double_area[:, None], 1e-12), out=np.zeros_like(raw, dtype=np.float64))
    centers = np.mean(triangles, axis=1)
    areas = double_area * 0.5
    return centers.astype(np.float64), normals.astype(np.float64), areas.astype(np.float64)


def _face_topology(faces: np.ndarray) -> FaceTopology:
    face_count = int(faces.shape[0])
    edge_dtype = np.int32 if int(np.max(faces)) <= np.iinfo(np.int32).max else np.int64
    face_dtype = np.int32 if face_count <= np.iinfo(np.int32).max else np.int64
    faces_compact = np.asarray(faces, dtype=edge_dtype)
    edges = np.concatenate([faces_compact[:, [0, 1]], faces_compact[:, [1, 2]], faces_compact[:, [2, 0]]], axis=0)
    edge_faces = np.tile(np.arange(face_count, dtype=face_dtype), 3)
    sorted_edges = np.sort(edges, axis=1)
    order = np.lexsort((sorted_edges[:, 1], sorted_edges[:, 0]))
    sorted_edges = sorted_edges[order]
    sorted_face_ids = edge_faces[order]
    changed = np.ones((len(sorted_edges),), dtype=bool)
    changed[1:] = np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1)
    starts = np.flatnonzero(changed)
    ends = np.concatenate([starts[1:], np.asarray([len(sorted_edges)], dtype=np.int64)])
    counts = ends - starts
    boundary_starts = starts[counts == 1]
    boundary_faces = sorted_face_ids[boundary_starts].astype(face_dtype, copy=False)
    pair_starts = starts[counts == 2]
    if pair_starts.size:
        adjacent_pairs = np.column_stack([sorted_face_ids[pair_starts], sorted_face_ids[pair_starts + 1]]).astype(face_dtype, copy=False)
        pair_edge_vertices = sorted_edges[pair_starts].astype(edge_dtype, copy=False)
    else:
        adjacent_pairs = np.zeros((0, 2), dtype=np.int64)
        pair_edge_vertices = np.zeros((0, 2), dtype=np.int64)
    nonmanifold_mask = counts > 2
    if np.any(nonmanifold_mask):
        nonmanifold_faces = np.unique(
            np.concatenate([
                sorted_face_ids[start:end]
                for start, end in zip(starts[nonmanifold_mask], ends[nonmanifold_mask])
            ])
        ).astype(face_dtype, copy=False)
    else:
        nonmanifold_faces = np.zeros((0,), dtype=np.int64)
    return FaceTopology(
        adjacent_pairs=adjacent_pairs,
        pair_edge_vertices=pair_edge_vertices,
        boundary_faces=boundary_faces,
        nonmanifold_faces=nonmanifold_faces,
        boundary_edge_count=int(boundary_starts.size),
        nonmanifold_edge_group_count=int(np.count_nonzero(nonmanifold_mask)),
    )


def _smoothstep01(values: np.ndarray) -> np.ndarray:
    x = np.clip(values, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _edge_normal_variation(normals: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros((0,), dtype=np.float32)
    a = normals[pairs[:, 0]]
    b = normals[pairs[:, 1]]
    dot = np.abs(np.sum(a * b, axis=1)).clip(0.0, 1.0)
    angles = np.arccos(dot)
    return np.clip(angles / (math.pi * 0.5), 0.0, 1.0).astype(np.float32)


def _count_confidence(counts: np.ndarray, minimum_samples: int) -> np.ndarray:
    minimum = max(int(minimum_samples), 1)
    return np.clip(np.asarray(counts, dtype=np.float32) / float(minimum), 0.0, 1.0)


def _summarize(values: np.ndarray, valid: np.ndarray | None = None) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if valid is not None:
        arr = arr[np.asarray(valid, dtype=bool).reshape(-1)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0.0, "min": 0.0, "q25": 0.0, "median": 0.0, "mean": 0.0, "q75": 0.0, "q90": 0.0, "q99": 0.0, "max": 0.0}
    qs = np.quantile(arr, [0.25, 0.5, 0.75, 0.9, 0.99])
    return {
        "count": float(arr.size),
        "min": float(np.min(arr)),
        "q25": float(qs[0]),
        "median": float(qs[1]),
        "mean": float(np.mean(arr)),
        "q75": float(qs[2]),
        "q90": float(qs[3]),
        "q99": float(qs[4]),
        "max": float(np.max(arr)),
    }


def _normalize_signal_by_quantiles(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    low_quantile: float,
    high_quantile: float,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float32)
    valid_mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    finite = values[valid_mask]
    low_q = float(np.clip(low_quantile, 0.0, 1.0))
    high_q = float(np.clip(high_quantile, 0.0, 1.0))
    if high_q <= low_q:
        high_q = min(1.0, low_q + 0.01)
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32), {
            "lowQuantile": low_q,
            "highQuantile": high_q,
            "lowValue": 0.0,
            "highValue": 1.0,
            "validCount": 0.0,
        }
    low_value = float(np.quantile(finite, low_q))
    high_value = float(np.quantile(finite, high_q))
    if high_value <= low_value + 1e-6:
        low_value = float(np.min(finite))
        high_value = float(np.max(finite))
    if high_value <= low_value + 1e-6:
        score = np.zeros_like(values, dtype=np.float32)
    else:
        score = _smoothstep01((values - low_value) / (high_value - low_value)).astype(np.float32)
    return np.where(valid_mask, score, 0.0).astype(np.float32), {
        "lowQuantile": low_q,
        "highQuantile": high_q,
        "lowValue": low_value,
        "highValue": high_value,
        "validCount": float(finite.size),
    }


def _compute_region_ids(
    *,
    adjacent_pairs: np.ndarray,
    edge_boundary_score: np.ndarray,
    connect_boundary_threshold: float,
    face_count: int,
) -> np.ndarray:
    face_count = int(face_count)
    if face_count == 0:
        return np.zeros((0,), dtype=np.int32)
    if len(adjacent_pairs) == 0:
        return np.arange(face_count, dtype=np.int32)
    connect = edge_boundary_score < float(connect_boundary_threshold)
    if not np.any(connect):
        return np.arange(face_count, dtype=np.int32)
    rows = adjacent_pairs[connect, 0].astype(np.int64, copy=False)
    cols = adjacent_pairs[connect, 1].astype(np.int64, copy=False)
    graph_rows = np.concatenate([rows, cols])
    graph_cols = np.concatenate([cols, rows])
    graph = sparse.coo_matrix(
        (np.ones((graph_rows.size,), dtype=bool), (graph_rows, graph_cols)),
        shape=(face_count, face_count),
        dtype=bool,
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    return labels.astype(np.int32, copy=False)


def _propagate_max(values: np.ndarray, pairs: np.ndarray, iterations: int, decay: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    if len(pairs) == 0 or int(iterations) <= 0:
        return out
    for _ in range(int(iterations)):
        neighbor = np.zeros_like(out)
        np.maximum.at(neighbor, pairs[:, 0], out[pairs[:, 1]] * float(decay))
        np.maximum.at(neighbor, pairs[:, 1], out[pairs[:, 0]] * float(decay))
        out = np.maximum(out, neighbor)
    return out.astype(np.float32)


def _region_statistics(
    *,
    face_region_id: np.ndarray,
    face_area: np.ndarray,
    face_centers: np.ndarray,
    scores: dict[str, np.ndarray],
    min_region_faces: int,
    detail_quantile: float,
    boundary_quantile: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    region_count = int(np.max(face_region_id)) + 1 if face_region_id.size else 0
    order = np.argsort(face_region_id, kind="stable")
    sorted_region_id = face_region_id[order]
    changed = np.ones((len(order),), dtype=bool)
    changed[1:] = sorted_region_id[1:] != sorted_region_id[:-1]
    starts = np.flatnonzero(changed)
    ends = np.concatenate([starts[1:], np.asarray([len(order)], dtype=np.int64)])
    region_face_count = np.zeros((region_count,), dtype=np.int32)
    region_area = np.zeros((region_count,), dtype=np.float32)
    region_centroid = np.zeros((region_count, 3), dtype=np.float32)
    region_small = np.zeros((region_count,), dtype=bool)
    region_scores = {name: np.zeros((region_count,), dtype=np.float32) for name in scores}
    regions: list[dict[str, Any]] = []
    high_quantile_scores = {"face_detail_weight", "face_protect_weight", "face_type_detail_score", "face_type_boundary_score", "face_edge_weight"}
    for start, end in zip(starts, ends):
        ids = order[start:end]
        rid = int(sorted_region_id[start])
        areas = np.asarray(face_area[ids], dtype=np.float64)
        area_sum = float(np.sum(areas))
        if area_sum > 1e-12:
            centroid = np.sum(face_centers[ids] * areas[:, None], axis=0) / area_sum
        else:
            centroid = np.mean(face_centers[ids], axis=0)
        region_face_count[rid] = int(ids.size)
        region_area[rid] = float(area_sum)
        region_centroid[rid] = centroid.astype(np.float32)
        region_small[rid] = bool(ids.size < int(min_region_faces))
        row: dict[str, Any] = {
            "id": rid,
            "faceCount": int(ids.size),
            "surfaceArea": float(area_sum),
            "centroid": [float(v) for v in centroid],
            "isSmallComponent": bool(region_small[rid]),
        }
        for name, values in scores.items():
            local = np.asarray(values[ids], dtype=np.float32)
            if local.size == 0:
                agg_value = mean_value = max_value = 0.0
            else:
                mean_value = float(np.mean(local))
                max_value = float(np.max(local))
                if name == "face_edge_weight":
                    agg_value = float(np.quantile(local, float(boundary_quantile)))
                elif name in high_quantile_scores:
                    agg_value = float(np.quantile(local, float(detail_quantile)))
                else:
                    agg_value = mean_value
            region_scores[name][rid] = agg_value
            short = name.removeprefix("face_")
            row[short] = agg_value
            row[f"{short}Mean"] = mean_value
            row[f"{short}Max"] = max_value
        regions.append(row)
    arrays: dict[str, np.ndarray] = {
        "region_face_count": region_face_count,
        "region_area": region_area,
        "region_centroid": region_centroid,
        "region_small": region_small,
    }
    for name, values in region_scores.items():
        short = name.removeprefix("face_")
        arrays[f"region_{short}"] = values
        arrays[f"face_region_{short}"] = values[face_region_id]
    arrays["face_small_region"] = region_small[face_region_id]
    return arrays, regions


def _colorize_scalar(values: np.ndarray, valid: np.ndarray | None = None, *, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid_mask = np.isfinite(values) if valid is None else np.asarray(valid, dtype=bool) & np.isfinite(values)
    normalized = np.clip((values - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    mapped = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and values.ndim == 1 and rgb.shape[1] == 1:
        rgb = rgb[:, 0, :]
    rgb[~valid_mask] = np.array([36, 36, 36], dtype=np.uint8)
    return rgb


def _region_id_colors(region_id: np.ndarray) -> np.ndarray:
    ids = np.asarray(region_id, dtype=np.uint32)
    x = ids * np.uint32(1664525) + np.uint32(1013904223)
    colors = np.stack(
        [
            ((x >> np.uint32(16)) & np.uint32(255)).astype(np.uint8),
            ((x >> np.uint32(8)) & np.uint32(255)).astype(np.uint8),
            (x & np.uint32(255)).astype(np.uint8),
        ],
        axis=1,
    )
    colors[region_id < 0] = np.array([36, 36, 36], dtype=np.uint8)
    return colors


def _region_type_colors(region_type: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            [120, 120, 120],
            [56, 190, 130],
            [242, 166, 56],
            [220, 60, 54],
            [86, 150, 230],
        ],
        dtype=np.uint8,
    )
    clipped = np.clip(np.asarray(region_type, dtype=np.int32), 0, len(palette) - 1)
    return palette[clipped]


def _write_face_color_ply(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate([np.asarray(colors, dtype=np.uint8), np.full((faces.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def write_debug_meshes(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    arrays: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    scalar_keys = [
        "face_planar_weight",
        "face_detail_weight",
        "face_edge_weight",
        "face_uncertainty",
        "face_simplify_weight",
        "face_protect_weight",
        "face_plane_project_weight",
        "face_support_confidence",
        "face_normal_curvature_score",
        "face_variation_score",
        "face_normal_gradient",
        "face_normal_variation",
        "face_normal_curvature",
        "face_type_planar_score",
        "face_type_detail_score",
        "face_type_boundary_score",
        "face_type_uncertain_score",
        "face_type_mixed_score",
        "face_region_planar_weight",
        "face_region_detail_weight",
        "face_region_simplify_weight",
        "face_region_protect_weight",
    ]
    for key in scalar_keys:
        if key not in arrays:
            continue
        out_path = debug_dir / f"{key}.ply"
        colors = _colorize_scalar(arrays[key], vmin=0.0, vmax=1.0)
        _write_face_color_ply(vertices, faces, colors, out_path)
        paths[key] = out_path
    if "face_region_id" in arrays:
        region_id_path = debug_dir / "face_region_id.ply"
        _write_face_color_ply(vertices, faces, _region_id_colors(arrays["face_region_id"]), region_id_path)
        paths["face_region_id"] = region_id_path
    if "face_region_type" in arrays:
        region_type_path = debug_dir / "face_region_type.ply"
        _write_face_color_ply(vertices, faces, _region_type_colors(arrays["face_region_type"]), region_type_path)
        paths["face_region_type"] = region_type_path
    return paths


def _evidence_signal(evidence: dict[str, np.ndarray], preferred: str, fallback: str, face_count: int) -> np.ndarray:
    key = preferred if preferred in evidence else fallback
    values = np.asarray(evidence[key], dtype=np.float32).reshape(-1)
    if values.shape[0] != face_count:
        raise ValueError(f"Evidence array {key!r} has {values.shape[0]} values, expected {face_count}.")
    return values


def compute_mesh_region_scores(
    config: MeshRegionScoreConfig,
    progress: ProgressFn | None = None,
) -> MeshRegionScoreResult:
    progress = progress or _progress_default
    mesh_path = config.mesh_path.resolve()
    evidence_npz = config.evidence_npz.resolve()
    out_dir = config.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_npz = out_dir / f"{config.scores_name}.npz"
    regions_json = out_dir / "mesh_regions.json"
    summary_json = out_dir / f"{config.scores_name}_summary.json"
    if (scores_npz.exists() or regions_json.exists()) and not config.overwrite:
        raise FileExistsError(f"Region score outputs already exist under {out_dir}. Re-run with --overwrite.")

    vertices, faces = evidence_utils.load_mesh_arrays(mesh_path)
    faces = np.asarray(faces, dtype=np.int64)
    face_count = int(faces.shape[0])
    face_centers, mesh_normals, computed_area = _face_geometry(vertices, faces)
    evidence = _load_evidence(evidence_npz)
    if evidence["view_count"].shape[0] != face_count:
        raise ValueError(f"Evidence face count ({evidence['view_count'].shape[0]}) does not match mesh face count ({face_count}).")

    face_area = np.asarray(evidence.get("face_area", computed_area), dtype=np.float32)
    if face_area.shape[0] != face_count:
        face_area = computed_area.astype(np.float32)
    stored_normals = np.asarray(evidence.get("mesh_face_normals", mesh_normals), dtype=np.float64)
    if stored_normals.shape == mesh_normals.shape:
        mesh_normals = _safe_normalize(stored_normals)

    progress(f"[regions] Mesh: {mesh_path} ({vertices.shape[0]:,} vertices, {face_count:,} faces)")
    progress(f"[regions] Evidence: {evidence_npz}")
    progress("[regions] Policy: minimal two-region remesh weights: planar simplify/snap vs detail protect")

    topology = _face_topology(faces)
    pairs = topology.adjacent_pairs.astype(np.int64, copy=False)
    pair_mesh_variation = _edge_normal_variation(mesh_normals, pairs)
    low = math.radians(float(config.mesh_crease_low_deg)) / (math.pi * 0.5)
    high = math.radians(float(config.mesh_crease_high_deg)) / (math.pi * 0.5)
    pair_mesh_crease = _smoothstep01((pair_mesh_variation - low) / max(high - low, 1e-6)).astype(np.float32)
    face_mesh_crease = np.zeros((face_count,), dtype=np.float32)
    if len(pairs):
        np.maximum.at(face_mesh_crease, pairs[:, 0], pair_mesh_crease)
        np.maximum.at(face_mesh_crease, pairs[:, 1], pair_mesh_crease)

    normal_count = np.asarray(evidence["normal_count"], dtype=np.int32)
    view_count = np.asarray(evidence["view_count"], dtype=np.int32)
    normal_valid = normal_count > 0
    support_normal = _count_confidence(normal_count, int(config.min_normal_samples))
    support_view = np.clip(
        np.asarray(evidence["mean_valid_support"], dtype=np.float32) / max(float(config.target_view_support), 1e-6),
        0.0,
        1.0,
    )
    face_support_confidence = np.clip(0.85 * support_normal + 0.15 * support_view, 0.0, 1.0).astype(np.float32)

    normal_variation = _evidence_signal(evidence, "normal_variation_q75", "mean_normal_variation", face_count).clip(0.0, 1.0)
    normal_curvature = _evidence_signal(evidence, "normal_curvature_q75", "mean_normal_curvature", face_count)
    normal_gradient = _evidence_signal(evidence, "mean_normal_gradient", "mean_normal_gradient", face_count).clip(0.0, 1.0)
    normal_variation = np.where(normal_valid, normal_variation, 0.0).astype(np.float32)
    normal_curvature = np.where(normal_valid, normal_curvature, 0.0).astype(np.float32)
    normal_gradient = np.where(normal_valid, normal_gradient, 0.0).astype(np.float32)

    face_normal_curvature_score, curvature_normalization = _normalize_signal_by_quantiles(
        normal_curvature,
        normal_valid,
        low_quantile=float(config.curvature_low_quantile),
        high_quantile=float(config.curvature_high_quantile),
    )
    face_variation_score, variation_normalization = _normalize_signal_by_quantiles(
        normal_variation,
        normal_valid,
        low_quantile=float(config.variation_low_quantile),
        high_quantile=float(config.variation_high_quantile),
    )
    face_gradient_score, gradient_normalization = _normalize_signal_by_quantiles(
        normal_gradient,
        normal_valid,
        low_quantile=float(config.curvature_low_quantile),
        high_quantile=float(config.curvature_high_quantile),
    )

    detail_signal = np.maximum.reduce(
        [
            face_normal_curvature_score,
            float(config.variation_detail_weight) * face_variation_score,
            0.50 * face_gradient_score,
            float(config.mesh_crease_detail_weight) * face_mesh_crease,
        ]
    ).astype(np.float32).clip(0.0, 1.0)
    face_detail_weight = (detail_signal * face_support_confidence).astype(np.float32).clip(0.0, 1.0)

    planarity_variation = np.clip(1.0 - float(config.planarity_variation_penalty) * face_variation_score, 0.0, 1.0)
    face_planar_weight = (
        face_support_confidence
        * (1.0 - face_normal_curvature_score)
        * planarity_variation
        * (1.0 - 0.50 * face_mesh_crease)
        * (1.0 - 0.50 * face_detail_weight)
    ).astype(np.float32).clip(0.0, 1.0)

    face_planarity_contrast = np.zeros((face_count,), dtype=np.float32)
    edge_boundary_score = np.zeros((len(pairs),), dtype=np.float32)
    pair_planarity_contrast = np.zeros((len(pairs),), dtype=np.float32)
    face_edge_weight = np.zeros((face_count,), dtype=np.float32)
    if len(pairs):
        pair_planarity_contrast = np.abs(face_planar_weight[pairs[:, 0]] - face_planar_weight[pairs[:, 1]]).astype(np.float32)
        pair_detail = np.maximum(face_detail_weight[pairs[:, 0]], face_detail_weight[pairs[:, 1]])
        edge_boundary_score = np.maximum.reduce(
            [
                pair_mesh_crease,
                pair_detail,
                _smoothstep01((pair_planarity_contrast - 0.20) / 0.55).astype(np.float32),
            ]
        ).astype(np.float32).clip(0.0, 1.0)
        np.maximum.at(face_planarity_contrast, pairs[:, 0], pair_planarity_contrast)
        np.maximum.at(face_planarity_contrast, pairs[:, 1], pair_planarity_contrast)
        np.maximum.at(face_edge_weight, pairs[:, 0], edge_boundary_score)
        np.maximum.at(face_edge_weight, pairs[:, 1], edge_boundary_score)

    smooth_iters = max(int(config.score_smooth_iterations), 0)
    face_edge_weight = _propagate_max(face_edge_weight, pairs, smooth_iters, 0.70)
    face_detail_weight = _propagate_max(np.maximum(face_detail_weight, 0.50 * face_edge_weight), pairs, smooth_iters, 0.60)

    face_uncertainty = np.maximum.reduce(
        [
            1.0 - face_support_confidence,
            np.where(normal_valid, 0.0, 1.0).astype(np.float32),
            np.where(view_count > 0, 0.0, 0.75).astype(np.float32),
        ]
    ).astype(np.float32).clip(0.0, 1.0)

    face_plane_project_weight = (
        face_planar_weight
        * (1.0 - face_detail_weight)
        * (1.0 - 0.70 * face_edge_weight)
        * (1.0 - 0.75 * face_uncertainty)
    ).astype(np.float32).clip(0.0, 1.0)
    face_simplify_weight = face_plane_project_weight.astype(np.float32)
    face_protect_weight = np.maximum.reduce(
        [
            face_detail_weight,
            0.70 * face_edge_weight,
            0.75 * face_uncertainty,
        ]
    ).astype(np.float32).clip(0.0, 1.0)

    face_type_planar_score = face_planar_weight.astype(np.float32)
    face_type_detail_score = np.maximum(face_detail_weight, 0.50 * face_edge_weight).astype(np.float32).clip(0.0, 1.0)
    face_type_boundary_score = face_edge_weight.astype(np.float32)
    face_type_uncertain_score = face_uncertainty.astype(np.float32)
    face_type_mixed_score = (1.0 - np.maximum.reduce([face_type_planar_score, face_type_detail_score, face_type_uncertain_score])).astype(np.float32).clip(0.0, 1.0)

    face_scores = {
        "face_planar_weight": face_planar_weight,
        "face_detail_weight": face_detail_weight,
        "face_edge_weight": face_edge_weight,
        "face_uncertainty": face_uncertainty,
        "face_simplify_weight": face_simplify_weight,
        "face_protect_weight": face_protect_weight,
        "face_plane_project_weight": face_plane_project_weight,
        "face_support_confidence": face_support_confidence.astype(np.float32),
        "face_normal_curvature": normal_curvature.astype(np.float32),
        "face_normal_variation": normal_variation.astype(np.float32),
        "face_normal_gradient": normal_gradient.astype(np.float32),
        "face_normal_curvature_score": face_normal_curvature_score.astype(np.float32),
        "face_variation_score": face_variation_score.astype(np.float32),
        "face_gradient_score": face_gradient_score.astype(np.float32),
        "face_detail_signal": detail_signal.astype(np.float32),
        "face_planarity_contrast": face_planarity_contrast.astype(np.float32),
        "face_planarity": face_planar_weight.astype(np.float32),
        "face_detail": face_detail_weight.astype(np.float32),
        "face_boundary": face_edge_weight.astype(np.float32),
        "face_type_planar_score": face_type_planar_score,
        "face_type_detail_score": face_type_detail_score,
        "face_type_boundary_score": face_type_boundary_score,
        "face_type_uncertain_score": face_type_uncertain_score,
        "face_type_mixed_score": face_type_mixed_score,
    }

    face_region_id = _compute_region_ids(
        adjacent_pairs=pairs,
        edge_boundary_score=edge_boundary_score,
        connect_boundary_threshold=float(config.connect_boundary_threshold),
        face_count=face_count,
    )
    region_arrays, regions = _region_statistics(
        face_region_id=face_region_id,
        face_area=face_area,
        face_centers=face_centers,
        scores=face_scores,
        min_region_faces=int(config.min_region_faces),
        detail_quantile=float(config.region_detail_quantile),
        boundary_quantile=float(config.region_boundary_quantile),
    )

    type_score_names = ["uncertain", "planar", "detail", "boundary", "mixed"]
    dominant_type_score_names = ["uncertain", "planar", "detail", "mixed"]
    region_count = int(region_arrays["region_face_count"].shape[0])
    if region_count:
        dominant_stack = np.stack([region_arrays[f"region_type_{name}_score"] for name in dominant_type_score_names], axis=1)
        region_type_score_stack = np.stack([region_arrays[f"region_type_{name}_score"] for name in type_score_names], axis=1)
        region_uncertain = region_arrays["region_type_uncertain_score"]
        region_planar = region_arrays["region_type_planar_score"]
        region_detail = region_arrays["region_type_detail_score"]
        region_type = np.full((region_count,), REGION_TYPE_CODES["mixed"], dtype=np.int16)
        region_type[region_uncertain >= 0.70] = REGION_TYPE_CODES["uncertain"]
        planar_mask = (region_planar >= float(config.planar_threshold)) & (region_planar >= region_detail) & (region_uncertain < 0.70)
        detail_mask = (region_detail >= float(config.detail_threshold)) & ~planar_mask & (region_uncertain < 0.70)
        region_type[planar_mask] = REGION_TYPE_CODES["planar"]
        region_type[detail_mask] = REGION_TYPE_CODES["detail"]
    else:
        region_type = np.zeros((0,), dtype=np.int16)
        dominant_stack = np.zeros((0, len(dominant_type_score_names)), dtype=np.float32)
        region_type_score_stack = np.zeros((0, len(type_score_names)), dtype=np.float32)
    face_region_type = region_type[face_region_id]
    region_arrays["region_type"] = region_type
    region_arrays["face_region_type"] = face_region_type
    region_arrays["dominant_type_score_stack"] = dominant_stack.astype(np.float32)
    region_arrays["region_type_score_stack"] = region_type_score_stack.astype(np.float32)
    for row in regions:
        rid = int(row["id"])
        type_code = int(region_type[rid])
        row["typeScores"] = {name: float(region_arrays[f"region_type_{name}_score"][rid]) for name in type_score_names}
        row["dominantTypeCode"] = type_code
        row["dominantTypeName"] = REGION_TYPE_NAMES.get(type_code, "unknown")
        row["typeCode"] = type_code
        row["typeName"] = REGION_TYPE_NAMES.get(type_code, "unknown")

    arrays: dict[str, np.ndarray] = {
        **face_scores,
        **region_arrays,
        "face_region_id": face_region_id,
        "face_region_type": face_region_type,
        "normal_sample_confidence": support_normal.astype(np.float32),
        "view_support_confidence": support_view.astype(np.float32),
        "edge_adjacent_pairs": pairs.astype(np.int32),
        "edge_boundary_score": edge_boundary_score.astype(np.float32),
        "edge_mesh_crease": pair_mesh_crease.astype(np.float32),
        "edge_planarity_contrast": pair_planarity_contrast.astype(np.float32),
    }
    np.savez_compressed(scores_npz, **arrays)

    debug_meshes: dict[str, Path] = {}
    if config.write_debug_meshes:
        debug_meshes = write_debug_meshes(vertices=vertices, faces=faces, arrays=arrays, out_dir=out_dir)

    region_type_counts = {REGION_TYPE_NAMES[type_code]: int(np.count_nonzero(face_region_type == type_code)) for type_code in sorted(REGION_TYPE_NAMES)}
    region_count_by_type = {REGION_TYPE_NAMES[type_code]: int(np.count_nonzero(region_type == type_code)) for type_code in sorted(REGION_TYPE_NAMES)}
    regions_payload = {
        "stageName": "omega_mesh_region_scores",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "regionTypeNames": {str(key): value for key, value in REGION_TYPE_NAMES.items()},
        "regions": regions,
    }
    _write_json(regions_json, regions_payload)

    summary = {
        "stageName": "omega_mesh_region_scores",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {**asdict(config), "mesh_path": str(mesh_path), "evidence_npz": str(evidence_npz), "out_dir": str(out_dir)},
        "policy": {
            "usesRgb": False,
            "usesGuide01": False,
            "normalDriven": True,
            "targetLengthIsDensityField": False,
            "notes": [
                "Only two remesh behaviors are modeled: plane-like regions simplify/snap; detail or non-planar regions protect.",
                "Plane-like weight is high when robust normal curvature and normal variation are low with enough normal-map support.",
                "Detail weight is high when robust normal curvature, variation, mesh crease, or local boundary evidence is high.",
                "face_plane_project_weight is the planar cleanup mask used by the QEM stage for fitted-plane projection.",
                "No target edge-length policy is used. Large and oblique triangles are allowed in confident planar regions.",
            ],
            "curvatureNormalization": curvature_normalization,
            "variationNormalization": variation_normalization,
            "gradientNormalization": gradient_normalization,
        },
        "inputs": {
            "meshPath": str(mesh_path),
            "evidenceNpz": str(evidence_npz),
            "evidenceSummary": str(_summary_path_for_evidence(evidence_npz)),
        },
        "mesh": {
            "vertexCount": int(vertices.shape[0]),
            "faceCount": face_count,
            "surfaceArea": float(np.sum(face_area)),
            "adjacentFacePairs": int(len(pairs)),
            "openBoundaryEdges": int(topology.boundary_edge_count),
            "nonmanifoldEdgeGroups": int(topology.nonmanifold_edge_group_count),
        },
        "coverage": {
            "viewFaceFraction": float(np.count_nonzero(view_count > 0) / max(face_count, 1)),
            "normalFaceFraction": float(np.count_nonzero(normal_valid) / max(face_count, 1)),
        },
        "scoreStats": {name: _summarize(values) for name, values in face_scores.items()},
        "regionScoreStats": {
            key: _summarize(value)
            for key, value in region_arrays.items()
            if key.startswith("region_") and value.ndim == 1 and value.dtype.kind in {"f", "i", "u", "b"}
        },
        "classification": {
            "dominantFaceTypeCounts": region_type_counts,
            "dominantRegionCountByType": region_count_by_type,
            "faceTypeCounts": region_type_counts,
            "regionCountByType": region_count_by_type,
            "regionCount": int(len(regions)),
            "smallRegionCount": int(np.count_nonzero(region_arrays["region_small"])),
            "minRegionFaces": int(config.min_region_faces),
            "typeScoreNames": type_score_names,
            "dominantTypeScoreNames": dominant_type_score_names,
        },
        "outputs": {
            "scoresNpz": str(scores_npz),
            "regionsJson": str(regions_json),
            "summaryJson": str(summary_json),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
    }
    _write_json(summary_json, summary)
    progress(f"[regions] Wrote {scores_npz}")
    progress(f"[regions] Wrote {regions_json}")
    progress(f"[regions] Wrote {summary_json}")
    progress(
        "[regions] Regions: {:,} | debug faces: planar={:,} detail={:,} mixed={:,} uncertain={:,}".format(
            len(regions),
            region_type_counts["planar"],
            region_type_counts["detail"],
            region_type_counts["mixed"],
            region_type_counts["uncertain"],
        )
    )
    return MeshRegionScoreResult(
        scores_npz=scores_npz,
        regions_json=regions_json,
        summary_json=summary_json,
        debug_meshes=debug_meshes,
        face_count=face_count,
        region_count=len(regions),
    )


def _config_from_args(args: argparse.Namespace) -> MeshRegionScoreConfig:
    model_dir = Path(args.model_dir).expanduser().resolve()
    evidence_npz = resolve_evidence_npz(model_dir, args.evidence_npz)
    mesh_path = resolve_mesh_for_scores(model_dir, args.mesh, evidence_npz, int(args.iteration))
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir is not None else evidence_npz.parent
    return MeshRegionScoreConfig(
        mesh_path=mesh_path,
        evidence_npz=evidence_npz,
        out_dir=out_dir,
        scores_name=str(args.scores_name),
        planar_threshold=float(args.planar_threshold),
        detail_threshold=float(args.detail_threshold),
        connect_boundary_threshold=float(args.connect_boundary_threshold),
        score_smooth_iterations=int(args.score_smooth_iterations),
        region_detail_quantile=float(args.region_detail_quantile),
        region_boundary_quantile=float(args.region_boundary_quantile),
        curvature_low_quantile=float(args.curvature_low_quantile),
        curvature_high_quantile=float(args.curvature_high_quantile),
        variation_low_quantile=float(args.variation_low_quantile),
        variation_high_quantile=float(args.variation_high_quantile),
        variation_detail_weight=float(args.variation_detail_weight),
        planarity_variation_penalty=float(args.planarity_variation_penalty),
        mesh_crease_detail_weight=float(args.mesh_crease_detail_weight),
        min_region_faces=int(args.min_region_faces),
        min_normal_samples=int(args.min_normal_samples),
        target_view_support=float(args.target_view_support),
        mesh_crease_low_deg=float(args.mesh_crease_low_deg),
        mesh_crease_high_deg=float(args.mesh_crease_high_deg),
        write_debug_meshes=bool(args.debug_meshes),
        overwrite=bool(args.overwrite),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert OMeGa mesh evidence into minimal planar/detail remesh policy scores.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Mesh used by evidence projection. Defaults from evidence summary.")
    parser.add_argument("--iteration", type=int, default=-1, help="Mesh iteration to use when --mesh is omitted.")
    parser.add_argument("--evidence-npz", type=Path, default=None, help="Defaults to <model-dir>/remesh/evidence/mesh_face_evidence.npz.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to the evidence npz directory.")
    parser.add_argument("--scores-name", default="mesh_region_scores")
    parser.add_argument("--planar-threshold", type=float, default=0.60, help="Debug threshold for treating a face/region as plane-like.")
    parser.add_argument("--detail-threshold", type=float, default=0.45, help="Debug threshold for treating a face/region as detail/non-planar.")
    parser.add_argument("--connect-boundary-threshold", type=float, default=0.45, help="Connected-component split threshold from detail/crease contrast.")
    parser.add_argument("--score-smooth-iterations", type=int, default=0, help="Optional neighbor max-propagation iterations for detail/edge scores.")
    parser.add_argument("--region-detail-quantile", type=float, default=0.85)
    parser.add_argument("--region-boundary-quantile", type=float, default=0.90)
    parser.add_argument("--curvature-low-quantile", type=float, default=0.25)
    parser.add_argument("--curvature-high-quantile", type=float, default=0.90)
    parser.add_argument("--variation-low-quantile", type=float, default=0.50)
    parser.add_argument("--variation-high-quantile", type=float, default=0.95)
    parser.add_argument("--variation-detail-weight", type=float, default=0.35)
    parser.add_argument("--planarity-variation-penalty", type=float, default=0.35)
    parser.add_argument("--mesh-crease-detail-weight", type=float, default=0.35)
    parser.add_argument("--min-region-faces", type=int, default=8)
    parser.add_argument("--min-normal-samples", type=int, default=2)
    parser.add_argument("--target-view-support", type=float, default=0.05)
    parser.add_argument("--mesh-crease-low-deg", type=float, default=25.0)
    parser.add_argument("--mesh-crease-high-deg", type=float, default=80.0)
    parser.add_argument("--debug-meshes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    compute_mesh_region_scores(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
