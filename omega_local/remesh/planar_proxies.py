"""Planar proxy extraction for Phase 3 OMeGa local remeshing.

This module consumes continuous policy fields and groups confident planar faces
across low-boundary edges.  It fits weighted planes for inspection and later
operation gates, but it does not project vertices or change topology.
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

from omega_local.remesh.policy import build_mesh_topology


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class PlanarProxyConfig:
    mesh_path: Path
    policy_npz: Path
    output_dir: Path
    proxies_name: str = "proxies"
    min_proxy_faces: int = 24
    seed_planar_score: float = 0.35
    max_seed_detail: float = 0.75
    max_seed_unexplained: float = 0.50
    max_connect_boundary_score: float = 0.45
    fit_trim_quantile: float = 0.90
    fit_iterations: int = 3
    max_plane_rmse: float = 0.0
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class PlanarProxyResult:
    proxies_json: Path
    proxies_npz: Path
    summary_json: Path
    debug_meshes: dict[str, Path]
    proxy_count: int


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(int(size), dtype=np.int64)
        self.rank = np.zeros((int(size),), dtype=np.int8)

    def find(self, item: int) -> int:
        item = int(item)
        parent = int(self.parent[item])
        if parent != item:
            parent = self.find(parent)
            self.parent[item] = parent
        return parent

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


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


def _load_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh object from {mesh_path}: {type(mesh)!r}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    return vertices, faces


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_area = np.linalg.norm(raw_normals, axis=1)
    normals = np.divide(raw_normals, np.maximum(double_area[:, None], 1e-12), out=np.zeros_like(raw_normals))
    return np.mean(triangles, axis=1), normals, 0.5 * double_area


def _load_policy(path: Path, face_count: int) -> dict[str, np.ndarray]:
    data = np.load(path)
    policy = {key: data[key] for key in data.files}
    required = [
        "face_area",
        "face_support",
        "face_detail_posterior",
        "face_unexplained_offset",
        "face_planar_score",
        "edge_faces",
        "edge_boundary_score",
    ]
    missing = [key for key in required if key not in policy]
    if missing:
        raise KeyError(f"Policy file is missing required arrays: {missing}")
    for key in ["face_support", "face_detail_posterior", "face_unexplained_offset", "face_planar_score"]:
        if np.asarray(policy[key]).shape[0] != face_count:
            raise ValueError(f"Policy array {key!r} does not match mesh face count {face_count}.")
    return policy


def _fit_weighted_plane(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    points = points[valid]
    weights = weights[valid]
    if points.shape[0] < 3:
        return None
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        return None
    centroid = np.sum(points * weights[:, None], axis=0) / weight_sum
    centered = points - centroid
    cov = (centered * weights[:, None]).T @ centered / weight_sum
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return None
    return centroid, normal / norm


def _fit_robust_plane(points: np.ndarray, weights: np.ndarray, trim_quantile: float, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    active = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0.0)
    if np.count_nonzero(active) < 3:
        return None
    for _ in range(max(int(iterations), 1)):
        plane = _fit_weighted_plane(points[active], weights[active])
        if plane is None:
            return None
        centroid, normal = plane
        residual = np.abs((points - centroid) @ normal)
        finite = active & np.isfinite(residual)
        if np.count_nonzero(finite) < 6:
            break
        cutoff = float(np.quantile(residual[finite], float(np.clip(trim_quantile, 0.5, 1.0))))
        next_active = finite & (residual <= max(cutoff, 1e-9))
        if np.array_equal(next_active, active):
            break
        active = next_active
    plane = _fit_weighted_plane(points[active], weights[active])
    if plane is None:
        return None
    centroid, normal = plane
    residual = np.abs((points - centroid) @ normal)
    return centroid, normal, residual


def _initial_components(policy: dict[str, np.ndarray], config: PlanarProxyConfig) -> np.ndarray:
    planar_score = np.asarray(policy["face_planar_score"], dtype=np.float32)
    detail = np.asarray(policy["face_detail_posterior"], dtype=np.float32)
    selected = (
        (planar_score >= float(config.seed_planar_score))
        & (detail <= float(config.max_seed_detail))
    )
    selected_ids = np.flatnonzero(selected)
    labels = np.full(planar_score.shape[0], -1, dtype=np.int32)
    if selected_ids.size == 0:
        return labels
    local = np.full(planar_score.shape[0], -1, dtype=np.int64)
    local[selected_ids] = np.arange(selected_ids.size, dtype=np.int64)
    uf = _UnionFind(selected_ids.size)
    edge_faces = np.asarray(policy["edge_faces"], dtype=np.int64)
    boundary_score = np.asarray(policy["edge_boundary_score"], dtype=np.float32)
    for edge_id, (a, b) in enumerate(edge_faces):
        if b < 0:
            continue
        if not selected[a] or not selected[b]:
            continue
        if boundary_score[edge_id] > float(config.max_connect_boundary_score):
            continue
        uf.union(int(local[a]), int(local[b]))
    root_to_label: dict[int, int] = {}
    for face_id in selected_ids:
        root = uf.find(int(local[face_id]))
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[face_id] = root_to_label[root]
    return labels


def _categorical_colors(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    colors = np.zeros(ids.shape + (3,), dtype=np.uint8)
    valid = ids >= 0
    x = ids[valid].astype(np.uint64) + 1
    colors[valid, 0] = ((x * 73 + 41) % 255).astype(np.uint8)
    colors[valid, 1] = ((x * 151 + 83) % 255).astype(np.uint8)
    colors[valid, 2] = ((x * 211 + 17) % 255).astype(np.uint8)
    colors[~valid] = np.array([42, 42, 42], dtype=np.uint8)
    return colors


def _colorize_scalar(values: np.ndarray, valid: np.ndarray, *, vmin: float, vmax: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    normalized = np.clip((values - float(vmin)) / max(float(vmax - vmin), 1e-6), 0.0, 1.0)
    mapped = cv2.applyColorMap(np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = cv2.cvtColor(mapped, cv2.COLOR_BGR2RGB)
    if rgb.ndim == 3 and values.ndim == 1 and rgb.shape[1] == 1:
        rgb = rgb[:, 0, :]
    rgb[~valid] = np.array([38, 38, 38], dtype=np.uint8)
    return rgb


def _write_face_color_ply(vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray, out_path: Path) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate([colors.astype(np.uint8), np.full((faces.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def _scalar_range(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)[np.asarray(valid, dtype=bool) & np.isfinite(values)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(finite, 0.02))
    hi = float(np.quantile(finite, 0.98))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    return lo, hi


def _write_debug_meshes(vertices: np.ndarray, faces: np.ndarray, arrays: dict[str, np.ndarray], out_dir: Path) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes" / "proxies"
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    proxy_id = arrays["face_proxy_id"]
    path = debug_dir / "face_proxy_id.ply"
    _write_face_color_ply(vertices, faces, _categorical_colors(proxy_id), path)
    paths["face_proxy_id"] = path
    for name in ["face_proxy_residual", "face_proxy_weight", "face_proxy_boundary"]:
        values = np.asarray(arrays[name], dtype=np.float32)
        valid = proxy_id >= 0 if name != "face_proxy_boundary" else values > 0
        vmin, vmax = (0.0, 1.0) if name != "face_proxy_residual" else _scalar_range(values, valid)
        path = debug_dir / f"{name}.ply"
        _write_face_color_ply(vertices, faces, _colorize_scalar(values, valid, vmin=vmin, vmax=vmax), path)
        paths[name] = path
    return paths


def extract_planar_proxies(config: PlanarProxyConfig, progress: ProgressFn | None = None) -> PlanarProxyResult:
    progress = progress or _progress_default
    mesh_path = config.mesh_path.expanduser().resolve()
    policy_npz = config.policy_npz.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    proxies_json = output_dir / f"{config.proxies_name}.json"
    proxies_npz = output_dir / f"{config.proxies_name}.npz"
    summary_json = output_dir / f"{config.proxies_name}_summary.json"
    if proxies_json.exists() and not config.overwrite:
        raise FileExistsError(f"Proxy output exists: {proxies_json}. Re-run with overwrite=True.")

    vertices, faces = _load_mesh_arrays(mesh_path)
    centers, mesh_normals, face_area = _face_geometry(vertices, faces)
    policy = _load_policy(policy_npz, int(faces.shape[0]))
    topology = build_mesh_topology(faces)
    initial_labels = _initial_components(policy, config)

    face_proxy_id = np.full(faces.shape[0], -1, dtype=np.int32)
    face_proxy_residual = np.zeros(faces.shape[0], dtype=np.float32)
    face_proxy_weight = np.zeros(faces.shape[0], dtype=np.float32)
    proxy_entries: list[dict[str, Any]] = []
    next_proxy_id = 0
    planar_score = np.asarray(policy["face_planar_score"], dtype=np.float32)
    support = np.asarray(policy["face_support"], dtype=np.float32)
    weights_all = np.asarray(policy["face_area"], dtype=np.float32) * support * planar_score

    progress(f"[proxies] Mesh: {mesh_path} ({faces.shape[0]:,} faces)")
    progress(f"[proxies] Policy: {policy_npz}")
    for component_id in sorted(int(v) for v in np.unique(initial_labels) if int(v) >= 0):
        component_faces = np.flatnonzero(initial_labels == component_id)
        if component_faces.size < int(config.min_proxy_faces):
            continue
        fit = _fit_robust_plane(
            centers[component_faces],
            weights_all[component_faces],
            trim_quantile=float(config.fit_trim_quantile),
            iterations=int(config.fit_iterations),
        )
        if fit is None:
            continue
        centroid, normal, residual_local = fit
        weights = weights_all[component_faces]
        weight_sum = max(float(np.sum(weights)), 1e-12)
        rmse = math.sqrt(float(np.sum(weights * residual_local * residual_local) / weight_sum))
        if float(config.max_plane_rmse) > 0.0 and rmse > float(config.max_plane_rmse):
            continue
        proxy_id = next_proxy_id
        next_proxy_id += 1
        d = -float(np.dot(normal, centroid))
        face_proxy_id[component_faces] = proxy_id
        face_proxy_residual[component_faces] = residual_local.astype(np.float32)
        face_proxy_weight[component_faces] = np.clip(weights / max(float(np.quantile(weights, 0.95)), 1e-6), 0.0, 1.0).astype(np.float32)
        proxy_entries.append(
            {
                "id": int(proxy_id),
                "faceCount": int(component_faces.size),
                "area": float(np.sum(face_area[component_faces])),
                "weightSum": float(np.sum(weights)),
                "centroid": [float(v) for v in centroid],
                "normal": [float(v) for v in normal],
                "d": float(d),
                "rmse": float(rmse),
                "residualMedian": float(np.median(residual_local)),
                "residualQ90": float(np.quantile(residual_local, 0.90)),
                "planarScoreMean": float(np.mean(planar_score[component_faces])),
                "supportMean": float(np.mean(support[component_faces])),
                "detailMean": float(np.mean(np.asarray(policy["face_detail_posterior"])[component_faces])),
                "unexplainedOffsetMean": float(np.mean(np.asarray(policy["face_unexplained_offset"])[component_faces])),
            }
        )

    edge_proxy_boundary = np.zeros(topology.edge_faces.shape[0], dtype=np.uint8)
    for edge_id, (a, b) in enumerate(topology.edge_faces):
        if b < 0:
            edge_proxy_boundary[edge_id] = 1 if face_proxy_id[a] >= 0 else 0
            continue
        pa = int(face_proxy_id[a])
        pb = int(face_proxy_id[b])
        edge_proxy_boundary[edge_id] = 1 if (pa >= 0 or pb >= 0) and pa != pb else 0
    face_proxy_boundary = np.zeros(faces.shape[0], dtype=np.float32)
    for local_id in range(3):
        face_proxy_boundary = np.maximum(face_proxy_boundary, edge_proxy_boundary[topology.face_edges[:, local_id]].astype(np.float32))

    arrays = {
        "face_proxy_id": face_proxy_id,
        "face_proxy_residual": face_proxy_residual,
        "face_proxy_weight": face_proxy_weight,
        "face_proxy_boundary": face_proxy_boundary.astype(np.float32),
        "edge_proxy_boundary": edge_proxy_boundary,
        "edge_faces": topology.edge_faces.astype(np.int64),
        "face_edges": topology.face_edges.astype(np.int64),
    }
    np.savez_compressed(proxies_npz, **arrays)
    _write_json(proxies_json, {"stageName": "omega_planar_proxies", "createdAtUtc": datetime.now(timezone.utc).isoformat(), "proxies": proxy_entries})
    debug_meshes = _write_debug_meshes(vertices, faces, arrays, output_dir) if config.write_debug_meshes else {}
    summary = {
        "stageName": "omega_planar_proxies",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "mesh_path": str(mesh_path),
            "policy_npz": str(policy_npz),
            "output_dir": str(output_dir),
        },
        "counts": {
            "initialPlanarFaces": int(np.count_nonzero(initial_labels >= 0)),
            "proxyFaces": int(np.count_nonzero(face_proxy_id >= 0)),
            "proxyCount": int(len(proxy_entries)),
            "proxyBoundaryEdges": int(np.count_nonzero(edge_proxy_boundary)),
        },
        "stats": {
            "face_proxy_residual": _summarize(face_proxy_residual, face_proxy_id >= 0),
            "face_proxy_weight": _summarize(face_proxy_weight, face_proxy_id >= 0),
            "proxy_face_count": _summarize(np.asarray([entry["faceCount"] for entry in proxy_entries], dtype=np.float32)),
            "proxy_rmse": _summarize(np.asarray([entry["rmse"] for entry in proxy_entries], dtype=np.float32)),
        },
        "outputs": {
            "proxiesJson": str(proxies_json),
            "proxiesNpz": str(proxies_npz),
            "summaryJson": str(summary_json),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
    }
    _write_json(summary_json, summary)
    progress(f"[proxies] Wrote {proxies_json}")
    progress(f"[proxies] Wrote {proxies_npz}")
    return PlanarProxyResult(
        proxies_json=proxies_json,
        proxies_npz=proxies_npz,
        summary_json=summary_json,
        debug_meshes=debug_meshes,
        proxy_count=int(len(proxy_entries)),
    )


def _resolve_path(path: Path, base: Path) -> Path:
    raw = path.expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [raw.resolve(), (base / raw).resolve()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _optional_path(path: Path | None) -> Path | None:
    if path is None or str(path).strip() in {"", "."}:
        return None
    return path


def _existing_file(path: Path, description: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{description} does not exist: {path}")
    if path.is_dir():
        raise SystemExit(
            f"{description} points to a directory, not a file: {path}\n"
            "If this came from an environment variable, re-export it or omit the flag to use the default."
        )
    return path


def _config_from_args(args: argparse.Namespace) -> PlanarProxyConfig:
    model_dir = args.model_dir.expanduser().resolve()
    mesh_arg = _optional_path(args.mesh)
    policy_arg = _optional_path(args.policy_npz)
    output_arg = _optional_path(args.output_dir)
    mesh_path = _existing_file(
        _resolve_path(mesh_arg, model_dir) if mesh_arg is not None else model_dir / "remesh" / "local" / "preclean_mesh.ply",
        "Mesh",
    )
    policy_npz = _existing_file(
        _resolve_path(policy_arg, model_dir) if policy_arg is not None else model_dir / "remesh" / "local" / "policy.npz",
        "Policy npz",
    )
    output_dir = output_arg.expanduser().resolve() if output_arg is not None else model_dir / "remesh" / "local"
    return PlanarProxyConfig(
        mesh_path=mesh_path,
        policy_npz=policy_npz,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract planar proxies from Phase 3 remesh policy fields.")
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--policy-npz", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--proxies-name", default="proxies")
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
    extract_planar_proxies(_config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
