"""Stage 2: extract view-normal evidence for OMeGa local remeshing.

This module is the clean orchestration layer for the redesigned evidence stage:

1. load the same prepared OMeGa camera frames and normal maps used for training;
2. render deterministic face-id/depth view buffers for the precleaned mesh;
3. compute per-view normal derivatives and robust tolerances;
4. transfer the view evidence into face-aligned arrays without remeshing.

The outputs are meant to be inspected before any policy or topology operation
uses them.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import imageio.v2 as imageio
import numpy as np
import trimesh

from omega_local.remesh.normal_maps import NormalGeometryConfig, compute_normal_geometry
from omega_local.remesh.transfer import EvidenceAccumulator, TransferConfig, summarize
from omega_local.remesh.view_buffers import render_mesh_view_buffer, resize_camera_inputs


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class MeshEvidenceConfig:
    mesh_path: Path
    data_dir: Path
    out_dir: Path
    view_buffer_dir: Path
    evidence_name: str = "evidence"
    data_factor: int = 1
    test_every: int = 8
    frame_stride: int = 4
    max_frames: int = 32
    max_buffer_width: int = 1600
    near_plane: float = 0.2
    edge_thickness: int = 1
    target_edge_min_px: float = 2.0
    target_edge_max_px: float = 96.0
    high_gradient_quantile: float = 0.85
    local_variance_radius_px: int = 2
    coverage_threshold: float = 0.15
    support_view_k0: float = 3.0
    min_offset_radius_px: float = 2.0
    max_offset_radius_px: float = 16.0
    write_view_buffers: bool = True
    write_debug_meshes: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class MeshEvidenceResult:
    evidence_npz: Path
    summary_json: Path
    frame_manifest_jsonl: Path
    view_buffer_dir: Path
    debug_meshes: dict[str, Path]
    face_count: int
    frame_count: int


def _progress_default(message: str) -> None:
    print(message, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _relative_or_absolute(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_path(path: Path, base: Path) -> Path:
    raw = path.expanduser()
    if raw.is_absolute():
        return raw.resolve()
    candidates = [
        raw.resolve(),
        (base / raw).resolve(),
        (_repo_root() / raw).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1]


def _latest_step_file(folder: Path, prefix: str) -> Path | None:
    found: list[tuple[int, Path]] = []
    for path in folder.glob(f"{prefix}_*_rank0.ply"):
        try:
            found.append((int(path.name.split("_")[1]), path))
        except (IndexError, ValueError):
            continue
    return sorted(found)[-1][1] if found else None


def resolve_latest_omega_mesh(model_dir: Path, requested: Path | None = None, iteration: int = -1) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if requested is not None:
        path = _resolve_path(requested, model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Requested mesh does not exist: {path}")
        return path

    preclean = model_dir / "remesh" / "local" / "preclean_mesh.ply"
    if preclean.exists():
        return preclean

    plys_dir = model_dir / "plys"
    if int(iteration) > 0:
        for step in (int(iteration), int(iteration) - 1, int(iteration) + 1):
            if step < 0:
                continue
            candidate = plys_dir / f"mesh_{step}_rank0.ply"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Requested mesh iteration was not found under {plys_dir}: {iteration}")

    latest = _latest_step_file(plys_dir, "mesh")
    if latest is not None:
        return latest
    raise FileNotFoundError(f"No OMeGa mesh found under {plys_dir}")


def resolve_dataset_dir(model_dir: Path, requested: Path | None = None) -> Path:
    model_dir = model_dir.expanduser().resolve()
    if requested is not None:
        path = _resolve_path(requested, model_dir)
        if path.exists():
            return path
        raise FileNotFoundError(f"Requested dataset dir does not exist: {path}")

    cfg_path = model_dir / "cfg.json"
    if cfg_path.exists():
        cfg = _read_json(cfg_path)
        data_dir = cfg.get("data_dir")
        if data_dir:
            path = _resolve_path(Path(str(data_dir)), model_dir.parent.parent)
            if path.exists():
                return path
    default = model_dir.parent / "dataset"
    if default.exists():
        return default
    raise FileNotFoundError(f"Could not resolve OMeGa dataset dir for {model_dir}")


def dataset_manifest_metadata(data_dir: Path) -> dict[str, Any]:
    manifest_path = data_dir.resolve().parent / "dataset_manifest.json"
    if not manifest_path.exists():
        return {"manifestPath": str(manifest_path), "exists": False}
    payload = _read_json(manifest_path)
    normal_maps = payload.get("normalMaps", {})
    frames = payload.get("frames", [])
    return {
        "manifestPath": str(manifest_path),
        "exists": True,
        "normalSource": normal_maps.get("source") if isinstance(normal_maps, dict) else None,
        "normalFrameCount": len(normal_maps.get("frames", [])) if isinstance(normal_maps, dict) else 0,
        "frameCount": len(frames) if isinstance(frames, list) else 0,
        "imageMode": payload.get("imageMode"),
        "poseSource": payload.get("poseSource"),
    }


def _import_colmap_dataset() -> tuple[Any, Any, Any, Any, Any]:
    repo_root = _repo_root()
    examples_root = repo_root / "examples"
    for path in (examples_root, repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from datasets.colmap import (  # type: ignore
            Parser,
            _crop_guide_maps,
            _load_building_depth_guide_npz,
            _remap_guide_maps,
            _resize_guide_maps,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Could not import OMeGa COLMAP dataset parser.") from exc
    return Parser, _load_building_depth_guide_npz, _resize_guide_maps, _remap_guide_maps, _crop_guide_maps


def load_omega_parser(
    *,
    data_dir: Path,
    data_factor: int,
    test_every: int,
    load_normal_maps: bool,
    load_guides: bool,
    guide_dir: str | None,
) -> Any:
    Parser, _, _, _, _ = _import_colmap_dataset()
    return Parser(
        data_dir=str(data_dir),
        factor=int(data_factor),
        normalize=False,
        vfm_init=False,
        test_every=int(test_every),
        load_normal_maps=bool(load_normal_maps),
        load_building_depth_guides=bool(load_guides and guide_dir),
        building_depth_guide_dir=guide_dir,
    )


def select_frame_indices(total_frames: int, frame_stride: int, max_frames: int) -> list[int]:
    indices = list(range(int(total_frames)))[:: max(int(frame_stride), 1)]
    if int(max_frames) > 0:
        indices = indices[: int(max_frames)]
    return indices


def _safe_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, np.maximum(norms, eps), out=np.zeros_like(vectors, dtype=np.float64))


def load_frame_inputs(parser: Any, index: int) -> dict[str, Any]:
    """Load and undistort/crop RGB and normal-map inputs like OMeGa training."""

    _, load_guide_npz, resize_guides, remap_guides, crop_guides = _import_colmap_dataset()
    image = imageio.imread(parser.image_paths[index])[..., :3]
    normal_map = None
    normal_valid = None
    if getattr(parser, "load_normal_maps", False) and parser.normal_maps_paths is not None:
        normal_map = imageio.imread(parser.normal_maps_paths[index])[..., :3]
        if parser.normal_mask_paths is not None:
            mask_path = Path(parser.normal_mask_paths[index])
            if mask_path.exists():
                invalid = imageio.imread(mask_path)
                if invalid.ndim == 3:
                    invalid = invalid[..., 0]
                normal_valid = invalid < 128

    guide = None
    if getattr(parser, "load_building_depth_guides", False) and parser.building_depth_guide_paths is not None:
        guide = load_guide_npz(parser.building_depth_guide_paths[index])
        guide = resize_guides(guide, width=int(image.shape[1]), height=int(image.shape[0]))

    camera_id = parser.camera_ids[index]
    K = np.asarray(parser.Ks_dict[camera_id], dtype=np.float64).copy()
    params = parser.params_dict[camera_id]
    if len(params) > 0:
        mapx, mapy = parser.mapx_dict[camera_id], parser.mapy_dict[camera_id]
        x, y, w, h = parser.roi_undist_dict[camera_id]
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        image = image[y : y + h, x : x + w]
        if normal_map is not None:
            normal_map = cv2.remap(normal_map, mapx, mapy, cv2.INTER_LINEAR)
            normal_map = normal_map[y : y + h, x : x + w]
        if normal_valid is not None:
            normal_valid = cv2.remap(normal_valid.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST).astype(bool)
            normal_valid = normal_valid[y : y + h, x : x + w]
        if guide is not None:
            guide = crop_guides(remap_guides(guide, mapx, mapy), x, y, w, h)

    if normal_map is not None:
        normal_map = (np.asarray(normal_map, dtype=np.float32) / 255.0 - 0.5) * 2.0
        normal_map = _safe_normalize(normal_map.astype(np.float64)).astype(np.float32)
        if normal_valid is None:
            normal_valid = np.isfinite(normal_map).all(axis=2) & (np.linalg.norm(normal_map, axis=2) > 0.25)

    return {
        "image_name": str(parser.image_names[index]),
        "camera_id": int(camera_id),
        "image": np.asarray(image[..., :3], dtype=np.uint8),
        "K": K,
        "camtoworld": np.asarray(parser.camtoworlds[index], dtype=np.float64),
        "normal_map_cam": normal_map,
        "normal_valid": normal_valid,
        "guide": guide,
    }


def load_mesh_arrays(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh object from {mesh_path}: {type(mesh)!r}")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        raise ValueError(f"Mesh has no triangles: {mesh_path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Expected triangular faces in {mesh_path}, got shape {faces.shape}")
    return vertices, faces


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


def _write_face_color_ply(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray,
    out_path: Path,
) -> None:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgba = np.concatenate(
        [np.asarray(colors, dtype=np.uint8), np.full((faces.shape[0], 1), 255, dtype=np.uint8)],
        axis=1,
    )
    mesh.visual.face_colors = rgba
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)


def _scalar_range(values: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float32)
    finite = arr[np.asarray(valid, dtype=bool) & np.isfinite(arr)]
    finite = finite[finite > 0.0]
    if finite.size == 0:
        return 0.0, 1.0
    lo = float(np.quantile(finite, 0.02))
    hi = float(np.quantile(finite, 0.98))
    if hi <= lo + 1e-9:
        hi = lo + 1.0
    return lo, hi


def write_debug_meshes(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    evidence: dict[str, np.ndarray],
    out_dir: Path,
) -> dict[str, Path]:
    debug_dir = out_dir / "debug_meshes" / "evidence"
    debug_dir.mkdir(parents=True, exist_ok=True)
    normal_valid = evidence["normal_count"] > 0
    specs: dict[str, tuple[str, np.ndarray, np.ndarray, float, float]] = {
        "face_support": ("face_support", evidence["face_support"], evidence["view_count"] > 0, 0.0, 1.0),
        "face_view_disagreement": (
            "face_view_disagreement",
            evidence["face_view_disagreement"],
            normal_valid,
            *_scalar_range(evidence["face_view_disagreement"], normal_valid),
        ),
        "face_normal_kappa": (
            "face_normal_kappa",
            evidence["face_normal_kappa"],
            normal_valid,
            *_scalar_range(evidence["face_normal_kappa"], normal_valid),
        ),
        "face_normal_gradient_max": ("face_normal_gradient_max", evidence["face_normal_gradient_max"], normal_valid, 0.0, 1.0),
        "face_detail_target_length": (
            "face_detail_target_length",
            evidence["face_detail_target_length"],
            normal_valid,
            *_scalar_range(evidence["face_detail_target_length"], normal_valid),
        ),
        "face_position_tolerance": (
            "face_position_tolerance",
            evidence["face_position_tolerance"],
            normal_valid,
            *_scalar_range(evidence["face_position_tolerance"], normal_valid),
        ),
        "face_direction_confidence": (
            "face_direction_confidence",
            evidence["face_direction_confidence"],
            normal_valid,
            0.0,
            1.0,
        ),
        "offset_normal_gradient_max": (
            "offset_normal_gradient_max",
            evidence["offset_normal_gradient_max"],
            evidence["offset_high_gradient_pixel_count"] > 0,
            *_scalar_range(evidence["offset_normal_gradient_max"], evidence["offset_high_gradient_pixel_count"] > 0),
        ),
    }
    paths: dict[str, Path] = {}
    for out_name, (_key, values, valid, vmin, vmax) in specs.items():
        out_path = debug_dir / f"{out_name}.ply"
        colors = _colorize_scalar(values, valid, vmin=float(vmin), vmax=float(vmax))
        _write_face_color_ply(vertices=vertices, faces=faces, colors=colors, out_path=out_path)
        paths[out_name] = out_path
    return paths


def _write_frame_buffer_npz(
    path: Path,
    *,
    frame: dict[str, Any],
    K: np.ndarray,
    image_scale: float,
    view_buffer: Any,
    normal_geometry: Any,
    offset_delta_px: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        face_id=view_buffer.face_id,
        depth=view_buffer.depth,
        mesh_normal_rgb=view_buffer.mesh_normal_rgb,
        mesh_edge_mask=view_buffer.mesh_edge_mask.astype(np.uint8),
        normal_gradient_g=normal_geometry.gradient_g.astype(np.float32),
        normal_gradient_normalized=normal_geometry.gradient_normalized.astype(np.float32),
        normal_target_edge_length_px=normal_geometry.target_edge_length_px.astype(np.float32),
        normal_high_gradient_mask=normal_geometry.high_gradient_mask.astype(np.uint8),
        normal_structure_tensor=normal_geometry.structure_tensor.astype(np.float32),
        normal_direction_confidence=normal_geometry.direction_confidence.astype(np.float32),
        normal_local_variance=normal_geometry.local_normal_variance.astype(np.float32),
        normal_local_gradient_g=normal_geometry.local_gradient_g.astype(np.float32),
        normal_local_gradient_normalized=normal_geometry.local_gradient_normalized.astype(np.float32),
        normal_local_structure_tensor=normal_geometry.local_structure_tensor.astype(np.float32),
        normal_local_direction_confidence=normal_geometry.local_direction_confidence.astype(np.float32),
        normal_local_high_gradient_mask=normal_geometry.local_high_gradient_mask.astype(np.uint8),
        normal_valid=normal_geometry.normal_valid.astype(np.uint8),
        visible_face_ids=view_buffer.visible_face_ids,
        K=np.asarray(K, dtype=np.float64),
        camtoworld=np.asarray(frame["camtoworld"], dtype=np.float64),
        image_name=np.array(str(frame["image_name"])),
        camera_id=np.array(int(frame["camera_id"]), dtype=np.int32),
        image_scale=np.array(float(image_scale), dtype=np.float32),
        width=np.array(int(view_buffer.width), dtype=np.int32),
        height=np.array(int(view_buffer.height), dtype=np.int32),
        normal_noise_sigma_g=np.array(float(normal_geometry.noise_sigma_g), dtype=np.float32),
        normal_tolerance_tau_g=np.array(float(normal_geometry.tolerance_tau_g), dtype=np.float32),
        normal_tolerance_tau_n=np.array(float(normal_geometry.tolerance_tau_n), dtype=np.float32),
        normal_tolerance_tau_a=np.array(float(normal_geometry.tolerance_tau_a), dtype=np.float32),
        normal_high_gradient_threshold=np.array(float(normal_geometry.high_gradient_threshold), dtype=np.float32),
        offset_delta_px=np.array(float(offset_delta_px), dtype=np.float32),
    )


def compute_mesh_face_evidence(config: MeshEvidenceConfig, progress: ProgressFn | None = None) -> MeshEvidenceResult:
    progress = progress or _progress_default
    mesh_path = config.mesh_path.expanduser().resolve()
    data_dir = config.data_dir.expanduser().resolve()
    out_dir = config.out_dir.expanduser().resolve()
    view_buffer_dir = config.view_buffer_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    view_buffer_dir.mkdir(parents=True, exist_ok=True)

    evidence_npz = out_dir / f"{config.evidence_name}.npz"
    summary_json = out_dir / f"{config.evidence_name}_summary.json"
    frame_manifest_jsonl = out_dir / f"{config.evidence_name}_frames.jsonl"
    if evidence_npz.exists() and not config.overwrite:
        raise FileExistsError(f"Evidence output exists: {evidence_npz}. Re-run with --overwrite to replace it.")

    dataset_metadata = dataset_manifest_metadata(data_dir)
    parser = load_omega_parser(
        data_dir=data_dir,
        data_factor=int(config.data_factor),
        test_every=int(config.test_every),
        load_normal_maps=True,
        load_guides=False,
        guide_dir=None,
    )
    frame_indices = select_frame_indices(len(parser.image_names), int(config.frame_stride), int(config.max_frames))
    if not frame_indices:
        raise ValueError("No frames selected for Stage 2 evidence extraction.")

    vertices, faces = load_mesh_arrays(mesh_path)
    accumulator = EvidenceAccumulator(
        vertices=vertices,
        faces=faces,
        config=TransferConfig(
            coverage_threshold=float(config.coverage_threshold),
            support_view_k0=float(config.support_view_k0),
            min_offset_radius_px=float(config.min_offset_radius_px),
            max_offset_radius_px=float(config.max_offset_radius_px),
        ),
    )

    progress(f"[evidence] Mesh: {mesh_path} ({vertices.shape[0]:,} vertices, {faces.shape[0]:,} faces)")
    progress(f"[evidence] Dataset: {data_dir}")
    progress(f"[evidence] Prepared normal source: {dataset_metadata.get('normalSource') or '<unknown>'}")
    progress(f"[evidence] Frames: {len(frame_indices):,} selected from {len(parser.image_names):,}")
    progress(f"[evidence] Buffer width cap: {config.max_buffer_width if config.max_buffer_width > 0 else 'source resolution'}")

    normal_cfg = NormalGeometryConfig(
        target_edge_min_px=float(config.target_edge_min_px),
        target_edge_max_px=float(config.target_edge_max_px),
        high_gradient_quantile=float(config.high_gradient_quantile),
        local_variance_radius_px=int(config.local_variance_radius_px),
    )

    frame_rows: list[dict[str, Any]] = []
    for ordinal, parser_index in enumerate(frame_indices, start=1):
        frame = load_frame_inputs(parser, int(parser_index))
        if frame["normal_map_cam"] is None:
            raise ValueError(f"Frame has no normal map in prepared dataset: {frame['image_name']}")
        image, normal_map, normal_valid, K, image_scale = resize_camera_inputs(
            image=frame["image"],
            normal_map=frame["normal_map_cam"],
            normal_valid=frame["normal_valid"],
            K=frame["K"],
            max_width=int(config.max_buffer_width),
        )
        normal_geometry = compute_normal_geometry(normal_map, normal_valid, normal_cfg)
        view_buffer = render_mesh_view_buffer(
            vertices=vertices,
            faces=faces,
            face_normals_world=accumulator.normals,
            K=K,
            camtoworld=frame["camtoworld"],
            image_shape=image.shape[:2],
            near_plane=float(config.near_plane),
            edge_thickness=int(config.edge_thickness),
        )
        frame_summary = accumulator.add_frame(
            view_buffer=view_buffer,
            normal_geometry=normal_geometry,
            K=K,
            camtoworld=frame["camtoworld"],
            normal_map_cam=normal_map,
        )

        stem = Path(str(frame["image_name"])).stem
        buffer_path = view_buffer_dir / "frames" / f"{ordinal:04d}_{stem}.npz"
        if config.write_view_buffers:
            _write_frame_buffer_npz(
                buffer_path,
                frame=frame,
                K=K,
                image_scale=float(image_scale),
                view_buffer=view_buffer,
                normal_geometry=normal_geometry,
                offset_delta_px=float(frame_summary.offset_delta_px),
            )

        row = {
            "ordinal": int(ordinal),
            "parserIndex": int(parser_index),
            "imageName": str(frame["image_name"]),
            "cameraId": int(frame["camera_id"]),
            "imageWidth": int(image.shape[1]),
            "imageHeight": int(image.shape[0]),
            "imageScale": float(image_scale),
            "visibleFaceCount": int(frame_summary.visible_face_count),
            "visiblePixelCount": int(frame_summary.visible_pixel_count),
            "normalFaceCount": int(frame_summary.normal_face_count),
            "normalPixelCount": int(frame_summary.normal_pixel_count),
            "highGradientPixelCount": int(frame_summary.high_gradient_pixel_count),
            "offsetDeltaPx": float(frame_summary.offset_delta_px),
            "offsetAssociatedPixelCount": int(frame_summary.offset_associated_pixel_count),
            "normalNoiseSigmaG": float(normal_geometry.noise_sigma_g),
            "normalToleranceTauG": float(normal_geometry.tolerance_tau_g),
            "normalToleranceTauN": float(normal_geometry.tolerance_tau_n),
            "normalToleranceTauA": float(normal_geometry.tolerance_tau_a),
            "normalHighGradientThreshold": float(normal_geometry.high_gradient_threshold),
            "normalLocalVarianceRadiusPx": int(config.local_variance_radius_px),
            "viewBufferPath": str(buffer_path),
        }
        frame_rows.append(row)
        progress(
            "[evidence {:04d}/{:04d}] {} visible_faces={:,} normal_faces={:,} delta={:.2f}px".format(
                ordinal,
                len(frame_indices),
                frame["image_name"],
                frame_summary.visible_face_count,
                frame_summary.normal_face_count,
                frame_summary.offset_delta_px,
            )
        )

    evidence = accumulator.finalize(selected_frame_count=len(frame_indices))
    np.savez_compressed(evidence_npz, **evidence)
    debug_meshes: dict[str, Path] = {}
    if config.write_debug_meshes:
        debug_meshes = write_debug_meshes(vertices=vertices, faces=faces, evidence=evidence, out_dir=out_dir)

    normal_valid_faces = evidence["normal_count"] > 0
    summary = {
        "stageName": "omega_view_normal_evidence",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "config": {
            **asdict(config),
            "mesh_path": str(mesh_path),
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "view_buffer_dir": str(view_buffer_dir),
        },
        "preparedDataset": dataset_metadata,
        "mesh": {
            "path": str(mesh_path),
            "vertexCount": int(vertices.shape[0]),
            "faceCount": int(faces.shape[0]),
            "surfaceArea": float(np.sum(evidence["face_area"])),
        },
        "frames": {
            "selectedCount": int(len(frame_indices)),
            "totalParserFrames": int(len(parser.image_names)),
            "frameStride": int(config.frame_stride),
            "maxFrames": int(config.max_frames),
            "manifest": str(frame_manifest_jsonl),
        },
        "coverage": {
            "visibleFaceFraction": float(np.count_nonzero(evidence["view_count"] > 0) / max(faces.shape[0], 1)),
            "normalFaceFraction": float(np.count_nonzero(normal_valid_faces) / max(faces.shape[0], 1)),
            "supportedFaceFraction": float(np.count_nonzero(evidence["face_support"] > 0.0) / max(faces.shape[0], 1)),
            "offsetDetailFaceFraction": float(
                np.count_nonzero(evidence["offset_high_gradient_pixel_count"] > 0) / max(faces.shape[0], 1)
            ),
        },
        "stats": {
            "visible_view_count": summarize(evidence["visible_view_count"]),
            "normal_view_count": summarize(evidence["normal_view_count"]),
            "normal_pixel_count": summarize(evidence["normal_pixel_count"]),
            "normal_coverage_mean": summarize(evidence["normal_coverage_mean"], evidence["view_count"] > 0),
            "face_support": summarize(evidence["face_support"], evidence["view_count"] > 0),
            "face_view_disagreement": summarize(evidence["face_view_disagreement"], normal_valid_faces),
            "face_normal_kappa": summarize(evidence["face_normal_kappa"], normal_valid_faces),
            "face_normal_gradient_max": summarize(evidence["face_normal_gradient_max"], normal_valid_faces),
            "face_detail_target_length": summarize(evidence["face_detail_target_length"], normal_valid_faces),
            "face_position_tolerance": summarize(evidence["face_position_tolerance"], normal_valid_faces),
            "face_direction_confidence": summarize(evidence["face_direction_confidence"], normal_valid_faces),
            "offset_high_gradient_pixel_count": summarize(evidence["offset_high_gradient_pixel_count"]),
            "offset_normal_gradient_max": summarize(
                evidence["offset_normal_gradient_max"],
                evidence["offset_high_gradient_pixel_count"] > 0,
            ),
        },
        "outputs": {
            "evidenceNpz": str(evidence_npz),
            "summaryJson": str(summary_json),
            "frameManifestJsonl": str(frame_manifest_jsonl),
            "viewBufferDir": str(view_buffer_dir),
            "debugMeshes": {name: str(path) for name, path in debug_meshes.items()},
        },
        "notes": [
            "The mesh is rasterized into face-id/depth buffers before evidence transfer.",
            "Normal derivatives and tolerances are computed per view before projection to mesh faces.",
            "Face support is normalized by visible pixel coverage and view count, then penalized by cross-view kappa disagreement.",
            "Offset evidence associates high-gradient normal pixels to the nearest rendered mesh pixel inside a measured per-view radius.",
            "This stage writes evidence only; it does not modify topology or geometry.",
        ],
    }
    _write_json(summary_json, summary)
    _write_jsonl(frame_manifest_jsonl, frame_rows)
    progress(f"[evidence] Wrote {evidence_npz}")
    progress(f"[evidence] Wrote {summary_json}")
    return MeshEvidenceResult(
        evidence_npz=evidence_npz,
        summary_json=summary_json,
        frame_manifest_jsonl=frame_manifest_jsonl,
        view_buffer_dir=view_buffer_dir,
        debug_meshes=debug_meshes,
        face_count=int(faces.shape[0]),
        frame_count=len(frame_indices),
    )


def _config_from_args(args: argparse.Namespace) -> MeshEvidenceConfig:
    model_dir = Path(args.model_dir).expanduser().resolve()
    cfg_path = model_dir / "cfg.json"
    model_cfg = _read_json(cfg_path) if cfg_path.exists() else {}
    mesh_path = resolve_latest_omega_mesh(model_dir, args.mesh, int(args.iteration))
    data_dir = resolve_dataset_dir(model_dir, args.data_dir)
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else model_dir / "remesh" / "local"
    view_buffer_dir = (
        args.view_buffer_dir.expanduser().resolve()
        if args.view_buffer_dir is not None
        else model_dir / "remesh" / "view_buffers"
    )
    return MeshEvidenceConfig(
        mesh_path=mesh_path,
        data_dir=data_dir,
        out_dir=out_dir,
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
        local_variance_radius_px=int(args.local_variance_radius_px),
        coverage_threshold=float(args.coverage_threshold),
        support_view_k0=float(args.support_view_k0),
        min_offset_radius_px=float(args.min_offset_radius_px),
        max_offset_radius_px=float(args.max_offset_radius_px),
        write_view_buffers=bool(args.view_buffers),
        write_debug_meshes=bool(args.debug_meshes),
        overwrite=bool(args.overwrite),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract redesigned OMeGa view-normal evidence: view buffers, normal geometry, and face fields."
    )
    parser.add_argument("model_dir", type=Path, help="Completed OMeGa result directory.")
    parser.add_argument("--mesh", type=Path, default=None, help="Defaults to <model-dir>/remesh/local/preclean_mesh.ply when present.")
    parser.add_argument("--iteration", type=int, default=-1, help="Raw OMeGa mesh iteration when --mesh is omitted and no preclean mesh exists.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Prepared OMeGa dataset dir. Defaults to cfg.json/data_dir.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/local.")
    parser.add_argument("--view-buffer-dir", type=Path, default=None, help="Defaults to <model-dir>/remesh/view_buffers.")
    parser.add_argument("--evidence-name", default="evidence")
    parser.add_argument("--data-factor", type=int, default=None)
    parser.add_argument("--test-every", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=32, help="0 means all selected parser frames.")
    parser.add_argument("--max-buffer-width", type=int, default=1600, help="0 keeps source prepared camera resolution.")
    parser.add_argument("--near-plane", type=float, default=None)
    parser.add_argument("--edge-thickness", type=int, default=1)
    parser.add_argument("--target-edge-min-px", type=float, default=2.0)
    parser.add_argument("--target-edge-max-px", type=float, default=96.0)
    parser.add_argument("--high-gradient-quantile", type=float, default=0.85)
    parser.add_argument("--local-variance-radius-px", type=int, default=2)
    parser.add_argument("--coverage-threshold", type=float, default=0.15)
    parser.add_argument("--support-view-k0", type=float, default=3.0)
    parser.add_argument("--min-offset-radius-px", type=float, default=2.0)
    parser.add_argument("--max-offset-radius-px", type=float, default=16.0)
    parser.add_argument("--view-buffers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug-meshes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compute_mesh_face_evidence(_config_from_args(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
