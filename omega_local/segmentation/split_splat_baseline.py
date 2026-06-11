"""Split&Splat-style mask propagation baseline for OMeGa captures.

This adapter ports the Split&Splat "Split" stage into the OMeGa folder
conventions. It intentionally does not consume SAI3D labels. The method starts
from per-view SAM2 automatic proposals, projects a 3D point set into every view
with depth visibility, grows global point labels by proposal overlap, clusters
those point labels, then writes per-view instance masks.

The implementation follows the public Split&Splat propagation script while
replacing its hard-coded ScanNet/COLMAP layout with our common proposal source:

    <model_dir>/segmentation/baselines/<proposal_source_name>/

Outputs are written under:

    <model_dir>/segmentation/baselines/split_splat/
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
import scipy.spatial
import trimesh
from sklearn.cluster import DBSCAN

from omega_local.segmentation.proposal_source import resolve_proposal_source
from omega_local.segmentation.sai3d_baseline import (
    THIRD_PARTY_ROOT,
    PointSet,
    _build_point_set,
    _build_raycast_scene,
    _build_sam2_image_predictor,
    _greedy_coreset_2d,
    _load_label_map,
    _load_mesh,
    _overlay,
    _predict_sam2_mask,
    _project_vertices,
    _render_depth_m,
    _resolve_input_mesh,
    _write_labeled_point_cloud,
)


EPSILON = 0.02
ACCURACY_LABELS = 0.7


@dataclass(frozen=True)
class SplitSplatPaths:
    baseline_dir: Path
    dataset_dir: Path
    image_dir: Path
    pose_dir: Path
    mask_dir: Path
    depth_dir: Path
    point_label_dir: Path
    view_mask_dir: Path
    view_mask_masks_dir: Path
    view_mask_overlay_dir: Path
    by_instance_dir: Path
    projected_dir: Path
    projected_masks_dir: Path
    projected_overlay_dir: Path
    assigned_dir: Path
    assigned_masks_dir: Path
    assigned_overlay_dir: Path
    prompt_dir: Path
    prompt_overlay_dir: Path
    sam2_dir: Path
    sam2_masks_dir: Path
    sam2_overlay_dir: Path
    summary_dir: Path
    scene_name: str

    @property
    def frame_manifest(self) -> Path:
        return self.dataset_dir / "frame_manifest.jsonl"

    @property
    def dataset_summary(self) -> Path:
        return self.dataset_dir / "dataset_summary.json"

    @property
    def summary(self) -> Path:
        return self.baseline_dir / "baseline_summary.json"

    @property
    def scene_mesh(self) -> Path:
        return self.dataset_dir / "scene_mesh.ply"

    @property
    def points_path(self) -> Path:
        return self.point_label_dir / "points.npy"

    @property
    def point_labels_path(self) -> Path:
        return self.point_label_dir / "point_labels.npy"

    @property
    def clustered_point_labels_path(self) -> Path:
        return self.point_label_dir / "clustered_point_labels.npy"

    @property
    def projection_cache_path(self) -> Path:
        return self.point_label_dir / "projection_cache.npz"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug_token(value: str) -> str:
    out: list[str] = []
    last = False
    for char in str(value).strip().lower():
        if char.isalnum():
            out.append(char)
            last = False
        elif not last:
            out.append("_")
            last = True
    return "".join(out).strip("_") or "split_splat"


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def _require_file(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_file():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def _require_dir(path: Path, description: str) -> Path:
    if not path.exists() or not path.is_dir():
        raise SystemExit(f"{description} does not exist: {path}")
    return path


def resolve_paths(
    model_dir: Path,
    *,
    baseline_name: str = "split_splat",
    scene_name: str | None = None,
    output_dir: Path | None = None,
) -> SplitSplatPaths:
    baseline_dir = (
        output_dir.expanduser().resolve()
        if output_dir is not None
        else model_dir / "segmentation" / "baselines" / _slug_token(baseline_name)
    )
    scene_slug = _slug_token(scene_name or f"{model_dir.parent.parent.name}_{model_dir.name}")
    dataset_dir = baseline_dir / "dataset"
    return SplitSplatPaths(
        baseline_dir=baseline_dir,
        dataset_dir=dataset_dir,
        image_dir=dataset_dir / "images" / scene_slug,
        pose_dir=dataset_dir / "poses" / scene_slug,
        mask_dir=dataset_dir / "masks" / scene_slug,
        depth_dir=dataset_dir / "depth" / scene_slug,
        point_label_dir=baseline_dir / "point_labels",
        view_mask_dir=baseline_dir / "view_masks",
        view_mask_masks_dir=baseline_dir / "view_masks" / "masks",
        view_mask_overlay_dir=baseline_dir / "view_masks" / "overlays",
        by_instance_dir=baseline_dir / "view_masks" / "by_instance",
        projected_dir=baseline_dir / "view_masks" / "projected_points",
        projected_masks_dir=baseline_dir / "view_masks" / "projected_points" / "masks",
        projected_overlay_dir=baseline_dir / "view_masks" / "projected_points" / "overlays",
        assigned_dir=baseline_dir / "view_masks" / "proposal_assigned",
        assigned_masks_dir=baseline_dir / "view_masks" / "proposal_assigned" / "masks",
        assigned_overlay_dir=baseline_dir / "view_masks" / "proposal_assigned" / "overlays",
        prompt_dir=baseline_dir / "view_masks" / "sam2_prompts",
        prompt_overlay_dir=baseline_dir / "view_masks" / "sam2_prompts" / "overlays",
        sam2_dir=baseline_dir / "view_masks" / "sam2_fallback",
        sam2_masks_dir=baseline_dir / "view_masks" / "sam2_fallback" / "masks",
        sam2_overlay_dir=baseline_dir / "view_masks" / "sam2_fallback" / "overlays",
        summary_dir=baseline_dir / "summaries",
        scene_name=scene_slug,
    )


def _write_depth_png(path: Path, depth_m: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 1, np.iinfo(np.uint16).max).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), depth_mm):
        raise RuntimeError(f"Failed to write depth png: {path}")
    values = depth_m[valid]
    return {
        "coverage": float(np.count_nonzero(valid) / max(valid.size, 1)),
        "minDepthMeters": float(values.min()) if values.size else 0.0,
        "medianDepthMeters": float(np.median(values)) if values.size else 0.0,
        "maxDepthMeters": float(values.max()) if values.size else 0.0,
    }


def prepare(args: argparse.Namespace, paths: SplitSplatPaths) -> dict[str, Any]:
    model_dir = args.model_dir.expanduser().resolve()
    source_mesh = _resolve_input_mesh(model_dir, args.mesh, int(args.iteration))
    proposal_source = resolve_proposal_source(
        model_dir,
        proposal_source_name=str(args.proposal_source_name),
        proposal_source_dir=args.proposal_source_dir,
    )
    source_rows = proposal_source.rows(int(args.source_frame_stride), int(args.max_frames))

    if paths.dataset_dir.exists() and any(paths.dataset_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Split&Splat dataset already exists: {paths.dataset_dir}. Pass --overwrite to replace it.")
    if args.overwrite and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
    for directory in [paths.image_dir, paths.pose_dir, paths.mask_dir, paths.depth_dir, paths.summary_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    mesh = _load_mesh(source_mesh)
    mesh.export(paths.scene_mesh)
    scene = _build_raycast_scene(source_mesh)

    rows: list[dict[str, Any]] = []
    depth_coverages: list[float] = []
    for out_index, row in enumerate(source_rows):
        source_frame_id = proposal_source.frame_id(row, out_index)
        color_src = proposal_source.image_path(row)
        pose_src = proposal_source.pose_path(row)
        mask_src = proposal_source.mask_path(row, out_index)

        rgb = cv2.imread(str(color_src), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Could not read proposal RGB frame: {color_src}")
        height, width = rgb.shape[:2]
        pose = np.loadtxt(pose_src, dtype=np.float64)
        fx = float(row["fx"])
        fy = float(row["fy"])
        cx = float(row["cx"])
        cy = float(row["cy"])

        color_path = paths.image_dir / f"{out_index}.jpg"
        pose_path = paths.pose_dir / f"{out_index}.txt"
        mask_path = paths.mask_dir / f"mask_{out_index}.png"
        depth_path = paths.depth_dir / f"{out_index}.png"
        if not cv2.imwrite(str(color_path), rgb, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]):
            raise RuntimeError(f"Failed to write RGB frame: {color_path}")
        np.savetxt(pose_path, pose, fmt="%.10f")
        mask = _load_label_map(mask_src, expected_shape=(height, width))
        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to write proposal mask: {mask_path}")

        depth = _render_depth_m(
            scene,
            pose_world_t_cam=pose,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width,
            height=height,
            chunk_rows=int(args.raycast_chunk_rows),
        )
        depth_stats = _write_depth_png(depth_path, depth)
        depth_coverages.append(float(depth_stats["coverage"]))

        rows.append(
            {
                "splitSplatFrameId": int(out_index),
                "proposalSource": str(proposal_source.baseline_dir),
                "sourceFrameId": int(source_frame_id),
                "scanID": str(row.get("scanID", "")),
                "frameID": int(row.get("frameID", source_frame_id)),
                "imageName": str(row.get("imageName", color_src.name)),
                "sourceImagePath": str(row.get("sourceImagePath", color_src)),
                "colorPath": str(color_path.relative_to(paths.dataset_dir)),
                "posePath": str(pose_path.relative_to(paths.dataset_dir)),
                "maskPath": str(mask_path.relative_to(paths.dataset_dir)),
                "depthPath": str(depth_path.relative_to(paths.dataset_dir)),
                "width": int(width),
                "height": int(height),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "depthStats": depth_stats,
                "inputLabelCount": int(np.count_nonzero(np.unique(mask) > 0)),
                "inputCoverage": float(np.count_nonzero(mask > 0) / max(mask.size, 1)),
            }
        )
        print(
            f"[split-splat prepare {out_index + 1:04d}/{len(source_rows):04d}] "
            f"source={source_frame_id} labels={rows[-1]['inputLabelCount']} "
            f"mask_cov={rows[-1]['inputCoverage']:.3f} depth_cov={depth_stats['coverage']:.3f}"
        )

    _write_jsonl(paths.frame_manifest, rows)
    summary = {
        "stage": "prepare",
        "timestampUtc": _now(),
        "method": "Stage common SAM2 automatic proposals, RGB frames, poses, OMeGa mesh depth, and scene mesh for Split&Splat propagation.",
        "modelDir": str(model_dir),
        "sceneName": paths.scene_name,
        "sourceMesh": str(source_mesh),
        "proposalSourceName": str(args.proposal_source_name),
        "proposalSourceDir": str(proposal_source.baseline_dir),
        "frameCount": int(len(rows)),
        "depthCoverageMean": float(np.mean(depth_coverages)) if depth_coverages else 0.0,
        "outputs": {
            "datasetDir": str(paths.dataset_dir),
            "imageDir": str(paths.image_dir),
            "poseDir": str(paths.pose_dir),
            "maskDir": str(paths.mask_dir),
            "depthDir": str(paths.depth_dir),
            "sceneMesh": str(paths.scene_mesh),
            "frameManifest": str(paths.frame_manifest),
        },
    }
    _write_json(paths.dataset_summary, summary)
    _write_json(paths.summary_dir / "prepare_summary.json", summary)
    return summary


def _load_depth_m(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read depth image: {path}")
    return raw.astype(np.float32) / 1000.0


def _project_visible_points(
    points: np.ndarray,
    *,
    pose_world_t_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    depth_m: np.ndarray,
    epsilon: float,
) -> dict[str, np.ndarray]:
    z, u, v = _project_vertices(points, pose_world_t_cam, fx=fx, fy=fy, cx=cx, cy=cy)
    bounded = (z > 0.0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = np.nonzero(bounded)[0]
    if valid.size == 0:
        return {
            "indices": np.zeros(0, dtype=np.int32),
            "u": np.zeros(0, dtype=np.int32),
            "v": np.zeros(0, dtype=np.int32),
            "z": np.zeros(0, dtype=np.float32),
        }
    sampled = depth_m[v[valid], u[valid]]
    keep = (sampled > 0.0) & (np.abs(sampled - z[valid]) <= float(epsilon))
    valid = valid[keep]
    if valid.size == 0:
        return {
            "indices": np.zeros(0, dtype=np.int32),
            "u": np.zeros(0, dtype=np.int32),
            "v": np.zeros(0, dtype=np.int32),
            "z": np.zeros(0, dtype=np.float32),
        }

    linear = v[valid].astype(np.int64) * int(width) + u[valid].astype(np.int64)
    order = np.lexsort((z[valid], linear))
    sorted_linear = linear[order]
    sorted_valid = valid[order]
    first = np.r_[0, np.nonzero(sorted_linear[1:] != sorted_linear[:-1])[0] + 1]
    chosen = sorted_valid[first]
    return {
        "indices": chosen.astype(np.int32, copy=False),
        "u": u[chosen].astype(np.int32, copy=False),
        "v": v[chosen].astype(np.int32, copy=False),
        "z": z[chosen].astype(np.float32, copy=False),
    }


def _erode_split_splat(mask: np.ndarray, *, a: float = 0.10, b: float = 0.30) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return binary
    size = float(np.count_nonzero(binary) / max(binary.size, 1))
    if size < float(a):
        kernel_size, iterations = 3, 3
    elif size < float(b):
        kernel_size, iterations = 5, 2
    else:
        kernel_size, iterations = 5, 2
    eroded = cv2.erode(binary.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations) > 0
    return eroded if np.any(eroded) else binary


def _erode_new_object(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    if not np.any(binary):
        return binary
    size = float(np.count_nonzero(binary) / max(binary.size, 1))
    if size < 0.05:
        kernel_size, iterations = 3, 2
    elif size < 0.10:
        kernel_size, iterations = 5, 2
    else:
        kernel_size, iterations = 5, 3
    eroded = cv2.erode(binary.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8), iterations=iterations) > 0
    return eroded if np.any(eroded) else binary


def _point_ids_in_mask(proj: dict[str, np.ndarray], mask: np.ndarray) -> np.ndarray:
    if proj["indices"].size == 0 or not np.any(mask):
        return np.zeros(0, dtype=np.int32)
    inside = mask[proj["v"], proj["u"]]
    return proj["indices"][inside].astype(np.int32, copy=False)


def _largest_cluster_point_ids(points: np.ndarray, point_ids: np.ndarray, *, eps: float, min_samples: int) -> np.ndarray:
    point_ids = np.asarray(point_ids, dtype=np.int64)
    if point_ids.size <= max(int(min_samples), 1):
        return point_ids.astype(np.int32, copy=False)
    xyz = points[point_ids]
    try:
        labels = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(xyz)
    except ValueError:
        return point_ids.astype(np.int32, copy=False)
    valid = labels >= 0
    if not np.any(valid):
        return point_ids.astype(np.int32, copy=False)
    cluster_ids, counts = np.unique(labels[valid], return_counts=True)
    largest = int(cluster_ids[int(np.argmax(counts))])
    keep = labels == largest
    if np.count_nonzero(keep) <= 10:
        return point_ids.astype(np.int32, copy=False)
    return point_ids[keep].astype(np.int32, copy=False)


def _get_label(votes: dict[int, float]) -> tuple[int, float]:
    if not votes:
        return 0, 0.0
    total = float(sum(float(value) for value in votes.values()))
    if total <= 0.0:
        return 0, 0.0
    label, weight = max(votes.items(), key=lambda item: float(item[1]))
    return int(label), float(weight) / total


def _reindex_labels(
    label_points: dict[int, set[int]],
    point_votes: list[dict[int, float]],
) -> tuple[dict[int, set[int]], list[dict[int, float]], dict[int, int]]:
    old_to_new = {old: new + 1 for new, old in enumerate(sorted(label_points.keys()))}
    new_label_points = {old_to_new[old]: set(points) for old, points in label_points.items() if old in old_to_new}
    new_point_votes: list[dict[int, float]] = []
    for votes in point_votes:
        new_votes = {old_to_new[label]: weight for label, weight in votes.items() if label in old_to_new}
        new_point_votes.append(new_votes)
    return new_label_points, new_point_votes, old_to_new


def _update_label_points(
    label_points: dict[int, set[int]],
    point_votes: list[dict[int, float]],
    *,
    min_points: int,
    accuracy: float,
) -> tuple[dict[int, set[int]], list[dict[int, float]], dict[int, int]]:
    for point_id, votes in enumerate(point_votes):
        major_label, prob = _get_label(votes)
        for label in list(label_points.keys()):
            if (label != major_label or prob < float(accuracy)) and point_id in label_points[label]:
                label_points[label].discard(point_id)
            elif label == major_label and prob >= float(accuracy):
                label_points[label].add(point_id)

    removed = [label for label, points in label_points.items() if len(points) < int(min_points)]
    for label in removed:
        for point_id in list(label_points[label]):
            if label in point_votes[point_id]:
                del point_votes[point_id][label]
        del label_points[label]
    return _reindex_labels(label_points, point_votes)


def _add_vote(
    *,
    point_id: int,
    label: int,
    weight: float,
    label_points: dict[int, set[int]],
    point_votes: list[dict[int, float]],
) -> None:
    if int(label) <= 0:
        return
    votes = point_votes[int(point_id)]
    votes[int(label)] = float(votes.get(int(label), 0.0)) + float(weight)
    label_points.setdefault(int(label), set()).add(int(point_id))


def _initialize_from_first_view(
    *,
    points: np.ndarray,
    proposal_mask: np.ndarray,
    proj: dict[str, np.ndarray],
    label_points: dict[int, set[int]],
    point_votes: list[dict[int, float]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    next_label = max(label_points.keys(), default=0) + 1
    for local_label in np.unique(proposal_mask):
        local_id = int(local_label)
        if local_id <= 0:
            continue
        mask = proposal_mask == local_id
        eroded = _erode_split_splat(mask)
        point_ids = _point_ids_in_mask(proj, eroded)
        point_ids = _largest_cluster_point_ids(
            points,
            point_ids,
            eps=float(args.cluster_eps),
            min_samples=int(args.cluster_min_samples),
        )
        if point_ids.size == 0:
            records.append({"localLabel": local_id, "globalLabel": 0, "pointCount": 0, "decision": "empty"})
            continue
        label = int(next_label)
        next_label += 1
        label_points[label] = set()
        for point_id in point_ids:
            for old_label in list(point_votes[int(point_id)].keys()):
                if old_label in label_points:
                    label_points[old_label].discard(int(point_id))
            _add_vote(point_id=int(point_id), label=label, weight=1.25, label_points=label_points, point_votes=point_votes)
        records.append(
            {
                "localLabel": local_id,
                "globalLabel": int(label),
                "pointCount": int(point_ids.size),
                "decision": "new_initial_label",
            }
        )
    return records


def _virtual_mask_from_labels(
    *,
    proj: dict[str, np.ndarray],
    point_votes: list[dict[int, float]],
    clustered_point_labels: np.ndarray | None,
    width: int,
    height: int,
    accuracy: float,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.uint16)
    if proj["indices"].size == 0:
        return out
    for point_id, x, y in zip(proj["indices"], proj["u"], proj["v"], strict=True):
        label = 0
        if clustered_point_labels is not None:
            if int(point_id) < clustered_point_labels.shape[0]:
                label = int(clustered_point_labels[int(point_id)])
        else:
            label, prob = _get_label(point_votes[int(point_id)])
            if prob < float(accuracy):
                label = 0
        if label > 0 and out[int(y), int(x)] == 0:
            out[int(y), int(x)] = np.uint16(min(label, np.iinfo(np.uint16).max))
    return out


def _positive_labels_in(mask_labels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.unique(mask_labels[np.asarray(mask, dtype=bool)])
    return values[values > 0].astype(np.int32, copy=False)


def _majority_label(mask_labels: np.ndarray, mask: np.ndarray) -> int:
    values = mask_labels[np.asarray(mask, dtype=bool)]
    values = values[values > 0]
    if values.size == 0:
        return 0
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[int(np.argmax(counts))])


def _propagate_one_view(
    *,
    points: np.ndarray,
    proposal_mask: np.ndarray,
    proj: dict[str, np.ndarray],
    label_points: dict[int, set[int]],
    point_votes: list[dict[int, float]],
    virtual_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    assigned_labels: dict[int, int] = {}

    for local_label in np.unique(proposal_mask):
        local_id = int(local_label)
        if local_id <= 0:
            continue
        mask = proposal_mask == local_id
        overlap_eroded = _erode_split_splat(mask, a=0.05, b=0.50)
        labels_in_mask = _positive_labels_in(virtual_mask, overlap_eroded)

        if labels_in_mask.size == 0:
            eroded = _erode_new_object(mask)
            point_ids = _point_ids_in_mask(proj, eroded)
            point_ids = _largest_cluster_point_ids(
                points,
                point_ids,
                eps=float(args.cluster_eps),
                min_samples=int(args.cluster_min_samples),
            )
            if point_ids.size == 0:
                assignments.append({"localLabel": local_id, "globalLabel": 0, "decision": "empty_new", "pointCount": 0})
                continue
            label = max(label_points.keys(), default=0) + 1
            label_points[label] = set()
            for point_id in point_ids:
                weight = 1.25 if not point_votes[int(point_id)] else 1.0
                _add_vote(point_id=int(point_id), label=label, weight=weight, label_points=label_points, point_votes=point_votes)
            assignments.append({"localLabel": local_id, "globalLabel": int(label), "decision": "new_label", "pointCount": int(point_ids.size)})
            continue

        if labels_in_mask.size == 1:
            label = int(labels_in_mask[0])
            assigned_labels[label] = local_id
            assignments.append({"localLabel": local_id, "globalLabel": label, "decision": "single_overlap"})
            continue

        major = _majority_label(virtual_mask, overlap_eroded)
        eroded = _erode_split_splat(mask, a=0.05, b=0.30)
        point_ids = _point_ids_in_mask(proj, eroded)
        point_ids = _largest_cluster_point_ids(
            points,
            point_ids,
            eps=float(args.cluster_eps),
            min_samples=int(args.cluster_min_samples),
        )
        for point_id in point_ids:
            _add_vote(point_id=int(point_id), label=major, weight=0.75, label_points=label_points, point_votes=point_votes)
        assignments.append(
            {
                "localLabel": local_id,
                "globalLabel": int(major),
                "decision": "multi_overlap_majority",
                "overlapLabels": [int(value) for value in labels_in_mask.tolist()],
                "pointCount": int(point_ids.size),
            }
        )

    for label, local_id in assigned_labels.items():
        mask = proposal_mask == int(local_id)
        eroded = _erode_split_splat(mask, a=0.05, b=0.30)
        point_ids = _point_ids_in_mask(proj, eroded)
        point_ids = _largest_cluster_point_ids(
            points,
            point_ids,
            eps=float(args.cluster_eps),
            min_samples=int(args.cluster_min_samples),
        )
        for point_id in point_ids:
            weight = 1.25 if not point_votes[int(point_id)] else 1.0
            _add_vote(point_id=int(point_id), label=int(label), weight=weight, label_points=label_points, point_votes=point_votes)
    return assignments


def _majority_point_labels(point_votes: list[dict[int, float]], *, accuracy: float) -> np.ndarray:
    labels = np.zeros(len(point_votes), dtype=np.int32)
    for point_id, votes in enumerate(point_votes):
        label, prob = _get_label(votes)
        if label > 0 and prob >= float(accuracy):
            labels[point_id] = int(label)
    return labels


def _cluster_point_labels(points: np.ndarray, labels: np.ndarray, *, eps: float, min_samples: int) -> tuple[np.ndarray, dict[str, Any]]:
    clustered = np.zeros(labels.shape[0], dtype=np.int32)
    label_summaries: list[dict[str, Any]] = []
    for label in np.unique(labels):
        label_i = int(label)
        if label_i <= 0:
            continue
        point_ids = np.nonzero(labels == label_i)[0]
        if point_ids.size == 0:
            continue
        xyz = points[point_ids]
        try:
            cluster_ids = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(xyz)
        except ValueError:
            cluster_ids = np.zeros(point_ids.shape[0], dtype=np.int32)
        valid = cluster_ids >= 0
        if not np.any(valid):
            continue
        unique, counts = np.unique(cluster_ids[valid], return_counts=True)
        if unique.size > 3:
            keep_ids = point_ids
            decision = "keep_all_many_clusters"
        else:
            largest = int(unique[int(np.argmax(counts))])
            keep_ids = point_ids[cluster_ids == largest]
            decision = "keep_largest_cluster"
        if keep_ids.size < 10:
            continue
        clustered[keep_ids] = label_i
        label_summaries.append(
            {
                "label": label_i,
                "inputPointCount": int(point_ids.size),
                "keptPointCount": int(keep_ids.size),
                "clusterCount": int(unique.size),
                "decision": decision,
            }
        )
    positive = clustered > 0
    old_to_new: dict[int, int] = {}
    remapped = np.zeros_like(clustered)
    for new_label, old_label in enumerate(np.unique(clustered[positive]), start=1):
        old_to_new[int(old_label)] = int(new_label)
        remapped[clustered == int(old_label)] = int(new_label)
    for item in label_summaries:
        item["remappedLabel"] = int(old_to_new.get(int(item["label"]), 0))
    summary = {
        "labelCountBefore": int(np.count_nonzero(np.unique(labels) > 0)),
        "labelCountAfter": int(np.count_nonzero(np.unique(remapped) > 0)),
        "labeledPointCountBefore": int(np.count_nonzero(labels > 0)),
        "labeledPointCountAfter": int(np.count_nonzero(remapped > 0)),
        "labels": label_summaries,
    }
    return remapped.astype(np.int32, copy=False), summary


def _write_projection_cache(path: Path, projection_cache: list[dict[str, np.ndarray]], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frame_ids": np.asarray([int(row["splitSplatFrameId"]) for row in rows], dtype=np.int32)
    }
    for idx, proj in enumerate(projection_cache):
        payload[f"indices_{idx}"] = proj["indices"].astype(np.int32, copy=False)
        payload[f"u_{idx}"] = proj["u"].astype(np.int32, copy=False)
        payload[f"v_{idx}"] = proj["v"].astype(np.int32, copy=False)
        payload[f"z_{idx}"] = proj["z"].astype(np.float32, copy=False)
    np.savez_compressed(path, **payload)


def _load_projection_cache(path: Path) -> list[dict[str, np.ndarray]]:
    data = np.load(path)
    frame_ids = data["frame_ids"]
    out: list[dict[str, np.ndarray]] = []
    for idx in range(frame_ids.shape[0]):
        out.append(
            {
                "indices": data[f"indices_{idx}"].astype(np.int32, copy=False),
                "u": data[f"u_{idx}"].astype(np.int32, copy=False),
                "v": data[f"v_{idx}"].astype(np.int32, copy=False),
                "z": data[f"z_{idx}"].astype(np.float32, copy=False),
            }
        )
    return out


def _write_label_point_clouds(paths: SplitSplatPaths, points: np.ndarray, point_labels: np.ndarray, clustered: np.ndarray) -> dict[str, Any]:
    paths.point_label_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths.points_path, points.astype(np.float32, copy=False))
    np.save(paths.point_labels_path, point_labels.astype(np.int32, copy=False))
    np.save(paths.clustered_point_labels_path, clustered.astype(np.int32, copy=False))
    labeled_points = paths.point_label_dir / "labeled_points.ply"
    clustered_points = paths.point_label_dir / "clustered_labeled_points.ply"
    _write_labeled_point_cloud(labeled_points, points, point_labels)
    _write_labeled_point_cloud(clustered_points, points, clustered)
    return {
        "points": str(paths.points_path),
        "pointLabels": str(paths.point_labels_path),
        "clusteredPointLabels": str(paths.clustered_point_labels_path),
        "labeledPoints": str(labeled_points),
        "clusteredLabeledPoints": str(clustered_points),
    }


def propagate(args: argparse.Namespace, paths: SplitSplatPaths) -> dict[str, Any]:
    rows = _read_jsonl(_require_file(paths.frame_manifest, "Split&Splat frame manifest"))
    if not rows:
        raise SystemExit(f"No frames found in {paths.frame_manifest}")
    mesh = _load_mesh(_require_file(paths.scene_mesh, "Split&Splat scene mesh"))
    point_set = _build_point_set(mesh, args)
    points = point_set.points.astype(np.float32, copy=False)
    point_votes: list[dict[int, float]] = [dict() for _ in range(points.shape[0])]
    label_points: dict[int, set[int]] = {}

    order_indices = list(range(len(rows)))
    if str(args.process_order) == "reverse":
        order_indices = list(reversed(order_indices))

    projection_cache: list[dict[str, np.ndarray]] = [None] * len(rows)  # type: ignore[list-item]
    history: list[dict[str, Any]] = []
    initialized = False
    for order_position, row_index in enumerate(order_indices):
        row = rows[row_index]
        frame_id = int(row["splitSplatFrameId"])
        width = int(row["width"])
        height = int(row["height"])
        pose = np.loadtxt(paths.dataset_dir / str(row["posePath"]), dtype=np.float64)
        depth = _load_depth_m(paths.dataset_dir / str(row["depthPath"]))
        proposal_mask = _load_label_map(paths.dataset_dir / str(row["maskPath"]), expected_shape=(height, width))
        proj = _project_visible_points(
            points,
            pose_world_t_cam=pose,
            fx=float(row["fx"]),
            fy=float(row["fy"]),
            cx=float(row["cx"]),
            cy=float(row["cy"]),
            width=width,
            height=height,
            depth_m=depth,
            epsilon=float(args.projection_epsilon),
        )
        projection_cache[row_index] = proj

        if not initialized:
            assignments = _initialize_from_first_view(
                points=points,
                proposal_mask=proposal_mask,
                proj=proj,
                label_points=label_points,
                point_votes=point_votes,
                args=args,
            )
            initialized = True
            action = "initialize"
        else:
            label_points, point_votes, _ = _update_label_points(
                label_points,
                point_votes,
                min_points=int(args.min_label_points),
                accuracy=float(args.accuracy_labels),
            )
            virtual_mask = _virtual_mask_from_labels(
                proj=proj,
                point_votes=point_votes,
                clustered_point_labels=None,
                width=width,
                height=height,
                accuracy=float(args.accuracy_labels),
            )
            assignments = _propagate_one_view(
                points=points,
                proposal_mask=proposal_mask,
                proj=proj,
                label_points=label_points,
                point_votes=point_votes,
                virtual_mask=virtual_mask,
                args=args,
            )
            action = "propagate"

        history.append(
            {
                "frameId": frame_id,
                "orderPosition": int(order_position),
                "action": action,
                "visiblePointCount": int(proj["indices"].size),
                "proposalLabelCount": int(np.count_nonzero(np.unique(proposal_mask) > 0)),
                "activeLabelCount": int(len(label_points)),
                "assignments": assignments,
            }
        )
        print(
            f"[split-splat propagate {order_position + 1:04d}/{len(rows):04d}] "
            f"frame={frame_id} action={action} visible={proj['indices'].size} labels={len(label_points)}"
        )

    label_points, point_votes, _ = _update_label_points(
        label_points,
        point_votes,
        min_points=int(args.min_label_points),
        accuracy=float(args.accuracy_labels),
    )
    point_labels = _majority_point_labels(point_votes, accuracy=float(args.accuracy_labels))
    clustered_labels, cluster_summary = _cluster_point_labels(
        points,
        point_labels,
        eps=float(args.final_cluster_eps),
        min_samples=int(args.final_cluster_min_samples),
    )
    _write_projection_cache(paths.projection_cache_path, projection_cache, rows)
    label_outputs = _write_label_point_clouds(paths, points, point_labels, clustered_labels)
    summary = {
        "stage": "propagate",
        "timestampUtc": _now(),
        "method": "Split&Splat mask-to-3D propagation: first-view initialization, virtual-mask overlap propagation, point-label majority voting, and DBSCAN label cleanup.",
        "frameCount": int(len(rows)),
        "processOrder": str(args.process_order),
        "pointSet": point_set.summary,
        "parameters": {
            "projectionEpsilon": float(args.projection_epsilon),
            "accuracyLabels": float(args.accuracy_labels),
            "minLabelPoints": int(args.min_label_points),
            "clusterEps": float(args.cluster_eps),
            "clusterMinSamples": int(args.cluster_min_samples),
            "finalClusterEps": float(args.final_cluster_eps),
            "finalClusterMinSamples": int(args.final_cluster_min_samples),
        },
        "labelCount": int(np.count_nonzero(np.unique(point_labels) > 0)),
        "clusteredLabelCount": int(np.count_nonzero(np.unique(clustered_labels) > 0)),
        "labeledPointCount": int(np.count_nonzero(point_labels > 0)),
        "clusteredLabeledPointCount": int(np.count_nonzero(clustered_labels > 0)),
        "clusterSummary": cluster_summary,
        "history": history,
        "outputs": {
            **label_outputs,
            "projectionCache": str(paths.projection_cache_path),
        },
    }
    _write_json(paths.summary_dir / "propagate_summary.json", summary)
    return summary


def _label_overlay(rgb_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return _overlay(rgb_bgr, labels)


def _boundary_overlay(rgb_bgr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    if labels.shape[:2] != rgb.shape[:2]:
        labels = cv2.resize(labels, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    boundary[:, :-1] |= labels[:, 1:] != labels[:, :-1]
    boundary[1:, :] |= labels[1:, :] != labels[:-1, :]
    boundary[:-1, :] |= labels[1:, :] != labels[:-1, :]
    boundary &= labels > 0
    boundary = cv2.dilate(boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
    out = rgb.copy()
    out[boundary] = np.array([255, 50, 25], dtype=np.uint8)
    return out


def _draw_prompt_overlay(rgb_bgr: np.ndarray, projected: np.ndarray, prompt_records: list[dict[str, Any]]) -> np.ndarray:
    out = _overlay(rgb_bgr, projected, alpha=0.28)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    for record in prompt_records:
        points = np.asarray(record.get("points", []), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            continue
        for x_f, y_f in points:
            x = int(round(float(x_f)))
            y = int(round(float(y_f)))
            cv2.circle(bgr, (x, y), 4, (35, 255, 35), thickness=-1)
            cv2.circle(bgr, (x, y), 6, (255, 255, 255), thickness=1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _write_instance_mask(by_instance_dir: Path, label: int, frame_id: int, mask: np.ndarray) -> None:
    out_dir = by_instance_dir / f"{int(label):04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{frame_id:06d}.png"), np.asarray(mask, dtype=np.uint8) * 255)


def _merge_instance_into_label_map(label_map: np.ndarray, label: int, mask: np.ndarray, *, fill_only_empty: bool = False) -> None:
    binary = np.asarray(mask, dtype=bool)
    if fill_only_empty:
        binary &= label_map == 0
    label_map[binary] = np.uint16(min(int(label), np.iinfo(np.uint16).max))


def _final_masks_for_view(
    *,
    row: dict[str, Any],
    proposal_mask: np.ndarray,
    projected: np.ndarray,
    proj: dict[str, np.ndarray],
    clustered_labels: np.ndarray,
    rgb_bgr: np.ndarray,
    predictor: Any | None,
    paths: SplitSplatPaths,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    height, width = proposal_mask.shape
    frame_id = int(row["splitSplatFrameId"])
    assigned = np.zeros((height, width), dtype=np.uint16)
    sam2_fallback = np.zeros((height, width), dtype=np.uint16)
    prompt_records: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    created: set[int] = set()

    for local_label in np.unique(proposal_mask):
        local_id = int(local_label)
        if local_id <= 0:
            continue
        proposal_binary = proposal_mask == local_id
        eroded = _erode_split_splat(proposal_binary)
        labels_in_mask = _positive_labels_in(projected, eroded)
        if labels_in_mask.size == 0:
            assignments.append({"localLabel": local_id, "globalLabel": 0, "decision": "no_projected_label"})
            continue
        if labels_in_mask.size == 1:
            label = int(labels_in_mask[0])
            decision = "single_overlap"
        else:
            label = _majority_label(projected, eroded)
            decision = "multi_overlap_majority"
        if label <= 0:
            continue
        _merge_instance_into_label_map(assigned, label, proposal_binary)
        _write_instance_mask(paths.by_instance_dir, label, frame_id, proposal_binary)
        created.add(label)
        assignments.append(
            {
                "localLabel": local_id,
                "globalLabel": int(label),
                "decision": decision,
                "overlapLabels": [int(value) for value in labels_in_mask.tolist()],
                "proposalPixels": int(np.count_nonzero(proposal_binary)),
            }
        )

    visible_labels = set(int(value) for value in np.unique(projected) if int(value) > 0)
    missing_labels = sorted(visible_labels - created)
    fallback_records: list[dict[str, Any]] = []
    if predictor is not None and missing_labels:
        predictor.set_image(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))

    for label in missing_labels:
        point_match = (proj["indices"].size > 0) & (clustered_labels[proj["indices"]] == int(label))
        if isinstance(point_match, bool) or not np.any(point_match):
            fallback_records.append({"globalLabel": int(label), "decision": "no_points"})
            continue
        points_xy = np.stack([proj["u"][point_match], proj["v"][point_match]], axis=1).astype(np.float32, copy=False)
        if points_xy.shape[0] < int(args.sam2_min_prompt_points):
            fallback_records.append({"globalLabel": int(label), "decision": "too_few_points", "pointCount": int(points_xy.shape[0])})
            continue
        prompts = _greedy_coreset_2d(points_xy, count=int(args.sam2_prompt_count), alpha=float(args.sam2_prompt_alpha))
        prompt_records.append({"label": int(label), "points": prompts.tolist()})
        if predictor is None:
            fallback_records.append({"globalLabel": int(label), "decision": "sam2_disabled", "pointCount": int(points_xy.shape[0])})
            continue
        mask = _predict_sam2_mask(predictor, prompts)
        if mask is None or not np.any(mask):
            fallback_records.append({"globalLabel": int(label), "decision": "sam2_empty", "pointCount": int(points_xy.shape[0])})
            continue
        _merge_instance_into_label_map(sam2_fallback, label, mask)
        _write_instance_mask(paths.by_instance_dir, label, frame_id, mask)
        fallback_records.append(
            {
                "globalLabel": int(label),
                "decision": "sam2_fallback",
                "pointCount": int(points_xy.shape[0]),
                "promptCount": int(prompts.shape[0]),
                "maskPixels": int(np.count_nonzero(mask)),
            }
        )

    final = assigned.copy()
    for label in np.unique(sam2_fallback):
        label_i = int(label)
        if label_i <= 0:
            continue
        _merge_instance_into_label_map(final, label_i, sam2_fallback == label_i, fill_only_empty=True)
    return final, assigned, sam2_fallback, assignments, fallback_records + prompt_records


def view_masks(args: argparse.Namespace, paths: SplitSplatPaths) -> dict[str, Any]:
    rows = _read_jsonl(_require_file(paths.frame_manifest, "Split&Splat frame manifest"))
    points = np.load(_require_file(paths.points_path, "Split&Splat points")).astype(np.float32, copy=False)
    clustered_labels = np.load(_require_file(paths.clustered_point_labels_path, "Split&Splat clustered point labels")).astype(np.int32, copy=False)
    projection_cache = _load_projection_cache(_require_file(paths.projection_cache_path, "Split&Splat projection cache"))

    if paths.view_mask_dir.exists() and bool(args.overwrite):
        shutil.rmtree(paths.view_mask_dir)
    for directory in [
        paths.view_mask_masks_dir,
        paths.view_mask_overlay_dir,
        paths.by_instance_dir,
        paths.projected_masks_dir,
        paths.projected_overlay_dir,
        paths.assigned_masks_dir,
        paths.assigned_overlay_dir,
        paths.prompt_overlay_dir,
        paths.sam2_masks_dir,
        paths.sam2_overlay_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    predictor: Any | None = None
    sam_context: Any = nullcontext()
    sam2_meta: dict[str, Any] = {"enabled": False}
    if not bool(args.skip_sam2_fallback):
        predictor, sam2_root, sam2_checkpoint, sam2_config, sam_context = _build_sam2_image_predictor(args)
        sam2_meta = {
            "enabled": True,
            "sam2Root": sam2_root,
            "sam2Checkpoint": sam2_checkpoint,
            "sam2Config": sam2_config,
        }

    pred_rows: list[dict[str, Any]] = []
    try:
        for row_index, (row, proj) in enumerate(zip(rows, projection_cache, strict=True)):
            frame_id = int(row["splitSplatFrameId"])
            width = int(row["width"])
            height = int(row["height"])
            rgb_bgr = cv2.imread(str(paths.dataset_dir / str(row["colorPath"])), cv2.IMREAD_COLOR)
            if rgb_bgr is None:
                raise ValueError(f"Could not read RGB frame: {paths.dataset_dir / str(row['colorPath'])}")
            proposal_mask = _load_label_map(paths.dataset_dir / str(row["maskPath"]), expected_shape=(height, width))
            projected = _virtual_mask_from_labels(
                proj=proj,
                point_votes=[],
                clustered_point_labels=clustered_labels,
                width=width,
                height=height,
                accuracy=float(args.accuracy_labels),
            )
            final, assigned, sam2_fallback, assignments, prompt_records = _final_masks_for_view(
                row=row,
                proposal_mask=proposal_mask,
                projected=projected,
                proj=proj,
                clustered_labels=clustered_labels,
                rgb_bgr=rgb_bgr,
                predictor=predictor,
                paths=paths,
                args=args,
            )

            final_png = paths.view_mask_masks_dir / f"{frame_id:06d}.png"
            final_npy = paths.view_mask_masks_dir / f"{frame_id:06d}.npy"
            assigned_png = paths.assigned_masks_dir / f"{frame_id:06d}.png"
            assigned_npy = paths.assigned_masks_dir / f"{frame_id:06d}.npy"
            projected_png = paths.projected_masks_dir / f"{frame_id:06d}.png"
            projected_npy = paths.projected_masks_dir / f"{frame_id:06d}.npy"
            sam2_png = paths.sam2_masks_dir / f"{frame_id:06d}.png"
            sam2_npy = paths.sam2_masks_dir / f"{frame_id:06d}.npy"
            cv2.imwrite(str(final_png), final)
            cv2.imwrite(str(assigned_png), assigned)
            cv2.imwrite(str(projected_png), projected)
            cv2.imwrite(str(sam2_png), sam2_fallback)
            np.save(final_npy, final)
            np.save(assigned_npy, assigned)
            np.save(projected_npy, projected)
            np.save(sam2_npy, sam2_fallback)

            final_overlay = paths.view_mask_overlay_dir / f"{frame_id:06d}.png"
            assigned_overlay = paths.assigned_overlay_dir / f"{frame_id:06d}.png"
            projected_overlay = paths.projected_overlay_dir / f"{frame_id:06d}.png"
            prompt_overlay = paths.prompt_overlay_dir / f"{frame_id:06d}.png"
            sam2_overlay = paths.sam2_overlay_dir / f"{frame_id:06d}.png"
            cv2.imwrite(str(final_overlay), cv2.cvtColor(_label_overlay(rgb_bgr, final), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(assigned_overlay), cv2.cvtColor(_label_overlay(rgb_bgr, assigned), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(projected_overlay), cv2.cvtColor(_label_overlay(rgb_bgr, projected), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(prompt_overlay), cv2.cvtColor(_draw_prompt_overlay(rgb_bgr, projected, prompt_records), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(sam2_overlay), cv2.cvtColor(_label_overlay(rgb_bgr, sam2_fallback), cv2.COLOR_RGB2BGR))

            final_coverage = float(np.count_nonzero(final > 0) / max(final.size, 1))
            row_summary = {
                "splitSplatFrameId": frame_id,
                "sourceFrameId": int(row.get("sourceFrameId", frame_id)),
                "scanID": row.get("scanID", ""),
                "frameID": int(row.get("frameID", frame_id)),
                "imageName": row.get("imageName", ""),
                "maskPng": str(final_png),
                "maskNpy": str(final_npy),
                "overlay": str(final_overlay),
                "labelCount": int(np.count_nonzero(np.unique(final) > 0)),
                "coverage": final_coverage,
                "projectedPointMaskNpy": str(projected_npy),
                "projectedPointOverlay": str(projected_overlay),
                "assignedProposalMaskNpy": str(assigned_npy),
                "assignedProposalOverlay": str(assigned_overlay),
                "sam2FallbackMaskNpy": str(sam2_npy),
                "sam2FallbackOverlay": str(sam2_overlay),
                "promptOverlay": str(prompt_overlay),
                "assignments": assignments,
                "promptRecords": prompt_records,
            }
            pred_rows.append(row_summary)
            print(
                f"[split-splat view-mask {row_index + 1:04d}/{len(rows):04d}] "
                f"frame={frame_id} labels={row_summary['labelCount']} cov={final_coverage:.3f}"
            )
    finally:
        if sam_context is not None:
            sam_context.__exit__(None, None, None)

    predictions = paths.view_mask_dir / "predictions.jsonl"
    _write_jsonl(predictions, pred_rows)
    summary = {
        "stage": "view_masks",
        "timestampUtc": _now(),
        "method": "Split&Splat final per-view masks: match final clustered 3D point labels to automatic SAM2 proposals, then use SAM2 coreset point prompts for visible labels without a matched proposal.",
        "frameCount": int(len(rows)),
        "sam2": sam2_meta,
        "coverage": {
            "min": float(min((row["coverage"] for row in pred_rows), default=0.0)),
            "mean": float(np.mean([row["coverage"] for row in pred_rows])) if pred_rows else 0.0,
            "max": float(max((row["coverage"] for row in pred_rows), default=0.0)),
        },
        "outputs": {
            "viewMaskDir": str(paths.view_mask_dir),
            "viewMaskMasksDir": str(paths.view_mask_masks_dir),
            "viewMaskOverlayDir": str(paths.view_mask_overlay_dir),
            "viewMaskPredictions": str(predictions),
            "viewMaskByInstanceDir": str(paths.by_instance_dir),
            "projectedPointMaskDir": str(paths.projected_dir),
            "projectedPointMaskMasksDir": str(paths.projected_masks_dir),
            "projectedPointMaskOverlayDir": str(paths.projected_overlay_dir),
            "assignedProposalMaskDir": str(paths.assigned_dir),
            "assignedProposalMaskMasksDir": str(paths.assigned_masks_dir),
            "assignedProposalMaskOverlayDir": str(paths.assigned_overlay_dir),
            "sam2FallbackMaskDir": str(paths.sam2_dir),
            "sam2FallbackMaskMasksDir": str(paths.sam2_masks_dir),
            "sam2FallbackMaskOverlayDir": str(paths.sam2_overlay_dir),
            "promptOverlayDir": str(paths.prompt_overlay_dir),
        },
    }
    _write_json(paths.summary_dir / "view_masks_summary.json", summary)
    return summary


def _merge_stage_summary(paths: SplitSplatPaths, stage_results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for path in [
        paths.summary_dir / "prepare_summary.json",
        paths.summary_dir / "propagate_summary.json",
        paths.summary_dir / "view_masks_summary.json",
    ]:
        if path.exists():
            summaries[path.stem] = _read_json(path)
    outputs: dict[str, Any] = {
        "datasetDir": str(paths.dataset_dir),
        "imageDir": str(paths.image_dir),
        "poseDir": str(paths.pose_dir),
        "maskDir": str(paths.mask_dir),
        "depthDir": str(paths.depth_dir),
        "sceneMesh": str(paths.scene_mesh),
        "frameManifest": str(paths.frame_manifest),
        "points": str(paths.points_path),
        "pointLabels": str(paths.point_labels_path),
        "clusteredPointLabels": str(paths.clustered_point_labels_path),
        "labeledPoints": str(paths.point_label_dir / "labeled_points.ply"),
        "clusteredLabeledPoints": str(paths.point_label_dir / "clustered_labeled_points.ply"),
        "projectionCache": str(paths.projection_cache_path),
        "viewMaskDir": str(paths.view_mask_dir),
        "viewMaskMasksDir": str(paths.view_mask_masks_dir),
        "viewMaskOverlayDir": str(paths.view_mask_overlay_dir),
        "viewMaskPredictions": str(paths.view_mask_dir / "predictions.jsonl"),
        "viewMaskByInstanceDir": str(paths.by_instance_dir),
        "projectedPointMaskMasksDir": str(paths.projected_masks_dir),
        "projectedPointMaskOverlayDir": str(paths.projected_overlay_dir),
        "assignedProposalMaskMasksDir": str(paths.assigned_masks_dir),
        "assignedProposalMaskOverlayDir": str(paths.assigned_overlay_dir),
        "sam2FallbackMaskMasksDir": str(paths.sam2_masks_dir),
        "sam2FallbackMaskOverlayDir": str(paths.sam2_overlay_dir),
        "promptOverlayDir": str(paths.prompt_overlay_dir),
    }
    payload = {
        "stage": "split_splat",
        "timestampUtc": _now(),
        "baselineName": str(args.baseline_name),
        "sceneName": paths.scene_name,
        "baselineDir": str(paths.baseline_dir),
        "method": "Official-like Split&Splat Split stage port for OMeGa: automatic SAM2 proposals, depth-visible 3D point propagation, point-label voting, DBSCAN cleanup, proposal matching, and SAM2 coreset fallback.",
        "activeStageResults": stage_results,
        "summaries": summaries,
        "outputs": outputs,
    }
    _write_json(paths.summary, payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an official-like Split&Splat mask propagation baseline in OMeGa conventions.")
    parser.add_argument("--baseline", default="split_splat", help=argparse.SUPPRESS)
    parser.add_argument("--stage", choices=("prepare", "propagate", "view_masks", "all"), default="all")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--baseline-name", default="split_splat")
    parser.add_argument("--scene-name", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--proposal-source-name", default="view_proposals_1024")
    parser.add_argument("--proposal-source-dir", type=Path, default=None)
    parser.add_argument("--source-frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--raycast-chunk-rows", type=int, default=64)

    parser.add_argument("--point-source", choices=("mesh_vertices", "mesh_area_samples", "point_cloud"), default="mesh_area_samples")
    parser.add_argument("--point-sample-count", type=int, default=160000)
    parser.add_argument("--point-sample-seed", type=int, default=71)
    parser.add_argument("--point-cloud", type=Path, default=None)
    parser.add_argument("--point-cloud-max-points", type=int, default=0)
    parser.add_argument("--point-cloud-sample-seed", type=int, default=71)

    parser.add_argument("--process-order", choices=("reverse", "forward"), default="reverse")
    parser.add_argument("--projection-epsilon", type=float, default=EPSILON)
    parser.add_argument("--accuracy-labels", type=float, default=ACCURACY_LABELS)
    parser.add_argument("--min-label-points", type=int, default=5)
    parser.add_argument("--cluster-eps", type=float, default=0.10)
    parser.add_argument("--cluster-min-samples", type=int, default=10)
    parser.add_argument("--final-cluster-eps", type=float, default=0.05)
    parser.add_argument("--final-cluster-min-samples", type=int, default=10)

    parser.add_argument("--skip-sam2-fallback", action="store_true")
    parser.add_argument("--view-mask-sam2-root", type=Path, default=THIRD_PARTY_ROOT / "sam2")
    parser.add_argument("--view-mask-sam2-checkpoint", type=Path, default=None)
    parser.add_argument("--view-mask-sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--sam2-prompt-count", type=int, default=10)
    parser.add_argument("--sam2-prompt-alpha", type=float, default=0.3)
    parser.add_argument("--sam2-min-prompt-points", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_dir = args.model_dir.expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {model_dir}")
    paths = resolve_paths(
        model_dir,
        baseline_name=str(args.baseline_name),
        scene_name=args.scene_name,
        output_dir=args.output_dir,
    )
    stages = ["prepare", "propagate", "view_masks"] if args.stage == "all" else [str(args.stage)]
    stage_results: list[dict[str, Any]] = []
    for stage in stages:
        if stage == "prepare":
            stage_results.append(prepare(args, paths))
        elif stage == "propagate":
            stage_results.append(propagate(args, paths))
        elif stage == "view_masks":
            stage_results.append(view_masks(args, paths))
        else:
            raise SystemExit(f"Unsupported stage: {stage}")
    _merge_stage_summary(paths, stage_results, args)
    print(f"[split-splat] Wrote baseline outputs to {paths.baseline_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
