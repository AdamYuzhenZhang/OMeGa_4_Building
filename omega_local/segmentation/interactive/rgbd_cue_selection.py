"""RGB-D cue-selection interactive segmentation.

This follows the CVPR 2016 "Interactive Segmentation on RGBD Images via Cue
Selection" formulation, with one deliberate capture-specific substitution:

1. SLIC superpixels are graph nodes.
2. LAB, predicted depth, and StableNormal cues each build
   foreground/background geodesic confidence maps.
3. Each node receives one of six labels: foreground/background crossed with
   color/depth/normal cue.
4. Alpha-beta swap optimizes the resulting multi-label MRF.
5. GrabCut refines only boundary superpixels.

The original paper derives normals from captured RGB-D depth. In this editor
both depth and normal are predicted, so the normal cue uses StableNormal while
the depth cue still uses Depth Anything V2.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .geometry_selection import (
    GrowSelection,
    _float_param,
    _load_depth,
    _load_normal,
    _resize_rgb_to,
    _strength_name,
)


_LABEL_COLOR_FG = 0
_LABEL_COLOR_BG = 1
_LABEL_DEPTH_FG = 2
_LABEL_DEPTH_BG = 3
_LABEL_NORMAL_FG = 4
_LABEL_NORMAL_BG = 5
_LABELS = tuple(range(6))
_MAX_COST = 1.0e8
_DEFAULT_SUPERPIXELS = 800
_DEFAULT_PAIRWISE_WEIGHT = 0.1
_BACKGROUND_PRIOR_THRESHOLD = 0.5


@dataclass(frozen=True)
class _CueGraph:
    segments: np.ndarray
    adjacency: list[set[int]]
    lab: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    count: np.ndarray


@dataclass(frozen=True)
class _CueRun:
    rgb: np.ndarray
    valid: np.ndarray
    graph: _CueGraph
    prompt_points: list[tuple[int, int, int]]
    fg_seeds: set[int]
    explicit_bg_seeds: set[int]
    auto_bg_seeds: set[int]
    bg_seeds: set[int]
    confidences: tuple[np.ndarray, np.ndarray, np.ndarray]
    data_cost: np.ndarray
    cue_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]]
    cue_sigmas: tuple[float, float, float]
    labels: np.ndarray
    mrf_mask: np.ndarray
    final_mask: np.ndarray
    boundary_ids: set[int]
    smoothness: float
    requested_superpixels: int
    component_seed: tuple[int, int]
    normal_cue_source: str


def cue_select_rgbd(
    *,
    rgb: np.ndarray,
    depth_path: Path,
    normal_path: Path,
    seed_xy: tuple[float, float],
    strength: str,
    smoothness: float | None = None,
    prompts: list[dict[str, Any]] | None = None,
    n_segments: int | None = None,
) -> GrowSelection:
    """Run the RGB-D cue-selection MRF and return the foreground pixels."""

    run = _run_rgbd_cue_selection(
        rgb=rgb,
        depth_path=depth_path,
        normal_path=normal_path,
        seed_xy=seed_xy,
        smoothness=smoothness,
        prompts=prompts,
        n_segments=n_segments,
    )
    return _grow_selection_from_run(run, _strength_name(strength))


def rgbd_cue_debug(
    *,
    rgb: np.ndarray,
    depth_path: Path,
    normal_path: Path,
    seed_xy: tuple[float, float],
    strength: str,
    smoothness: float | None = None,
    prompts: list[dict[str, Any]] | None = None,
    n_segments: int | None = None,
) -> dict[str, Any]:
    """Run RGB-D cue selection and return transparent debug overlays."""

    run = _run_rgbd_cue_selection(
        rgb=rgb,
        depth_path=depth_path,
        normal_path=normal_path,
        seed_xy=seed_xy,
        smoothness=smoothness,
        prompts=prompts,
        n_segments=n_segments,
    )
    selection = _grow_selection_from_run(run, _strength_name(strength)).to_json()
    panels = _debug_panels(run)
    return {
        "mode": "rgbd-cue-select-debug",
        "width": int(run.rgb.shape[1]),
        "height": int(run.rgb.shape[0]),
        "selection": selection,
        "summary": _debug_summary(run),
        "panels": panels,
    }


def _run_rgbd_cue_selection(
    *,
    rgb: np.ndarray,
    depth_path: Path,
    normal_path: Path,
    seed_xy: tuple[float, float],
    smoothness: float | None,
    prompts: list[dict[str, Any]] | None,
    n_segments: int | None,
) -> _CueRun:
    try:
        import cv2  # type: ignore[import-not-found]
        import maxflow  # type: ignore[import-not-found]
        from scipy.sparse import csr_matrix  # type: ignore[import-not-found]
        from scipy.sparse.csgraph import dijkstra  # type: ignore[import-not-found]
        from skimage.segmentation import slic  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - missing dependency path
        raise RuntimeError("RGB-D Cue requires opencv, scipy, scikit-image, and PyMaxflow.") from exc

    depth, valid = _load_depth(depth_path)
    height, width = depth.shape
    rgb = _resize_rgb_to(np.asarray(rgb), width, height)
    normal_feature, normal_valid = _stable_normal_feature(cv2, normal_path, (height, width))
    valid = valid & normal_valid
    seed_x, seed_y = _nearest_valid_evidence_seed(valid, seed_xy)

    segment_count = int(np.clip(int(n_segments or _DEFAULT_SUPERPIXELS), 200, 3200))
    lambda_weight = _float_param(smoothness, _DEFAULT_PAIRWISE_WEIGHT, 0.01, 0.8)

    prompt_points = _prompt_points(prompts, (seed_x, seed_y), width, height)
    component_seed_x, component_seed_y = _foreground_component_seed(prompt_points, (seed_x, seed_y), valid)
    depth_feature = _paper_depth_feature(depth, valid)
    segments = slic(rgb, n_segments=segment_count, compactness=10.0, start_label=0, enforce_connectivity=True, channel_axis=-1)
    graph = _build_cue_graph(cv2, rgb, depth_feature, normal_feature, segments)

    fg_seeds, bg_seeds = _prompt_superpixels(graph.segments, prompt_points)
    explicit_bg_seeds = set(bg_seeds)
    if not fg_seeds:
        fg_seeds.add(int(graph.segments[component_seed_y, component_seed_x]))

    lab_edges, depth_edges, normal_edges = _cue_edge_costs(graph)
    auto_bg_seeds: set[int] = set()
    if not bg_seeds:
        auto_bg_seeds = _boundary_background_prior(
            graph=graph,
            fg_seeds=fg_seeds,
            depth_edges=depth_edges,
            csr_matrix=csr_matrix,
            dijkstra=dijkstra,
        )
        bg_seeds |= auto_bg_seeds
    if not bg_seeds:
        auto_bg_seeds = _image_boundary_superpixels(graph.segments)
        bg_seeds |= auto_bg_seeds
    bg_seeds -= fg_seeds
    auto_bg_seeds = {sid for sid in auto_bg_seeds if sid in bg_seeds}
    if not bg_seeds:
        raise ValueError("RGB-D Cue could not infer background seeds. Add a negative click with the - operation.")

    confidences = _cue_confidences(
        graph=graph,
        fg_seeds=fg_seeds,
        bg_seeds=bg_seeds,
        lab_edges=lab_edges,
        depth_edges=depth_edges,
        normal_edges=normal_edges,
        csr_matrix=csr_matrix,
        dijkstra=dijkstra,
    )
    data_cost = _data_costs(confidences, fg_seeds, bg_seeds)
    cue_sigmas = _cue_sigmas(lab_edges, depth_edges, normal_edges)
    labels = _optimize_alpha_beta(
        maxflow_module=maxflow,
        graph=graph,
        data_cost=data_cost,
        cue_edges=(lab_edges, depth_edges, normal_edges),
        cue_sigmas=cue_sigmas,
        smoothness=lambda_weight,
        iterations=5,
    )

    super_fg = (labels % 2) == 0
    mrf_mask = _enforce_seed_superpixels(super_fg[graph.segments], graph.segments, fg_seeds, bg_seeds)
    boundary_ids = _boundary_superpixels(graph.segments, mrf_mask)
    final_mask = _grabcut_boundary_refine(cv2, rgb, graph.segments, mrf_mask, prompt_points, fg_seeds, bg_seeds)
    final_mask = _enforce_seed_superpixels(final_mask, graph.segments, fg_seeds, bg_seeds)
    final_mask &= valid

    return _CueRun(
        rgb=rgb,
        valid=valid,
        graph=graph,
        prompt_points=prompt_points,
        fg_seeds=fg_seeds,
        explicit_bg_seeds=explicit_bg_seeds,
        auto_bg_seeds=auto_bg_seeds,
        bg_seeds=bg_seeds,
        confidences=confidences,
        data_cost=data_cost,
        cue_edges=(lab_edges, depth_edges, normal_edges),
        cue_sigmas=cue_sigmas,
        labels=labels,
        mrf_mask=mrf_mask,
        final_mask=final_mask,
        boundary_ids=boundary_ids,
        smoothness=lambda_weight,
        requested_superpixels=segment_count,
        component_seed=(component_seed_x, component_seed_y),
        normal_cue_source="stable_normal",
    )


def _grow_selection_from_run(run: _CueRun, strength_name: str) -> GrowSelection:
    label_hist = np.bincount(run.labels, minlength=6).astype(np.int64)
    component_seed_x, component_seed_y = run.component_seed
    return GrowSelection(
        mode="rgbd-cue-select",
        strength=strength_name,
        seed=(component_seed_x, component_seed_y),
        mask=run.final_mask,
        parameters={
            "smoothnessLambda": float(run.smoothness),
            "superpixels": float(len(run.graph.count)),
            "requestedSuperpixels": float(run.requested_superpixels),
            "foregroundSeeds": float(len(run.fg_seeds)),
            "backgroundSeeds": float(len(run.bg_seeds)),
            "enforcedSeedSuperpixels": float(len(run.fg_seeds) + len(run.bg_seeds)),
            "backgroundPriorThreshold": float(_BACKGROUND_PRIOR_THRESHOLD),
            "normalCueUsesStableNormal": 1.0,
            "componentSeedX": float(component_seed_x),
            "componentSeedY": float(component_seed_y),
            "colorFgLabels": float(label_hist[_LABEL_COLOR_FG]),
            "colorBgLabels": float(label_hist[_LABEL_COLOR_BG]),
            "depthFgLabels": float(label_hist[_LABEL_DEPTH_FG]),
            "depthBgLabels": float(label_hist[_LABEL_DEPTH_BG]),
            "normalFgLabels": float(label_hist[_LABEL_NORMAL_FG]),
            "normalBgLabels": float(label_hist[_LABEL_NORMAL_BG]),
        },
    )


def _debug_summary(run: _CueRun) -> dict[str, Any]:
    data_energy, smooth_energy = _energy_terms(
        run.graph,
        run.labels,
        run.data_cost,
        run.cue_edges,
        run.cue_sigmas,
        run.smoothness,
    )
    changed = np.count_nonzero(run.mrf_mask != run.final_mask)
    label_hist = np.bincount(run.labels, minlength=6).astype(np.int64)
    return {
        "smoothnessLambda": float(run.smoothness),
        "superpixels": int(len(run.graph.count)),
        "requestedSuperpixels": int(run.requested_superpixels),
        "foregroundSeeds": int(len(run.fg_seeds)),
        "explicitBackgroundSeeds": int(len(run.explicit_bg_seeds)),
        "autoBackgroundSeeds": int(len(run.auto_bg_seeds)),
        "backgroundSeeds": int(len(run.bg_seeds)),
        "normalCueSource": run.normal_cue_source,
        "mrfAreaPixels": int(np.count_nonzero(run.mrf_mask)),
        "finalAreaPixels": int(np.count_nonzero(run.final_mask)),
        "grabCutChangedPixels": int(changed),
        "dataEnergy": float(data_energy),
        "smoothEnergy": float(smooth_energy),
        "totalEnergy": float(data_energy + smooth_energy),
        "labelCounts": {
            "colorFg": int(label_hist[_LABEL_COLOR_FG]),
            "colorBg": int(label_hist[_LABEL_COLOR_BG]),
            "depthFg": int(label_hist[_LABEL_DEPTH_FG]),
            "depthBg": int(label_hist[_LABEL_DEPTH_BG]),
            "normalFg": int(label_hist[_LABEL_NORMAL_FG]),
            "normalBg": int(label_hist[_LABEL_NORMAL_BG]),
        },
    }


def _debug_panels(run: _CueRun) -> list[dict[str, str]]:
    panels = [
        (
            "seeds",
            "Seeds + Superpixels",
            "SLIC boundaries, foreground seed superpixels, explicit background seeds, and automatic boundary background seeds.",
            _seed_superpixel_overlay(run),
        ),
        (
            "color-confidence",
            "Color Confidence",
            "Foreground confidence from LAB geodesic distances.",
            _confidence_overlay(run, 0),
        ),
        (
            "depth-confidence",
            "Depth Confidence",
            "Foreground confidence from depth geodesic distances.",
            _confidence_overlay(run, 1),
        ),
        (
            "normal-confidence",
            "Normal Confidence",
            "Foreground confidence from StableNormal geodesic distances.",
            _confidence_overlay(run, 2),
        ),
        (
            "cue-winner",
            "Cue Winner",
            "Final optimized six-label state: cue type crossed with foreground/background.",
            _cue_winner_overlay(run),
        ),
        (
            "foreground-margin",
            "Foreground Margin",
            "Best background unary minus best foreground unary; green favors foreground, magenta favors background, gray is ambiguous.",
            _margin_overlay(run),
        ),
        (
            "cut-cost",
            "Cut Edge Cost",
            "MRF cut edges over superpixel boundaries. Cyan is easy to cut; red is expensive but was still cut.",
            _cut_cost_overlay(run),
        ),
        (
            "grabcut-change",
            "GrabCut Change",
            "Boundary refinement changes: green added pixels, magenta removed pixels, white final boundary.",
            _grabcut_change_overlay(run),
        ),
    ]
    return [
        {
            "id": panel_id,
            "label": label,
            "description": description,
            "overlayPng": _encode_rgba_png(image),
        }
        for panel_id, label, description, image in panels
    ]


def _seed_superpixel_overlay(run: _CueRun) -> np.ndarray:
    segments = run.graph.segments
    rgba = np.zeros(segments.shape + (4,), dtype=np.uint8)
    boundary = _superpixel_boundary_mask(segments)
    rgba[boundary] = np.array([255, 255, 255, 90], dtype=np.uint8)
    if run.auto_bg_seeds:
        rgba[np.isin(segments, np.asarray(sorted(run.auto_bg_seeds), dtype=np.int32))] = np.array([255, 171, 64, 108], dtype=np.uint8)
    if run.explicit_bg_seeds:
        rgba[np.isin(segments, np.asarray(sorted(run.explicit_bg_seeds), dtype=np.int32))] = np.array([255, 78, 92, 128], dtype=np.uint8)
    if run.fg_seeds:
        rgba[np.isin(segments, np.asarray(sorted(run.fg_seeds), dtype=np.int32))] = np.array([69, 255, 159, 132], dtype=np.uint8)
    rgba[boundary] = np.maximum(rgba[boundary], np.array([255, 255, 255, 100], dtype=np.uint8))
    return rgba


def _confidence_overlay(run: _CueRun, cue_index: int) -> np.ndarray:
    values = run.confidences[cue_index][run.graph.segments]
    rgba = _scalar_ramp_rgba(values, _CONFIDENCE_RAMP, alpha=142)
    rgba[~run.valid, 3] = 0
    return rgba


def _cue_winner_overlay(run: _CueRun) -> np.ndarray:
    labels = run.labels[run.graph.segments]
    colors = np.array(
        [
            [255, 180,  58, 145],  # color foreground
            [118,  77,  41, 125],  # color background
            [ 72, 220, 255, 145],  # depth foreground
            [ 55,  92, 222, 125],  # depth background
            [ 91, 239, 126, 145],  # normal foreground
            [200,  96, 255, 125],  # normal background
        ],
        dtype=np.uint8,
    )
    rgba = colors[np.clip(labels, 0, 5)]
    rgba = rgba.copy()
    rgba[~run.valid, 3] = 0
    rgba[_superpixel_boundary_mask(run.graph.segments)] = np.array([255, 255, 255, 115], dtype=np.uint8)
    return rgba


def _margin_overlay(run: _CueRun) -> np.ndarray:
    fg_cost = np.min(run.data_cost[:, [_LABEL_COLOR_FG, _LABEL_DEPTH_FG, _LABEL_NORMAL_FG]], axis=1)
    bg_cost = np.min(run.data_cost[:, [_LABEL_COLOR_BG, _LABEL_DEPTH_BG, _LABEL_NORMAL_BG]], axis=1)
    margin = (bg_cost - fg_cost)[run.graph.segments]
    rgba = _signed_margin_rgba(margin)
    rgba[~run.valid, 3] = 0
    rgba[_superpixel_boundary_mask(run.graph.segments)] = np.array([255, 255, 255, 70], dtype=np.uint8)
    return rgba


def _cut_cost_overlay(run: _CueRun) -> np.ndarray:
    segments = run.graph.segments
    rgba = np.zeros(segments.shape + (4,), dtype=np.uint8)
    boundary = _superpixel_boundary_mask(segments)
    rgba[boundary] = np.array([255, 255, 255, 46], dtype=np.uint8)
    cut_cost = _cut_edge_cost_image(run)
    cut = np.isfinite(cut_cost)
    if np.any(cut):
        denom = max(float(run.smoothness), 1.0e-6)
        t = np.clip(cut_cost[cut] / denom, 0.0, 1.0)
        rgba[cut] = _ramp_colors(t, _CUT_COST_RAMP, alpha=225)
    return rgba


def _grabcut_change_overlay(run: _CueRun) -> np.ndarray:
    rgba = np.zeros(run.final_mask.shape + (4,), dtype=np.uint8)
    added = run.final_mask & ~run.mrf_mask
    removed = run.mrf_mask & ~run.final_mask
    final_boundary = _mask_boundary(run.final_mask)
    rgba[added] = np.array([75, 255, 142, 135], dtype=np.uint8)
    rgba[removed] = np.array([255, 80, 205, 145], dtype=np.uint8)
    rgba[final_boundary] = np.array([255, 255, 255, 210], dtype=np.uint8)
    return rgba


def _cut_edge_cost_image(run: _CueRun) -> np.ndarray:
    segments = run.graph.segments
    labels = run.labels
    mask = run.mrf_mask
    out = np.full(segments.shape, np.nan, dtype=np.float32)
    cache: dict[tuple[int, int], float] = {}
    for dy, dx in ((0, 1), (1, 0)):
        a_seg = segments[: segments.shape[0] - dy if dy else segments.shape[0], : segments.shape[1] - dx if dx else segments.shape[1]]
        b_seg = segments[dy:, dx:]
        a_mask = mask[: mask.shape[0] - dy if dy else mask.shape[0], : mask.shape[1] - dx if dx else mask.shape[1]]
        b_mask = mask[dy:, dx:]
        changed = (a_seg != b_seg) & (a_mask != b_mask)
        if not np.any(changed):
            continue
        aa = a_seg[changed].astype(np.int32, copy=False)
        bb = b_seg[changed].astype(np.int32, copy=False)
        values = np.empty(aa.shape[0], dtype=np.float32)
        for index, (u_raw, v_raw) in enumerate(zip(aa.tolist(), bb.tolist())):
            u = int(u_raw)
            v = int(v_raw)
            key = (u, v) if u < v else (v, u)
            if key not in cache:
                cache[key] = _pairwise_label_cost(u, v, int(labels[u]), int(labels[v]), run.cue_edges, run.cue_sigmas, run.smoothness)
            values[index] = float(cache[key])
        view_a = out[: out.shape[0] - dy if dy else out.shape[0], : out.shape[1] - dx if dx else out.shape[1]]
        view_b = out[dy:, dx:]
        view_a[changed] = values
        view_b[changed] = values
    return out


def _superpixel_boundary_mask(segments: np.ndarray) -> np.ndarray:
    boundary = np.zeros(segments.shape, dtype=bool)
    if segments.shape[1] > 1:
        changed = segments[:, 1:] != segments[:, :-1]
        boundary[:, 1:] |= changed
        boundary[:, :-1] |= changed
    if segments.shape[0] > 1:
        changed = segments[1:, :] != segments[:-1, :]
        boundary[1:, :] |= changed
        boundary[:-1, :] |= changed
    return boundary


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    if mask.shape[1] > 1:
        changed = mask[:, 1:] != mask[:, :-1]
        boundary[:, 1:] |= changed
        boundary[:, :-1] |= changed
    if mask.shape[0] > 1:
        changed = mask[1:, :] != mask[:-1, :]
        boundary[1:, :] |= changed
        boundary[:-1, :] |= changed
    return boundary


def _scalar_ramp_rgba(values: np.ndarray, ramp: np.ndarray, *, alpha: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    colors = _ramp_colors(np.clip(values.reshape(-1), 0.0, 1.0), ramp, alpha=alpha)
    return colors.reshape(values.shape + (4,))


def _signed_margin_rgba(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape + (4,), dtype=np.uint8)
    scale = max(float(np.percentile(np.abs(finite), 92.0)), 1.0e-6)
    t = np.clip(values / scale, -1.0, 1.0)
    rgba = np.zeros(values.shape + (4,), dtype=np.uint8)
    positive = t >= 0
    strength = np.abs(t)
    rgba[..., 0] = np.where(positive, 88, 255).astype(np.uint8)
    rgba[..., 1] = np.where(positive, 235, 88).astype(np.uint8)
    rgba[..., 2] = np.where(positive, 136, 212).astype(np.uint8)
    rgba[..., 3] = np.rint(55 + 155 * strength).astype(np.uint8)
    rgba[~np.isfinite(values), 3] = 0
    return rgba


def _ramp_colors(values: np.ndarray, ramp: np.ndarray, *, alpha: int) -> np.ndarray:
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0).reshape(-1)
    stops = ramp[:, 0]
    colors = ramp[:, 1:]
    out = np.stack([np.interp(x, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    rgba = np.zeros((x.shape[0], 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    rgba[:, 3] = int(alpha)
    return rgba


def _encode_rgba_png(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


_CONFIDENCE_RAMP = np.array(
    [
        [0.00,  42,  55, 125],
        [0.25,  52, 143, 210],
        [0.50, 220, 220, 220],
        [0.75, 245, 169,  70],
        [1.00, 235,  76,  72],
    ],
    dtype=np.float32,
)

_CUT_COST_RAMP = np.array(
    [
        [0.00,  68, 235, 255],
        [0.45, 252, 220,  83],
        [1.00, 255,  78,  92],
    ],
    dtype=np.float32,
)


def _nearest_valid_evidence_seed(valid: np.ndarray, seed_xy: tuple[float, float]) -> tuple[int, int]:
    from .geometry_selection import _nearest_valid_seed

    return _nearest_valid_seed(valid, seed_xy, valid.shape[1], valid.shape[0], max_radius=24)


def _prompt_points(
    prompts: list[dict[str, Any]] | None,
    default_seed: tuple[int, int],
    width: int,
    height: int,
) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for prompt in prompts or []:
        try:
            x = int(round(float(prompt.get("sourceX", prompt.get("x", default_seed[0])))))
            y = int(round(float(prompt.get("sourceY", prompt.get("y", default_seed[1])))))
            label = 1 if int(prompt.get("label", 1)) > 0 else 0
        except (TypeError, ValueError):
            continue
        if 0 <= x < width and 0 <= y < height:
            out.append((x, y, label))
    if not out:
        out.append((int(default_seed[0]), int(default_seed[1]), 1))
    return out


def _foreground_component_seed(
    prompts: list[tuple[int, int, int]],
    fallback_seed: tuple[int, int],
    valid: np.ndarray,
) -> tuple[int, int]:
    for x, y, label in reversed(prompts):
        if label <= 0:
            continue
        if 0 <= x < valid.shape[1] and 0 <= y < valid.shape[0] and valid[y, x]:
            return int(x), int(y)
    return int(fallback_seed[0]), int(fallback_seed[1])


def _paper_depth_feature(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(depth, dtype=np.float32).copy()
    finite = depth[valid & np.isfinite(depth) & (depth > 0)]
    if finite.size == 0:
        return np.zeros(depth.shape, dtype=np.float32)
    out[~(valid & np.isfinite(out) & (out > 0))] = float(np.median(finite))
    return out.astype(np.float32, copy=False)


def _stable_normal_feature(cv2: Any, normal_path: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    normal, valid = _load_normal(normal_path)
    height, width = int(shape[0]), int(shape[1])
    if normal.shape[0] != height or normal.shape[1] != width:
        normal = cv2.resize(normal.astype(np.float32, copy=False), (width, height), interpolation=cv2.INTER_LINEAR)
        valid = cv2.resize(valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    normal = _normalize_vectors(normal)
    valid &= np.isfinite(normal).all(axis=-1)
    normal = normal.astype(np.float32, copy=True)
    normal[~valid] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return normal, valid


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return (vectors / np.maximum(norm, 1.0e-8)).astype(np.float32, copy=False)


def _build_cue_graph(cv2: Any, rgb: np.ndarray, depth: np.ndarray, normal: np.ndarray, segments: np.ndarray) -> _CueGraph:
    labels = segments.astype(np.int32, copy=False)
    n = int(labels.max()) + 1
    flat = labels.reshape(-1)
    count = np.bincount(flat, minlength=n).astype(np.float64)
    count_safe = np.maximum(count, 1.0)

    lab_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = np.zeros((n, 3), dtype=np.float64)
    for channel in range(3):
        lab[:, channel] = np.bincount(flat, weights=lab_img[..., channel].reshape(-1), minlength=n) / count_safe

    depth_mean = np.bincount(flat, weights=depth.reshape(-1), minlength=n) / count_safe

    normal_mean = np.zeros((n, 3), dtype=np.float64)
    for channel in range(3):
        normal_mean[:, channel] = np.bincount(flat, weights=normal[..., channel].reshape(-1), minlength=n) / count_safe
    normal_norm = np.linalg.norm(normal_mean, axis=1, keepdims=True)
    normal_mean = normal_mean / np.maximum(normal_norm, 1.0e-8)

    adjacency = _segment_adjacency(labels, n)
    return _CueGraph(
        segments=labels,
        adjacency=adjacency,
        lab=lab.astype(np.float32),
        depth=depth_mean.astype(np.float32),
        normal=normal_mean.astype(np.float32),
        count=count.astype(np.int64),
    )


def _segment_adjacency(labels: np.ndarray, n: int) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(n)]
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = labels[max(0, dy) : labels.shape[0] + min(0, dy), max(0, dx) : labels.shape[1] + min(0, dx)]
        b = labels[max(0, -dy) : labels.shape[0] - max(0, dy), max(0, -dx) : labels.shape[1] - max(0, dx)]
        changed = a != b
        if not np.any(changed):
            continue
        aa = a[changed].reshape(-1)
        bb = b[changed].reshape(-1)
        for u, v in zip(aa.tolist(), bb.tolist()):
            if u == v:
                continue
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
    return adjacency


def _prompt_superpixels(segments: np.ndarray, prompts: list[tuple[int, int, int]]) -> tuple[set[int], set[int]]:
    fg: set[int] = set()
    bg: set[int] = set()
    for x, y, label in prompts:
        sid = int(segments[y, x])
        if label > 0:
            fg.add(sid)
        else:
            bg.add(sid)
    return fg, bg


def _enforce_seed_superpixels(mask: np.ndarray, segments: np.ndarray, fg_seeds: set[int], bg_seeds: set[int]) -> np.ndarray:
    out = np.asarray(mask, dtype=bool).copy()
    if fg_seeds:
        out[np.isin(segments, np.asarray(sorted(fg_seeds), dtype=np.int32))] = True
    if bg_seeds:
        out[np.isin(segments, np.asarray(sorted(bg_seeds), dtype=np.int32))] = False
    return out


def _cue_edge_costs(graph: _CueGraph) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]]:
    lab_edges: dict[tuple[int, int], float] = {}
    depth_edges: dict[tuple[int, int], float] = {}
    normal_edges: dict[tuple[int, int], float] = {}
    for u, neighbors in enumerate(graph.adjacency):
        for v in neighbors:
            if v <= u:
                continue
            key = (u, v)
            lab_edges[key] = float(np.linalg.norm(graph.lab[u] - graph.lab[v]))
            depth_edges[key] = float(abs(float(graph.depth[u]) - float(graph.depth[v])))
            cosine = float(np.dot(graph.normal[u], graph.normal[v]))
            normal_edges[key] = float(1.0 - np.clip(cosine, -1.0, 1.0))
    return lab_edges, depth_edges, normal_edges


def _boundary_background_prior(
    *,
    graph: _CueGraph,
    fg_seeds: set[int],
    depth_edges: dict[tuple[int, int], float],
    csr_matrix: Any,
    dijkstra: Any,
) -> set[int]:
    boundary = _image_boundary_superpixels(graph.segments)
    if not boundary:
        return set()
    depth_dist = _dijkstra_from_seeds(graph, depth_edges, fg_seeds, csr_matrix, dijkstra)
    finite = depth_dist[np.isfinite(depth_dist)]
    if finite.size == 0:
        return set(boundary)
    max_dist = max(float(np.max(finite)), 1.0e-8)
    return {sid for sid in boundary if float(depth_dist[sid]) / max_dist > _BACKGROUND_PRIOR_THRESHOLD}


def _image_boundary_superpixels(segments: np.ndarray) -> set[int]:
    ids: set[int] = set()
    ids.update(np.unique(segments[0, :]).astype(int).tolist())
    ids.update(np.unique(segments[-1, :]).astype(int).tolist())
    ids.update(np.unique(segments[:, 0]).astype(int).tolist())
    ids.update(np.unique(segments[:, -1]).astype(int).tolist())
    return ids


def _cue_confidences(
    *,
    graph: _CueGraph,
    fg_seeds: set[int],
    bg_seeds: set[int],
    lab_edges: dict[tuple[int, int], float],
    depth_edges: dict[tuple[int, int], float],
    normal_edges: dict[tuple[int, int], float],
    csr_matrix: Any,
    dijkstra: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _confidence_for_edges(graph, lab_edges, fg_seeds, bg_seeds, csr_matrix, dijkstra),
        _confidence_for_edges(graph, depth_edges, fg_seeds, bg_seeds, csr_matrix, dijkstra),
        _confidence_for_edges(graph, normal_edges, fg_seeds, bg_seeds, csr_matrix, dijkstra),
    )


def _confidence_for_edges(
    graph: _CueGraph,
    edges: dict[tuple[int, int], float],
    fg_seeds: set[int],
    bg_seeds: set[int],
    csr_matrix: Any,
    dijkstra: Any,
) -> np.ndarray:
    d_fg = _dijkstra_from_seeds(graph, edges, fg_seeds, csr_matrix, dijkstra)
    d_bg = _dijkstra_from_seeds(graph, edges, bg_seeds, csr_matrix, dijkstra)
    denom = d_fg + d_bg
    confidence = d_bg / np.maximum(denom, 1.0e-8)
    confidence[~np.isfinite(confidence)] = 0.5
    return np.clip(confidence, 0.0, 1.0).astype(np.float32, copy=False)


def _dijkstra_from_seeds(
    graph: _CueGraph,
    edges: dict[tuple[int, int], float],
    seeds: set[int],
    csr_matrix: Any,
    dijkstra: Any,
) -> np.ndarray:
    n = len(graph.count)
    if not seeds:
        return np.full(n, np.inf, dtype=np.float32)
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []
    values = np.asarray(list(edges.values()), dtype=np.float32)
    fallback = float(np.percentile(values, 25.0)) if values.size else 1.0
    fallback = max(fallback, 1.0e-6)
    min_weights = []
    for u, neighbors in enumerate(graph.adjacency):
        local = []
        for v in neighbors:
            key = (u, v) if u < v else (v, u)
            local.append(float(edges.get(key, fallback)))
        if local:
            min_weights.append(min(local))
    average_min_weight = float(np.mean(min_weights)) if min_weights else fallback
    for u, neighbors in enumerate(graph.adjacency):
        for v in neighbors:
            key = (u, v) if u < v else (v, u)
            cost = float(edges.get(key, fallback))
            if cost < average_min_weight:
                cost /= 3.0
            row.append(u)
            col.append(v)
            data.append(max(cost, 1.0e-6))
    matrix = csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)
    dist = dijkstra(matrix, directed=False, indices=np.asarray(sorted(seeds), dtype=np.int32), min_only=True)
    return np.asarray(dist, dtype=np.float32)


def _data_costs(confidences: tuple[np.ndarray, np.ndarray, np.ndarray], fg_seeds: set[int], bg_seeds: set[int]) -> np.ndarray:
    n = confidences[0].shape[0]
    data = np.zeros((n, 6), dtype=np.float64)
    for cue_idx, confidence in enumerate(confidences):
        fg_label = cue_idx * 2
        bg_label = cue_idx * 2 + 1
        data[:, fg_label] = 1.0 - confidence
        data[:, bg_label] = confidence
    for sid in fg_seeds:
        data[sid, 1::2] = _MAX_COST
        data[sid, 0::2] = 0.0
    for sid in bg_seeds:
        data[sid, 0::2] = _MAX_COST
        data[sid, 1::2] = 0.0
    return data


def _cue_sigmas(
    lab_edges: dict[tuple[int, int], float],
    depth_edges: dict[tuple[int, int], float],
    normal_edges: dict[tuple[int, int], float],
) -> tuple[float, float, float]:
    return (_sigma(lab_edges, 1.0), _sigma(depth_edges, 1.0), _sigma(normal_edges, 0.5))


def _sigma(edges: dict[tuple[int, int], float], scale: float) -> float:
    if not edges:
        return 1.0
    values = np.asarray(list(edges.values()), dtype=np.float64)
    sigma = float(np.max(values))
    return max(sigma * sigma * float(scale), 1.0e-12)


def _optimize_alpha_beta(
    *,
    maxflow_module: Any,
    graph: _CueGraph,
    data_cost: np.ndarray,
    cue_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]],
    cue_sigmas: tuple[float, float, float],
    smoothness: float,
    iterations: int,
) -> np.ndarray:
    labels = np.argmin(data_cost, axis=1).astype(np.int32)
    old_energy = _energy(graph, labels, data_cost, cue_edges, cue_sigmas, smoothness)
    for _iteration in range(max(1, int(iterations))):
        for alpha in _LABELS:
            for beta in _LABELS:
                if beta <= alpha:
                    continue
                labels = _alpha_beta_swap(maxflow_module, graph, labels, data_cost, cue_edges, cue_sigmas, smoothness, alpha, beta)
        new_energy = _energy(graph, labels, data_cost, cue_edges, cue_sigmas, smoothness)
        if new_energy >= old_energy - 1.0e-7:
            break
        old_energy = new_energy
    return labels


def _alpha_beta_swap(
    maxflow_module: Any,
    graph: _CueGraph,
    labels: np.ndarray,
    data_cost: np.ndarray,
    cue_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]],
    cue_sigmas: tuple[float, float, float],
    smoothness: float,
    alpha: int,
    beta: int,
) -> np.ndarray:
    active = np.nonzero((labels == alpha) | (labels == beta))[0]
    if active.size == 0:
        return labels
    active_index = {int(node): idx for idx, node in enumerate(active.tolist())}
    graph_cut = maxflow_module.Graph[float](int(active.size), int(active.size) * 6)
    nodes = graph_cut.add_nodes(int(active.size))

    for node in active.tolist():
        idx = active_index[int(node)]
        alpha_cost = float(data_cost[node, alpha])
        beta_cost = float(data_cost[node, beta])
        for neighbor in graph.adjacency[node]:
            if neighbor in active_index:
                continue
            alpha_cost += _pairwise_label_cost(node, neighbor, alpha, int(labels[neighbor]), cue_edges, cue_sigmas, smoothness)
            beta_cost += _pairwise_label_cost(node, neighbor, beta, int(labels[neighbor]), cue_edges, cue_sigmas, smoothness)
        graph_cut.add_tedge(nodes[idx], beta_cost, alpha_cost)

    for node in active.tolist():
        idx = active_index[int(node)]
        for neighbor in graph.adjacency[node]:
            if neighbor <= node or neighbor not in active_index:
                continue
            jdx = active_index[int(neighbor)]
            weight = _pairwise_label_cost(node, neighbor, alpha, beta, cue_edges, cue_sigmas, smoothness)
            graph_cut.add_edge(nodes[idx], nodes[jdx], weight, weight)

    graph_cut.maxflow()
    next_labels = labels.copy()
    for node in active.tolist():
        idx = active_index[int(node)]
        next_labels[node] = alpha if graph_cut.get_segment(nodes[idx]) == 0 else beta
    return next_labels


def _energy(
    graph: _CueGraph,
    labels: np.ndarray,
    data_cost: np.ndarray,
    cue_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]],
    cue_sigmas: tuple[float, float, float],
    smoothness: float,
) -> float:
    data_energy, smooth_energy = _energy_terms(graph, labels, data_cost, cue_edges, cue_sigmas, smoothness)
    return data_energy + smooth_energy


def _energy_terms(
    graph: _CueGraph,
    labels: np.ndarray,
    data_cost: np.ndarray,
    cue_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]],
    cue_sigmas: tuple[float, float, float],
    smoothness: float,
) -> tuple[float, float]:
    total = float(np.sum(data_cost[np.arange(labels.shape[0]), labels]))
    smooth_total = 0.0
    for u, neighbors in enumerate(graph.adjacency):
        for v in neighbors:
            if v <= u:
                continue
            smooth_total += _pairwise_label_cost(u, v, int(labels[u]), int(labels[v]), cue_edges, cue_sigmas, smoothness)
    return total, smooth_total


def _pairwise_label_cost(
    u: int,
    v: int,
    label_u: int,
    label_v: int,
    cue_edges: tuple[dict[tuple[int, int], float], dict[tuple[int, int], float], dict[tuple[int, int], float]],
    cue_sigmas: tuple[float, float, float],
    smoothness: float,
) -> float:
    if label_u == label_v:
        return 0.0
    cue_u = label_u // 2
    cue_v = label_v // 2
    key = (u, v) if u < v else (v, u)
    w_u = _smooth_weight_for_cue(cue_edges[cue_u].get(key, 0.0), cue_sigmas[cue_u], smoothness)
    w_v = _smooth_weight_for_cue(cue_edges[cue_v].get(key, 0.0), cue_sigmas[cue_v], smoothness)
    return min(w_u, w_v)


def _smooth_weight_for_cue(distance: float, sigma_squared: float, smoothness: float) -> float:
    return float(smoothness) * float(np.exp(-(float(distance) ** 2) / max(2.0 * float(sigma_squared), 1.0e-12)))


def _grabcut_boundary_refine(
    cv2: Any,
    rgb: np.ndarray,
    segments: np.ndarray,
    mask: np.ndarray,
    prompts: list[tuple[int, int, int]],
    fg_seeds: set[int],
    bg_seeds: set[int],
) -> np.ndarray:
    if not np.any(mask) or np.all(mask):
        return mask
    boundary_ids = _boundary_superpixels(segments, mask)
    if not boundary_ids:
        return mask
    gc_mask = np.where(mask, cv2.GC_FGD, cv2.GC_BGD).astype(np.uint8)
    boundary = np.isin(segments, np.asarray(sorted(boundary_ids), dtype=np.int32))
    gc_mask[boundary & mask] = cv2.GC_PR_FGD
    gc_mask[boundary & ~mask] = cv2.GC_PR_BGD
    if fg_seeds:
        gc_mask[np.isin(segments, np.asarray(sorted(fg_seeds), dtype=np.int32))] = cv2.GC_FGD
    if bg_seeds:
        gc_mask[np.isin(segments, np.asarray(sorted(bg_seeds), dtype=np.int32))] = cv2.GC_BGD
    for x, y, label in prompts:
        radius = 4
        y0 = max(0, y - radius)
        y1 = min(gc_mask.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(gc_mask.shape[1], x + radius + 1)
        gc_mask[y0:y1, x0:x1] = cv2.GC_FGD if label > 0 else cv2.GC_BGD
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return mask
    return (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)


def _boundary_superpixels(segments: np.ndarray, mask: np.ndarray) -> set[int]:
    ids: set[int] = set()
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a_seg = segments[max(0, dy) : segments.shape[0] + min(0, dy), max(0, dx) : segments.shape[1] + min(0, dx)]
        b_seg = segments[max(0, -dy) : segments.shape[0] - max(0, dy), max(0, -dx) : segments.shape[1] - max(0, dx)]
        a_mask = mask[max(0, dy) : mask.shape[0] + min(0, dy), max(0, dx) : mask.shape[1] + min(0, dx)]
        b_mask = mask[max(0, -dy) : mask.shape[0] - max(0, dy), max(0, -dx) : mask.shape[1] - max(0, dx)]
        changed = a_mask != b_mask
        if not np.any(changed):
            continue
        ids.update(a_seg[changed].astype(int).tolist())
        ids.update(b_seg[changed].astype(int).tolist())
    return ids
