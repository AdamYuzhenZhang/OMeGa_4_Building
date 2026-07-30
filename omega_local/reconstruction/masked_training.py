"""Shared loss helpers for object-wise Gaussian reconstruction."""

from __future__ import annotations

import json
from pathlib import Path


_MASK_LOSS_LINE = "        Ll1_mask = l1_loss(mask, GT_mask)\n"
_RGB_LOSS_LINE = "        Ll1 = l1_loss(image, gt_image)\n"
_SCENE_LINE = "    scene = Scene(dataset, gaussians)\n"
_MASK_LOSS_EXPRESSION = "Ll1_mask * 0.25"
_DENSIFY_CONDITION_LINE = (
    "            if iteration < opt.densify_until_iter < opt.max_num_splats:\n"
)
_DENSIFY_CALL_LINE = (
    "                    gaussians.densify_and_prune("
    "opt.densify_grad_threshold, 0.005, scene.cameras_extent, "
    "size_threshold, radii)\n"
)


def load_normalized_view_weights(
    path: Path,
    *,
    mask_dir: Path,
) -> dict[str, float]:
    """Normalize manual/propagated weights over this region's positive views."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = {
        str(row.get("imageStem") or ""): max(
            float(row.get("weight", 1.0)),
            0.0,
        )
        for row in payload.get("frames", [])
        if str(row.get("imageStem") or "")
    }
    eligible = {
        item.stem for item in mask_dir.glob("*.png") if item.is_file()
    }
    if not eligible:
        raise ValueError(f"Region mask directory is empty: {mask_dir}")
    mean = sum(raw.get(stem, 1.0) for stem in eligible) / len(eligible)
    if mean <= 0:
        raise ValueError(f"Region view weights have zero mean: {path}")
    return {stem: value / mean for stem, value in raw.items()}


def sanitize_identity_masks(scene) -> None:
    """Premultiply identity RGB by the alpha that the camera loader drops."""
    for attribute in ("train_cameras", "test_cameras", "black_cameras"):
        camera_sets = getattr(scene, attribute, {})
        for cameras in camera_sets.values():
            for camera in cameras:
                alpha = getattr(camera, "alpha_mask", None)
                if alpha is not None:
                    camera.original_mask = camera.original_mask * alpha


def mask_foreground(target_mask):
    """Return binary support from a binary or RGB identity target."""
    import torch

    if target_mask.ndim == 2:
        target_mask = target_mask.unsqueeze(0)
    return torch.any(target_mask > 1.0e-6, dim=0, keepdim=True).to(
        target_mask.dtype
    )


def balanced_mask_l1(rendered_mask, target_mask):
    """Balance colored identity-mask error over foreground and background."""
    import torch

    if rendered_mask.ndim == 2:
        rendered_mask = rendered_mask.unsqueeze(0)
    if target_mask.ndim == 2:
        target_mask = target_mask.unsqueeze(0)
    error = torch.abs(rendered_mask - target_mask).mean(
        dim=0,
        keepdim=True,
    )
    foreground = mask_foreground(target_mask)
    background = 1.0 - foreground
    foreground_loss = (error * foreground).sum() / foreground.sum().clamp_min(
        1.0
    )
    background_loss = (error * background).sum() / background.sum().clamp_min(
        1.0
    )
    return 0.5 * (foreground_loss + background_loss)


def foreground_l1(rendered_rgb, target_rgb, target_mask):
    """Normalize RGB supervision by foreground pixels instead of image area."""
    weight = mask_foreground(target_mask)
    while weight.ndim < rendered_rgb.ndim:
        weight = weight.unsqueeze(0)
    weight = weight.expand_as(rendered_rgb)
    return (
        (rendered_rgb - target_rgb).abs() * weight
    ).sum() / weight.sum().clamp_min(1.0)


def enforce_max_splats(gaussians, maximum: int) -> int:
    """Prune the lowest-opacity overflow after one densification event."""
    import torch

    limit = max(int(maximum), 1)
    count = int(gaussians.get_xyz.shape[0])
    if count <= limit:
        return 0
    opacity = gaussians.get_opacity.detach().reshape(-1)
    overflow = count - limit
    indices = torch.topk(opacity, k=overflow, largest=False).indices
    prune_mask = torch.zeros(count, dtype=torch.bool, device=opacity.device)
    prune_mask[indices] = True
    missing_tmp_radii = gaussians.tmp_radii is None
    if missing_tmp_radii:
        gaussians.tmp_radii = torch.zeros(
            count, dtype=opacity.dtype, device=opacity.device
        )
    gaussians.prune_points(prune_mask)
    if missing_tmp_radii:
        gaussians.tmp_radii = None
    return overflow


def patch_densification_cap_source(source: str) -> str:
    """Correct the released iteration/count mix-up for custom runs only."""
    expected = (
        (_DENSIFY_CONDITION_LINE, "densification condition"),
        (_DENSIFY_CALL_LINE, "densify-and-prune call"),
    )
    for needle, label in expected:
        if source.count(needle) != 1:
            raise RuntimeError(
                f"Released train.py no longer has one expected {label}; "
                "refusing an ambiguous densification-cap patch."
            )
    source = source.replace(
        _DENSIFY_CONDITION_LINE,
        (
            "            if iteration < opt.densify_until_iter and "
            "gaussians.get_xyz.shape[0] < opt.max_num_splats:\n"
        ),
        1,
    )
    return source.replace(
        _DENSIFY_CALL_LINE,
        _DENSIFY_CALL_LINE
        + "                    _enforce_max_splats("
        + "gaussians, opt.max_num_splats)\n",
        1,
    )


def patch_foreground_balanced_source(
    source: str,
    *,
    mask_loss_weight: float,
) -> str:
    """Patch the released trainer narrowly, failing if its source has changed."""
    expected = (
        (_MASK_LOSS_LINE, "mask-loss line"),
        (_RGB_LOSS_LINE, "RGB-loss line"),
        (_MASK_LOSS_EXPRESSION, "mask-loss expression"),
        (_SCENE_LINE, "Scene construction line"),
    )
    for needle, label in expected:
        if source.count(needle) != 1:
            raise RuntimeError(
                f"Released train.py no longer has one expected {label}; "
                "refusing an ambiguous foreground-balanced loss patch."
            )
    source = source.replace(
        _SCENE_LINE,
        _SCENE_LINE + "    _sanitize_identity_masks(scene)\n",
        1,
    )
    source = source.replace(
        _MASK_LOSS_LINE,
        (
            "        Ll1_mask = _balanced_mask_l1(mask, GT_mask) * "
            "_view_weight(viewpoint_cam.image_name)\n"
        ),
        1,
    )
    source = source.replace(
        _RGB_LOSS_LINE,
        "        Ll1 = _foreground_l1(image, gt_image, GT_mask)\n",
        1,
    )
    return source.replace(
        _MASK_LOSS_EXPRESSION,
        f"Ll1_mask * {float(mask_loss_weight):.17g}",
        1,
    )
