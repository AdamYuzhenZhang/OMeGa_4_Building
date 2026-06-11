"""Region-weighted rendered-depth regularization for OMeGa-4-Building.

This loss is intentionally separate from the upstream-style trainer logic.  It
implements the first extension idea:

    L_depth = lambda_d * mean_p w(p) * rho(D_render(p) - D_prior(p))

where rho is a robust Huber penalty and w(p) is built from Guide01's soft
geometry categories.  Planar pixels can pull the mesh strongly toward a depth
prior, ridge/detail pixels can be weakened, and unknown pixels are ignored.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def _as_bhw(tensor: Tensor) -> Tensor:
    """Return tensor as [B, H, W], accepting common OMeGa raster shapes."""

    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        return tensor[..., 0]
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        return tensor[:, 0]
    if tensor.ndim == 3:
        return tensor
    if tensor.ndim == 2:
        return tensor[None]
    raise ValueError(f"Expected depth-like tensor with 2-4 dims, got {tuple(tensor.shape)}")


def _huber(error: Tensor, delta: float) -> Tensor:
    """Huber penalty in meters.

    rho(e) = 0.5 * e^2 / delta,  if |e| <= delta
           = |e| - 0.5 * delta,  otherwise

    This behaves like L2 near zero but avoids letting inconsistent monocular
    depth outliers dominate the mesh update.
    """

    abs_error = torch.abs(error)
    delta_t = torch.as_tensor(max(float(delta), 1e-6), device=error.device, dtype=error.dtype)
    quadratic = 0.5 * error.square() / delta_t
    linear = abs_error - 0.5 * delta_t
    return torch.where(abs_error <= delta_t, quadratic, linear)


def compute_depth_regularization_loss(
    *,
    render_depth: Tensor,
    render_alpha: Tensor | None,
    batch: dict[str, Any],
    depth_lambda: float,
    huber_delta_m: float,
    min_depth_m: float,
    max_depth_m: float,
    min_alpha: float,
    planar_weight: float,
    detail_weight: float,
    ridge_weight: float,
    distance_decay_m: float,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute weighted depth-prior loss.

    Inputs are one training batch.  The guide maps are produced by Guide01 and
    loaded by ``examples/datasets/colmap.py``.  The rendered depth is compared
    in the image plane, so gradients flow from the rasterized splat depth back
    to OMeGa's mesh vertices through ``update_gs``.
    """

    device = render_depth.device
    dtype = render_depth.dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    if "building_depth_prior" not in batch:
        return zero, zero, {"used_depth_pixels": 0.0, "mean_depth_weight": 0.0, "mean_abs_depth_error_m": 0.0}

    rendered = _as_bhw(render_depth).to(dtype=dtype)
    prior = _as_bhw(batch["building_depth_prior"].to(device)).to(dtype=dtype)
    valid = _as_bhw(batch["building_depth_valid"].to(device)).bool()
    p_planar = _as_bhw(batch["building_p_local_planar"].to(device)).to(dtype=dtype).clamp(0.0, 1.0)
    p_detail = _as_bhw(batch["building_p_detail"].to(device)).to(dtype=dtype).clamp(0.0, 1.0)
    p_ridge = _as_bhw(batch["building_p_ridge"].to(device)).to(dtype=dtype).clamp(0.0, 1.0)

    finite = torch.isfinite(rendered) & torch.isfinite(prior)
    depth_ok = prior >= float(min_depth_m)
    if float(max_depth_m) > 0.0:
        depth_ok = depth_ok & (prior <= float(max_depth_m))
    if render_alpha is not None and float(min_alpha) > 0.0:
        depth_ok = depth_ok & (_as_bhw(render_alpha).to(device) >= float(min_alpha))

    category_weight = (
        float(planar_weight) * p_planar
        + float(detail_weight) * p_detail
        + float(ridge_weight) * p_ridge
    )
    if float(distance_decay_m) > 0.0:
        # Optional confidence falloff: nearby depth priors are usually more
        # reliable in phone captures, while far facade/sky estimates are weaker.
        category_weight = category_weight * torch.exp(-prior.clamp_min(0.0) / float(distance_decay_m))

    weight = torch.where(valid & finite & depth_ok, category_weight, torch.zeros_like(category_weight))
    weight_sum = weight.sum()
    if weight_sum <= 1e-8:
        return zero, zero, {"used_depth_pixels": 0.0, "mean_depth_weight": 0.0, "mean_abs_depth_error_m": 0.0}

    error = rendered - prior
    unscaled_loss = (weight * _huber(error, float(huber_delta_m))).sum() / weight_sum.clamp_min(1e-8)
    scaled_loss = float(depth_lambda) * unscaled_loss
    mean_abs_error = (weight * torch.abs(error)).sum() / weight_sum.clamp_min(1e-8)
    used = torch.count_nonzero(weight > 0).to(dtype=dtype)
    stats = {
        "used_depth_pixels": float(used.detach().cpu()),
        "mean_depth_weight": float(weight[weight > 0].mean().detach().cpu()),
        "mean_abs_depth_error_m": float(mean_abs_error.detach().cpu()),
    }
    return scaled_loss, unscaled_loss, stats

