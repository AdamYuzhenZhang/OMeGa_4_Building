"""Minimal torch-scatter compatibility used by released ObjectGS.

ObjectGS only consumes the reduced values returned by ``scatter_max`` during
anchor growth. PyTorch's native ``scatter_reduce_`` supplies that operation on
current CUDA/PyTorch releases, avoiding an obsolete binary dependency.
"""

from __future__ import annotations

import torch


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
    fill_value: float | None = None,
) -> tuple[torch.Tensor, None]:
    if out is not None:
        raise NotImplementedError("ObjectGS compatibility does not use out=")
    normalized_dim = dim if dim >= 0 else src.ndim + dim
    if normalized_dim < 0 or normalized_dim >= src.ndim:
        raise IndexError(f"Invalid scatter dimension {dim} for {src.shape}.")
    expanded_index = index
    if expanded_index.shape != src.shape:
        expanded_index = expanded_index.expand_as(src)
    if dim_size is None:
        dim_size = (
            int(expanded_index.max().item()) + 1
            if expanded_index.numel()
            else 0
        )
    shape = list(src.shape)
    shape[normalized_dim] = int(dim_size)
    if fill_value is None:
        fill_value = (
            float("-inf")
            if src.dtype.is_floating_point
            else torch.iinfo(src.dtype).min
        )
    reduced = torch.full(
        shape,
        fill_value,
        dtype=src.dtype,
        device=src.device,
    )
    reduced.scatter_reduce_(
        normalized_dim,
        expanded_index,
        src,
        reduce="amax",
        include_self=True,
    )
    return reduced, None
