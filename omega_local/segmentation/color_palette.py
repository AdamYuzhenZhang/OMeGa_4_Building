"""Shared deterministic colors for segmentation labels."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import numpy as np


def categorical_rgb(
    labels: np.ndarray,
    *,
    background: Sequence[int] = (0, 0, 0),
) -> np.ndarray:
    """Map integer labels to bright, deterministic RGB colors."""
    values = np.asarray(labels, dtype=np.int64)
    colors = np.empty((*values.shape, 3), dtype=np.uint8)
    colors[...] = np.asarray(background, dtype=np.uint8)
    for label in np.unique(values[values > 0]):
        label_int = int(label)
        hue = (0.071 + label_int * 0.618033988749895) % 1.0
        saturation = 0.68 + 0.10 * ((label_int * 37) % 3) / 2.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, 1.0)
        colors[values == label_int] = np.rint(
            np.asarray(rgb) * 255.0
        ).astype(np.uint8)
    return colors
