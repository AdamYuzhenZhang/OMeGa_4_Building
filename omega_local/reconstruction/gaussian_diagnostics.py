"""Numerical diagnostics for standard 3D Gaussian Splatting PLY files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from plyfile import PlyData


_SH_C0 = 0.28209479177387814


def summarize_gaussian_ply(path: Path) -> dict[str, Any]:
    """Summarize appearance and geometry fields without rendering the model."""
    vertices = PlyData.read(path, mmap="r")["vertex"].data
    names = set(vertices.dtype.names or ())
    summary: dict[str, Any] = {
        "path": str(path),
        "gaussianCount": int(vertices.shape[0]),
        "propertyCount": len(names),
    }
    if not {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        if {"red", "green", "blue"}.issubset(names):
            rgb = (
                np.column_stack(
                    [vertices[channel] for channel in ("red", "green", "blue")]
                ).astype(np.float64)
                / 255.0
            )
            summary["appearance"] = {
                "rgbMean": rgb.mean(axis=0).tolist(),
                "nearWhiteRgbFraction": float(
                    np.all(rgb > 0.98, axis=1).mean()
                ),
                "brightRgbFraction": float(
                    np.all(rgb > 0.9, axis=1).mean()
                ),
            }
        return summary

    dc = np.column_stack(
        [vertices[f"f_dc_{channel}"] for channel in range(3)]
    ).astype(np.float64)
    dc_rgb = np.clip(0.5 + _SH_C0 * dc, 0.0, 1.0)
    white = np.all(dc_rgb > 0.98, axis=1)
    summary["appearance"] = {
        "dcRgbMean": dc_rgb.mean(axis=0).tolist(),
        "nearWhiteDcFraction": float(white.mean()),
        "brightDcFraction": float(np.all(dc_rgb > 0.9, axis=1).mean()),
        "nonfiniteDcCount": int(np.any(~np.isfinite(dc), axis=1).sum()),
    }

    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name.rsplit("_", 1)[1]),
    )
    if rest_names:
        rest = np.column_stack([vertices[name] for name in rest_names]).astype(
            np.float64
        )
        amplitude = np.max(np.abs(rest), axis=1)
        summary["appearance"]["shRestAbsolutePercentiles"] = (
            np.percentile(amplitude, [50, 95, 99, 100]).tolist()
        )

    if {"scale_0", "scale_1", "scale_2"}.issubset(names):
        scale = np.exp(
            np.column_stack(
                [vertices[f"scale_{axis}"] for axis in range(3)]
            ).astype(np.float64)
        )
        max_scale = np.max(scale, axis=1)
        summary["geometry"] = {
            "maxScalePercentiles": np.percentile(
                max_scale, [50, 95, 99, 100]
            ).tolist(),
            "nearWhiteMaxScaleP95": (
                float(np.percentile(max_scale[white], 95))
                if np.any(white)
                else None
            ),
        }

    if "opacity" in names:
        logits = np.asarray(vertices["opacity"], dtype=np.float64)
        opacity = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        summary["opacity"] = {
            "percentiles": np.percentile(opacity, [5, 50, 95]).tolist(),
            "nearWhiteMedian": (
                float(np.median(opacity[white])) if np.any(white) else None
            ),
            "nearWhiteVisibleFraction": float(
                np.logical_and(white, opacity > 0.05).mean()
            ),
            "nearWhiteOpaqueFraction": float(
                np.logical_and(white, opacity > 0.5).mean()
            ),
        }
    return summary
