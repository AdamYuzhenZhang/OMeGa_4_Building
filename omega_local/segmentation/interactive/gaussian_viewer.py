"""Configuration for the embedded SuperSplat Gaussian viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def viewer_status(public_dir: Path) -> dict[str, Any]:
    index = public_dir / "index.html"
    return {
        "available": index.is_file(),
        "viewerId": "supersplat",
        "displayName": "SuperSplat Viewer",
        "message": (
            "Ready"
            if index.is_file()
            else "Build the local SuperSplat viewer with scripts/setup_gaussian_viewer.sh"
        ),
        "supports": ["3dgs_ply"],
    }


def viewer_settings() -> dict[str, Any]:
    return {
        "version": 2,
        # 3DGS SH colors are already optimized against the input image values.
        # SuperSplat's default unmodified path matches the training renderer more
        # closely; display tone mapping makes the reconstruction look too dark.
        "tonemapping": "none",
        # This Split&Splat model contains low-opacity splats with SH colors
        # above the display range. Accumulate them in floating point and clamp
        # only when composing the final image, as the training renderer does.
        "highPrecisionRendering": True,
        "background": {"color": [0.035, 0.043, 0.05]},
        "postEffectSettings": {
            "sharpness": {"enabled": False, "amount": 0},
            "bloom": {"enabled": False, "intensity": 1, "blurLevel": 2},
            "grading": {
                "enabled": False,
                "brightness": 0,
                "contrast": 1,
                "saturation": 1,
                "tint": [1, 1, 1],
            },
            "vignette": {
                "enabled": False,
                "intensity": 0.5,
                "inner": 0.3,
                "outer": 0.75,
                "curvature": 1,
            },
            "fringing": {"enabled": False, "intensity": 0.5},
        },
        "animTracks": [],
        "cameras": [],
        "annotations": [],
        "startMode": "default",
    }
