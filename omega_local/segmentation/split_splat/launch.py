"""Small compatibility launcher for released scripts with GUI backend assumptions."""

from __future__ import annotations

import os
import runpy
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


def _paper_mask_refinement_source(source: str) -> str:
    """Apply the two paper-defined controls omitted by the released script."""
    camera_limit = "if(iter>len(cameras)/2): break"
    missing_mask_gate = "if(iou_score_new<0.05): continue"
    if source.count(camera_limit) != 1:
        raise RuntimeError(
            "Released mask refiner no longer has the expected half-camera limit."
        )
    if source.count(missing_mask_gate) != 1:
        raise RuntimeError(
            "Released mask refiner no longer has the expected missing-mask gate."
        )
    return source.replace(
        camera_limit,
        "if False: break  # Adapter: paper processes every view.",
    ).replace(
        missing_mask_gate,
        "if(mask is None and iou_score_new<0.95): continue",
    )


def _rank_safe_convex_hull(
    convex_hull: Callable[..., Any],
    points: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Treat every point as boundary when a 2D set has rank below two."""
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    if (
        values.ndim == 2
        and values.shape[0] > 0
        and values.shape[1] == 2
        and np.linalg.matrix_rank(values - np.mean(values, axis=0)) < 2
    ):
        return SimpleNamespace(
            vertices=np.arange(values.shape[0], dtype=np.int64)
        )
    return convex_hull(points, *args, **kwargs)


def _guard_rank_deficient_convex_hull() -> None:
    """Keep upstream's hull exclusion defined for collinear image prompts."""
    import scipy.spatial

    original = scipy.spatial.ConvexHull

    def guarded(points: Any, *args: Any, **kwargs: Any) -> Any:
        return _rank_safe_convex_hull(
            original,
            points,
            *args,
            **kwargs,
        )

    scipy.spatial.ConvexHull = guarded


def _guard_empty_point_projection() -> None:
    """Keep the released CUDA extension from launching a zero-block grid."""
    import torch  # noqa: F401 - loads libc10 before the native extension
    import point_projection_cuda

    def guarded(function: Callable[..., Any]) -> Callable[..., Any]:
        def call(points: Any, *args: Any, **kwargs: Any) -> Any:
            if int(points.shape[0]) == 0:
                return None
            return function(points, *args, **kwargs)

        return call

    point_projection_cuda.pcd2D = guarded(point_projection_cuda.pcd2D)
    point_projection_cuda.pcd2D_mask = guarded(
        point_projection_cuda.pcd2D_mask
    )


def _normalize_camera_image_names() -> None:
    """Adapt the ScanNet-only refiner's suffix assumptions to COLMAP cameras."""
    import scene.cameras

    original = scene.cameras.Camera.__init__

    def normalized(self: Any, *args: Any, **kwargs: Any) -> None:
        if "image_name" in kwargs:
            kwargs["image_name"] = Path(str(kwargs["image_name"])).stem
        elif len(args) >= 11:
            values = list(args)
            values[10] = Path(str(values[10])).stem
            args = tuple(values)
        original(self, *args, **kwargs)

    scene.cameras.Camera.__init__ = normalized


def main() -> int:
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--normalize-camera-image-names", action="store_true")
    parser.add_argument("--paper-mask-refinement", action="store_true")
    parser.add_argument("script")
    options, script_args = parser.parse_known_args(sys.argv[1:])
    script = Path(options.script).expanduser().resolve()
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib

    matplotlib.use("Agg", force=True)
    original_use = matplotlib.use

    def headless_use(_backend, *args, **kwargs):
        return original_use("Agg", force=True)

    matplotlib.use = headless_use
    _guard_rank_deficient_convex_hull()
    _guard_empty_point_projection()
    if options.normalize_camera_image_names:
        _normalize_camera_image_names()
    sys.argv = [str(script), *script_args]
    if options.paper_mask_refinement:
        source = _paper_mask_refinement_source(
            script.read_text(encoding="utf-8")
        )
        namespace = {
            "__file__": str(script),
            "__name__": "__main__",
            "__package__": None,
            "__cached__": None,
        }
        exec(compile(source, str(script), "exec"), namespace)
    else:
        runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
