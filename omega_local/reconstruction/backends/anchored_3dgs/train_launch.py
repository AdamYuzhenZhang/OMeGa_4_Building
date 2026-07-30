"""Run released Split&Splat training from a full Gaussian initializer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from omega_local.reconstruction.masked_training import (
    balanced_mask_l1,
    foreground_l1,
    load_normalized_view_weights,
    patch_foreground_balanced_source,
    sanitize_identity_masks,
)


_SCENE_SETUP = (
    "    scene = Scene(dataset, gaussians)\n"
    "    gaussians.training_setup(opt)\n"
)


def _load_view_weights(
    path: Path,
    *,
    mask_dir: Path,
) -> dict[str, float]:
    return load_normalized_view_weights(path, mask_dir=mask_dir)


def _view_weight(image_name: str) -> float:
    stem = Path(str(image_name)).stem
    return float(_ANCHORED_VIEW_WEIGHTS.get(stem, 1.0))


def _load_anchored_gaussians(gaussians, scene, dataset) -> None:
    """Load geometry/appearance while retaining this region's identity color."""
    identity = gaussians._id[:1].detach().clone()
    gaussians.load_ply(
        str(_ANCHORED_INITIALIZER),
        scene.getTrainCameras(),
        dataset.train_test_exp,
    )
    gaussians._id = identity.repeat(gaussians.get_xyz.shape[0], 1).contiguous()


def _patched_source(source: str, *, mask_loss_weight: float = 1.0) -> str:
    if source.count(_SCENE_SETUP) != 1:
        raise RuntimeError(
            "Released train.py no longer has the expected Scene setup; "
            "refusing an ambiguous anchored-initializer patch."
        )
    source = source.replace(
        _SCENE_SETUP,
        (
            "    scene = Scene(dataset, gaussians)\n"
            "    _load_anchored_gaussians(gaussians, scene, dataset)\n"
            "    gaussians.training_setup(opt)\n"
        ),
        1,
    )
    return patch_foreground_balanced_source(
        source,
        mask_loss_weight=mask_loss_weight,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an exact full-Gaussian initializer and normalized manual-"
            "anchor mask weights to an isolated released Split&Splat trainer."
        )
    )
    parser.add_argument("--upstream-train", type=Path, required=True)
    parser.add_argument("--initializer-gaussians", type=Path, required=True)
    parser.add_argument("--training-view-weights", type=Path, required=True)
    parser.add_argument("--training-mask-dir", type=Path, required=True)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    train_path = args.upstream_train.expanduser().resolve()
    initializer = args.initializer_gaussians.expanduser().resolve()
    weights_path = args.training_view_weights.expanduser().resolve()
    mask_dir = args.training_mask_dir.expanduser().resolve()
    for label, path in (
        ("Released train.py", train_path),
        ("Gaussian initializer", initializer),
        ("Training-view weights", weights_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Training mask directory does not exist: {mask_dir}"
        )

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    source = _patched_source(
        train_path.read_text(encoding="utf-8"),
        mask_loss_weight=args.mask_loss_weight,
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(train_path.parent))
    sys.argv = [str(train_path), *train_args]
    namespace: dict[str, Any] = {
        "__file__": str(train_path),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
        "_ANCHORED_INITIALIZER": initializer,
        "_ANCHORED_VIEW_WEIGHTS": _load_view_weights(
            weights_path,
            mask_dir=mask_dir,
        ),
        "_balanced_mask_l1": balanced_mask_l1,
        "_foreground_l1": foreground_l1,
        "_load_anchored_gaussians": _load_anchored_gaussians,
        "_view_weight": _view_weight,
        "_sanitize_identity_masks": sanitize_identity_masks,
    }
    globals().update(
        {
            "_ANCHORED_INITIALIZER": initializer,
            "_ANCHORED_VIEW_WEIGHTS": namespace["_ANCHORED_VIEW_WEIGHTS"],
        }
    )
    exec(compile(source, str(train_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
