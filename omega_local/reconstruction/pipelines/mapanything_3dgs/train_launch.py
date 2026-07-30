"""Run the released masked 3DGS trainer with manual-frame mask weights."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from omega_local.reconstruction.masked_training import (
    balanced_mask_l1,
    enforce_max_splats,
    foreground_l1,
    load_normalized_view_weights,
    patch_densification_cap_source,
    patch_foreground_balanced_source,
    sanitize_identity_masks,
)


def _load_view_weights(path: Path, mask_dir: Path) -> dict[str, float]:
    return load_normalized_view_weights(path, mask_dir=mask_dir)


def _view_weight(image_name: str) -> float:
    return float(_MAPANYTHING_VIEW_WEIGHTS.get(Path(image_name).stem, 1.0))


def _patched_source(source: str, *, mask_loss_weight: float = 1.0) -> str:
    source = patch_foreground_balanced_source(
        source,
        mask_loss_weight=mask_loss_weight,
    )
    return patch_densification_cap_source(source)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-train", type=Path, required=True)
    parser.add_argument("--training-view-weights", type=Path, required=True)
    parser.add_argument("--training-mask-dir", type=Path, required=True)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    train_path = args.upstream_train.expanduser().resolve()
    weights_path = args.training_view_weights.expanduser().resolve()
    mask_dir = args.training_mask_dir.expanduser().resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Released trainer does not exist: {train_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Training-view weights do not exist: {weights_path}"
        )
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Training masks do not exist: {mask_dir}")

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    weights = _load_view_weights(weights_path, mask_dir)
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
        "_MAPANYTHING_VIEW_WEIGHTS": weights,
        "_balanced_mask_l1": balanced_mask_l1,
        "_enforce_max_splats": enforce_max_splats,
        "_foreground_l1": foreground_l1,
        "_view_weight": _view_weight,
        "_sanitize_identity_masks": sanitize_identity_masks,
    }
    globals()["_MAPANYTHING_VIEW_WEIGHTS"] = weights
    exec(compile(source, str(train_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
