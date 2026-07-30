"""Run released Split&Splat composition training with paper parameters."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from omega_local.reconstruction.masked_training import (
    balanced_mask_l1,
    foreground_l1,
    patch_foreground_balanced_source,
    sanitize_identity_masks,
)


_MASK_EXPRESSION = "Ll1_mask * 0.25"
_TRAINING_SETUP = "gaussians.training_setup(opt)"


def _reset_opacity_before_training(gaussians) -> None:
    """Apply the composition opacity cap before Adam captures the parameter."""
    import torch

    capped_opacity = torch.minimum(
        gaussians.get_opacity,
        torch.full_like(gaussians.get_opacity, 0.01),
    )
    reset = gaussians.inverse_opacity_activation(capped_opacity)
    gaussians._opacity = torch.nn.Parameter(reset.detach().requires_grad_(True))


def _patched_source(
    source: str,
    *,
    mask_loss_weight: float,
    paper_reset_opacity: bool,
    foreground_balanced: bool = False,
) -> str:
    if foreground_balanced:
        source = patch_foreground_balanced_source(
            source,
            mask_loss_weight=mask_loss_weight,
        )
    else:
        if source.count(_MASK_EXPRESSION) != 1:
            raise RuntimeError(
                "Released train.py no longer has the expected mask-loss expression; "
                "refusing an ambiguous runtime patch."
            )
        source = source.replace(
            _MASK_EXPRESSION,
            f"Ll1_mask * {float(mask_loss_weight):.17g}",
            1,
        )
    if paper_reset_opacity:
        if source.count(_TRAINING_SETUP) != 1:
            raise RuntimeError(
                "Released train.py no longer has one training_setup call; "
                "cannot apply the paper's composition opacity reset safely."
            )
        source = source.replace(
            _TRAINING_SETUP,
            (
                "if dataset.composition:\n"
                "        _reset_opacity_before_training(gaussians)\n"
                f"    {_TRAINING_SETUP}"
            ),
            1,
        )
    return source


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply explicit Split&Splat composition parameters to an isolated "
            "execution of the released training script."
        )
    )
    parser.add_argument("--upstream-train", type=Path, required=True)
    parser.add_argument("--mask-loss-weight", type=float, required=True)
    parser.add_argument("--paper-reset-opacity", action="store_true")
    parser.add_argument("--foreground-balanced", action="store_true")
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    train_path = args.upstream_train.expanduser().resolve()
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    source = _patched_source(
        train_path.read_text(encoding="utf-8"),
        mask_loss_weight=args.mask_loss_weight,
        paper_reset_opacity=bool(args.paper_reset_opacity),
        foreground_balanced=bool(args.foreground_balanced),
    )

    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(train_path.parent))
    sys.argv = [str(train_path), *train_args]
    namespace = {
        "__file__": str(train_path),
        "__name__": "__main__",
        "__package__": None,
        "__cached__": None,
        "_reset_opacity_before_training": _reset_opacity_before_training,
        "_balanced_mask_l1": balanced_mask_l1,
        "_foreground_l1": foreground_l1,
        "_view_weight": lambda _image_name: 1.0,
        "_sanitize_identity_masks": sanitize_identity_masks,
    }
    exec(compile(source, str(train_path), "exec"), namespace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
