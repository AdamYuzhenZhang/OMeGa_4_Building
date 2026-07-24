"""Isolated entry point for third-party memory VOS propagation backends."""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image


def _prepend_import_root(root: Path) -> None:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _parse_indices(value: str) -> list[int]:
    return sorted({int(item) for item in str(value).split(",") if item.strip()})


def _resolve_torch_device(requested: str):
    import torch

    key = str(requested or "auto").strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(key)


def _run_xmem(args: argparse.Namespace) -> None:
    _prepend_import_root(args.root)
    from inference.run_on_video import run_on_video

    run_on_video(
        imgs_in_path=str(args.frames_dir),
        masks_in_path=str(args.masks_dir),
        masks_out_path=str(args.output_dir),
        frames_with_masks=_parse_indices(args.source_indices),
        compute_iou=False,
        print_progress=False,
        original_memory_mechanism=False,
        save_overlay=False,
        overwrite_config={
            "model": str(args.checkpoint),
            "size": int(args.internal_size),
            "mem_every": 10,
            "save_masks": True,
        },
    )


def _load_index_mask(path: Path, object_count: int, device):
    import torch

    labels = np.asarray(Image.open(path), dtype=np.uint8)
    channels = [torch.from_numpy(labels == object_id) for object_id in range(1, object_count + 1)]
    return torch.stack(channels, dim=0).to(device=device, dtype=torch.float32)


def _load_image_tensor(path: Path, device):
    from torchvision.transforms.functional import to_tensor

    with Image.open(path) as image:
        return to_tensor(image.convert("RGB")).to(device=device)


def _run_cutie(args: argparse.Namespace) -> None:
    _prepend_import_root(args.root)

    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict

    from cutie.inference.inference_core import InferenceCore
    from cutie.model.cutie import CUTIE

    device = _resolve_torch_device(args.device)
    config_dir = args.root / "cutie" / "config"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="video_config")
    with open_dict(cfg):
        cfg.device = str(device)
        cfg.weights = str(args.checkpoint)
        cfg.max_internal_size = int(args.internal_size)
        cfg.num_objects = int(args.object_count)

    torch.set_grad_enabled(False)
    network = CUTIE(cfg).to(device).eval()
    weights = torch.load(args.checkpoint, map_location=device)
    network.load_weights(weights)
    processor = InferenceCore(network, cfg=cfg)
    objects = list(range(1, int(args.object_count) + 1))
    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    source_indices = set(_parse_indices(args.source_indices))
    if not frame_paths:
        raise ValueError(f"No staged frames found in {args.frames_dir}")

    use_amp = bool(cfg.amp and device.type == "cuda")
    amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
    output_dir = args.output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode(), amp_context:
        # Cutie's official process_video path commits every supplied mask to
        # permanent memory before traversing the sequence.
        for local_index in sorted(source_indices):
            image = _load_image_tensor(frame_paths[local_index], device)
            mask = _load_index_mask(args.masks_dir / f"{local_index:05d}.png", args.object_count, device)
            processor.step(
                image,
                mask,
                objects=objects,
                idx_mask=False,
                force_permanent=True,
            )

        for local_index, frame_path in enumerate(frame_paths):
            image = _load_image_tensor(frame_path, device)
            if local_index in source_indices:
                mask = _load_index_mask(args.masks_dir / f"{local_index:05d}.png", args.object_count, device)
                output_prob = processor.step(
                    image,
                    mask,
                    objects=objects,
                    idx_mask=False,
                    end=local_index == len(frame_paths) - 1,
                )
            else:
                output_prob = processor.step(image, end=local_index == len(frame_paths) - 1)
            labels = processor.output_prob_to_mask(output_prob).detach().cpu().numpy().astype(np.uint8)
            Image.fromarray(labels, mode="L").save(output_dir / f"{local_index:05d}.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an isolated memory-VOS propagation backend.")
    parser.add_argument("--backend", choices=("xmem", "cutie"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--masks-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-indices", required=True)
    parser.add_argument("--object-count", type=int, required=True)
    parser.add_argument("--internal-size", type=int, default=480)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.root = args.root.expanduser().resolve()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.frames_dir = args.frames_dir.expanduser().resolve()
    args.masks_dir = args.masks_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.backend == "xmem":
        _run_xmem(args)
    else:
        _run_cutie(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
