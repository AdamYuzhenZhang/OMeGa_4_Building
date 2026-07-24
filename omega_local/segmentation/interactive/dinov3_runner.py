"""Isolated, resumable DINOv3 evidence extraction for V2-SAM."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .v2sam_cache import DinoFeatureCache, DinoFrame


def _resolve_device(value: str):
    import torch

    key = str(value or "auto").strip().lower()
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(key)


def _load_frames(path: Path) -> list[DinoFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("frames", payload) if isinstance(payload, dict) else payload
    frames = [
        DinoFrame(
            local_index=int(row["localIndex"]),
            frame_id=int(row["frameId"]),
            image_path=Path(str(row["imagePath"])).expanduser().resolve(),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        for row in rows
    ]
    frames.sort(key=lambda frame: frame.local_index)
    if [frame.local_index for frame in frames] != list(range(len(frames))):
        raise ValueError("DINOv3 manifest indices must be contiguous and zero-based.")
    return frames


def prepare_dinov3_cache(
    frames: list[DinoFrame],
    *,
    root: Path,
    repo_root: Path,
    checkpoint: Path,
    device,
    image_size: int = 768,
    patch_size: int = 16,
) -> DinoFeatureCache:
    cache = DinoFeatureCache(
        root,
        model=None,
        device=device,
        checkpoint=checkpoint,
        image_size=image_size,
        patch_size=patch_size,
    )
    if not cache.features_ready(frames):
        import torch

        repo_text = str(repo_root)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)
        model = torch.hub.load(
            str(repo_root / "third_parts" / "dinov3"),
            "dinov3_vitl16",
            source="local",
            weights=str(checkpoint),
        ).eval().to(device)
        cache.model = model
        cache.prepare(frames)
        cache.model = None
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cache.build_pca_visualizations(frames)
    return cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the V2-SAM-compatible DINOv3 evidence cache.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--patch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames = _load_frames(args.manifest.expanduser().resolve())
    device = _resolve_device(args.device)
    prepare_dinov3_cache(
        frames,
        root=args.root.expanduser().resolve(),
        repo_root=args.repo_root.expanduser().resolve(),
        checkpoint=args.checkpoint.expanduser().resolve(),
        device=device,
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
