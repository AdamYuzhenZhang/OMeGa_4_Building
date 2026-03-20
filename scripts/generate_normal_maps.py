"""
Generate monocular normal maps using Stable Normal.
https://github.com/hugoycj/StableNormal

Output format: RGB PNG images in <data_dir>/normal_maps/
Normal values are encoded as (normal + 1) / 2 * 255, i.e., [-1, 1] -> [0, 255].

Usage:
    python scripts/generate_normal_maps.py --data_dir /path/to/colmap_dataset

    # Process at lower resolution for speed, then resize back
    python scripts/generate_normal_maps.py --data_dir /path/to/colmap_dataset --max_size 1024
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Generate normal maps using Stable Normal")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COLMAP dataset")
    parser.add_argument("--max_size", type=int, default=None,
                        help="Max image dimension for inference (resize back to original after)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    output_dir = data_dir / "normal_maps"
    output_dir.mkdir(exist_ok=True)

    # Collect image paths
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix in extensions
    ])
    print(f"Found {len(image_paths)} images in {images_dir}")

    # Load Stable Normal via torch.hub (official API)
    predictor = torch.hub.load(
        "Stable-X/StableNormal", "StableNormal", trust_repo=True,
    )

    for img_path in tqdm(image_paths, desc="Generating normal maps"):
        # Output path: same name but .png extension
        out_name = img_path.stem + ".png"
        out_path = output_dir / out_name

        if out_path.exists():
            continue

        # Load image
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)

        # Optionally resize for faster inference
        if args.max_size is not None:
            w, h = original_size
            scale = min(args.max_size / max(w, h), 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                image = image.resize((new_w, new_h), Image.BILINEAR)

        # Run inference — returns a PIL Image with normals encoded as RGB
        with torch.no_grad():
            normal_pil = predictor(image)  # PIL Image, RGB, [0, 255]

        # Resize back to original resolution if needed
        if args.max_size is not None and scale < 1.0:
            normal_pil = normal_pil.resize(original_size, Image.BILINEAR)
            normal_pil.save(out_path)
        else:
            normal_pil.save(out_path)

    print(f"Normal maps saved to {output_dir}")


if __name__ == "__main__":
    main()
