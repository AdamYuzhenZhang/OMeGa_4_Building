"""Human-readable documentation for exported segmentation mask packages."""

from __future__ import annotations

from typing import Any


def render_share_readme(
    manifest: dict[str, Any],
    regions: dict[str, Any],
    keyframes: dict[str, Any],
) -> str:
    region_lines = "\n".join(
        f"| {row['id']} | {row['name']} | `{row['colorHex']}` | "
        f"{', '.join(str(value) for value in row['manualFrameIds']) or '-'} |"
        for row in regions["regions"]
    )
    resolution = manifest["maskEncoding"]["resolution"]
    anchors = ", ".join(str(value) for value in keyframes["manualAnchorFrameIds"])
    return f"""# {manifest['dataset']} Segmentation Masks

This package pairs {manifest['frameCount']} indexed masks with stable,
dataset-level region IDs. The {manifest['manualAnchorCount']} user-confirmed
anchor masks conditioned **{manifest['method']['name']}** propagation across
the capture.

## Contents

- `masks/NNNNNN.png`: single-channel indexed mask for frame `NNNNNN`.
- `frames.jsonl`: image/mask correspondence, coverage, active region IDs, and
  per-region pixel areas for every frame.
- `regions.json`: stable dataset-level region IDs, names, and display colors.
- `keyframes.json`: manual anchors used by propagation and separately listed
  system-suggested keyframes.
- `manifest.json`: package provenance, encoding, and summary statistics.
- `checksums.sha256`: integrity hashes for every other package file.

The folder is self-contained for decoding the segmentation annotations. RGB
pixels are not duplicated; each `frames.jsonl` row records the corresponding
capture-relative `sourceImage` path and original resolution.

## Mask Semantics

- Masks are `{resolution['width']} x {resolution['height']}` 16-bit PNGs.
- A pixel value is the persistent region ID listed in `regions.json`.
- `0` means unassigned/background; it is not a persistent region.
- Region IDs have the same meaning in every frame.
- Regions do not overlap because each pixel stores exactly one ID.
- Files use the source image orientation. The editor's optional 90-degree
  portrait display rotation is not baked into this package.
- Manual-anchor masks are preserved exactly in the corresponding files under
  `masks/`. Other masks are propagated candidates and may require downstream
  fusion or quality review.

Manual anchor frame IDs: `{anchors}`.

Each `frames.jsonl` row records `frameId`, `imageName`, `sourceImage`,
`maskPath`, both resolutions, anchor/keyframe flags, coverage, active
`regionIds`, and per-region pixel counts in `regionAreas`.

## Reading The Data

```python
import json
from pathlib import Path

import numpy as np
from PIL import Image

root = Path("{manifest['method']['id']}")
frames = [
    json.loads(line)
    for line in (root / "frames.jsonl").read_text().splitlines()
    if line.strip()
]
regions = json.loads((root / "regions.json").read_text())

row = frames[0]
mask = np.asarray(Image.open(root / row["maskPath"]), dtype=np.uint16)
door_id = next(r["id"] for r in regions["regions"] if r["name"] == "Door")
door = mask == door_id
```

To align a mask to the original RGB resolution, resize with nearest-neighbor
sampling so integer IDs are never interpolated:

```python
mask_full = np.asarray(
    Image.fromarray(mask).resize(
        (row["sourceResolution"]["width"], row["sourceResolution"]["height"]),
        resample=Image.Resampling.NEAREST,
    ),
    dtype=np.uint16,
)
```

## Persistent Regions

| ID | Name | Display color | Manually labeled frames |
|---:|---|---|---|
{region_lines}

## Quality Note

Stable IDs encode the intended cross-frame correspondence; they do not
guarantee that every propagated pixel is correct. Coverage is not an accuracy
score. Use `isManualAnchor` when downstream methods should give user-confirmed
frames higher weight.
"""


__all__ = ["render_share_readme"]
