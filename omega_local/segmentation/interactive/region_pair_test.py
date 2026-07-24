"""Shared input selection for focused source-target region transfer tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .sam2_video_propagation import LabeledPropagationSource, PropagationFrame


@dataclass(frozen=True)
class RegionPairTestInput:
    method_id: str
    output_policy: str
    target_frame: PropagationFrame
    target_local_index: int
    source: LabeledPropagationSource
    region_id: int
    region_row: dict[str, Any]
    fingerprint: str

    @property
    def target_frame_id(self) -> int:
        return int(self.target_frame.frame_id)

    @property
    def source_frame_id(self) -> int:
        return int(self.source.frame_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "mode": "single_region_pair",
            "methodId": self.method_id,
            "sourcePolicy": "first_manual_occurrence",
            "outputPolicy": self.output_policy,
            "targetFrameId": self.target_frame_id,
            "sourceFrameId": self.source_frame_id,
            "regionId": int(self.region_id),
            "sourceAreaPixels": int(np.count_nonzero(self.source.labels == self.region_id)),
            "fingerprint": self.fingerprint,
        }


def build_region_pair_test_input(
    *,
    method_id: str,
    output_policy: str,
    target_frame_id: int,
    region_id: int,
    frames: list[PropagationFrame],
    region_status: dict[str, Any],
    region_maps_dir: Path,
) -> RegionPairTestInput:
    if region_id <= 0:
        raise ValueError("Region-pair testing needs a selected persistent region.")
    frame_index_by_id = {int(frame.frame_id): index for index, frame in enumerate(frames)}
    target_local_index = frame_index_by_id.get(int(target_frame_id))
    if target_local_index is None:
        raise KeyError(target_frame_id)

    region_row = next(
        (
            dict(row)
            for row in region_status.get("regions", [])
            if isinstance(row, dict) and int(row.get("id", 0)) == int(region_id)
        ),
        None,
    )
    if region_row is None:
        raise KeyError(f"Persistent region {region_id} does not exist.")

    manual_frame_ids = {
        int(value)
        for value in region_row.get("frameIds", [])
        if int(value) in frame_index_by_id
    }
    if not manual_frame_ids:
        raise ValueError("The selected region has no manual source mask available.")
    source_frame_id = min(manual_frame_ids, key=frame_index_by_id.__getitem__)
    if source_frame_id == int(target_frame_id):
        raise ValueError(
            "The open target is this region's first manual source frame. Open another frame "
            "to test transfer from the first occurrence."
        )

    source_local_index = frame_index_by_id[source_frame_id]
    source_frame = frames[source_local_index]
    map_path = region_maps_dir / f"{source_frame_id:06d}.npy"
    if not map_path.exists():
        raise FileNotFoundError(f"First-occurrence region map is missing: {map_path}")
    labels = np.load(map_path)
    expected_shape = (int(source_frame.height), int(source_frame.width))
    if labels.ndim != 2 or labels.shape != expected_shape:
        raise ValueError(
            f"Persistent region map for frame {source_frame_id} has shape {labels.shape}; "
            f"expected {expected_shape}."
        )
    isolated = np.zeros(expected_shape, dtype=np.uint16)
    isolated[np.asarray(labels) == int(region_id)] = np.uint16(region_id)
    if not np.any(isolated):
        raise ValueError(f"Region {region_id} has no pixels in its first manual frame {source_frame_id}.")

    source = LabeledPropagationSource(
        local_index=int(source_local_index),
        frame_id=int(source_frame_id),
        labels=isolated,
    )
    target_frame = frames[target_local_index]
    digest = hashlib.sha256()
    digest.update(b"omega-focused-region-pair-v1\0")
    digest.update(f"method:{method_id}:policy:{output_policy}\0".encode("utf-8"))
    digest.update(
        f"source:{source.frame_id}:target:{target_frame_id}:region:{region_id}\0".encode("ascii")
    )
    digest.update(source.labels.tobytes(order="C"))
    for frame in (source_frame, target_frame):
        image_path = frame.image_path.expanduser().resolve()
        stat = image_path.stat()
        digest.update(f"{image_path}:{stat.st_size}:{stat.st_mtime_ns}\0".encode("utf-8"))
    return RegionPairTestInput(
        method_id=str(method_id),
        output_policy=str(output_policy),
        target_frame=target_frame,
        target_local_index=int(target_local_index),
        source=source,
        region_id=int(region_id),
        region_row=region_row,
        fingerprint=digest.hexdigest(),
    )
