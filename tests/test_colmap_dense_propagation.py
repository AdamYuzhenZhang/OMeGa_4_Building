import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from omega_local.segmentation.interactive.colmap_dense_propagation import (
    ColmapDenseConfig,
    ColmapDensePropagationSession,
    densify_sparse_colmap_labels,
)
from omega_local.segmentation.interactive.sam2_video_propagation import PropagationFrame
def _config(**overrides) -> ColmapDenseConfig:
    values = {
        "source_method_id": "colmap_tracks",
        "source_run_dir": Path("unused"),
        "evidence_root": Path("unused"),
        "target_superpixels": 500,
        "compactness": 6.0,
        "max_seed_distance_pixels": 80.0,
        "max_boundary_barrier": 1.25,
        "min_competitor_margin": 0.05,
    }
    values.update(overrides)
    return ColmapDenseConfig(**values)


def _two_surface_evidence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = 48, 72
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :36] = np.array([185, 82, 56], dtype=np.uint8)
    rgb[:, 36:] = np.array([48, 92, 190], dtype=np.uint8)
    depth = np.ones((height, width), dtype=np.float32)
    depth[:, 36:] = 2.0
    normal = np.zeros((height, width, 3), dtype=np.float32)
    normal[:, :36, 2] = 1.0
    normal[:, 36:, 0] = 1.0
    valid = np.ones((height, width), dtype=bool)
    return rgb, depth, valid, normal, valid


def test_dense_recovery_fills_surfaces_without_crossing_strong_boundary() -> None:
    rgb, depth, depth_valid, normal, normal_valid = _two_surface_evidence()
    sparse = np.zeros(depth.shape, dtype=np.uint16)
    sparse[24, 12] = 4
    sparse[24, 60] = 9

    dense, summary = densify_sparse_colmap_labels(
        rgb=rgb,
        depth=depth,
        depth_valid=depth_valid,
        normal=normal,
        normal_valid=normal_valid,
        sparse_labels=sparse,
        config=_config(),
    )

    assert dense[24, 12] == 4
    assert dense[24, 60] == 9
    assert np.mean(dense[:, :32] == 4) > 0.70
    assert np.mean(dense[:, 40:] == 9) > 0.70
    assert np.count_nonzero(dense[:, :32] == 9) == 0
    assert np.count_nonzero(dense[:, 40:] == 4) == 0
    assert summary["densePixelCount"] > summary["sparsePixelCount"]


def test_dense_recovery_preserves_unknown_beyond_seed_radius() -> None:
    rgb, depth, depth_valid, normal, normal_valid = _two_surface_evidence()
    sparse = np.zeros(depth.shape, dtype=np.uint16)
    sparse[24, 12] = 4

    dense, _summary = densify_sparse_colmap_labels(
        rgb=rgb,
        depth=depth,
        depth_valid=depth_valid,
        normal=normal,
        normal_valid=normal_valid,
        sparse_labels=sparse,
        config=_config(max_seed_distance_pixels=14.0),
    )

    assert dense[24, 12] == 4
    assert dense[24, 60] == 0
    assert np.count_nonzero(dense == 0) > dense.size // 2


def test_dense_recovery_consumes_saved_sparse_layer() -> None:
    rgb, depth, valid, normal, _normal_valid = _two_surface_evidence()
    sparse = np.zeros(depth.shape, dtype=np.uint16)
    sparse[24, 12] = 4
    sparse[24, 60] = 9

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        source_dir = root / "colmap_tracks"
        (source_dir / "label_maps").mkdir(parents=True)
        (source_dir / "input.json").write_text(json.dumps({"fingerprint": "fixed-input"}), encoding="utf-8")
        (source_dir / "summary.json").write_text("{}", encoding="utf-8")
        np.save(source_dir / "label_maps" / "000000.npy", sparse)

        evidence = root / "view_evidence"
        (evidence / "depth_npz").mkdir(parents=True)
        (evidence / "normal_npz").mkdir(parents=True)
        np.savez_compressed(evidence / "depth_npz" / "000000.npz", depth_m=depth, valid=valid)
        np.savez_compressed(evidence / "normal_npz" / "000000.npz", normal=normal, valid_mask=valid)

        image_path = root / "000000.jpg"
        Image.fromarray(rgb, mode="RGB").save(image_path)
        frame = PropagationFrame(
            frame_id=0,
            image_name=image_path.name,
            image_path=image_path,
            width=rgb.shape[1],
            height=rgb.shape[0],
        )
        config = _config(source_run_dir=source_dir, evidence_root=evidence)
        saved: list[np.ndarray] = []
        rows = ColmapDensePropagationSession(config).recover_from_source(
            run_input=SimpleNamespace(frames=(frame,), sources=(), fingerprint="fixed-input"),
            save_callback=lambda _frame, labels: saved.append(labels.copy()),
        )

    assert len(saved) == 1
    assert np.count_nonzero(saved[0]) > np.count_nonzero(sparse)
    assert rows[0]["sourceMethodId"] == "colmap_tracks"
    assert rows[0]["sparsePixelCount"] == 2
