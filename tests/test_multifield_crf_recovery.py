from pathlib import Path

import numpy as np

from omega_local.segmentation.interactive.colmap_point_field import (
    ColmapPointFieldConfig,
    ProjectedPointField,
    _load_segmented_point_labels,
)
from omega_local.segmentation.interactive.multifield_crf_recovery import (
    MultiFieldCrfConfig,
    densify_multifield_crf,
)


def _config(**overrides) -> MultiFieldCrfConfig:
    values = {
        "source_method_id": "colmap_tracks",
        "source_run_dir": Path("unused"),
        "evidence_root": Path("unused"),
        "point_field": ColmapPointFieldConfig(Path("unused"), Path("unused")),
        "target_superpixels": 500,
        "max_seed_distance_pixels": 80.0,
    }
    values.update(overrides)
    return MultiFieldCrfConfig(**values)


def _two_surface_case() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    ProjectedPointField,
    np.ndarray,
]:
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
    xy = np.asarray(
        [[12, 12], [12, 24], [12, 36], [60, 12], [60, 24], [60, 36]],
        dtype=np.int32,
    )
    positions = np.column_stack(
        (xy[:, 0] / 10.0, xy[:, 1] / 10.0, np.r_[np.zeros(3), np.ones(3)])
    ).astype(np.float32)
    point_normal = np.concatenate(
        (np.tile([0.0, 0.0, 1.0], (3, 1)), np.tile([1.0, 0.0, 0.0], (3, 1))),
        axis=0,
    ).astype(np.float32)
    field = ProjectedPointField(
        point_ids=np.arange(6, dtype=np.int64),
        pixel_xy=xy,
        positions=positions,
        normals=point_normal,
        labels=np.asarray([4, 4, 4, 9, 9, 9], dtype=np.uint16),
    )
    return rgb, depth, normal, valid, field, np.zeros((height, width), dtype=np.uint16)


def test_multifield_crf_recovers_two_surfaces_without_crossing_boundary() -> None:
    rgb, depth, normal, valid, field, sparse = _two_surface_case()
    sparse[24, 12] = 4
    sparse[24, 60] = 9

    dense, summary = densify_multifield_crf(
        rgb=rgb,
        depth=depth,
        depth_valid=valid,
        normal=normal,
        normal_valid=valid,
        sparse_labels=sparse,
        point_field=field,
        config=_config(),
    )

    assert dense[24, 12] == 4
    assert dense[24, 60] == 9
    assert np.mean(dense[:, :32] == 4) > 0.90
    assert np.mean(dense[:, 40:] == 9) > 0.90
    assert np.count_nonzero(dense[:, :32] == 9) == 0
    assert np.count_nonzero(dense[:, 40:] == 4) == 0
    assert summary["visiblePointCount"] == 6
    assert "2d_3d_pairwise" in summary["paperTermsRetained"]


def test_multifield_crf_preserves_unknown_outside_support_radius() -> None:
    rgb, depth, normal, valid, field, sparse = _two_surface_case()
    sparse[24, 12] = 4
    left_only = ProjectedPointField(
        point_ids=field.point_ids[:3],
        pixel_xy=field.pixel_xy[:3],
        positions=field.positions[:3],
        normals=field.normals[:3],
        labels=field.labels[:3],
    )

    dense, _summary = densify_multifield_crf(
        rgb=rgb,
        depth=depth,
        depth_valid=valid,
        normal=normal,
        normal_valid=valid,
        sparse_labels=sparse,
        point_field=left_only,
        config=_config(max_seed_distance_pixels=18.0),
    )

    assert dense[24, 12] == 4
    assert dense[24, 60] == 0
    assert np.count_nonzero(dense == 0) > dense.size // 2


def test_segmented_point_labels_restore_colmap_id_space(tmp_path: Path) -> None:
    path = tmp_path / "segmented.npz"
    np.savez_compressed(
        path,
        point_ids=np.asarray([2, 7], dtype=np.int64),
        labels=np.asarray([4, 9], dtype=np.uint16),
    )

    labels = _load_segmented_point_labels(path, max_point_id=8)

    assert labels.shape == (9,)
    assert labels[2] == 4
    assert labels[7] == 9
    assert np.count_nonzero(labels) == 2
