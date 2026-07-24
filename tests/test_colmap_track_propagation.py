from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d

from omega_local.segmentation.interactive.colmap_track_propagation import (
    _PinholePixelTransform,
    ColmapTrackConfig,
    _label_points_from_anchors,
    _map_observation_pixels,
    _rasterize_exclusive_points,
    _save_segmented_colmap_points,
)
from omega_local.segmentation.interactive.sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
)


class _Camera:
    width = 8
    height = 6


class _Point:
    def __init__(self, x: float, y: float, point_id: int) -> None:
        self.x = x
        self.y = y
        self.point3D_id = point_id

    def has_point3D(self) -> bool:
        return True


class _Image:
    def __init__(self, points: list[_Point]) -> None:
        self.camera = _Camera()
        self.points2D = points


def _frame(frame_id: int) -> PropagationFrame:
    return PropagationFrame(
        frame_id=frame_id,
        image_name=f"{frame_id:06d}.jpg",
        image_path=Path(f"{frame_id:06d}.jpg"),
        width=8,
        height=6,
    )


def test_anchor_votes_keep_winner_and_reject_tie() -> None:
    frames = [_frame(0), _frame(1)]
    images = {
        0: _Image([_Point(2, 2, 1), _Point(4, 2, 2)]),
        1: _Image([_Point(2, 2, 1), _Point(4, 2, 2)]),
    }
    labels_a = np.zeros((6, 8), dtype=np.uint16)
    labels_b = np.zeros((6, 8), dtype=np.uint16)
    labels_a[2, 2] = 7
    labels_b[2, 2] = 7
    labels_a[2, 4] = 7
    labels_b[2, 4] = 9
    sources = [
        LabeledPropagationSource(local_index=0, frame_id=0, labels=labels_a),
        LabeledPropagationSource(local_index=1, frame_id=1, labels=labels_b),
    ]

    point_labels, summary = _label_points_from_anchors(
        sources,
        frames,
        images,
        np.array([False, True, True], dtype=bool),
    )

    assert point_labels.tolist() == [0, 7, 0]
    assert summary == {
        "positiveAnchorObservations": 4,
        "votedPointCount": 2,
        "labeledPointCount": 1,
        "tiedPointCount": 1,
    }


def test_sparse_raster_keeps_only_unambiguous_point_pixels() -> None:
    pixels = np.array([[2, 2], [2, 2], [5, 4]], dtype=np.int32)
    labels = np.array([3, 4, 8], dtype=np.uint16)

    raster = _rasterize_exclusive_points(pixels, labels, width=8, height=6, radius=0)

    assert raster[2, 2] == 0
    assert raster[4, 5] == 8
    assert np.count_nonzero(raster) == 1


def test_pinhole_export_rotation_maps_colmap_pixels_to_editor_grid() -> None:
    transform = _PinholePixelTransform(
        source_width=8,
        source_height=6,
        output_width=8,
        output_height=6,
        input_camera_model="SIMPLE_PINHOLE",
        input_camera_params=(4.0, 3.5, 2.5),
        input_distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        output_camera_params=(4.0, 4.0, 3.5, 2.5),
        rotation="180",
        pre_rotation_width=8,
        pre_rotation_height=6,
    )

    mapped = _map_observation_pixels(
        _Image([]),
        _frame(0),
        np.array([[1.0, 2.0]], dtype=np.float64),
        transform,
    )

    np.testing.assert_allclose(mapped, np.array([[6.0, 3.0]]), atol=1e-6)


def test_segmented_colmap_export_uses_persistent_region_colors(tmp_path: Path) -> None:
    source_path = tmp_path / "aligned.ply"
    positions = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=np.float64)
    source_colors = np.asarray([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(positions)
    cloud.colors = o3d.utility.Vector3dVector(source_colors.astype(np.float64) / 255.0)
    assert o3d.io.write_point_cloud(str(source_path), cloud, write_ascii=False)

    reconstruction = SimpleNamespace(
        points3D={
            9: SimpleNamespace(color=source_colors[2]),
            2: SimpleNamespace(color=source_colors[0]),
            5: SimpleNamespace(color=source_colors[1]),
        }
    )
    point_labels = np.zeros(10, dtype=np.uint16)
    point_labels[5] = 3
    point_labels[9] = 4
    config = ColmapTrackConfig(
        model_path=tmp_path,
        aligned_points_path=source_path,
        segmented_cache_path=tmp_path / "segmented.npz",
        segmented_ply_path=tmp_path / "segmented.ply",
        segmented_summary_path=tmp_path / "segmented.json",
    )

    summary = _save_segmented_colmap_points(
        reconstruction,
        point_labels,
        [
            {"id": 3, "name": "door", "color": "hsl(0 100% 50%)"},
            {"id": 4, "name": "arch", "color": "hsl(120 100% 50%)"},
        ],
        config,
    )

    assert summary is not None
    assert summary["labeledPointCount"] == 2
    with np.load(config.segmented_cache_path) as data:
        np.testing.assert_allclose(data["positions"], positions[1:])
        np.testing.assert_array_equal(data["labels"], np.asarray([3, 4], dtype=np.uint16))
        np.testing.assert_array_equal(data["point_ids"], np.asarray([5, 9], dtype=np.int64))
        np.testing.assert_array_equal(data["colors"], np.asarray([[255, 0, 0], [0, 255, 0]], dtype=np.uint8))
