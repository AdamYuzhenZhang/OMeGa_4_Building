from pathlib import Path

import numpy as np

from omega_local.segmentation.interactive.feedforward_point_propagation import (
    label_points_from_anchors,
    project_visible_points,
    rasterize_projected_point_labels,
)
from omega_local.segmentation.interactive.sam2_video_propagation import (
    LabeledPropagationSource,
    PropagationFrame,
)


def _frame(frame_id: int = 0) -> PropagationFrame:
    return PropagationFrame(
        frame_id=frame_id,
        image_name=f"{frame_id:06d}.jpg",
        image_path=Path(f"{frame_id:06d}.jpg"),
        width=9,
        height=7,
        fx=4.0,
        fy=4.0,
        cx=4.0,
        cy=3.0,
        pose_world_from_camera=np.eye(4, dtype=np.float64),
    )


def test_projection_keeps_only_front_depth_band_per_pixel() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 2.00],
            [0.0, 0.0, 2.03],
            [0.0, 0.0, 2.20],
            [0.5, 0.0, 2.00],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    ids, pixels, depth = project_visible_points(
        points,
        _frame(),
        depth_band_m=0.05,
        depth_band_relative=0.0,
    )

    np.testing.assert_array_equal(ids, np.asarray([0, 1, 3], dtype=np.int64))
    np.testing.assert_array_equal(pixels, np.asarray([[4, 3], [4, 3], [5, 3]], dtype=np.int32))
    np.testing.assert_allclose(depth, np.asarray([2.0, 2.03, 2.0], dtype=np.float32))


def test_anchor_voting_keeps_majority_and_rejects_tie() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]], dtype=np.float32)
    frames = [_frame(0), _frame(1)]
    labels_a = np.zeros((7, 9), dtype=np.uint16)
    labels_b = np.zeros((7, 9), dtype=np.uint16)
    labels_a[3, 4] = 7
    labels_b[3, 4] = 7
    labels_a[3, 5] = 7
    labels_b[3, 5] = 9
    sources = [
        LabeledPropagationSource(local_index=0, frame_id=0, labels=labels_a),
        LabeledPropagationSource(local_index=1, frame_id=1, labels=labels_b),
    ]

    point_labels, summary = label_points_from_anchors(
        points,
        sources,
        frames,
        depth_band_m=0.05,
        depth_band_relative=0.0,
    )

    np.testing.assert_array_equal(point_labels, np.asarray([7, 0], dtype=np.uint16))
    assert summary["labeledPointCount"] == 1
    assert summary["tiedPointCount"] == 1
    assert summary["positiveAnchorObservations"] == 4


def test_sparse_raster_uses_persistent_point_labels() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0]], dtype=np.float32)
    point_labels = np.asarray([4, 9], dtype=np.uint16)

    labels, point_count = rasterize_projected_point_labels(
        points,
        point_labels,
        _frame(),
        depth_band_m=0.05,
        depth_band_relative=0.0,
        radius=0,
    )

    assert point_count == 2
    assert labels[3, 4] == 4
    assert labels[3, 5] == 9
    assert np.count_nonzero(labels) == 2
