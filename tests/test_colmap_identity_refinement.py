from pathlib import Path

import numpy as np

from omega_local.segmentation.interactive.colmap_identity_refinement import (
    ColmapIdentityRefinementConfig,
    _relabel_components,
)
from omega_local.segmentation.interactive.colmap_track_propagation import ColmapTrackConfig


def _config() -> ColmapIdentityRefinementConfig:
    return ColmapIdentityRefinementConfig(
        colmap=ColmapTrackConfig(model_path=Path("unused")),
        source_method_id="sam2_video",
        source_run_dir=Path("unused"),
    )


def test_confident_track_consensus_relabels_component_without_changing_shape() -> None:
    source = np.zeros((6, 10), dtype=np.uint16)
    source[1:5, 1:5] = 7
    source_footprint = source > 0
    pixels = np.asarray([[2, 2], [3, 2], [2, 3], [3, 3]], dtype=np.int32)
    regions = np.asarray([9, 9, 9, 7], dtype=np.uint16)
    weights = np.asarray([1.0, 1.0, 1.0, 0.15], dtype=np.float32)

    refined, diagnostics = _relabel_components(source, pixels, regions, weights, _config())

    np.testing.assert_array_equal(refined > 0, source_footprint)
    assert np.all(refined[1:5, 1:5] == 9)
    assert diagnostics["relabeledComponentCount"] == 1
    assert diagnostics["relabeledPixelCount"] == 16
    assert diagnostics["changes"][0]["sourceRegionId"] == 7
    assert diagnostics["changes"][0]["targetRegionId"] == 9


def test_ambiguous_track_consensus_keeps_video_identity() -> None:
    source = np.zeros((6, 10), dtype=np.uint16)
    source[1:5, 1:5] = 7
    pixels = np.asarray([[2, 2], [3, 2], [2, 3], [3, 3]], dtype=np.int32)
    regions = np.asarray([8, 8, 9, 9], dtype=np.uint16)
    weights = np.ones(4, dtype=np.float32)

    refined, diagnostics = _relabel_components(source, pixels, regions, weights, _config())

    np.testing.assert_array_equal(refined, source)
    assert diagnostics["ambiguousComponentCount"] == 1
    assert diagnostics["relabeledComponentCount"] == 0


def test_sparse_track_support_abstains_instead_of_overriding_video() -> None:
    source = np.zeros((6, 10), dtype=np.uint16)
    source[1:5, 1:5] = 7
    pixels = np.asarray([[2, 2], [3, 2]], dtype=np.int32)
    regions = np.asarray([9, 9], dtype=np.uint16)
    weights = np.ones(2, dtype=np.float32)

    refined, diagnostics = _relabel_components(source, pixels, regions, weights, _config())

    np.testing.assert_array_equal(refined, source)
    assert diagnostics["lowSupportComponentCount"] == 1
    assert diagnostics["relabeledComponentCount"] == 0


def test_large_component_requires_area_scaled_track_support() -> None:
    source = np.full((100, 200), 7, dtype=np.uint16)
    pixels = np.asarray([[20, 20], [40, 40], [60, 60]], dtype=np.int32)
    regions = np.asarray([9, 9, 9], dtype=np.uint16)
    weights = np.ones(3, dtype=np.float32)

    refined, diagnostics = _relabel_components(source, pixels, regions, weights, _config())

    np.testing.assert_array_equal(refined, source)
    assert diagnostics["insufficientDensityComponentCount"] == 1
    assert diagnostics["relabeledComponentCount"] == 0
