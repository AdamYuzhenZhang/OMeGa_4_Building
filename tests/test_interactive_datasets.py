from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omega_local.segmentation.interactive.colmap_dataset import (
    _display_point_mask,
    _image_sort_key,
)
from omega_local.segmentation.interactive.datasets import (
    ActiveEditorState,
    EditorDatasetRegistry,
    EditorDatasetSpec,
    load_dataset_specs,
)


class _State:
    def __init__(self, name: str) -> None:
        self.name = name

    def value(self) -> str:
        return self.name


def _spec(dataset_id: str) -> EditorDatasetSpec:
    return EditorDatasetSpec(
        dataset_id=dataset_id,
        name=dataset_id,
        kind="omega",
        model_dir=Path("/tmp") / dataset_id,
        baseline_name="baseline",
        default_rotate_frames=dataset_id == "portrait",
    )


def test_dataset_registry_delegates_to_request_context() -> None:
    portrait = _spec("portrait")
    square = _spec("square")
    registry = EditorDatasetRegistry(
        [(portrait, _State("portrait")), (square, _State("square"))],
        default_dataset_id="portrait",
    )
    active = ActiveEditorState(registry)

    assert active.value() == "portrait"
    token = registry.activate("square")
    try:
        assert active.value() == "square"
        assert registry.summary()["activeDatasetId"] == "square"
    finally:
        registry.reset(token)
    assert active.value() == "portrait"


def test_dataset_registry_rejects_shared_output_directory() -> None:
    first = _spec("first")
    second = EditorDatasetSpec(
        dataset_id="second",
        name="second",
        kind=first.kind,
        model_dir=first.model_dir,
        baseline_name=first.baseline_name,
        default_rotate_frames=False,
    )

    try:
        EditorDatasetRegistry(
            [(first, _State("first")), (second, _State("second"))],
            default_dataset_id="first",
        )
    except ValueError as exc:
        assert "independent output directories" in str(exc)
    else:
        raise AssertionError("Expected a shared editor output directory to be rejected")


def test_dataset_summary_exposes_canonical_output_directory() -> None:
    spec = _spec("square")
    summary = spec.summary(active=True)

    assert summary["outputDir"] == str(spec.output_dir)
    assert summary["active"] is True


def test_load_omega_dataset_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "datasets.json"
    registry_path.write_text(
        json.dumps(
            {
                "defaultDatasetId": "square",
                "datasets": [
                    {
                        "id": "square",
                        "name": "Square Capture",
                        "kind": "omega",
                        "modelDir": "./model",
                        "baselineName": "native",
                        "defaultRotateFrames": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    specs, default_id = load_dataset_specs(registry_path)

    assert default_id == "square"
    assert specs[0].model_dir == (tmp_path / "model").resolve()
    assert specs[0].baseline_name == "native"
    assert specs[0].default_rotate_frames is False


def test_native_colmap_frame_order_is_stream_then_numeric_frame() -> None:
    names = [
        "cam_front_f10.png",
        "cam_back_f2.png",
        "cam_front_f2.png",
        "cam_back_f10.png",
        "cam_back_f1.png",
    ]
    assert sorted(names, key=_image_sort_key) == [
        "cam_back_f1.png",
        "cam_back_f2.png",
        "cam_back_f10.png",
        "cam_front_f2.png",
        "cam_front_f10.png",
    ]


def test_native_colmap_display_filter_rejects_extreme_outlier() -> None:
    positions = np.concatenate(
        [
            np.stack(
                [
                    np.linspace(-1.0, 1.0, 200),
                    np.zeros(200),
                    np.ones(200),
                ],
                axis=1,
            ),
            np.asarray([[1.0e8, 0.0, 0.0]]),
        ],
        axis=0,
    )
    errors = np.ones(positions.shape[0], dtype=np.float64)
    tracks = np.full(positions.shape[0], 3, dtype=np.int32)

    selected, summary = _display_point_mask(positions, errors, tracks)

    assert not bool(selected[-1])
    assert int(np.count_nonzero(selected)) > 190
    assert summary["method"] == "track_error_then_robust_distance_quantile"
