from pathlib import Path

import numpy as np

from omega_local.segmentation.interactive.colmap_sam2_prompt_recovery import (
    ColmapSam2PromptConfig,
    recover_with_sam2_point_prompts,
)
from omega_local.segmentation.interactive.sam2_session import Sam2Config


class _FakePredictor:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def predict_candidates(
        self,
        *,
        frame_id: int,
        image_rgb: np.ndarray,
        points_xy: np.ndarray,
        point_labels: np.ndarray,
        multimask: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        del frame_id, multimask
        self.calls.append((points_xy.copy(), point_labels.copy()))
        height, width = image_rgb.shape[:2]
        positive = points_xy[point_labels > 0]
        is_left = float(positive[:, 0].mean()) < width / 2

        correct = np.full((height, width), -2.0, dtype=np.float32)
        if is_left:
            correct[4 : height - 4, 2 : width // 2 - 2] = 2.0
        else:
            correct[4 : height - 4, width // 2 + 2 : width - 2] = 2.0
        # This candidate has a higher SAM score but violates every negative prompt.
        overlarge = np.full((height, width), 2.0, dtype=np.float32)
        missed = np.full((height, width), -2.0, dtype=np.float32)
        return np.stack((correct, overlarge, missed)), np.array([0.82, 0.98, 0.30], dtype=np.float32)


def _config() -> ColmapSam2PromptConfig:
    return ColmapSam2PromptConfig(
        source_method_id="colmap_tracks",
        source_run_dir=Path("/unused"),
        sam2=Sam2Config(
            root=Path("/unused"),
            checkpoint=Path("/unused/checkpoint.pt"),
            config="unused.yaml",
            device="cpu",
        ),
        positive_prompt_count=6,
        negative_prompt_count=6,
        prompt_clearance_pixels=2.0,
        min_positive_prompts=2,
    )


def test_sam2_point_recovery_uses_competing_regions_as_negative_prompts() -> None:
    sparse = np.zeros((40, 60), dtype=np.uint16)
    sparse[10:15, 8:13] = 1
    sparse[25:30, 18:23] = 1
    sparse[8:13, 45:50] = 2
    sparse[24:29, 37:42] = 2
    predictor = _FakePredictor()

    dense, stats = recover_with_sam2_point_prompts(
        frame_id=7,
        image_rgb=np.zeros((40, 60, 3), dtype=np.uint8),
        sparse_labels=sparse,
        predictor=predictor,
        config=_config(),
    )

    assert len(predictor.calls) == 2
    assert all(np.any(labels == 0) and np.any(labels == 1) for _points, labels in predictor.calls)
    assert dense[20, 10] == 1
    assert dense[20, 50] == 2
    assert dense[20, 30] == 0
    assert np.array_equal(dense[sparse > 0], sparse[sparse > 0])
    assert stats["acceptedRegionCount"] == 2
    assert stats["rejectedRegionCount"] == 0


def test_sam2_point_recovery_keeps_sparse_support_when_prompts_are_insufficient() -> None:
    sparse = np.zeros((20, 30), dtype=np.uint16)
    sparse[4, 5] = 3
    config = ColmapSam2PromptConfig(
        **{**_config().__dict__, "min_positive_prompts": 4},
    )
    predictor = _FakePredictor()

    dense, stats = recover_with_sam2_point_prompts(
        frame_id=2,
        image_rgb=np.zeros((20, 30, 3), dtype=np.uint8),
        sparse_labels=sparse,
        predictor=predictor,
        config=config,
    )

    assert not predictor.calls
    assert np.array_equal(dense, sparse)
    assert stats["acceptedRegionCount"] == 0
    assert stats["rejectedRegionCount"] == 1
