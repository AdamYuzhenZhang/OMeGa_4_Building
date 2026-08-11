"""Run the released SAM1 proposal plus SAM2 tracking stage unchanged."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.split_splat.contract import atomic_write_json, now_utc

from .contract import LEVELS, OfficialSegmentThenSplatConfig, OfficialSegmentThenSplatPaths
from .execution import run_command


Progress = Callable[[str], None]


def run_official_autoseg(
    config: OfficialSegmentThenSplatConfig,
    paths: OfficialSegmentThenSplatPaths,
    *,
    progress: Progress | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    script = (
        config.source_root
        / "third_party"
        / "AutoSeg-SAM2"
        / "auto-mask-fast.py"
    )
    if not script.is_file():
        raise FileNotFoundError(f"Released AutoSeg-SAM2 script is missing: {script}")
    _prepare_runtime(config, paths)
    level_rows = []
    for level in LEVELS:
        level_summary = paths.autoseg_stage / f"{level}.json"
        output_dir = paths.autoseg_output / level / "final-output"
        complete = len(list(output_dir.glob("mask_*.npy"))) == _frame_count(paths)
        if complete and not dry_run:
            level_rows.append({"level": level, "cacheHit": True})
            continue
        if not dry_run and (paths.autoseg_output / level).exists():
            shutil.rmtree(paths.autoseg_output / level)
        if progress is not None:
            progress(f"AutoSeg-SAM2 {level}: starting.")
        command = [
            str(config.python),
            "-u",
            str(script),
            "--video_path",
            str(paths.images_dir),
            "--output_dir",
            str(paths.autoseg_output),
            "--level",
            level,
            "--batch_size",
            str(config.batch_size),
            "--detect_stride",
            str(config.detect_stride),
        ]
        result = run_command(
            command,
            cwd=paths.autoseg_runtime,
            log_path=paths.autoseg_stage / "logs" / f"{level}.log",
            environment=_autoseg_environment(config),
            progress=progress,
            dry_run=dry_run,
        )
        if not dry_run:
            count = len(list(output_dir.glob("mask_*.npy")))
            expected = _frame_count(paths)
            if count != expected:
                raise RuntimeError(
                    f"AutoSeg-SAM2 {level} produced {count}/{expected} final masks."
                )
            atomic_write_json(
                level_summary,
                {
                    "schemaVersion": 1,
                    "status": "complete",
                    "level": level,
                    "frameCount": count,
                    "outputDir": str(output_dir),
                    "timestampUtc": now_utc(),
                },
            )
        level_rows.append({"level": level, **result})
    summary = {
        "schemaVersion": 1,
        "stage": "autoseg",
        "timestampUtc": now_utc(),
        "method": "Released AutoSeg-SAM2",
        "levels": list(LEVELS),
        "frameCount": _frame_count(paths),
        "detectStride": int(config.detect_stride),
        "sam1PointsPerSide": 32,
        "predictedIoUThreshold": 0.7,
        "stabilityThreshold": 0.85,
        "outputDir": str(paths.autoseg_output),
        "runs": level_rows,
    }
    if not dry_run:
        atomic_write_json(paths.stage_summary("autoseg"), summary)
    return summary


def _autoseg_environment(
    config: OfficialSegmentThenSplatConfig,
) -> dict[str, str]:
    sam1_root = (
        config.source_root
        / "third_party"
        / "AutoSeg-SAM2"
        / "submodule"
        / "segment-anything-1"
    )
    if not (sam1_root / "segment_anything" / "automatic_mask_generator.py").is_file():
        raise FileNotFoundError(
            f"Released multigranularity SAM1 fork is missing: {sam1_root}"
        )
    prior = os.environ.get("PYTHONPATH", "")
    return {
        "PYTHONPATH": (
            str(sam1_root)
            if not prior
            else f"{sam1_root}{os.pathsep}{prior}"
        ),
        # AutoSeg's object batch is memory-only: reducing it preserves prompts
        # and masks while avoiding allocator fragmentation on long sequences.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def _prepare_runtime(
    config: OfficialSegmentThenSplatConfig,
    paths: OfficialSegmentThenSplatPaths,
) -> None:
    links = {
        paths.autoseg_runtime / "checkpoints" / "sam1" / "sam_vit_h_4b8939.pth": config.sam1_checkpoint,
        paths.autoseg_runtime / "checkpoints" / "sam2" / "sam2_hiera_large.pt": config.sam2_checkpoint,
    }
    for link, target in links.items():
        if not target.is_file():
            raise FileNotFoundError(f"Segmentation checkpoint is missing: {target}")
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)


def _frame_count(paths: OfficialSegmentThenSplatPaths) -> int:
    with paths.frame_map.open("r", encoding="utf-8") as handle:
        return sum(1 for row in handle if row.strip())

