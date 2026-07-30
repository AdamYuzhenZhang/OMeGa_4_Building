"""Storage cleanup for completed Split&Splat training stages."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def cleanup_completed_models(
    model_root: Path,
    *,
    final_iteration: int,
) -> dict[str, Any]:
    """Keep final models while removing training-only copies and checkpoints."""
    report = _new_report(
        policy="keep_final_models",
        root=model_root,
        final_iteration=int(final_iteration),
    )
    if not model_root.exists() or model_root.is_symlink():
        return report

    for pattern, category in (
        ("input.ply", "inputCopies"),
        ("events.out.tfevents.*", "tensorboardEvents"),
        ("chkpnt*.pth", "optimizerCheckpoints"),
    ):
        for path in model_root.rglob(pattern):
            _remove_file(path, report, category)

    final_name = f"iteration_{int(final_iteration)}"
    iteration_dirs = sorted(
        (
            path
            for path in model_root.rglob("iteration_*")
            if path.is_dir() and not path.is_symlink()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in iteration_dirs:
        if directory.name != final_name:
            _remove_tree(directory, report, "intermediateIterations")
    return report


def cleanup_completed_composition(
    paths: Any,
    stage_summary: dict[str, Any],
) -> dict[str, Any]:
    """Drop resumable merge models only after all final outputs are complete."""
    report = _new_report(
        policy="keep_final_outputs",
        root=paths.splat_composition,
    )
    if stage_summary.get("status") != "complete":
        report["skipped"] = "stage_not_complete"
        return report

    outputs = stage_summary.get("outputs")
    if not isinstance(outputs, dict):
        report["skipped"] = "outputs_missing"
        return report
    required_files = (
        "fullGaussians",
        "instanceIdGaussians",
        "instanceLabels",
        "pointsCache",
    )
    missing = [
        key
        for key in required_files
        if not Path(str(outputs.get(key) or "")).is_file()
    ]
    required_dirs = (
        paths.splat_outputs / "individual_objects",
        paths.splat_outputs / "split_rgb_objects",
    )
    missing.extend(
        str(path)
        for path in required_dirs
        if not path.is_dir()
    )
    if missing:
        report["skipped"] = "final_outputs_incomplete"
        report["missing"] = missing
        return report

    if paths.splat_composition.exists():
        _remove_tree(
            paths.splat_composition,
            report,
            "completedCompositionWorkspace",
        )
    return report


def cleanup_report_changed(report: dict[str, Any]) -> bool:
    return bool(report.get("removedFiles") or report.get("removedDirectories"))


def _new_report(
    *,
    policy: str,
    root: Path,
    final_iteration: int | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "policy": policy,
        "root": str(root),
        "removedBytes": 0,
        "removedFiles": 0,
        "removedDirectories": 0,
        "categories": {},
    }
    if final_iteration is not None:
        report["retainedIteration"] = int(final_iteration)
    return report


def _remove_file(
    path: Path,
    report: dict[str, Any],
    category: str,
) -> None:
    if not path.is_file() and not path.is_symlink():
        return
    size = path.lstat().st_size
    path.unlink(missing_ok=True)
    _record(report, category, files=1, directories=0, size=size)


def _remove_tree(
    path: Path,
    report: dict[str, Any],
    category: str,
) -> None:
    if not path.exists() and not path.is_symlink():
        return
    size, files, directories = _tree_size(path)
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)
    _record(
        report,
        category,
        files=files,
        directories=max(directories, 1),
        size=size,
    )


def _tree_size(path: Path) -> tuple[int, int, int]:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size, 1, 0
    size = 0
    files = 0
    directories = 1
    for child in path.iterdir():
        child_size, child_files, child_directories = _tree_size(child)
        size += child_size
        files += child_files
        directories += child_directories
    return size, files, directories


def _record(
    report: dict[str, Any],
    category: str,
    *,
    files: int,
    directories: int,
    size: int,
) -> None:
    report["removedBytes"] += int(size)
    report["removedFiles"] += int(files)
    report["removedDirectories"] += int(directories)
    categories = report["categories"]
    row = categories.setdefault(
        category,
        {"removedBytes": 0, "removedFiles": 0, "removedDirectories": 0},
    )
    row["removedBytes"] += int(size)
    row["removedFiles"] += int(files)
    row["removedDirectories"] += int(directories)
