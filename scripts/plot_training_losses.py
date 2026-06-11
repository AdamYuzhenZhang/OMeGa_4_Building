#!/usr/bin/env python3
"""Plot training losses from ``stats/train_loss.jsonl``.

This script is intentionally small and dependency-light so it can be reused for
baseline and extension runs. It reads the JSONL records written during training,
detects numeric columns, and saves simple matplotlib line plots.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


CORE_LOSSES = ("loss", "l1loss", "ssimloss")
COUNT_COLUMNS = ("num_GS", "num_faces")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training loss curves from a result directory or train_loss.jsonl file."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Result directory, stats directory, or explicit train_loss.jsonl path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for plots. Defaults to <result>/stats/loss_plots.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving-average window in records. 1 disables smoothing.",
    )
    parser.add_argument(
        "--include-zero",
        action="store_true",
        help="Include columns that are present but zero for the whole run.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=140,
        help="Output PNG DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving. Mostly useful on local desktops.",
    )
    return parser.parse_args()


def resolve_train_loss_path(path: Path) -> tuple[Path, Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        train_loss_path = path
        stats_dir = path.parent
    elif (path / "train_loss.jsonl").exists():
        stats_dir = path
        train_loss_path = path / "train_loss.jsonl"
    elif (path / "stats" / "train_loss.jsonl").exists():
        stats_dir = path / "stats"
        train_loss_path = stats_dir / "train_loss.jsonl"
    else:
        raise SystemExit(
            f"Could not find train_loss.jsonl from {path}. "
            "Pass a result directory, stats directory, or the JSONL file itself."
        )
    return train_loss_path, stats_dir


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"Expected a JSON object on {path}:{line_number}")
            rows.append(payload)
    if not rows:
        raise SystemExit(f"No records found in {path}")

    # Result directories are often reused during local sweeps. The trainer
    # appends to train_loss.jsonl, so a rerun can leave duplicate steps from an
    # earlier setting. Keep the latest record for each step before plotting.
    by_step: dict[int, dict[str, object]] = {}
    duplicates = 0
    for row in rows:
        step = row.get("step")
        if not isinstance(step, (int, float)) or isinstance(step, bool) or not math.isfinite(float(step)):
            continue
        step_key = int(step)
        if step_key in by_step:
            duplicates += 1
        by_step[step_key] = row
    if by_step and len(by_step) != len(rows):
        print(
            f"[warning] Found {duplicates} duplicate-step records in {path}; "
            "keeping the latest record per step for plotting."
        )
        rows = [by_step[step] for step in sorted(by_step)]
    return rows


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def numeric_columns(rows: Iterable[dict[str, object]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in seen or key == "step":
                continue
            if is_number(value):
                ordered.append(key)
                seen.add(key)
    return ordered


def column_values(rows: list[dict[str, object]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        values.append(float(value) if is_number(value) else float("nan"))
    return values


def is_all_zero(values: list[float], eps: float = 1e-12) -> bool:
    finite_values = [value for value in values if math.isfinite(value)]
    return bool(finite_values) and all(abs(value) <= eps for value in finite_values)


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = [value for value in values[start : index + 1] if math.isfinite(value)]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def plot_columns(
    *,
    rows: list[dict[str, object]],
    columns: list[str],
    out_path: Path,
    title: str,
    y_label: str,
    smooth_window: int,
    dpi: int,
) -> bool:
    if not columns:
        return False

    steps = [float(row["step"]) for row in rows if is_number(row.get("step"))]
    if len(steps) != len(rows):
        raise SystemExit("Every train_loss row must contain a numeric `step` field.")

    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for column in columns:
        values = smooth(column_values(rows, column), smooth_window)
        if not any(math.isfinite(value) for value in values):
            continue
        ax.plot(steps, values, label=column)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return True


def main() -> int:
    args = parse_args()
    train_loss_path, stats_dir = resolve_train_loss_path(args.path)
    rows = load_jsonl(train_loss_path)
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else stats_dir / "loss_plots"

    columns = numeric_columns(rows)
    loss_columns = [
        column
        for column in columns
        if "loss" in column.lower() or column.lower() in {"distloss"}
    ]
    if not args.include_zero:
        loss_columns = [
            column for column in loss_columns if not is_all_zero(column_values(rows, column))
        ]

    core_columns = [column for column in CORE_LOSSES if column in loss_columns]
    mesh_columns = [
        column
        for column in loss_columns
        if column not in core_columns
    ]
    count_columns = [column for column in COUNT_COLUMNS if column in columns]

    written: list[Path] = []
    plot_jobs = [
        (
            core_columns,
            out_dir / "training_core_losses.png",
            "Core Training Losses",
            "loss",
        ),
        (
            mesh_columns,
            out_dir / "training_mesh_regularization_losses.png",
            "Mesh And Regularization Losses",
            "loss",
        ),
        (
            loss_columns,
            out_dir / "training_all_losses.png",
            "All Training Loss Columns",
            "loss",
        ),
        (
            count_columns,
            out_dir / "training_counts.png",
            "Model Size During Training",
            "count",
        ),
    ]

    for selected_columns, out_path, title, y_label in plot_jobs:
        if plot_columns(
            rows=rows,
            columns=list(selected_columns),
            out_path=out_path,
            title=title,
            y_label=y_label,
            smooth_window=max(1, int(args.smooth_window)),
            dpi=int(args.dpi),
        ):
            written.append(out_path)

    summary = {
        "trainLossJsonl": str(train_loss_path),
        "recordCount": len(rows),
        "firstStep": rows[0].get("step"),
        "lastStep": rows[-1].get("step"),
        "lossColumns": loss_columns,
        "countColumns": count_columns,
        "smoothWindow": max(1, int(args.smooth_window)),
        "plots": [str(path) for path in written],
    }
    summary_path = out_dir / "summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"[done] Read {len(rows)} records from {train_loss_path}")
    for path in written:
        print(f"[done] Wrote {path}")
    print(f"[done] Wrote {summary_path}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
