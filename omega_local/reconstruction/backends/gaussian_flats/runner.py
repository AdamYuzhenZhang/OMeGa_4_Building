"""Commands and streamed execution for the released Gaussian Flats code."""

from __future__ import annotations

import codecs
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.split_splat.contract import atomic_write_json, now_utc

from .contract import GaussianFlatsConfig, GaussianFlatsPaths


ProgressCallback = Callable[[str], None]


def train_command(config: GaussianFlatsConfig, paths: GaussianFlatsPaths) -> list[str]:
    return [
        str(config.python),
        "-u",
        str(config.gaussian_flats_root / "train_planar.py"),
        "-s",
        str(paths.scene_dir),
        "-m",
        str(paths.model_dir),
        "--mask_root",
        str(paths.plane_masks_dir),
        "--init_type",
        "sfm",
        "--eval",
        "--resolution",
        str(config.resolution),
        "--iterations",
        str(config.iterations),
        "--cap_max",
        str(config.cap_max),
        "--plane_fit_iter",
        str(config.plane_fit_iter),
        "--plane_fit_min_points",
        str(config.plane_fit_min_points),
        "--plane_sigma_res",
        str(config.plane_sigma_res),
        "--plane_sigma_dist",
        str(config.plane_sigma_dist),
        "--planar_mask_loss_weight",
        str(config.planar_mask_loss_weight),
        "--depthtv_loss_weight",
        str(config.depthtv_loss_weight),
        "--scale_reg",
        str(config.scale_reg),
        "--opacity_reg",
        str(config.opacity_reg),
        "--test_iterations",
        str(config.iterations),
        "--save_iterations",
        str(config.iterations),
    ]


def render_commands(config: GaussianFlatsConfig, paths: GaussianFlatsPaths) -> list[list[str]]:
    common = ["-m", str(paths.model_dir), "--iteration", str(config.iterations)]
    return [
        [
            str(config.python),
            "-u",
            str(config.gaussian_flats_root / "render.py"),
            *common,
            "--skip_video",
            "--skip_mesh",
        ],
        [
            str(config.python),
            "-u",
            str(config.gaussian_flats_root / "render_planar.py"),
            *common,
            "--skip_video",
            "--skip_mesh",
        ],
    ]


def mesh_commands(config: GaussianFlatsConfig, paths: GaussianFlatsPaths) -> list[list[str]]:
    common = ["-m", str(paths.model_dir), "--iteration", str(config.iterations)]
    return [
        [
            str(config.python),
            "-u",
            str(config.gaussian_flats_root / "render.py"),
            *common,
            "--skip_train",
            "--skip_test",
            "--skip_video",
            "--mesh_voxel_size",
            str(config.mesh_voxel_size),
        ],
        [
            str(config.python),
            "-u",
            str(config.gaussian_flats_root / "render_planar.py"),
            *common,
            "--skip_train",
            "--skip_test",
            "--skip_video",
            "--grid_resolution",
            str(config.planar_grid_resolution),
            "--tile_size",
            str(config.planar_tile_size),
        ],
    ]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    payload = {
        "command": command,
        "cwd": str(cwd),
        "startedUtc": now_utc(),
        "dryRun": bool(dry_run),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text("$ " + " ".join(command) + "\n", encoding="utf-8")
        return {**payload, "returnCode": 0, "finishedUtc": now_utc()}

    env = os.environ.copy()
    python_paths = [
        str(source)
        for source in (
            cwd,
            cwd / "submodules" / "diff-gaussian-rasterization",
            cwd / "submodules" / "simple-knn",
            cwd.parent / "torch_kdtree",
        )
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log:
        rendered = " ".join(command)
        log.write("$ " + rendered + "\n\n")
        log.flush()
        print(f"[gaussian-flats] running: {rendered}", file=sys.stderr, flush=True)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=False,
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        try:
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                decoded = decoder.decode(chunk)
                log.write(decoded)
                log.flush()
                sys.stderr.write(decoded)
                sys.stderr.flush()
                if progress is not None:
                    rows = (pending + decoded).replace("\r", "\n").split("\n")
                    pending = rows.pop()
                    for row in rows:
                        if row.strip():
                            progress(row.strip())
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        tail = decoder.decode(b"", final=True)
        if tail:
            log.write(tail)
            sys.stderr.write(tail)
            pending += tail
        if progress is not None and pending.strip():
            progress(pending.strip())
        return_code = process.wait()
    result = {**payload, "returnCode": return_code, "finishedUtc": now_utc()}
    atomic_write_json(log_path.with_suffix(".json"), result)
    if return_code != 0:
        raise RuntimeError(
            f"3D Gaussian Flats command failed with exit code {return_code}. "
            f"See {log_path}."
        )
    return result
