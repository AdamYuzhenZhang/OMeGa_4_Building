"""Launch released ObjectGS and discover its timestamped model output."""

from __future__ import annotations

import codecs
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    now_utc,
)

from .contract import ObjectGSConfig, ObjectGSPaths


ProgressCallback = Callable[[str], None]


def train_objectgs(
    config: ObjectGSConfig,
    paths: ObjectGSPaths,
    *,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    train_script = config.objectgs_root / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"ObjectGS train.py is missing: {train_script}")
    command = [
        str(config.python),
        "-u",
        str(train_script),
        "--config",
        str(paths.train_config),
        "--quiet",
    ]
    log_path = paths.logs_dir / "train.log"
    before = {
        path.resolve()
        for path in paths.model_parent.glob("*")
        if path.is_dir()
    }
    result = _run_command(
        command,
        cwd=config.objectgs_root,
        log_path=log_path,
        compat_root=Path(__file__).resolve().parent / "compat",
        objectgs_root=config.objectgs_root,
        progress=progress,
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "schemaVersion": 1,
            "stage": "train",
            "status": "planned",
            "command": command,
            "modelParent": str(paths.model_parent),
        }

    candidates = sorted(
        (
            path.resolve()
            for path in paths.model_parent.glob("*")
            if path.is_dir() and path.resolve() not in before
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        candidates = sorted(
            (path.resolve() for path in paths.model_parent.glob("*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime_ns,
        )
    if not candidates:
        raise RuntimeError(
            f"ObjectGS completed without a model under {paths.model_parent}."
        )
    model_dir = candidates[-1]
    final_ply = (
        model_dir
        / "point_cloud"
        / f"iteration_{config.iterations}"
        / "point_cloud.ply"
    )
    if not final_ply.is_file():
        raise FileNotFoundError(
            f"ObjectGS final anchor model is missing: {final_ply}"
        )
    pointer = {
        "schemaVersion": 1,
        "modelDir": str(model_dir),
        "iteration": int(config.iterations),
        "anchorPly": str(final_ply),
        "trainLog": str(log_path),
    }
    atomic_write_json(paths.model_pointer, pointer)
    return {
        "schemaVersion": 1,
        "stage": "train",
        "timestampUtc": now_utc(),
        "method": "Released ObjectGS 3D shared-scene training",
        **pointer,
        "command": result["command"],
    }


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    compat_root: Path,
    objectgs_root: Path,
    progress: ProgressCallback | None,
    dry_run: bool,
) -> dict[str, Any]:
    payload = {
        "command": command,
        "cwd": str(cwd),
        "startedUtc": now_utc(),
        "dryRun": bool(dry_run),
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        return {**payload, "returnCode": 0, "finishedUtc": now_utc()}

    env = os.environ.copy()
    gsplat_object_root = objectgs_root.parent / "gsplat-object"
    python_paths = [
        str(gsplat_object_root),
        str(compat_root),
        str(objectgs_root),
    ]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
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
                if not decoded:
                    continue
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
    result = {
        **payload,
        "returnCode": int(return_code),
        "finishedUtc": now_utc(),
    }
    atomic_write_json(log_path.with_suffix(".json"), result)
    if return_code != 0:
        raise RuntimeError(
            f"ObjectGS command failed with exit code {return_code}. "
            f"See {log_path}."
        )
    return result
