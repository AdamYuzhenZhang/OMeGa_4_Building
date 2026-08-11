"""Subprocess streaming for the released paper stages."""

from __future__ import annotations

import codecs
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.split_splat.contract import atomic_write_json, now_utc


Progress = Callable[[str], None]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str] | None = None,
    progress: Progress | None = None,
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
        return {**payload, "returnCode": 0, "finishedUtc": now_utc()}
    env = os.environ.copy()
    env.update(environment or {})
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
        assert process.stdout is not None
        pending = ""
        while True:
            chunk = process.stdout.read(4096)
            if not chunk:
                break
            decoded = decoder.decode(chunk)
            log.write(decoded)
            log.flush()
            pending += decoded.replace("\r", "\n")
            rows = pending.split("\n")
            pending = rows.pop()
            for row in rows:
                message = row.strip()
                if message:
                    print(message, flush=True)
                    if progress is not None:
                        progress(message)
        tail = decoder.decode(b"", final=True)
        if tail:
            log.write(tail)
            pending += tail
        if pending.strip() and progress is not None:
            progress(pending.strip())
        return_code = process.wait()
    result = {
        **payload,
        "returnCode": int(return_code),
        "finishedUtc": now_utc(),
        "log": str(log_path),
    }
    atomic_write_json(log_path.with_suffix(".json"), result)
    if return_code != 0:
        raise RuntimeError(
            f"Segment then Splat command failed with exit code {return_code}. "
            f"See {log_path}."
        )
    return result

