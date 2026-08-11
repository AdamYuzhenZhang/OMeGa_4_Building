"""Launch the released hard- or soft-identity static 3DGS trainer."""

from __future__ import annotations

import codecs
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from omega_local.segmentation.split_splat.contract import now_utc, read_json

from .contract import StaticSemantic3DGSConfig, StaticSemantic3DGSPaths


ProgressCallback = Callable[[str], None]


def train_static_semantic_3dgs(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
    *,
    progress: ProgressCallback | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if config.method == "segment_then_splat":
        command = _segment_then_splat_command(config, paths)
    elif config.method == "gaussian_grouping":
        command = _gaussian_grouping_command(config, paths)
    else:
        raise AssertionError(config.method)
    environment = _method_environment(config)
    log_path = paths.logs_dir / "train.log"
    result = _run_command(
        command,
        cwd=config.method_root,
        log_path=log_path,
        environment=environment,
        progress=progress,
        dry_run=dry_run,
    )
    if dry_run:
        return {
            "schemaVersion": 1,
            "stage": "train",
            "status": "planned",
            "method": config.display_name,
            "command": command,
            "modelOutput": str(paths.model_output),
        }

    final_dir = paths.model_output / "point_cloud" / f"iteration_{config.iterations}"
    final_ply = final_dir / "point_cloud.ply"
    if not final_ply.is_file():
        raise FileNotFoundError(
            f"{config.display_name} completed without its final PLY: {final_ply}"
        )
    outputs: dict[str, str] = {"nativeScenePly": str(final_ply)}
    if config.method == "segment_then_splat":
        hard_ids = paths.model_output / f"default_object_id_{config.iterations}.pth"
        if not hard_ids.is_file():
            raise FileNotFoundError(
                f"Segment then Splat hard object IDs are missing: {hard_ids}"
            )
        outputs["hardObjectIds"] = str(hard_ids)
    else:
        classifier = final_dir / "classifier.pth"
        if not classifier.is_file():
            raise FileNotFoundError(
                f"Gaussian Grouping classifier is missing: {classifier}"
            )
        outputs["classifier"] = str(classifier)

    return {
        "schemaVersion": 1,
        "stage": "train",
        "timestampUtc": now_utc(),
        "method": config.display_name,
        "identityModel": (
            "fixed_hard_object_id"
            if config.method == "segment_then_splat"
            else "learned_16d_identity_plus_classifier"
        ),
        "iterations": int(config.iterations),
        "densifyUntilIteration": int(config.densify_until_iter),
        "command": result["command"],
        "log": str(log_path),
        "outputs": outputs,
    }


def validate_runtime(config: StaticSemantic3DGSConfig) -> dict[str, Any]:
    if not config.python.is_file():
        raise FileNotFoundError(f"Python does not exist: {config.python}")
    train_script = config.method_root / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(
            f"{config.display_name} source does not exist: {config.method_root}"
        )
    probe = (
        "import inspect, diff_gaussian_rasterization as d; "
        "print(inspect.signature(d.GaussianRasterizer.forward))"
    )
    process = subprocess.run(
        [str(config.python), "-c", probe],
        cwd=str(config.method_root),
        env=_method_environment(config),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"{config.display_name} rasterizer import failed:\n"
            f"{process.stderr.strip()}"
        )
    signature = process.stdout.strip()
    required_token = (
        "means2D_densify"
        if config.method == "segment_then_splat"
        else "sh_objs"
    )
    if required_token not in signature:
        raise RuntimeError(
            f"{config.display_name} needs a different CUDA rasterizer. "
            f"Expected {required_token!r} in {signature!r}. Run the matching "
            "setup command before training."
        )
    return {
        "python": str(config.python),
        "methodRoot": str(config.method_root),
        "rasterizerSignature": signature,
    }


def _segment_then_splat_command(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
) -> list[str]:
    return [
        str(config.python),
        "-u",
        str(config.method_root / "train.py"),
        "-s",
        str(paths.dataset_dir),
        "-m",
        str(paths.model_output),
        "--iterations",
        str(config.iterations),
        "--resolution",
        str(config.resolution),
        "--num_sample_objects",
        str(config.num_sample_objects),
        "--densify_until_iter",
        str(config.densify_until_iter),
        "--partial_mask_iou",
        str(config.partial_mask_iou),
        "--stage1_iters",
        "0",
        "--stage2_iters",
        "0",
        "--load2gpu_on_the_fly",
        "--save_iterations",
        str(config.iterations),
        "--test_iterations",
        str(config.iterations),
    ]


def _gaussian_grouping_command(
    config: StaticSemantic3DGSConfig,
    paths: StaticSemantic3DGSPaths,
) -> list[str]:
    region_count = len(read_json(paths.shared_region_map)["regions"])
    return [
        str(config.python),
        "-u",
        str(config.method_root / "train.py"),
        "-s",
        str(paths.dataset_dir),
        "-m",
        str(paths.model_output),
        "-r",
        str(config.resolution),
        "--object_path",
        "object_mask",
        "--num_classes",
        str(region_count + 1),
        "--iterations",
        str(config.iterations),
        "--config_file",
        str(paths.dataset_dir / "train_config.json"),
        "--data_device",
        "cpu",
        "--save_iterations",
        str(config.iterations),
        "--test_iterations",
        str(config.iterations),
    ]


def _method_environment(config: StaticSemantic3DGSConfig) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if config.method == "segment_then_splat":
        extension_dir = config.native_extensions_dir
        if extension_dir is None:
            raise ValueError(
                "Segment then Splat requires --native-extensions-dir."
            )
        prior = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            str(extension_dir)
            if not prior
            else f"{extension_dir}{os.pathsep}{prior}"
        )
    return environment


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
    progress: ProgressCallback | None,
    dry_run: bool,
) -> dict[str, Any]:
    if dry_run:
        return {"command": command, "cwd": str(cwd), "dryRun": True}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
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
            lines = pending.split("\n")
            pending = lines.pop()
            for line in lines:
                message = line.strip()
                if message:
                    print(message, flush=True)
                    if progress is not None:
                        progress(message)
        tail = decoder.decode(b"", final=True)
        if tail:
            log.write(tail)
            pending += tail
        if pending.strip():
            print(pending.strip(), flush=True)
            if progress is not None:
                progress(pending.strip())
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"{cwd.name} command failed with exit code {return_code}. "
            f"See {log_path}."
        )
    return {
        "command": command,
        "cwd": str(cwd),
        "returnCode": return_code,
        "log": str(log_path),
    }
