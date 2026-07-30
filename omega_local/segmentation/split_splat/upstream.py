"""Released Split&Splat command construction and execution."""

from __future__ import annotations

import codecs
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from .contract import (
    SplitSplatRunConfig,
    SplitSplatRunPaths,
    atomic_write_json,
    now_utc,
    replace_symlink,
)


ProgressCallback = Callable[[str], None]


def _upstream_python_paths(config: SplitSplatRunConfig) -> list[str]:
    root = config.split_splat_root
    return [
        str(root.parent / "sam2"),
        str(root),
        str(root / "utils"),
        str(root / "point_projection"),
        str(root / "submodules" / "diff-gaussian-rasterization"),
        str(root / "submodules" / "simple-knn"),
        str(root / "submodules" / "fused-ssim"),
        str(Path(__file__).resolve().parents[3]),
    ]


def upstream_availability(config: SplitSplatRunConfig) -> dict[str, Any]:
    root = config.split_splat_root
    checkpoint = root / "checkpoints" / "sam2.1_hiera_large.pt"
    fallback_checkpoint = root.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    checkpoint_source = checkpoint if checkpoint.is_file() else fallback_checkpoint
    abi_result = subprocess.run(
        [str(config.python), "-c", "import sys; print(f'cpython-{sys.version_info.major}{sys.version_info.minor}')"],
        check=False,
        capture_output=True,
        text=True,
    )
    python_abi = abi_result.stdout.strip() if abi_result.returncode == 0 else "unknown"
    projection_extensions = sorted((root / "point_projection").glob("point_projection_cuda*.so"))
    compatible_projection = next(
        (path for path in projection_extensions if python_abi in path.name),
        None,
    )
    required = {
        "root": root,
        "autoSeg": root / "sam2" / "auto_seg.py",
        "maskPropagation": root / "sam2" / "mask_propagation.py",
        "maskOptimizer": root / "utils_mask" / "mask_optimizer_scannet.py",
        "train": root / "train.py",
        "sam2Checkpoint": checkpoint_source,
    }
    missing = [key for key, path in required.items() if not path.exists()]
    if compatible_projection is None:
        missing.append("pointProjectionPythonAbi")
    runtime_env = os.environ.copy()
    runtime_env["PYTHONPATH"] = os.pathsep.join(
        _upstream_python_paths(config)
        + ([runtime_env["PYTHONPATH"]] if runtime_env.get("PYTHONPATH") else [])
    )
    runtime_probe = subprocess.run(
        [
            str(config.python),
            "-c",
            (
                "import json, pycolmap, torch; "
                "import point_projection_cuda; "
                "import diff_gaussian_rasterization._C; "
                "import simple_knn._C; "
                "import fused_ssim_cuda; "
                "print(json.dumps({'cudaAvailable': torch.cuda.is_available(), "
                "'cudaVersion': torch.version.cuda, "
                "'deviceCount': torch.cuda.device_count(), "
                "'pycolmapVersion': pycolmap.__version__}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=runtime_env,
    )
    runtime_payload: dict[str, Any] = {}
    if runtime_probe.returncode == 0:
        try:
            runtime_payload = json.loads(runtime_probe.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            runtime_payload = {}
    if runtime_probe.returncode != 0:
        missing.append("nativeExtensionImports")
    elif not bool(runtime_payload.get("cudaAvailable")):
        missing.append("cudaRuntime")
    if runtime_payload.get("pycolmapVersion") != "3.11.1":
        missing.append("pycolmapVersion")
    return {
        "available": not missing,
        "paths": {
            **{key: str(path) for key, path in required.items()},
            "pointProjectionExtension": (
                str(compatible_projection)
                if compatible_projection is not None
                else str(root / "point_projection")
            ),
        },
        "missing": missing,
        "python": str(config.python),
        "pythonAbi": python_abi,
        "availablePointProjectionExtensions": [str(path) for path in projection_extensions],
        "runtimeProbe": {
            **runtime_payload,
            "returnCode": int(runtime_probe.returncode),
            "stderr": runtime_probe.stderr.strip()[-2000:],
        },
        "runtimeCheckpointPath": str(checkpoint),
    }


def ensure_upstream_runtime(config: SplitSplatRunConfig) -> None:
    checkpoint = config.split_splat_root / "checkpoints" / "sam2.1_hiera_large.pt"
    if checkpoint.is_file():
        return
    source = config.split_splat_root.parent / "sam2" / "checkpoints" / "sam2.1_hiera_large.pt"
    if not source.is_file():
        raise FileNotFoundError(f"SAM2 large checkpoint is missing: {checkpoint}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    replace_symlink(checkpoint, source)


def upstream_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def prepare_official_proposal_workspace(run_paths: SplitSplatRunPaths) -> Path:
    work = run_paths.proposals_dir / "work"
    if work.exists():
        shutil.rmtree(work)
    (work / "data").mkdir(parents=True, exist_ok=True)
    (work / "output").mkdir(parents=True, exist_ok=True)
    replace_symlink(work / "data" / run_paths.run_id, run_paths.dataset_dir)
    return work


def official_proposals_command(
    config: SplitSplatRunConfig,
    run_paths: SplitSplatRunPaths,
) -> list[str]:
    return [
        str(config.python),
        str(config.split_splat_root / "sam2" / "auto_seg.py"),
        "--scene",
        run_paths.run_id,
        "--scene_path",
        "./data",
    ]


def global_gs_command(
    config: SplitSplatRunConfig,
    run_paths: SplitSplatRunPaths,
) -> list[str]:
    return [
        str(config.python),
        str(config.split_splat_root / "train.py"),
        "-s",
        str(run_paths.dataset_dir),
        "-m",
        str(run_paths.global_gs_model),
        "--images",
        "images",
        "--depths",
        "depth",
        "--iterations",
        str(config.iterations),
        "--save_iterations",
        str(config.iterations),
        "--test_iterations",
        str(config.iterations),
        "--disable_viewer",
    ]


def prepare_split_workspace(run_paths: SplitSplatRunPaths) -> Path:
    work = run_paths.split_work_dir
    if work.exists():
        shutil.rmtree(work)
    scene_dir = work / "data" / run_paths.run_id
    sparse_dir = scene_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    (work / "output").mkdir(parents=True, exist_ok=True)
    replace_symlink(scene_dir / "images", run_paths.dataset_dir / "images")
    replace_symlink(scene_dir / "depth", run_paths.dataset_dir / "depth")
    replace_symlink(sparse_dir / "cameras.txt", run_paths.dataset_dir / "sparse" / "0" / "cameras.txt")
    replace_symlink(sparse_dir / "images.txt", run_paths.dataset_dir / "sparse" / "0" / "images.txt")
    replace_symlink(sparse_dir / "points3D.txt", run_paths.dataset_dir / "sparse" / "0" / "points3D.txt")
    replace_symlink(sparse_dir / "points3D.ply", run_paths.global_point_cloud)
    replace_symlink(
        work / "output" / f"{run_paths.run_id}_autoseg_mask",
        run_paths.proposal_binary_masks,
    )
    return work


def split_command(
    config: SplitSplatRunConfig,
    run_paths: SplitSplatRunPaths,
    *,
    verbose: bool,
) -> list[str]:
    command = [
        str(config.python),
        str(config.split_splat_root / "sam2" / "mask_propagation.py"),
        "-dataset",
        run_paths.run_id,
    ]
    if verbose:
        command.append("--verbose")
    return command


def instance_train_command(
    config: SplitSplatRunConfig,
    *,
    dataset_dir: Path,
    model_dir: Path,
    initial_pass: bool,
) -> list[str]:
    """Build the released per-instance 1k reconstruction command."""
    half = max(config.instance_iterations // 2, 1)
    command = [
        str(config.python),
        str(config.split_splat_root / "train.py"),
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(config.instance_iterations),
        "--is_instance",
        "--test_iterations",
        str(half),
        str(config.instance_iterations),
        "--save_iterations",
        str(half),
        str(config.instance_iterations),
        "--disable_viewer",
    ]
    if initial_pass:
        command.append("--init_rec")
    return command


def mask_refinement_command(
    config: SplitSplatRunConfig,
    run_paths: SplitSplatRunPaths,
    *,
    instance_id: int,
) -> list[str]:
    released = [
        str(config.python),
        str(config.split_splat_root / "utils_mask" / "mask_optimizer_scannet.py"),
        "-m",
        str(run_paths.splat_initial_models / str(instance_id)),
        "--instance_test",
        str(instance_id),
        "--scene",
        run_paths.run_id,
    ]
    return [
        str(config.python),
        "-m",
        "omega_local.segmentation.split_splat.launch",
        "--normalize-camera-image-names",
        "--paper-mask-refinement",
        *released[1:],
    ]


def composition_train_command(
    config: SplitSplatRunConfig,
    *,
    dataset_dir: Path,
    model_dir: Path,
    mask_weight: float,
    foreground_balanced: bool = False,
) -> list[str]:
    """Build a paper-weighted composition command without editing upstream."""
    half = max(config.composition_iterations // 2, 1)
    launcher = [
        str(config.python),
        "-m",
        "omega_local.segmentation.split_splat.train_launch",
        "--upstream-train",
        str(config.split_splat_root / "train.py"),
        "--mask-loss-weight",
        f"{float(mask_weight):.9g}",
        "--paper-reset-opacity",
    ]
    if foreground_balanced:
        launcher.append("--foreground-balanced")
    return [
        *launcher,
        "--",
        "-s",
        str(dataset_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(config.composition_iterations),
        "--composition",
        "--is_instance",
        "--densify_from_iter",
        "999999",
        "--test_iterations",
        str(half),
        str(config.composition_iterations),
        "--save_iterations",
        str(half),
        str(config.composition_iterations),
        "--disable_viewer",
    ]


def run_command(
    command: list[str],
    *,
    cwd: Path,
    config: SplitSplatRunConfig,
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
        log_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        print(
            f"[split-splat] planned command: {' '.join(command)}",
            file=sys.stderr,
            flush=True,
        )
        return {**payload, "returnCode": 0, "finishedUtc": now_utc()}

    env = os.environ.copy()
    python_paths = _upstream_python_paths(config)
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log:
        rendered_command = " ".join(command)
        log.write("$ " + rendered_command + "\n\n")
        log.flush()
        print(
            f"[split-splat] running: {rendered_command}",
            file=sys.stderr,
            flush=True,
        )
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
        pending_progress = ""
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
                    progress_text = (pending_progress + decoded).replace("\r", "\n")
                    progress_rows = progress_text.split("\n")
                    pending_progress = progress_rows.pop()
                    for row in progress_rows:
                        row = row.strip()
                        if row:
                            progress(row)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        decoded = decoder.decode(b"", final=True)
        if decoded:
            log.write(decoded)
            log.flush()
            sys.stderr.write(decoded)
            sys.stderr.flush()
            pending_progress += decoded
        if progress is not None and pending_progress.strip():
            progress(pending_progress.strip())
        return_code = process.wait()
    result = {**payload, "returnCode": int(return_code), "finishedUtc": now_utc()}
    atomic_write_json(log_path.with_suffix(".json"), result)
    if return_code != 0:
        raise RuntimeError(
            f"Split&Splat command failed with exit code {return_code}. See {log_path}"
        )
    return result


def write_compatibility_report(
    config: SplitSplatRunConfig,
    run_paths: SplitSplatRunPaths,
) -> dict[str, Any]:
    proposal_substitution = config.proposal_source == "propagation"
    report = {
        "schemaVersion": 1,
        "timestampUtc": now_utc(),
        "paper": "https://arxiv.org/html/2602.03809v1",
        "source": "https://github.com/LTTM/Split_and_Splat",
        "upstreamRoot": str(config.split_splat_root),
        "upstreamCommit": upstream_commit(config.split_splat_root),
        "availability": upstream_availability(config),
        "policy": (
            "paper_algorithm_using_released_training_and_split_scripts"
            if not proposal_substitution
            else "paper_split_algorithm_with_controlled_input_mask_substitution"
        ),
        "adapterChanges": [
            "Use an isolated per-stage working directory for fixed ./data and ./output paths.",
            "Stage immutable editor images under non-mutating .JPEG names.",
            "Write PINHOLE COLMAP text cameras for the editor's exact raster grid.",
            "Expose global Gaussian means as sparse/0/points3D.ply only inside the Split workspace.",
            "Convert upstream per-instance binary masks into an exclusive uint16 preview layer.",
            "Treat empty point sets as no-op projection calls instead of zero-block CUDA launches.",
            (
                "Stage each Split instance as an isolated COLMAP dataset while "
                "reusing the shared RGB/camera/depth support by symlink."
            ),
            (
                "Normalize camera image stems and apply the paper's all-view, "
                "tau_iou=0.95 mask-refinement controls at runtime. The released "
                "ScanNet script otherwise processes roughly half the cameras "
                "and uses tau_iou=0.05 for previously missing masks."
            ),
            (
                "Execute paper composition mask weights and opacity reset through "
                "a runtime launcher; the upstream train.py file is never edited."
            ),
            (
                "Store composition groups under bounded deterministic keys while "
                "retaining complete instance memberships in stage metadata."
            ),
            (
                "For the Depth Anything ablation, calibrate inverse depth to the "
                "global Gaussian geometry with the released median/MAD affine rule."
            ),
        ],
        "algorithmChanges": (
            []
            if not proposal_substitution
            else [
                (
                    "Replace the released four-grid automatic masks with the "
                    f"editor's {config.propagation_method} exclusive masks. "
                    "Global 3DGS training and the released Split propagation code "
                    "are unchanged."
                )
            ]
        ),
        "proposalSource": config.proposal_source,
        "propagationMethod": (
            config.propagation_method if proposal_substitution else None
        ),
        "knownReleaseIssues": [
            "The released auto_seg.py writes temporary NPZ files to ./data and mutates .jpg/.png names.",
            "The released general propagation script loads sparse/0/points3D.ply; the adapter supplies the paper's global GS means there.",
            "The released propagation script requests TkAgg; the launcher requests a headless backend but does not alter algorithm parameters.",
            (
                "The released run_all*.sh scripts use 1,000 iterations and a "
                "singular --save_iteration spelling; the adapter uses train.py's "
                "actual --save_iterations argument with the same checkpoints."
            ),
            (
                "The paper states tau_iou=0.95 for accepting a newly recovered "
                "missing mask, while released mask_optimizer_scannet.py gates at "
                "0.05 and stops after roughly half the cameras. The adapter "
                "applies the paper values without editing the upstream checkout."
            ),
            (
                "The README composition example uses weights 0.05, 0.1, 0.25, "
                "while the paper states +0.1 increments capped at 0.25. The "
                "adapter defaults to the paper schedule 0.05, 0.15, 0.25."
            ),
            (
                "Released composition folders concatenate all merged instance "
                "names and can exceed filesystem component limits on large "
                "scenes; the adapter uses bounded storage keys."
            ),
        ],
        "depthSource": config.depth_source,
        "faithfulDepthSource": config.depth_source == "murre",
        "depthNote": (
            "Murre is the paper-faithful DSLR depth source."
            if config.depth_source == "murre"
            else "Depth Anything V2 is an explicit diagnostic ablation, not the official baseline."
        ),
        "splatSettings": {
            "instanceIterations": config.instance_iterations,
            "compositionIterations": config.composition_iterations,
            "compositionMaskWeights": list(config.composition_mask_weights),
            "compositionDensification": False,
        },
    }
    atomic_write_json(run_paths.compatibility_report, report)
    return report
