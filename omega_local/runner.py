"""JSON-config runner for local OMeGa experiments.

The OMeGa trainer is intentionally left as a normal CLI.  This module is a thin
orchestration layer that expands environment variables in JSON configs, converts
config values to trainer flags, and launches ``examples/simple_trainer_meshgs.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
UNRESOLVED_ENV_RE = re.compile(r"\$(\{[^}]+\}|[A-Za-z_][A-Za-z0-9_]*)")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _find_unresolved_env(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, str) and UNRESOLVED_ENV_RE.search(value):
        found.append(f"{prefix or '<root>'}: {value}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_unresolved_env(item, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(_find_unresolved_env(item, child_prefix))
    return found


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    return payload


def load_config(path: Path) -> dict[str, Any]:
    """Load a config and optional parent configs listed in ``extends``."""

    path = path.expanduser().resolve()
    payload = _load_json(path)
    parents = payload.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list):
        raise ValueError(f"`extends` must be a string or list in {path}")

    merged: dict[str, Any] = {}
    for parent in parents:
        parent_path = Path(os.path.expandvars(str(parent))).expanduser()
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, load_config(parent_path))
    merged = _deep_merge(merged, payload)
    merged["_config_path"] = str(path)
    return _expand(merged)


def _parse_assignment(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ValueError(f"Expected KEY=VALUE assignment, got: {raw}")
    key, value = raw.split("=", 1)
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    parts = key.split(".")
    if len(parts) == 1:
        parts = ["args", parts[0]]
    return parts, _expand(parsed)


def apply_overrides(config: dict[str, Any], assignments: list[str]) -> dict[str, Any]:
    config = deepcopy(config)
    for raw in assignments:
        parts, value = _parse_assignment(raw)
        cursor = config
        for part in parts[:-1]:
            next_value = cursor.setdefault(part, {})
            if not isinstance(next_value, dict):
                raise ValueError(f"Cannot assign into non-object config key: {'.'.join(parts)}")
            cursor = next_value
        existing = cursor.get(parts[-1])
        if isinstance(existing, bool) and not isinstance(value, bool):
            if isinstance(value, str) and value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            else:
                raise ValueError(
                    f"Override {'.'.join(parts)} expects a boolean; use true or false, got {value!r}."
                )
        cursor[parts[-1]] = value
    return config


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def config_args_to_cli(args: dict[str, Any]) -> list[str]:
    cli: list[str] = []
    for key, value in args.items():
        if value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            cli.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, list):
            cli.append(flag)
            cli.extend(str(item) for item in value)
        else:
            cli.extend([flag, str(value)])
    return cli


def build_command(config: dict[str, Any], extra_args: list[str] | None = None) -> list[str]:
    trainer = Path(str(config.get("trainer", "examples/simple_trainer_meshgs.py")))
    if not trainer.is_absolute():
        trainer = REPO_ROOT / trainer
    python = str(config.get("python", sys.executable))
    trainer_args = config.get("args", {})
    if not isinstance(trainer_args, dict):
        raise ValueError("Config key `args` must be a JSON object.")
    command = [python, str(trainer), *config_args_to_cli(trainer_args)]
    if extra_args:
        command.extend(extra_args)
    return command


def build_env(config: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    pythonpath_parts = [str(REPO_ROOT), str(REPO_ROOT / "examples")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    for key, value in config.get("env", {}).items():
        env[str(key)] = str(value)
    return env


def write_resolved_run_record(config: dict[str, Any], command: list[str]) -> Path | None:
    args = config.get("args", {})
    result_dir = args.get("result_dir") if isinstance(args, dict) else None
    if not result_dir:
        return None
    out_dir = Path(str(result_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resolved_run_config.json"
    payload = {
        "config": config,
        "command": command,
        "commandString": shlex.join(command),
    }
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path


def build_post_train_commands(config: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Build local post-training utilities requested by the JSON config.

    These commands are intentionally top-level config options rather than
    trainer args, so they do not leak into OMeGa's CLI. The first use is loss
    plotting from ``stats/train_loss.jsonl`` after a training run completes.
    """

    post_train = config.get("post_train", {})
    if not isinstance(post_train, dict):
        return []
    if not bool(post_train.get("plot_losses", False)):
        return []

    args = config.get("args", {})
    result_dir = args.get("result_dir") if isinstance(args, dict) else None
    if not result_dir:
        raise SystemExit("post_train.plot_losses requires args.result_dir in the config.")

    command = [
        str(config.get("python", sys.executable)),
        str(REPO_ROOT / "scripts" / "plot_training_losses.py"),
        str(result_dir),
        "--smooth-window",
        str(int(post_train.get("plot_losses_smooth_window", post_train.get("smooth_window", 5)))),
    ]
    if post_train.get("plot_losses_out_dir"):
        command.extend(["--out-dir", str(post_train["plot_losses_out_dir"])])
    if bool(post_train.get("plot_losses_include_zero", False)):
        command.append("--include-zero")
    if post_train.get("plot_losses_dpi") is not None:
        command.extend(["--dpi", str(int(post_train["plot_losses_dpi"]))])
    return [("Plot training losses", command)]


def run_post_train_commands(config: dict[str, Any], *, env: dict[str, str]) -> int:
    commands = build_post_train_commands(config)
    for label, command in commands:
        print(f"=== Post Train: {label} ===")
        print(shlex.join(command))
        code = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode
        if code != 0:
            print(f"[warning] Post-training task failed with code {code}: {label}")
            return code
    return 0


def validate_config(config: dict[str, Any], *, dry_run: bool) -> None:
    unresolved = _find_unresolved_env(config)
    if unresolved:
        preview = "\n".join(f"  - {item}" for item in unresolved[:8])
        suffix = "" if len(unresolved) <= 8 else f"\n  ... and {len(unresolved) - 8} more"
        raise SystemExit(
            "Config still contains unresolved environment variables. "
            "Export variables such as PACKAGE_ROOT, or override values with --set.\n"
            f"{preview}{suffix}"
        )

    if dry_run:
        return

    args = config.get("args", {})
    if not isinstance(args, dict):
        return
    for key, label in (("data_dir", "OMeGa dataset"), ("path_to_mesh", "initial mesh")):
        value = args.get(key)
        if value and not Path(str(value)).exists():
            raise SystemExit(
                f"Missing {label}: {value}\n"
                "Run scan_processing/23_run_phone_omega.py export first, or override this path with --set."
            )
    resume_path = args.get("resume_from_checkpoint")
    if resume_path and not Path(str(resume_path)).exists():
        raise SystemExit(
            f"Missing resume checkpoint: {resume_path}\n"
            "Run the warmup config first, or override resume_from_checkpoint with --set."
        )
    if bool(args.get("building_depth_regularization_on", False)):
        guide_dir = args.get("building_depth_guide_dir")
        if not guide_dir or not Path(str(guide_dir)).exists():
            raise SystemExit(
                f"Missing Guide01 depth regularization directory: {guide_dir}\n"
                "Run scan_processing/Guide01_compute_phone_promptda_planarity.py first, "
                "or override building_depth_guide_dir with --set."
            )
    if bool(args.get("building_coplane_regularization_on", False)):
        if int(args.get("building_coplane_min_faces", 1)) < 3:
            raise SystemExit("building_coplane_min_faces must be at least 3.")
        if float(args.get("building_coplane_plane_distance_m", 0.0)) <= 0.0:
            raise SystemExit("building_coplane_plane_distance_m must be positive.")
        if float(args.get("building_coplane_normal_angle_deg", 0.0)) <= 0.0:
            raise SystemExit("building_coplane_normal_angle_deg must be positive.")
        if bool(args.get("building_coplane_depth_assist_on", False)) and not bool(args.get("building_depth_regularization_on", False)):
            raise SystemExit("building_coplane_depth_assist_on requires building_depth_regularization_on so Guide01 maps are loaded.")
        if int(args.get("building_coplane_depth_sample_stride", 1)) < 1:
            raise SystemExit("building_coplane_depth_sample_stride must be at least 1.")
        if int(args.get("building_coplane_depth_tile_size_px", 1)) < 1:
            raise SystemExit("building_coplane_depth_tile_size_px must be positive.")
        if float(args.get("building_coplane_depth_plane_distance_m", 0.0)) <= 0.0:
            raise SystemExit("building_coplane_depth_plane_distance_m must be positive.")


def run_config(
    config_path: Path,
    *,
    expected_kind: str,
    assignments: list[str],
    extra_args: list[str],
    dry_run: bool,
) -> int:
    config = apply_overrides(load_config(config_path), assignments)
    kind = str(config.get("kind", "baseline"))
    if kind != expected_kind:
        raise SystemExit(f"Config kind is {kind!r}, but this runner expects {expected_kind!r}.")
    validate_config(config, dry_run=dry_run)
    command = build_command(config, extra_args)
    record_path = None if dry_run else write_resolved_run_record(config, command)
    print("=== OMeGa Local Runner ===")
    print(f"Kind: {kind}")
    print(f"Config: {config['_config_path']}")
    if record_path is not None:
        print(f"Resolved config: {record_path}")
    post_train_commands = build_post_train_commands(config)
    print("Command:")
    print(shlex.join(command))
    if post_train_commands:
        print("Post-train commands:")
        for label, post_command in post_train_commands:
            print(f"  # {label}")
            print("  " + shlex.join(post_command))
    if dry_run:
        return 0

    env = build_env(config)
    train_code = subprocess.run(command, cwd=REPO_ROOT / "examples", env=env, check=False).returncode
    if train_code != 0:
        print(f"[warning] Training exited with code {train_code}; skipping post-training tasks.")
        return train_code
    post_code = run_post_train_commands(config, env=env)
    return post_code if post_code != 0 else train_code


def main(expected_kind: str) -> int:
    parser = argparse.ArgumentParser(description=f"Run a local OMeGa {expected_kind} config.")
    parser.add_argument("--config", required=True, type=Path, help="JSON config file.")
    parser.add_argument("--set", dest="assignments", action="append", default=[], help="Override config values, e.g. --set max_steps=1000 or --set args.result_dir=/tmp/run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without launching training.")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]
    try:
        return run_config(
            args.config,
            expected_kind=expected_kind,
            assignments=list(args.assignments),
            extra_args=extra_args,
            dry_run=bool(args.dry_run),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
