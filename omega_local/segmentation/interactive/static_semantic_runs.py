"""Discover completed joint static semantic 3DGS experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_METHODS = {
    "segment_then_splat": "Segment then Splat",
    "gaussian_grouping": "Gaussian Grouping",
}


def discover_static_semantic_3dgs_runs(
    interactive_dir: Path,
) -> list[dict[str, Any]]:
    """Return editor-facing rows for all completed joint static runs."""
    rows = [
        _status_row(payload)
        for payload in _experiment_payloads(interactive_dir)
    ]
    rows.sort(key=lambda row: str(row.get("timestampUtc") or ""), reverse=True)
    return rows


def live_static_semantic_3dgs_experiment(
    interactive_dir: Path,
    run_id: str,
) -> dict[str, Any] | None:
    """Resolve one discovered run to the artifact payload used by the viewer."""
    normalized = str(run_id).strip()
    return next(
        (
            payload
            for payload in _experiment_payloads(interactive_dir)
            if str(payload["runId"]) == normalized
        ),
        None,
    )


def _experiment_payloads(interactive_dir: Path) -> list[dict[str, Any]]:
    root = (
        interactive_dir
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
    )
    payloads: list[dict[str, Any]] = []
    for source_run in sorted(root.glob("*")):
        source_manifest = _read_json(source_run / "run.json")
        methods_root = source_run / "04_static_semantic_3dgs" / "runs"
        for method_id, display_name in _METHODS.items():
            for run_dir in sorted((methods_root / method_id).glob("*")):
                payload = _experiment_payload(
                    source_run,
                    source_manifest,
                    method_id,
                    display_name,
                    run_dir,
                )
                if payload is not None:
                    payloads.append(payload)
    payloads.extend(_paper_experiment_payloads(interactive_dir))
    return payloads


def _paper_experiment_payloads(
    interactive_dir: Path,
) -> list[dict[str, Any]]:
    root = (
        interactive_dir
        / "experiments"
        / "segment_then_splat_paper"
        / "runs"
    )
    payloads: list[dict[str, Any]] = []
    for run_dir in sorted(root.glob("*")):
        stage = _read_json(run_dir / "05_outputs" / "stage.json")
        manifest = _read_json(run_dir / "05_outputs" / "manifest.json")
        if stage.get("status") != "complete" or not manifest:
            continue
        scene_path = Path(str(manifest.get("scenePly") or ""))
        gaussian_count = int(manifest.get("gaussianCount", 0))
        if not scene_path.is_file() or gaussian_count <= 0:
            continue
        for level, level_row in dict(manifest.get("levels") or {}).items():
            payload = _paper_level_payload(
                run_dir,
                manifest,
                str(level),
                level_row,
            )
            if payload is not None:
                payloads.append(payload)
    return payloads


def _paper_level_payload(
    run_dir: Path,
    manifest: dict[str, Any],
    level: str,
    level_row: Any,
) -> dict[str, Any] | None:
    if not isinstance(level_row, dict):
        return None
    scene_path = Path(str(manifest["scenePly"]))
    gaussian_count = int(manifest["gaussianCount"])
    artifacts: dict[str, dict[str, Any]] = {
        "scene": {
            "displayName": f"Paper {level.title()} Scene",
            "path": str(scene_path),
            "format": "3dgs_ply",
            "pointCount": gaussian_count,
            "instanceId": 0,
            "artifactGroup": "composed_scene",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    }
    partition_count = 0
    for row in level_row.get("objects", []):
        if not isinstance(row, dict):
            continue
        path = Path(str(row.get("ply") or ""))
        count = int(row.get("gaussianCount", 0))
        if not path.is_file() or count <= 0:
            continue
        region_id = int(row.get("regionId", 0))
        partition_count += count
        artifacts[f"rgb_region_{region_id:06d}"] = {
            "displayName": str(row.get("name") or f"Object {region_id}"),
            "path": str(path),
            "format": "3dgs_ply",
            "pointCount": count,
            "instanceId": region_id,
            "artifactGroup": "rgb_objects",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    if partition_count != gaussian_count:
        return None
    run_name = run_dir.name
    assigned = int(level_row.get("assignedGaussianCount", 0))
    return {
        "schemaVersion": 1,
        "runId": f"static_segment_then_splat_paper_{run_name}_{level}",
        "methodId": "segment_then_splat_paper",
        "experimentFamily": "static_semantic_3dgs",
        "artifactRole": "reconstruction",
        "baseRunId": run_name,
        "reconstructionVariant": f"paper_{level}",
        "inputId": "official_autoseg_sam2",
        "displayName": f"Segment then Splat Paper · {level.title()}",
        "labelSpace": f"segment_then_splat_{level}_object",
        "geometrySource": "aligned_colmap_sparse",
        "maskMethodId": f"segment_then_splat_paper_{run_name}_{level}",
        "ready": True,
        "timestampUtc": str(manifest.get("timestampUtc") or ""),
        "canonicalRunDir": str(run_dir),
        "runDir": str(run_dir),
        "gaussianCount": gaussian_count,
        "foregroundGaussianCount": assigned,
        "backgroundGaussianCount": gaussian_count - assigned,
        "regionCount": int(level_row.get("objectCount", 0)),
        "gaussianArtifacts": artifacts,
    }


def _experiment_payload(
    source_run: Path,
    source_manifest: dict[str, Any],
    method_id: str,
    display_name: str,
    run_dir: Path,
) -> dict[str, Any] | None:
    output_dir = run_dir / "03_outputs"
    stage = _read_json(output_dir / "stage.json")
    manifest = _read_json(output_dir / "manifest.json")
    region_manifest = _read_json(output_dir / "regions.json")
    if stage.get("status") != "complete" or not manifest:
        return None

    scene_path = Path(str(manifest.get("scenePly") or ""))
    gaussian_count = int(manifest.get("gaussianCount", 0))
    if not scene_path.is_file() or gaussian_count <= 0:
        return None

    artifacts: dict[str, dict[str, Any]] = {
        "scene": {
            "displayName": f"{display_name} Joint Scene",
            "path": str(scene_path),
            "format": "3dgs_ply",
            "pointCount": gaussian_count,
            "instanceId": 0,
            "artifactGroup": "composed_scene",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }
    }
    region_count = 0
    for row in region_manifest.get("regions", []):
        if not isinstance(row, dict):
            continue
        point_count = int(row.get("gaussianCount", 0))
        path_value = str(row.get("ply") or "")
        path = Path(path_value) if path_value else None
        if point_count <= 0 or path is None or not path.is_file():
            continue
        region_id = int(row.get("persistentRegionId", 0))
        if region_id > 0:
            region_count += 1
        artifacts[f"rgb_region_{region_id:06d}"] = {
            "displayName": str(row.get("name") or f"Region {region_id}"),
            "path": str(path),
            "format": "3dgs_ply",
            "pointCount": point_count,
            "instanceId": region_id,
            "artifactGroup": "rgb_objects",
            "colorSpace": "rgb_sh",
            "radiometricAppearance": True,
        }

    partition_count = sum(
        int(row["pointCount"])
        for key, row in artifacts.items()
        if key != "scene"
    )
    if partition_count != gaussian_count:
        return None

    source_run_id = str(source_manifest.get("runId") or source_run.name)
    run_name = str(run_dir.name)
    run_id = f"static_{method_id}_{source_run_id}_{run_name}"
    return {
        "schemaVersion": 1,
        "runId": run_id,
        "methodId": method_id,
        "experimentFamily": "static_semantic_3dgs",
        "artifactRole": "reconstruction",
        "baseRunId": source_run_id,
        "reconstructionVariant": method_id,
        "inputId": str(source_manifest.get("propagationMethod") or ""),
        "displayName": display_name,
        "labelSpace": "persistent_region",
        "geometrySource": "omega_mapanything_initializer",
        "manualFrameWeight": int(source_manifest.get("manualFrameWeight", 1)),
        "ready": True,
        "timestampUtc": str(
            manifest.get("timestampUtc")
            or stage.get("timestampUtc")
            or ""
        ),
        "canonicalRunDir": str(run_dir),
        "runDir": str(run_dir),
        "gaussianCount": gaussian_count,
        "foregroundGaussianCount": int(
            manifest.get("foregroundGaussianCount", 0)
        ),
        "backgroundGaussianCount": int(
            manifest.get("backgroundGaussianCount", 0)
        ),
        "regionCount": region_count,
        "gaussianArtifacts": artifacts,
    }


def _status_row(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = []
    for variant_id, artifact in payload["gaussianArtifacts"].items():
        path = Path(str(artifact["path"]))
        artifacts.append(
            {
                "variantId": str(variant_id),
                "displayName": str(artifact["displayName"]),
                "format": str(artifact["format"]),
                "pointCount": int(artifact["pointCount"]),
                "labelCount": int(payload["regionCount"]),
                "labeledPointCount": int(payload["foregroundGaussianCount"]),
                "colorSpace": str(artifact["colorSpace"]),
                "unlabeledColor": "",
                "artifactGroup": str(artifact["artifactGroup"]),
                "instanceId": int(artifact["instanceId"]),
                "radiometricAppearance": bool(
                    artifact["radiometricAppearance"]
                ),
                "contentVersion": f"{path.stat().st_size:x}-{path.stat().st_mtime_ns:x}",
            }
        )
    return {
        key: value
        for key, value in payload.items()
        if key != "gaussianArtifacts"
    } | {"gaussianArtifacts": artifacts}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}
