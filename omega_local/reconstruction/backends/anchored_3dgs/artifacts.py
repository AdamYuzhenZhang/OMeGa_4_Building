"""Editor metadata for anchored reconstruction artifacts."""

from __future__ import annotations

from typing import Any

from omega_local.segmentation.split_splat.contract import (
    atomic_write_json,
    read_json,
)

from .contract import AnchoredSplatConfig, AnchoredSplatPaths


def register_anchored_artifacts(
    config: AnchoredSplatConfig,
    paths: AnchoredSplatPaths,
    editor_paths: Any,
) -> dict[str, Any]:
    experiment_path = (
        editor_paths.segmentation3d_dir
        / "runs"
        / f"{paths.run_id}_splat"
        / "experiment.json"
    )
    if not experiment_path.is_file():
        raise FileNotFoundError(
            f"Composed anchored viewer metadata is missing: {experiment_path}"
        )
    payload = read_json(experiment_path)
    gaussian_artifacts = dict(payload.get("gaussianArtifacts") or {})
    updated_artifacts = {}
    for key, artifact in gaussian_artifacts.items():
        updated = {
            **artifact,
            "colorSpace": (
                "rgb_sh"
                if bool(artifact.get("radiometricAppearance"))
                else "persistent_region"
            ),
        }
        if key == "instance_ids":
            updated["displayName"] = "Anchored Persistent Regions"
        elif key.startswith("composed_object_"):
            updated["displayName"] = (
                f"Anchored Region {int(updated['instanceId'])}"
            )
        elif key.startswith("composed_rgb_object_"):
            updated["displayName"] = (
                f"Anchored RGB Region {int(updated['instanceId'])}"
            )
        elif key.startswith("rgb_object_"):
            updated["displayName"] = (
                f"Source RGB Region {int(updated['instanceId'])}"
            )
        updated_artifacts[key] = updated
    result = {
        **payload,
        "methodId": "anchored_3dgs",
        "experimentFamily": "split_splat",
        "artifactRole": "reconstruction",
        "baseRunId": config.source_run_id,
        "reconstructionVariant": "anchored_3dgs",
        "inputId": config.source_run_id,
        "displayName": f"Anchored 3DGS: {config.run_id}",
        "labelSpace": "persistent_region",
        "geometrySource": "anchored_persistent_region_composition",
        "maskRefinement": config.mask_refinement,
        "canonicalRunDir": str(paths.run_dir),
        "gaussianArtifacts": updated_artifacts,
    }
    atomic_write_json(experiment_path, result)
    return {
        "experimentPath": str(experiment_path),
        "methodId": result["methodId"],
        "displayName": result["displayName"],
        "labelSpace": result["labelSpace"],
    }
