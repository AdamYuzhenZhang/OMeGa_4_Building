from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from omega_local.reconstruction.backends.gaussian_flats.visualization import (
    ensure_visualization_artifacts,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_viewer_artifacts_publish_native_masks_and_meshes_only(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    run_dir = (
        interactive
        / "reconstruction"
        / "runs"
        / "gaussian_flats"
        / "door_planar_official"
    )
    _write_json(
        run_dir / "run.json",
        {
            "runId": "door_planar_official",
            "iterations": 30_000,
            "planarRegion": "Door",
            "propagationMethod": "sam2_video",
        },
    )
    _write_json(
        run_dir / "01_input" / "planes.json",
        {"regionId": 6, "regionName": "Door"},
    )
    plane_mask = run_dir / "01_input" / "plane_masks" / "0" / "frame.png"
    plane_mask.parent.mkdir(parents=True)
    Image.fromarray(np.asarray([[0, 255], [255, 0]], dtype=np.uint8)).save(
        plane_mask
    )
    (run_dir / "01_input" / "frames.jsonl").write_text(
        json.dumps(
            {
                "frameId": 7,
                "imageName": "frame.jpg",
                "planeMask": str(plane_mask),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "02_model").mkdir(parents=True, exist_ok=True)
    mesh = trimesh.Trimesh(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        process=False,
    )
    mesh.export(run_dir / "02_model" / "fuse_post.ply")
    mesh.export(run_dir / "02_model" / "planar_mesh.obj")

    result = ensure_visualization_artifacts(run_dir)

    mesh_artifact = result["meshArtifacts"]["mesh_comparison"]
    assert mesh_artifact["partCount"] == 2
    assert mesh_artifact["triangleCount"] == 2
    mesh_manifest = json.loads(
        Path(mesh_artifact["manifestPath"]).read_text(encoding="utf-8")
    )
    assert {row["persistentRegionId"] for row in mesh_manifest["regions"]} == {0, 6}
    assert "gaussianArtifacts" not in result
    plane_artifact = result["planeMaskArtifacts"]["input_plane_masks"]
    assert plane_artifact["frameCount"] == 1
    assert plane_artifact["nonemptyFrameCount"] == 1
    overlay = Image.open(Path(plane_artifact["overlayDir"]) / "000007.png")
    assert overlay.mode == "RGBA"
    assert np.count_nonzero(np.asarray(overlay)[..., 3]) == 2

    experiment = (
        interactive
        / "3d_segmentation"
        / "runs"
        / "gaussian_flats_door_planar_official"
        / "experiment.json"
    )
    assert experiment.is_file()
    payload = json.loads(experiment.read_text(encoding="utf-8"))
    assert payload["experimentFamily"] == "gaussian_flats"
    assert payload["labelSpace"] == "persistent_region"
    assert "gaussianArtifacts" not in payload
    assert "input_plane_masks" in payload["planeMaskArtifacts"]
