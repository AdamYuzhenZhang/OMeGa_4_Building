from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from plyfile import PlyData

from omega_local.reconstruction.backends.objectgs.compat.torch_scatter import (
    scatter_max,
)
from omega_local.reconstruction.backends.objectgs.contract import (
    ObjectGSConfig,
    ObjectGSPaths,
)
from omega_local.reconstruction.backends.objectgs.mesh import (
    export_object_meshes,
)
from omega_local.reconstruction.backends.objectgs.prepare import (
    _remap_labels,
    _write_labeled_initializer,
    _write_training_config,
)
from omega_local.segmentation.interactive.reconstruction_runs import (
    discover_objectgs_runs,
)
from omega_local.segmentation.interactive.segmentation3d_manager import (
    Segmentation3DManager,
)


def _config(tmp_path: Path, **changes) -> ObjectGSConfig:
    values = {
        "model_dir": tmp_path / "model",
        "editor_baseline_name": "editor",
        "objectgs_root": tmp_path / "ObjectGS",
        "python": Path(sys.executable),
    }
    values.update(changes)
    return ObjectGSConfig(**values).normalized()


def _paths(tmp_path: Path) -> ObjectGSPaths:
    return ObjectGSPaths(
        source_run_dir=tmp_path / "mapanything",
        run_dir=tmp_path / "mapanything" / "04_shared_objectgs" / "runs" / "joint",
    )


def test_objectgs_outputs_are_parallel_to_per_region_splat(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)

    assert paths.source_run_dir / "03_splat" not in paths.run_dir.parents
    assert "04_shared_objectgs" in paths.run_dir.parts
    assert paths.predicted_label_maps.parent.name == "predicted_masks"


def test_objectgs_mesh_defaults_use_metric_bounded_geometry(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    assert config.mesh_voxel_size == 0.01
    assert config.mesh_resolution == 512
    assert config.mesh_max_triangles == 1_000_000


def test_objectgs_mesh_command_carries_resource_bounds(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths = _paths(tmp_path)
    model = tmp_path / "trained"
    model.mkdir()
    paths.model_pointer.parent.mkdir(parents=True)
    paths.model_pointer.write_text(
        json.dumps({"modelDir": str(model)}),
        encoding="utf-8",
    )

    result = export_object_meshes(
        config,
        paths,
        object(),
        dry_run=True,
    )

    command = result["command"]
    assert command[command.index("--mesh-voxel-size") + 1] == "0.01"
    assert command[command.index("--mesh-max-triangles") + 1] == "1000000"


def test_objectgs_label_remap_reserves_zero() -> None:
    source = np.asarray([[0, 12], [30, 12]], dtype=np.uint16)
    output = _remap_labels(source, {12: 1, 30: 2})

    assert output.dtype == np.uint8
    assert output.tolist() == [[0, 1], [2, 1]]


def test_initializer_marks_geometry_completed_points_unknown(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "points.npz"
    np.savez_compressed(
        cache,
        points=np.asarray(
            [[0, 0, 1], [1, 0, 1], [2, 0, 1]],
            dtype=np.float32,
        ),
        labels=np.asarray([7, 7, 9], dtype=np.int32),
        source_colors=np.asarray(
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            dtype=np.uint8,
        ),
    )
    provenance = tmp_path / "provenance.npy"
    np.save(provenance, np.asarray([2, 3, 1], dtype=np.uint8))
    output = tmp_path / "points3D.ply"

    summary = _write_labeled_initializer(
        cache,
        provenance,
        output,
        {7: 1, 9: 2},
        stride=1,
        geometry_completed_as_unknown=True,
    )
    labels = np.asarray(PlyData.read(output)["vertex"]["label"])

    assert labels.tolist() == [1, 0, 2]
    assert summary["sourcePointCount"] == 3
    assert summary["geometryCompletedAsUnknownCount"] == 1


def test_generated_config_uses_released_joint_object_loss(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        iterations=30_000,
        semantic_loss_weight=0.1,
        initializer_stride=2,
    )
    paths = _paths(tmp_path)
    template = (
        config.objectgs_root
        / "config"
        / "objectgs"
        / "3d"
        / "3dovs"
        / "config.yaml"
    )
    template.parent.mkdir(parents=True)
    template.write_text(
        yaml.safe_dump(
            {
                "model_params": {
                    "model_config": {
                        "name": "GaussianModel",
                        "kwargs": {"voxel_size": 0.001},
                    }
                },
                "pipeline_params": {},
                "optim_params": {
                    "iterations": 1,
                    "offset_lr_max_steps": 1,
                    "mlp_opacity_lr_max_steps": 1,
                    "mlp_color_lr_max_steps": 1,
                    "position_lr_max_steps": 1,
                    "update_until": 1,
                    "lambda_object_loss": 0.01,
                },
            }
        ),
        encoding="utf-8",
    )

    _write_training_config(config, paths)
    payload = yaml.safe_load(paths.train_config.read_text(encoding="utf-8"))

    assert payload["model_params"]["source_path"] == str(paths.dataset_dir)
    assert payload["model_params"]["ratio"] == 1
    assert payload["optim_params"]["iterations"] == 30_000
    assert payload["optim_params"]["lambda_object_loss"] == 0.1


def test_torch_scatter_compatibility_returns_maximum() -> None:
    source = torch.tensor([[1.0], [3.0], [2.0]])
    index = torch.tensor([[0], [0], [1]])

    reduced, argmax = scatter_max(source, index, dim=0)

    assert reduced.tolist() == [[3.0], [2.0]]
    assert argmax is None


def test_objectgs_progress_is_discovered_beside_mapanything(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    run = (
        interactive
        / "experiments"
        / "mapanything_region_3dgs"
        / "runs"
        / "map_run"
        / "04_shared_objectgs"
        / "runs"
        / "joint"
    )
    (run / "01_dataset").mkdir(parents=True)
    (run / "run.json").write_text(
        json.dumps(
            {
                "runId": "joint",
                "mapanythingRunId": "map_run",
                "iterations": 30_000,
            }
        ),
        encoding="utf-8",
    )
    (run / "01_dataset" / "stage.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    (run / "progress.json").write_text(
        json.dumps(
            {
                "status": "running",
                "message": "Training progress: 12000/30000",
            }
        ),
        encoding="utf-8",
    )

    rows = discover_objectgs_runs(interactive)

    assert len(rows) == 1
    assert rows[0]["mapanythingRunId"] == "map_run"
    assert rows[0]["status"] == "running"
    assert rows[0]["currentIteration"] == 12_000
    assert rows[0]["stages"][1]["status"] == "running"

    for stage in ("03_model", "04_outputs"):
        (run / stage).mkdir()
        (run / stage / "stage.json").write_text(
            json.dumps({"status": "complete"}),
            encoding="utf-8",
        )
    (run / "progress.json").write_text(
        json.dumps({"status": "complete", "message": "Rendered 214/214"}),
        encoding="utf-8",
    )

    completed = discover_objectgs_runs(interactive)[0]

    assert completed["status"] == "paused"
    assert completed["currentIteration"] == 30_000
    assert completed["overallProgress"] == 75.0
    assert completed["stages"][3]["status"] == "pending"

    (run / "05_meshes").mkdir()
    (run / "05_meshes" / "stage.json").write_text(
        json.dumps({"status": "complete"}),
        encoding="utf-8",
    )
    completed = discover_objectgs_runs(interactive)[0]

    assert completed["status"] == "complete"
    assert completed["overallProgress"] == 100.0


def test_objectgs_anchor_payload_and_rgb_render_are_dataset_scoped(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    root = interactive / "segmentation3d"
    run_id = "map_run_joint_anchors"
    experiment = root / "runs" / run_id
    canonical = interactive / "experiments" / "objectgs"
    rgb = canonical / "04_outputs" / "rgb_renders"
    frame_map = canonical / "01_dataset" / "frame_map.jsonl"
    experiment.mkdir(parents=True)
    rgb.mkdir(parents=True)
    frame_map.parent.mkdir(parents=True)
    points = experiment / "anchors.npz"
    np.savez_compressed(
        points,
        points=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        labels=np.asarray([3, 7], dtype=np.uint16),
    )
    frame_map.write_text(
        json.dumps({"frameId": 42}) + "\n",
        encoding="utf-8",
    )
    (rgb / "00000.png").write_bytes(b"png")
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "experimentFamily": "mapanything_objectgs",
                "canonicalRunDir": str(canonical),
                "pointsCachePath": str(points),
            }
        ),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(
        root,
        object(),
        max_points=10,
        seed=1,
    )

    payload = manager.result_points(run_id)

    assert payload["labels"] == [3, 7]
    assert [row["id"] for row in payload["regions"]] == [3, 7]
    assert manager.objectgs_rgb_render(run_id, 42) == rgb / "00000.png"


def test_objectgs_mesh_scene_exposes_independent_named_parts(
    tmp_path: Path,
) -> None:
    interactive = tmp_path / "interactive"
    root = interactive / "segmentation3d"
    run_id = "map_run_joint_anchors"
    experiment = root / "runs" / run_id
    meshes = interactive / "experiments" / "objectgs" / "05_meshes"
    region_dir = meshes / "objects" / "region_000003"
    region_dir.mkdir(parents=True)
    experiment.mkdir(parents=True)
    glb = region_dir / "mesh.glb"
    glb.write_bytes(b"glTF")
    manifest = meshes / "mesh_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "persistentRegionId": 3,
                        "name": "Door",
                        "glb": str(glb),
                        "vertexCount": 123,
                        "triangleCount": 234,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    regions = interactive / "regions"
    regions.mkdir()
    (regions / "regions.json").write_text(
        json.dumps(
            {
                "regions": [
                    {"id": 3, "name": "Entry Door", "color": "#123456"}
                ]
            }
        ),
        encoding="utf-8",
    )
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "runId": run_id,
                "experimentFamily": "mapanything_objectgs",
                "meshArtifacts": {
                    "object_meshes": {
                        "displayName": "Official Object Meshes",
                        "manifestPath": str(manifest),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manager = Segmentation3DManager(
        root,
        object(),
        max_points=10,
        seed=1,
    )

    scene = manager.mesh_scene(run_id, "object_meshes")

    assert scene["partCount"] == 1
    assert scene["triangleCount"] == 234
    assert scene["parts"][0]["partId"] == "region_000003"
    assert scene["parts"][0]["regionName"] == "Entry Door"
    assert scene["parts"][0]["regionColor"] == [18, 52, 86]
    assert manager.mesh_artifact(
        run_id,
        "object_meshes",
        "region_000003",
    ) == glb
