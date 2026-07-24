from pathlib import Path

import numpy as np

from omega_local.segmentation.interactive.omega_mesh_point_cloud import (
    OmegaMeshHybridPointCloudConfig,
    OmegaMeshPointCloudConfig,
    OmegaMeshSurfacePointCloudConfig,
    ensure_omega_mesh_hybrid_point_cloud,
    ensure_omega_mesh_point_cloud,
    ensure_omega_mesh_surface_point_cloud,
    omega_mesh_point_cloud_status,
    omega_mesh_surface_point_cloud_status,
)


def test_extracts_every_finite_mesh_vertex_without_surface_resampling(tmp_path: Path) -> None:
    import open3d as o3d

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)

    config = OmegaMeshPointCloudConfig(
        mesh_path=mesh_path,
        points_path=tmp_path / "points.ply",
        cache_path=tmp_path / "points.npz",
        summary_path=tmp_path / "points.json",
    )
    summary = ensure_omega_mesh_point_cloud(config)

    assert summary["sourceVertexCount"] == 4
    assert summary["sourceTriangleCount"] == 2
    assert summary["outputPointCount"] == 4
    assert summary["preservesEveryFiniteMeshVertex"] is True
    assert summary["surfaceResampling"] is False
    with np.load(config.cache_path) as payload:
        np.testing.assert_allclose(payload["positions"], vertices.astype(np.float32))
        np.testing.assert_array_equal(payload["colors"], np.full((4, 3), 255, dtype=np.uint8))
        np.testing.assert_array_equal(payload["source_indices"], np.arange(4, dtype=np.int64))

    status = omega_mesh_point_cloud_status(config)
    assert status["ready"] is True
    assert status["pointCount"] == 4


def test_area_sampled_voxel_cloud_is_uniform_and_stays_on_the_surface(tmp_path: Path) -> None:
    import open3d as o3d

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)

    config = OmegaMeshSurfacePointCloudConfig(
        mesh_path=mesh_path,
        cache_path=tmp_path / "surface.npz",
        summary_path=tmp_path / "surface.json",
        target_point_count=100,
        oversample_factor=8,
        seed=17,
    )
    summary = ensure_omega_mesh_surface_point_cloud(config)

    assert 80 <= summary["outputPointCount"] <= 120
    assert summary["surfaceResampling"] is True
    assert summary["voxelUniformization"] is True
    assert summary["componentFiltering"] is False
    assert summary["sourceIndexMeaning"] == "source triangle index"
    with np.load(config.cache_path) as payload:
        points = payload["positions"]
        assert points.shape[0] == summary["outputPointCount"]
        np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-6)
        assert np.all((points[:, :2] >= 0.0) & (points[:, :2] <= 1.0))
        assert np.all(np.isin(payload["source_indices"], [0, 1]))

    status = omega_mesh_surface_point_cloud_status(config)
    assert status["ready"] is True
    assert status["pointCount"] == summary["outputPointCount"]


def test_hybrid_keeps_clean_vertices_and_fills_unoccupied_surface_voxels(tmp_path: Path) -> None:
    import open3d as o3d

    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [10.0, 10.0, 10.0],  # unreferenced floating vertex
            [20.0, 20.0, 20.0],
            [20.0001, 20.0, 20.0],
            [20.0, 20.0001, 20.0],
        ],
        dtype=np.float64,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3], [5, 6, 7]], dtype=np.int32)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh_path = tmp_path / "mesh.ply"
    assert o3d.io.write_triangle_mesh(str(mesh_path), mesh, write_ascii=False)

    config = OmegaMeshHybridPointCloudConfig(
        mesh_path=mesh_path,
        points_path=tmp_path / "hybrid.ply",
        cache_path=tmp_path / "hybrid.npz",
        summary_path=tmp_path / "hybrid.json",
        surface_cache_path=tmp_path / "surface.npz",
        surface_summary_path=tmp_path / "surface.json",
        surface_target_point_count=100,
        surface_oversample_factor=8,
        min_component_faces=2,
        min_component_area_fraction=1e-4,
        seed=17,
    )
    summary = ensure_omega_mesh_hybrid_point_cloud(config)

    assert summary["retainedMeshVertexCount"] == 4
    assert summary["removedComponentCount"] == 1
    assert summary["surfaceCompletionPointCount"] > 0
    assert summary["outputPointCount"] > 4
    with np.load(config.cache_path) as payload:
        positions = payload["positions"]
        point_kinds = payload["point_kinds"]
        np.testing.assert_allclose(positions[:4], vertices[:4].astype(np.float32))
        assert np.all(point_kinds[:4] == 0)
        assert np.all(point_kinds[4:] == 1)
        assert not np.any(np.all(np.isclose(positions, vertices[4]), axis=1))
