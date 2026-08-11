"""Run ObjectGS object-depth rendering and bounded TSDF fusion.

This worker intentionally lives beside the compatibility shims. The upstream
3D exporter reads 2DGS-only normal buffers and its documented ``-1`` all-object
mode is not dispatched by the released script. The geometry algorithm remains
the released one: render each fixed-label anchor subset, fuse RGB-D with its
Open3D camera conversion, then retain its largest components. RGB-D frames are
integrated as they are rendered to keep memory bounded. A final QEM cap makes
the resulting geometry practical to store and inspect in the editor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
import yaml
from tqdm import tqdm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objectgs-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--region-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mesh-voxel-size", type=float, default=0.01)
    parser.add_argument("--mesh-resolution", type=int, default=512)
    parser.add_argument("--mesh-clusters", type=int, default=10)
    parser.add_argument("--mesh-max-triangles", type=int, default=1_000_000)
    return parser


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _post_process(
    mesh: o3d.geometry.TriangleMesh,
    clusters: int,
    max_triangles: int,
) -> o3d.geometry.TriangleMesh:
    """Apply the released component rule in place, then bound viewer geometry."""
    output = mesh
    if not output.has_triangles():
        return output
    output.remove_duplicated_triangles()
    output.remove_degenerate_triangles()
    output.remove_unreferenced_vertices()
    labels, counts, _ = output.cluster_connected_triangles()
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0:
        return output
    keep = min(max(int(clusters), 1), int(counts.size))
    threshold = max(int(np.sort(counts)[-keep]), 50)
    output.remove_triangles_by_mask(counts[labels] < threshold)
    output.remove_unreferenced_vertices()
    output.remove_degenerate_triangles()
    if len(output.triangles) > int(max_triangles):
        output = output.simplify_quadric_decimation(
            target_number_of_triangles=int(max_triangles),
        )
        output.remove_degenerate_triangles()
        output.remove_duplicated_triangles()
        output.remove_unreferenced_vertices()
    return output


def _write_glb(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
    if colors.shape != vertices.shape:
        colors = np.full(vertices.shape, 0.72, dtype=np.float32)
    rgba = np.column_stack(
        [
            np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8),
            np.full(vertices.shape[0], 255, dtype=np.uint8),
        ]
    )
    converted = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_colors=rgba,
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    converted.export(path, file_type="glb")


def _mesh_row(mesh: o3d.geometry.TriangleMesh) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bounds = (
        [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]
        if vertices.size
        else [[], []]
    )
    return {
        "vertexCount": int(vertices.shape[0]),
        "triangleCount": int(np.asarray(mesh.triangles).shape[0]),
        "bounds": bounds,
    }


def _resume_rows(
    output_dir: Path,
    *,
    model_dir: Path,
    mesh_voxel_size: float,
    mesh_resolution: int,
    mesh_clusters: int,
    mesh_max_triangles: int,
) -> dict[int, dict]:
    candidates = (
        output_dir / "mesh_manifest.partial.json",
        output_dir / "mesh_manifest.json",
    )
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if (
        str(payload.get("modelDir") or "") != str(model_dir)
        or float(payload.get("meshVoxelSize", -1)) != float(mesh_voxel_size)
        or int(payload.get("meshResolution", -1)) != int(mesh_resolution)
        or int(payload.get("meshClusters", -1)) != int(mesh_clusters)
        or int(payload.get("meshMaxTriangles", -1)) != int(mesh_max_triangles)
    ):
        return {}
    completed = {}
    for row in payload.get("regions", []):
        if not isinstance(row, dict):
            continue
        try:
            region_id = int(row["persistentRegionId"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            int(row.get("triangleCount", 0)) > 0
            and int(row.get("triangleCount", 0)) <= int(mesh_max_triangles)
            and Path(str(row.get("ply") or "")).is_file()
            and Path(str(row.get("glb") or "")).is_file()
        ):
            completed[region_id] = row
    return completed


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    objectgs_root = args.objectgs_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    sys.path.insert(0, str(objectgs_root))

    from gaussian_renderer import render
    from scene import Scene
    from utils.general_utils import parse_cfg
    from utils.mesh_utils import GaussianExtractor, to_cam_open3d

    class StreamingDepthExtractor(GaussianExtractor):
        @torch.no_grad()
        def reconstruction(self, viewpoint_stack):
            self.clean()
            self.viewpoint_stack = viewpoint_stack
            self.estimate_bounding_sphere()

        @torch.no_grad()
        def extract_mesh_bounded(
            self,
            *,
            voxel_size,
            sdf_trunc,
            depth_trunc,
        ):
            print("Running streaming TSDF volume integration ...", flush=True)
            print(f"voxel_size: {voxel_size}", flush=True)
            print(f"sdf_trunc: {sdf_trunc}", flush=True)
            print(f"depth_trunc: {depth_trunc}", flush=True)
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=float(voxel_size),
                sdf_trunc=float(sdf_trunc),
                color_type=(
                    o3d.pipelines.integration.TSDFVolumeColorType.RGB8
                ),
            )
            camera_parameters = to_cam_open3d(self.viewpoint_stack)
            for camera, camera_o3d in tqdm(
                zip(self.viewpoint_stack, camera_parameters),
                total=len(self.viewpoint_stack),
                desc="ObjectGS render + TSDF integration",
            ):
                package = self.render(camera, self.gaussians)
                rgb = package["render"].detach().cpu()
                depth = package["render_depth"].detach().cpu()
                if camera.alpha_mask is not None:
                    depth[camera.alpha_mask.detach().cpu() < 0.5] = 0
                color_image = np.asarray(
                    np.clip(rgb.permute(1, 2, 0).numpy(), 0.0, 1.0) * 255,
                    order="C",
                    dtype=np.uint8,
                )
                depth_image = np.asarray(
                    depth.permute(1, 2, 0).numpy(),
                    order="C",
                )
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_image),
                    o3d.geometry.Image(depth_image),
                    depth_trunc=float(depth_trunc),
                    convert_rgb_to_intensity=False,
                    depth_scale=1.0,
                )
                volume.integrate(
                    rgbd,
                    intrinsic=camera_o3d.intrinsic,
                    extrinsic=camera_o3d.extrinsic,
                )
                del package, rgb, depth, rgbd
            return volume.extract_triangle_mesh()

    config_path = model_dir / "config.yaml"
    with config_path.open("r", encoding="utf-8") as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    model_params, _, pipeline = parse_cfg(cfg)
    model_params.model_path = str(model_dir)
    model_config = model_params.model_config
    gaussians = getattr(__import__("scene"), model_config["name"])(
        **model_config["kwargs"]
    )
    scene = Scene(
        model_params,
        gaussians,
        load_iteration=-1,
        shuffle=False,
    )
    gaussians.eval()
    if getattr(gaussians, "active_sh_degree", None) is not None:
        gaussians.active_sh_degree = 0
    cameras = scene.getTrainCameras()
    region_payload = json.loads(args.region_map.read_text(encoding="utf-8"))
    regions = sorted(
        region_payload.get("regions", []),
        key=lambda row: int(row["persistentRegionId"]),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = _resume_rows(
        output_dir,
        model_dir=model_dir,
        mesh_voxel_size=float(args.mesh_voxel_size),
        mesh_resolution=int(args.mesh_resolution),
        mesh_clusters=int(args.mesh_clusters),
        mesh_max_triangles=int(args.mesh_max_triangles),
    )
    rows = []
    total = len(regions)
    for index, region in enumerate(regions):
        persistent_id = int(region["persistentRegionId"])
        objectgs_id = int(region["objectgsLabelId"])
        name = str(region.get("name") or f"Region {persistent_id}")
        cached = completed_rows.get(persistent_id)
        if cached is not None:
            rows.append(cached)
            print(
                f"[objectgs-mesh] region {index + 1}/{total}: "
                f"{persistent_id} ({name}) [cached]",
                flush=True,
            )
            continue
        print(
            f"[objectgs-mesh] region {index + 1}/{total}: "
            f"{persistent_id} ({name})",
            flush=True,
        )
        object_mask = gaussians.label_ids.squeeze() == objectgs_id
        region_dir = output_dir / "objects" / f"region_{persistent_id:06d}"
        region_dir.mkdir(parents=True, exist_ok=True)
        extractor = StreamingDepthExtractor(
            gaussians,
            render,
            pipeline,
            scene.background,
            object_mask,
        )
        extractor.reconstruction(cameras)
        depth_trunc = float(extractor.radius * 2.0 * 5.0)
        voxel_size = (
            float(args.mesh_voxel_size)
            if float(args.mesh_voxel_size) > 0
            else depth_trunc / float(args.mesh_resolution)
        )
        sdf_trunc = 5.0 * voxel_size
        mesh = extractor.extract_mesh_bounded(
            voxel_size=voxel_size,
            sdf_trunc=sdf_trunc,
            depth_trunc=depth_trunc,
        )
        raw_triangle_count = len(mesh.triangles)
        print(
            f"[objectgs-mesh] raw triangles: {raw_triangle_count:,}; "
            f"viewer cap: {int(args.mesh_max_triangles):,}",
            flush=True,
        )
        mesh = _post_process(
            mesh,
            args.mesh_clusters,
            args.mesh_max_triangles,
        )
        mesh.compute_vertex_normals()
        ply_path = region_dir / "mesh.ply"
        glb_path = region_dir / "mesh.glb"
        (region_dir / "mesh_raw.ply").unlink(missing_ok=True)
        glb_path.unlink(missing_ok=True)
        o3d.io.write_triangle_mesh(str(ply_path), mesh)
        if mesh.has_triangles():
            _write_glb(mesh, glb_path)
        rows.append(
            {
                "persistentRegionId": persistent_id,
                "objectgsLabelId": objectgs_id,
                "name": name,
                "rawPly": "",
                "ply": str(ply_path),
                "glb": str(glb_path) if glb_path.is_file() else "",
                "meshVoxelSize": float(args.mesh_voxel_size),
                "meshResolution": int(args.mesh_resolution),
                "meshMaxTriangles": int(args.mesh_max_triangles),
                "rawTriangleCount": int(raw_triangle_count),
                "voxelSize": voxel_size,
                "sdfTrunc": sdf_trunc,
                "depthTrunc": depth_trunc,
                **_mesh_row(mesh),
            }
        )
        del extractor, mesh
        torch.cuda.empty_cache()
        _atomic_json(
            output_dir / "mesh_manifest.partial.json",
            {
                "modelDir": str(model_dir),
                "meshVoxelSize": float(args.mesh_voxel_size),
                "meshResolution": int(args.mesh_resolution),
                "meshClusters": int(args.mesh_clusters),
                "meshMaxTriangles": int(args.mesh_max_triangles),
                "regions": rows,
                "completed": index + 1,
                "total": total,
            },
        )

    manifest = {
        "schemaVersion": 1,
        "method": "Released ObjectGS object render + bounded TSDF fusion",
        "modelDir": str(model_dir),
        "meshVoxelSize": float(args.mesh_voxel_size),
        "meshResolution": int(args.mesh_resolution),
        "meshClusters": int(args.mesh_clusters),
        "meshMaxTriangles": int(args.mesh_max_triangles),
        "cameraCount": len(cameras),
        "regions": rows,
    }
    _atomic_json(output_dir / "mesh_manifest.json", manifest)
    partial = output_dir / "mesh_manifest.partial.json"
    if partial.is_file():
        partial.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
