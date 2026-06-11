import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
try:
    from pycolmap import SceneManager
except ImportError:
    SceneManager = None
    import pycolmap
from plyfile import PlyData, PlyElement
from .normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)
import open3d as o3d
import pickle

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively get relative paths of files in a directory."""
    paths = []
    for dp, dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return paths


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_relative_path(base: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _resolve_guide_dir(data_dir: str, guide_dir: Optional[str]) -> Path:
    if not guide_dir:
        raise ValueError("building_depth_guide_dir is required when depth guides are enabled.")
    raw = Path(str(guide_dir)).expanduser()
    if raw.is_absolute():
        return raw
    data_path = Path(data_dir).resolve()
    candidates = [
        data_path / raw,
        data_path.parent / raw,
        data_path.parent.parent / raw,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_building_depth_guide_paths(data_dir: str, image_names: List[str], guide_dir: Optional[str]) -> List[str]:
    """Map OMeGa image names to Guide01 ``*.npz`` files.

    Guide01 is generated in the scan-processing package folder, while OMeGa
    trains from ``omega/dataset``.  The dataset manifest stores the bridge:
    ``imageName -> scanID/frameID``.  Guide01's manifest stores
    ``scanID/frameID -> guidePath``.
    """

    data_path = Path(data_dir).resolve()
    dataset_manifest_path = data_path.parent / "dataset_manifest.json"
    if not dataset_manifest_path.exists():
        raise ValueError(f"Missing OMeGa dataset manifest for guide lookup: {dataset_manifest_path}")
    with dataset_manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    capture_root = Path(str(dataset_manifest.get("captureRoot", data_path.parent.parent))).resolve()
    frame_by_image: Dict[str, tuple[str, int]] = {}
    for row in dataset_manifest.get("frames", []):
        if not isinstance(row, dict):
            continue
        frame_by_image[str(row["imageName"])] = (str(row["scanID"]), int(row["frameID"]))

    guide_root = _resolve_guide_dir(data_dir, guide_dir)
    guide_manifest_path = guide_root / "guides_manifest.jsonl"
    if not guide_manifest_path.exists():
        raise ValueError(f"Missing Guide01 manifest: {guide_manifest_path}")
    guide_by_frame: Dict[tuple[str, int], Path] = {}
    for row in _read_jsonl(guide_manifest_path):
        key = (str(row["scanID"]), int(row["frameID"]))
        guide_by_frame[key] = _resolve_relative_path(capture_root, str(row["guidePath"]))

    paths: List[str] = []
    missing: List[str] = []
    for image_name in image_names:
        key = frame_by_image.get(image_name)
        guide_path = None if key is None else guide_by_frame.get(key)
        if guide_path is None or not guide_path.exists():
            missing.append(f"{image_name} -> {key} -> {guide_path}")
        else:
            paths.append(str(guide_path))
    if missing:
        preview = "\n".join(missing[:8])
        raise ValueError(f"Missing Guide01 depth guides for {len(missing)} images. First missing:\n{preview}")
    print(f"[Parser] Loaded depth regularization guides from {guide_root}")
    return paths


def _load_building_depth_guide_npz(path: str) -> Dict[str, np.ndarray]:
    guide = np.load(path)
    required = ["depth_m", "valid_mask", "p_local_planar", "p_detail", "p_ridge"]
    missing = [key for key in required if key not in guide]
    if missing:
        raise ValueError(f"Guide01 file {path} is missing keys: {missing}")
    depth = np.asarray(guide["depth_m"], dtype=np.float32)
    return {
        "depth": np.where(np.isfinite(depth), depth, 0.0).astype(np.float32, copy=False),
        "valid": np.asarray(guide["valid_mask"], dtype=np.uint8),
        "p_local_planar": np.asarray(guide["p_local_planar"], dtype=np.float32),
        "p_detail": np.asarray(guide["p_detail"], dtype=np.float32),
        "p_ridge": np.asarray(guide["p_ridge"], dtype=np.float32),
    }


def _resize_guide_maps(guide: Dict[str, np.ndarray], width: int, height: int) -> Dict[str, np.ndarray]:
    if guide["depth"].shape[:2] == (height, width):
        return guide
    return {
        "depth": cv2.resize(guide["depth"], (width, height), interpolation=cv2.INTER_LINEAR),
        "valid": cv2.resize(guide["valid"], (width, height), interpolation=cv2.INTER_NEAREST),
        "p_local_planar": cv2.resize(guide["p_local_planar"], (width, height), interpolation=cv2.INTER_LINEAR),
        "p_detail": cv2.resize(guide["p_detail"], (width, height), interpolation=cv2.INTER_LINEAR),
        "p_ridge": cv2.resize(guide["p_ridge"], (width, height), interpolation=cv2.INTER_LINEAR),
    }


def _remap_guide_maps(guide: Dict[str, np.ndarray], mapx: np.ndarray, mapy: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "depth": cv2.remap(guide["depth"], mapx, mapy, cv2.INTER_LINEAR),
        "valid": cv2.remap(guide["valid"], mapx, mapy, cv2.INTER_NEAREST),
        "p_local_planar": cv2.remap(guide["p_local_planar"], mapx, mapy, cv2.INTER_LINEAR),
        "p_detail": cv2.remap(guide["p_detail"], mapx, mapy, cv2.INTER_LINEAR),
        "p_ridge": cv2.remap(guide["p_ridge"], mapx, mapy, cv2.INTER_LINEAR),
    }


def _crop_guide_maps(guide: Dict[str, np.ndarray], x: int, y: int, w: int, h: int) -> Dict[str, np.ndarray]:
    return {key: value[y : y + h, x : x + w] for key, value in guide.items()}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    colors = (colors * 255).astype(np.uint8)
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)

    return positions, colors, normals

class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        vfm_init: bool = False,
        test_every: int = 8,
        load_normal_maps: bool = False,
        load_building_depth_guides: bool = False,
        building_depth_guide_dir: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.vfm_init = vfm_init
        self.test_every = test_every
        self.load_normal_maps = load_normal_maps
        self.load_building_depth_guides = load_building_depth_guides

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."


        if SceneManager is not None:
            manager = SceneManager(colmap_dir)
            manager.load_cameras()
            manager.load_images()
            manager.load_points3D()
            imdata = manager.images
            cameras = manager.cameras
            points = manager.points3D.astype(np.float32)
            points_err = manager.point3D_errors.astype(np.float32)
            points_rgb = manager.point3D_colors.astype(np.uint8)
        else:
            reconstruction = pycolmap.Reconstruction(colmap_dir)
            manager = None
            imdata = reconstruction.images
            cameras = reconstruction.cameras
            points3d_values = list(reconstruction.points3D.values())
            if points3d_values:
                points = np.stack([np.asarray(point.xyz, dtype=np.float32) for point in points3d_values], axis=0)
                points_err = np.asarray([float(point.error) for point in points3d_values], dtype=np.float32)
                points_rgb = np.stack([np.asarray(point.color, dtype=np.uint8) for point in points3d_values], axis=0)
            else:
                points = np.empty((0, 3), dtype=np.float32)
                points_err = np.empty((0,), dtype=np.float32)
                points_rgb = np.empty((0, 3), dtype=np.uint8)

        # Extract extrinsic matrices in world-to-camera format.
        w2c_mats = []
        camera_ids = []
        Ks_dict = dict()
        params_dict = dict()
        imsize_dict = dict()  # width, height
        bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            if SceneManager is not None:
                rot = im.R()
                trans = im.tvec.reshape(3, 1)
            else:
                cam_from_world = im.cam_from_world()
                rot = np.asarray(cam_from_world.rotation.matrix(), dtype=np.float64)
                trans = np.asarray(cam_from_world.translation, dtype=np.float64).reshape(3, 1)
            w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
            w2c_mats.append(w2c)

            # support different camera intrinsics
            camera_id = im.camera_id
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]
            if SceneManager is not None:
                fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
                type_ = cam.camera_type
            else:
                fx, fy = cam.focal_length_x, cam.focal_length_y
                cx, cy = cam.principal_point_x, cam.principal_point_y
                type_ = cam.model_name
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            if type_ == 0 or type_ == "SIMPLE_PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            elif type_ == 1 or type_ == "PINHOLE":
                params = np.empty(0, dtype=np.float32)
                camtype = "perspective"
            if type_ == 2 or type_ == "SIMPLE_RADIAL":
                k1 = cam.k1 if SceneManager is not None else cam.params[3]
                params = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 3 or type_ == "RADIAL":
                if SceneManager is not None:
                    k1, k2 = cam.k1, cam.k2
                else:
                    k1, k2 = cam.params[3], cam.params[4]
                params = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 4 or type_ == "OPENCV":
                if SceneManager is not None:
                    k1, k2, p1, p2 = cam.k1, cam.k2, cam.p1, cam.p2
                else:
                    k1, k2, p1, p2 = cam.params[4], cam.params[5], cam.params[6], cam.params[7]
                params = np.array([k1, k2, p1, p2], dtype=np.float32)
                camtype = "perspective"
            elif type_ == 5 or type_ == "OPENCV_FISHEYE":
                if SceneManager is not None:
                    k1, k2, k3, k4 = cam.k1, cam.k2, cam.k3, cam.k4
                else:
                    k1, k2, k3, k4 = cam.params[4], cam.params[5], cam.params[6], cam.params[7]
                params = np.array([k1, k2, k3, k4], dtype=np.float32)
                camtype = "fisheye"
            assert (
                camtype == "perspective"
            ), f"Only support perspective camera model, got {type_}"

            params_dict[camera_id] = params

            # image size
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)

        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras."
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if not (type_ == 0 or type_ == 1):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        image_names = [imdata[k].name for k in imdata]

        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load images.
        if factor > 1:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        colmap_image_dir = os.path.join(data_dir, "images")
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
        for d in [image_dir, colmap_image_dir]:
            if not os.path.exists(d):
                raise ValueError(f"Image folder {d} does not exist.")

        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]

        # Load Paths of normal maps
        normal_maps_names = [f.replace('.jpg','.png').replace('.JPG', '.png')  for f in image_names]
        normal_maps_dir = os.path.join(data_dir, "normal_maps")
        normal_maps_paths = [os.path.join(normal_maps_dir, d) for d in normal_maps_names] if load_normal_maps else None
        normal_masks_dir = os.path.join(data_dir, "normal_aux")
        normal_mask_paths = (
            [os.path.join(normal_masks_dir, os.path.splitext(os.path.basename(d))[0] + "_invalid.png") for d in image_names]
            if load_normal_maps
            else None
        )
        building_depth_guide_paths = (
            _load_building_depth_guide_paths(data_dir, image_names, building_depth_guide_dir)
            if load_building_depth_guides
            else None
        )

        # 3D points and {image_name -> [point_idx]}
        points_normal = None
        point_indices = dict()

        if vfm_init:
            print("Loading VFM point cloud for initialization...")
            vfm_dir = os.path.join(data_dir, "vfm_sparse/0/")
            positions, colors, normals = fetchPly(os.path.join(vfm_dir, "points3D.ply"))
            # replace COLMAP points with VFM points
            points = positions
            points_rgb = colors
            points_normal = normals

        # image_id_to_name = {v: k for k, v in manager.name_to_image_id.items()}
        # for point_id, data in manager.point3D_id_to_images.items():
        #     for image_id, _ in data:
        #         image_name = image_id_to_name[image_id]
        #         point_idx = manager.point3D_id_to_point3D_idx[point_id]
        #         point_indices.setdefault(image_name, []).append(point_idx)
        # point_indices = {
        #     k: np.array(v).astype(np.int32) for k, v in point_indices.items()
        # }

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principle_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.normal_maps_paths = normal_maps_paths
        self.normal_mask_paths = normal_mask_paths
        self.building_depth_guide_paths = building_depth_guide_paths
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.points_normal = points_normal  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        actual_height, actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = actual_height / colmap_height, actual_width / colmap_width
        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]
            K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                K, params, (width, height), 0
            )
            mapx, mapy = cv2.initUndistortRectifyMap(
                K, params, None, K_undist, (width, height), cv2.CV_32FC1
            )
            self.Ks_dict[camera_id] = K_undist
            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.roi_undist_dict[camera_id] = roi_undist

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)

        self.scene_scale = np.max(dists)


class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        indices = np.arange(len(self.parser.image_names))
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        normal_map = imageio.imread(self.parser.normal_maps_paths[index])[..., :3] if self.parser.load_normal_maps else None
        normal_map_valid = None
        if self.parser.load_normal_maps and self.parser.normal_mask_paths is not None:
            normal_mask_path = self.parser.normal_mask_paths[index]
            if os.path.exists(normal_mask_path):
                invalid = imageio.imread(normal_mask_path)
                if invalid.ndim == 3:
                    invalid = invalid[..., 0]
                normal_map_valid = invalid < 128
        building_depth_guide = (
            _load_building_depth_guide_npz(self.parser.building_depth_guide_paths[index])
            if self.parser.load_building_depth_guides
            else None
        )
        
        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        if building_depth_guide is not None:
            building_depth_guide = _resize_guide_maps(
                building_depth_guide,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
        
        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            image = image[y : y + h, x : x + w]
            if normal_map is not None:
                normal_map = cv2.remap(normal_map, mapx, mapy, cv2.INTER_LINEAR)
                normal_map = normal_map[y : y + h, x : x + w]
            if normal_map_valid is not None:
                normal_map_valid = cv2.remap(normal_map_valid.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST).astype(bool)
                normal_map_valid = normal_map_valid[y : y + h, x : x + w]
            if building_depth_guide is not None:
                building_depth_guide = _remap_guide_maps(building_depth_guide, mapx, mapy)
                building_depth_guide = _crop_guide_maps(building_depth_guide, x, y, w, h)

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if normal_map is not None:
                normal_map = normal_map[y : y + self.patch_size, x : x + self.patch_size]
            if normal_map_valid is not None:
                normal_map_valid = normal_map_valid[y : y + self.patch_size, x : x + self.patch_size]
            if building_depth_guide is not None:
                building_depth_guide = _crop_guide_maps(building_depth_guide, x, y, self.patch_size, self.patch_size)
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "image_name": self.parser.image_names[index].split('.')[0],
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
        }

        if normal_map is not None:
            data["normal_map"] = torch.from_numpy((normal_map / 255.0 - 0.5) * 2).float()
            if normal_map_valid is not None:
                data["normal_map_valid"] = torch.from_numpy(normal_map_valid.astype(bool))
        if building_depth_guide is not None:
            data["building_depth_prior"] = torch.from_numpy(building_depth_guide["depth"]).float()
            data["building_depth_valid"] = torch.from_numpy(building_depth_guide["valid"].astype(bool))
            data["building_p_local_planar"] = torch.from_numpy(np.clip(building_depth_guide["p_local_planar"], 0.0, 1.0)).float()
            data["building_p_detail"] = torch.from_numpy(np.clip(building_depth_guide["p_detail"], 0.0, 1.0)).float()
            data["building_p_ridge"] = torch.from_numpy(np.clip(building_depth_guide["p_ridge"], 0.0, 1.0)).float()

        return data


if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/360_v2/garden")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=True, test_every=8
    )
    dataset = Dataset(parser, split="train")
    print(f"Dataset: {len(dataset)} images.")

    writer = imageio.get_writer("results/points.mp4", fps=30)
    for data in tqdm.tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
