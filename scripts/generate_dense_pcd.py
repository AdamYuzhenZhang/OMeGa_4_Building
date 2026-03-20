"""
Generate dense point cloud using VGGT for mesh initialization.
https://github.com/facebookresearch/vggt

Pipeline:
  1. Distribute all images into interleaved groups (strided sampling),
     so each group covers the entire scene uniformly
  2. Run VGGT point_map head on each group → direct 3D coordinates
  3. Per-group Sim(3) alignment from VGGT cameras to COLMAP cameras
  4. Merge aligned point clouds; voxel downsampling averages out
     inter-group inconsistencies
  5. Post-process: outlier removal, normal estimation

Usage:
    python scripts/generate_dense_pcd.py --data_dir /path/to/colmap_dataset

    # Control group size (default 60, adjust for VRAM)
    python scripts/generate_dense_pcd.py --data_dir /path/to/colmap_dataset --group_size 40

    # Fewer groups = faster; more groups = denser
    python scripts/generate_dense_pcd.py --data_dir /path/to/colmap_dataset --min_group_size 20

Prerequisites:
    1. Install VGGT: cd vggt && pip install -e .
       (see https://github.com/facebookresearch/vggt/blob/main/docs/package.md)
    2. COLMAP sparse reconstruction must exist at <data_dir>/sparse/0/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


VGGT_RESOLUTION = 518


def load_colmap_cameras(data_dir):
    """Load COLMAP cameras using the OMeGa Parser."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from examples.datasets.colmap import Parser

    parser = Parser(str(data_dir))
    return parser


def create_interleaved_groups(num_images, group_size, min_group_size=10):
    """Create interleaved groups that each cover the full scene uniformly.

    Example: 300 images, group_size=60 → 5 groups
        Group 0: [0, 5, 10, 15, ...]
        Group 1: [1, 6, 11, 16, ...]
        Group 2: [2, 7, 12, 17, ...]
        ...

    Args:
        num_images: total number of images
        group_size: max images per group (limited by VRAM)
        min_group_size: discard groups smaller than this

    Returns:
        list of index arrays, one per group
    """
    num_groups = max(1, (num_images + group_size - 1) // group_size)

    groups = []
    for g in range(num_groups):
        indices = np.arange(g, num_images, num_groups)
        if len(indices) >= min_group_size:
            groups.append(indices)

    return groups


def preprocess_images_for_vggt(image_paths, resolution=VGGT_RESOLUTION):
    """Load and preprocess images for VGGT inference.

    Returns:
        images: (S, 3, H, W) float32 tensor in [0, 1]
        valid_regions: list of (valid_h, valid_w) before padding
    """
    import torchvision.transforms.functional as TF

    tensors = []
    valid_regions = []

    for p in tqdm(image_paths, desc="Loading images", leave=False):
        img = Image.open(p).convert("RGB")
        orig_w, orig_h = img.size

        scale = resolution / max(orig_w, orig_h)
        new_w = int(round(orig_w * scale / 14)) * 14
        new_h = int(round(orig_h * scale / 14)) * 14

        valid_regions.append((new_h, new_w))

        img = img.resize((new_w, new_h), Image.BILINEAR)
        tensor = TF.to_tensor(img)
        tensors.append(tensor)

    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)

    padded = []
    for t in tensors:
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        if pad_h > 0 or pad_w > 0:
            t = F.pad(t, (0, pad_w, 0, pad_h), value=1.0)
        padded.append(t)

    images = torch.stack(padded, dim=0)
    return images, valid_regions


def compute_sim3_alignment(src_points, dst_points):
    """Compute Sim(3) transform from src to dst using Umeyama algorithm.

    Finds: dst = s * R @ src + t
    """
    assert src_points.shape == dst_points.shape
    n = src_points.shape[0]

    src_mean = src_points.mean(axis=0)
    dst_mean = dst_points.mean(axis=0)

    src_centered = src_points - src_mean
    dst_centered = dst_points - dst_mean

    src_var = np.sum(src_centered ** 2) / n

    H = (dst_centered.T @ src_centered) / n
    U, D, Vt = np.linalg.svd(H)

    d = np.linalg.det(U) * np.linalg.det(Vt)
    S = np.eye(3)
    if d < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / src_var
    t = dst_mean - s * R @ src_mean

    return s, R, t


def apply_sim3(points, s, R, t):
    """Apply Sim(3) transform: dst = s * R @ src + t"""
    return (s * (R @ points.T)).T + t


def process_group(model, images, valid_regions, group_indices,
                  all_image_paths, camtoworlds_colmap,
                  conf_threshold_pct, device, dtype):
    """Run VGGT on one group and return aligned points + colors.

    Returns:
        aligned_points: (N, 3) in COLMAP coords
        colors: (N, 3) float in [0, 1]
        sim3_scale: the scale factor used
    """
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Select images for this group
    group_images = images[group_indices]  # (S, 3, H, W)
    imgs_batch = group_images.unsqueeze(0).to(device)  # (1, S, 3, H, W)
    img_shape = imgs_batch.shape[-2:]

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(imgs_batch)

            # point_map: direct 3D coordinates, globally consistent within group
            point_map, point_conf = model.point_head(
                aggregated_tokens_list, imgs_batch, ps_idx
            )

            # Cameras for Sim(3) alignment
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            vggt_extrinsic, _ = pose_encoding_to_extri_intri(pose_enc, img_shape)

    # To numpy
    point_map = point_map.squeeze(0).cpu().numpy()      # (S, H, W, 3)
    point_conf = point_conf.squeeze(0).cpu().numpy()     # (S, H, W)
    vggt_extri = vggt_extrinsic.squeeze(0).cpu().numpy()  # (S, 3, 4) w2c

    # VGGT camera centers
    S = len(group_indices)
    vggt_w2c_4x4 = np.zeros((S, 4, 4))
    vggt_w2c_4x4[:, :3, :] = vggt_extri
    vggt_w2c_4x4[:, 3, 3] = 1.0
    vggt_c2w = np.linalg.inv(vggt_w2c_4x4)
    vggt_centers = vggt_c2w[:, :3, 3]

    # COLMAP camera centers
    colmap_centers = camtoworlds_colmap[group_indices, :3, 3]

    # Sim(3) alignment
    s, R, t = compute_sim3_alignment(vggt_centers, colmap_centers)

    # Extract and filter points
    imgs_np = group_images.permute(0, 2, 3, 1).numpy()  # (S, H, W, 3)

    all_pts = []
    all_cols = []

    for i in range(S):
        valid_h, valid_w = valid_regions[group_indices[i]]

        conf = point_conf[i, :valid_h, :valid_w]
        pts = point_map[i, :valid_h, :valid_w]
        colors = imgs_np[i, :valid_h, :valid_w]

        conf_flat = conf.reshape(-1)
        threshold_val = np.percentile(conf_flat, conf_threshold_pct)
        mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        pts_filtered = pts.reshape(-1, 3)[mask]
        colors_filtered = colors.reshape(-1, 3)[mask]

        valid = np.isfinite(pts_filtered).all(axis=1)
        if valid.sum() > 0:
            norms = np.linalg.norm(pts_filtered[valid], axis=1)
            norm_thresh = np.percentile(norms, 99.5)
            valid &= np.linalg.norm(pts_filtered, axis=1) < norm_thresh

        all_pts.append(pts_filtered[valid])
        all_cols.append(colors_filtered[valid])

    group_points = np.concatenate(all_pts, axis=0)
    group_colors = np.concatenate(all_cols, axis=0)

    # Apply Sim(3) to transform points to COLMAP coords
    aligned_points = apply_sim3(group_points, s, R, t)

    # Cleanup
    del aggregated_tokens_list, ps_idx, point_map, point_conf, pose_enc
    torch.cuda.empty_cache()

    return aligned_points, group_colors, s


def main():
    parser = argparse.ArgumentParser(description="Generate dense point cloud using VGGT")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COLMAP dataset")
    parser.add_argument("--group_size", type=int, default=60,
                        help="Max images per group (limited by VRAM)")
    parser.add_argument("--min_group_size", type=int, default=10,
                        help="Discard groups smaller than this")
    parser.add_argument("--conf_threshold_pct", type=float, default=30.0,
                        help="Percentile threshold for point confidence filtering (0-100)")
    parser.add_argument("--target_num_points", type=int, default=500000,
                        help="Target number of points after voxel downsampling (0 = no downsampling)")
    parser.add_argument("--estimate_normals", action="store_true", default=True,
                        help="Estimate normals for screened Poisson reconstruction")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=int, default=VGGT_RESOLUTION,
                        help="VGGT inference resolution (default: 518)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = data_dir / "vfm_sparse" / "0"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "points3D.ply"

    # Import VGGT
    try:
        from vggt.models.vggt import VGGT
    except ImportError:
        print("Error: VGGT is not installed.")
        print("Install from: https://github.com/facebookresearch/vggt")
        print("  git clone https://github.com/facebookresearch/vggt.git")
        print("  cd vggt && pip install -e .")
        sys.exit(1)

    # Load COLMAP data
    print("Loading COLMAP cameras...")
    colmap_parser = load_colmap_cameras(data_dir)
    all_image_paths = colmap_parser.image_paths
    camtoworlds_colmap = colmap_parser.camtoworlds  # (N, 4, 4), c2w

    num_total = len(all_image_paths)
    print(f"Found {num_total} images in COLMAP")

    # --- Step 1: Create interleaved groups ---
    groups = create_interleaved_groups(
        num_total, args.group_size, args.min_group_size
    )
    print(f"\nCreated {len(groups)} interleaved groups "
          f"(group_size={args.group_size}, stride={len(groups)})")
    for i, g in enumerate(groups):
        print(f"  Group {i}: {len(g)} images "
              f"(indices {g[0]}, {g[1] if len(g)>1 else '?'}, ..., {g[-1]})")

    # --- Step 2: Preprocess all images once ---
    print("\nPreprocessing all images...")
    all_images, all_valid_regions = preprocess_images_for_vggt(
        all_image_paths, resolution=args.resolution
    )

    # Load VGGT model
    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval().to(args.device)

    dtype = torch.bfloat16 if (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16

    # --- Step 3: Process each group ---
    all_points_list = []
    all_colors_list = []

    for group_idx, group_indices in enumerate(groups):
        print(f"\n--- Group {group_idx+1}/{len(groups)}: "
              f"{len(group_indices)} images ---")

        aligned_pts, colors, sim3_scale = process_group(
            model, all_images, all_valid_regions, group_indices,
            all_image_paths, camtoworlds_colmap,
            args.conf_threshold_pct, args.device, dtype
        )

        print(f"  {len(aligned_pts):,} points, Sim(3) scale={sim3_scale:.4f}")
        all_points_list.append(aligned_pts)
        all_colors_list.append(colors)

    # Merge all groups
    all_points = np.concatenate(all_points_list, axis=0)
    all_colors = np.concatenate(all_colors_list, axis=0)
    print(f"\nTotal: {len(all_points):,} points from {len(groups)} groups")

    # --- Step 4: Post-processing ---
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(
        all_colors if all_colors.dtype in (np.float64, np.float32)
        else all_colors.astype(np.float64) / 255.0
    )

    # Statistical outlier removal
    print("Removing statistical outliers...")
    n_before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"  Kept {len(pcd.points):,} / {n_before:,} points "
          f"({len(pcd.points)/n_before*100:.1f}%)")

    # Voxel downsampling (averages positions within each voxel,
    # naturally smoothing inter-group inconsistencies)
    if args.target_num_points > 0 and len(pcd.points) > args.target_num_points:
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_extent = np.max(bbox.get_extent())
        voxel_lo, voxel_hi = 1e-6, bbox_extent / 2.0

        print(f"Voxel downsampling to ~{args.target_num_points:,} points...")
        for _ in range(30):
            voxel_mid = (voxel_lo + voxel_hi) / 2.0
            pcd_down = pcd.voxel_down_sample(voxel_mid)
            n = len(pcd_down.points)
            if n > args.target_num_points:
                voxel_lo = voxel_mid
            else:
                voxel_hi = voxel_mid
            if abs(n - args.target_num_points) / args.target_num_points < 0.05:
                break

        pcd = pcd_down
        print(f"  Downsampled to {len(pcd.points):,} points "
              f"(voxel_size={voxel_mid:.6f})")

    # Estimate normals
    if args.estimate_normals:
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
        print("  Normals estimated and oriented.")

    # Save
    o3d.io.write_point_cloud(str(output_path), pcd)
    print(f"\nPoint cloud saved to {output_path} ({len(pcd.points):,} points)")


if __name__ == "__main__":
    main()
