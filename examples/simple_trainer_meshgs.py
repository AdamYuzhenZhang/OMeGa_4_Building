import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
import imageio
import open3d as o3d
import gc
import nerfview
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tyro
import viser
from datasets.colmap import (
    Dataset,
    Parser,
    _crop_guide_maps,
    _load_building_depth_guide_npz,
    _remap_guide_maps,
    _resize_guide_maps,
)
from datasets.traj import generate_interpolated_path
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from utils import (AppearanceOptModule, CameraOptModule,
                   rgb_to_sh, set_random_seed, colormap,
                   l1_loss, save_ply, merge_ply,
                   save_image, depth_gray2rgb, normal_rgb2sn, post_process_mesh, ssim,
                   init_colors_from_pc, rotation_matrix_to_quaternion, calculate_means, calculate_scales, calculate_Rs)
from collections import defaultdict

from mesh_opt import mesh_merge
from gsplat.distributed import cli
from gsplat.rendering import rasterization_2dgs, rasterization_2dgs_inria_wrapper
from gsplat.strategy import MeshGSStrategy
from gsplat.utils import normalized_quat_to_rotmat

from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing

import cv2

try:
    from omega_local.viz.mesh_training_preview import (
        resize_rgb_and_intrinsics_for_preview,
        update_mesh_training_contact_sheet,
        write_mesh_training_preview,
    )
except Exception:  # pragma: no cover - optional local diagnostics only.
    resize_rgb_and_intrinsics_for_preview = None
    update_mesh_training_contact_sheet = None
    write_mesh_training_preview = None

try:
    from omega_local.viz.regularization_preview import (
        update_regularization_contact_sheet,
        write_regularization_preview,
    )
except Exception:  # pragma: no cover - optional local diagnostics only.
    update_regularization_contact_sheet = None
    write_regularization_preview = None

try:
    from omega_local.losses.depth_regularization import compute_depth_regularization_loss
except Exception:  # pragma: no cover - optional local extension only.
    compute_depth_regularization_loss = None

try:
    from omega_local.losses.coplane_regularization import (
        build_coplane_targets,
        build_depth_plane_hypotheses,
        compute_coplane_regularization_loss,
        should_refresh_coplane_targets,
    )
except Exception:  # pragma: no cover - optional local extension only.
    build_coplane_targets = None
    build_depth_plane_hypotheses = None
    compute_coplane_regularization_loss = None
    should_refresh_coplane_targets = None


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt file. If provide, it will skip training and render a video
    ckpt: Optional[List[str]] = None
    # Local training resume checkpoint. Unlike --ckpt, this continues training
    # and restores mesh vertices/faces, splat parameters, and optimizer states.
    resume_from_checkpoint: Optional[str] = None

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 1
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000

    # Number of position lr steps
    pos_lr_steps: int = 30_000

    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.01
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.2
    # Far plane clipping distance
    far_plane: float = 200

    # GSs with opacity below this value will be pruned
    prune_opa: float = 0.05
    # GSs larger than this scene-scale fraction are pruned after the first opacity reset.
    prune_scale3d: float = 0.1

    grow_scale3d: float = 0.01

    grow_grad2d: float = 0.0002
    
    # Start refining GSs after this iteration
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration
    refine_stop_iter: int = 15_000
    # Reset opacities every this steps
    reset_every: int = 3000
    # Refine GSs every this steps
    refine_every: int = 100
    # Pause refining GSs until this number of steps after reset
    pause_refine_after_reset: int = 0

    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use absolute gradient for pruning. This typically requires larger --grow_grad2d, e.g., 0.0008 or 0.0006
    absgrad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False
    # Whether to use revised opacity heuristic from arXiv:2404.06109 (experimental)
    revised_opacity: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Load normal map
    normal_map_on: bool = False
    # Enable monocular normal loss (L_n)
    mono_normal_loss: bool = False
    # Weight for monocular normal loss
    mono_normal_lambda: float = 0.5
    # Iteration to start monocular normal regularization
    mono_normal_loss_start_iter: int = 3_000

    # Edge Loss
    edge_loss: bool = False
    # Weight for edge loss
    edge_lambda: float = 1e-2

    # Distortion loss. (experimental)
    dist_loss: bool = False
    # Weight for distortion loss
    dist_lambda: float = 1e-2
    # Iteration to start distortion loss regulerization
    dist_start_iter: int = 3_000

    # OMeGa-4-Building depth regularization. Disabled by default so baseline
    # configs stay comparable to OMeGa. When enabled, Guide01 depth priors add:
    #   L_depth = lambda_d * mean_p w(p) * rho(D_render(p) - D_prior(p)).
    building_depth_regularization_on: bool = False
    building_depth_guide_dir: str = ""
    building_depth_regularization_start_iter: int = 3_000
    building_depth_lambda: float = 0.05
    building_depth_huber_m: float = 0.05
    building_depth_min_m: float = 0.05
    building_depth_max_m: float = 15.0
    building_depth_min_alpha: float = 0.05
    building_depth_planar_weight: float = 1.0
    building_depth_detail_weight: float = 0.25
    building_depth_ridge_weight: float = 0.0
    building_depth_distance_decay_m: float = 0.0

    # OMeGa-4-Building coplane regularization. This is a mesh-space structural
    # prior: every refresh window, current faces that already agree in normal
    # and plane offset are grouped into stop-gradient target planes, then the
    # optimizer penalizes vertex-to-plane distance and face-normal deviation.
    building_coplane_regularization_on: bool = False
    building_coplane_start_iter: int = 3_000
    building_coplane_refresh_every: int = 500
    building_coplane_point_lambda: float = 0.05
    building_coplane_normal_lambda: float = 0.005
    building_coplane_huber_m: float = 0.02
    building_coplane_normal_angle_deg: float = 8.0
    building_coplane_plane_distance_m: float = 0.03
    building_coplane_min_faces: int = 24
    building_coplane_min_group_area_m2: float = 0.01
    building_coplane_min_face_area_m2: float = 1e-7
    building_coplane_max_groups: int = 4096

    # Optional depth-assisted coplane grouping. The depth guide proposes broad
    # planes from locally planar pixels; mesh faces still need to pass current
    # distance/normal checks before joining those larger groups.
    building_coplane_depth_assist_on: bool = False
    building_coplane_depth_planar_threshold: float = 0.65
    building_coplane_depth_sample_stride: int = 8
    building_coplane_depth_tile_size_px: int = 96
    building_coplane_depth_min_tile_points: int = 64
    building_coplane_depth_tile_fit_max_error_m: float = 0.04
    building_coplane_depth_plane_distance_m: float = 0.08
    building_coplane_depth_normal_angle_deg: float = 15.0
    building_coplane_depth_min_plane_points: int = 600
    building_coplane_depth_max_planes: int = 32

    # Model for splatting.
    model_type: Literal["2dgs", "2dgs-inria"] = "2dgs"

    # Log with wandb
    wandb_on: bool = False
    # Wandb project name
    wandb_project: str = "OMeGa"
    # Wandb group name
    wandb_group: str = "DDP"
    # Dump information to tensorboard every this steps
    log_every: int = 50
    # Save training images to tensorboard
    log_save_image: bool = False

    # Dust3r
    vfm_init: bool = False

    # mesh loss
    mesh_loss: bool = False
    mesh_loss_start_iter: int = 1000
    mesh_normal_consistency_lambda: float = 1.0
    mesh_smooth_lambda: float = 40.0

    # mesh prune
    mesh_prune_opacity_threshold: float = 0.005

    path_to_mesh: str = ""

    # init num2gs
    num_mesh2gs: int = 1

    # learning rate for parameters
    lr_vertices: float = 1.6e-4

    lr_uvsum: float = 0.01

    lr_uratio: float = 0.01

    lr_scalelambda: float = 0.01

    lr_quat: float = 1e-3

    lr_rot2d: float = 1e-2
    
    lr_sh0: float = 2.5e-3
    
    lr_shN: float = 2.5e-3 / 20
    
    lr_opa: float = 5e-2

    # remove mesh faces settings
    remove_cd: float = 0.001
    remove_cd_norm: float = 0.00
    remove_every: int = 500
    remove_start_iter: int = 5000

    min_rel_scale: float = 0.1
    max_rel_scale: float = 1.5

    # mesh split
    mesh_refine_start_iter: int = 3_000

    mesh_split_topk: float = 0.02

    split_every: int = 500

    # Local debug preview. This does not change optimization; it writes a
    # fixed-camera RGB/splat/mesh/normal/wire contact sheet during training.
    mesh_preview_on: bool = False
    mesh_preview_every: int = 500
    mesh_preview_frame: int = 0
    mesh_preview_max_width: int = 900
    mesh_preview_max_faces: int = 0
    mesh_preview_near_plane: float = 0.05
    mesh_preview_contact_sheet_images: int = 24
    mesh_preview_rotate_clockwise: bool = True
    mesh_preview_wire_thickness: int = 1
    mesh_preview_wire_alpha: float = 0.30

    # Local extension preview. It uses the same fixed camera frame as the mesh
    # preview and visualizes what the depth/coplane losses see: prior depth,
    # rendered depth, weighted residuals, coplane groups, and plane residuals.
    regularization_preview_on: bool = False
    regularization_preview_every: int = 500
    regularization_preview_max_width: int = 900
    regularization_preview_contact_sheet_images: int = 24
    regularization_preview_depth_residual_scale_m: float = 0.25
    regularization_preview_coplane_residual_vmax_m: float = 0.05

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.pos_lr_steps = int(self.pos_lr_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.refine_start_iter = int(self.refine_start_iter * factor)
        self.refine_stop_iter = int(self.refine_stop_iter * factor)
        self.reset_every = int(self.reset_every * factor)
        self.refine_every = int(self.refine_every * factor)
        self.building_depth_regularization_start_iter = int(self.building_depth_regularization_start_iter * factor)
        self.building_coplane_start_iter = int(self.building_coplane_start_iter * factor)
        self.building_coplane_refresh_every = max(1, int(self.building_coplane_refresh_every * factor))
        self.regularization_preview_every = max(1, int(self.regularization_preview_every * factor))

def create_mesh_anchors_with_optimizers(
    parser: Parser,
    cfg: Config,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    # init mesh peoperties
    mesh = o3d.io.read_triangle_mesh(cfg.path_to_mesh)
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()

    vertices = torch.nn.Parameter(
        torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=device, requires_grad=True)
    )  # [N, 3]

    faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long, device=device, requires_grad=False)  # [F, 3]
    
    params = [
        ("vertices", torch.nn.Parameter(vertices), cfg.lr_vertices * scene_scale),
    ]

    mesh_anchors = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizers = {
        name: (torch.optim.Adam)(
            [{"params": mesh_anchors[name], "lr": lr * math.sqrt(BS)}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return mesh_anchors, faces, optimizers

def init_gs_from_mesh(
        parser: Parser,
        cfg: Config, 
        mesh_params,
        feature_dim: Optional[int] = None,
        device="cuda"):
    """
    mesh2gs: convert mehs faces to gs ellipses
    
    @Learnable parameters:
    1. lambda: [N,2] -> means = A + u·AB + v·AC, where u + v <= 1
    2. scale_lambda: [N, 2]
    3. rot_2d: [N, 2]
    4. opacities: [N]
    5. sh0: [N, 1]
    6. shN: [N, 15]
    
    @Intermediate Parameters:
    1. means: [N, 3]
    2. scales: [N, 3]
    3. quats: [N, 4]

    @ Index
    1. index: [N]
    """
    # points cloud
    init_points = torch.from_numpy(parser.points).float().to(device)             # [n, 3]
    init_colors = torch.from_numpy(parser.points_rgb / 255.0).float().to(device) # [n, 3]

    num_faces = mesh_params["faces"].shape[0]

    faces = mesh_params["faces"]
    vertices = mesh_params["vertices"]
    triangles = vertices[faces]

    assert cfg.num_mesh2gs == 1
    means = calculate_means(triangles)
    scales = calculate_scales(triangles)
    Rs = calculate_Rs(triangles)

    ####################### learnable parameters #######################
    A = vertices[faces[:, 0], :].unsqueeze(1).repeat(1, cfg.num_mesh2gs, 1).reshape(num_faces * cfg.num_mesh2gs, 3)
    B = vertices[faces[:, 1], :].unsqueeze(1).repeat(1, cfg.num_mesh2gs, 1).reshape(num_faces * cfg.num_mesh2gs, 3)
    C = vertices[faces[:, 2], :].unsqueeze(1).repeat(1, cfg.num_mesh2gs, 1).reshape(num_faces * cfg.num_mesh2gs, 3)
    AB = B - A
    AC = C - A

    AO = means - A

    M = torch.stack([AB, AC], dim=-1)  # [F, 3, 2]

    uv = torch.linalg.lstsq(M, AO.unsqueeze(-1)).solution.squeeze(-1)

    u = uv[:, :1]    # [F, 1] 
    v = uv[:, 1:]    # [F, 1]

    uv_sum = torch.logit((u + v).clamp(0.0001, 0.9999))
    u_ratio = torch.logit((u / (u + v)).clamp(0.0001, 0.9999))

    scale_lambda =  torch.logit(0.5 * torch.ones((num_faces * cfg.num_mesh2gs, 2), dtype=torch.float, device=device))

    rot_2d = torch.Tensor([[1, 0]]).repeat(num_faces * cfg.num_mesh2gs, 1).to(device)

    opacities = torch.logit(cfg.init_opa * torch.ones((num_faces * cfg.num_mesh2gs), dtype=torch.float, device=device))   # [F]

    params = {
        "uv_sum": uv_sum,
        "u_ratio": u_ratio,
        "scale_lambda": scale_lambda,
        "rot_2d": rot_2d,
        "opacities": opacities
    }

    if feature_dim is None:
        N = means.shape[0]
        rgbs = init_colors_from_pc(init_points, init_colors, means)
        colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)

        sh0 = colors[:, :1, :].to(device)
        shN = colors[:, 1:, :].to(device)

        params.update({
            "sh0": sh0,
            "shN": shN,
        })
    else:
        N = means.shape[0]
        rgbs = init_colors_from_pc(init_points, init_colors, means)
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        colors = torch.logit(rgbs)  # [N, 3]
        params.update({
            "features":  features,
            "colors":  colors,
        })

    params = {n: torch.nn.Parameter(v.to(device)) for n, v in params.items()}

    ####################### Intermediate parameters #######################
    scale_range = cfg.max_rel_scale - cfg.min_rel_scale
    scales = torch.cat([scales * (torch.sigmoid(scale_lambda) * scale_range + cfg.min_rel_scale), torch.ones_like(scales)[:,:1]], dim=-1)
    scales = torch.log(scales)

    normalized_rot_2d = F.normalize(rot_2d, dim=-1)
    R_0 = normalized_rot_2d[..., 0:1] * Rs[..., 0] + normalized_rot_2d[..., 1:2] * Rs[..., 1]
    R_1 = -normalized_rot_2d[..., 1:2] * Rs[..., 0] + normalized_rot_2d[..., 0:1] * Rs[..., 1]
    R_2 = Rs[..., 2]
    R = torch.cat([R_0[..., None], R_1[..., None], R_2[..., None]], dim=-1)
    quats = rotation_matrix_to_quaternion(R)

    gs2mesh_index = torch.arange(num_faces, dtype=torch.long, device=device).unsqueeze(-1).repeat(1, cfg.num_mesh2gs).reshape(-1)

    params.update({
        "means": means,
        "scales": scales,
        "quats": quats,
        "index": gs2mesh_index
    })
    
    return params


def update_gs(cfg: Config, mesh_params, gs_params, optimizers, world_size):

    faces = mesh_params["faces"]
    vertices = mesh_params["vertices"]
    gs2mesh_index = gs_params["index"]
    
    # update gs means & quats based on leanable paramss
    uv_sum = gs_params["uv_sum"]
    u_ratio = gs_params["u_ratio"]
    u = torch.sigmoid(uv_sum) * torch.sigmoid(u_ratio)
    v = torch.sigmoid(uv_sum) - u

    triangles = vertices[faces][gs2mesh_index]

    means = (
        triangles[:,0]
        + u * (triangles[:,1] - triangles[:,0])
        + v * (triangles[:,2] - triangles[:,0])
    )

    scale_lambda = gs_params["scale_lambda"]
    scales = calculate_scales(triangles)
    
    scale_range = cfg.max_rel_scale - cfg.min_rel_scale
    scales = torch.cat([scales * (torch.sigmoid(scale_lambda) * scale_range + cfg.min_rel_scale), torch.ones_like(scales)[:,:1]], dim=-1)
    scales = torch.log(scales)

    Rs = calculate_Rs(triangles)
    rot_2d = gs_params["rot_2d"]
    normalized_rot_2d = F.normalize(rot_2d, dim=-1)
    R_0 = normalized_rot_2d[..., 0:1] * Rs[..., 0] + normalized_rot_2d[..., 1:2] * Rs[..., 1]
    R_1 = -normalized_rot_2d[..., 1:2] * Rs[..., 0] + normalized_rot_2d[..., 0:1] * Rs[..., 1]
    R_2 = Rs[..., 2]
    R = torch.cat([R_0[..., None], R_1[..., None], R_2[..., None]], dim=-1)
    quats = rotation_matrix_to_quaternion(R)

    params = {
        "means": means,
        "scales": scales,
        "quats": quats
    }

    gs_params.update(params)

class Runner:
    """Engine for training and testing."""

    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config, run) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        self.run = run
        self.distributed = world_size > 1
        self.resume_start_step = 0

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/plys"
        os.makedirs(self.ply_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.eval_dir = f"{cfg.result_dir}/eval"
        os.makedirs(self.eval_dir, exist_ok=True)
        self.vis_dir = f"{cfg.result_dir}/vis"
        os.makedirs(self.vis_dir, exist_ok=True)
        self.mesh_preview_dir = f"{cfg.result_dir}/mesh_previews"
        if cfg.mesh_preview_on:
            os.makedirs(self.mesh_preview_dir, exist_ok=True)
        self.regularization_preview_dir = f"{cfg.result_dir}/regularization_previews"
        if cfg.regularization_preview_on:
            os.makedirs(self.regularization_preview_dir, exist_ok=True)
            if write_regularization_preview is None or update_regularization_contact_sheet is None:
                raise ImportError("omega_local regularization preview renderer is unavailable. Check this checkout and PYTHONPATH.")
        self.mesh_preview_frame_cache = None
        self.building_coplane_targets = None
        if cfg.building_depth_regularization_on:
            if compute_depth_regularization_loss is None:
                raise ImportError("omega_local depth regularization module is unavailable. Use the local runner so PYTHONPATH includes this repo.")
            os.makedirs(self.stats_dir, exist_ok=True)
            with open(f"{self.stats_dir}/building_depth_regularization_config.json", "w") as f:
                json.dump(
                    {
                        "equation": "L_depth = lambda_d * mean_p w(p) * rho(D_render(p) - D_prior(p))",
                        "guideDir": cfg.building_depth_guide_dir,
                        "startIter": int(cfg.building_depth_regularization_start_iter),
                        "lambda": float(cfg.building_depth_lambda),
                        "huberMeters": float(cfg.building_depth_huber_m),
                        "categoryWeights": {
                            "localPlanar": float(cfg.building_depth_planar_weight),
                            "detail": float(cfg.building_depth_detail_weight),
                            "ridge": float(cfg.building_depth_ridge_weight),
                        },
                        "distanceDecayMeters": float(cfg.building_depth_distance_decay_m),
                    },
                    f,
                    indent=2,
                )
            print(
                "Enabled OMeGa-4-Building rendered-depth regularization. "
                "Guide01 planar/detail/ridge probabilities weight the depth loss."
            )
        if cfg.building_coplane_regularization_on:
            if build_coplane_targets is None or compute_coplane_regularization_loss is None or should_refresh_coplane_targets is None:
                raise ImportError("omega_local coplane regularization module is unavailable. Use the local runner so PYTHONPATH includes this repo.")
            os.makedirs(self.stats_dir, exist_ok=True)
            with open(f"{self.stats_dir}/building_coplane_regularization_config.json", "w") as f:
                json.dump(
                    {
                        "equation": "L_coplane = lambda_p * mean_f rho(n_g dot x_f + d_g) + lambda_n * mean_f (1 - |normal_f dot n_g|)",
                        "startIter": int(cfg.building_coplane_start_iter),
                        "refreshEvery": int(cfg.building_coplane_refresh_every),
                        "pointLambda": float(cfg.building_coplane_point_lambda),
                        "normalLambda": float(cfg.building_coplane_normal_lambda),
                        "huberMeters": float(cfg.building_coplane_huber_m),
                        "grouping": {
                            "normalAngleDegrees": float(cfg.building_coplane_normal_angle_deg),
                            "planeDistanceMeters": float(cfg.building_coplane_plane_distance_m),
                            "minFaces": int(cfg.building_coplane_min_faces),
                            "minGroupAreaM2": float(cfg.building_coplane_min_group_area_m2),
                            "minFaceAreaM2": float(cfg.building_coplane_min_face_area_m2),
                            "maxGroups": int(cfg.building_coplane_max_groups),
                        },
                    },
                    f,
                    indent=2,
                )
            print(
                "Enabled OMeGa-4-Building coplane regularization. "
                "Mesh faces are periodically grouped into cached plane targets."
            )

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=False,
            vfm_init=cfg.vfm_init,
            test_every=cfg.test_every,
            load_normal_maps=cfg.normal_map_on,
            load_building_depth_guides=cfg.building_depth_regularization_on,
            building_depth_guide_dir=cfg.building_depth_guide_dir,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Precompute training camera positions for nearest-neighbor appearance embedding lookup
        if cfg.app_opt:
            train_indices = self.trainset.indices
            self.train_cam_positions = torch.from_numpy(
                self.parser.camtoworlds[train_indices, :3, 3]
            ).float()  # [N_train, 3]

        # Model
        feature_dim = 32 if cfg.app_opt else None
        mesh_anchors, faces, self.optimizers = create_mesh_anchors_with_optimizers(
            self.parser,
            cfg,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )

        self.mesh_params = {**mesh_anchors, "faces": faces}
        
        self.splats = init_gs_from_mesh(self.parser, 
                                        self.cfg, 
                                        mesh_params=self.mesh_params, 
                                        feature_dim=feature_dim,
                                        device=self.device)

        splats_params = [
            ("uv_sum", self.splats["uv_sum"], cfg.lr_uvsum),
            ("u_ratio", self.splats["u_ratio"], cfg.lr_uratio),
            ("scale_lambda", self.splats["scale_lambda"], cfg.lr_scalelambda),
            ("rot_2d", self.splats["rot_2d"], cfg.lr_rot2d),
            ("opacities", self.splats["opacities"], cfg.lr_opa),
        ]
        if cfg.app_opt:
            splats_params += [
                ("features",  self.splats["features"], 2.5e-3),
                ("colors", self.splats["colors"], 2.5e-3),
                ]
        else:
            splats_params += [
                ("sh0", self.splats["sh0"], cfg.lr_sh0),
                ("shN", self.splats["shN"], cfg.lr_shN),
            ]

        BS = cfg.batch_size * world_size
        self.optimizers.update({
            name: (torch.optim.Adam)(
                [{"params": params, "lr": lr * math.sqrt(BS)}],
                eps=1e-15 / math.sqrt(BS),
                betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            ) for name, params, lr in splats_params
        })
        
        print("Saving initial Gaussians")
        save_ply(os.path.join(self.cfg.result_dir,"plys/init.ply"), self.splats)
        
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        self.model_type = cfg.model_type

        if self.model_type == "2dgs":
            key_for_gradient = "gradient_2dgs"
        else:
            key_for_gradient = "means2d"

        assert cfg.reset_every >= cfg.pause_refine_after_reset, "[ERROR] reset_every must be >= pause_refine_after_reset."
        # Strategy
        self.strategy = MeshGSStrategy(
            verbose=True,
            prune_opa=cfg.prune_opa,
            prune_scale3d=cfg.prune_scale3d,
            grow_scale3d=cfg.grow_scale3d,
            grow_grad2d=cfg.grow_grad2d,
            pause_refine_after_reset=cfg.pause_refine_after_reset,
            refine_start_iter=cfg.refine_start_iter,
            refine_stop_iter=cfg.refine_stop_iter,
            refine_every=cfg.refine_every,
            reset_every=cfg.reset_every,
            split_every=cfg.split_every,
            mesh_refine_start_iter=cfg.mesh_refine_start_iter,
            remove_cd=cfg.remove_cd,
            remove_cd_norm=cfg.remove_cd_norm,
            remove_every=cfg.remove_every,
            remove_start_iter=cfg.remove_start_iter,
            revised_opacity=cfg.revised_opacity,
            absgrad=cfg.absgrad,
            key_for_gradient=key_for_gradient,
        )

        # self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(
            scene_scale = self.scene_scale
        )
        self.mesh_state = self.strategy.initialize_mesh_state()

        # Metrics & Loss
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        if cfg.resume_from_checkpoint is not None:
            self._restore_training_checkpoint(Path(cfg.resume_from_checkpoint))

        # Losses & Metrics.

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]
        
        image_ids = kwargs.pop("image_ids", None)

        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        if self.model_type == "2dgs":
            rasterize_fnc = rasterization_2dgs
        elif self.model_type == "2dgs-inria":
            rasterize_fnc = rasterization_2dgs_inria_wrapper

        renders, info = rasterize_fnc(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.distributed,
            **kwargs,
        )

        if self.model_type == "2dgs":
            (
                render_colors,
                render_alphas,
                render_normals,
                normals_from_depth,
                render_distort,
                render_median,
                render_depths,
            ) = renders
        elif self.model_type == "2dgs-inria":
            render_colors, render_depths, render_alphas = renders
            render_normals = info["normals_rend"]
            normals_from_depth = info["normals_surf"]
            render_distort = info["render_distloss"]
            render_median = render_colors[..., 3]

        return (
            render_colors,
            render_alphas,
            render_normals,
            normals_from_depth,
            render_distort,
            render_median,
            render_depths,
            info,
        )

    def _load_mesh_preview_frame(self):
        """Load one fixed parser-frame image and camera for debug previews."""

        if self.mesh_preview_frame_cache is not None:
            return self.mesh_preview_frame_cache

        frame_index = int(np.clip(self.cfg.mesh_preview_frame, 0, len(self.parser.image_names) - 1))
        image = imageio.imread(self.parser.image_paths[frame_index])[..., :3]
        building_depth_guide = None
        if self.parser.load_building_depth_guides and self.parser.building_depth_guide_paths is not None:
            building_depth_guide = _load_building_depth_guide_npz(self.parser.building_depth_guide_paths[frame_index])
            building_depth_guide = _resize_guide_maps(
                building_depth_guide,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
            )
        camera_id = self.parser.camera_ids[frame_index]
        K = self.parser.Ks_dict[camera_id].copy()
        params = self.parser.params_dict[camera_id]
        camtoworld = self.parser.camtoworlds[frame_index]

        if len(params) > 0:
            mapx, mapy = self.parser.mapx_dict[camera_id], self.parser.mapy_dict[camera_id]
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            image = image[y : y + h, x : x + w]
            if building_depth_guide is not None:
                building_depth_guide = _remap_guide_maps(building_depth_guide, mapx, mapy)
                building_depth_guide = _crop_guide_maps(building_depth_guide, x, y, w, h)

        self.mesh_preview_frame_cache = {
            "frame_index": frame_index,
            "image": image,
            "K": K,
            "camtoworld": camtoworld,
            "building_depth_guide": building_depth_guide,
        }
        return self.mesh_preview_frame_cache

    @torch.no_grad()
    def _maybe_write_mesh_preview(self, step: int, max_steps: int, *, phase: str = "pre") -> None:
        """Write a fixed-view preview without changing OMeGa's optimization."""

        if not self.cfg.mesh_preview_on or self.world_rank != 0:
            return
        if (
            resize_rgb_and_intrinsics_for_preview is None
            or write_mesh_training_preview is None
            or update_mesh_training_contact_sheet is None
        ):
            raise ImportError("omega_local mesh preview renderer is unavailable. Check this checkout and PYTHONPATH.")

        every = max(int(self.cfg.mesh_preview_every), 1)
        if phase == "pre":
            should_write = step == 0 or step % every == 0
        elif phase == "final":
            should_write = step == max_steps - 1
        else:
            should_write = False
        if not should_write:
            return

        frame = self._load_mesh_preview_frame()
        rgb_preview, K_preview, _ = resize_rgb_and_intrinsics_for_preview(
            frame["image"],
            frame["K"],
            max_width=int(self.cfg.mesh_preview_max_width),
        )
        preview_height, preview_width = rgb_preview.shape[:2]

        # Scheduled "pre" previews are called immediately after the normal
        # training-loop update_gs call, so do not recompute derived splat tensors
        # under no_grad here; doing so would detach the graph used by the
        # upcoming training render. The final preview happens after the last
        # optimizer step, so it needs one explicit refresh.
        if phase == "final":
            update_gs(self.cfg, self.mesh_params, self.splats, self.optimizers, self.world_size)
        camtoworld = torch.from_numpy(frame["camtoworld"]).float().to(self.device)
        K = torch.from_numpy(K_preview).float().to(self.device)
        splat_colors, _, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=camtoworld[None],
            Ks=K[None],
            width=int(preview_width),
            height=int(preview_height),
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB+ED",
        )
        splat_rgb = torch.clamp(splat_colors[0, ..., :3], 0.0, 1.0).detach().cpu().numpy()
        splat_rgb = (splat_rgb * 255.0).astype(np.uint8)

        vertices = self.mesh_params["vertices"].detach().cpu().numpy()
        faces = self.mesh_params["faces"].detach().cpu().numpy()
        summary = write_mesh_training_preview(
            out_dir=Path(self.mesh_preview_dir),
            step=step,
            frame_index=int(frame["frame_index"]),
            phase=phase,
            vertices=vertices,
            faces=faces,
            K=K_preview,
            camtoworld=frame["camtoworld"],
            rgb=rgb_preview,
            splat_rgb=splat_rgb,
            max_width=0,
            near_plane=float(self.cfg.mesh_preview_near_plane),
            max_faces=int(self.cfg.mesh_preview_max_faces),
            rotate_clockwise=bool(self.cfg.mesh_preview_rotate_clockwise),
            wire_thickness=int(self.cfg.mesh_preview_wire_thickness),
            wire_alpha=float(self.cfg.mesh_preview_wire_alpha),
        )
        update_mesh_training_contact_sheet(
            preview_dir=Path(self.mesh_preview_dir),
            max_images=int(self.cfg.mesh_preview_contact_sheet_images),
        )
        with open(f"{self.stats_dir}/mesh_preview.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(
            "[mesh-preview] "
            f"step={step} phase={phase} frame={summary['frame_index']} faces_rendered={summary['rendered_face_count']} "
            f"path={summary['path']}"
        )

    @torch.no_grad()
    def _maybe_write_regularization_preview(self, step: int, max_steps: int, *, phase: str = "pre") -> None:
        """Write fixed-frame diagnostics for depth and coplane regularizers."""

        if not self.cfg.regularization_preview_on or self.world_rank != 0:
            return
        if not (self.cfg.building_depth_regularization_on or self.cfg.building_coplane_regularization_on):
            return
        if write_regularization_preview is None or update_regularization_contact_sheet is None:
            raise ImportError("omega_local regularization preview renderer is unavailable. Check this checkout and PYTHONPATH.")

        every = max(int(self.cfg.regularization_preview_every), 1)
        if phase == "pre":
            should_write = step == 0 or step % every == 0
        elif phase == "final":
            should_write = step == max_steps - 1
        else:
            should_write = False
        if not should_write:
            return

        if self.cfg.building_coplane_regularization_on and step >= self.cfg.building_coplane_start_iter:
            self._maybe_refresh_building_coplane_targets(step)

        frame = self._load_mesh_preview_frame()
        rgb_preview, K_preview, _ = resize_rgb_and_intrinsics_for_preview(
            frame["image"],
            frame["K"],
            max_width=int(self.cfg.regularization_preview_max_width),
        )
        preview_height, preview_width = rgb_preview.shape[:2]

        # Like mesh previews, scheduled pre-step diagnostics use the existing
        # update_gs result. Final diagnostics happen after the last optimizer
        # step, so refresh the mesh-controlled splat tensors once.
        if phase == "final":
            update_gs(self.cfg, self.mesh_params, self.splats, self.optimizers, self.world_size)
        camtoworld = torch.from_numpy(frame["camtoworld"]).float().to(self.device)
        K = torch.from_numpy(K_preview).float().to(self.device)
        _, alphas, _, _, _, _, render_depths, _ = self.rasterize_splats(
            camtoworlds=camtoworld[None],
            Ks=K[None],
            width=int(preview_width),
            height=int(preview_height),
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB+D",
            distloss=False,
        )

        guide = frame.get("building_depth_guide") or {}
        vertices = self.mesh_params["vertices"].detach().cpu().numpy()
        faces = self.mesh_params["faces"].detach().cpu().numpy()
        summary = write_regularization_preview(
            out_dir=Path(self.regularization_preview_dir),
            step=step,
            frame_index=int(frame["frame_index"]),
            phase=phase,
            rotate_clockwise=bool(self.cfg.mesh_preview_rotate_clockwise),
            rgb=frame["image"],
            K=frame["K"],
            camtoworld=frame["camtoworld"],
            rendered_depth=render_depths.detach().cpu().numpy(),
            rendered_alpha=alphas.detach().cpu().numpy(),
            depth_prior=guide.get("depth"),
            depth_valid=guide.get("valid"),
            p_planar=guide.get("p_local_planar"),
            p_detail=guide.get("p_detail"),
            p_ridge=guide.get("p_ridge"),
            vertices=vertices,
            faces=faces,
            coplane_targets=self.building_coplane_targets,
            max_width=int(self.cfg.regularization_preview_max_width),
            near_plane=float(self.cfg.mesh_preview_near_plane),
            depth_min_m=float(self.cfg.building_depth_min_m),
            depth_max_m=float(self.cfg.building_depth_max_m),
            depth_alpha_min=float(self.cfg.building_depth_min_alpha),
            depth_planar_weight=float(self.cfg.building_depth_planar_weight),
            depth_detail_weight=float(self.cfg.building_depth_detail_weight),
            depth_ridge_weight=float(self.cfg.building_depth_ridge_weight),
            depth_distance_decay_m=float(self.cfg.building_depth_distance_decay_m),
            residual_scale_m=float(self.cfg.regularization_preview_depth_residual_scale_m),
            coplane_residual_vmax_m=float(self.cfg.regularization_preview_coplane_residual_vmax_m),
        )
        update_regularization_contact_sheet(
            preview_dir=Path(self.regularization_preview_dir),
            max_images=int(self.cfg.regularization_preview_contact_sheet_images),
        )
        with open(f"{self.stats_dir}/regularization_preview.jsonl", "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(
            "[regularization-preview] "
            f"step={step} phase={phase} frame={summary['frame_index']} "
            f"depth_pixels={int(summary['depth_used_pixels'])} "
            f"coplane_faces={int(summary['active_coplane_faces'])} "
            f"path={summary['path']}"
        )

    @torch.no_grad()
    def _maybe_refresh_building_coplane_targets(self, step: int) -> None:
        """Refresh stop-gradient plane targets for coplane regularization.

        This is intentionally outside the differentiable loss.  Plane grouping
        is a structural hypothesis built from the current mesh; the following
        optimization window then treats those fitted planes as fixed targets.
        """

        if not self.cfg.building_coplane_regularization_on:
            return
        if not should_refresh_coplane_targets(
            targets=self.building_coplane_targets,
            face_count=int(self.mesh_params["faces"].shape[0]),
            step=int(step),
            refresh_every=int(self.cfg.building_coplane_refresh_every),
        ):
            return

        depth_planes = None
        if bool(self.cfg.building_coplane_depth_assist_on):
            if build_depth_plane_hypotheses is None:
                raise ImportError("omega_local depth-assisted coplane grouping is unavailable. Check PYTHONPATH.")
            frame = self._load_mesh_preview_frame()
            guide = frame.get("building_depth_guide")
            if guide is not None:
                # Depth-assisted grouping is a soft merge hint, not a direct
                # hard assignment. We fit broad planes from high-confidence
                # locally planar depth pixels, then only accept mesh faces that
                # already lie near those planes with compatible normals.
                depth_planes = build_depth_plane_hypotheses(
                    depth=guide["depth"],
                    valid=guide["valid"],
                    p_planar=guide["p_local_planar"],
                    K=frame["K"],
                    camtoworld=frame["camtoworld"],
                    device=self.device,
                    dtype=self.mesh_params["vertices"].dtype,
                    min_depth_m=float(self.cfg.building_depth_min_m),
                    max_depth_m=float(self.cfg.building_depth_max_m),
                    planar_threshold=float(self.cfg.building_coplane_depth_planar_threshold),
                    sample_stride=int(self.cfg.building_coplane_depth_sample_stride),
                    tile_size_px=int(self.cfg.building_coplane_depth_tile_size_px),
                    min_tile_points=int(self.cfg.building_coplane_depth_min_tile_points),
                    tile_fit_max_error_m=float(self.cfg.building_coplane_depth_tile_fit_max_error_m),
                    normal_angle_deg=float(self.cfg.building_coplane_depth_normal_angle_deg),
                    plane_distance_m=float(self.cfg.building_coplane_depth_plane_distance_m),
                    min_plane_points=int(self.cfg.building_coplane_depth_min_plane_points),
                    max_planes=int(self.cfg.building_coplane_depth_max_planes),
                )

        self.building_coplane_targets = build_coplane_targets(
            vertices=self.mesh_params["vertices"],
            faces=self.mesh_params["faces"],
            step=int(step),
            normal_angle_deg=float(self.cfg.building_coplane_normal_angle_deg),
            plane_distance_m=float(self.cfg.building_coplane_plane_distance_m),
            min_faces=int(self.cfg.building_coplane_min_faces),
            min_group_area_m2=float(self.cfg.building_coplane_min_group_area_m2),
            min_face_area_m2=float(self.cfg.building_coplane_min_face_area_m2),
            max_groups=int(self.cfg.building_coplane_max_groups),
            depth_planes=depth_planes,
            depth_plane_distance_m=float(self.cfg.building_coplane_depth_plane_distance_m),
            depth_normal_angle_deg=float(self.cfg.building_coplane_depth_normal_angle_deg),
        )
        stats = dict(self.building_coplane_targets.stats)
        if self.world_rank == 0:
            with open(f"{self.stats_dir}/building_coplane_groups.jsonl", "a") as f:
                f.write(json.dumps(stats) + "\n")
            print(
                "[building-coplane] "
                f"step={step} groups={int(stats['accepted_groups'])} "
                f"faces={int(stats['accepted_faces'])}/{int(stats['total_faces'])} "
                f"coverage={stats['accepted_face_fraction']:.3f} "
                f"depth_planes={int(stats.get('depth_plane_hypotheses', 0))} "
                f"depth_faces={int(stats.get('depth_assisted_faces', 0))}"
            )

    def _learnable_splat_names(self) -> list[str]:
        if self.cfg.app_opt:
            return ["uv_sum", "u_ratio", "scale_lambda", "rot_2d", "features", "colors", "opacities"]
        return ["uv_sum", "u_ratio", "scale_lambda", "rot_2d", "sh0", "shN", "opacities"]

    def _tensor_checkpoint_dict(self, values: dict) -> dict:
        payload = {}
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                payload[key] = value.detach().cpu()
        return payload

    def _bind_optimizer_param(self, name: str, param: torch.nn.Parameter) -> None:
        if name in self.optimizers:
            self.optimizers[name].param_groups[0]["params"] = [param]

    def _restore_training_checkpoint(self, path: Path) -> None:
        """Restore a local training-resume checkpoint.

        This is distinct from OMeGa's ``--ckpt`` mesh-export path.  Continuing
        mesh optimization needs more than saved splat tensors: the mesh
        vertices/faces, trainable splat parameters, optimizer state, and next
        global step must be restored so baseline and extension branches share
        exactly the same warmup.
        """

        if not path.exists():
            raise FileNotFoundError(f"Missing resume checkpoint: {path}")
        print(f"Restoring OMeGa training state from {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "mesh_params" not in ckpt or "splats" not in ckpt:
            raise ValueError(f"Checkpoint is not a training-resume checkpoint: {path}")

        mesh_payload = ckpt["mesh_params"]
        self.mesh_params["vertices"] = torch.nn.Parameter(mesh_payload["vertices"].to(self.device).float())
        self.mesh_params["faces"] = mesh_payload["faces"].to(self.device).long()
        self._bind_optimizer_param("vertices", self.mesh_params["vertices"])

        learnable = set(self._learnable_splat_names())
        for key, value in ckpt["splats"].items():
            if not isinstance(value, torch.Tensor):
                continue
            restored = value.to(self.device)
            self.splats[key] = torch.nn.Parameter(restored) if key in learnable else restored
            if key in learnable:
                self._bind_optimizer_param(key, self.splats[key])

        optimizer_payload = ckpt.get("optimizers", {})
        for name, state in optimizer_payload.items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state)

        self.strategy_state.update(ckpt.get("strategy_state", {}))
        self.mesh_state.update(ckpt.get("mesh_state", {}))
        self.resume_start_step = int(ckpt.get("next_step", int(ckpt.get("step", -1)) + 1))
        print(f"Resuming training at global step {self.resume_start_step}")

        if self.cfg.pose_opt and "pose_adjust" in ckpt:
            module = self.pose_adjust.module if self.world_size > 1 else self.pose_adjust
            module.load_state_dict(ckpt["pose_adjust"])
        if self.cfg.app_opt and "app_module" in ckpt:
            module = self.app_module.module if self.world_size > 1 else self.app_module
            module.load_state_dict(ckpt["app_module"])

    def _save_training_checkpoint(self, step: int, global_tic: float) -> None:
        """Save a post-optimizer checkpoint suitable for branching experiments."""

        with torch.no_grad():
            update_gs(self.cfg, self.mesh_params, self.splats, self.optimizers, self.world_size)

        mem = torch.cuda.max_memory_allocated() / 1024**3
        stats = {
            "mem": mem,
            "ellipse_time": time.time() - global_tic,
            "num_GS": len(self.splats["means"]),
            "num_faces": len(self.mesh_params["faces"]),
            "step": int(step),
            "next_step": int(step) + 1,
        }
        print("Step: ", step, stats)
        with open(f"{self.stats_dir}/train_step{step:05d}.json", "w") as f:
            json.dump(stats, f)

        data = {
            "schema": "omega_4_building.training_resume.v1",
            "step": int(step),
            "next_step": int(step) + 1,
            "splats": self._tensor_checkpoint_dict(self.splats),
            "mesh_params": self._tensor_checkpoint_dict(self.mesh_params),
            "optimizers": {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()},
            "strategy_state": self._tensor_checkpoint_dict(self.strategy_state),
            "mesh_state": self._tensor_checkpoint_dict(self.mesh_state),
        }
        if self.cfg.pose_opt:
            data["pose_adjust"] = (self.pose_adjust.module if self.world_size > 1 else self.pose_adjust).state_dict()
        if self.cfg.app_opt:
            data["app_module"] = (self.app_module.module if self.world_size > 1 else self.app_module).state_dict()

        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

        point_cloud_path = f"{self.ply_dir}/ply_{step:05d}_rank{self.world_rank}.ply"
        save_ply(point_cloud_path, self.splats)

        vertices = self.mesh_params["vertices"].detach().cpu().numpy()
        faces = self.mesh_params["faces"].cpu().numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        o3d.io.write_triangle_mesh(f"{self.ply_dir}/mesh_{step:05d}_rank{self.world_rank}.ply", mesh)

        if self.world_size > 1:
            dist.barrier()
            if self.world_rank == 0:
                merge_ply(self.ply_dir, step, self.world_size)

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.json", "w") as f:
                json.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = int(self.resume_start_step)
        if init_step >= max_steps:
            raise ValueError(f"resume_from_checkpoint starts at step {init_step}, but max_steps is {max_steps}.")

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["vertices"], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
            ),
            # torch.optim.lr_scheduler.ExponentialLR(
            #     self.optimizers["means"], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
            # ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / cfg.pos_lr_steps)
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if step % 100 == 0:
                torch.cuda.empty_cache()
            
            update_gs(cfg, self.mesh_params, self.splats, self.optimizers, world_size)
            self._maybe_write_mesh_preview(step, max_steps, phase="pre")
            self._maybe_write_regularization_preview(step, max_steps, phase="pre")

            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # get valid mask
            h, w = pixels.shape[1:3]
            mask = torch.ones((1, h, w, 1), dtype=torch.bool).to(device)

            normal_map_valid = None
            if self.cfg.mono_normal_loss or self.cfg.normal_map_on:
                normal_map = data["normal_map"].to(device)
                if "normal_map_valid" in data:
                    normal_map_valid = data["normal_map_valid"].to(device).bool()
            
            # forward
            (
                renders,
                alphas,
                render_normals,
                render_normals_from_depth,
                render_distort,
                render_median,
                render_depths,
                info,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+D",
                distloss=self.cfg.dist_loss,
            )
            
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None
            
            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.strategy.step_pre_backward(
                info=info,
            )
            
            # loss
            l1loss = l1_loss(colors, pixels, mask)
            
            ssimloss = 1.0 - ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), mask=mask)

            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            
            mesh_normal_consistency_loss = torch.tensor(0.0, device=device)
            mesh_smooth_loss = torch.tensor(0.0, device=device)
            scaled_mesh_normal_loss = torch.tensor(0.0, device=device)
            scaled_mesh_smooth_loss = torch.tensor(0.0, device=device)
            if cfg.mesh_loss and step >= cfg.mesh_loss_start_iter:
                mesh = Meshes(verts=[self.mesh_params["vertices"]], faces=[self.mesh_params["faces"]])

                mesh_normal_consistency_loss = mesh_normal_consistency(mesh)
                scaled_mesh_normal_loss = cfg.mesh_normal_consistency_lambda * mesh_normal_consistency_loss
                loss += scaled_mesh_normal_loss

                mesh_smooth_loss = mesh_laplacian_smoothing(mesh)
                scaled_mesh_smooth_loss = (cfg.mesh_smooth_lambda / self.strategy_state['scene_scale']) * mesh_smooth_loss
                loss += scaled_mesh_smooth_loss

            # plane normals with pseudo normals
            mono_normal_loss = torch.tensor(0.0, device=device)
            if self.cfg.mono_normal_loss and step >= self.cfg.mono_normal_loss_start_iter:
                normal_map = normal_map @ camtoworlds[0, :3, :3].T      # Transform to world coordinate
                normal_map = normal_map.squeeze(0)                      # [H, W, 3]

                normals = render_normals.clone().squeeze(0)             # [H, W, 3]: world coordinate
                normals = normals / torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-6)  # normalize

                normal_loss_mask = mask
                if normal_map_valid is not None:
                    normal_loss_mask = normal_loss_mask & normal_map_valid.unsqueeze(-1)
                normal_loss_mask_f = normal_loss_mask.squeeze(0).float()

                abs_diff = torch.abs(normal_map - normals)
                alpha = 2.0
                beta = 2.0
                focal_weight = alpha * (1 - torch.exp(-beta * abs_diff))
                weighted_diff = focal_weight * abs_diff * normal_loss_mask_f
                valid_normal_values = normal_loss_mask_f.sum() * 3.0
                if bool((valid_normal_values > 0).detach().item()):
                    mono_normal_loss = self.cfg.mono_normal_lambda * weighted_diff.sum() / valid_normal_values.clamp(min=1.0)
                
                loss += mono_normal_loss

            distloss = torch.tensor(0.0, device=device)
            curr_dist_lambda = 0.0
            if cfg.dist_loss:
                if step > cfg.dist_start_iter:
                    curr_dist_lambda = cfg.dist_lambda
                else:
                    curr_dist_lambda = 0.0
                distloss = render_distort.mean()
                loss += distloss * curr_dist_lambda

            building_depth_loss = torch.tensor(0.0, device=device)
            scaled_building_depth_loss = torch.tensor(0.0, device=device)
            building_depth_stats = {
                "used_depth_pixels": 0.0,
                "mean_depth_weight": 0.0,
                "mean_abs_depth_error_m": 0.0,
            }
            if cfg.building_depth_regularization_on and step >= cfg.building_depth_regularization_start_iter:
                # OMeGa-4-Building depth prior:
                #   L_depth = lambda_d * mean_p w(p) * rho(D_render(p) - D_prior(p)).
                #
                # w(p) is built from Guide01 soft geometry probabilities. Planar
                # pixels can have high weight, ridge/detail pixels can be weak,
                # and unknown pixels are ignored. D_render is differentiable, so
                # this loss moves the mesh-controlled splats and therefore the
                # underlying mesh vertices.
                (
                    scaled_building_depth_loss,
                    building_depth_loss,
                    building_depth_stats,
                ) = compute_depth_regularization_loss(
                    render_depth=render_depths,
                    render_alpha=alphas,
                    batch=data,
                    depth_lambda=float(cfg.building_depth_lambda),
                    huber_delta_m=float(cfg.building_depth_huber_m),
                    min_depth_m=float(cfg.building_depth_min_m),
                    max_depth_m=float(cfg.building_depth_max_m),
                    min_alpha=float(cfg.building_depth_min_alpha),
                    planar_weight=float(cfg.building_depth_planar_weight),
                    detail_weight=float(cfg.building_depth_detail_weight),
                    ridge_weight=float(cfg.building_depth_ridge_weight),
                    distance_decay_m=float(cfg.building_depth_distance_decay_m),
                )
                loss += scaled_building_depth_loss

            building_coplane_loss = torch.tensor(0.0, device=device)
            building_coplane_point_loss = torch.tensor(0.0, device=device)
            building_coplane_normal_loss = torch.tensor(0.0, device=device)
            building_coplane_stats = {
                "active_faces": 0.0,
                "mean_abs_plane_distance_m": 0.0,
            }
            if cfg.building_coplane_regularization_on and step >= cfg.building_coplane_start_iter:
                # OMeGa-4-Building coplane prior:
                #   L_coplane =
                #       lambda_p * mean_f rho(n_g dot x_f + d_g)
                #     + lambda_n * mean_f (1 - |normal_f dot n_g|).
                #
                # Plane groups (n_g, d_g) are refreshed without gradients every
                # few hundred iterations from faces that already look coplanar.
                # This makes the loss a soft structural pull toward larger flat
                # architectural regions without hard-snapping vertices.
                self._maybe_refresh_building_coplane_targets(step)
                (
                    building_coplane_loss,
                    building_coplane_point_loss,
                    building_coplane_normal_loss,
                    building_coplane_stats,
                ) = compute_coplane_regularization_loss(
                    vertices=self.mesh_params["vertices"],
                    faces=self.mesh_params["faces"],
                    targets=self.building_coplane_targets,
                    point_lambda=float(cfg.building_coplane_point_lambda),
                    normal_lambda=float(cfg.building_coplane_normal_lambda),
                    huber_delta_m=float(cfg.building_coplane_huber_m),
                )
                loss += building_coplane_loss
            
            loss.backward()
            
            # accumulate faces grads
            if self.mesh_params['vertices'].grad is not None:
                vertices = self.mesh_params['vertices'].clone().detach()
                faces = self.mesh_params['faces'].clone().detach()
                vertices_grad = self.mesh_params['vertices'].grad.clone().detach()
                faces_vertices_grad = vertices_grad[faces]
                faces_vertices = vertices[faces]
                
                # projection
                edge1 = faces_vertices[:, 1] - faces_vertices[:, 0]
                edge2 = faces_vertices[:, 2] - faces_vertices[:, 0]
                normals = torch.cross(edge1, edge2, dim=1)
                normals_norm = normals.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
                normals = normals / normals_norm            # [F,3]
                normals_expanded = normals.unsqueeze(1)     # [F, 1, 3]
                projections = torch.sum(faces_vertices_grad * normals_expanded, dim=2).abs()
                faces_grad = projections.sum(dim=1) 
                info['faces_grad'] = faces_grad
            else:
                info['faces_grad'] = None
            
            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.dist_loss:
                desc += f"dist loss={distloss.item():.6f}"
            if cfg.building_depth_regularization_on:
                desc += f" depth={scaled_building_depth_loss.item():.6f}"
            if cfg.building_coplane_regularization_on:
                desc += f" coplane={building_coplane_loss.item():.6f}"
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.log_every > 0 and step % cfg.log_every == 0:
                loss_stats = {
                    "step": int(step),
                    "loss": float(loss.detach().cpu()),
                    "l1loss": float(l1loss.detach().cpu()),
                    "ssimloss": float(ssimloss.detach().cpu()),
                    "mesh_normal_consistency_loss": float(mesh_normal_consistency_loss.detach().cpu()),
                    "scaled_mesh_normal_consistency_loss": float(scaled_mesh_normal_loss.detach().cpu()),
                    "mesh_smooth_loss": float(mesh_smooth_loss.detach().cpu()),
                    "scaled_mesh_smooth_loss": float(scaled_mesh_smooth_loss.detach().cpu()),
                    "mono_normal_loss": float(mono_normal_loss.detach().cpu()),
                    "distloss": float(distloss.detach().cpu()),
                    "dist_lambda": float(curr_dist_lambda),
                    "building_depth_loss": float(building_depth_loss.detach().cpu()),
                    "scaled_building_depth_loss": float(scaled_building_depth_loss.detach().cpu()),
                    "building_depth_used_pixels": float(building_depth_stats["used_depth_pixels"]),
                    "building_depth_mean_weight": float(building_depth_stats["mean_depth_weight"]),
                    "building_depth_mean_abs_error_m": float(building_depth_stats["mean_abs_depth_error_m"]),
                    "building_coplane_loss": float(building_coplane_loss.detach().cpu()),
                    "building_coplane_point_loss": float(building_coplane_point_loss.detach().cpu()),
                    "building_coplane_normal_loss": float(building_coplane_normal_loss.detach().cpu()),
                    "building_coplane_active_faces": float(building_coplane_stats["active_faces"]),
                    "building_coplane_mean_abs_plane_distance_m": float(building_coplane_stats["mean_abs_plane_distance_m"]),
                    "num_GS": int(len(self.splats["means"])),
                    "num_faces": int(len(self.mesh_params["faces"])),
                }
                with open(f"{self.stats_dir}/train_loss.jsonl", "a") as f:
                    f.write(json.dumps(loss_stats) + "\n")

            if world_rank == 0 and cfg.wandb_on and cfg.log_every > 0 and step % cfg.log_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.run.log({
                    "train/loss": loss.item(),
                    "train/l1loss": l1loss.item(),
                    "train/ssimloss": ssimloss.item(),
                    "train/num_GS": len(self.splats["means"]),
                    "train/mem": mem}, step)
                if cfg.mono_normal_loss:
                    self.run.log({"train/mono_normal_loss": mono_normal_loss.item()}, step)
                if cfg.dist_loss:
                    self.run.log({"train/distloss": distloss.item()}, step)
                if cfg.building_depth_regularization_on:
                    self.run.log(
                        {
                            "train/building_depth_loss": building_depth_loss.item(),
                            "train/scaled_building_depth_loss": scaled_building_depth_loss.item(),
                            "train/building_depth_used_pixels": building_depth_stats["used_depth_pixels"],
                        },
                        step,
                    )
                if cfg.building_coplane_regularization_on:
                    self.run.log(
                        {
                            "train/building_coplane_loss": building_coplane_loss.item(),
                            "train/building_coplane_point_loss": building_coplane_point_loss.item(),
                            "train/building_coplane_normal_loss": building_coplane_normal_loss.item(),
                            "train/building_coplane_active_faces": building_coplane_stats["active_faces"],
                        },
                        step,
                    )
                if cfg.log_save_image and step % (cfg.log_every * 5) == 0:
                    canvas = torch.cat([pixels, colors[..., :3]], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.run.log({"train/render": wandb.Image(canvas, caption=f"step: {step}")})

            if step > 0 and step % 1000 == 0:
                # save .ply
                point_cloud_path = f"{self.ply_dir}/ply_{step:05d}_rank{self.world_rank}.ply"
                save_ply(point_cloud_path, self.splats)

                # save mesh
                vertices = self.mesh_params["vertices"].detach().cpu().numpy()
                faces = self.mesh_params["faces"].cpu().numpy()
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                o3d.io.write_triangle_mesh(f"{self.ply_dir}/mesh_{step:05d}_rank{self.world_rank}.ply", mesh)
                if world_size > 1:
                    dist.barrier()

                    if world_rank == 0:
                        # merge ply of diffrent world ranks
                        merge_ply(self.ply_dir, step, self.world_size)

            should_save_checkpoint = step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1

            # Densify
            self.strategy.step_post_backward(
                mesh_params=self.mesh_params,
                gs_params=self.splats,
                cfg=cfg,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mesh_state=self.mesh_state,
                step=step,
                info=info,
                packed=cfg.packed,
            )
            
            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )
            torch.cuda.empty_cache()

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Remove degradation faces
            self.strategy.prune_degradation_faces(
                cfg=self.cfg,
                gs_params=self.splats,
                mesh_params=self.mesh_params,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mesh_state=self.mesh_state,
            )
            self._maybe_write_mesh_preview(step, max_steps, phase="final")
            self._maybe_write_regularization_preview(step, max_steps, phase="final")
            if should_save_checkpoint:
                self._save_training_checkpoint(step, global_tic)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                # self.render_traj(step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = {"psnr": [], "ssim": [], "lpips": [], "l1": [], "ssim_loss": [], "render_loss": []}
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]

            # Find nearest training camera for appearance embedding
            if cfg.app_opt:
                val_pos = camtoworlds[0, :3, 3]  # [3]
                dists = torch.norm(self.train_cam_positions.to(device) - val_pos[None], dim=-1)
                nn_embed_ids = dists.argmin().unsqueeze(0)  # [1]
            else:
                nn_embed_ids = None

            tic = time.time()
            (
                colors,
                alphas,
                normals,
                normals_from_depth,
                render_distort,
                render_median,
                render_depths,
                _,
            ) = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
                image_ids=nn_embed_ids,
            )  # [1, H, W, 3]
            colors = torch.clamp(colors, 0.0, 1.0)
            colors = colors[..., :3]  # Take RGB channels
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            if world_rank == 0:
                # write images
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}.png", (canvas * 255).astype(np.uint8)
                )

                pixels = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W]
                colors = colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                l1_value = torch.mean(torch.abs(colors - pixels))
                ssim_loss_value = 1.0 - ssim(colors, pixels)
                render_loss_value = l1_value * (1.0 - cfg.ssim_lambda) + ssim_loss_value * cfg.ssim_lambda
                metrics["psnr"].append(self.psnr(colors, pixels))
                metrics["ssim"].append(self.ssim(colors, pixels))
                metrics["lpips"].append(self.lpips(colors, pixels))
                metrics["l1"].append(l1_value)
                metrics["ssim_loss"].append(ssim_loss_value)
                metrics["render_loss"].append(render_loss_value)

                # # write median depths
                # render_median = (render_median - render_median.min()) / (render_median.max() - render_median.min())
                # render_median = render_median.detach().cpu().squeeze(0).unsqueeze(-1).repeat(1, 1, 3).numpy()
                # # render_median = render_median.detach().cpu().squeeze(0).repeat(1, 1, 3).numpy()
                # imageio.imwrite(
                #     f"{self.eval_dir}/val_{i:04d}_median_depth_{step}.png", (render_median * 255).astype(np.uint8)
                # )
        
                # write normals
                normals = (normals * 0.5 + 0.5).squeeze(0).cpu().numpy()
                normals_output = (normals * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}_normal_{step}.png", normals_output
                )

                # write normals from depth
                normals_from_depth *= alphas.squeeze(0).detach()
                normals_from_depth = (normals_from_depth * 0.5 + 0.5).cpu().numpy()
                normals_from_depth = (normals_from_depth - np.min(normals_from_depth)) / (
                    np.max(normals_from_depth) - np.min(normals_from_depth)
                )
                normals_from_depth_output = (normals_from_depth * 255).astype(np.uint8)
                if len(normals_from_depth_output.shape) == 4:
                    normals_from_depth_output = normals_from_depth_output.squeeze(0)
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}_normals_from_depth_{step}.png",
                    normals_from_depth_output,
                )

                # write distortions
                render_dist = render_distort
                dist_max = torch.max(render_dist)
                dist_min = torch.min(render_dist)
                render_dist = (render_dist - dist_min) / (dist_max - dist_min)
                render_dist = (
                    colormap(render_dist.cpu().numpy()[0])
                    .permute((1, 2, 0))
                    .numpy()
                    .astype(np.uint8)
                )
                imageio.imwrite(
                    f"{self.eval_dir}/val_{i:04d}_distortions_{step}.png", render_dist
                )

        if world_rank == 0:
            ellipse_time /= len(valloader)

            val_psnr = torch.stack(metrics["psnr"]).mean()
            val_ssim = torch.stack(metrics["ssim"]).mean()
            val_lpips = torch.stack(metrics["lpips"]).mean()
            val_l1 = torch.stack(metrics["l1"]).mean()
            val_ssim_loss = torch.stack(metrics["ssim_loss"]).mean()
            val_render_loss = torch.stack(metrics["render_loss"]).mean()
            print(
                f"PSNR: {val_psnr.item():.3f}, SSIM: {val_ssim.item():.4f}, LPIPS: {val_lpips.item():.3f}, "
                f"L1: {val_l1.item():.4f}, render_loss: {val_render_loss.item():.4f} "
                f"Time: {ellipse_time:.3f}s/image "
                f"Number of GS: {len(self.splats['means'])}"
            )
            # save stats as json
            stats = {
                "psnr": val_psnr.item(),
                "ssim": val_ssim.item(),
                "lpips": val_lpips.item(),
                "l1": val_l1.item(),
                "ssim_loss": val_ssim_loss.item(),
                "render_loss": val_render_loss.item(),
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            }
            with open(f"{self.stats_dir}/val_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to wandb
            if self.cfg.wandb_on:
                self.run.log({f"val/{k}": v for k, v in stats.items()}, step)

    @torch.no_grad()
    def render(self, step: int):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[:]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for i in tqdm.trange(len(camtoworlds), desc="Rendering"):
            renders, _, normals, _, _, render_depths, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]

            colors = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0).cpu().numpy()  # [H, W, 3]
            depths = render_depths.squeeze()  # [H, W, 1]
            normals = torch.nn.functional.normalize(normals[0], dim=2)
            normals = normals.cpu().numpy()
            
            save_image(colors, f"{self.render_dir}/render_{i:04d}.png")
            depth_gray2rgb(depths, f"{self.render_dir}/depth_{i:04d}.png", 0.1, 5)
            normal_rgb2sn(normals, f"{self.render_dir}/normal_{i:04d}.png")

    @torch.no_grad()
    def export_mesh(self, depth_trunc=3, mesh_res=1024):
        voxel_size = depth_trunc / mesh_res
        sdf_trunc = 5.0 * voxel_size
        
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        cfg = self.cfg
        device = self.device

        camtoworlds = self.parser.camtoworlds[:]
        camtoworlds = generate_interpolated_path(camtoworlds, 1)  # [N, 3, 4]
        camtoworlds = np.concatenate(
            [
                camtoworlds,
                np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds), axis=0),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds = torch.from_numpy(camtoworlds).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for i in tqdm.trange(len(camtoworlds), desc="Exporting mesh"):
            renders, _, normals, _, _, render_depths, _ = self.rasterize_splats(
                camtoworlds=camtoworlds[i : i + 1],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]          
            rgb = torch.clamp(renders[0, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
            depth = render_depths.squeeze()  # [H, W]

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(rgb.cpu().numpy() * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
            # ndc2pix = torch.tensor([
            #     [width / 2, 0, 0, (width-1) / 2],
            #     [0, height / 2, 0, (height-1) / 2],
            #     [0, 0, 0, 1]]).float().cuda().T
            # intrins =  (K @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=width, height=height,
                cx = K[0,2].item(),
                cy = K[1,2].item(), 
                fx = K[0,0].item(), 
                fy = K[1,1].item()
            )
            extrinsic=np.linalg.inv(np.asarray((camtoworlds[i]).cpu().numpy()))
            volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=extrinsic)
            del renders, normals, rgb, depth
            gc.collect()
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(os.path.join(self.cfg.result_dir, "fuse.ply"), mesh)
        print("mesh saved at {}".format(os.path.join(self.cfg.result_dir, "fuse.ply")))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
        o3d.io.write_triangle_mesh(os.path.join(self.cfg.result_dir, "fuse_post.ply"), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(self.cfg.result_dir, "fuse_post.ply")))

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    run = wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group,
    ) if world_rank == 0 and cfg.wandb_on else None
    
    runner = Runner(local_rank, world_rank, world_size, cfg, run)

    if cfg.ckpt is not None:
        # run eval only
        if world_rank == 0:
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=True)
                for file in cfg.ckpt
            ]
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
            # runner.eval(step=ckpts[0]["step"])
            # runner.render_traj(step=ckpts[0]["step"])
            runner.export_mesh()
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    wandb.setup()
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
