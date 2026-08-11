# 3D Gaussian Flats Door-Plane Baseline

## Purpose

This bridge runs the released [3D Gaussian Flats](https://github.com/theialab/3dgs-flats)
pipeline on the existing OMeGa COLMAP scene and interactive segmentation. It
tests one controlled hybrid reconstruction:

```text
all persistent-region masks retained as semantic provenance
Door (persistent region 6) exposed as the only plane-mask track
        -> released hybrid 2D/3D Gaussian training
        -> RGB/depth and planar debug renders
        -> full TSDF mesh and explicit Door planar mesh
```

The upstream repository is kept unmodified at `third_party/3dgs-flats`. The
adapter lives in
`omega_local/reconstruction/backends/gaussian_flats/` and invokes the released
`train_planar.py`, `render.py`, and `render_planar.py` entry points.

Semantic masks other than Door still supervise ordinary RGB reconstruction
only indirectly through their RGB pixels. They are not declared planar and
therefore remain represented by unconstrained 3D Gaussians. A semantic region
mask does not force all its geometry into one plane unless it is explicitly
exported under `plane_masks/<plane-id>/`.

## Inputs

- RGB, poses, intrinsics, and SfM initializer:
  `<omega_stable_mesh>/dataset/{images,sparse}`
- Persistent-region labels:
  `<interactive>/proposals/propagation/sam2_video/label_maps`
- Region identity and name:
  `<interactive>/regions/regions.json`

Preparation resolves `Door` by name to persistent region ID `6`, writes binary
Door masks using the exact COLMAP image stems, and validates the released
loader's constraints. The current data has 197 nonempty Door masks; 182 exceed
the upstream exclusive `128 * 128` area cutoff, well above its 11-view minimum.

## Paper Settings

The default run uses the paper's 30,000-iteration schedule and explicitly sets:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `plane_fit_iter` | 3,500 | 3D-only warm-up before fitting planes |
| `plane_fit_min_points` | 100 | minimum accepted plane inliers |
| `plane_sigma_res` | 0.01 | perpendicular relocation sigma |
| `plane_sigma_dist` | 0.3 | in-plane relocation sigma |
| `planar_mask_loss_weight` | 0.1 | rendered plane-mask loss |
| `depthtv_loss_weight` | 0.1 | depth total-variation regularizer |
| `scale_reg` | 0.01 | Gaussian scale regularizer |
| `opacity_reg` | 0.01 | Gaussian opacity regularizer |

`cap_max=1,000,000` is a memory-safe scene setting for the 32 GB GPU used by
this pipeline, not a universal paper constant. The released DSLR configurations
choose a different cap per scene. A 2,000,000 cap exhausted GPU memory in the
planar KD-tree on this capture; lowering image resolution would not directly
bound that operation.
The source images and masks are already at the pipeline's common 1024-pixel
working resolution, so the adapter uses `resolution=1` and does not downsample
them again.

Mesh extraction uses a `0.02` scene-unit TSDF voxel and planar grid by default.
For this metric-aligned capture that is approximately 2 cm. The released renderer.s
automatic value was about 9.2 mm and produced an unnecessarily large mesh for
this scene.

## Commands

One-time isolated native runtime setup:

```bash
cd /home/yz2332/projects/digitalTwin

third_party/OMeGa_4_Building/scripts/setup_gaussian_flats_env.sh
```

Run the four inspectable stages:

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export OMEGA_BUILDING_ROOT=$DT_ROOT/third_party/OMeGa_4_Building
export PYTHON=$DT_ROOT/.venv/bin/python
export OMEGA_RESULT_DIR=$DT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/omega_stable_mesh/model_baseline_stronger_30000

$PYTHON $OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_flats.py \
  --stage prepare \
  --model-dir $OMEGA_RESULT_DIR \
  --planar-region Door

$PYTHON $OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_flats.py \
  --stage train \
  --model-dir $OMEGA_RESULT_DIR \
  --planar-region Door

$PYTHON $OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_flats.py \
  --stage render \
  --model-dir $OMEGA_RESULT_DIR \
  --planar-region Door

$PYTHON $OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_flats.py \
  --stage mesh \
  --model-dir $OMEGA_RESULT_DIR \
  --planar-region Door
```

The same pipeline can be launched with `--stage all`. Completed stages are
cached. Training is deliberately final-only: it evaluates and exports at
iteration 30,000 without writing resume checkpoints or intermediate models. If
training is interrupted, rerun the same command; the adapter discards the
partial model and starts a clean train while keeping prepared inputs. A
successful full run removes transient logs and training progress automatically.

## Interactive Visualization

Open the editor and use `Prepare Data > Gaussian Flats`. The run card reports
the four pipeline stages and live training iteration. Results appear as soon as
their source stage is complete:

- **Hybrid RGB · Flat / Free 3D** loads one hybrid scene. The left inspector
  toggles the exact Door-flat and unconstrained 3D Gaussian partitions. `RGB`
  shows learned appearance; `Regions` colors the two representations for
  diagnosis.
- **Full + Flat Meshes** loads the full TSDF surface and explicit Door planar
  surface as two mesh parts. Toggle either part in the left inspector and use
  `RGB` or `Regions` coloring. The meshes overlap by design when both are on.

The viewer partitions the fused world-space `point_cloud.ply` with the
released model's per-Gaussian `plane_ids`; it does not infer planarity again
from the input Door mask. This makes the flat toggle an exact view of training.

The editor is available at `http://127.0.0.1:8787` and through Tailscale at
`http://100.66.11.29:8787`.

## Outputs

The run is stored under:

```text
<interactive>/reconstruction/runs/gaussian_flats/door_planar_official/
  01_input/
    scene/                 # links to immutable RGB and COLMAP data
    semantic_label_maps/   # link to all persistent-region labels
    persistent_regions.json
    propagation_config.json
    plane_masks/0/         # Door-only binary plane track
    frames.jsonl
    planes.json
  02_model/                # final released model, renders, and meshes
  03_outputs/              # stable links to final useful artifacts
  stages/                   # compact completed-stage metadata
```

Important final artifacts are the full hybrid `point_cloud.ply`,
`point_cloud_planar.ply`, fitted `planes.json`, `plane_to_mask_id_30000.json`,
the complete `fuse_post.ply` TSDF mesh, and `planar_mesh.obj`. The regular
hybrid model contains both Door-bound 2D Gaussians and unconstrained 3D
Gaussians. `point_cloud_planar.ply` stores the released optimizer's complete
plane-local training state; the adapter derives the actual flat and free-3D
subsets from the fused `point_cloud.ply` plus `planes.json/plane_ids` under
`03_outputs/gaussians/`.
