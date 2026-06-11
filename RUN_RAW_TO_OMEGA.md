# Raw Captures To OMeGa And 2DGS

This runbook keeps the main pipeline compact:

1. Process the iPhone capture.
2. Run feed-forward priors on the iPhone capture: MASt3R and MapAnything.
3. Train OMeGa on iPhone with PromptDA normals.
4. Train OMeGa on iPhone with StableNormal normals.
5. Train 2DGS on iPhone.
6. Run the same style of pipeline on RGB camera data, when a DSLR capture is available.

## 0. Variables

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export PYTHON="$DT_ROOT/.venv/bin/python"

export RAW_CAPTURE="$DT_ROOT/data/raw_runs/grove_entrance_0521"
export CAPTURE_NAME="$(basename "$RAW_CAPTURE")"
export OUT_ROOT="$DT_ROOT/data/scan_processing_outputs"
export PACKAGE_ROOT="$OUT_ROOT/$CAPTURE_NAME"

export COLMAP_BIN="$DT_ROOT/third_party/colmap_cuda/install/bin/colmap"
export PROMPTDA_ROOT="$DT_ROOT/third_party/PromptDA"
export PROMPTDA_CKPT=depth-anything/prompt-depth-anything-vitl
export MAST3R_ROOT="$DT_ROOT/third_party/mast3r"
export STABLENORMAL_ROOT="$DT_ROOT/third_party/StableNormal"
export TWO_DGS_ROOT="$DT_ROOT/third_party/2d-gaussian-splatting"
export OMEGA_BUILDING_ROOT="$DT_ROOT/third_party/OMeGa_4_Building"

export WORKING_SIZE=1024
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Mesh preview frame index is zero-based: 0 = first frame, 3 = fourth frame.
export OMEGA_MESH_PREVIEW_FRAME=0
```

Optional DSLR branch:

```bash
export DSLR_CAPTURE="$DT_ROOT/data/camera/grove_entrance_dslr_0521"
export DSLR_NAME="$(basename "$DSLR_CAPTURE")"
export DSLR_PACKAGE_ROOT="$OUT_ROOT/$DSLR_NAME"
export DSLR_PINHOLE_NAME="${DSLR_NAME}_pinhole"
export DSLR_PINHOLE_PACKAGE_ROOT="$OUT_ROOT/$DSLR_PINHOLE_NAME"
```

For all visualization scripts that support frame sampling, this runbook uses
`--frame-stride 4 --max-frames 32`. The pose-alignment and PromptDA depth debug
scripts currently render their own fixed summaries, so they are left unchanged.

## 1. Process The iPhone Capture

Step `00` extracts RGB/depth/confidence files and converts ARKit poses to the
project OpenCV `world_T_cam` convention.

```bash
$PYTHON "$DT_ROOT/scan_processing/00_extract_capture_data.py" \
  "$RAW_CAPTURE" \
  --out-root "$OUT_ROOT" \
  --overwrite
```

Step `01` runs sparse COLMAP with ARKit pose priors. Step `02` aligns the
registered COLMAP poses back into the metric ARKit frame and writes the final
pose manifest used by later steps.

```bash
$PYTHON "$DT_ROOT/scan_processing/01_run_phone_colmap_sparse.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --colmap-bin "$COLMAP_BIN" \
  --matcher sequential \
  --sequential-overlap 40 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/02_finalize_phone_poses.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --overwrite
```

Check the ARKit/COLMAP alignment:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis03_visualize_phone_pose_alignment.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --overwrite
```

Run PromptDA after pose finalization. For exterior captures, avoid clipping far
depth during inference; clamp depth later when exporting OMeGa inputs.

```bash
$PYTHON "$DT_ROOT/scan_processing/04_run_phone_promptda.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --promptda-root "$PROMPTDA_ROOT" \
  --checkpoint "$PROMPTDA_CKPT" \
  --max-image-size "$WORKING_SIZE" \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis01_visualize_phone_depth_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --overwrite
```

Key outputs:

```text
$PACKAGE_ROOT/cameras/poses/final_poses.jsonl
$PACKAGE_ROOT/cameras/depth_promptda/predictions.jsonl
```

## 2. Feed-Forward Priors On iPhone

### 2.1 MASt3R

Initialize MASt3R submodules once:

```bash
git -C "$MAST3R_ROOT" submodule update --init --recursive
```

Run MASt3R for the OMeGa VFM point-cloud initializer:

```bash
$PYTHON "$DT_ROOT/scan_processing/22_run_phone_mast3r.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --mast3r-root "$MAST3R_ROOT" \
  --model-id naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric \
  --scene-graph logwin \
  --winsize 4 \
  --min-conf-thr 1.5 \
  --voxel-size-m 0.01 \
  --remove-outliers \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis22_visualize_phone_mast3r_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

### 2.2 MapAnything

Run MapAnything with the finalized step-02 poses. For outdoor scenes, keep the
depth cap broad enough for the visible facade and entry area.

```bash
$PYTHON "$DT_ROOT/scan_processing/11_run_phone_mapanything.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-id facebook/map-anything \
  --max-image-size 1600 \
  --minibatch-size 1 \
  --apply-confidence-mask \
  --confidence-percentile 10 \
  --use-multiview-confidence \
  --voxel-size-m 0.01 \
  --max-depth-m 15.0 \
  --remove-outliers \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis06_visualize_phone_mapanything_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Key outputs:

```text
$PACKAGE_ROOT/pointcloud/mast3r/phone_mast3r_points.ply
$PACKAGE_ROOT/pointcloud/mapanything/phone_mapanything_points.ply
```

## 3. OMeGa On iPhone With PromptDA Normals

This is the preferred iPhone OMeGa route when PromptDA is guided by clean iPhone
LiDAR depth. It exports to the default `$PACKAGE_ROOT/omega` folder.

```bash
$PYTHON "$DT_ROOT/scan_processing/23_run_phone_omega.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --colmap-bin "$COLMAP_BIN" \
  --normal-source promptda \
  --max-normal-depth-m 15.0 \
  --point-cloud-source mast3r \
  --seed-voxel-size-m 0.03 \
  --max-init-mesh-points 250000 \
  --init-normal-radius-m 0.08 \
  --poisson-depth 8 \
  --poisson-density-quantile 0.08 \
  --init-mesh-cleaning pymeshlab \
  --init-mesh-min-component-faces 100 \
  --init-mesh-close-holes-size 80 \
  --overwrite
```

Preview the exact mesh and normal maps that OMeGa will use:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24b_visualize_phone_omega_input_normals.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name omega \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24_visualize_phone_omega_mesh_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name omega \
  --mesh-source init \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Train OMeGa in two steps. The warmup checkpoint is the reusable base for later custom branches; the second command is the proper 30k run.

```bash
export PACKAGE_ROOT="$OUT_ROOT/$CAPTURE_NAME"
export OMEGA_MESH_PREVIEW_FRAME=0

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_warmup.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/warmup_2999.json" \
  --mesh-preview-frame "$OMEGA_MESH_PREVIEW_FRAME"
```

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/baseline_stronger_from_warmup_30000.json" \
  --set args.mesh_preview_frame="$OMEGA_MESH_PREVIEW_FRAME"
```

Key outputs:

```text
$PACKAGE_ROOT/omega/dataset/
$PACKAGE_ROOT/omega/init_mesh.ply
$PACKAGE_ROOT/omega/model_warmup_2999/
$PACKAGE_ROOT/omega/model_baseline_stronger_30000/
```

## 4. OMeGa On iPhone With StableNormal Normals

Use this branch to compare StableNormal against PromptDA-derived normals while
keeping the same iPhone poses. The StableNormal script name says `dslr` for
historical reasons, but it reads any package with `final_poses.jsonl`.

```bash
$PYTHON "$DT_ROOT/scan_processing/D05_run_dslr_stablenormal.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --stable-normal-root "$STABLENORMAL_ROOT" \
  --variant stable \
  --yoso-version yoso-normal-v0-3 \
  --processing-resolution 1536 \
  --num-inference-steps 10 \
  --ensemble-size 1 \
  --batch-size 1 \
  --data-type outdoor \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/VisD05_visualize_dslr_stablenormal_normals.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Export a separate OMeGa package under `$PACKAGE_ROOT/omega_stable_mesh`. This
keeps the PromptDA branch untouched.

```bash
$PYTHON "$DT_ROOT/scan_processing/23_run_phone_omega.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --omega-output-name omega_stable_mesh \
  --colmap-bin "$COLMAP_BIN" \
  --normal-source stablenormal \
  --point-cloud-source mast3r \
  --seed-voxel-size-m 0.04 \
  --max-init-mesh-points 220000 \
  --init-normal-radius-m 0.10 \
  --poisson-depth 8 \
  --poisson-density-quantile 0.10 \
  --init-mesh-cleaning pymeshlab \
  --init-mesh-min-component-faces 150 \
  --init-mesh-close-holes-size 100 \
  --overwrite
```

Preview the OMeGa StableNormal inputs:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24b_visualize_phone_omega_input_normals.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name omega_stable_mesh \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24c_compare_phone_omega_input_normals.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --reference-omega-output-name omega \
  --candidate-omega-output-name omega_stable_mesh \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24_visualize_phone_omega_mesh_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name omega_stable_mesh \
  --mesh-source init \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Train OMeGa in two steps. The warmup checkpoint is the reusable base for later custom branches; the second command is the proper 30k run from that warmup.

```bash
export PACKAGE_ROOT="$OUT_ROOT/$CAPTURE_NAME"
export OMEGA_MESH_PREVIEW_FRAME=0

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_warmup.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/warmup_stable_mesh_2999.json" \
  --mesh-preview-frame "$OMEGA_MESH_PREVIEW_FRAME"
```

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/baseline_stable_mesh_stronger_from_warmup_30000.json" \
  --set args.mesh_preview_frame="$OMEGA_MESH_PREVIEW_FRAME"
```

Key outputs:

```text
$PACKAGE_ROOT/cameras/normal_stablenormal/
$PACKAGE_ROOT/omega_stable_mesh/dataset/
$PACKAGE_ROOT/omega_stable_mesh/init_mesh.ply
$PACKAGE_ROOT/omega_stable_mesh/model_warmup_2999/
$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000/
```

Later OMeGa experiments, including custom remeshing branches, should resume from
the warmup checkpoint written by the first command rather than changing the data
export step.

## 5. 2DGS On iPhone

Train 2DGS from the finalized phone poses. The `auto` seed prefers COLMAP dense
plus merged fill when available; otherwise choose a specific point source.

```bash
$PYTHON "$DT_ROOT/scan_processing/16_run_phone_2dgs.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --two-dgs-root "$TWO_DGS_ROOT" \
  --point-cloud-source auto \
  --seed-voxel-size-m 0.01 \
  --run-training \
  --iterations 30000 \
  --save-iterations 30000 \
  --run-render \
  --unbounded \
  --observed-only-tsdf \
  --observed-tsdf-min-observations 2 \
  --mesh-res 512 \
  --overwrite
```

If the phone run has no reliable dense COLMAP cloud, use the PromptDA voxel seed:

```bash
$PYTHON "$DT_ROOT/scan_processing/16_run_phone_2dgs.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --two-dgs-root "$TWO_DGS_ROOT" \
  --point-cloud-source promptda_voxel \
  --seed-voxel-size-m 0.01 \
  --max-seed-points 300000 \
  --densify-until-iter 7000 \
  --densification-interval 300 \
  --densify-grad-threshold 0.0005 \
  --run-training \
  --iterations 30000 \
  --save-iterations 30000 \
  --run-render \
  --unbounded \
  --observed-only-tsdf \
  --observed-tsdf-min-observations 2 \
  --mesh-res 512 \
  --overwrite
```

Preview renders and mesh:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis11_visualize_phone_2dgs_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --two-dgs-root "$TWO_DGS_ROOT" \
  --iteration -1 \
  --resolution -1 \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis12_visualize_phone_2dgs_mesh_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --mesh-source auto \
  --render-backend triangle \
  --triangle-max-faces 0 \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Key outputs:

```text
$PACKAGE_ROOT/2d_gaussian_splatting/model/
$PACKAGE_ROOT/pointcloud/2d_gaussian_splatting/phone_2dgs_gaussians.ply
$PACKAGE_ROOT/pointcloud/2d_gaussian_splatting/phone_2dgs_mesh_post.ply
```

## 6. Same Pipeline On RGB Camera Data

The DSLR capture is RGB-only: no ARKit poses, no LiDAR depth, and no clean
PromptDA metric prompt. The reliable route is:

1. Run DSLR-only COLMAP.
2. Align the whole DSLR COLMAP model to the phone COLMAP frame with one robust Sim(3).
3. Export an undistorted, rotated PINHOLE package.
4. Run MapAnything and StableNormal on the PINHOLE package.
5. Train OMeGa and 2DGS from the cleaned PINHOLE package.

### 6.1 Prepare And Register

```bash
$PYTHON "$DT_ROOT/scan_processing/D00_prepare_dslr_capture.py" \
  "$DSLR_CAPTURE" \
  --out-root "$OUT_ROOT" \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/D01_run_dslr_colmap_sparse.py" \
  "$DSLR_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --colmap-bin "$COLMAP_BIN" \
  --camera-model SIMPLE_RADIAL \
  --matcher sequential \
  --sequential-overlap 40 \
  --sequential-loop-detection \
  --max-image-size 4096 \
  --sift-max-num-features 32768 \
  --sift-peak-threshold 0.004 \
  --sift-domain-size-pooling \
  --guided-matching \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/D01b_align_dslr_colmap_to_phone.py" \
  "$DSLR_PACKAGE_ROOT" \
  --phone-package "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --colmap-bin "$COLMAP_BIN" \
  --dslr-camera-model SIMPLE_RADIAL \
  --anchor-matcher cross-pairs \
  --max-image-size 4096 \
  --sift-max-num-features 32768 \
  --sift-peak-threshold 0.004 \
  --sift-domain-size-pooling \
  --guided-matching \
  --abs-pose-max-error 10.0 \
  --abs-pose-min-num-inliers 80 \
  --min-anchor-observations 100 \
  --overwrite
```

Check the DSLR trajectory and phone-frame alignment:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisD01_visualize_dslr_phone_alignment.py" \
  "$DSLR_PACKAGE_ROOT" \
  --phone-package "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --max-sparse-points 50000 \
  --overwrite
```

Export the clean PINHOLE package. The 180-degree rotation is baked into both
images and poses.

```bash
$PYTHON "$DT_ROOT/scan_processing/D01c_export_dslr_pinhole_package.py" \
  "$DSLR_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --output-name "$DSLR_PINHOLE_NAME" \
  --max-output-size 4096 \
  --undistort-alpha 0.0 \
  --output-rotation 180 \
  --overwrite
```

Key output:

```text
$DSLR_PINHOLE_PACKAGE_ROOT/cameras/poses/final_poses.jsonl
$DSLR_PINHOLE_PACKAGE_ROOT/pointcloud/colmap_sparse/colmap_sparse_points.ply
```

### 6.2 Feed-Forward Priors

```bash
$PYTHON "$DT_ROOT/scan_processing/11_run_phone_mapanything.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --max-image-size 1600 \
  --minibatch-size 1 \
  --apply-confidence-mask \
  --confidence-percentile 10 \
  --use-multiview-confidence \
  --voxel-size-m 0.01 \
  --max-depth-m 15.0 \
  --remove-outliers \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/D05_run_dslr_stablenormal.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --stable-normal-root "$STABLENORMAL_ROOT" \
  --variant stable \
  --yoso-version yoso-normal-v0-3 \
  --processing-resolution 1536 \
  --num-inference-steps 10 \
  --ensemble-size 1 \
  --batch-size 1 \
  --data-type outdoor \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis06_visualize_phone_mapanything_debug.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/VisD05_visualize_dslr_stablenormal_normals.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

### 6.3 OMeGa On DSLR With StableNormal

Export to `$DSLR_PINHOLE_PACKAGE_ROOT/omega_stable_mesh`. StableNormal supplies
normal maps; MapAnything supplies the coarse point/mesh initializer.

```bash
$PYTHON "$DT_ROOT/scan_processing/23_run_phone_omega.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --omega-output-name omega_stable_mesh \
  --colmap-bin "$COLMAP_BIN" \
  --normal-source stablenormal \
  --point-cloud-source mapanything \
  --seed-voxel-size-m 0.05 \
  --max-init-mesh-points 180000 \
  --init-normal-radius-m 0.12 \
  --poisson-depth 7 \
  --poisson-density-quantile 0.12 \
  --init-mesh-cleaning pymeshlab \
  --init-mesh-min-component-faces 200 \
  --init-mesh-close-holes-size 120 \
  --overwrite
```

Preview OMeGa inputs:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24b_visualize_phone_omega_input_normals.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name omega_stable_mesh \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24_visualize_phone_omega_mesh_debug.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name omega_stable_mesh \
  --mesh-source init \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Train OMeGa in two steps. The warmup checkpoint is the reusable base for later custom branches; the second command is the proper 30k run from that warmup.

```bash
export PACKAGE_ROOT="$DSLR_PINHOLE_PACKAGE_ROOT"
export OMEGA_MESH_PREVIEW_FRAME=0

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_warmup.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/warmup_stable_mesh_2999.json" \
  --mesh-preview-frame "$OMEGA_MESH_PREVIEW_FRAME"
```

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/baseline_stable_mesh_stronger_from_warmup_30000.json" \
  --set args.mesh_preview_frame="$OMEGA_MESH_PREVIEW_FRAME"
```

Later OMeGa experiments, including custom remeshing branches, should resume from
the warmup checkpoint written by the first command rather than changing the data
export step.

### 6.4 2DGS On DSLR

Train the 2DGS baseline from the cleaned PINHOLE COLMAP poses and sparse COLMAP
seed points.

```bash
$PYTHON "$DT_ROOT/scan_processing/D03_run_dslr_2dgs.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --two-dgs-root "$TWO_DGS_ROOT" \
  --point-cloud-source colmap_sparse \
  --seed-voxel-size-m 0.01 \
  --resolution -1 \
  --run-training \
  --iterations 30000 \
  --save-iterations 30000 \
  --overwrite
```

After training, export the mesh:

```bash
$PYTHON "$DT_ROOT/scan_processing/D03_run_dslr_2dgs.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --two-dgs-root "$TWO_DGS_ROOT" \
  --run-render \
  --depth-ratio 1.0 \
  --depth-trunc 10.0 \
  --voxel-size 0.02 \
  --sdf-trunc 0.08 \
  --num-cluster 10 \
  --overwrite
```

Preview renders and mesh:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis11_visualize_phone_2dgs_debug.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --two-dgs-root "$TWO_DGS_ROOT" \
  --iteration -1 \
  --resolution -1 \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis12_visualize_phone_2dgs_mesh_debug.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --mesh-source auto \
  --render-backend triangle \
  --triangle-max-faces 0 \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```


## 7. OMeGa Remesh Experiments

Remesh commands now live in `RUN_OMEGA_REMESH.md`. That file documents the
current DSLR StableNormal run under:

```text
/home/yz2332/projects/digitalTwin/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/omega_stable_mesh
```

It covers OMeGa data preparation, OMeGa warmup/baseline training, post-hoc
remesh, face-count sweeps, and mesh-evidence projection. The remesh and training
logic stays in this OMeGa checkout; visualization remains in `scan_processing/`.

## 8. Export And View OMeGa Splats

Set `PACKAGE_ROOT` and `OMEGA_RESULT_DIR` to the branch you want to inspect.

iPhone PromptDA example:

```bash
export PACKAGE_ROOT="$OUT_ROOT/$CAPTURE_NAME"
export OMEGA_RESULT_DIR="$PACKAGE_ROOT/omega/model_baseline_stronger_30000"
export OMEGA_DATASET_DIR="$PACKAGE_ROOT/omega/dataset"
```

iPhone StableNormal example:

```bash
export PACKAGE_ROOT="$OUT_ROOT/$CAPTURE_NAME"
export OMEGA_RESULT_DIR="$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000"
export OMEGA_DATASET_DIR="$PACKAGE_ROOT/omega_stable_mesh/dataset"
```

DSLR StableNormal example:

```bash
export PACKAGE_ROOT="$DSLR_PINHOLE_PACKAGE_ROOT"
export OMEGA_RESULT_DIR="$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000"
export OMEGA_DATASET_DIR="$PACKAGE_ROOT/omega_stable_mesh/dataset"
```

Export native OMeGa/2DGS surfel splats:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/export_omega_splats_to_ply.py" \
  "$OMEGA_RESULT_DIR" \
  --format 2dgs \
  --out "$OMEGA_RESULT_DIR/web_exports/omega_final_2dgs.ply" \
  --sh-degree 0
```

Host an exported OMeGa 2DGS-style PLY in the local Viser viewer:

```bash
$PYTHON "$DT_ROOT/third_party/2D-GS-Viser-Viewer/viewer.py" \
  "$OMEGA_RESULT_DIR/web_exports/omega_final_2dgs.ply" \
  -s "$OMEGA_DATASET_DIR" \
  --host 0.0.0.0 \
  --port 8080 \
  --reorient disable \
  --sh_degree 0 \
  --cameras-json "$PACKAGE_ROOT/2d_gaussian_splatting/model/cameras.json" \
  --show_cameras \
  --no_edit_panel
```

Host the trained 2DGS baseline:

```bash
$PYTHON "$DT_ROOT/third_party/2D-GS-Viser-Viewer/viewer.py" \
  "$PACKAGE_ROOT/2d_gaussian_splatting/model" \
  -s "$PACKAGE_ROOT/2d_gaussian_splatting/dataset" \
  --iterations 30000 \
  --host 0.0.0.0 \
  --port 8080 \
  --reorient disable \
  --show_cameras \
  --no_edit_panel
```

Only run one viewer on port `8080` at a time. Then open:

```text
http://<ubuntu-ip>:8080
```
