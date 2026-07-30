# Run OMeGa Segmentation

This file keeps only the active segmentation pipeline:

```text
common 2D proposal initializer
  -> SAM2Object on the common source
  -> SAI3D on the common source with proposal-gated view-mask refinement
  -> SAI3D point-cloud variants for MapAnything and COLMAP evidence
  -> staged released Split&Splat and anchored-mask experiments
```

## 1. Environment

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export PYTHON=$DT_ROOT/.venv/bin/python
export OMEGA_BUILDING_ROOT=$DT_ROOT/third_party/OMeGa_4_Building
export SAM2OBJECT_ROOT=$DT_ROOT/third_party/SAM2Object
export SAI3D_ROOT=$DT_ROOT/third_party/SAI3D_DT
export PACKAGE_ROOT=$DT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole
export OMEGA_RESULT_DIR=$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000
export OUT_ROOT=$DT_ROOT/data/scan_processing_outputs
export SAM2_CHECKPOINT=$DT_ROOT/third_party/sam2/checkpoints/sam2.1_hiera_large.pt
export PROPOSAL_NAME=view_proposals_1024
export MAPANYTHING_POINTS=$PACKAGE_ROOT/pointcloud/mapanything/phone_mapanything_points.ply
export COLMAP_POINTS=$PACKAGE_ROOT/pointcloud/colmap_sparse/colmap_sparse_points.ply
export XMEM_ROOT=$DT_ROOT/third_party/XMem2
export CUTIE_ROOT=$DT_ROOT/third_party/Cutie
export V2SAM_ROOT=$DT_ROOT/third_party/V2-SAM
export V2SAM_PYTHON=$HOME/miniconda3/envs/v2sam-dt/bin/python
export VGGT_S_ROOT=$DT_ROOT/third_party/VGGT-S
export VGGT_S_PYTHON=$DT_ROOT/.venv/bin/python
```

Resolution convention:

- the staged proposal source defines the pixel grid for interactive
  segmentation;
- the current recommended run uses `1024` px width, so SAM2 proposals, Step 3
  SAM2 masks, SAM2 video propagation, StableNormal evidence, Depth Anything V2
  evidence, proposal overlays, and selected pixels all live at `1024 x 683`;
- to test full DSLR resolution, use new output names such as
  `view_proposals_fullres` and `sai3d_area_samples_fullres_dense`, and set
  `--image-width 0` / `--proposal-image-width 0`; do not mix those outputs with
  the 1024 run;
- SAM2 itself still uses its pretrained 1024-square image encoder internally and
  upsamples masks back to the input frame grid, so full-res mainly improves
  output-mask alignment and raster precision rather than changing SAM2's encoder
  feature resolution.

Install lightweight dependencies if needed:

```bash
$PYTHON -m pip install open3d natsort matplotlib tqdm progressbar2 opencv-python scipy scikit-image PyMaxflow plyfile supervision hydra-core iopath
```

## 2. Common 2D Proposal Initializer

This stages the DSLR frames once at 1024px width and creates independent SAM2
automatic proposal masks for every view. Set `--image-width 0` and use a new
`--baseline-name` to keep original DSLR resolution.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline view_proposals \
  --stage all \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name "$PROPOSAL_NAME" \
  --capture-root "$PACKAGE_ROOT" \
  --image-width 1024 \
  --frame-stride 1 \
  --max-frames 0 \
  --sam2-root "$DT_ROOT/third_party/sam2" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --device cuda:0 \
  --points-per-side 64 \
  --max-masks-per-frame 120 \
  --overwrite
```

Visualize the initializer:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega08_visualize_view_proposals.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name "$PROPOSAL_NAME" \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 560 \
  --overwrite
```

Inspect the overlay, label map, and boundary map. This source has no cross-view
identity; it is local 2D proposal evidence for downstream methods.

## 3. SAM2Object On The Common Source

SAM2Object consumes `view_proposals_1024` for the ordered RGB/pose sequence,
then runs its own SAM2 keyframe segmentation, forward/reverse tracking, merge,
and mesh-graph consistency.

Prepare SAM2Object's video layout:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sam2object \
  --stage prepare \
  --model-dir "$OMEGA_RESULT_DIR" \
  --proposal-source-name "$PROPOSAL_NAME" \
  --source-frame-stride 1 \
  --max-frames 0 \
  --python "$PYTHON" \
  --overwrite
```

Run tracking and merge:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sam2object \
  --stage track \
  --model-dir "$OMEGA_RESULT_DIR" \
  --sam2object-root "$SAM2OBJECT_ROOT" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --python "$PYTHON" \
  --device cuda:0 \
  --keyframe-mode fixed \
  --keyframe-stride 10 \
  --max-masks-per-keyframe 80 \
  --overwrite
```

Run graph consolidation:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sam2object \
  --stage consolidate \
  --model-dir "$OMEGA_RESULT_DIR" \
  --python "$PYTHON" \
  --graph-from-points-thres 0.9 \
  --graph-thres-connect 0.9,0.3,5 \
  --graph-thres-merge 200 \
  --overwrite
```

Visualize:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega03_visualize_sam2object_baseline.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 520 \
  --overwrite
```

Inspect forward/reverse masks, merged masks, consistent masks, and
`graph_consistency/labeled_mesh.ply`.

## 4. SAI3D On The Common Source

SAI3D consumes `view_proposals_1024` masks, renders OMeGa mesh depth for
visibility, builds superpoints over mesh vertices or area-balanced surface
samples, and runs SAI3D-style multi-view graph merging. The final view-mask step
is a SAI3D finalizer inspired by Split&Splat: project the consistent SAI3D 3D
labels as a virtual mask, assign automatic SAM2 proposals by depth-visible 3D
support, and use SAM2 point prompting only as a fallback for visible objects
without a trusted proposal.

Clean the old default SAI3D run if you want a fresh replacement:

```bash
rm -rf "$OMEGA_RESULT_DIR/segmentation/baselines/sai3d"
```

Prepare the SAI3D RGB-D/mask dataset:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage prepare \
  --model-dir "$OMEGA_RESULT_DIR" \
  --proposal-source-name "$PROPOSAL_NAME" \
  --source-frame-stride 1 \
  --max-frames 0 \
  --overwrite
```

Observe the vertex-graph inputs:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage observe \
  --model-dir "$OMEGA_RESULT_DIR" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source mesh_vertices \
  --superpoint-mode voxel \
  --superpoint-target-count 8000 \
  --overwrite
```

Segment the vertex graph. This produces the actual SAI3D 3D labels:
`mesh_labels/point_labels.npy`, `mesh_labels/vertex_labels.npy`,
`mesh_labels/face_labels.npy`, `mesh_labels/labeled_points.ply`, and
`mesh_labels/labeled_mesh.ply`.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage segment \
  --model-dir "$OMEGA_RESULT_DIR" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source mesh_vertices \
  --superpoint-mode voxel \
  --superpoint-target-count 8000 \
  --thres-connect 0.9,0.5,5 \
  --thres-merge 200 \
  --overwrite
```

Project the saved SAI3D labels to views without SAM2. This is the clean
"SAI3D 3D labels only" comparison:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage view_masks \
  --model-dir "$OMEGA_RESULT_DIR" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source mesh_vertices \
  --view-mask-refine-mode geometry \
  --overwrite
```

Visualize the projected-only result:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega04_visualize_sai3d_baseline.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 500 \
  --output-dir "$PACKAGE_ROOT/visualizations/omega_segmentation/sai3d/${OMEGA_RESULT_DIR##*/}_sai3d_projected_only" \
  --overwrite
```

Then rerun only the final SAI3D proposal-gated refinement step:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage view_masks \
  --model-dir "$OMEGA_RESULT_DIR" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source mesh_vertices \
  --view-mask-refine-mode split_splat \
  --view-mask-sam2-root "$DT_ROOT/third_party/sam2" \
  --view-mask-sam2-checkpoint "$SAM2_CHECKPOINT" \
  --view-mask-sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --view-mask-sam2-prompt-count 10 \
  --view-mask-prompt-erode-px 5 \
  --view-mask-sam2-min-iou 0.28 \
  --view-mask-sam2-min-geometry-recall 0.35 \
  --overwrite
```

Visualize observation evidence and the proposal-gated final segmentation:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega05_visualize_sai3d_observe.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 500 \
  --output-dir "$PACKAGE_ROOT/visualizations/omega_segmentation/sai3d_observe/${OMEGA_RESULT_DIR##*/}_sai3d" \
  --overwrite

$PYTHON "$DT_ROOT/scan_processing/VisSegOmega04_visualize_sai3d_baseline.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 500 \
  --output-dir "$PACKAGE_ROOT/visualizations/omega_segmentation/sai3d/${OMEGA_RESULT_DIR##*/}_sai3d_split_splat" \
  --overwrite
```

## 5. SAI3D Dense Area-Sampled 1024 Pipeline

This is the current recommended SAI3D mesh run. It keeps the 1024px common
SAM2 proposal source, uses triangle-area-balanced OMeGa mesh samples instead of
raw mesh vertices, raises the superpoint target, runs SAI3D observe/segment/view
masks, and writes both observe and final debug visualizations. Use
`--proposal-image-width 0` with a new baseline/proposal name for a full-res run.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_sai3d_pipeline.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --capture-root "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --python "$PYTHON" \
  --baseline-name sai3d_area_samples_1024_dense \
  --proposal-name "$PROPOSAL_NAME" \
  --sai3d-root "$SAI3D_ROOT" \
  --sam2-root "$DT_ROOT/third_party/sam2" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --proposal-image-width 1024 \
  --point-source mesh_area_samples \
  --point-sample-count 160000 \
  --point-sample-seed 71 \
  --superpoint-mode voxel \
  --superpoint-target-count 30000 \
  --thres-connect 0.9,0.5,5 \
  --thres-merge 200 \
  --view-mask-refine-mode split_splat \
  --visualization-frame-stride 4 \
  --visualization-max-frames 32 \
  --visualization-panel-cell-width 420 \
  --visualization-contact-cols 1 \
  --clean-baseline \
  --overwrite
```

The wrapper reuses `$PROPOSAL_NAME` if it already exists. Add
`--regenerate-proposals` only when you want to rebuild the 1024 SAM2 proposals.

Full-resolution variant for final-quality tests:

```bash
export FULL_PROPOSAL_NAME=view_proposals_fullres
export FULL_BASELINE_NAME=sai3d_area_samples_fullres_dense

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_sai3d_pipeline.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --capture-root "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --python "$PYTHON" \
  --baseline-name "$FULL_BASELINE_NAME" \
  --proposal-name "$FULL_PROPOSAL_NAME" \
  --sai3d-root "$SAI3D_ROOT" \
  --sam2-root "$DT_ROOT/third_party/sam2" \
  --sam2-checkpoint "$SAM2_CHECKPOINT" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --proposal-image-width 0 \
  --point-source mesh_area_samples \
  --point-sample-count 160000 \
  --point-sample-seed 71 \
  --superpoint-mode voxel \
  --superpoint-target-count 30000 \
  --thres-connect 0.9,0.5,5 \
  --thres-merge 200 \
  --view-mask-refine-mode split_splat \
  --skip-visualization \
  --clean-baseline \
  --overwrite
```

Then launch the editor with `--baseline-name "$FULL_BASELINE_NAME"`. In that
run, all interactive masks and evidence are stored on the original DSLR frame
grid.

If this SAI3D baseline already has `mesh_labels/point_labels.npy` and
`mesh_labels/face_labels.npy`, rerun only the new final mask stage:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage view_masks \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_area_samples_1024_dense \
  --proposal-source-name "$PROPOSAL_NAME" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source mesh_area_samples \
  --point-sample-count 160000 \
  --point-sample-seed 71 \
  --view-mask-refine-mode split_splat \
  --view-mask-sam2-root "$DT_ROOT/third_party/sam2" \
  --view-mask-sam2-checkpoint "$SAM2_CHECKPOINT" \
  --view-mask-sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --overwrite
```

Inspect:

- `visualizations/omega_segmentation/sai3d_observe/...`: superpoints, rendered
  depth visibility, sampled 2D labels, and support.
- `visualizations/omega_segmentation/sai3d/...`: full chain from input
  proposals to SAI3D 3D labels, accepted/rejected proposal assignments,
  projected geometry support, SAM2 fallback prompts/candidates, final masks,
  and boundaries.
- `mesh_labels/labeled_points.ply`: direct area-sampled SAI3D 3D labels.
- `mesh_labels/labeled_mesh.ply`: transferred mesh labels.

## 6. SAI3D External Point-Cloud Tests

These variants keep the same 2D proposal source but replace the SAI3D graph
points with raw reconstruction point clouds. The OMeGa mesh is still staged for
rendered mesh outputs, but external point-cloud graph points use point z-buffer
visibility. Inspect `mesh_labels/labeled_points.ply` for the actual point-cloud
labels before judging the transferred mesh labels.

### 6.1 MapAnything Points

MapAnything points give broad coverage and good semantic segmentation, but the
cloud is noisy/layered and less self-consistent across views. Treat this as
useful broad evidence for 2D mask recovery rather than as a clean final 3D
segmentation.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage prepare \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_mapanything_points_1024 \
  --proposal-source-name "$PROPOSAL_NAME" \
  --source-frame-stride 1 \
  --max-frames 0 \
  --overwrite

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage segment \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_mapanything_points_1024 \
  --proposal-source-name "$PROPOSAL_NAME" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source point_cloud \
  --point-cloud "$MAPANYTHING_POINTS" \
  --point-cloud-max-points 300000 \
  --point-visibility-source point_zbuffer \
  --point-zbuffer-depth-band 0.02 \
  --superpoint-mode voxel \
  --superpoint-target-count 20000 \
  --thres-connect 0.9,0.5,5 \
  --thres-merge 200 \
  --view-mask-refine-mode split_splat \
  --view-mask-sam2-root "$DT_ROOT/third_party/sam2" \
  --view-mask-sam2-checkpoint "$SAM2_CHECKPOINT" \
  --view-mask-sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --overwrite
```

### 6.2 COLMAP Sparse Points

COLMAP sparse points cover key detailed regions and sample strong local visual
structure. The distribution is not a continuous surface, so the useful setting
is conservative: many small superpoints, short graph edges, strict merge
threshold, and no small-region post-merge. This produces a high-detail
oversegmentation that can be merged later.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage prepare \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_colmap_sparse_points_1024_sparse_graph \
  --proposal-source-name "$PROPOSAL_NAME" \
  --source-frame-stride 1 \
  --max-frames 0 \
  --overwrite

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline sai3d \
  --stage segment \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_colmap_sparse_points_1024_sparse_graph \
  --proposal-source-name "$PROPOSAL_NAME" \
  --sai3d-root "$SAI3D_ROOT" \
  --point-source point_cloud \
  --point-cloud "$COLMAP_POINTS" \
  --point-cloud-max-points 0 \
  --point-visibility-source point_zbuffer \
  --point-zbuffer-depth-band 0.02 \
  --superpoint-mode voxel \
  --superpoint-target-count 20000 \
  --graph-max-edge-length 0.35 \
  --max-neighbor-distance 1 \
  --thres-connect 0.9 \
  --thres-merge 0 \
  --view-mask-refine-mode split_splat \
  --view-mask-sam2-root "$DT_ROOT/third_party/sam2" \
  --view-mask-sam2-checkpoint "$SAM2_CHECKPOINT" \
  --view-mask-sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --overwrite
```

Visualize either point-cloud run:

```bash
for NAME in sai3d_mapanything_points_1024 sai3d_colmap_sparse_points_1024_sparse_graph; do
  $PYTHON "$DT_ROOT/scan_processing/VisSegOmega05_visualize_sai3d_observe.py" \
    "$PACKAGE_ROOT" \
    --out-root "$OUT_ROOT" \
    --model-dir "$OMEGA_RESULT_DIR" \
    --baseline-name "$NAME" \
    --frame-stride 4 \
    --max-frames 32 \
    --panel-cell-width 680 \
    --overwrite

  $PYTHON "$DT_ROOT/scan_processing/VisSegOmega04_visualize_sai3d_baseline.py" \
    "$PACKAGE_ROOT" \
    --out-root "$OUT_ROOT" \
    --model-dir "$OMEGA_RESULT_DIR" \
    --baseline-name "$NAME" \
    --frame-stride 4 \
    --max-frames 32 \
    --panel-cell-width 680 \
    --overwrite
done
```

## 7. Interactive Editor And Propagation Baselines

The editor registers seven first-stage anchor-to-mask methods: SAM2 video,
bounded SAM2, XMem++, Cutie, sparse COLMAP Tracks, OMeGa Initializer Points, and
OMeGa Final Points. They receive one
validated packet of complete-frame persistent-region maps and write the same
persistent-ID candidate format. Dense recovery is a separate second stage whose
three decoders consume either saved sparse point layer: `Superpixel Geodesic`,
the Xie et al. `2D/3D CRF` adaptation, and SAMPro3D-inspired `SAM2 Point
Prompts`. The same `Geometry Pass` control also exposes `COLMAP Verify Video`,
which reads SAM2 Video explicitly, preserves its mask pixels, and corrects only
component region IDs supported by reliable COLMAP tracks.
V2-SAM and VGGT-S are focused source-target diagnostics for one selected
region, not all-frame propagation methods. Methods never overwrite
user-confirmed region maps or each other's outputs.

The official repositories and checkpoints should exist at:

```text
$XMEM_ROOT/saves/XMem.pth
$CUTIE_ROOT/weights/cutie-base-mega.pth
$V2SAM_ROOT/weights/sam2/sam2_hiera_large.pt
$V2SAM_ROOT/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
$V2SAM_ROOT/weights/visual_ego2exo_full.pth
$V2SAM_ROOT/weights/fusion_ego2exo_full.pth
$VGGT_S_ROOT/official_ckpts/main_exp.pth
```

Launch the current 1024-grid editor:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_editor.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_area_samples_1024_dense \
  --host 0.0.0.0 \
  --port 8787 \
  --max-points 180000 \
  --label-source raw \
  --sai3d-root "$SAI3D_ROOT" \
  --xmem-root "$XMEM_ROOT" \
  --xmem-checkpoint "$XMEM_ROOT/saves/XMem.pth" \
  --cutie-root "$CUTIE_ROOT" \
  --cutie-checkpoint "$CUTIE_ROOT/weights/cutie-base-mega.pth" \
  --v2sam-root "$V2SAM_ROOT" \
  --v2sam-python "$V2SAM_PYTHON" \
  --v2sam-profile ego2exo \
  --vggts-root "$VGGT_S_ROOT" \
  --vggts-python "$VGGT_S_PYTHON" \
  --xmem-size 480 \
  --cutie-size 480 \
  --memory-vos-device cuda
```

Step 1 can display the aligned RGB COLMAP sparse cloud. `Cleaned (SOR)` switches
between the immutable raw cache and the derived Open3D `20`-neighbor,
`2.0`-standard-deviation cleanup; the cleaned NPZ, PLY, and JSON audit file are
generated automatically under `interactive/data/point_clouds/`. After
`COLMAP Tracks` runs, `Segmented` displays only voted SfM points using their
persistent-region colors; its NPZ, PLY, and JSON summary are saved beside the
raw and cleaned caches.

Step 1 calls this source `OMeGa Initialization -> Initializer Points`. It is the
exact feed-forward VFM cloud copied into the active OMeGa dataset. For this DSLR
StableNormal run, the source is MapAnything:

```text
$PACKAGE_ROOT/pointcloud/mapanything/phone_mapanything_points.ply  # 10,977,572 raw points
$PACKAGE_ROOT/omega_stable_mesh/dataset/vfm_sparse/0/points3D.ply # 367,740 points at 5 cm
$PACKAGE_ROOT/omega_stable_mesh/init_mesh.ply                     # Poisson/PyMeshLab init mesh
```

The VFM cloud is the exact point set consumed by OMeGa and is the default source
for the `OMeGa Initializer Points` geometry pass. Complete-frame region maps label visible points using
calibrated projection, a per-pixel point z-buffer, and cross-anchor majority
voting; ties remain unknown. `Region Labels` visualizes the resulting point
labels under the same initialization group. Its three dense methods use the
same second-stage contracts as COLMAP.
To test any other registered feed-forward cloud in the same world frame, launch
with `--feedforward-point-cloud /path/to/points.ply`; the optional matching mesh
can be recorded with `--feedforward-init-mesh /path/to/init_mesh.ply`.

Step 1 calls the final source `OMeGa Optimized Mesh -> Mesh Vertices`. It is a
separate reconstruction-backed source, discovered by default in the strongest
sibling run:

```text
$PACKAGE_ROOT/omega_stable_mesh/model_baseline_strongest_30000/plys/mesh_29999_rank0.ply
```

It keeps three point representations under
`interactive/data/point_clouds/`:

- `omega_final_vertices.{ply,npz,json}` contains every finite mesh vertex. It is
  useful for inspecting OMeGa's native detail-adaptive density.
- `omega_final_surface_completion.{npz,json}` stores area-sampled, voxelized
  candidates used only to fill sparse faces in the clean hybrid cloud.
- `omega_final_clean_hybrid.{ply,npz,json}` removes unreferenced vertices and
  negligible disconnected components, retains every remaining native vertex,
  then adds one on-surface sample only in occupied surface voxels without a
  retained vertex. Step 1 exposes this source as `Clean Hybrid`.

Run `OMeGa Final Points` to vote persistent IDs onto the clean hybrid cloud and
enable `Projected Regions`. The Geometry Pass then offers `Final Superpixels`,
`Final 2D/3D CRF`, and `Final SAM2 Prompts`. Override discovery only when needed
with `--omega-final-mesh /path/to/final_mesh.ply`.

Step 6 runs registered 3D segmentation methods on this strongest OMeGa
geometry. The first active experiment is `SAI3D + SAM2 Automatic`: it reuses the
214 Step 2 proposal label maps, thins the clean hybrid cloud to the selected
point budget, builds SAI3D voxel
superpoints, and runs the original progressive 3D grouping. The default
`400k points / 8k superpoints` follows SAI3D's original graph scale; `4k` and
`12k` provide coarse/fine comparisons. `All points` means the complete clean
hybrid cloud. Every completed run appears in the dedicated Step 1
`SAI3D Segmentation` card, labeled by its evidence source and anchor weight.
Each result is an independently toggleable point-cloud layer, so automatic and
propagated runs can be compared directly.
Outputs are isolated under:

```text
interactive/3d_segmentation/inputs/sam2_auto/
interactive/3d_segmentation/runs/sai3d_sam2_auto/
```

`Propagated + Manual Anchors` accepts any completed full-run propagation layer,
with `SAM2 Video` selected by default. Completed keyframes replace the propagated
mask with the user-confirmed persistent-region map. `Manual weight` controls how
strongly those views affect SAI3D (`2x`, `4x`, or `8x`; default `4x`) through
exact integer view replication. This preserves SAI3D's original affinity and
progressive grouping code while making manual evidence explicitly stronger.

The weighted run is isolated by source and weight, for example:

```text
interactive/3d_segmentation/inputs/propagated_weighted_sam2_video_w4/
interactive/3d_segmentation/runs/sai3d_propagated_weighted_sam2_video_w4/
```

In Step 5, mark edited frames complete, choose a full propagation method, and
click `Run`. The same selector controls `Preview`; every result is shown above
the original SAM2 anything proposals. For a focused transfer check, select a
persistent region, open a different target frame, choose `V2-SAM` or `VGGT-S`
in the `Pair Test` selector, and click `Test Pair`. The earliest manual frame
containing that region is the sole source. The test replaces only the target
frame in the chosen method's candidate layer and leaves every other frame and
method untouched. Step 3 creates its candidate-layer toggles from the same
registry. Saved methods use one layout:

```text
interactive/proposals/propagation/registry.json
interactive/proposals/propagation/sam2_video/
interactive/proposals/propagation/sam2_bounded/
interactive/proposals/propagation/xmem/
interactive/proposals/propagation/cutie/
interactive/proposals/propagation/colmap_tracks/
interactive/proposals/propagation/colmap_verify_video/
interactive/proposals/propagation/colmap_dense_superpixels/
interactive/proposals/propagation/colmap_dense_multifield_crf/
interactive/proposals/propagation/colmap_dense_sam2_prompts/
interactive/proposals/propagation/feedforward_points/
interactive/proposals/propagation/feedforward_dense_superpixels/
interactive/proposals/propagation/feedforward_dense_multifield_crf/
interactive/proposals/propagation/feedforward_dense_sam2_prompts/
interactive/proposals/propagation/omega_final_points/
interactive/proposals/propagation/omega_final_dense_superpixels/
interactive/proposals/propagation/omega_final_dense_multifield_crf/
interactive/proposals/propagation/omega_final_dense_sam2_prompts/
interactive/proposals/propagation/v2sam_pair/
interactive/proposals/propagation/vggts_pair/
interactive/view_evidence/dinov3_vitl16_768/
```

Each method directory contains `input.json`, `config.json`, `progress.json`,
`summary.json`, `frames.jsonl`, and matching `label_maps/`, `overlays/`, and
`metadata/` directories. A full propagation becomes stale whenever its manual
anchor maps change; source-dependent recovery and SAI3D inputs reject that
result until it is regenerated. Focused source-target pair diagnostics remain
independent because they are not complete dataset layers.

`COLMAP Tracks` automatically uses the retained original DSLR SfM model under
the sibling `grove_entrance_dslr_0521` capture. It rejects the OMeGa seed model
because that model has no feature observations. SfM keypoints are mapped to the
editor grid using the D01c pinhole export's recorded undistortion and pixel
rotation. Complete frames remain the original dense user masks; other frames
contain only sparse labeled track markers. This first test intentionally
performs no proposal or dense-mask recovery.

Use the separate Step 5 `Geometry Pass` row for source-dependent methods.
`COLMAP Verify Video` requires a current `Video` run and previews its verified
IDs above the source while preserving the source mask footprint. The other
methods require `COLMAP Tracks` or `OMeGa Initializer Points` first. `Superpixel
Geodesic` requires the Step 1 StableNormal and Depth Anything V2 caches and
fills local support using RGB/depth/normal
boundary costs. `SAM2 Point Prompts` requires SAM2, samples cleaned COLMAP
observations as positive prompts, and uses competing region observations as
negative prompts. `2D/3D CRF` additionally requires the segmented COLMAP point
cache produced by `COLMAP Tracks`; it jointly updates image-superpixel and
visible-point labels using 2D, 3D, and projection-coupling terms. Each preview
shows dense recovery above the exact sparse COLMAP source. Every dense method
reads that same source and saves an independent result, so decoder comparisons
never rerun or overwrite COLMAP track voting.

The 480 setting follows each memory-VOS repository's standard inference default
while all saved labels remain on the staged 1024-pixel frame grid. Use `-1` for
either size only when testing native-grid inference; it is substantially more
costly.

Step 1 owns DINOv3 evidence beside StableNormal and Depth Anything V2. Click
`Generate` to extract missing/stale feature frames or resume an interrupted
run; once complete, the control becomes `Regenerate`. V2-SAM checks the cache
signature before inference and does not load the DINOv3 model when all features
and PCA maps are current. If evidence is missing, V2-SAM can still fill the
cache as a safety fallback.

V2-SAM runs the official Anchor, Visual, and Fusion experts for one selected
manual source-target region pair and reimplements the paper's PCCS point-cycle
selection for that pair. DINOv3 ViT-L/16 uses the released V2-SAM setting of a
768-pixel feature height and a 16-pixel patch size. The cache also stores
dataset-consistent PCA images for evidence inspection. Learned Visual/Fusion
weights use the released `ego2exo` profile and should be evaluated as a
baseline rather than an architecture-tuned model.

VGGT-S uses the official [VGGT-S source](https://github.com/buaa-colalab/VGGT-S),
[paper](https://arxiv.org/abs/2604.13596), and released `main_exp.pth` Union
Segmentation Head. It samples 50 points from the complete source mask for the
initial VGGT target localization, removes the 10 median-distance outliers,
clusters five target prompts, conditions the head on the full cropped source
mask and pairwise VGGT features, and performs the released initial pass plus
one refinement. It saves the recovered 1024-grid mask directly. The checkpoint
was trained for Ego-Exo transfer, so this is a reproducible domain-transfer
baseline rather than an architecture-specialized model.

```text
interactive/view_evidence/dinov3_vitl16_768/features/   # float16 patch features
interactive/view_evidence/dinov3_vitl16_768/metadata/   # per-frame provenance
interactive/view_evidence/dinov3_vitl16_768/pca/        # viewable RGB PCA maps
interactive/view_evidence/dinov3_vitl16_768/summary.json
```

After a V2-SAM run finishes, choose `Image -> DINOv3 PCA` in the editor toolbar
to inspect the cached feature field on the active frame. The PCA basis is shared
across the dataset, so colors are comparable between views. DINOv3 ViT-L/16
produces one feature per 16-pixel patch: a 1024-by-683 staged image therefore
has only about a 71-by-48 feature grid after the official aspect-preserving
768-pixel-height resize. Enlarging that patch grid makes the PCA view look
blurry; it is a feature visualization, not a low-resolution RGB input or a
pixel-accurate mask. DINOv3 itself can accept larger divisible dimensions, but
V2-SAM fixes sparse correspondence to 768 in its released configuration.
Higher-resolution features should therefore be treated as a separate ablation,
not mixed into the official baseline. Point clouds,
proposal layers, selected pixels, and persistent-region overlays remain separate
viewer layers.


## Split&Splat Paper Baseline

The staged runner keeps the released Split&Splat algorithm isolated from the
editor dataset. Dataset geometry is cached independently from segmentation
proposals:

```text
interactive/experiments/split_splat/
  _shared/<dataset-cache-id>/   # staged input + global depth-regularized GS
  runs/<run-id>/                # proposal source + Split labels and masks
```

Runs with the same model, editor raster, depth source/path, and GS iteration
count derive the same cache ID automatically. The official-proposal and
SAM2-Video-proposal runs therefore reuse the global reconstruction.
Pass `--shared-id` only when an explicit stable cache name is useful.

Set the shared paths:

```bash
export PROJECT_ROOT=/home/yz2332/projects/digitalTwin
export OMEGA_BUILDING_ROOT=$PROJECT_ROOT/third_party/OMeGa_4_Building
export OMEGA_RESULT_DIR=$PROJECT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/omega_stable_mesh/model_baseline_stronger_30000
export SEG_BASELINE=sai3d_area_samples_1024_dense
export SPLIT_SPLAT_ROOT=$PROJECT_ROOT/third_party/Split_and_Splat
export SPLIT_SPLAT_PYTHON=$PROJECT_ROOT/.venv_split_splat/bin/python
```

Create or refresh the isolated environment without modifying `.venv`:

```bash
$OMEGA_BUILDING_ROOT/scripts/setup_split_splat_env.sh
```

Build the local [SuperSplat Viewer](https://github.com/playcanvas/supersplat-viewer)
once. The setup uses an existing Node.js installation when available, or places
an isolated Node.js toolchain under `$PROJECT_ROOT/.toolchains`:

```bash
$OMEGA_BUILDING_ROOT/scripts/setup_gaussian_viewer.sh
```

The status command must report `"available": true` before a long stage starts.
If it reports only `cudaRuntime` and `nvidia-smi` reports a driver/library
version mismatch, reboot once to load the newly installed NVIDIA kernel module.
This is a machine-driver state issue, not a Python package conflict.

The current editor has a complete 1024-pixel Depth Anything V2 cache, so these
commands are directly runnable:

```bash
SS_COMMON=(
  --model-dir $OMEGA_RESULT_DIR
  --editor-baseline-name $SEG_BASELINE
  --split-splat-root $SPLIT_SPLAT_ROOT
  --python $SPLIT_SPLAT_PYTHON
  --depth-source editor_depth
)
```

Check source, checkpoint, proposal source, and stage status without starting a
job:

```bash
$SPLIT_SPLAT_PYTHON $OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py \
  --stage status --run-id split_splat_official \
  --proposal-source official_auto "${SS_COMMON[@]}"
```

### Complete Two-Run Command

Run this block from `OMeGa_4_Building`. It prepares the shared dataset, trains
the global Gaussian scene once, and then runs Split with both proposal sources:

```bash
(
set -euo pipefail

# Shared dataset preparation.
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage prepare \
  --run-id split_splat_official \
  --proposal-source official_auto \
  "${SS_COMMON[@]}"

# Shared depth-regularized global 3DGS training.
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage global_gs \
  --run-id split_splat_official \
  --proposal-source official_auto \
  "${SS_COMMON[@]}"

# Experiment A: released four-grid SAM2 proposals.
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage proposals \
  --run-id split_splat_official \
  --proposal-source official_auto \
  "${SS_COMMON[@]}"

"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage split \
  --run-id split_splat_official \
  --proposal-source official_auto \
  "${SS_COMMON[@]}"

# Experiment B: our SAM2 Video propagated proposals.
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage proposals \
  --run-id split_splat_sam2_video \
  --proposal-source propagation \
  --propagation-method sam2_video \
  "${SS_COMMON[@]}"

"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage split \
  --run-id split_splat_sam2_video \
  --proposal-source propagation \
  --propagation-method sam2_video \
  "${SS_COMMON[@]}"
)
```

The parentheses keep strict error handling inside a subshell. A failed stage
therefore stops this block without closing the interactive terminal that
launched it.

Both runs derive the same shared cache ID and therefore reuse the staged
dataset and 30,000-iteration global 3DGS. `proposals` either runs the released
1/4/8/16-point SAM2 grids or converts the existing exclusive SAM2 Video label
maps to Split&Splat binary masks. `split` is the same released
depth-visible 3D label-voting code in both experiments. Source persistent
region IDs are recorded in `source_mappings/`; final Split IDs remain
run-local.

### Anchored 3D Split Adaptation

Experiment C keeps the editor's persistent IDs and mask boundaries instead of
asking released Split to rediscover instances and recover masks with SAM2. It
reuses the same shared global 3DGS:

```bash
(
set -euo pipefail

"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage proposals \
  --run-id split_splat_anchored_sam2_video \
  --proposal-source propagation \
  --propagation-method sam2_video \
  --split-method anchored_3d \
  --manual-frame-weight 8 \
  "${SS_COMMON[@]}"

"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage split \
  --run-id split_splat_anchored_sam2_video \
  --proposal-source propagation \
  --propagation-method sam2_video \
  --split-method anchored_3d \
  --manual-frame-weight 8 \
  "${SS_COMMON[@]}"
)
```

For global Gaussian mean \(g\) and persistent region \(r\), the method
accumulates observations after z-buffering the global 3DGS against itself:

```text
E(g, r) = manual_frame_weight * manual_votes(g, r)
        + propagated_votes(g, r)
```

Completed manual frames are hard identity anchors. Background `0` is unknown
and casts no negative vote. Predicted depth is not used as a hard visibility
gate. Every Gaussian with direct positive evidence is assigned to its winning
region; confidence records ambiguity instead of rejecting the point. The small
unobserved remainder inherits a confidence-weighted local 3D neighbor
consensus, producing one owner for every shared Gaussian.

The completed 3D ownership field is projected back into propagated views. When
its majority is clear, a whole connected source component changes identity.
Relabeling requires at least `max(4, 0.05% of component pixels)` projected
samples and `55%` majority. The method never adds or removes foreground pixels,
and manual masks pass through exactly.

The editor exposes:

- `Anchored Masks`: dense, shape-preserving persistent-region masks;
- `Complete 3D Region Ownership`: projected ownership used to correct
  propagated component identity;
- `Labeled Global 3DGS`: persistent-region labels on the shared geometry.

Machine-readable diagnostics include per-Gaussian confidence and provenance,
per-frame relabel counts, and `training_view_weights.json`. The isolated
anchored reconstruction below consumes these artifacts directly. Do not run
released `splat_prepare` or `all_splat` for this experiment; those stages belong
to the released reconstruction path and invoke its own mask logic.

### Anchored Splat Reconstruction

Experiment D reconstructs persistent regions in an isolated output tree:

```text
interactive/reconstruction/runs/anchored_3dgs/<run-id>/
```

It preserves each region's complete Gaussian state, applies manual-frame
weights to mask supervision, disables densification during the anchored
warm-up, then prunes low-opacity, oversized, and repeatedly
mask-unsupported floaters before reusing the paper's collision-driven
composition. The default run performs no
SAM2 refinement and no redundant second per-region training pass:

```bash
AS_COMMON=(
  --model-dir "$OMEGA_RESULT_DIR"
  --editor-baseline-name "$SEG_BASELINE"
  --split-splat-root "$SPLIT_SPLAT_ROOT"
  --python "$SPLIT_SPLAT_PYTHON"
  --source-run-id split_splat_anchored_sam2_video
)

"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_anchored_splat.py" \
  --stage all \
  --run-id anchored_splat_no_refine \
  --mask-refinement none \
  --mask-loss-weight 1 \
  --floater-opacity-threshold 0.005 \
  --floater-min-visible-views 3 \
  --overwrite-stage \
  "${AS_COMMON[@]}"
```

The stages are `prepare`, `initial`, `refine_masks`, `refined`, and `compose`.
Each trained object records `floater_pruning.json` before composition; it lists
low-opacity, oversized-world-scale, and multi-view-unsupported removals
separately. Anchored composition also writes
`post_composition_floater_pruning.json` after opacity-reset optimization.
This command intentionally replaces the earlier `anchored_splat_no_refine`
reconstruction while retaining the shared anchored Split. After this complete
rerun, omit `--overwrite-stage` to resume or reuse the corrected stages.

After changing the anchored ownership or training policy, regenerate the
anchored Split with `--overwrite-stage`, then regenerate all anchored
reconstruction stages with the same flag. Shared dataset preparation and the
30,000-iteration global 3DGS remain cached.

After `compose` completes, refresh the editor. Step 1 groups this reconstruction
under `Anchored 3D Adaptation` beside the released Official and SAM2 Video
experiments. `Composed 3DGS` loads the exact final RGB splats once. The left
panel switches between RGB and persistent-region colors and controls the same
per-region visibility in both modes. Separate no-refinement and paper-SAM2 runs
remain separate reconstruction entries.

Run the paper SAM2 refiner only as a separate ablation. It restores manual
anchor masks afterward and cannot alter D0 or the released baselines:

```bash
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_anchored_splat.py" \
  --stage all \
  --run-id anchored_splat_paper_sam2 \
  --mask-refinement paper_sam2 \
  "${AS_COMMON[@]}"
```

The first backend trains only views where that region has a nonempty mask,
matching the released instance loader. Absent views are excluded rather than
treated as negative background. Pixel-level known/unknown supervision remains
a future controlled ablation.

### MapAnything Region 3DGS

This custom pipeline is parallel to Split&Splat. It labels the exact
MapAnything point cloud used to initialize OMeGa, preserves manual mask shapes,
corrects only propagated component IDs, and trains each persistent region from
its RGB point subset with ordinary 3DGS densification. See
`MAPANYTHING_REGION_3DGS.md` for the contract and math.

```bash
MAP3D_PYTHON="$PROJECT_ROOT/.venv_split_splat/bin/python"
MAP3D_COMMON=(
  --model-dir "$OMEGA_RESULT_DIR"
  --editor-baseline-name "$SEG_BASELINE"
  --split-splat-root "$SPLIT_SPLAT_ROOT"
  --python "$MAP3D_PYTHON"
  --run-id mapanything_sam2_video
  --propagation-method sam2_video
  --manual-frame-weight 8
)

# Rebuild point ownership, masks, point-initialized region 3DGS models,
# and their direct composition. Ambiguous initializer points are completed
# from local 3D consensus instead of being dropped.
"$MAP3D_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_mapanything_3dgs.py" \
  --stage all \
  --instance-iterations 30000 \
  --minimum-region-points 32 \
  --minimum-positive-views 2 \
  --max-num-splats 100000 \
  --mask-loss-weight 1 \
  --floater-opacity-threshold 0.005 \
  --floater-min-visible-views 3 \
  --overwrite-stage \
  "${MAP3D_COMMON[@]}"
```

The 30,000-iteration quality run matches the standard full-3DGS optimization
horizon and permits densification through iteration 15,000, subject to the
per-region cap. The `100000` cap is
kept because the current composed MapAnything model already contains 2.32
million Gaussians; increasing it first would mostly increase storage and
overfitting risk rather than address optimization maturity.

The command replaces the earlier `mapanything_sam2_video` Split and Splat
stages so the completed point ownership and corrected mask objective are both
used. It does not alter editor masks or propagation results. After this complete
rerun, omit `--overwrite-stage` to resume or reuse the corrected stages.
Refresh the editor after Split or Splat completes; Step 1
shows the masks, projected point support, segmented points, composed scene, and
individual trained regions under `MapAnything Region 3DGS`.

### Splat Both Split Results

Run these only after the corresponding `split` stage completes. `all_splat`
executes five resumable phases: instance dataset preparation, initial
per-instance reconstruction, Gaussian-guided SAM2 mask refinement, refined
per-instance reconstruction, and collision-driven composition.

```bash
# Experiment A: official four-grid SAM2 -> Split -> Splat.
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage all_splat \
  --run-id split_splat_official \
  --proposal-source official_auto \
  "${SS_COMMON[@]}"

# Experiment B: our SAM2 Video masks -> Split -> the same Splat stages.
"$SPLIT_SPLAT_PYTHON" \
  "$OMEGA_BUILDING_ROOT/scripts/run_omega_split_splat.py" \
  --stage all_splat \
  --run-id split_splat_sam2_video \
  --proposal-source propagation \
  --propagation-method sam2_video \
  "${SS_COMMON[@]}"
```

The default Splat settings follow the paper and released ScanNet scripts:

- initial instance training: `1,000` iterations with the released
  `--init_rec` mask dilation;
- Gaussian reprojection: five spatially distributed positive SAM2 prompts and
  paper IoU selection on every camera; a previously missing mask is accepted
  only when its IoU with the rendered Gaussian mask exceeds `0.95`;
- refined instance training: another `1,000` iterations;
- composition: `1,000` iterations per collision merge, no densification,
  opacity reset before each merge, instance-aware RGBA mask loading, and mask
  weights `0.05 0.15 0.25` across rounds, clamped to `0.25` afterward.

The released `mask_optimizer_scannet.py` stops after roughly half the cameras
and uses `0.05` instead of the paper's `0.95` missing-mask threshold. The
compatibility launcher changes only those two controls at runtime; the
upstream checkout remains untouched. Composition pair scores use paper Eq. 4,
the directional fraction of one object's Gaussians inside the other's AABB.
For an unordered pair, the larger of the two directional scores is used.

The paper states that the composition weight increases by `0.1` from `0.05`
to `0.25`, giving `0.05, 0.15, 0.25`. The released README instead lists
`0.05, 0.1, 0.25`, while `train.py` hardcodes `0.25`. The adapter uses the
paper schedule explicitly and records it in every run manifest; it does not
edit the upstream checkout. Override it only for a documented ablation:

```text
--composition-mask-weights 0.05 0.10 0.25
```

Each completed instance model is cached independently. Re-running the same
phase skips an object when its expected `iteration_1000/point_cloud.ply`
already exists. Do not pass `--overwrite-stage` when resuming an interrupted
run.

The global dataset, images, camera support, metric depth conversion, and 30k
global 3DGS remain under `_shared/<cache>/`. Everything affected by the mask
source remains under its run:

```text
runs/<run-id>/splat/
  01_instance_datasets/  # copied masks; shared images/cameras are symlinks
  02_initial_models/     # first 1k object reconstructions
  03_mask_refinement/    # per-instance completion markers
  04_refined_models/     # second 1k object reconstructions
  05_composition/        # collision rounds, pair datasets, pair models
  06_outputs/            # composed scenes and post-composition object models
```

The final files are:

```text
06_outputs/composed_scene.ply
06_outputs/composed_scene_instance_ids.ply
06_outputs/composed_scene_instance_labels.npy
06_outputs/composed_scene_points.npz
06_outputs/individual_objects/object_<instance-id>.ply
06_outputs/split_rgb_objects/object_<instance-id>.ply
```

After completion, Step 1 distinguishes appearance from instance geometry. The
shared global model is the radiometric RGB reconstruction. `split_rgb_objects`
contains RGB subsets of that model selected by the Split labels.
`individual_objects` contains the paper's post-composition object geometry and
is displayed with categorical ID colors. The released Split point writer omits
RGB from its per-instance initializer, so these composed files must not be
presented as photometric reconstructions. The adapter restores global DC color
when preparing future per-instance training datasets, preventing the released
renderer's exact-black initialization from becoming a dead SH gradient.

The upstream Split result is a collection of binary masks per instance, and
different instances may overlap in one view. The editor's `Split Masks
(Preview)` layer is therefore only an exclusive visualization: it assigns
overlap pixels to the smaller mask so thin instances remain visible. It never
changes the raw instance masks used by Splat. `conflictPixels` in the Split
summary records the amount of overlap.

The later `Splat-Refined Masks` layer is the last persisted per-view mask set in
the released pipeline. After initial per-instance 3DGS training, the
`splat_masks` stage projects five Gaussian prompts per visible object into each
view, asks SAM2 for a candidate, and retains the candidate only when its IoU
against the rendered object silhouette improves upon the existing mask. These
per-instance masks supervise refined object training and composition.
Composition does not emit another per-view mask revision. The editor again
resolves overlaps only for display and leaves the upstream binary masks intact.

For the `editor_depth` ablation, `split` first calibrates each Depth Anything
inverse-depth map to the shared global Gaussian geometry using Split&Splat's
median/deviation affine rule. The 214 calibrated visibility maps are cached
once under `_shared/<cache>/02_global_gs/split_depth_editor_depth/` and reused
by every proposal run. Murre depth follows the native metric-depth path and is
not calibrated.

The released `split` command has two full-view passes: progressive 3D label
voting, then SAM2 mask recovery. On this 214-frame scene, the official
automatic proposals can retain hundreds of provisional labels and take well
over an hour. Continuous tqdm output means it is working. `Ctrl+C` safely
terminates the child process and records the stage as cancelled.

The compatibility launcher also handles projected prompt sets that are exactly
collinear. The released coreset sampler otherwise asks Qhull for a 2D convex
hull and fails on these valid rank-1 image-space observations. Only this
degenerate case uses the sampler's existing all-points fallback; ordinary 2D
hulls are unchanged. After an interrupted `split`, rerun the same stage command
without `--overwrite-stage`: the interrupted workspace is rebuilt, while the
shared 30k global GS, calibrated depth cache, and proposal masks are reused.

Use `--overwrite-stage` only on the stage that must be regenerated.
Overwriting `prepare` or `global_gs` invalidates the shared geometry used by
both mask experiments.

The runner prints stage starts, cache hits, frame counts, upstream training
progress, proposal statistics, and final artifact registration to the terminal.
Progress is written to `stderr`, while the final stage summary remains JSON on
`stdout`. The same upstream output is retained in:

```text
_shared/<cache>/logs/global_gs.log
runs/<run-id>/logs/proposals.log
runs/<run-id>/logs/split.log
runs/<run-id>/logs/splat_initial/<instance-id>.log
runs/<run-id>/logs/splat_masks/<instance-id>.log
runs/<run-id>/logs/splat_refined/<instance-id>.log
runs/<run-id>/logs/splat_compose/<round-and-pair>.log
```

The latest machine-readable state is always available in
`runs/<run-id>/progress.json`.

For the paper-faithful DSLR depth input, replace the depth argument in
`SS_COMMON` with:

```text
--depth-source murre --depth-dir /path/to/murre/metric_depth
```

The current `editor_depth` run is a documented Depth Anything V2 ablation, not
the paper-faithful Murre depth setting.

After a staged run completes, refresh the editor page. Step 1 groups its
artifacts under one compact Split&Splat experiment:

- `Input Proposals`, `Split Masks (Preview)`, and `Splat-Refined Masks` compare
  the input, post-Split, and final persisted post-SAM2 Splat masks;
- `Labeled Global 3DGS` shows the Split-stage labels on the shared scene;
- `Composed 3DGS` loads exact final RGB partitions once; the left panel toggles
  RGB/region colors and controls the shared visibility of every instance.

The shared global RGB reconstruction is listed once because all experiments
reuse it. The official, released SAM2 Video, and anchored SAM2 Video
experiments remain separate. Step 3 lists dense results under
**Split-Refined Masks** and anchored sparse projections under **3D Evidence**.
Released Split&Splat IDs use the `split_splat_instance` namespace and are never
interpreted as designer persistent-region IDs. The anchored run deliberately
keeps the `persistent_region` namespace. ID-colored scenes preserve each
Gaussian's position, opacity, scale, and rotation; labeled splats use the same
bright deterministic colors as the masks. Unlabeled Split-stage splats are
transparent. Semantic viewer files use a display-only 0.04 opacity floor so
categorical labels remain legible; the canonical trained PLY is unchanged.
Checking a Gaussian result renders it in place with the editor camera, frame
intrinsics, portrait orientation, pan, and zoom. Selecting a point-cloud source
switches back to software points.

The in-place renderer is the editor's single Gaussian viewer path.
The embedded path uses unmodified Gaussian colors (`tonemapping: none`) and
high-precision accumulation. This avoids clipping low-opacity HDR splats before
their contributions are composed into the final display image.

The `proposals` stage is also registered in Step 3 under **Frame Proposals** as
a read-only `Split Input: ...` layer. For the official baseline this previews
the released four-grid SAM2 masks. The colored editor view is necessarily an
exclusive label map; the Split stage still consumes the untouched, potentially
overlapping binary masks from the paper code.

Useful debugging artifacts:

- `split/01_proposals/label_maps/` and `overlays/`: exact proposal inputs;
- `split/01_proposals/source_mappings/`: SAM2 Video persistent-ID provenance;
- `_shared/<cache>/02_global_gs/model/point_cloud/iteration_30000/point_cloud.ply`:
  the full upstream model, including Split identity descriptors;
- `_shared/<cache>/02_global_gs/viewer_gaussians.ply`: render-only RGB 3DGS
  with the unused identity descriptors removed;
- `split/02_point_labels/labeled_points.npz`: Gaussian means colored by final
  Split instance ID in editor Step 6;
- `split/02_point_labels/point_labels.npy`: one Split label aligned to every
  row of the global Gaussian model;
- `split/02_point_labels/point_label_confidence.npy`: anchored-run confidence
  for each global Gaussian;
- `split/02_point_labels/point_label_provenance.npy`: anchored-run provenance
  (`0` unlabeled, `1` propagated evidence, `2` manual-anchor evidence);
- `split/02_point_labels/projected_support/`: sparse projected 3D labels used
  to diagnose relabel decisions;
- `split/02_point_labels/segmented_gaussians.ply`: global Gaussian geometry,
  opacity, scale, and rotation with view-independent instance colors;
- `split/03_consistent_masks/overlays/`: final per-view colored masks;
- `split/03_consistent_masks/label_maps/`: final exclusive `uint16` masks;
- `split/03_consistent_masks/training_view_weights.json`: anchored manual and
  propagated frame weights staged for a future weighted Splat trainer;
- `split/02_point_labels/upstream_instances/<id>/`: released per-instance
  binary masks and labeled PLY files.
- `splat/06_outputs/composed_scene_instance_ids.ply`: final composed scene
  with view-independent instance colors;
- `splat/06_outputs/individual_objects/`: categorical viewer geometry for each
  final composed instance;
- `splat/06_outputs/split_rgb_objects/`: true RGB subsets of the shared global
  reconstruction for the instances retained by the Splat stage.

The upstream PLY contains 384 descriptor channels needed by Split and is much
larger than a render file. The registration pass strips only non-render
channels for `viewer_gaussians.ply`; it keeps all RGB spherical harmonics.
`segmented_gaussians.ply` keeps the same Gaussian geometry but uses degree-zero
label colors, which makes labels stable from every viewpoint and keeps the
debug artifact smaller. Completed-stage cache hits refresh these registrations
without retraining.

The embedded viewer is currently connected to Split&Splat's standard 3DGS
outputs. Its WebGL canvas sits behind the editor's transparent annotation
canvas, while the editor remains the sole owner of navigation and projection.
Its renderer adapter is intentionally separate from the software point renderer
so a disk-aware 2DGS mode can be added without changing the segmentation UI or
artifact contract.

The setup script rebuilds all released CUDA extensions for the isolated
CPython 3.12 environment. The staged runner reuses
`third_party/sam2/checkpoints/sam2.1_hiera_large.pt` through a symlink, so it
does not download a duplicate checkpoint.

### Export Persistent Masks

Export a portable package containing indexed masks, persistent-region
metadata, manual anchors, frame correspondence, parsing instructions, and
checksums:

```bash
export SEGMENTATION_SHARE=$DT_ROOT/data/shared/grove_entrance_dslr_0521_segmentation_share

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/export_omega_segmentation_share.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_area_samples_1024_dense \
  --method sam2_video \
  --output-dir "$SEGMENTATION_SHARE/sam2_video" \
  --overwrite

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/export_omega_segmentation_share.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_area_samples_1024_dense \
  --method xmem \
  --output-dir "$SEGMENTATION_SHARE/xmem" \
  --overwrite
```

The package is self-contained except for RGB images. Its `frames.jsonl` maps
each `uint16` indexed mask back to the capture-relative RGB path. Manual
anchors and propagated frames share the same persistent-region ID space.
