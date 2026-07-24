# Run OMeGa Segmentation

This file keeps only the active segmentation pipeline:

```text
common 2D proposal initializer
  -> SAM2Object on the common source
  -> SAI3D on the common source with proposal-gated view-mask refinement
  -> official-like Split&Splat propagation on the common source
  -> SAI3D point-cloud variants for MapAnything and COLMAP evidence
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
export SPLIT_SPLAT_NAME=split_splat_area_samples_1024
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

## 7. Official-Like Split&Splat Propagation Baseline

This tests Split&Splat's split-stage logic separately from SAI3D. It starts from
the common SAM2 automatic proposals, projects area-sampled OMeGa mesh points
through rendered mesh depth, grows global 3D point labels by proposal overlap,
clusters the 3D labels, then writes per-view instance masks by matching final 3D
labels back to SAM2 proposals with SAM2 point-prompt fallback.

Run the full baseline:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline split_splat \
  --stage all \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name "$SPLIT_SPLAT_NAME" \
  --proposal-source-name "$PROPOSAL_NAME" \
  --point-source mesh_area_samples \
  --point-sample-count 160000 \
  --point-sample-seed 71 \
  --process-order reverse \
  --view-mask-sam2-root "$DT_ROOT/third_party/sam2" \
  --view-mask-sam2-checkpoint "$SAM2_CHECKPOINT" \
  --view-mask-sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --overwrite
```

If point propagation already exists and you only want to regenerate final view
masks:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_baselines.py" \
  --baseline split_splat \
  --stage view_masks \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name "$SPLIT_SPLAT_NAME" \
  --view-mask-sam2-root "$DT_ROOT/third_party/sam2" \
  --view-mask-sam2-checkpoint "$SAM2_CHECKPOINT" \
  --view-mask-sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --overwrite
```

Visualize:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega10_visualize_split_splat_baseline.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name "$SPLIT_SPLAT_NAME" \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 520 \
  --overwrite
```

Inspect:

- `point_labels/labeled_points.ply`: raw propagated 3D point labels;
- `point_labels/clustered_labeled_points.ply`: DBSCAN-cleaned final point labels;
- `view_masks/projected_points/`: final clustered labels projected as sparse
  virtual masks;
- `view_masks/proposal_assigned/`: automatic proposals assigned to global IDs;
- `view_masks/sam2_fallback/`: SAM2 coreset fallback masks for visible labels
  missed by proposal matching;
- `view_masks/masks/`: final exclusive label maps for our visualization/tooling;
- `view_masks/by_instance/`: per-instance binary masks, closest to upstream
  Split&Splat's training-mask layout.

## 8. Interactive Editor And Propagation Baselines

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
hybrid cloud. Every completed run appears in Step 1
under `OMeGa Optimized Mesh -> SAI3D Results`, labeled by its evidence source
and anchor weight. Each result is an independently toggleable point-cloud layer,
so automatic and propagated runs can be compared directly.
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
