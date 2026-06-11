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
```

Install lightweight dependencies if needed:

```bash
$PYTHON -m pip install open3d natsort matplotlib tqdm opencv-python scipy plyfile supervision hydra-core iopath
```

## 2. Common 2D Proposal Initializer

This stages the DSLR frames once at 1024px width and creates independent SAM2
automatic proposal masks for every view.

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
masks, and writes both observe and final debug visualizations.

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
