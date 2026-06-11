# OMeGa Segmentation Pipeline

Status: active design notes for the OMeGa building fork.

The goal is designer-useful segmentation for OMeGa scenes: stable parts across
views, clean enough per-view masks for later training control, and 3D labels
that can drive mesh/splat grouping, remesh policies, and per-region
optimization.

## Current Direction

The current OMeGa mesh is still an optimization product. It can be noisy,
incomplete, folded, and over-dense near boundaries. For that reason, we should
not make mesh-only segmentation the core path.

The active pipeline is:

```text
common 2D proposal initializer
  -> SAM2Object tracking / graph-consistency baseline
  -> SAI3D multi-view superpoint graph baseline
  -> SAI3D-to-view proposal-gated finalizer inspired by Split&Splat
  -> official-like Split&Splat propagation baseline
  -> SAI3D point-cloud evidence variants
```

The initializer is the only place that stages DSLR frames from the capture
package. Downstream methods consume its staged RGB/pose frames and per-view
proposal masks through one shared contract:

```text
<proposal_source>/baseline_summary.json
<proposal_source>/dataset/frame_manifest.jsonl
<proposal_source>/view_masks/masks/
```

This keeps SAM2Object, SAI3D, and Split&Splat separate as methods while avoiding
hidden fallbacks where one baseline secretly initializes another.

Gaussian Grouping is tracked separately because it is not a post-hoc
segmentation baseline over our common proposal source. It is a training-time
3DGS method with its own DEVA/SAM pseudo-label preparation and its own Gaussian
identity-feature optimization loop. We keep it faithful to the original method
instead of feeding it SAM2Object, SAI3D, mesh, point-cloud, or lifted-mask
outputs.

## Implemented Files

```text
third_party/OMeGa_4_Building/
  RUN_OMEGA_SEGMENTATION.md
  RUN_OMEGA_GAUSSIAN_GROUPING.md
  SEGMENTATION_IDEAS.md
  scripts/
    run_omega_segmentation_baselines.py
    run_omega_gaussian_grouping.py
  omega_local/
    segmentation/
      proposal_source.py
      view_proposals_baseline.py
      sam2object_baseline.py
      sai3d_baseline.py
      split_splat_baseline.py
      gaussian_grouping_pipeline.py

scan_processing/
  VisSegOmega03_visualize_sam2object_baseline.py
  VisSegOmega04_visualize_sai3d_baseline.py
  VisSegOmega05_visualize_sai3d_observe.py
  VisSegOmega08_visualize_view_proposals.py
  VisSegOmega09_visualize_gaussian_grouping.py
  VisSegOmega10_visualize_split_splat_baseline.py
```

The active outputs live under:

```text
<model_dir>/segmentation/baselines/view_proposals_1024/
<model_dir>/segmentation/baselines/sam2object/
<model_dir>/segmentation/baselines/sai3d/
<model_dir>/segmentation/baselines/sai3d_area_samples/
<model_dir>/segmentation/baselines/split_splat/
<model_dir>/segmentation/baselines/sai3d_mapanything_points_1024/
<model_dir>/segmentation/baselines/sai3d_colmap_sparse_points_1024_sparse_graph/
<model_dir>/segmentation/gaussian_grouping/deva_scale4/
```

## 0. Separate Faithful Baseline: Gaussian Grouping

Existing method:

- Gaussian Grouping: https://github.com/lkeab/gaussian-grouping
- Paper: Gaussian Grouping: Segment and Edit Anything in 3D Scenes, ECCV 2024.

Implementation:

```text
scripts/run_omega_gaussian_grouping.py
omega_local/segmentation/gaussian_grouping_pipeline.py
scan_processing/VisSegOmega09_visualize_gaussian_grouping.py
RUN_OMEGA_GAUSSIAN_GROUPING.md
```

Role:

Gaussian Grouping is a training-time 3D Gaussian Splatting segmentation
baseline. Unlike SAM2Object and SAI3D, it should not consume our common 2D
proposal initializer or any of our lifted 3D labels. The fair test is to stage
our OMeGa COLMAP dataset into its expected layout, run its original DEVA/SAM
pseudo-label script, train its Gaussian identity fields, and render its native
mask outputs.

Math / logic:

- Each Gaussian stores the normal 3DGS appearance parameters plus a learned
  compact object / identity feature.
- The modified rasterizer renders both RGB and a per-pixel object-feature map.
- A 1x1 classifier maps rendered object features to pseudo-label IDs.
- Cross-entropy against DEVA/SAM per-view pseudo labels supervises the identity
  field during training.
- A 3D spatial consistency regularizer encourages nearby Gaussians to have
  similar object-label probabilities.

Current OMeGa staging:

- Source dataset: `<capture>/omega_stable_mesh/dataset`
- Staged dataset: `<gaussian-grouping>/data/<dataset_name>`
- Full RGB and COLMAP sparse reconstruction are symlinked.
- `images_4/` is created for Gaussian Grouping's pseudo-label script.
- Training uses `-r 4`, so the RGB resolution matches the generated
  `object_mask/*.png` labels.

What to inspect:

- `object_mask/`: DEVA/SAM pseudo labels used for training;
- `model/train/ours_<iter>/objects_pred/`: rendered predicted masks;
- `model/train/ours_<iter>/objects_feature16/`: PCA visualization of learned
  identity features;
- `model/train/ours_<iter>/concat/`: native Gaussian Grouping side-by-side
  render output;
- `VisSegOmega09` contact sheets for quick comparison.

## 1. Common 2D Proposal Initializer

Implementation:

```text
omega_local/segmentation/view_proposals_baseline.py
omega_local/segmentation/proposal_source.py
scan_processing/VisSegOmega08_visualize_view_proposals.py
```

Purpose:

- stage posed DSLR RGB frames once;
- choose the working image resolution once;
- write camera poses and intrinsics in the baseline folder;
- run SAM2 automatic mask generation independently per view;
- convert overlapping SAM2 masks into one integer proposal map per view.

Math / logic:

- SAM2 automatic masks produce local 2D candidate regions.
- Each mask has predicted IoU, stability, and area.
- We sort candidates by a score from predicted IoU and stability.
- Higher-scoring masks claim pixels first.
- Later masks only keep currently unoccupied pixels.
- The output is one mutually exclusive integer label map per frame.

Important limitation:

- This source has no cross-view identity. It only provides local 2D proposals.
- Cross-view reasoning belongs to SAM2Object, SAI3D, or the Split&Splat
  propagation baseline.

## 2. SAM2Object On The Common Source

Existing method:

- SAM2Object: https://github.com/jihuaizhaohd/SAM2Object
- Project page: https://jihuaizhaohd.github.io/SAM2Object/

Implementation:

```text
omega_local/segmentation/sam2object_baseline.py
scan_processing/VisSegOmega03_visualize_sam2object_baseline.py
```

Role:

SAM2Object is our video-propagation baseline. It consumes the initializer's
ordered RGB/pose frames, then runs SAM2Object-style keyframe segmentation,
forward/reverse propagation, merge, and mesh-graph consolidation.

Math / logic:

- Keyframes get SAM2 automatic masks.
- SAM2 video propagation tracks those masks through the ordered view sequence.
- A reverse pass tracks from the opposite direction.
- The merge step fills forward-mask holes with compatible reverse masks.
- The graph step projects OMeGa mesh vertices into views using rendered depth.
- Neighboring mesh vertices have high affinity when they are co-visible and
  receive the same tracked 2D mask ID.
- Progressive thresholds merge high-confidence components first, then weaker
  compatible components.

What to inspect:

- forward and reverse tracking masks;
- merged local masks before graph consistency;
- consistent view masks after graph relabeling;
- `graph_consistency/labeled_mesh.ply`.

Observed behavior:

- useful as a representative video-propagation baseline;
- can over-merge architectural pieces into large facade regions;
- per-view coverage and identity can still drift.

## 3. SAI3D On The Common Source

Existing method:

- SAI3D: https://github.com/yd-yin/SAI3D
- Project page: https://yd-yin.github.io/SAI3D/
- Local editable fork: `third_party/SAI3D_DT`

Related method used for our view-mask refinement:

- Split&Splat: https://github.com/LTTM/Split_and_Splat
- Paper: https://arxiv.org/abs/2602.03809

Implementation:

```text
omega_local/segmentation/sai3d_baseline.py
scan_processing/VisSegOmega05_visualize_sai3d_observe.py
scan_processing/VisSegOmega04_visualize_sai3d_baseline.py
```

Role:

SAI3D is our current best base for cross-view 3D identity. It consumes the
initializer's proposal masks, renders OMeGa mesh depth for visibility, builds a
3D graph over mesh vertices, area-balanced mesh samples, or external point
clouds, creates local superpoints, and runs SAI3D-style multi-view graph
merging.

Math / logic:

- A 3D primitive is visible in a view if mesh-depth rendering agrees with its
  projected depth.
- For external point clouds, visibility can instead use a per-view point
  z-buffer: keep the nearest projected point per pixel, with a small depth
  band for nearby samples.
- Visible primitives sample the local 2D proposal label at their projected pixel.
- Each primitive accumulates a distribution over observed 2D proposal labels.
- Neighboring primitives have high affinity when their visible proposal-label
  distributions agree across shared views.
- SAI3D progressively merges primitives from strict to relaxed thresholds.
- Small components can be merged into compatible neighbors.

Our SAI3D finalizer extension:

```text
SAI3D global 3D labels
  -> project visible labeled graph points into each view
  -> render dense geometry support from face labels
  -> use projected labels as virtual masks
  -> assign automatic SAM2 proposals by eroded overlap with 3D support
  -> prompt SAM2 only for visible labels without trusted proposals
  -> accept fallback masks only if they pass geometry IoU/recall gates
  -> write final per-view global-ID masks
```

This borrows the proposal-matching idea from Split&Splat while keeping SAI3D as
our source of cross-view identity. It is not the full Split&Splat split stage.
The implementation keeps two separate stages:

- `--stage segment`: compute SAI3D's consistent 3D labels and save point,
  vertex, and face labels;
- `--stage view_masks`: project the saved 3D labels to views, then choose one
  of three finalizers: pure geometry projection, old SAM2-per-label prompting,
  or the current proposal-gated finalizer mode.

The geometry projection is the consistency anchor. In the current
`split_splat` mode, automatic SAM2 proposals are the primary 2D boundary source,
while SAM2 point prompting is a fallback for visible SAI3D objects that were
missed or rejected by proposal assignment. The name is historical; it should be
read as a SAI3D finalizer inspired by Split&Splat, not as the full method.

Current point sources:

- `mesh_vertices`: direct OMeGa mesh vertices. This inherits OMeGa's vertex
  density.
- `mesh_area_samples`: triangle-area-balanced samples on the OMeGa mesh. This
  reduces vertex-density bias and gives broad planar surfaces fairer support.
- `point_cloud`: external PLY point clouds used directly as SAI3D graph points.
  This lets us test raw MapAnything and COLMAP geometry evidence without
  relying only on the current OMeGa mesh.

Point-cloud implementation details:

- External point-cloud superpoints use non-empty voxel count targeting rather
  than full bounding-box volume. This matters for sparse/detail-biased clouds
  like COLMAP, where a volume-derived voxel size can create a few huge
  superpoints.
- `--graph-max-edge-length` optionally rejects long kNN graph edges. This is
  important when the point cloud is not a continuous surface and kNN would
  otherwise bridge empty space.
- `mesh_labels/labeled_points.ply` is the direct point-cloud segmentation
  output. Use it before judging `labeled_mesh.ply`, which transfers point labels
  back to the OMeGa mesh.

What to inspect:

- `observe` visualization: superpoints, visible support, 2D labels on points,
  and positive-support maps;
- after `segment`: inspect `mesh_labels/labeled_points.ply`,
  `mesh_labels/labeled_mesh.ply`, and the observe visualization;
- after `view_masks --view-mask-refine-mode geometry`: inspect the pure SAI3D
  projected masks;
- after `view_masks --view-mask-refine-mode split_splat`: inspect accepted
  proposal relabels, rejected proposals, projected point support, dense geometry
  support, SAM2 fallback prompts/candidates, and final geometry-gated masks;
- `mesh_labels/labeled_points.ply` for external point clouds;
- `mesh_labels/labeled_mesh.ply` for mesh-based runs and transferred labels.

Observed behavior:

- cleaner cross-view identities than raw 2D proposals;
- still limited by proposal quality and OMeGa visibility;
- final 3D labels are useful seeds;
- the proposal-gated finalizer is the current SAI3D-side test for converting
  those seeds into sharper per-view masks.

Current recommended mesh-based SAI3D test:

```text
view_proposals_1024
  -> mesh_area_samples with 160k sampled surface points
  -> voxel superpoints with target count 30k
  -> SAI3D progressive merge at 0.9,0.8,0.7,0.6,0.5
  -> proposal-gated finalizer and SAM2 fallback
  -> comprehensive observe/final visualization
```

Why this variant matters:

- 1024px proposals preserve more facade/architectural detail than the old
  low-resolution run.
- Area-balanced graph points avoid over-weighting where the OMeGa mesh happens
  to have dense vertices.
- A higher superpoint target keeps the graph locally editable before SAI3D
  progressively merges it.
- The comprehensive final visualization is meant to debug the whole chain:
  input proposals, rendered mesh visibility, observed graph labels,
  superpoint support, solved 3D labels, geometry support, SAM2 prompts,
  candidates, final masks, and boundaries.

### 3.1 MapAnything Point-Cloud Test

Baseline:

```text
sai3d_mapanything_points_1024
```

Input:

```text
<capture>/pointcloud/mapanything/phone_mapanything_points.ply
```

Settings:

- `--point-source point_cloud`
- `--point-cloud-max-points 300000`
- `--point-visibility-source point_zbuffer`
- `--superpoint-target-count 20000`
- default progressive merge: `--thres-connect 0.9,0.5,5`

Observed behavior:

- broad coverage and good semantic grouping;
- useful segmentation signal despite the current OMeGa mesh being incomplete;
- point positions are noisy/layered and not perfectly consistent across views;
- best interpreted as broad 3D evidence for later 2D mask cleanup, not as a
  final clean designer-facing 3D segmentation.

### 3.2 COLMAP Sparse Point-Cloud Test

Baseline:

```text
sai3d_colmap_sparse_points_1024_sparse_graph
```

Input:

```text
<capture>/pointcloud/colmap_sparse/colmap_sparse_points.ply
```

Settings:

- `--point-source point_cloud`
- `--point-cloud-max-points 0`
- `--point-visibility-source point_zbuffer`
- `--superpoint-target-count 20000`
- `--graph-max-edge-length 0.35`
- `--max-neighbor-distance 1`
- `--thres-connect 0.9`
- `--thres-merge 0`

Observed behavior:

- COLMAP points cover many key detail regions and edges;
- 2D labels sampled onto visible points are often very good;
- default SAI3D progressive merging was too aggressive for this sparse graph;
- conservative sparse-graph settings produce a much better high-detail
  oversegmentation;
- the result can look fragmented in final 3D labels, but that is acceptable and
  useful for the next stage.

Current conclusion:

```text
fragment first, merge later
```

For sparse COLMAP evidence, preserving local detail is more valuable than
forcing large object regions inside SAI3D. Merging should happen later using
2D consistency, user grouping/splitting, StableNormal/depth agreement, and
object-level training goals.

## 4. Official-Like Split&Splat Propagation Baseline

Existing method:

- Split&Splat: https://github.com/LTTM/Split_and_Splat
- Paper: https://arxiv.org/abs/2602.03809

Implementation:

```text
omega_local/segmentation/split_splat_baseline.py
scan_processing/VisSegOmega10_visualize_split_splat_baseline.py
```

Role:

This is the faithful test of Split&Splat's split-stage logic in our OMeGa
conventions. It does not consume SAI3D labels. It consumes the same common SAM2
automatic proposal source and a chosen 3D point set.

Math / logic:

- Project the 3D point set into each view using rendered depth visibility.
- Initialize global 3D labels from the first processed view's eroded SAM2
  proposals.
- For each later view, project current global point labels into a sparse
  "virtual mask".
- Assign each local SAM2 proposal by eroded overlap with that virtual mask.
- If a proposal has no overlap, create a new global label and vote its visible
  points into that label.
- If a proposal overlaps one label, update that label with the proposal's
  visible points.
- If a proposal overlaps multiple labels, vote points into the majority label.
- After all views, keep point labels whose majority vote probability is at least
  `0.7`.
- Cluster each 3D label with DBSCAN, keeping the largest cluster unless the label
  has many clusters, matching the upstream Split&Splat heuristic.
- Write per-view masks by matching final clustered 3D labels back to automatic
  SAM2 proposals; run SAM2 coreset point prompting only for visible labels with
  no matched proposal.

What to inspect:

- `point_labels/labeled_points.ply`: raw propagated point labels;
- `point_labels/clustered_labeled_points.ply`: final clustered point labels;
- `view_masks/projected_points/`: sparse virtual masks from final 3D labels;
- `view_masks/proposal_assigned/`: local SAM2 proposals assigned to global IDs;
- `view_masks/sam2_fallback/`: masks generated only for visible labels missed by
  proposal assignment;
- `view_masks/masks/`: final exclusive label maps for our tooling;
- `view_masks/by_instance/`: per-instance binary masks closer to the upstream
  Split&Splat output format.

Why this matters:

- It tests the actual Split&Splat split idea separately from SAI3D.
- It tells us whether their proposal-to-3D voting logic is enough for our
  architectural scene before we add our own user edits or StableNormal cues.
- If it fails, we can say the failure belongs to the baseline assumptions rather
  than our SAI3D extension.

## 5. Current Baseline Takeaways

- SAM2Object represents video-propagation behavior: useful, but can merge
  architectural pieces too broadly.
- SAI3D on OMeGa mesh or area samples gives better surface continuity, but is
  limited by the quality and completeness of the current OMeGa mesh.
- The official-like Split&Splat baseline directly tests automatic SAM2 proposal
  propagation through 3D point labels, without SAI3D.
- SAI3D on MapAnything gives broad coverage and good semantic support, but the
  raw point cloud is noisy.
- SAI3D on COLMAP sparse points gives strong detail/local structure support, but
  should remain conservative and oversegmented.
- The most promising custom extension should combine these complementary
  sources instead of selecting one hard 3D substrate.

## Next Research Step

Use SAI3D's 3D labels as the current segmentation endpoint, then study how those
labels should affect OMeGa optimization:

- train or refine different objects/parts with different settings;
- use labels to select remesh constraints and target detail levels;
- later add user edits as hard constraints on top of the same proposal-source
  and 3D-graph structure.
