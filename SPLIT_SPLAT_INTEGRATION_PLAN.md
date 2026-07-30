# Split&Splat Integration Plan

## Goal

Add a staged Split&Splat experiment to the interactive segmentation pipeline:

```text
posed RGB + COLMAP + monocular depth
  -> official Split&Splat automatic masks OR imported editor propagation masks
  -> depth-regularized global 3DGS
  -> 3D-consistent instance labels and view masks
  -> one 3DGS per instance
  -> geometry-guided mask refinement
  -> per-instance retraining
  -> progressive composition
```

The first experiment preserves the method described by Split&Splat and the
released implementation. The implemented second controlled experiment replaces
only the initial mask source with one of our user-conditioned Step 5 outputs.

Paper: https://arxiv.org/html/2602.03809v1

Source: https://github.com/LTTM/Split_and_Splat

Local source: `third_party/Split_and_Splat`

This file specifies the paper adapter and its controlled adaptations. The
backend-neutral partition and reconstruction contracts are defined separately
in
[`SEGMENTATION_RECONSTRUCTION_ARCHITECTURE.md`](SEGMENTATION_RECONSTRUCTION_ARCHITECTURE.md)
so generalization does not change the released baseline.
The parallel 3DGS reconstruction that consumes Experiment C is designed in
[`ANCHORED_SPLAT_DESIGN.md`](ANCHORED_SPLAT_DESIGN.md).

## Fidelity Policy

For the first run:

- preserve the upstream stage order, SAM2 settings, mask-merging rules, depth
  visibility test, erosion, DBSCAN filtering, label voting, per-instance
  training, mask refinement, retraining, collision ordering, and composition
  loss schedule;
- call code from `third_party/Split_and_Splat` through a narrow adapter instead
  of copying GPL-licensed implementation into the OMeGa fork;
- patch only data paths, camera/image naming, progress reporting, resumability,
  and clear release defects that prevent the paper's stated algorithm from
  running;
- record the upstream commit, all patched files, parameters, input hashes, and
  deviations in `compatibility_report.json`;
- do not use OMeGa geometry, SAI3D labels, StableNormal, persistent-region
  weights, or custom fusion in this baseline.

The paper and public release are not perfectly identical. For example, the
paper specifies a depth-regularized dense 3DGS for Split, while one released
ScanNet script loads the ScanNet mesh and the general script loads COLMAP
points. The primary baseline follows the paper's dense-3DGS route using the
released 3DGS implementation and records every source-level compatibility patch.

## Canonical Inputs

Use one frozen input manifest for every stage:

- staged 1024-pixel-long-edge RGB frames used by the editor;
- rescaled camera intrinsics for that exact image grid;
- COLMAP cameras, poses, tracks, and sparse points from
  `omega_stable_mesh/dataset/sparse/0`;
- monocular depth at the same frame dimensions;
- the Split&Splat SAM2.1 Hiera-L checkpoint and upstream four-grid settings.

For the DSLR sequence, the paper uses Murre when measured depth is unavailable
because it is designed to be scale-consistent with SfM. Murre is not bundled in
the Split&Splat repository, so it is an explicit dependency for the faithful
baseline. Depth Anything V2 may be tested later as an ablation, not silently
used as the official depth source.

Before any segmentation run, validate:

1. every RGB, depth, mask, and camera has the same frame identity;
2. every raster has the expected dimensions;
3. rescaled intrinsics reproduce COLMAP tracks with low reprojection error;
4. monocular depth is scale-aligned with visible COLMAP points;
5. the coordinate convention matches the Split&Splat projection code.

## Experiment A: Faithful Split&Splat

Run ID: `split_splat_official`

### Step 5A: Prepare Split&Splat Inputs

Create an isolated adapter workspace without renaming or modifying source
images. Export the camera, image, depth, and COLMAP layouts expected by the
upstream code. Save the mapping between editor frame IDs, source image names,
and Split&Splat frame names.

Outputs:

```text
split/01_input/
  manifest.json
  frame_map.jsonl
  images/
  depth/
  colmap/sparse/0/
  camera_validation.json
```

### Step 5B: Generate Upstream Automatic Masks

Run Split&Splat's four SAM2 automatic-mask configurations:

- `uber_huge`: 1 point per side;
- `very_coarse`: 4 points per side;
- `coarse`: 8 points per side;
- `fine`: 16 points per side.

Preserve the separate binary masks and apply the upstream coarse-to-fine merge.
These are not equivalent to the editor's current exclusive SAM2 label maps.

Monitor:

- each grid's masks;
- the merged automatic masks;
- proposal count, overlap, coverage, and uncovered pixels per frame.

### Step 5C: Train the Initial Global 3DGS

Initialize from the frozen COLMAP sparse cloud and train the released
depth-regularized 3DGS on the full scene. The Gaussian means form the dense
point set used by Split.

Monitor:

- RGB render versus input;
- rendered depth versus aligned monocular depth;
- depth residual and valid-depth coverage;
- Gaussian count, pruning, and densification;
- the dense Gaussian means in the 3D viewer.

This stage produces geometry for segmentation consistency. It is not one of
the final per-instance reconstructions.

### Step 5D: Propagate Labels Through 3D

Apply the upstream Split logic:

1. project dense Gaussian means into each camera;
2. retain surface-consistent points using rendered/predicted depth;
3. erode masks before collecting point support;
4. remove isolated support with DBSCAN;
5. initialize global labels from the first processed view;
6. sequentially remap later local masks by overlap with projected labels;
7. accumulate per-point label votes with the upstream first-observation bias;
8. assign a final label by normalized majority vote and reject weak points;
9. reproject labeled points and merge compatible over-segmented masks.

Outputs:

- a dense GS point set with global instance ID and vote confidence;
- view-consistent per-instance binary masks;
- an exclusive label map for editor visualization only;
- unlabeled/uncertain support kept explicitly as ID `0`.

Step 5 registers the final masks as a read-only candidate layer named
`Split&Splat Split`. Its IDs belong to a run-local Split&Splat namespace, not
the editor's persistent-region namespace.

### Step 6: Inspect Split Geometry

Step 6 exposes two point-cloud views from the same run:

- `Split&Splat Global GS`: dense Gaussian means before instance assignment;
- `Split&Splat Labeled GS`: the same points colored by final global instance
  ID, with rejected or unlabeled points gray.

Optional diagnostics include vote confidence, observation count, DBSCAN
inliers/outliers, and per-frame visibility. Step 6 does not rerun Split; it
visualizes the canonical Step 5 artifacts.

### Step 7A: Prepare Per-Instance Training Sets

For each global instance:

- export its binary masks and masked multi-view RGB;
- select the matching labeled dense points as its initializer;
- retain the original world coordinate system and cameras;
- write a stable global-instance registry.

Do not invent a background/context component in the official baseline.
Unlabeled regions remain background as in the method. Their coverage must be
reported so an incomplete scene cannot be mistaken for a complete
reconstruction.

### Step 7B: Initial Per-Instance Reconstruction

Run the released per-instance 3DGS training independently for every retained
instance with its original iteration schedule and losses.

Monitor:

- input masks and masked RGB;
- RGB, opacity, and depth renders;
- per-instance Gaussian count and coverage;
- skipped or failed instances.

### Step 7C: Geometry-Guided Mask Refinement

For each instance and view:

1. render the visible instance Gaussians at full opacity;
2. project visible Gaussian means;
3. sample a compact spatially distributed positive-point set with the
   upstream greedy farthest-point procedure;
4. prompt SAM2 to obtain a candidate mask;
5. compare the original and candidate masks by IoU against the rendered
   geometry mask;
6. keep the better candidate; when no original mask exists, accept the SAM2
   candidate only above the paper's `tau_iou=0.95`.

Visualize `original mask | rendered geometry | SAM2 candidate | chosen mask`
and save the two IoU scores and the decision for every view.

The released ScanNet utility contains two discrepancies: it stops after about
half the cameras and uses `0.05` for the missing-mask gate. Our runtime adapter
processes every camera and applies the paper's `0.95`, without modifying the
third-party checkout.

### Step 7D: Retrain Instances

Retrain every retained instance using the chosen refined masks and the released
refinement schedule. Keep this output separate from the initial instance
models so the effect of mask refinement is measurable.

### Step 7E: Compose the Scene

Follow the released composition process:

1. compute paper Eq. 4 directional AABB collision scores and rank each
   unordered pair by `max(C_ab, C_ba)`;
2. merge the highest-collision compatible instance sets;
3. concatenate their Gaussians and retain instance identity;
4. reset opacity;
5. disable densification during short joint refinement;
6. progressively increase the mask-consistency weight following the upstream
   schedule;
7. repeat until one composed scene remains.

Monitor the collision matrix, merge tree, pre/post-merge renders, boundary
leakage, Gaussian counts, and mask loss. Composition improves splat overlap and
rendered boundaries; it does not guarantee mesh connectivity or a watertight
surface.

## Experiment B: User-Conditioned Mask Substitution

Implemented run ID: `split_splat_sam2_video`

Keep COLMAP, depth, initial global 3DGS, Split label propagation, per-instance
training, refinement, retraining, and composition identical to Experiment A.
Change only the initial mask source:

```text
Experiment A: Split&Splat four-grid SAM2 masks
Experiment B: selected Step 5 propagated masks + complete manual anchors
```

Convert every nonzero stable region ID in each frame into one binary input mask.
Feed those masks through the same Split stage rather than bypassing its 3D
consistency logic. Save the persistent-region provenance of every input binary
mask. Split&Splat may merge or renumber those masks, so its final instance IDs
remain a separate run-local namespace.

The `proposals` stage now dispatches between:

- `--proposal-source official_auto`: released 1/4/8/16-grid SAM2 masks;
- `--proposal-source propagation --propagation-method sam2_video`: the
  editor's exclusive persistent-region maps converted to binary masks.

Both runs derive the same `_shared/<dataset-cache-id>` and therefore reuse the
same staged cameras, depth, and global 3DGS.

The canonical pasteable command sequence is maintained in
[`RUN_OMEGA_SEGMENTATION.md`](RUN_OMEGA_SEGMENTATION.md#complete-two-run-command).
Its execution order is:

```text
split_splat_official:
  prepare -> global_gs -> proposals -> split
          -> splat_prepare -> splat_initial -> splat_masks
          -> splat_refined -> splat_compose

split_splat_sam2_video:
                         proposals -> split
                         splat_prepare -> splat_initial -> splat_masks
                         -> splat_refined -> splat_compose

split_splat_anchored_sam2_video:
                         proposals -> anchored_3d split
                         -> splat_prepare
                         -> optional unweighted splat_initial
```

The second and third runs resolve the same shared cache automatically. They
must pass `--proposal-source propagation --propagation-method sam2_video` on
their run-specific stages.

Experiment B remains the controlled proposal-source substitution: it does not
add custom loss weights or freeze manual anchors inside released Split&Splat.
Experiment C is explicitly our method. It preserves persistent IDs and source
mask shapes, uses completed frames as hard identity anchors during global-3DGS
voting, and only relabels connected propagated components when projected
Gaussians with manual-anchor provenance provide decisive evidence.
Propagated-only 3D labels may seed reconstruction but cannot relabel their own
source masks. Experiment C does not invoke SAM2 in the Split stage.

Experiment C writes paper-compatible instance masks and labeled Gaussian
subsets plus a `training_view_weights.json` sidecar. The sidecar is staged by
`splat_prepare`, but the released trainer does not consume it yet. Weighted
training and removal of the released Splat-stage SAM2 refinement are the next
explicit adaptation, not hidden behavior in the current baseline.

## Shared Run Layout

```text
interactive/experiments/split_splat/
  _shared/<dataset-cache-id>/
    shared.json
    01_input/
    02_global_gs/
    03_splat_support/
  runs/<run_id>/
    run.json
    compatibility_report.json
    progress.json
    logs/
    split/
      01_proposals/
      02_point_labels/
      03_consistent_masks/
    splat/
      01_instance_datasets/
      02_initial_models/
      03_mask_refinement/
      04_refined_models/
      05_composition/
      06_outputs/
    metrics/
```

The dataset cache is canonical for cameras, images, depth, and global 3DGS.
Every mask experiment references that cache and owns only proposal-dependent
artifacts. Step 5 layers and Step 6 point-cloud entries reference these files
instead of duplicating large masks or PLY files.

## Backend And UI Design

Use one `SplitSplatExperimentManager` shared by Steps 5-7.

- Step 5 owns `prepare`, `global_gs`, `proposals`, and `split`.
- Step 6 lists the unsegmented and labeled GS point artifacts.
- Step 7 owns `splat_initial`, `splat_masks`, `splat_refined`, and
  `splat_compose`.
  `compose`.
- The command-line runner and web UI call the same manager and stage contract.
- Each stage has `not_ready`, `ready`, `running`, `complete`, `failed`, and
  `stale` states.
- Long jobs stream frame/instance/iteration progress and write logs.

Terminal progress is emitted on `stderr` so `stdout` remains a valid JSON stage
summary. Upstream output is simultaneously retained in stage log files, and
the latest throttled state is written to each run's `progress.json`.

Extend the proposal-layer contract with an explicit label namespace. Existing
propagation outputs use `persistent_region`; the official Split&Splat output
uses `split_splat_instance`. This prevents the UI from treating a run-local
instance ID as one of the designer's persistent regions.

## Validation And Comparison

First verify each stage independently:

1. upstream four-grid masks match a direct run of `auto_seg.py`;
2. global GS camera renders align exactly with editor frames;
3. projected points agree with depth and COLMAP track orientation;
4. labeled points and final masks reproduce the upstream propagation rules;
5. instance initializers contain only the intended global label;
6. refinement decisions reproduce saved IoU values;
7. composition follows the collision order and loss schedule.

Compare Experiments A and B using:

- final per-view mask coverage, boundary F-score, and held-out anchor IoU;
- cross-view identity consistency;
- labeled and unlabeled 3D coverage;
- per-instance and composed RGB quality;
- rendered mask IoU and cross-instance leakage;
- missing regions, duplicate geometry, overlap, and cracks;
- Gaussian count, runtime, and peak VRAM.

## Implementation Order

1. Add the shared experiment manifest, stage state, and artifact registry.
2. Build and validate the non-destructive dataset adapter.
3. Integrate Murre depth and camera/depth scale checks.
4. Run upstream multi-grid SAM2 and expose its diagnostics.
5. Train and inspect the global depth-regularized 3DGS.
6. Integrate Split propagation and register Step 5/6 outputs.
7. Integrate initial per-instance training and visualization.
8. Integrate mask refinement and per-instance retraining.
9. Integrate progressive composition and final rendering.
10. Freeze Experiment A.
11. Add Experiment B by swapping only the mask-source adapter.
12. Run the same metrics and visual comparisons for both experiments.

Do not begin by patching OMeGa training. The first milestone is a complete,
externally reproducible Split&Splat baseline with every departure from upstream
visible in its run manifest.


## Implementation Status

The Split and Splat experiment runner is implemented in
`omega_local/segmentation/split_splat/` and
`scripts/run_omega_split_splat.py`:

- isolated run manifest, compatibility report, progress, and stage summaries;
- exact editor-grid image/camera/depth adapter with all-frame pose validation;
- isolated runtime pinned to the released `pycolmap==3.11.1` camera API;
- released four-grid SAM2 invocation in a disposable workspace;
- released depth-regularized global 3DGS invocation;
- released Split propagation with global Gaussian means supplied as its dense
  point support;
- shared inverse-depth calibration for the Depth Anything ablation, using the
  released median/deviation affine rule and projected global Gaussian means;
- empty-point CUDA launch protection and cancellation-safe child-process
  cleanup without changing non-empty projection math;
- canonical binary instance masks plus an exclusive editor-only preview;
- anchored global-Gaussian label voting that preserves persistent IDs, gives
  completed manual frames hard identity priority, and relabels source-mask
  components without changing their foreground support;
- per-Gaussian anchored confidence/provenance, sparse projected 3D support, and
  staged per-view training weights;
- read-only Step 5 mask registration and Step 6 global/labeled point-cloud
  registration;
- embedded SuperSplat rendering of the shared RGB 3DGS and each run's
  label-colored 3DGS, with Split-only descriptor fields excluded from browser
  artifacts;
- shared camera, RGB, and metric-depth support for Splat without duplicating
  those files between proposal runs;
- resumable per-instance initial training, released Gaussian/SAM2 mask
  refinement over every camera with the paper's missing-mask threshold, and
  refined training with the source's 1,000-iteration ScanNet settings;
- collision-driven composition with paper mask weights, opacity reset,
  instance-aware mask loading, densification disabled, final RGB/instance-ID
  Gaussian artifacts, and separately viewable post-composition object 3DGS
  files.

The staged runner is the only Split&Splat implementation in this fork. It keeps
the released paper baseline and the anchored adaptation under separate run IDs
and output directories.
