# Interactive Multi-View Segmentation For Reconstruction-Ready Captures

This document reframes the segmentation work as a research project rather than
only an editor feature. The target output is a consistent set of per-frame masks
for downstream reconstruction, such as object-wise 3DGS, layer-wise 3DGS,
region-conditioned OMeGa training, or later mesh/remesh control.

The central idea is:

```text
The user defines the segmentation intent on a small number of useful views.
The system uses SAM2, camera geometry, and optional 3D evidence to propagate,
check, and export that intent as a reconstruction-ready multi-view mask dataset.
```

This is intentionally not category segmentation. In architectural and graphics
capture tasks, the desired region may be an object, a facade plane, a material
patch, a trim element, a detail cluster, an occluder, or an arbitrary seam that
only matters for reconstruction. These boundaries can be ambiguous,
task-dependent, and impossible for a generic foundation model to infer without
user intent.

## 1. Research Thesis

Foundation segmentation models are useful accelerators, but they are biased
toward object-like regions. Reconstruction workflows often need task-defined
regions that do not correspond to semantic objects.

Our research question:

```text
How can a small number of user edits on a posed image capture dataset be turned
into a consistent, reconstruction-ready segmentation across all frames?
```

The key claim should be modest and sharp:

- SAM2 proposes and tracks candidate regions.
- The user decides what the regions mean.
- Geometry and view connectivity make the labels consistent across frames.
- The final product is not a pretty visualization; it is a structured mask
  dataset ready for reconstruction supervision.

## 2. Lessons From Baselines

The baseline work points to a clear direction.

### 2.1 Mesh-Based Or 3D-First Segmentation

Methods such as GeoSAM2-style mesh prompting or raw mesh segmentation are not a
good first source of designer-facing masks for our current setting. The OMeGa
mesh can be spatially useful, but it is still reconstructed, incomplete, and
locally messy around thin structures, railings, edges, and details.

What we learned:

- 3D labels are consistent because they live in one shared representation.
- Pixel boundaries are often not clean enough.
- Missing or folded geometry creates incomplete or wrong visible masks.
- The mesh is useful as soft visibility/evidence, not as the only truth.

### 2.2 SAM2Object / Video Propagation

SAM2 video propagation is valuable when a region is clear and persists across
nearby frames. It can preserve small objects better than independent automatic
mask generation.

What we learned:

- It is a good propagation accelerator.
- It should be local and reviewable, not blindly applied to the whole dataset.
- For unordered capture sequences, propagation should follow a view graph, not
  only frame index.
- It still inherits SAM2's object bias and can merge facade elements that a
  designer wants separate.

### 2.3 SAI3D

SAI3D is the strongest automatic baseline for persistent identity because it
fuses 2D proposal evidence through a 3D point/superpoint graph.

What we learned:

- The final 3D labels are much more consistent across views than independent
  2D masks.
- The quality depends heavily on the point cloud or mesh samples.
- Sparse COLMAP points can preserve useful details but are incomplete.
- OMeGa area samples give broad coverage but inherit mesh defects.
- The 3D labels are a good evidence layer, but the final per-view masks still
  need user control and clean boundaries.

### 2.4 Split&Splat / Proposal-To-3D Logic

Split&Splat-like logic is useful because it treats SAM2 proposals as 2D regions
that can be checked against projected 3D evidence.

What we learned:

- Proposal assignment to projected 3D support is a good primitive.
- The automatic result is still limited by the original SAM2 proposal quality.
- It is useful as a refinement/fusion idea, not enough by itself for
  task-defined building parts.

### 2.5 Gaussian Grouping / DEVA

Gaussian Grouping and DEVA-style object tracking are good references for
training-time identity features and video object mask propagation. In our tests,
the pseudo labels tended to over-group architectural elements and leave coverage
gaps.

What we learned:

- Optimizing identity features is interesting for later reconstruction.
- DEVA-like object tracking is not enough for facade/material/detail regions.
- This is a later optional extension, not the core interactive pipeline.

## 3. Project Contributions

The strongest contribution is not another fully automatic segmentation method.
It is an intent-driven multi-view segmentation system for reconstruction.

### Contribution A: User-Defined Reconstruction Regions

The system explicitly separates region intent from semantic category.

A persistent region can mean:

- object
- facade plane
- material patch
- trim/detail
- reconstruction layer
- user-defined seam
- ignore or uncertain area

This is important because reconstruction users care about controllable regions,
not only semantic object classes.

### Contribution B: Editable SAM2 Proposal Layer

SAM2 automatic masks are treated as editable local proposals, not final labels.
The user can pick, lasso, add/subtract, use SAM2 prompts, and update the proposal
map of a frame.

This gives a clear intermediate representation:

```text
RGB frame
  -> SAM2 automatic proposals
  -> user-corrected frame-local proposals
  -> confirmed persistent region IDs
```

The proposal ID is local to one frame. The region ID is global across the
dataset. Keeping these separate is essential.

### Contribution C: Keyframe-Centered Interactive Propagation

The user edits selected keyframes. SAM2 video propagation and geometry checks
turn those edits into candidate masks in nearby views.

The user should not need to label every image. The system should instead:

- select useful keyframes;
- propagate confirmed regions to overlapping frames;
- show uncertain or conflicting frames for review;
- let the user add more edits only where the system is unsure.

### Contribution D: Reconstruction-Ready Export

The output is a structured dataset:

- exclusive per-frame label maps;
- per-region visible masks;
- confidence maps;
- boundary uncertainty or trimaps;
- persistent region metadata;
- provenance: user-edited, propagated, geometry-confirmed, uncertain.

This is the bridge from segmentation to reconstruction.

## 4. Core Data Model

The project should use a small number of explicit data layers.

### 4.1 Raw SAM2 Proposals

Per-frame automatic proposal maps:

```text
frame_i:
  proposal_label_map[y, x] = local proposal id
```

These are independent per frame. Proposal `12` in one frame has no relationship
to proposal `12` in another frame.

Current editor status:

- implemented as a single integer label map per frame;
- stored under the interactive proposal run;
- visualized as overlay maps and proposal cards.

### 4.2 Edited Frame Proposals

The editable copy of the proposal label map for each frame.

User actions such as pick, lasso, and SAM2 prompts should update this same
representation. After an edit, the result should still be:

```text
frame_i:
  edited_proposal_label_map[y, x] = local proposal id
```

This keeps manual edits and automatic proposals compatible with later stages.

### 4.3 Persistent Regions

Persistent region IDs are global across the dataset:

```text
region_001 = "facade left plane"
region_002 = "railing"
region_003 = "door trim"
```

A user assigns one or more edited frame proposals, lasso regions, or SAM2 masks
to a persistent region. This assignment is the main statement of intent.

Recommended representation:

```text
regions.json
  region_id
  name
  color
  created_from
  user_confirmed_frames
  notes

keyframe_labels/frame_000123.png
  pixel -> region id, ignore, or uncertain
```

### 4.4 Propagated Region Hypotheses

Propagation should not immediately overwrite final labels. It should produce
hypotheses with confidence:

```text
candidate_masks/
  frame_000124/
    region_001_from_frame_000123_sam2.png
    region_001_from_frame_000123_sam2.json
```

Each candidate stores:

- region ID;
- source keyframe;
- method;
- confidence;
- propagation distance;
- SAM2 score if available;
- geometric consistency score if available;
- boundary uncertainty.

### 4.5 Final Label Maps

The final exported output is exclusive:

```text
final_label_maps/frame_000124.png
  each visible pixel -> one region ID, ignore, or uncertain
```

This is the dataset consumed by reconstruction.

## 5. Proposed Pipeline

The pipeline should be modular. Each phase writes inspectable outputs before the
next phase consumes them.

### Phase 0: Input Staging

Inputs:

- RGB images;
- COLMAP camera poses and intrinsics;
- optional COLMAP sparse tracks;
- optional predicted depth;
- optional predicted normals;
- optional rough point cloud or OMeGa mesh.

Outputs:

- ordered frame manifest;
- consistent image paths and camera metadata;
- optional geometry evidence paths.

This phase should not depend on segmentation.

### Phase 1: Proposal Initialization

Run SAM2 automatic mask generation for every frame at a chosen working
resolution.

Outputs:

- raw proposal label maps;
- proposal metadata;
- proposal overlays;
- proposal summaries.

This is already mostly implemented in the editor/pipeline. The important rule is
that this stage only creates local frame proposals. It should not assign
cross-view identities.

Current implementation:

- the editor can generate `interactive/proposals/sam2_auto_<width>/`;
- generated proposal maps are one integer label map per frame;
- loading a ready proposal run creates/repairs
  `interactive/proposals/sam2_auto_<width>_edited/`;
- edited proposal maps keep the same representation as raw proposal maps;
- proposal IDs are local to one frame and are not persistent region IDs.

### Phase 2: Keyframe Selection

Select frames that are useful for user edits.

The keyframe score should combine:

- pose/view coverage;
- overlap with other frames;
- image reliability;
- segmentation boundary usefulness;
- camera motion or appearance change;
- optional rare-region visibility.

A practical first version is a greedy coverage selector over a pose-based view
graph:

```text
view_graph edge(i, j)
  = exp(-rotation_angle(i, j)^2 / sigma_r^2)
  * exp(-translation_distance(i, j)^2 / sigma_t^2)

intrinsic_score(i)
  = 0.35 image_reliability
  + 0.30 boundary_usefulness
  + 0.20 proposal_usefulness
  + 0.15 rare_visibility

gain(i | S)
  = new_pose_graph_coverage(i | S)
  + 0.25 intrinsic_score(i)
  - 0.25 redundancy(i | S)
```

Current editor status:

- a segmentation-aware greedy keyframe selector exists;
- image reliability uses sharpness, exposure, contrast, and depth coverage;
- boundary usefulness uses RGB edges, SAM2 proposal boundaries, and depth edges;
- proposal usefulness uses SAM2 proposal coverage, count, entropy, and tiny-mask
  fraction;
- rare visibility uses low overlap with nearby pose-graph neighbors;
- pose-graph coverage uses camera centers and rotations; COLMAP track overlap is
  left as a later optional upgrade;
- proposal/depth evidence is read from the editor proposal maps when available,
  falling back to the staged proposal/depth paths in the frame manifest;
- selected frames are shown in the filmstrip.

Research point:

- show that keyframe selection reduces user effort while preserving coverage.

### Phase 3: User Proposal Editing On Keyframes

The user fixes the local proposal map of a keyframe.

Tools:

- generate local view evidence for inspection and future geometry-aware
  selection;
- switch the active frame background between RGB, predicted normal, depth,
  normal-edge, and depth-edge views;
- pick proposal;
- lasso pixels;
- add/subtract selection;
- SAM2 positive/negative prompts;
- run SAM2 to create a temporary selection;
- accept selection into the edited proposal map;
- cancel/undo;
- sort proposals by ID or area.

Output:

```text
interactive/view_evidence/
  normal_rgb/000124.png
  normal_npz/000124.npz
  normal_edges/000124.png
  depth_rgb/000124.png
  depth_npz/000124.npz
  depth_edges/000124.png
  metadata/000124.json

interactive/proposals/sam2_auto_<width>_edited/
  label_maps/000124.npy
  label_maps/000124.png
  overlays/000124.png
  metadata/000124.json
```

This phase is about correcting local 2D regions. It still does not require
global region identity.

Current editor status:

- `View Evidence` generation is implemented in the Phase 3 panel;
- the panel has one StableNormal button and one Depth Anything V2 button;
- each button generates its local evidence if missing, or regenerates only that
  evidence type if it already exists;
- generated data is stored in one editor-local evidence cache;
- the frame dropdown can visualize RGB, normal, depth, normal edges, and depth
  edges by swapping only the background image; point clouds, proposal overlays,
  and selected-pixel overlays remain separately controlled;
- the editor does not read the staged OMeGa/SAI3D normal or depth maps for this
  step; normals are generated with StableNormal and depth with Depth Anything V2.

Next geometry-aware selection operators should consume these saved evidence maps
rather than recomputing them inside the UI:

- normal-similarity flood fill for planar regions;
- depth-discontinuity or normal-edge snapping for lasso boundaries;
- mixed RGB/depth/normal graph cut refinement around a user-selected region.

### Phase 4: User Region Definition

The user maps edited local proposals into persistent region IDs.

Core interactions:

- create a new region from selected proposal pixels;
- assign selected proposal pixels to an existing region;
- merge persistent regions;
- split a region by drawing/editing a subset;
- rename a region;
- mark pixels as ignore;
- mark ambiguous boundary as uncertain.

Output:

```text
keyframe_region_maps/frame_i.png
regions.json
edit_log.jsonl
```

This is the first stage where user intent becomes explicit.

Critical design decision:

- proposal edits are local cleanup;
- region assignment is semantic/task intent;
- these should stay separate in UI and file structure.

### Phase 5: Local Propagation Candidates

For each confirmed keyframe region, propagate it to nearby frames.

Propagation methods to test:

- SAM2 video propagation over a local ordered window;
- propagation along a view graph instead of raw frame index;
- optional optical flow for adjacent image order;
- optional homography propagation for near-planar regions;
- optional projected 3D point support from COLMAP, SAI3D, or OMeGa samples.

The output is candidate masks, not final masks.

The current interactive propagation popup is exactly the right first test:

- top: propagated candidate;
- bottom: original local proposals;
- frame window around source keyframe;
- portrait orientation for inspection.

Next improvement:

- store candidate masks and confidence, not only preview them.

### Phase 6: Multi-View Consistency And Fusion

Fuse user-confirmed keyframes and propagated candidates into one visible label
map per frame.

For each pixel or proposal region, compute competing region scores:

```text
score(region r, frame i, pixel x)
  = user_confirmed_weight
  + propagation_confidence
  + proposal_overlap_score
  + geometry_consistency
  - boundary_uncertainty
```

User-confirmed pixels are hard constraints. Propagated and automatic evidence are
soft constraints.

Possible geometry consistency terms:

- projected COLMAP point labels agree across views;
- projected SAI3D/point-cloud label support agrees with the candidate;
- candidate boundary aligns with depth/normal/RGB discontinuities;
- candidate is visible and not behind rendered mesh/depth support;
- candidate does not conflict strongly with another high-confidence region.

Conflict resolution:

```text
if user-confirmed label exists:
    keep user label
else if best score is clearly above second best:
    assign best region
else if all scores are weak:
    mark ignore or unlabeled
else:
    mark uncertain
```

This should produce:

- exclusive visible label maps;
- confidence maps;
- uncertainty bands;
- conflict reports for user review.

### Phase 7: Review Loop

The system should guide the user to the next useful edit.

Review candidates:

- frames with high uncertain area;
- frames where two propagated regions conflict;
- frames far from any edited keyframe;
- frames where geometry consistency disagrees with propagation;
- frames containing rare regions with low support.

This can become an active-learning loop:

```text
edit keyframe
  -> propagate
  -> fuse
  -> rank uncertain frames
  -> user edits one more frame
  -> repeat
```

This is a strong research direction because it directly measures reduction in
manual work.

### Phase 8: Reconstruction-Ready Export

Export:

```text
segmented_dataset/
  images/
  cameras/
  label_maps/
  visible_masks/
    region_001/
    region_002/
  confidence/
  uncertainty/
  trimaps/
  metadata/
    regions.json
    frame_manifest.jsonl
    propagation_log.jsonl
    conflicts.jsonl
    export_config.json
```

Downstream reconstruction can use:

- positive pixels: high-confidence region mask;
- negative pixels: other visible regions;
- ignore pixels: uncertain bands;
- confidence maps: loss weights;
- persistent IDs: object-wise or layer-wise training splits.

## 6. Current Implementation Snapshot

The current editor is already a useful first prototype for Phases 1 to 3:
navigation, SAM2 proposal initialization, keyframe hints, local proposal
editing, and a non-destructive propagation probe.

### 6.1 Launch

Set the usual project paths:

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export PYTHON=$DT_ROOT/.venv/bin/python
export OMEGA_BUILDING_ROOT=$DT_ROOT/third_party/OMeGa_4_Building
export PACKAGE_ROOT=$DT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole
export OMEGA_RESULT_DIR=$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000
```

Launch the editor on the current dense SAI3D area-sample run:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_editor.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_area_samples_1024_dense \
  --host 0.0.0.0 \
  --port 8787 \
  --max-points 180000 \
  --label-source raw
```

The default `--label-source raw` loads the sampled SAI3D input points as an
unsegmented point cloud, so all points render as label `0`. This matches the new
pipeline: start from raw capture/point evidence, then let the user define
segmentation intent through proposals and regions.

Optional label sources:

```text
--label-source raw     sampled input points, all label 0
--label-source saved   previously saved interactive_labels.npy
--label-source sai3d   SAI3D final point_labels.npy
--label-source auto    saved, then sai3d, then raw
```

Open locally:

```text
http://127.0.0.1:8787
```

Open over Tailscale or LAN with plain HTTP:

```text
http://100.66.11.29:8787
```

The editor server is plain HTTP. A URL beginning with `https://` will not work
unless we later add TLS or Tailscale Serve.

### 6.2 Viewer Behavior

Current viewer behavior:

- center canvas shows the raw or labeled 3D point set;
- bottom filmstrip shows all frames;
- clicking a frame aligns the center view to that frame camera;
- portrait rotation is enabled for easier DSLR inspection;
- frame-to-frame camera motion is interpolated;
- during interpolation the RGB background is hidden and points remain visible;
- in an active frame, pan/zoom changes the 2D sub-frustum, not the capture
  camera pose, so RGB and projected points stay aligned;
- right-drag can leave the selected frame and continue in free 3D orbit.

This viewer design is important for later user studies because it lets users
move between image-space editing and 3D spatial context without changing the
underlying capture camera geometry.

### 6.3 SAM2 Proposal Layer

The editor keeps SAM2 proposals separate from SAI3D labels and from future
persistent region IDs.

Raw proposal output:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/<baseline-name>/interactive/proposals/sam2_auto_<width>/
  config.json
  progress.json
  summary.json
  frames.jsonl
  label_maps/
    000000.npy
    000000.png
  overlays/
    000000.png
  metadata/
    000000.json
```

Editable proposal output:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/<baseline-name>/interactive/proposals/sam2_auto_<width>_edited/
  label_maps/
    000000.npy
    000000.png
  overlays/
    000000.png
  metadata/
    000000.json
```

Both raw and edited proposals use the same representation:

```text
label_map[y, x] = 0 for unassigned/background
label_map[y, x] = positive local proposal ID
```

Proposal IDs are local to one frame and are intentionally not reindexed after
edits. Gaps are acceptable because downstream code should treat proposal IDs as
opaque labels.

### 6.4 Current Proposal Editing Tools

The current webpage also has a local `View Evidence` generator for Phase 3
debugging and future geometry-aware selection. It writes:

```text
interactive/view_evidence/
  normal_rgb/
  normal_npz/
  normal_edges/
  depth_rgb/
  depth_npz/
  depth_edges/
  metadata/
  summary.json
```

The evidence cache is local to the editor:

- the editor keeps a single local evidence cache;
- startup reads the cache if it exists;
- the StableNormal button generates or regenerates only normal outputs;
- the Depth Anything V2 button generates or regenerates only depth outputs;
- the viewer dropdown changes only the displayed background image, not camera
  projection, source-pixel coordinates, proposal overlays, or point visibility.

The current webpage supports a unified frame-local pixel selection model.
Different tools produce the same kind of selected pixel mask:

- `Pick`: click an existing proposal in the active frame;
- `Lasso`: draw a raw pixel region;
- `SAM2`: place positive and negative prompts;
- `Run SAM2`: convert prompts into a temporary selected region;
- `Deselect`: clear the current selected region;
- `Update Proposal`: save the composed selected pixels into the editable
  proposal label map;
- `Cancel`: discard the temporary selection;
- `Cmd/Ctrl+Z`: undo transient selection edits.

Selection semantics:

- plain action replaces the current selection;
- `Shift` adds;
- `Option`/`Alt` subtracts;
- the `+` and `-` buttons can pin add/subtract mode;
- the canvas shows one final composed selected boundary, not separate tool
  histories.

`Run SAM2` does not directly create a saved proposal. It produces a temporary
selection. The user can refine that selection with pick/lasso, then press
`Update Proposal` to save it into the edited proposal label map.

`Update Proposal` saves in the same format as the original SAM2 proposals:

- if the selected pixels fully cover an existing proposal ID, that ID can be
  reused;
- if the selection is partial or newly drawn, the edit creates a new
  `max_id + 1` proposal ID;
- unselected leftovers keep their old IDs;
- overlays and metadata are regenerated immediately.

This preserves the clean invariant:

```text
manual proposal edits are still just proposal label maps
```

### 6.5 Keyframe Suggestions

The editor can detect suggested frames for early user inspection. The current
implementation is a segmentation-aware greedy selector:

```text
intrinsic_score
  = 0.35 image_reliability
  + 0.30 boundary_usefulness
  + 0.20 proposal_usefulness
  + 0.15 rare_visibility

greedy_gain
  = pose_graph_new_coverage
  + 0.25 intrinsic_score
  - 0.25 redundancy_to_selected_keyframes
```

Evidence terms:

- `image_reliability`: Laplacian sharpness, exposure, contrast, and rendered
  depth coverage;
- `boundary_usefulness`: RGB Canny edges, SAM2 proposal label-map boundaries,
  and depth discontinuities;
- `proposal_usefulness`: SAM2 proposal coverage, proposal count, proposal area
  entropy, and tiny-proposal penalty;
- `rare_visibility`: pose-graph novelty, so unusual views are not swallowed by
  dense clusters;
- `pose_graph_new_coverage`: how many not-yet-covered neighboring views this
  keyframe can represent for later propagation.

This keeps the selector simpler than a full COLMAP-track optimization while
testing the important behavior we need: useful edit anchors that are reliable,
boundary-rich, and well connected to nearby views.

The selector uses the editor's generated/edited proposal label maps when they
exist. If they do not exist yet, it falls back to the staged `maskPath` and
`depthPath` entries in the SAI3D frame manifest. This makes keyframe detection
usable before an interactive proposal run, while still letting Phase 1 proposal
outputs inform Phase 2 once available.

For the current 214-frame DSLR run, the default automatic budget selected 27
keyframes.

Stored output:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/<baseline-name>/interactive/keyframes/keyframes.json
```

### 6.6 Propagation Test

The current propagation section is an experimental probe, not saved dataset
state yet.

Backend logic:

1. Compose the current active-frame pixel selection.
2. Build a short temporary image sequence around the selected frame.
3. Insert the selection as a SAM2-video mask prompt on the source frame.
4. Run SAM2 video propagation forward and backward.
5. Return temporary overlays for display only.

The popup compares:

- top: propagated edited mask over RGB;
- bottom: current SAM2 proposal overlay over RGB.

This is exactly the right role for propagation at this stage: it tells us
whether a user-confirmed edit can reduce nearby-frame editing work before we add
saved candidates, confidence, geometry checks, or final fusion.

### 6.7 Existing Data Source

The editor currently expects a completed SAI3D-style baseline directory:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/sai3d_area_samples_1024_dense/
  dataset/frame_manifest.jsonl
  dataset/posed_images/.../*.jpg
  dataset/posed_images/.../*.txt
  dataset/scans/.../points.pts
  mesh_labels/point_labels.npy
```

For the new pipeline, this SAI3D baseline is mainly a convenient source of
staged frames, cameras, and sampled points. It does not mean SAI3D labels are
the source of truth.

## 7. What Should Be Modular

The implementation should avoid one monolithic segmentation script.

Recommended modules:

```text
omega_local/segmentation/interactive/
  app.py                       # web API
  state.py                     # editor state
  proposals.py                 # raw/editable SAM2 proposal maps
  keyframes.py                 # keyframe selection
  view_evidence.py             # Phase 3 evidence job orchestration
  view_evidence_models.py      # StableNormal / Depth Anything runtime adapters
  view_evidence_images.py      # normal/depth visualization helpers
  view_evidence_store.py       # evidence cache, status, and metadata helpers
  sam2_session.py              # single-image SAM2 prompts
  sam2_video_propagation.py    # local propagation tests
  mask_edits.py                # compose selections and edit label maps
  static/
    css/                       # app shell, viewer, panels, filmstrip
    js/                        # frontend state, viewport, phases 1-3 tools

future modules:
  regions.py                   # persistent region IDs and assignments
  propagation_store.py         # save propagated candidates
  fusion.py                    # confidence scoring and final label maps
  export_dataset.py            # reconstruction-ready export
```

Recommended output layout:

```text
<model_dir>/segmentation/interactive/
  proposals/
    sam2_auto_<width>/
    sam2_auto_<width>_edited/
  keyframes/
  regions/
  propagation/
  fused/
  exports/
```

## 8. Critical Risks

### Risk 1: SAM2 Does Not Understand Reconstruction Intent

SAM2 often segments object-like regions, not facade planes, material changes, or
arbitrary reconstruction seams.

Mitigation:

- treat SAM2 only as an accelerator;
- make lasso/brush edits first-class;
- make user-confirmed region maps hard constraints;
- do not force automatic proposal identities.

### Risk 2: Capture Images Are Not A True Video

SAM2 video propagation assumes temporal coherence. DSLR capture frames can jump
in viewpoint or revisit the scene from different angles.

Mitigation:

- propagate only over local windows or view-graph neighborhoods;
- use pose/overlap/keyframe scores to choose propagation targets;
- keep propagation as candidates until reviewed or fused.

### Risk 3: 3D Geometry Is Useful But Imperfect

OMeGa mesh, point clouds, and depth estimates can be incomplete near thin
objects, railings, and high-detail boundaries.

Mitigation:

- use geometry as consistency evidence, not absolute truth;
- preserve uncertainty where geometry and image evidence disagree;
- allow user edits to override geometry.

### Risk 4: Final Masks Need Clean Boundaries

3D-consistent labels can still project to fuzzy or sparse per-view masks.

Mitigation:

- use edited SAM2/local proposals for pixel boundaries;
- use 3D evidence mainly for identity and consistency;
- export trimaps/uncertainty around hard boundaries.

### Risk 5: Too Many Regions Can Become Unmanageable

Architectural scenes can have many useful parts.

Mitigation:

- keep region hierarchy optional;
- start with flat region IDs;
- support merge/rename/color/group operations;
- let downstream export choose subsets or groups.

## 9. Evaluation Plan

The project should be evaluated as an interactive reconstruction tool, not only
as automatic segmentation.

### User Effort

Measure:

- number of edited keyframes;
- number of clicks/lasso strokes;
- time to reach usable masks;
- how many additional frames are requested by the review loop.

### Mask Quality

On a manually annotated subset:

- IoU per region;
- boundary F-score;
- false merge and false split counts;
- uncertain-pixel rate;
- coverage of visible target regions.

### Multi-View Consistency

Use posed views and optional sparse points:

- label agreement of shared COLMAP tracks;
- consistency of projected region support;
- conflict rate between propagated candidates;
- identity stability across wide viewpoint changes.

### Reconstruction Utility

Run downstream tests:

- object-wise or region-wise 3DGS quality;
- leakage between regions;
- ability to train/remove/edit a region independently;
- whether masks improve geometry/remesh settings or layer separation.

### Ablations

Compare:

- independent SAM2 proposals only;
- SAM2Object/video propagation only;
- SAI3D automatic labels only;
- user keyframes without propagation;
- user keyframes with SAM2 propagation;
- user keyframes with propagation plus geometry consistency;
- with and without keyframe selection.

## 10. Recommended Build Order

### Milestone 1: Stabilize Proposal Editing

Current status: mostly implemented.

Finish:

- raw point cloud plus RGB frame viewer;
- SAM2 proposal generation/loading;
- editable proposal maps;
- pick/lasso/SAM2 temporary selection;
- update proposal map;
- undo and explicit save behavior where needed;
- clear distinction between raw proposals and edited proposals.

Success criterion:

```text
A user can fix SAM2 proposal mistakes on one frame, and the edited proposal map
is stored in the same format as the original SAM2 proposal map.
```

### Milestone 2: Add Persistent Region IDs

Implement the layer above proposals.

Needed UI:

- region list;
- create region from current selection;
- assign current selection to existing region;
- rename/color region;
- merge/split region;
- mark ignore/uncertain.

Success criterion:

```text
A user can turn corrected frame-local proposals into named global regions on
selected keyframes.
```

### Milestone 3: Store Propagation Candidates

Extend the current SAM2 propagation preview into a saved candidate generator.

Needed:

- choose source keyframe and region;
- propagate to previous/next local window or view-graph neighbors;
- save candidate masks and confidence metadata;
- visualize propagated candidate vs edited proposal vs RGB.

Success criterion:

```text
One edited region on a keyframe produces inspectable candidate masks on nearby
frames without overwriting final labels.
```

### Milestone 4: Fuse Into Per-Frame Label Maps

Implement a first confidence-based resolver.

Needed:

- user-confirmed keyframes as hard labels;
- propagated candidates as soft labels;
- overlap resolution;
- uncertain-band output;
- simple confidence maps.

Success criterion:

```text
The system exports one exclusive visible label map per frame, plus confidence
and uncertainty, from a small number of edited keyframes.
```

### Milestone 5: Add View-Graph-Aware Keyframe And Review Loop

Turn keyframe detection into an active assistant.

Needed:

- pose/overlap-aware view graph;
- keyframe recommendation;
- uncertain-frame ranking after fusion;
- UI highlights for frames needing review.

Success criterion:

```text
The system recommends where the user should edit next and reduces total manual
work compared with uniform frame sampling.
```

### Milestone 6: Reconstruction Export And Downstream Test

Package the masks for reconstruction.

Needed:

- final dataset export;
- region metadata;
- confidence/uncertainty maps;
- training split scripts or examples for object-wise/layer-wise 3DGS/OMeGa.

Success criterion:

```text
The exported masks can directly supervise a downstream reconstruction run.
```

## 11. Near-Term Decision

The next implementation step should not be another automatic baseline. It should
be the persistent region layer on top of the current proposal editor.

Reason:

- proposal editing already fixes SAM2's local mistakes;
- propagation only makes sense after a proposal is assigned to a global region;
- final output requires persistent region IDs, not only edited proposal maps.

Immediate next task:

```text
Add a region panel and keyframe region map:
1. create region from selected proposal pixels;
2. assign selected pixels to existing region;
3. save keyframe region map;
4. visualize persistent region colors on the active frame;
5. use one persistent region as input to the existing SAM2 propagation preview.
```

This keeps the project modular and moves directly toward the research goal:
turning user intent into consistent multi-view masks for reconstruction.
