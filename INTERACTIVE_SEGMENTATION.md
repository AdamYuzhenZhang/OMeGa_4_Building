# Interactive Multi-View Segmentation For Reconstruction Supervision

This document reframes the segmentation work as a research project rather than
only an editor feature. The implemented output is a multi-view segmentation
supervision package for downstream reconstruction, such as object-wise 3DGS,
layer-wise 3DGS, region-conditioned OMeGa training, or later mesh/remesh
control. Dense, boundary-refined masks for every frame remain a later fusion
product rather than a prerequisite for using the current data.

The central idea is:

```text
The user defines segmentation intent directly on a small number of posed source
images, before a final dense representation is required. The system uses SAM2,
camera geometry, and optional 3D evidence to accelerate editing, suggest regions
in later frames, and package that intent as hard 2D anchors, soft candidate
masks, and sparse but globally linked 3D labels.
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
How can user-defined, potentially non-semantic region intent be transferred
from a small set of edited views into identity-consistent supervision for a
posed image collection while minimizing correction effort?
```

The key claim should be modest and sharp:

- SAM2 proposes and tracks candidate regions, but those candidates are not
  trusted as final labels.
- The user decides what the regions mean by creating confirmed keyframe anchors.
- Propagation turns confirmed anchors into editable priors for later frames.
- Sparse geometry provides globally consistent region identity even when it
  cannot provide dense image coverage.
- Geometry and view connectivity can later fuse or refine this evidence into
  dense masks when a downstream task requires them.
- The current product is not a pretty visualization or perfect ground truth; it
  is a structured supervision package with explicit evidence provenance.

### 1.1 Why Operate On Capture Frames

The method should be described as **pre-dense-reconstruction segmentation**, not
as segmentation before any reconstruction. COLMAP poses and sparse tracks are
already geometric products, but the pipeline does not require a finished mesh,
NeRF, or 3DGS whose errors could constrain the segmentation.

This addresses a real dependency cycle:

```text
segmentation is needed to reconstruct regions cleanly
  -> dense-representation segmentation requires a reconstruction first
  -> errors in that reconstruction limit the segmentation
```

The proposed ordering is instead:

```text
posed RGB images + sparse camera geometry
  -> user-defined persistent regions
  -> hard keyframe masks + soft candidates + sparse 3D labels
  -> region-aware dense reconstruction
  -> optional 3D-backed dense per-view masks
```

Source images retain color, material boundaries, thin structures, and
pixel-level edges that may be blurred, merged, omitted, or folded in a dense
reconstruction. Frame-space editing also lets the user express a region whose
meaning comes from the downstream task rather than from an existing 3D object.

This is not a claim that 2D interaction is always better than 3D interaction.
The representations have complementary strengths:

- frame-space interaction preserves source-image evidence and does not inherit
  dense-geometry errors;
- 3D interaction provides immediate global identity and occlusion-aware spatial
  context;
- the intended final system combines image-space intent with later 2D-3D
  consistency and fusion.

### 1.2 Testable Research Hypotheses

The publication should test three concrete hypotheses:

1. Iterative keyframe editing with propagated suggestions reaches a target mask
   quality with less user effort than independent frame editing.
2. Hard user anchors plus multi-view fusion produce better identity consistency
   than propagation alone without sacrificing image-space boundary quality.
3. Frame-first interaction is especially beneficial for non-object regions,
   fine boundaries, and scenes where the available dense reconstruction is
   incomplete or inaccurate.

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

### 2.6 Direct 3D And Representation-Backed Interaction

The closest comparison is not one method but two families:

- direct point-cloud interaction, represented by
  [Easy3D](https://simonelli-andrea.github.io/easy3d) and
  [AGILE3D](https://arxiv.org/abs/2306.00977), where users click directly on a
  voxelized or point-based 3D scene;
- rendered-view interaction on an existing dense representation, represented by
  [SAGA](https://arxiv.org/abs/2312.00860), SA3D, and ISRF, where 2D prompts are
  transferred into an already trained 3DGS or radiance field.

These methods have an important advantage: their labels live in one 3D
representation and are therefore globally consistent. Their limitation for our
setting is equally important: they assume that a sufficiently complete dense
scene already exists. Object-oriented 3D feature models can also require more
interaction for object parts or task-defined subdivisions.

Our comparison should not try to prove that frame interaction dominates every
3D workflow. It should test when source-frame interaction is more faithful and
whether later fusion can recover the global consistency normally supplied by a
3D representation.

## 3. Project Contributions

The strongest contribution is not another fully automatic segmentation method.
It is an intent-driven, pre-dense-reconstruction segmentation system for posed
image collections.

### Contribution A: Intent Before Dense Geometry

The system defines reconstruction regions on the original posed observations
before requiring the dense representation that those regions will supervise.
The output is representation-agnostic: the same supervision package can guide
OMeGa, object-wise or layer-wise 3DGS, a mesh pipeline, or another reconstruction
method.

### Contribution B: User-Defined Reconstruction Regions

The system explicitly separates region intent from semantic category.

A persistent region can represent, or later be extended to represent:

- object
- facade plane
- material patch
- trim/detail
- reconstruction layer
- user-defined seam
- ignore or uncertain area

This is important because reconstruction users care about controllable regions,
not only semantic object classes.

### Contribution C: Shared Pixel Selection From Multiple Tools

SAM2 automatic masks, propagated region candidates, lasso, SAM2 prompts, normal
grow, and RGB-D cue are treated as tools for producing the same temporary
selected-pixel mask. They are accelerators for local frame editing, not final
labels.

This gives a clear intermediate representation:

```text
RGB frame
  -> SAM2 automatic proposals
  -> selected pixels from pick/lasso/SAM2/geometry tools
  -> confirmed persistent region IDs
```

The proposal ID is local to one frame. The selected-pixel mask is temporary. The
region ID is global across the dataset. Keeping these separate is essential.

### Contribution D: Keyframe Anchors And Proposal Propagation

The user edits selected keyframes and marks a frame complete only when the
visible regions in that view are intentionally labeled. Complete keyframes are
hard anchors. SAM2 video propagation turns those anchors into candidate region
proposals in other frames.

The user should not need to label every image. The system should instead:

- select useful keyframes;
- propagate confirmed regions as editable suggestion layers;
- let the user accept, correct, or ignore those suggestions on later keyframes;
- let the user add more edits only where the system is unsure.

### Contribution E: Reconstruction Supervision With Provenance

The implemented editor assembles three complementary forms of supervision:

- complete user-edited keyframes as hard, exclusive 2D region anchors;
- propagation outputs as soft, method-specific candidate masks;
- COLMAP track labels as sparse, globally linked 3D region evidence.

A later export/fusion stage can additionally produce:

- exclusive per-frame label maps;
- per-region visible masks;
- confidence maps;
- boundary uncertainty or trimaps;
- persistent region metadata;
- provenance: user-edited, propagated, geometry-confirmed, uncertain.

This distinction is important. The current package is already valid input for
3D segmentation or region-aware reconstruction, but it must not be described as
complete dense per-view ground truth.

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

### 4.2 Frame-Local Pixel Selection

Step 3 is only a selection workspace. User actions such as pick, lasso, SAM2
prompts, normal grow, and RGB-D cue all produce the same transient selected
pixel mask:

```text
frame_i:
  selected[y, x] = true or false
```

The selected pixels are not a proposal layer and are not saved as final labels.
They can be refined by multiple tools, cleared, or turned into confirmed regions
in Step 4.

### 4.3 Confirmed Persistent Regions

Persistent region IDs are global across the dataset:

```text
region_001 = "facade left plane"
region_002 = "railing"
region_003 = "door trim"
```

A user turns selected pixels into a persistent region assignment. This
assignment is the main statement of intent.

Current representation:

```text
regions.json
  id
  name
  color
  assignments
  frameIds / frameAreas
  frameCount / pixelCount

keyframe_region_maps/000123.npy
  pixel -> region id or unlabeled 0
```

Each frame stores one exclusive region label map. A pixel can belong to only one
confirmed region at a time. Region operations overwrite pixels in that map, so
adding pixels to one region automatically steals them from any previous region.

Current Step 4 operations:

- `New Region`: selected pixels create a new global region ID;
- `Assign to Region`: selected pixels replace the active region on this frame;
- `Add Pixels`: selected pixels are added to the active region on this frame;
- `Subtract Pixels`: selected pixels are set back to background;
- `Clear Frame`: the active region is removed from the current frame only;
- `Delete All`: the active region is removed from metadata and all frame maps
  after explicit confirmation.

### 4.4 Propagated Region Proposal Layer

Propagation must not overwrite confirmed labels. Every method produces an
independent region-aware candidate layer:

```text
interactive/proposals/propagation/<method-id>/
  label_maps/000124.npy
  overlays/000124.png
  metadata/000124.json
```

Each propagated label ID is a persistent region ID, so the layer carries a
region suggestion while still acting only as a proposal prior. It is useful for
later keyframes because it often tracks user-confirmed regions better than fresh
SAM2 automatic proposals. The user can pick from this layer, refine the selected
pixels, and commit them to a confirmed region in Step 4.

Propagated proposal metadata must preserve its source persistent region:

```text
labelId == regionId
regionId
sourceFrameId
sourceFrameIds
proposalKind = propagated_region
```

Names and display colors are not duplicated into propagated proposal metadata.
The persistent region table owns region display properties, so renaming a
persistent region immediately changes propagated suggestions and proposal lists
that point to that region ID.

In the editor this appears as a Step 4 suggestion. When the selected pixels come
from a propagated proposal, Step 4 shows the source region name and manual-frame
previews. The user can `Accept Source` to write the selected propagated pixels
into that persistent region on the current frame, or leave the suggestion unused
and create a new region from the selected pixels.

The editor shows independently toggleable candidate proposal layers:

- `SAM2`: original per-frame automatic proposals;
- one layer for each ready propagation or recovery backend, labeled by its
  registered method name.

Pick only targets layers that are visible and ready. Candidate proposals remain
soft priors; confirmed region maps are the edited output. This is the core
design choice: propagation helps the next manual edit, but it is not treated as
truth.

### 4.5 Current Multi-View Supervision Package

Step 5 is a valid stopping point for reconstruction experiments. One run
contains:

```text
interactive/regions/
  regions.json                         # stable global identities
  keyframe_region_maps/*.npy           # hard user anchors

interactive/proposals/propagation/<method-id>/
  label_maps/*.npy                     # soft candidate masks
  overlays/*.png                       # derived visual previews
  metadata/*.json                      # source anchors and provenance

interactive/data/point_clouds/
  colmap_sparse_segmented.npz          # sparse 3D region identities
  colmap_sparse_segmented.ply
  colmap_sparse_segmented.json
  omega_final_vertices.npz             # exact strongest-mesh vertices
  omega_final_vertices.ply
  omega_final_clean_hybrid.npz         # cleaned vertices + sparse-face fill
  omega_final_clean_hybrid.ply
  omega_final_clean_hybrid_segmented.npz
  omega_final_clean_hybrid_segmented.ply
```

The evidence has different authority:

- completed keyframe region maps are hard labels;
- propagated masks are soft candidates and may be incomplete or wrong;
- segmented COLMAP tracks are sparse but globally consistent identity links;
- unsupported pixels remain unknown rather than becoming background.

This package can supervise segmentation after reconstruction, or segmentation
during reconstruction with lower weights on propagated and sparse evidence. It
does not claim that every visible pixel already has a clean final label.

### 4.6 Future Dense Label Maps

Dense label maps are an optional later export stage, after enough keyframes have
been confirmed and 3D-backed fusion has been evaluated. The exported output is
exclusive:

```text
final_label_maps/frame_000124.png
  each visible pixel -> one region ID, ignore, or uncertain
```

These maps are useful when a downstream method requires dense per-frame labels.
They should be produced by a separate fusion/refinement step that treats
user-confirmed keyframes as hard constraints and propagated/automatic
candidates as soft evidence.

## 5. Proposed Pipeline

The pipeline should be modular. Each phase writes inspectable outputs before the
next phase consumes them.

The editor UI and the research phases map as follows:

| Editor step | Implemented responsibility | Research phase |
| --- | --- | --- |
| 1. Prepare Data | Stage point clouds and generate StableNormal, Depth Anything V2, and DINOv3 evidence | Phase 0 |
| 2. SAM2 Proposals | Generate/load frame-local SAM2 proposals and suggest keyframes | Phases 1-2 |
| 3. Edit Proposals | Produce one transient selected-pixel mask with pick, lasso, SAM2, Normal Grow, or RGB-D Cue | Phase 3 |
| 4. Persistent Regions | Commit selected pixels to stable dataset-level region IDs and mark complete anchor frames | Phase 4 |
| 5. Propagate Regions | Run parallel candidate backends, explicit geometry passes, pair tests, and sparse COLMAP labeling | Phase 5 |
| 6. Segment in 3D | Run SAI3D over cleaned hybrid OMeGa points using automatic or anchor-weighted 2D evidence | Phase 6 |
| 7. Reconstruct Regions | Train region-aware 3DGS, 2DGS, or OMeGa components from hard 2D anchors, soft Step 5 candidates, and Step 6 3D labels | Phase 7 |

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

Current editor support:

- Step 1 exposes the reconstruction samples and aligned COLMAP sparse cloud as
  independent visibility layers. Toggling one never changes the indices used by
  the other or by 3D point selection.
- The aligned source is discovered at
  `pointcloud/colmap_sparse/colmap_sparse_points.ply`; the editor does not need
  another command-line path.
- On first use it stages a finite, exact-deduplicated copy cropped only to the
  padded reconstruction working bounds under
  `interactive/data/point_clouds/colmap_sparse.npz`.
- The cache preserves XYZ, RGB, and each point's source-row index so a later
  consistency module can join the visible points back to the original COLMAP
  track table without coordinate matching.
- The COLMAP layer is rendered with its source RGB colors. Its deterministic
  display subset is color-ordered internally only to keep canvas rendering
  responsive; this does not change the cache or point-track correspondence.
- `Cleaned (SOR)` switches that same display layer between the immutable raw
  cache and a derived Open3D statistical-outlier-removal result. The cleaned
  result uses `20` neighbors and a `2.0` standard-deviation ratio while
  preserving source RGB and source-row indices.
- Derived outputs live beside the raw cache as
  `colmap_sparse_cleaned.npz`, `colmap_sparse_cleaned.ply`, and
  `colmap_sparse_cleaned.json`. The JSON records the exact input signature,
  parameters, bounds, and retained/removed point counts.
- Running `COLMAP Tracks` also exports the positively labeled SfM points as
  `colmap_sparse_segmented.npz`, `colmap_sparse_segmented.ply`, and
  `colmap_sparse_segmented.json`. Their colors are the canonical persistent-
  region colors. Step 1 exposes this as the mutually exclusive `Segmented`
  COLMAP display option beside `Cleaned (SOR)`.

Proven cleanup methods should be compared as separate derived layers while the
raw COLMAP cache remains immutable:

1. **COLMAP observation filtering:** use PyCOLMAP's
   [`filter_all_points3D`](https://colmap.github.io/pycolmap/pycolmap.html)
   for reprojection error, negative depth, and triangulation angle, plus
   `filter_points3D_with_short_tracks` when testing a minimum track length.
2. **Statistical outlier removal:** use the established
   [Open3D statistical filter](https://www.open3d.org/docs/latest/tutorial/Advanced/pointcloud_outlier_removal.html)
   or its equivalent PCL implementation, which rejects points whose mean
   neighborhood distance is unusually large.

The source reconstruction already passes COLMAP's default `4 px` reprojection
and `1.5 degree` triangulation-angle filter. Its visible floaters correlate much
more strongly with short tracks: `93.8%` of the sparsest spatial one percent are
two-view tracks, compared with `34.2%` overall. On the staged cloud, the
implemented Open3D baseline uses the documented conservative example setting
(`20` neighbors, `2.0` standard deviations). It removes `16,905 / 557,040`
points (`3.0%`), of which `89.1%` are two-view tracks.
Radius filtering and unconditional deletion of every two-view track should not
be the first defaults because point density varies strongly and valid thin
architectural details may also have short support.

### Phase 1: Proposal Initialization

Run SAM2 automatic mask generation for every frame at a chosen working
resolution.

Outputs:

- raw proposal label maps;
- proposal metadata;
- proposal overlays;
- proposal summaries.

This is implemented in the editor/pipeline. The important rule is that this
stage only creates local frame proposals. It does not assign cross-view
identities.

Current implementation:

- the editor can generate `interactive/proposals/sam2_auto_<width>/`;
- generated proposal maps are one integer label map per frame;
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
- revise keyframe selection around a view graph rather than only frame order:
  cluster nearby/overlapping views and choose anchors that cover the graph while
  still favoring frames with reliable boundaries and useful proposal structure.
  This should support both the current video-propagation path and a future
  view-based propagation path.

### Phase 3: Pixel Selection On Keyframes

The user creates and refines a selected pixel region on a keyframe. This phase
does not write labels; it prepares pixels for Step 4 region operations.

Tools:

- generate local view evidence for inspection and future geometry-aware
  selection;
- switch the active frame background between RGB, predicted normal, depth,
  normal-edge, and depth-edge views;
- pick proposal;
- lasso pixels;
- normal-similarity grow from a clicked seed;
- RGB-D Cue graph cut from foreground/background clicks;
- add/subtract selection;
- optional `Lock regions` constraint that removes pixels already assigned to
  persistent regions from the transient selected-pixel mask;
- SAM2 positive/negative prompts;
- auto-run SAM2 prompts to create a temporary selection;
- clear selected pixels;
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
```

This phase is about creating a local 2D selection. It still does not require
global region identity, and it does not modify SAM2 or propagated proposal
layers. When `Lock regions` is enabled, each selection operation is composed
normally first, then intersected with the complement of the current frame's
persistent region map. This lets users select around confirmed regions without
accidentally stealing pixels from them; turning it off restores unrestricted
selection and is required when intentionally subtracting from confirmed regions.

Resolution policy:

- the staged frame manifest is the single source of truth for every 2D product
  in the interactive editor;
- RGB display images, SAM2 automatic proposals, Step 3 SAM2 prompted masks,
  SAM2 video propagation masks, StableNormal evidence, Depth Anything V2
  evidence, proposal overlays, selected pixels, and region maps must all
  share this same pixel grid;
- for the current `sai3d_area_samples_1024_dense` run, that grid is
  `1024 x 683`;
- original `sourceImagePath` frames may be higher resolution, but they should
  not be mixed into this editor run; full-resolution experiments should stage a
  separate full-resolution proposal source and baseline.

Current editor status:

- `View Evidence` generation is implemented in the Step 1 preparation panel
  next to raw point-cloud status and visibility;
- the preparation panel is organized as four sibling blocks: point clouds,
  StableNormal, Depth Anything V2, and DINOv3;
- the StableNormal block exposes only scene type and StableNormal mode
  (`stable` or `turbo`); model resolution, diffusion steps, ensemble, and batch
  size stay fixed as a quality preset;
- the Depth Anything V2 block exposes the checkpoint selector;
- each button generates its local evidence if missing, or regenerates only that
  evidence type if it already exists;
- incomplete evidence caches are resumable: the editor validates the PNG, NPZ,
  edge PNG, and metadata for each frame, skips only complete frames, and the
  button shows `Resume` until the selected evidence type reaches every frame;
- `Regenerate` is reserved for a complete evidence type and clears only that
  target before rebuilding it;
- generated data is stored in one editor-local evidence cache;
- evidence inference uses the staged manifest frame image, so StableNormal,
  Depth Anything V2, SAM2 proposals, Step 3 SAM2 masks, SAM2 video
  propagation, proposal overlays, and selected pixels all live on the same pixel
  grid;
- a full-resolution editor run is obtained by staging a full-resolution proposal
  source, not by mixing original capture images with a downsampled manifest;
- StableNormal uses the same quality path as the DSLR OMeGa preparation run:
  full `stable` variant, `1536` long-side processing resolution, `outdoor`
  masking, `10` inference steps, and one ensemble sample;
- Depth Anything V2 defaults to the outdoor `Large` checkpoint instead of the
  previous fast `Small` checkpoint;
- the frame dropdown can visualize RGB, normal, depth, normal edges, and depth
  edges by swapping only the background image; point clouds, proposal overlays,
  and selected-pixel overlays remain separately controlled;
- the editor does not read the staged OMeGa/SAI3D normal or depth maps for this
  step; normals are generated with StableNormal and depth with Depth Anything V2;
- `Normal Grow` and `RGB-D Cue` consume the saved evidence maps and return
  temporary selected pixels, exactly like lasso or SAM2 masks;
- `Normal Grow` exposes a continuous normal-angle slider;
- `RGB-D Cue` exposes the MRF smoothness `lambda` used by the paper.

Implemented geometry-aware selection operators:

- `Normal Grow`: flood-fill connected pixels whose StableNormal direction stays
  within the selected normal-angle threshold. This is a fast planar-region
  selector for architecture-style surfaces.
- `RGB-D Cue`: follows Feng et al., CVPR 2016, "Interactive Segmentation on RGBD
  Images via Cue Selection." We keep the paper's SLIC superpixels, geodesic
  foreground/background confidence maps, six labels
  `{color, depth, normal} x {foreground, background}`, alpha-beta swap, automatic
  image-boundary background prior, and GrabCut boundary refinement. Our
  substitution is cue data: Depth Anything V2 provides the depth cue, while
  StableNormal replaces the paper's depth-derived normal cue. `+` clicks are
  foreground cues and `-` clicks are background cues. Defaults are
  `lambda = 0.10` and `800` target superpixels.

Future geometry-aware selection operators should consume these saved evidence
maps rather than recomputing them inside the UI:

- depth-discontinuity or normal-edge snapping for lasso boundaries;
- strokes/lasso-to-foreground/background cue support for RGB-D Cue.

### Phase 4: User Region Definition

The user commits the transient selected pixels to persistent region IDs.

Implemented interactions:

- create a new region from selected pixels;
- replace an existing region's pixels on the current frame;
- add selected pixels to an existing region;
- subtract selected pixels from an existing region;
- rename or delete a persistent region;
- clear a region from one frame;
- mark a fully reviewed frame complete so it becomes a hard propagation anchor.

Future region roles can be explicit:

- `persistent`: important reconstruction region that should be propagated and
  tracked across frames;
- `context`: labeled local/background context that helps the user organize a
  frame but should not be propagated as an object target;
- `ignore`: pixels excluded from reconstruction supervision;
- `uncertain`: ambiguous pixels or boundaries reserved for later review/fusion.

The proposed `context` role is important for architecture captures. Some elements appear
only in a few views or are not part of the reconstruction task. Treating them as
full persistent regions can make SAM2 propagation more confusing, so they should
eventually be labeled locally without becoming propagation anchors. Context,
ignore, uncertainty, hierarchy, and direct persistent-region merge/split are
not implemented in the current Step 4 data model.

Output:

```text
interactive/regions/
  regions.json
  keyframe_region_maps/
    000123.npy
    000123.png
  overlays/
    000123.png
  edit_log.jsonl
```

This is the first stage where user intent becomes explicit.

Critical design decision:

- Step 3 selection is local cleanup and does not carry semantic identity;
- Step 4 region assignment is semantic/task intent;
- these should stay separate in UI and file structure.

Current implementation:

- the left panel displays persistent regions above active-frame proposals;
- the right panel has a Persistent Regions step after Edit Proposals;
- `New Region` creates a dataset-level region from the current selected pixels;
- `Assign to Region` replaces the active region on the current frame with
  selected pixels;
- `Add Pixels` adds selected pixels to the active region on the current frame;
- `Subtract Pixels` sets selected pixels back to background in the region map;
- `Rename` appears beside the name field only when a region is selected;
- `Clear Frame` and `Delete All` live inside the selected-region card because
  they act on the selected region;
- selecting a region shows a selected-region card with its name, current-frame
  status, a close button, and a horizontal manual-reference strip with the
  keyframes where that region was manually created or assigned;
- manual-reference frames are derived from current region pixels, so clearing
  the last pixels of a region on a frame removes that frame from the strip;
- each region has stable `id`, `name`, `color`, assignments, frame count, and
  pixel count;
- each keyframe region map is an integer label map where
  `region_map[y, x] = persistent_region_id`;
- assignment records track the contributing local proposal IDs, but local
  proposal IDs remain frame-local and non-semantic.
- a frame can be marked `Frame complete` after all visible persistent regions
  for that view have been labeled. This is separate from merely having some
  region pixels on the frame.

### Phase 5: Multi-Anchor Propagated Proposals

Complete keyframes are hard anchors. Each time the user finishes another
keyframe and marks it complete, the propagated proposal layer is rebuilt from
all complete keyframes, not incrementally accumulated from one source frame.
Partially labeled frames can still store persistent regions, but they are not
used as SAM2 video conditioning inputs.

All first-stage anchor-to-mask methods are parallel backends. Each receives the
same user edits and writes an independent candidate layer. A second-stage
geometry pass may consume one named candidate layer, but that dependency must
be explicit in the registry, fingerprinted, and shown as the comparison
reference. Methods never silently overwrite or fuse another result.

#### Shared Anchor-To-Mask Contract

Every backend receives:

- the canonical staged frame manifest at the editor resolution;
- the persistent-region registry with stable region IDs;
- every complete-frame region map as a hard user anchor;
- camera intrinsics and poses;
- an optional evidence bundle containing COLMAP tracks, pointmaps, depth,
  normals, local SAM2 proposals, and the view graph;
- a method-specific configuration stored with the run.

Every backend produces one complete `RegionCandidateLayer`:

```text
label_map[y, x] = 0                       for no candidate
label_map[y, x] = stable persistent ID    for a region candidate
confidence[y, x] in [0, 1]                when available
uncertain[y, x] in {0, 1}                 for unsupported or conflicting pixels
```

Each frame also records the source anchor frame IDs, method ID, runtime,
configuration, and evidence provenance. If a model does not expose a meaningful
confidence, it writes `NaN` and identifies the confidence as unavailable rather
than inventing a calibrated score.

Contract invariants:

- all methods use the same frame IDs, image grid, and persistent region IDs;
- complete user anchors are copied exactly into every result;
- missing geometry or weak correspondence remains unsupported evidence;
- methods never overwrite user regions or another method's output;
- outputs are streamed frame-by-frame so progress and partial previews work;
- the web app discovers methods from a registry and renders every result with
  the same layer, picker, and comparison components.

Current anchor-to-mask methods:

- `video propagation`: the current SAM2 video mode, using multiple complete
  keyframes as conditioning frames inside one ordered sequence;
- `bounded propagation`: a parallel SAM2 mode that adds complete keyframes as
  conditioning masks, then propagates only inside anchor-bounded intervals.
  Each interval is run once forward from the left anchor and once backward from
  the right anchor, keeping useful intermediate memory while limiting long
  chronological drift. Agreement and one-sided support are retained; conflicting
  labels are selected by anchor distance and SAM2 logit only when their weighted
  margin is at least `0.20`, otherwise the pixel remains unknown. This fusion is
  our controlled ablation, not an official SAM2 mode;
- `XMem++`: an external memory-VOS baseline that commits every complete
  keyframe mask to permanent memory before traversing the capture sequence;
- `Cutie`: an external memory-VOS baseline that commits every complete
  keyframe to permanent object memory and propagates with object-level memory
  features;
- `COLMAP Tracks`: a geometry-only diagnostic that labels original SfM points
  from completed-frame persistent regions and renders only their recorded 2D
  observations in other frames. It performs no SAM inference, proposal
  scoring, nearest-neighbor fill, or dense mask recovery. The same run saves a
  region-colored 3D point cloud for direct inspection in Step 1;
- `COLMAP Verify Video`: an explicit identity-refinement pass over the saved
  `SAM2 Video` layer. It preserves every foreground pixel and changes only a
  connected component's persistent region ID when exact COLMAP tracks provide
  enough reliable, high-margin votes. Missing, sparse, or mixed evidence keeps
  the video ID unchanged. This follows the 3D-guided mask-association principle
  of [MV3DIS](https://arxiv.org/html/2604.08916) and the global identity
  consensus of [MaskClustering](https://arxiv.org/html/2401.07745), adapted to
  sparse COLMAP tracks rather than dense RGB-D geometry;
- `Superpixel Geodesic` dense recovery: reads the completed `COLMAP Tracks`
  label maps as immutable hard seeds, then grows them over fine SLIC
  superpixels using RGB, Depth Anything V2, and StableNormal boundaries.
  Distance, boundary-barrier, and competing-label checks preserve an explicit
  unknown state instead of forcing every pixel into a region;
- `SAM2 Point Prompts` dense recovery: reads the same sparse label maps,
  coreset-samples each region as positive SAM2 prompts, removes prompt
  candidates too close to competing labels, and samples competing regions as
  negatives. It selects SAM2 multimask candidates by predicted IoU and prompt
  agreement, resolves overlap using per-pixel SAM2 logits, and restores exact
  COLMAP observations as hard labels. This adapts the projected-3D-prompt
  principle of [SAMPro3D](https://arxiv.org/abs/2311.17707); it does not claim
  to reproduce SAMPro3D's separate 3D instance-consolidation task;
- `V2-SAM`: a focused wide-baseline source-target diagnostic using the official
  Anchor, Visual, and Fusion experts. The open frame is the target, the selected
  persistent region supplies the identity, and its earliest manual occurrence
  is the sole source. PCCS selects the expert with the smallest normalized
  point-cycle error; that expert mask is saved directly, without cross-source
  consensus or merging.
- `VGGT-S`: a second focused source-target diagnostic using official VGGT
  correspondences and the released Union Segmentation Head. It consumes the
  complete source mask, pairwise VGGT feature maps, five sampled source/target
  points, and one iterative mask-refinement pass.

Sequence, memory, and sparse COLMAP methods are first-stage anchor-to-mask
controls. The Step 5 `Geometry Pass` registry contains explicit source-dependent
second stages: dense decoders read a saved sparse point layer, while COLMAP
Verify Video reads SAM2 Video and refines only identity. Every pass writes an
independent candidate layer. The focused pair tests isolate whether learned
wide-baseline correspondence works for a specific region before any larger
propagation strategy is considered. The implemented Superpixel Geodesic and
SAM2 Point Prompts decoders are retained as diagnostics; on the grove capture
they did not recover boundaries or unsupported thin structures reliably enough
to replace video candidates or become final labels.

The output is candidate masks, not final masks. Manual keyframe region maps stay
authoritative and are never overwritten by propagation.

Together with the hard region maps and segmented COLMAP cloud, these candidates
already form the current multi-view supervision package. A reconstruction loss
can respect their authority explicitly, for example:

```text
L_seg = L_hard_anchor
      + lambda_candidate * L_propagated
      + lambda_track * L_sparse_3D
```

with `lambda_candidate < 1`, visibility checks on sparse tracks, and no loss on
unsupported or uncertain pixels. Exact loss terms belong to the downstream
reconstruction experiment rather than to the editor.

The interactive propagation popup shows:

- top: propagated candidate, initially empty/loading and filled as each frame is
  generated;
- bottom: original local proposals, shown immediately as the propagation prior
  reference;
- all frames in the ordered capture sequence;
- portrait orientation for inspection.

The target comparison mode is driven by the method registry:

- select one or more completed methods without changing the anchor packet;
- show the same frame in synchronized method columns;
- keep RGB, raw SAM2 proposals, and confirmed anchors available as fixed
  reference columns;
- show run state, source anchors, confidence availability, and uncertainty;
- allow any ready method result to be toggled as a pickable proposal layer.

Current implementation:

- Step 5 discovers methods from one backend registry. Full-dataset methods use
  uniform `Run` and `Preview` actions. A second registry-driven selector exposes
  V2-SAM and VGGT-S through one `Test Pair` action;
- a run uses the opened frame only as the trigger, then builds one immutable
  input packet from every complete frame with a positive persistent
  `region_map`;
- the input packet validates frame dimensions and persistent IDs and stores a
  fingerprint so different methods can be compared on exactly the same anchors;
- for each persistent region ID, all manually accepted masks for that region are
  inserted into one SAM2 video state as conditioning masks for the same object ID;
- SAM2 starts its propagation pass at the first frame in the ordered capture
  sequence. All complete keyframes remain conditioning masks, including anchors
  later in the sequence, so the start frame controls the sequential prediction
  order rather than replacing the multi-anchor constraints;
- SAM2 propagates through the full ordered frame list and writes only its own
  `sam2_video` candidate layer;
- bounded SAM2 writes only its own `sam2_bounded` candidate layer;
- the official [XMem++](https://github.com/mbzuai-metaverse/XMem2) and
  [Cutie](https://github.com/hkchengrex/Cutie) repositories are called through
  isolated adapters behind the same backend interface;
- XMem++ and Cutie use dense temporary object IDs internally, then map their
  outputs back to stable persistent region IDs. Completed anchors are copied
  exactly into the saved output rather than replaced by model predictions;
- COLMAP Tracks loads the retained original DSLR SfM model rather than OMeGa's
  trackless dense seed cloud. A point receives the unique persistent ID with
  the most positive completed-frame observations; exact vote ties remain
  unknown. Points with track length below `2` or reprojection error above `4`
  pixels are rejected. Original distorted COLMAP observations are transformed
  through the recorded D01c pinhole undistortion, output rotation, and editor
  resize before either anchor sampling or target rasterization. Target masks
  contain only those observations, rasterized with a one-pixel marker radius;
  conflicting labels at the same raster pixel are cleared. Completed anchor
  frames remain exact user maps by the shared hard-anchor contract;
- both memory baselines use their published 480-pixel shorter-edge inference
  default internally and restore masks to the editor's staged 1024-pixel grid.
  `--xmem-size -1` and `--cutie-size -1` enable a more expensive native-grid
  comparison;
- the official [V2-SAM](https://github.com/jaychempan/V2-SAM) Anchor, Visual,
  and Fusion experts run in an isolated Python environment. The release does
  not provide a standalone PCCS runner, so pairwise expert choice reimplements
  the paper's point-cycle rule. We deliberately keep this as one region and one
  source-target pair rather than claiming an unproven multi-keyframe extension;
- DINOv3 ViT-L/16 features are extracted once at the official 768-pixel feature
  height and cached per frame under
  `interactive/view_evidence/dinov3_vitl16_768/`. Dataset-consistent PCA images
  are saved beside the feature cache for later UI inspection. Because ViT-L/16
  emits a 16-pixel patch grid, the enlarged PCA image is intentionally coarse
  and should not be read as a pixel-level prediction. Step 1 owns an explicit,
  resumable DINOv3 generation control beside StableNormal and Depth Anything;
  V2-SAM validates and reuses the cache without loading DINOv3 when it is
  current;
- the official [VGGT-S](https://github.com/buaa-colalab/VGGT-S) source,
  [paper](https://arxiv.org/abs/2604.13596), and released `main_exp.pth` head are
  connected without modifying VGGT's track head. The adapter calls the same
  feature extractor and tracker explicitly, uses the published `518` input,
  50-point locator, 10-point median-outlier removal, five-point segmentation
  prompts, source-mask conditioning, dynamic 3x target crop, and one refinement
  pass. Its checkpoint is trained for Ego-Exo objects, so architectural transfer
  quality is an empirical domain-shift test;
- the released V2-SAM sparse-correspondence configuration fixes DINOv3 feature
  height to `768`. DINOv3 supports larger inputs technically, but a 1024-height
  variant changes token count, memory cost, and the scale at which V2-SAM's
  correspondence thresholds were tuned, so it remains a future ablation rather
  than the default method;
- V2-SAM stores per-frame source scores, expert cycle errors, selected experts,
  confidence, margin, and uncertainty under
  `interactive/proposals/propagation/v2sam_pair/v2sam_diagnostics/`;
- the released learned Visual and Fusion weights use the `ego2exo` profile.
  They are an official reproducible baseline, but their domain does not match
  architectural capture. The Anchor expert and PCCS are training-free;
- generated propagated overlays are saved frame-by-frame during SAM2 video
  inference, so the preview can update live instead of waiting for the full run;
- propagation uses the same SAM2 checkpoint/config and the same staged frame
  resolution as the editor; when the staged frame is already a correctly sized
  JPEG, it is symlinked or copied into SAM2's temporary video folder instead of
  being re-encoded;
- each propagation method is stored as an independent proposal layer. None of
  them overwrite raw SAM2 proposals, persistent keyframe maps, or final labels;
- every saved candidate map is validated against the staged frame dimensions
  and registered persistent IDs, and complete anchors are copied exactly;
- Step 3 layer toggles and Step 5 method options are generated from the registry,
  so a new backend does not require method-specific UI branches;
- propagated proposal metadata stores `sourceFrameIds` for the persistent region
  that produced the suggestion, so Step 4 can show its manual reference frames.

Next improvement:

- freeze and export one supervision manifest that references hard anchors,
  method-specific soft candidates, and segmented sparse tracks;
- add synchronized multi-method columns to the comparison preview;
- evaluate the package in 3D segmentation and region-aware reconstruction;
- use method disagreement and unsupported areas to improve keyframe review.

### Phase 6: 3D Segmentation Experiments And Future Fusion

Step 6 now provides a registry-driven place to test existing 3D segmentation
methods against the strongest optimized OMeGa geometry. Its first backend is
SAI3D. Each run is defined by these independent choices:

- a 3D method, initially SAI3D;
- a 2D evidence source;
- for propagated evidence, a completed propagation result and manual-frame weight;
- a point and superpoint budget.

The first implemented input is **SAM2 Automatic**. It reuses the exact 1024-pixel
proposal label maps already generated in Step 2 and uses a cleaned hybrid point
cloud extracted from the strongest OMeGa mesh.
The editor stages only a small proposal-source manifest; it does not duplicate
the RGB images, poses, or masks. SAI3D then performs its original progressive
superpoint grouping from per-view mask agreement. The editor defaults to `8k`
superpoints, matching SAI3D's original scale; its dense affinity implementation
makes the earlier `60k` experimental setting unnecessarily expensive.

The parallel **Propagated + Manual Anchors** input uses any completed full-run
propagation layer; `SAM2 Video` is the default. The selected propagation supplies
one persistent-region label map per view. On frames marked complete, the
propagated map is replaced by the user-confirmed region map. Those manual views
then receive the selected `2x`, `4x`, or `8x` confidence, with `4x` as the
default:

```text
affinity(a, b) =
  sum_i w_i confidence_i(a, b) similarity_i(a, b)
  / sum_i w_i confidence_i(a, b)

w_i = w_anchor  for complete user-confirmed keyframes
      w_soft    for propagated frames, with w_anchor > w_soft
```

The implementation realizes integer `w_i` by repeating the corresponding label
and visibility column before SAI3D computes affinity. Repetition is exactly
equivalent to the weighted numerator and denominator above. It leaves SAI3D's
per-view label distribution, confidence equation, graph, threshold schedule,
and progressive region growing unchanged. The staged manifest and experiment
summary record each frame's weight and whether it was a manual anchor.

Run outputs live under:

```text
interactive/3d_segmentation/
  inputs/propagated_weighted_<source-id>_w<weight>/
  runs/sai3d_propagated_weighted_<source-id>_w<weight>/
    dataset/
    mesh_labels/
    logs/
    experiment.json
```

Step 1 gives `SAI3D Segmentation` its own result card beside the geometric
point-cloud sources. Each run has an independent checkbox and point-cloud
cache, labeled by its 2D evidence source and manual-anchor weight. Multiple
results can be shown together for comparison, and changing Step 6 settings does
not change which saved run a Step 1 row references. These viewer layers do not
mutate the editor's baseline samples, optimized mesh vertices, or persistent
regions. Step 6 is responsible only for choosing and running experiments.

Split&Splat registrations additionally expose full Gaussian artifacts in a
separate Step 1 `Split&Splat` card. The shared global RGB reconstruction appears
once. Each experiment then presents its input proposals, final Split masks,
Split-labeled global 3DGS, categorical composed geometry, an RGB object selector
derived from the shared reconstruction, and a composed-geometry object
selector. This distinction is intentional: the released per-object
Split&Splat initializer is geometry-only, while the shared global model carries
the valid radiometric appearance. The official and SAM2 Video experiments
remain visually separate without expanding hundreds of objects into hundreds
of controls.

The corresponding per-view results also appear as read-only Step 3 layers under
`Frame Proposals` and `Split-Refined Masks`; 2D and 3D use the same run-local IDs
and deterministic colors. Checking a Gaussian artifact renders it directly in
the main camera-synchronized viewport and hides the software point layers.
Selecting any point-cloud source switches back. This camera-synchronized
viewport is the only Gaussian viewer path in the editor, avoiding duplicate
model loads and divergent display settings.
The embedded renderer uses no display tone mapping and a high-precision
floating-point accumulation target so low-opacity, high-dynamic-range Gaussian
colors follow the Split&Splat training renderer as closely as the browser
allows.

The parallel **Anchored 3D Split** experiment addresses a specific failure of
the released Split stage on user-authored architectural regions. Released Split
rediscovers run-local identities and uses SAM2 to recover masks; this can change
correct manual boundaries or remove intended regions. The anchored path instead:

1. replaces propagated maps with exact persistent-region maps on completed
   manual frames;
2. z-buffers the shared global 3DGS against itself and accumulates region votes
   on its Gaussian means, with completed frames weighted more strongly;
3. keeps persistent IDs rather than clustering or renumbering them;
4. assigns every directly observed Gaussian and completes the small unobserved
   remainder by confidence-weighted local 3D consensus;
5. projects the complete ownership field into each view and relabels a whole
   connected source-mask component when the 3D majority is clear;
6. preserves every foreground pixel and passes manual masks through exactly;
7. partitions every global Gaussian into a persistent-region initializer.

This is intentionally closer to SAI3D-style 3D association than to SAM2 mask
refinement. It uses geometry to correct identity, not to invent a cleaner 2D
boundary. Confidence and provenance describe ownership reliability but do not
delete uncertain geometry. The editor exposes the dense anchored masks,
complete projected 3D ownership, and labeled global 3DGS as separate
diagnostics. Per-Gaussian confidence/provenance and per-frame manual weights
are saved for the later reconstruction stage.

The optimized mesh exposes two user-facing point representations.
`Mesh Vertices` preserves every finite vertex and therefore reveals OMeGa's
native, highly nonuniform tessellation. `Clean Hybrid` first removes
unreferenced vertices and only disconnected components that are negligible by
both face count and surface area. It retains every remaining native vertex,
then adds one area-sampled surface point only in a voxel with no retained
vertex. This preserves extra samples in complex regions while setting a
coverage floor on large planar faces. Step 5 geometry projection and new Step 6
experiments both use this hybrid source; raw vertices remain available for
diagnosis.

After these 3D tests, a later fusion stage may optionally turn complete
user-confirmed keyframes and propagated candidates into one visible label map
per frame. This remains future work and is not required for calling the current
Step 5 outputs valid reconstruction supervision.

This phase is where SAM2Object-, SAI3D-, GeoSAM2-, or Split&Splat-style ideas
become useful again. The difference is that their input evidence should no
longer be raw automatic SAM2 proposals alone. It should be the stronger evidence
created by this editor:

- complete user-confirmed keyframe region maps as hard anchors;
- propagated region proposals as soft, reviewable candidates;
- 3D point/mesh visibility as consistency evidence;
- RGB/depth/normal boundaries as refinement cues.

This is the intended research split: interaction defines reliable region intent;
later multi-view fusion turns those anchors into dense masks.

#### Three Evidence Tiers

The planned methods cover three complementary evidence tiers:

1. **Exact sparse geometry:** original COLMAP tracks transfer user labels only
   along measured multi-view observations. This evidence is precise but sparse.
2. **Learned cross-view correspondence:** focused V2-SAM and VGGT-S pair
   diagnostics, plus future 3AM and MV-SAM-style features, test relations across
   large camera changes where image sequence order is weak.
3. **Dense 2D candidates:** SAM2 automatic proposals, video-propagated masks,
   and RGB/depth/normal boundaries recover full-resolution image boundaries.

Geometry establishes identity and association. Dense 2D candidates establish
the final visible support. Sparse or unreliable geometry is represented as
unknown evidence instead of being interpreted as background.

Boundary quality is handled here, not in propagation. Propagated masks can have
fuzzy edges, seams, holes, or local over-merge artifacts because SAM2 and SAM2
automatic proposals are imperfect. Fusion should treat propagated masks as soft
region evidence or interior support, then refine final boundaries using RGB
edges, StableNormal edges, Depth Anything discontinuities, local proposals, and
3D visibility. The final export should include uncertainty/trimap regions around
hard boundaries when evidence disagrees.

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

#### First Geometry-Backed Test: COLMAP Track Consensus

The first diagnostic uses the sparse reconstruction as a correspondence graph
and deliberately rasterizes only its labeled observations. A
[COLMAP 3D point](https://colmap.github.io/format.html#points3d-txt) stores a
track of `(image_id, point2D_idx)` observations, so one point provides measured
pixel correspondences across several views. This stage asks whether the sparse
identity transfer is correct before any dense decoder can hide its failures.

For point `p`, persistent region `r`, and complete anchor set `A`, accumulate:

```text
vote_p(r) = sum over observations (i, x) in track(p), i in A
              [anchor_i(x) == r]

label_p = argmax_r vote_p(r)
```

Anchor pixels are hard evidence. A unique winning label is accepted; exact ties
remain unknown. Before voting, points with track length below `2` or COLMAP
reprojection error above `4` pixels are rejected.

For target frame `i`, the backend visits only the observations already recorded
in that image's COLMAP track. The distorted SfM keypoint is mapped through the
per-frame pinhole export metadata, including its recorded `180`-degree rotation
for the grove dataset, then resized onto the editor grid. A labeled point is
drawn with a one-pixel marker radius. Pixels touched by different region IDs are cleared.
There is no arbitrary reprojection into unseen cameras, nearest-neighbor fill,
SAM call, or proposal assignment. Complete anchor frames remain the exact user
masks under the shared backend contract.

#### Second Geometry-Backed Test: COLMAP-Verified Video Identity

This pass tests whether sparse geometry can correct video identity drift without
being asked to reconstruct a boundary. It reads the complete `SAM2 Video` label
map and treats each connected component as an immutable pixel shape.

For component `C` and persistent region `r`, exact COLMAP observations inside
the component vote with anchor-consensus, reprojection-error, and track-length
weights:

```text
G(C, r) = sum_{p observed in C} weight(p) [label(p) == r]
          / sum_{p observed in C} weight(p)
```

The source identity is replaced only when the winning alternative has at least
three tracks, at least `72%` of weighted support, a `0.28` margin over the next
candidate, and enough observations for the component area (one winning track
per `5000` pixels). Otherwise the pass abstains by retaining the video identity.
This area-scaled gate prevents a few sparse or floating points from relabeling a
large mask. Complete anchors are copied exactly, and the foreground footprint is
unchanged across the source and result.

The run writes `propagation_colmap_verify_video` as a separate toggleable layer.
In Step 5, first run `Video`, then choose `COLMAP Verify Video` under `Geometry
Pass`. Its preview places the verified result above the original Video layer.
Per-frame diagnostics record agreements, changed components, low-support
abstentions, ambiguous votes, and every source-to-target ID change.

The implemented dense-recovery stages consume the saved `COLMAP Tracks` maps
without rerunning COLMAP voting or camera projection. `Superpixel Geodesic`
treats their exact observations as hard seeds on
approximately `10,000` fine SLIC superpixels per 1024-grid frame. It does not
alter the sparse source. A superpixel becomes a propagation seed only when one
persistent ID owns at least
`75%` of its positive sparse pixels; mixed seed nodes remain unknown. Adjacent
superpixels receive a boundary cost

```text
B(p, q) = 0.02
        + 0.45 normalized_RGB_boundary
        + 0.30 normalized_log_depth_boundary
        + 0.25 normalized_normal_angle_boundary
```

For each region, marker-controlled minimax propagation records the path whose
largest crossed boundary is smallest:

```text
D_r(q) = min over paths seed(r) -> q  max edge B(p, q)
```

A node is assigned only when its best region is within `160` pixels of a seed,
has barrier at most `1.25`, and beats the next region by at least `0.10` after a
small distance penalty. Otherwise it remains label `0`.

`SAM2 Point Prompts` instead treats the sparse labels as 3D-grounded positive
and negative evidence for the shared SAM2 image predictor. Each visible region
receives at most 16 spatially diverse positive prompts and 16 nearby competing
negative prompts. Candidate masks must retain at least `75%` of positive
prompts, reject at least `75%` of negatives, and cover less than `90%` of the
image. Surviving region masks compete per pixel using their SAM2 logits. Pixels
outside every accepted mask remain unknown.

`2D/3D CRF` is a controlled adaptation of
[Semantic Instance Annotation of Street Scenes by 3D to 2D Label Transfer](https://openaccess.thecvf.com/content_cvpr_2016/papers/Xie_Semantic_Instance_Annotation_CVPR_2016_paper.pdf)
(Xie et al., CVPR 2016). It keeps two random fields: fine image superpixels
`X` and visible COLMAP points `Y`. Its Potts energy is

```text
E(X, Y) = unary_2D(X) + unary_3D(Y)
        + pairwise_2D(X) + pairwise_3D(Y)
        + projected_pairwise_2D_3D(X, Y)
```

The 2D unary uses the color distribution and distance to projected labeled
points. The 2D pairwise graph combines local RGB/depth/StableNormal boundaries
with bilateral neighbors. The 3D graph connects spatial neighbors and weights
them by local PCA-normal agreement. A projected point is coupled to its image
superpixel, and eight damped mean-field iterations update both fields. Pixels
are decoded only when confidence, competitor margin, normalized entropy, and
support distance pass their thresholds; label `0` remains an explicit unknown.

This test is not presented as the paper's unreleased full implementation. Fine
superpixels and local/bilateral k-NN graphs approximate its fully connected
pixel kernels. Persistent labels voted from manual keyframes replace its rough
street-scene cuboids/ellipsoids; the street-specific fold/curb unary and learned
weights are unavailable and omitted. The retained 2D unary, 3D unary, 2D
pairwise, 3D pairwise, 2D/3D coupling, and mean-field structure make the test a
useful direct adaptation rather than an unrelated fill heuristic.

Original sparse pixels and complete manual anchors are copied exactly by all
dense decoders. Conservative abstention is important: textureless support may
fill between nearby observations, while an unsupported surface or ambiguous
seam is not silently painted. The point-field interface is intentionally
separate from inference, so a later MASt3R adapter can replace sparse COLMAP
observations without changing the CRF or its output contract.

`Superpixel Geodesic` is our first dense decoder and multi-region adaptation of the
sparse-3D-link and superpixel graph
formulation in
[Multi-view Object Segmentation in Space and Time](https://openaccess.thecvf.com/content_iccv_2013/papers/Djelouah_Multi-view_Object_Segmentation_2013_ICCV_paper.pdf).
The paper jointly optimizes binary foreground/background labels with graph
cuts; our decoder uses hard persistent-ID seeds and confidence-aware geodesic
growth so many user-defined regions and an explicit unknown label can coexist.
Future proposal lifting, graph-cut, random-walker, or learned recovery methods
must implement the same sparse-layer-to-dense-layer contract, making them
parallel ablations rather than modifications of COLMAP Tracks.

A later proposal-fusion stage can also let labeled tracks score existing 2D
proposal nodes:

```text
support(frame i, proposal q, region r)
  = weighted labeled-track observations inside q for r
    / weighted labeled-track observations inside q
```

That stage could combine sparse but reliable correspondences with foundation
model proposals. A later global graph can
connect proposals that share labeled tracks and use multi-view consensus before
local RGB/normal/depth boundary refinement.

This design is grounded in three prior results:

- [Interactive Object Segmentation from Multi-View Images](https://doi.org/10.1016/j.jvcir.2013.02.012)
  alternates 3D graph-cut support and precise 2D segmentation from a few edited
  views. It is close to our interaction problem, although it is a binary
  foreground method and no maintained implementation was located.
- [Sparse Multi-View Consistency for Object Segmentation](https://inria.hal.science/hal-01115557)
  demonstrates that sparse 3D samples can communicate segmentation evidence
  across calibrated views without requiring dense reconstruction.
- [MaskClustering](https://pku-epic.github.io/MaskClustering/) represents 2D
  masks as graph nodes and uses agreement across many views rather than only
  adjacent-frame overlap. Its released implementation assumes RGB-D scans and a
  reconstructed point cloud, so we should reuse its global consensus principle
  around our COLMAP tracks and hard user anchors rather than force its complete
  ScanNet-oriented pipeline onto this dataset.

For the current grove capture, the retained original COLMAP model contains
`583,405` tracked points and `3,330,632` image observations, with mean track
length `5.71`. Coverage will still be weak on textureless walls, so track
consensus should be high-precision evidence with an explicit unknown state, not
the sole dense-mask generator.

#### Ordered Parallel Test Queue

Implement and evaluate one backend at a time. A backend advances only after it
can read the fixed anchor set, satisfy the shared contract, and appear beside
the controls in the web comparison view. Availability below was checked in July
2026.

| Order | Backend | Prior work and status | Test hypothesis |
| --- | --- | --- | --- |
| 0 | SAM2 Video, SAM2 Bounded, XMem++, Cutie | Existing controls; official external repositories are already connected | Quantify the limit of chronological and memory-based propagation from exactly the same anchors. |
| 1 | COLMAP Track Consensus | **Implemented as sparse masks and a segmented 3D point export.** Our adapter is grounded in [Sparse Multi-View Consistency](https://inria.hal.science/hal-01115557) and [Interactive Object Segmentation from Multi-View Images](https://doi.org/10.1016/j.jvcir.2013.02.012). | Test whether exact sparse correspondences provide high-precision region identity on wide-baseline views. Sparse observations remain a useful output even where dense recovery abstains. |
| 2 | 2D/3D CRF Dense Recovery | **Implemented as an adaptation.** Retains the multi-field energy and mean-field inference from [Xie et al. 2016](https://www.cvlibs.net/projects/label_transfer/), while documenting substitutions required by its unavailable code, parameters, and street primitives. | Test whether joint image/point regularization recovers cleaner dense masks than independent superpixel growth or SAM2 point prompts. Start with COLMAP, then swap only the point-field provider to MASt3R. |
| 3 | V2-SAM Focused Pair | **Implemented as a diagnostic.** [Official code](https://github.com/jaychempan/V2-SAM) and [paper](https://arxiv.org/abs/2511.20886). Official Anchor, Visual, and Fusion experts are retained for one selected persistent region and one manual-source/open-target pair; PCCS is reimplemented from the paper because its runner is absent from the release. | Test whether wide-baseline DINOv3 correspondence works at all for a chosen architectural region before comparing it with chronological VOS. Report Anchor/Visual/Fusion choices separately because the released learned weights use the `ego2exo` domain. |
| 4 | VGGT-S Focused Pair | **Implemented as a diagnostic.** [Official code](https://github.com/buaa-colalab/VGGT-S) and [paper](https://arxiv.org/abs/2604.13596). The released Union Segmentation Head transfers the first manual source mask to the open target through VGGT tracking, pair features, and iterative refinement. | Compare dense source-mask-conditioned geometric transfer with V2-SAM expert selection on exactly the same source-region-target tuple. Report the released checkpoint's Ego-Exo training domain as a limitation. |
| 5 | COLMAP-Verified Video Identity | **Implemented as our sparse-geometry adaptation of [MV3DIS](https://arxiv.org/html/2604.08916) and [MaskClustering](https://arxiv.org/html/2401.07745).** It consumes SAM2 Video explicitly, preserves mask pixels, and conservatively relabels connected components from exact COLMAP observations. | Test whether sparse tracks can correct persistent-ID drift while abstaining on unsupported thin geometry. Evaluate correction precision and coverage separately; this is not a boundary-refinement claim. |
| 6 | SAI3D Hard-Anchor Adapter | [Official SAI3D project](https://yd-yin.github.io/SAI3D/); source is already integrated locally | Replace automatic semantic evidence with user region votes, run progressive 3D superpoint consensus, and recover image boundaries from 2D candidates. |
| 7 | MaskClustering Anchor Adapter | [Official code](https://github.com/PKU-EPIC/MaskClustering) assumes RGB-D and reconstructed points | Test global mask-node consensus with hard user anchors. Adapt only after the sparse-track baseline shows whether the available geometry supports reliable proposal links. |
| 8 | MV3DIS-Style 3D-Guided Matching | [Paper](https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MV3DIS_Multi-View_Mask_Matching_via_3D_Guides_for_Zero-Shot_3D_CVPR_2026_paper.html); [repository](https://github.com/zybjn/MV3DIS) is currently incomplete | Use coarse 3D projections as common references for selecting consistent 2D candidates, with visibility/depth reliability. Wait for a usable release before claiming an exact reproduction. |
| 9 | 3AM | [Project page](https://jayisaking.github.io/3AM-Page/); code is announced but unavailable | Replace sequence-only matching with MUSt3R geometry-aware features inside SAM2 when an official implementation is released. |
| 10 | MV-SAM | [Paper and project](https://jaesung-choe.github.io/mv_sam); code is currently marked for release | Test joint multi-view prompt segmentation with pointmap coordinates and geometry confidence when source and weights are available. |

`Surface-SOS`, `PanSt3R`, `SA3D`, `ISRF`, and `NVOS` remain deferred
representation-backed comparisons. They require optimizing a dense scene
representation and therefore answer a different systems question from the
frame-first, pre-reconstruction workflow.

For every completed test, use one frozen comparison packet:

- identical complete-frame anchors and persistent region registry;
- identical staged images and evaluation frame set;
- region IoU and boundary F-score where ground truth exists;
- anchor preservation, coverage, uncertainty, and cross-view identity metrics;
- runtime, peak memory, and user corrections needed on the next keyframe;
- fixed UI rows for frames and columns for methods.

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

Hierarchical segmentation is a later extension rather than part of the first
coarse pass. A persistent region should eventually support an optional
`parent_id`, so a coarse region such as `door`, `stairs`, or `arch` can be
split in a second pass into panels, ornaments, bricks, or trim. This keeps the
main workflow focused while leaving a clean path to coarse-to-fine region
definition.

### Phase 7: Region-Aware Reconstruction

The stable artifact boundaries, backend interface, baseline-isolation rules,
and controlled experiment matrix are specified in
[`SEGMENTATION_RECONSTRUCTION_ARCHITECTURE.md`](SEGMENTATION_RECONSTRUCTION_ARCHITECTURE.md).

Step 5 is a reasonable replacement for raw automatic SAM2 masks wherever a
method exposes per-view masks as supervision. The precise claim is:

```text
User-conditioned Step 5 candidates provide more intent-aligned mask evidence
than independent SAM2 Automatic proposals, while complete keyframes remain hard
annotations and unsupported propagated pixels remain soft or unknown.
```

This is an interface-level substitution, not a claim that every propagated mask
is final ground truth or that an entire prior pipeline can be skipped.
Compatibility depends on what the downstream method expects:

- **SAI3D:** direct fit. The current Step 6 adapter already replaces automatic
  proposal evidence with stable persistent IDs and gives manual anchors higher
  weight.
- **Split&Splat:** direct at the per-view-mask boundary. Experiment B replaces
  its SAM2 Automatic initialization while retaining released Split. Experiment
  C additionally replaces released Split mask recovery with anchored 3D
  identity association so user-authored shapes and IDs survive. Instance
  reconstruction and composition remain useful downstream.
- **Gaussian Grouping:** compatible through an adapter that renders stable
  region IDs and confidence-weighted identity targets. Step 5 replaces the
  SAM/DEVA pseudo-label source, not the Gaussian identity optimization.
- **Per-object 3DGS, 2DGS, and OMeGa:** direct once hard, soft, negative, and
  unknown pixels are kept distinct and Step 6 points are exported by region.
- **SAGA:** not a literal drop-in. Its scale-gated affinity field is trained from
  overlapping multi-scale SAM proposals, while our persistent regions are
  exclusive task labels.
- **SAM2Object:** Step 5 largely replaces its propagation role. Feeding the
  result back into another long SAM2 propagation is not automatically useful.

#### Relevant Reconstruction Methods

| Method | Main inputs | Reconstruction organization | Main outputs | Lesson for Step 7 |
| --- | --- | --- | --- | --- |
| [Split&Splat](https://arxiv.org/html/2602.03809v1) | Posed RGB, SfM/depth, SAM2-derived view-consistent instance masks, labeled point subsets | Trains one 3DGS per instance, refines masks from rendered geometry, merges colliding Gaussian sets, then jointly refines the composition | Separate instance Gaussian sets plus one composed scene with instance descriptors | Closest executable baseline for independent warm-up followed by scene-wise composition |
| [ObjectSDF++](https://qianyiwu.github.io/objectsdf%2B%2B) | Posed RGB and instance masks | Jointly renders multiple object SDFs with occlusion-aware object opacity and an object-distinction regularizer | Whole-scene and separate object meshes | Separate object ownership still needs joint visibility and collision reasoning |
| [RICO](https://arxiv.org/abs/2303.08605) | Posed RGB, semantic masks, monocular depth, and normals | Joint compositional SDF with object-background depth constraints and smoothness in unobserved space | Per-object and combined meshes | Depth/normal priors can regularize architectural regions, but indoor-background assumptions should not be copied blindly |
| [Gaussian Object Carver](https://arxiv.org/html/2412.02075) | RGB, instance labels, monocular depth, and normals | One shared 3DGS with per-Gaussian semantics, followed by segmented point extraction and optional object completion | Shared scene, object point sets, and optional watertight object meshes | A shared scene can preserve context while still exporting object geometry; the announced code is not currently public |
| [Gaussian Grouping](https://arxiv.org/abs/2312.00732) | RGB and SAM/DEVA identity supervision | One 3DGS with a differentiably rendered identity feature and local 3D consistency | One editable, grouped Gaussian scene | Strong shared-scene baseline; region identity need not require separate training |
| [vMAP](https://arxiv.org/abs/2302.01838) | RGB-D video, poses, and object masks | One compact implicit model per object, optimized in a vectorized map | Separate watertight object fields | Supports modular object models, although its online RGB-D setting differs from our posed-image dataset |
| [Direct Object-Level Reconstruction via Probabilistic Gaussian Splatting](https://arxiv.org/abs/2603.14316) | Posed images and continuous foreground probabilities | Filters the SfM initializer and trains a compact single-object 2DGS with per-Gaussian foreground probability | One compact object-level 2DGS and probability masks | Closest mathematical reference for using Step 5 confidence as soft supervision rather than thresholding every candidate |

The local Split&Splat source confirms the important implementation sequence:

1. create one image/mask/camera folder per instance;
2. initialize each instance from its labeled point subset;
3. train each Gaussian model independently with RGB and rendered-opacity mask
   losses;
4. refine masks by prompting SAM2 from projected instance Gaussians and checking
   agreement with rendered geometry;
5. progressively merge spatially colliding Gaussian sets;
6. reset opacity, stop densification during composition, and jointly refine with
   progressively stronger mask consistency.

This means Step 5 can improve Split&Splat's supervision, but the composition
stage is still needed to repair independent-model overlap, missing context, and
boundary disagreement. Keep Split&Splat as an external baseline behind a data
adapter; its repository combines code under different license terms, so its
implementation should not be copied into the Apache-licensed OMeGa fork without
an explicit license review.

The current anchored reconstruction consumes `training_view_weights.json`,
weights the rendered-opacity mask loss on completed manual frames, and starts
from each region's exact full-attribute Gaussian subset. Its warm-up disables
densify-and-prune so no initializer Gaussian is discarded. Released SAM2 mask
refinement remains an optional isolated ablation rather than part of the
default anchored result.

#### Region Identity Versus Reconstruction Ownership

A persistent `region_id` records designer intent. A `component_id` records which
parameters and geometry are optimized together. They should not be forced to be
identical.

Examples:

- a detached door or railing can map one region to one component;
- facade plane, attached trim, and an arch seam can retain separate region IDs
  but share one component so they do not reconstruct as overlapping shells;
- a hierarchical door may have child region IDs for panels and ornaments while
  one component owns their common geometry;
- context or ignore regions supervise visibility but do not need a standalone
  model.

This separation lets the UI preserve semantic/task labels while the
reconstruction chooses a stable ownership partition.

#### Step 7 Input Contract

```text
interactive/reconstruction/input/
  supervision_manifest.json
  components.json
  regions.json
  frames.jsonl
  hard_region_maps/             # complete user-edited keyframes
  candidate_layers/<method>/    # Step 5 label/confidence/unknown maps
  segmented_points/<method>/    # Step 6 points with stable region IDs
  view_evidence/                # optional depth and StableNormal
```

`supervision_manifest.json` freezes the exact Step 5 backend, run fingerprint,
anchor set, and Step 6 source. `components.json` maps persistent regions to
trainable components and stores per-component settings such as primitive
budget, planar regularization, normal/depth weights, subdivision, remeshing, and
training iterations.

For frame `i`, region `r`, and pixel `p`, construct a target `M_ir(p)` and weight
`W_ir(p)`:

```text
W_ir(p) = 1                         for a complete manual anchor
        = lambda_soft * c_ir(p)     for a propagated candidate
        = 0                         for unknown or conflicting evidence
```

where `0 < lambda_soft < 1` and `c_ir` is a confidence only when the backend
provides a meaningful one. A completed keyframe can provide trusted negatives
for its absent visible regions. An incomplete or unsupported frame must not be
silently interpreted as background.

All components are rendered together to obtain scene color `C_i`, depth, and
per-region/component opacity `A_ir`. A minimal joint objective is:

```text
L = L_scene_rgb
  + lambda_mask * sum_irp W_ir(p) BCE(A_ir(p), M_ir(p))
  + sum_k L_geometry(k)
  + lambda_separate * L_component_overlap
  + lambda_boundary * L_shared_boundary
```

`L_geometry(k)` is backend-specific. For OMeGa it can include its photometric,
normal, mesh, and splat regularizers with region-specific weights. For 2DGS or
3DGS it includes the original rendering and geometry regularization. Independent
warm-up should compute RGB loss only on trusted foreground support and use
opacity loss on trusted positives/negatives; multiplying the RGB image by an
incomplete mask and treating every black pixel as background would bake Step 5
errors into the model.

#### Reconstruction Modes

Implement three comparable modes behind one adapter contract:

1. **Shared scene baseline:** one 3DGS/2DGS/OMeGa scene with a rendered region-ID
   or probability head, following Gaussian Grouping-style identity supervision.
2. **Independent component baseline:** initialize and optimize one model per
   component, then concatenate or compose them. This is the closest
   Split&Splat baseline and exposes leakage caused by independent training.
3. **Compositional reconstruction:** independently warm-start components, then
   render all components together with common cameras and jointly refine
   visibility, boundaries, and overlap. This is the preferred direction.

For OMeGa, each component should own a mesh, its face-bound splats, optimizer
state, and remeshing policy. The renderer depth-sorts splats from every component
in one scene. Remeshing must not cross persistent-region boundaries unless those
regions deliberately share a component; shared component boundaries need
coincidence/stitching constraints rather than two independently drifting edges.

Step 6 is an initializer, not immutable truth. Region-labeled points seed the
appropriate component, high-confidence unlabeled geometry can remain shared
context, and unsupported regions can fall back to the global initializer rather
than disappearing.

#### Step 7 Outputs

```text
interactive/reconstruction/runs/<backend>/<run_id>/
  config.json
  supervision_manifest.json
  components/<component_id>/
    model/
    mesh/
    splats/
    region_ids.json
  composed/
    model/
    mesh/
  renders/
    rgb/
    region_id/
    opacity/
    depth/
    normal/
  metrics.json
```

All components remain in the original COLMAP/OMeGa world coordinate system, so
separate exports can be edited independently and loaded together without a
second alignment.

#### First Step 7 Experiment

Do not patch OMeGa's training loop first. Freeze one supervision packet and test
three representative components: a large facade plane, a detailed door/arch,
and a thin railing.

Compare:

1. one full-scene 2DGS/3DGS baseline;
2. independently masked component models;
3. independent warm-up followed by joint compositional refinement.

Use the same RGB frames, cameras, hard anchors, one selected Step 5 candidate
layer, and one Step 6 segmented initializer. Measure held-out RGB quality,
region silhouette IoU and boundary F-score, cross-region leakage, component
coverage, overlap/cracks at shared boundaries, geometry error where a reference
exists, and primitive count. This experiment determines whether region-wise
ownership helps before introducing the additional mesh/splat coupling and
remeshing complexity of OMeGa.

### Cross-Phase Review Loop

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

### Phase 8: Post-Reconstruction Dense-Mask Export

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

The current editor implements the six operational stages from preparation
through 3D segmentation. Phase 7 now has a frozen reconstruction contract, but
no region-aware training backend is implemented yet. It exports region-colored
COLMAP points from
exact track consensus, region-colored feed-forward initializer points, and
region-colored samples from the final strongest OMeGa mesh. The projected
clouds use calibrated z-buffer visibility and anchor voting. All three sparse
point layers feed parallel Superpixel Geodesic, 2D/3D CRF, and SAM2
point-prompt dense recovery tests. Future packaged downstream export remains
separate from these candidate layers.

For the current DSLR StableNormal OMeGa run, `step_23_run_phone_omega_stable_mesh.json`
records MapAnything as the seed source. OMeGa voxelized 10,977,572 raw points to
367,740 points at 5 cm, copied that exact cloud to
`omega_stable_mesh/dataset/vfm_sparse/0/points3D.ply`, and generated
`omega_stable_mesh/init_mesh.ply` from 180,000 sampled seed points with Poisson
reconstruction and PyMeshLab repair. The editor discovers those two artifacts
from `--model-dir`; `--feedforward-point-cloud` and
`--feedforward-init-mesh` allow an aligned MASt3R, MapAnything, or other
feed-forward result to replace them without changing propagation or dense
recovery code.

The final-reconstruction branch tests the favorable case in which a good OMeGa
geometry already exists before segmentation. The editor discovers
`model_baseline_strongest_30000/plys/mesh_29999_rank0.ply`, extracts all finite
optimized vertices exactly once, and stages them as
`interactive/data/point_clouds/omega_final_vertices.{ply,npz,json}`. This is not
uniform surface resampling: retaining the mesh vertices preserves OMeGa's
adaptive density and exact optimized positions. The source mesh has no vertex
colors, so the raw cloud is white; `OMeGa Final Points` produces the persistent-
region-colored version. `Final Superpixels`, `Final 2D/3D CRF`, and `Final SAM2
Prompts` then reuse the same stage-two decoders as the initializer experiment.

Current grove-entrance result with eight complete anchors and 26 persistent
regions: the strongest mesh contributed 1,768,807 finite vertices, of which
1,613,227 received an unambiguous persistent ID. Sparse projected coverage is 70.4% on average across 214 views, compared with
31.8% for OMeGa Initializer Points and
3.7% for COLMAP Tracks. `Final Superpixels` raises mean coverage to 97.4%;
`Final 2D/3D CRF` reaches 96.4% and abstains slightly more. These are coverage
statistics, not accuracy claims: the saved side-by-side layers must still be
inspected for boundary leakage and wrong 3D identity transfer.

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

The editor discovers the checked-out memory propagation baselines at
`$DT_ROOT/third_party/XMem2` and `$DT_ROOT/third_party/Cutie`. Their default
checkpoints are `XMem2/saves/XMem.pth` and
`Cutie/weights/cutie-base-mega.pth`. Step 5 provides independent `Video`,
`Bounded`, `XMem++`, and `Cutie` runs and previews.

The default `--label-source raw` loads the sampled SAI3D input points as an
unsegmented point cloud, so all points render as label `0`. This matches the new
pipeline: start from raw capture/point evidence, then let the user define
segmentation intent through proposals and regions.

Optional label sources:

```text
--label-source raw     sampled input points, all label 0
--label-source sai3d   SAI3D final point_labels.npy
--label-source auto    sai3d, then raw
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

### 6.3 Proposal Layers

The editor keeps frame-local SAM2 proposals, anchor-conditioned persistent
region candidates, and user-confirmed persistent region maps as distinct data
types.

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

Every anchor-to-mask result uses one method-keyed layout:

```text
interactive/proposals/propagation/
  registry.json
  <method-id>/
    config.json
    input.json
    progress.json
    summary.json
    frames.jsonl
    label_maps/
      000000.npy
      000000.png
    confidence/
      000000.npz
    uncertainty/
      000000.png
    overlays/
      000000.png
    metadata/
      000000.json
```

`registry.json` stores method ID, display name, availability, and label space.
Each method directory stores the exact input packet and fingerprint, method
configuration, live progress, and validated output. The web app reads this
registry through the layer-status API, so adding a backend does not require
method-specific layer, runner, or preview code.

Raw SAM2 automatic proposals remain frame-local:

```text
sam2_label_map[y, x] = 0 for no proposal
sam2_label_map[y, x] = positive frame-local proposal ID
```

Every anchor-to-mask backend uses the persistent-region label space:

```text
region_candidate_map[y, x] = 0 for no candidate
region_candidate_map[y, x] = stable persistent region ID
```

The picker can target any visible ready layer, but it preserves the layer's
declared label space. Local proposal IDs never become semantic IDs implicitly.
Persistent region candidates retain their source region and source anchor IDs
through selection and assignment.

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
- the StableNormal button in the Step 1 preparation panel generates or
  regenerates only normal outputs;
- the Depth Anything V2 button in the Step 1 preparation panel generates or
  regenerates only depth outputs;
- if a previous evidence job was interrupted, `Generate` keeps compatible
  existing frames and computes the missing frames;
- older preview-quality evidence is ignored by status until it is regenerated
  with the current preparation quality preset;
- the viewer dropdown changes only the displayed background image, not camera
  projection, source-pixel coordinates, proposal overlays, or point visibility.

The current webpage supports a unified frame-local pixel selection model.
Different tools produce the same kind of selected pixel mask:

- `Pick`: click an existing proposal in the active frame;
- `Lasso`: draw a raw pixel region;
- `Normal Grow`: click a surface and flood-fill connected pixels with similar
  StableNormal direction;
- `RGB-D Cue`: add foreground/background clicks and solve the
  superpixel-based RGB-D cue-selection MRF from Feng et al.; the returned mask
  replaces the current selected pixels;
- `SAM2`: place positive and negative prompts;
- `Deselect`: clear the current selected region;
- `Cancel`: discard the temporary selection;
- `Cmd/Ctrl+Z`: undo transient selection edits.

Selection semantics:

- plain action replaces the current selection;
- `Shift` adds;
- `Option`/`Alt` subtracts;
- the `+` and `-` buttons can pin add/subtract mode;
- the canvas shows one final composed selected boundary, not separate tool
  histories.

SAM2 prompting does not directly create a saved proposal. It produces a
temporary selection. The user can refine that selection with pick/lasso, normal
grow, or RGB-D cue, then use Step 4 region operations to commit the pixels.

This preserves the clean invariant:

```text
Step 3 selects pixels; Step 4 commits selected pixels to confirmed region maps
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

The selector uses the editor's generated SAM2 proposal label maps when they
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

### 6.6 Propagated Layer Rebuild

The current propagation section writes a saved proposal layer, not accepted
labels.

Backend logic:

1. Validate that at least one frame is marked complete.
2. Collect every complete frame that has a positive persistent `region_map`.
3. For each persistent region ID, add every accepted keyframe mask for that
   region to one SAM2 video inference state using the same object ID.
4. Run SAM2 video propagation from the first ordered frame through the full
   sequence. Completed frames before or after any target frame stay in SAM2's
   conditioning memory as mask prompts.
5. Convert per-object SAM2 logits to one exclusive label map by assigning each
   pixel to the strongest positive object logit, or background if no region logit
   is positive.
6. Save the result as the `Propagated` proposal layer with source-region and
   source-keyframe metadata.

The popup compares:

- top: propagated region proposal overlay over RGB;
- bottom: current SAM2 proposal overlay over RGB when available, otherwise RGB
  only.

This is the intended role for propagation at this stage: user-confirmed regions
create better region-aware proposals for later frames, and later user edits add
more hard anchors before the propagated layer is rebuilt again.

### 6.7 Existing Data Source

The editor currently uses a SAI3D-style baseline directory as its staged capture
container:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/sai3d_area_samples_1024_dense/
  dataset/frame_manifest.jsonl
  dataset/posed_images/.../*.jpg
  dataset/posed_images/.../*.txt
  dataset/scans/.../points.pts
  mesh_labels/point_labels.npy             # optional SAI3D diagnostic labels
```

For the new pipeline, this SAI3D baseline is mainly a convenient source of
staged frames, cameras, and sampled points. `point_labels.npy` is not required
for the default `--label-source raw` workflow, and SAI3D labels are not the
source of truth.

## 7. What Should Be Modular

The implementation should avoid one monolithic segmentation script.

Current module boundaries:

```text
omega_local/segmentation/interactive/
  app.py, state.py, paths.py   # web API, orchestration, and canonical paths
  proposals.py, keyframes.py   # local proposals and edit-frame suggestions
  regions.py, mask_edits.py    # persistent IDs and selected-pixel operations
  view_evidence*.py            # StableNormal / Depth Anything preparation
  dinov3_*.py                  # reusable DINOv3 evidence cache
  geometry_selection.py        # Normal Grow
  rgbd_cue_selection.py        # Feng et al. RGB-D cue-selection MRF
  sam2_session.py              # single-image SAM2 selection
  propagation_backends.py      # shared candidate-layer registry and contract
  sam2_*propagation.py         # video and bounded SAM2 backends
  memory_vos_*.py              # XMem++ and Cutie adapters
  colmap_*propagation.py       # sparse tracks and dense-recovery diagnostics
  colmap_identity_refinement.py # shape-preserving Video ID verification
  feedforward_point_*.py        # shared calibrated arbitrary-cloud projection
  omega_mesh_point_cloud.py     # exact final-mesh vertex extraction
  region_pair_test.py          # focused source-target test contract
  v2sam_*.py, vggts_*.py       # focused learned correspondence adapters
  point_cloud_sources.py       # reconstruction/COLMAP display sources
  segmentation3d_contract.py   # method/input/run registry contract
  sai3d_experiment.py          # SAI3D adapter over editor mask sources
  segmentation3d_manager.py    # background jobs and labeled-point results
  static/
    css/                       # app shell, viewer, panels, filmstrip
    js/                        # viewer and phase-specific controls

future modules:
  fusion.py                    # confidence scoring and final label maps
  export_dataset.py            # packaged supervision/dense-mask export
```

Current output layout:

```text
<model_dir>/segmentation/baselines/<baseline-name>/interactive/
  data/point_clouds/
  view_evidence/
  proposals/
    sam2_auto_<width>/
    propagation/<method-id>/
  keyframes/
  regions/
  3d_segmentation/
    inputs/<input-id>/
    runs/<method-id>_<input-id>/
```

Future `fused/` and `exports/` directories should be added only when their data
contracts are implemented.

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
as an automatic segmenter. Mask quality alone is insufficient because a fully
manual workflow can eventually produce excellent masks. The primary result
should be a **quality-versus-user-effort curve**: quality reached after a fixed
time or interaction budget, and effort required to reach a fixed quality.

### 9.1 Dataset Strategy

No single public dataset covers both standard 3D segmentation and the
task-defined architectural regions that motivate this work. Use a tiered
benchmark:

| Dataset | Role | Strength | Limitation |
| --- | --- | --- | --- |
| [ScanNet](https://www.scan-net.org/ScanNet/) | Standard compatibility test | RGB-D frames, poses, reconstructed meshes, 3D instances, projected 2D labels, and direct compatibility with interactive 3D baselines | Lower-resolution video capture and mostly object-centric labels |
| [ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/) | Primary public real-scene benchmark | Registered high-resolution DSLR images, iPhone RGB-D, laser geometry, COLMAP poses, and semantic/instance annotations | Indoor scenes and annotations that still follow semantic objects more than reconstruction intent |
| [Hypersim](https://github.com/apple/ml-hypersim) | Controlled diagnostic benchmark | Exact geometry, poses, depth, normals, materials, and dense pixel labels | Synthetic appearance and camera trajectories |
| Custom architectural captures | Core intent benchmark | Facade planes, arches, trim, railings, material regions, and arbitrary reconstruction seams | Requires expert reference annotations and multiple scenes |

ScanNet++ is the most important public dataset for our setting. Its official
toolbox can rasterize laser-mesh semantic and instance labels into registered
DSLR or iPhone views, allowing the same result to be measured in 2D and 3D.

A small pilot can begin with several ScanNet++ scenes and a limited set of
well-covered DSLR views. The final evaluation must include multiple custom
architectural scenes; the grove entrance alone is a demonstration, not enough
evidence for generalization.

### 9.2 Two Segmentation Tracks

Evaluate two distinct tasks:

1. **Standard instance segmentation:** doors, chairs, tables, fixtures, and
   other regions already represented by public ground truth. This establishes
   compatibility with common segmentation benchmarks.
2. **Reconstruction-intent segmentation:** facade or wall subdivisions, object
   parts, thin structures, material regions, detail groups, and explicitly
   specified seams. This tests the actual contribution.

For the intent track, every participant must receive the same visual task
specification and expert-reviewed reference masks. A separate open-ended study
can let designers define their own decomposition, but those sessions should be
evaluated for control, usefulness, and consistency rather than against a unique
ground truth that does not exist.

### 9.3 Fair Frame-Space And 3D Comparison

Compare representative interaction families on the same scenes and target
regions:

- our frame-first editor operating on original posed images;
- a direct point-cloud method such as Easy3D or AGILE3D;
- a representation-backed method such as SAGA on a 3DGS trained from the same
  images;
- frame-local SAM2-assisted annotation and propagation-only baselines.

The comparison should evaluate 3D methods under two geometry conditions:

- **oracle geometry:** the dataset laser scan or ground-truth mesh, giving the
  direct 3D method its best possible input;
- **practical geometry:** a point cloud, mesh, or 3DGS reconstructed from the
  same posed images used by our method.

This separates intrinsic interaction quality from sensitivity to reconstruction
errors. Project each 3D result into the held-out image views for 2D evaluation,
and lift/fuse our per-view labels onto the reference scan for 3D evaluation.

Use user time as the main effort measure. A 3D click, lasso stroke, SAM2 prompt,
and region assignment do not have equivalent costs, so raw click count alone is
not a fair cross-interface metric.

### 9.4 Quantitative Measures

**User effort:**

- total user time and compute waiting time, reported separately;
- number of edited keyframes and revisited frames;
- positive/negative clicks, lasso strokes, region operations, and undo actions;
- accepted propagated area versus manually corrected area;
- additional frames requested by the review loop.

**Per-view mask quality:**

- region IoU/Jaccard;
- boundary F-score;
- visible-region coverage;
- false merge and false split counts;
- uncertain-pixel rate;
- quality after each completed keyframe and after each interaction round.

**Multi-view and 3D consistency:**

- label agreement of shared COLMAP tracks;
- identity switches across views;
- projected-region conflicts and cycle consistency;
- per-region 3D IoU, completeness, and contamination on a reference scan;
- 3D boundary distance;
- AP25/AP50 only where standard instance matching is meaningful.

Sparse track agreement should not be reported alone because it under-samples
boundaries and thin structures. Report interior and boundary behavior
separately where possible.

**Reconstruction utility:**

- leakage between independently reconstructed regions;
- object or region removal/editing quality;
- held-out rendering quality for object-wise or layer-wise 3DGS;
- geometry, normal, or boundary quality when reference geometry is available;
- whether region-conditioned OMeGa or remeshing produces cleaner, more editable
  structure.

The reconstruction test is required to support the term
"reconstruction-ready"; good-looking overlays alone are not sufficient.

### 9.5 User Study

Use a within-subject, counterbalanced study so each participant experiences the
frame-first workflow and the selected baseline workflows without ordering bias.
Recruit reconstruction practitioners, architects, designers, or experienced 3D
content creators for the main study; a novice group can be reported separately.
Determine the final participant count from a pilot and power analysis.

The controlled study should measure time-to-quality on fixed region tasks. A
smaller open-ended study should examine whether participants can express useful
decompositions that automatic object segmentation does not provide.

Record every interaction before the study begins:

- timestamp and active frame;
- selected tool and operation;
- proposal/region source and affected pixel count;
- propagation, undo, completion, and frame-navigation events;
- intermediate masks after every accepted operation.

Subjective measures should include workload, perceived control, trust in
propagated suggestions, and whether the resulting decomposition matches the
participant's reconstruction intent.

### 9.6 Baselines And Ablations

Automatic and interactive baselines serve different purposes and should be
reported separately.

Main baselines:

- independent SAM2 proposals or frame-local SAM2-assisted editing;
- SAM2 Video, SAM2 Bounded, XMem++, Cutie, and sparse COLMAP Tracks from fixed
  user anchors;
- COLMAP Track Consensus plus 2D proposal decoding;
- focused V2-SAM and VGGT-S source-target region tests;
- SAI3D automatic labels and the planned hard-anchor adapter;
- MaskClustering anchor consensus when the geometry adapter is available;
- Easy3D or AGILE3D direct point-cloud interaction;
- SAGA or another representation-backed 3DGS interaction method.

System ablations:

- user keyframes without propagated suggestions;
- user keyframes with long-video propagation;
- user keyframes with bounded propagation;
- exact tracks alone versus learned correspondence alone;
- track-guided V2-SAM candidate fusion;
- candidate fusion with and without dense 2D boundary recovery;
- with and without segmentation-aware keyframe recommendation;
- with and without depth/normal selection tools;
- propagation-only final masks versus hard anchors plus multi-view fusion;
- oracle geometry versus practical reconstructed geometry.

The central expected result is not that every propagated frame improves. It is
that reviewable propagation lowers later correction cost, while hard anchors
and final fusion preserve user intent and recover consistency.

### 9.7 Publication Positioning

The paper should not be framed as a collection of existing selection tools. Its
core claim is the intent-before-geometry workflow and the measured benefit of
iteratively turning sparse user anchors into multi-view reconstruction labels.

- An HCI/UIST-style paper should emphasize formative designer needs, interaction
  design, quality-versus-effort results, and a controlled study with domain
  users.
- A graphics/3D-vision paper should additionally contribute a principled
  anchor-constrained fusion method, a reproducible benchmark, and quantitative
  2D/3D comparisons.
- A reconstruction-centered paper must show that the exported regions improve
  independent editing, layer separation, geometry, or remeshing downstream.

The intended direction is a hybrid systems and graphics contribution: the user
study establishes that the workflow expresses intent efficiently, while fusion
and reconstruction experiments establish that the result is technically useful.

## 10. Recommended Build Order

### Milestone 1: Stabilize Proposal Editing

Current status: implemented.

Available:

- raw point cloud plus RGB frame viewer;
- SAM2 proposal generation/loading;
- pick/lasso/SAM2 temporary selection;
- normal grow and RGB-D cue selection;
- undo and transient selected-pixel state;
- clear distinction between selected pixels and candidate proposals.

Success criterion:

```text
A user can create a high-quality selected pixel mask on one frame without
modifying the source SAM2 or propagated proposal layers.
```

### Milestone 2: Add Persistent Region IDs

Current status: implemented for flat persistent regions.

Available UI:

- region list;
- create region from current selection;
- assign current selection to existing region;
- rename/delete region;
- assign, replace, add, subtract, and clear frame-local pixels;
- complete-frame anchor marking and manual-frame references.

Success criterion:

```text
A user can turn selected pixels into named global regions on selected
keyframes.
```

Context/ignore/uncertain roles, hierarchy, and direct region merge/split remain
future extensions.

### Milestone 3: Parallel Anchor-To-Mask Backends

Use complete persistent-region keyframes as fixed inputs to independent,
comparable candidate-mask methods.

Current status: implemented. The shared registry, anchor loader, input
fingerprint, output validator, result writer, progress interface, dynamic layer
toggles, and method-selectable run/preview UI cover Video, Bounded, XMem++,
Cutie, and COLMAP Tracks. The Geometry Pass controls expose COLMAP Verify Video,
Superpixel Geodesic, 2D/3D CRF, and SAM2 Point Prompts as independent
source-dependent layers. V2-SAM and VGGT-S share a focused region-pair action,
and COLMAP Tracks exports the segmented sparse 3D cloud.

Remaining:

- add synchronized multi-method comparison columns;
- formalize quality, coverage, uncertainty, and correction-effort evaluation.

Success criterion:

```text
One frozen set of edited keyframes produces directly comparable candidate masks
from every enabled backend without overwriting final labels.
```

### Milestone 4: Fuse Into Per-Frame Label Maps

Current status: future/optional dense-output stage. The current hard anchors,
soft candidates, and sparse 3D labels are already a usable supervision package.

Implement a first confidence-based resolver only where dense masks are required.

Needed:

- complete user-confirmed keyframes as hard labels;
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

### Milestone 6: Region-Aware Reconstruction And Downstream Test

Freeze the current evidence tiers, then compare shared-scene, independent-
component, and jointly composed reconstruction without requiring dense masks in
every frame.

Needed:

- supervision manifest with hard/soft/sparse provenance;
- stable region-to-component ownership and per-component settings;
- confidence, uncertainty, and explicit unknown handling;
- shared-scene, independent-component, and compositional 3DGS/2DGS adapters;
- an OMeGa adapter only after the simpler reconstruction comparison is
  understood.

Success criterion:

```text
The same supervision packet can train and compare all reconstruction modes
without treating propagated or unsupported pixels as ground truth.
```

## 11. Near-Term Direction

The current implementation should stay centered on manual keyframe anchors and
proposal priors. Propagation is useful because it gives the user better starting
points on later keyframes, especially after earlier frames have been corrected,
but it should not be judged as the final segmentation result. The long-video
and bounded experiments can remain as proposal baselines; further propagation
tuning is secondary to completing and evaluating the end-to-end research claim.

Reason:

- selected-pixel tools let the user correct SAM2's local mistakes;
- persistent regions encode designer intent across the dataset;
- complete keyframes are the only trusted propagation anchors;
- propagated masks are region-aware suggestions for later editing;
- segmented COLMAP tracks already transfer persistent identity consistently in
  sparse 3D;
- perfect dense masks are optional later products of 3D-backed
  fusion/refinement, not a condition for starting reconstruction experiments.

Immediate research tasks:

```text
1. Freeze a supervision manifest that records hard keyframe anchors, each soft
   candidate layer, segmented COLMAP points, uncertainty, and provenance.
2. Use this package in a first 3D segmentation or region-conditioned OMeGa/3DGS
   experiment with evidence-specific loss weights and explicit unknown pixels.
3. Measure how user anchors, propagation priors, and sparse tracks affect 2D
   quality, 3D identity consistency, reconstruction leakage, and user effort.
4. Add synchronized multi-method comparison columns and log the full
   quality-versus-effort trajectory.
5. Use a reconstructed 3D label field to refine/reproject dense masks only after
   the downstream experiment shows where dense coverage is actually required.
6. Add a ScanNet++ adapter and compare frame-first editing against direct 3D
   interaction under oracle and practical geometry.
```

The key publication test is whether the system reaches accurate 2D boundaries
and consistent 3D region identity with less user effort, especially for
task-defined regions that object-oriented segmentation does not naturally
represent.
