# Segmentation Test Report

Status: summary of the segmentation baselines and prototypes tested on
`grove_entrance_dslr_0521_pinhole`, using the OMeGa stronger mesh output under:

```text
data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/
  omega_stable_mesh/model_baseline_stronger_30000
```

## Goal

The long-term goal is designer-friendly 3D reconstruction: the scene should be
split into meaningful architectural and object parts so OMeGa can optimize,
remesh, and export each region with appropriate constraints. In practice we
need both:

- consistent 3D labels for mesh/splats/points;
- clean per-view masks that can be used later as training supervision or user
  editing handles.

The key tension is that standard segmentation methods are usually object-centric
while our desired units are often building elements: walls, arches, steps,
railings, facade components, seams, and sculptural details. These parts may not
have strong RGB boundaries, and the current reconstructed mesh can be noisy or
incomplete.

## High-Level Outcome

The most important result is:

```text
SAI3D-style 3D segmentation is currently the most useful automatic anchor.
2D/video mask methods are useful evidence, but they are not reliable final
building-element segmenters for this scene.
```

Mesh-only methods were not robust enough because they inherit the current OMeGa
mesh quality. Video and SAM/DEVA/Split&Splat-style methods can produce masks,
but tend to over-merge facade elements or miss smaller architectural parts. SAI3D
is stronger because it resolves identity in 3D: once a point/superpoint receives
a label, every view observes the same underlying label through projection.

This led to the interactive prototype: edit SAI3D's 3D labels directly from
frames, then reproject the edited 3D state into all views.

## Method Categories

### 1. Mesh-Rendered Or Mesh-Conditioned Methods

These methods start from the mesh or rely strongly on mesh-rendered geometric
views. They are attractive because they can output 3D face labels directly, but
they assume the mesh is a reliable representation of the scene.

Tested methods:

- GeoSAM2
- SAMPro3D
- SAM2Object graph consolidation over mesh vertices
- SAI3D on raw mesh vertices

Observed behavior:

- They are sensitive to missing or messy geometry.
- Boundaries inherit OMeGa mesh fuzziness, folded faces, and incomplete
  structures.
- They can be useful for coarse spatial support, but not enough by themselves
  for designer-facing building-element segmentation.

### 2. 3D Segmentation Methods

These methods produce or optimize labels on a 3D representation: mesh vertices,
sampled mesh points, point clouds, or Gaussians. This is the most promising
category for cross-view consistency.

Tested methods:

- SAMPro3D
- SAI3D on mesh vertices
- SAI3D on area-balanced mesh samples
- SAI3D on MapAnything points
- SAI3D on COLMAP sparse points
- Gaussian Grouping

Observed behavior:

- SAI3D produced the most consistent multi-view identity because the final
  labels live in 3D.
- Area-balanced mesh sampling improved SAI3D over raw mesh vertices by reducing
  OMeGa vertex-density bias.
- MapAnything points gave broad coverage and useful semantic grouping, but the
  point cloud was noisy and layered.
- COLMAP sparse points gave strong detail and edge evidence, but should remain
  conservative and oversegmented because the graph is sparse.
- Gaussian Grouping learned interesting identity features, visible in PCA, but
  its final predicted masks mostly inherited DEVA/SAM pseudo-label limitations.

### 3. Clean 2D / Video / Mask Propagation Methods

These are appealing because the final thing we want for training is often a
per-frame clean mask. They usually work well for object-level segmentation, but
our building-element targets are harder.

Tested methods:

- Common SAM2 automatic per-view proposal initializer
- SAM2Object
- SAI3D-to-view SAM2 prompting
- SAI3D proposal-gated view-mask finalizer inspired by Split&Splat
- Official-like Split&Splat propagation baseline
- Gaussian Grouping's DEVA/SAM pseudo labels

Observed behavior:

- SAM2 automatic masks can cover the scene well, but labels are only local to
  each view and often overlap, split, or merge differently across views.
- SAM2Object gives a representative video-propagation baseline, but frame order
  is only an approximate video for our DSLR capture. It can merge large facade
  regions and drift across viewpoint changes.
- Split&Splat-style propagation is helpful conceptually, but still depends on
  the quality of SAM2 proposals. If the initial 2D proposal merges wall, arch,
  and detail together, 3D voting stabilizes that mistake rather than recovering
  the desired split.
- SAM2 prompting from SAI3D points did not reliably create cleaner masks. It can
  produce hatching, holes, or inconsistent coverage when positive/negative
  prompts fall near fuzzy boundaries.
- DEVA in Gaussian Grouping tracks object-like masks, but building elements with
  weak boundaries are frequently over-grouped.

## Tested Baselines

### Common SAM2 View Proposals

Implementation:

```text
omega_local/segmentation/view_proposals_baseline.py
scan_processing/VisSegOmega08_visualize_view_proposals.py
```

Key idea:

Run SAM2 automatic mask generation independently on each DSLR view at the chosen
working resolution, then convert overlapping masks into a mutually exclusive
integer proposal map.

How we tried it:

- Used all 214 DSLR frames.
- Used the cleaned common output folder:

```text
<model_dir>/segmentation/baselines/view_proposals_1024
```

What we found:

- This is a useful shared initializer, not a final segmentation.
- It gives good local mask evidence and coverage.
- It has no cross-view identity, and architectural components can be merged or
  split differently from view to view.

### GeoSAM2

Implementation status:

Earlier baseline output exists under:

```text
<model_dir>/segmentation/baselines/geosam2
```

Key idea:

Render the mesh into multi-view geometric buffers, segment rendered views with
promptable 2D masks, then back-project the result to per-face 3D labels.

How we tried it:

- Staged the OMeGa mesh as a GeoSAM2 input mesh.
- Rendered textureless mesh views using Blender.
- Ran GeoSAM2 auto segmentation.
- Normalized the output into OMeGa-style face labels and a labeled mesh.

What we found:

- Auto mode produced essentially one large label for the mesh in our run.
- The rendered inputs were uncolored textureless mesh views, so visual facade
  cues were unavailable.
- The method is good for promptable mesh-part segmentation when geometry itself
  explains the part, but our OMeGa mesh is too messy and too visually ambiguous
  for automatic designer-level facade splits.

Conclusion:

GeoSAM2 is not the right core baseline for this stage. It may still be useful
later as an interactive mesh-part tool after geometry is cleaner.

### SAMPro3D

Implementation status:

Earlier baseline output exists under:

```text
<model_dir>/segmentation/baselines/sampro3d
```

Key idea:

Use SAM-style 2D proposals and 3D point/superpoint consistency to assign
segments in 3D.

How we tried it:

- Connected the SAMPro3D repository to our OMeGa dataset.
- Generated proposals and normalized predicted labels back to the OMeGa mesh.
- Visualized proposal progress and final mesh labels.

What we found:

- It produced recognizable 3D labels, but there were incomplete regions and
  messy boundaries.
- Like other proposal-driven approaches, the final quality was limited by 2D
  proposal coverage and by imperfect geometry.
- The result was useful as evidence, but not clean enough as a final
  architectural segmentation.

Conclusion:

SAMPro3D confirmed that 3D lifting helps consistency, but SAI3D was a cleaner
base for our later extensions.

### SAM2Object

Implementation:

```text
omega_local/segmentation/sam2object_baseline.py
scan_processing/VisSegOmega03_visualize_sam2object_baseline.py
```

Key idea:

Treat the ordered DSLR views as a pseudo-video. Run SAM2Object-style keyframe
segmentation, forward/reverse propagation, merging, and a mesh graph
consolidation step.

How we tried it:

- Prepared all 214 frames through the common proposal source.
- Ran forward/reverse mask propagation.
- Ran mesh-graph consolidation over the OMeGa mesh.
- Visualized forward, reverse, merged, and graph-consistent masks.

What we found:

- It is the right representative for video-propagation methods.
- It can track object-like regions, but facade/building components often get
  merged into larger masks.
- The final masks are not guaranteed to represent the same fine-grained element
  in every view.
- Because our views are DSLR captures with jumps between frames, the "video"
  assumption is only approximate.

Conclusion:

Good baseline for object propagation, not sufficient for precise architectural
part segmentation.

### SAI3D

Implementation:

```text
omega_local/segmentation/sai3d_baseline.py
scan_processing/VisSegOmega05_visualize_sai3d_observe.py
scan_processing/VisSegOmega04_visualize_sai3d_baseline.py
third_party/SAI3D_DT
```

Key idea:

Build a graph over 3D primitives, observe which 2D proposal labels they receive
across views, group local primitives into superpoints, and progressively merge
superpoints whose multi-view mask observations agree.

How we tried it:

- Raw OMeGa mesh vertices.
- Area-balanced OMeGa mesh samples.
- Higher resolution common proposal masks at 1024px width.
- Higher superpoint target for denser local evidence.
- MapAnything point cloud as the 3D graph.
- COLMAP sparse point cloud as the 3D graph.

Important run:

```text
<model_dir>/segmentation/baselines/sai3d_area_samples_1024_dense
```

What we found:

- SAI3D gives the best cross-view consistency because labels are resolved in
  3D.
- Area-balanced mesh sampling is better than raw vertices because large planar
  surfaces receive fairer support.
- The final `mesh_labels/labeled_points.ply` is often more meaningful than the
  final per-view masks.
- The per-view finalization step remains hard. Pure projection is consistent but
  sparse/fuzzy; SAM2 refinement can make boundaries sharper but can also
  introduce inconsistent coverage or overmerged masks.

Point-cloud variants:

- MapAnything gives broad coverage and useful semantic grouping, but noisy
  layered points make it less clean as a final 3D label source.
- COLMAP sparse points focus on key texture/detail regions. With conservative
  graph settings, they produce useful oversegmented detail evidence.

Conclusion:

SAI3D is the current best automatic endpoint. We should treat its 3D labels as
the canonical state, then improve masks by projection, local boundary cleanup,
and interactive edits.

### Split&Splat-Style Tests

Implementations:

```text
omega_local/segmentation/sai3d_baseline.py
omega_local/segmentation/split_splat_baseline.py
scan_processing/VisSegOmega10_visualize_split_splat_baseline.py
```

We tested two related ideas:

- a SAI3D finalizer inspired by Split&Splat;
- an official-like Split&Splat split-stage port that does not use SAI3D labels.

Key idea:

Use 3D point projection to turn per-view SAM2 masks into global labels. Later
views see a projected "virtual mask" from existing 3D labels, then local SAM2
proposals are matched to those labels by overlap. The method can also use SAM2
prompt fallback for visible labels that do not match a proposal.

How we tried it:

- Used 160k area-balanced OMeGa mesh samples.
- Used all 214 frames.
- Used common SAM2 proposals at 1024px.
- Wrote debug outputs for projected points, proposal assignment, SAM2 fallback,
  final masks, and per-instance masks.

What we found:

- It is a useful faithful baseline for mask-to-3D propagation.
- It still inherits initial SAM2 proposal errors.
- Some views have final masks that miss elements; other views let large masks
  replace smaller intended segments.
- The method is stronger when object proposals are already reliable. Our target
  facade/building elements often require splits that SAM2 did not propose
  cleanly.

Conclusion:

Split&Splat logic is useful for reasoning about projection and geometric IoU,
but it does not solve our main segmentation challenge automatically.

### Gaussian Grouping

Implementation:

```text
omega_local/segmentation/gaussian_grouping_pipeline.py
scan_processing/VisSegOmega09_visualize_gaussian_grouping.py
RUN_OMEGA_GAUSSIAN_GROUPING.md
third_party/gaussian-grouping
```

Key idea:

This is a training-time Gaussian Splatting segmentation method. It prepares
DEVA/SAM pseudo labels, trains Gaussians with identity features, and renders
predicted object masks and identity-feature PCA visualizations.

How we tried it:

- Staged the OMeGa COLMAP dataset into Gaussian Grouping's native layout.
- Tested scale 4 first, then scale 2 for sharper masks.
- Installed and built its modified rasterizer and DEVA/SAM dependencies.
- Tuned pseudo-label preparation parameters:
  - preserve small objects;
  - SAM predicted IoU threshold;
  - SAM points per side;
  - detection interval;
  - max object count.
- Trained and rendered the method faithfully, without feeding it SAI3D or our
  other custom labels.

What we found:

- The identity PCA visualization is promising: it shows that learned Gaussian
  features can separate different scene regions even when the final classifier
  masks are coarse.
- The pseudo labels and predicted masks still over-group many building elements.
- Lowering SAM points per side improved coverage in our test, but did not fully
  solve building-element granularity.
- The final result appears limited mainly by DEVA/SAM pseudo-label assumptions:
  they track object-like regions better than facade subparts.

Conclusion:

Gaussian Grouping is conceptually relevant for joint segmentation and 3DGS
optimization, but as a faithful baseline it did not produce the clean
designer-friendly masks we need. Its identity-feature learning remains
interesting for future optimization-integrated segmentation.

## Interactive Segmentation Prototype

Implementation:

```text
omega_local/segmentation/interactive/
scripts/run_omega_segmentation_editor.py
INTERACTIVE_SEGMENTATION.md
```

Key idea:

Make SAI3D's 3D labels editable. The user interacts through familiar RGB
frames, but edits change the underlying 3D point labels. Reprojection then makes
the edited state visible in all views without a separate propagation algorithm.

Current interface:

- central 3D point viewer;
- bottom frame strip;
- click a frame to align the 3D view to that camera;
- portrait rotation for easier DSLR viewing;
- smooth frame-to-frame interpolation;
- pan/zoom within an active frame using a sub-frustum rather than moving the
  capture camera;
- floating object/label list;
- local selection tools:
  - pick label;
  - lasso visible points;
  - SAM2 prompt mask;
- local edit actions:
  - assign selected points to an ID;
  - create new object from selection;
  - merge objects;
  - save edits separately from the original SAI3D labels;
  - undo local edits.

Why this matters:

The automatic baselines all struggle with designer intent. The editor gives us a
practical path: let automatic SAI3D provide a consistent 3D oversegmentation,
then let users group, split, and correct it at the level of 3D evidence.

## Main Failure Modes We Observed

### SAM-style proposals are object-centric

SAM2, DEVA, and video propagation often identify object-like regions, not
architectural design elements. A wall, arch, stair, and adjacent facade detail
can be treated as one object if image boundaries are weak.

### 2D masks are not naturally view-consistent

Even when masks look good in one frame, another frame may split or merge the
same region differently. Propagation helps, but only if the initial masks align
with the desired parts.

### Geometry helps consistency but not always boundary quality

3D labels are consistent across views, but projection can look fuzzy or sparse
because the OMeGa mesh/points are imperfect and boundaries are not perfectly
aligned with RGB edges.

### Mesh quality limits mesh-first segmentation

If a railing, fold, hole, or facade detail is absent or messy in the mesh, a
mesh-only segmenter cannot invent the correct boundary. It can only segment what
the mesh represents.

### Final per-view masks should not override 3D identity

The strongest lesson from Split&Splat/SAM2 refinement is that 2D masks should be
used as boundary evidence, not as the authority for object identity. Large 2D
masks can erase smaller valid 3D segments.

## Current Recommended Direction

Use SAI3D-style 3D labels as the canonical segmentation state:

```text
common SAM2 view proposals
  -> SAI3D over area-balanced OMeGa mesh samples
  -> optional SAI3D evidence over MapAnything / COLMAP points
  -> interactive 3D label editing
  -> project edited labels to all views
  -> optional local 2D boundary cleanup constrained by 3D identity
```

For the next stage, the per-view mask finalizer should be conservative:

- render/project the canonical 3D labels first;
- use RGB/SAM2/depth/normal boundaries only in a narrow local band;
- reject 2D proposals that merge multiple confident 3D labels;
- allow an `ignore` region near uncertain boundaries rather than forcing wrong
  full coverage;
- keep user edits as hard constraints.

## Presentation-Framing Summary

Slide title: `From Automatic Masks To Editable 3D Segmentation`

Key points:

- We tested mesh, 3D, video, splat-training, and Split&Splat-style segmentation
  baselines on the OMeGa grove entrance output.
- Mesh-only methods are limited by current reconstruction artifacts.
- 2D/video methods can produce clean object masks but often fail to split
  architectural elements consistently.
- SAI3D is currently the strongest anchor because its labels live in 3D and are
  naturally cross-view consistent.
- The promising direction is not fully automatic segmentation; it is automatic
  3D oversegmentation plus interactive designer correction.
- Final masks should be rendered from edited 3D labels, with SAM2/RGB/depth used
  only for local boundary refinement.

Representative visualizations:

```text
visualizations/omega_segmentation/view_proposals/
visualizations/omega_segmentation/sam2object/
visualizations/omega_segmentation/sai3d/
visualizations/omega_segmentation/sai3d_observe/
visualizations/omega_segmentation/split_splat/
visualizations/omega_segmentation/gaussian_grouping/
```

