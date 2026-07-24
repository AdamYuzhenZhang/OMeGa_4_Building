# Interactive Multi-View Segmentation: Presentation Outline

This note is the short, presentation-facing version of the interactive
segmentation project. It explains the motivation, core research question,
pipeline, and demo structure without going into every implementation detail.

## 1. Problem

Most recent multi-view segmentation systems are very good at making automatic
segmentation more consistent, but they still depend on the initial segmentation
prior. In practice that prior is usually SAM2 or a SAM-like object mask.

That is a problem for architectural and graphics reconstruction:

- a designer may want to split one wall into planar reconstruction regions;
- a facade element may not be a semantic object;
- trim, railings, arches, seams, and material regions can be visually ambiguous;
- the correct boundary is often defined by the downstream reconstruction task,
  not by object category.

So the failure mode is not only inconsistency. The deeper issue is intent. If
SAM2 proposes the wrong kind of region, then SAM2Object, SAI3D, GeoSAM2, or a
Split&Splat-style fusion step can make that wrong intent more consistent, but
they cannot know the designer's intended split.

## 2. Research Question

The project question is:

```text
How can a small number of user edits on a posed image capture dataset be turned
into a consistent, reconstruction-ready segmentation across all frames?
```

The important shift is:

```text
automatic segmentation as final answer
  -> user-defined segmentation intent, accelerated by automatic tools
```

SAM2 remains useful, but its role changes. It is not the source of truth. It is
one accelerator among several tools that help the user define persistent
regions.

## 3. Core Pipeline Story

The pipeline separates four ideas that are often mixed together:

| Concept | Meaning | Why It Exists |
| --- | --- | --- |
| Local proposals | Per-frame SAM2 or propagated candidate masks | Fast suggestions, but not trusted as final labels |
| Selected pixels | The temporary pixel region the user is currently editing | A common representation for pick, lasso, SAM2 prompts, normal grow, and RGB-D cue |
| Persistent regions | Named dataset-level regions such as `door`, `left facade plane`, or `railing` | The user's reconstruction intent |
| Propagated proposals | SAM2 video masks created from confirmed persistent-region keyframes | Suggestions for later frames, not ground truth |

The workflow is:

1. Generate automatic per-frame SAM2 proposals.
2. Choose useful keyframes for manual inspection.
3. On one keyframe, the user creates selected pixels with pick, lasso, SAM2,
   normal grow, or RGB-D cue.
4. The user commits selected pixels to named persistent regions.
5. Completed keyframes become hard anchors.
6. SAM2 video propagation uses those anchors to create region-aware proposal
   layers on other frames.
7. The user opens later keyframes, uses propagated proposals as a better prior,
   corrects them, and commits more persistent regions.
8. After enough keyframes are confirmed, a later fusion/export step produces
   reconstruction-ready masks for all frames.

This makes the interaction iterative: each edited keyframe improves the
suggestions on future keyframes, so the user should do less manual work over
time.

## 4. Why Each Step Makes Sense

| Step | What To Demo | Why It Is Designed This Way |
| --- | --- | --- |
| Raw capture context | Open the viewer, click frames, show the 3D points over posed images | The dataset is posed, so every 2D edit lives in a shared camera system |
| SAM2 proposals | Toggle the SAM2 layer and pick masks | SAM2 is a strong local proposal generator, but proposals are frame-local and object-biased |
| Local editing | Use lasso, SAM2 prompts, normal grow, or RGB-D cue to refine selected pixels | Different regions need different tools; all tools output the same selected-pixel mask |
| Persistent regions | Create or assign a named region from selected pixels | This is where user intent becomes explicit and global |
| Completion flag | Mark a keyframe complete | Only complete frames are trusted as propagation anchors, avoiding partially labeled views as hard constraints |
| Propagation | Rebuild the propagated layer and compare it against SAM2 proposals | Propagation is used as an editable prior for later frames, not as final segmentation |
| Iteration | Edit a later keyframe using propagated suggestions, then propagate again | The system becomes more helpful as more user-confirmed anchors are added |
| Final export | Explain the future exclusive label maps, confidence, and uncertainty | Reconstruction needs masks that are consistent, inspectable, and safe to use as supervision |

## 5. Main Claim

The contribution is not that we replace SAM2. The contribution is a structured
interactive pipeline that lets a user define task-specific multi-view regions,
then uses SAM2, geometry, normal/depth evidence, and camera connectivity to
amplify those edits into a reconstruction-ready dataset.

This is especially relevant when segmentation is used to control reconstruction:

- object-wise or layer-wise 3DGS;
- region-conditioned OMeGa optimization;
- per-part remeshing settings;
- material or facade decomposition;
- cleaner editability for designers.

## 6. Suggested Demo Flow

Use the app demo to make the research idea concrete:

1. Show why raw SAM2 proposals are useful but not enough.
2. Edit one keyframe with pick, lasso, SAM2 prompts, normal grow, or RGB-D cue.
3. Turn selected pixels into named persistent regions.
4. Mark the frame complete.
5. Propagate from complete frames and show the propagated layer as suggestions.
6. Open a later keyframe and show that the propagated suggestions are easier to
   correct than starting from SAM2 proposals alone.
7. Explain that final reconstruction-ready masks come from a later
   multi-anchor fusion/export stage.

The demo message should stay simple:

```text
The user defines the meaning. The system spreads that intent across views.
```
