# Interactive SAM2 Proposal Segmentation

Status: new design direction for designer-controlled building-element
segmentation.

## Motivation

Our segmentation tests show a consistent limitation: SAM2, DEVA, and
SAM2Object-style video propagation are strong at object-like masks, but weak at
designer-facing building elements. Facade parts, arches, stair pieces, wall
panels, trims, and sculptural details often do not have the object boundaries
that these methods expect.

This means user interaction should happen earlier than our current final
projection stage. Instead of asking automatic SAM2 masks to define semantics,
we should let users edit SAM2 proposals in selected keyframes, assign labels,
and then treat those edits as high-confidence evidence during 3D fusion.

## Core Principle

The canonical segmentation should still live in 3D, but the user should define
or correct semantics in 2D keyframes.

```text
automatic SAM2 proposals
  -> user-edited keyframe masks
  -> optional propagation to overlapping frames
  -> weighted SAI3D-style 3D fusion
  -> consistent 3D labels
  -> projected per-view masks
  -> optional constrained 2D boundary cleanup
```

SAM2 should propose regions and sharpen local boundaries. SAI3D-style 3D fusion
should enforce multi-view identity. User edits should define the semantic
building-part labels.

## Inputs We Already Have

- Posed DSLR frames and intrinsics from the OMeGa dataset folder.
- SAM2 automatic proposal maps from the common view-proposal initializer.
- SAI3D 3D points or superpoints sampled from the OMeGa mesh, MapAnything point
  cloud, or COLMAP sparse point cloud.
- Visibility, projection, and depth-consistency code used by the SAI3D and
  Split&Splat-style tests.
- The interactive point editor, including frame navigation, SAM2 prompts,
  lasso selection, object list, undo, and explicit save.
- The editor now opens from the sampled SAI3D input point cloud by default:
  labels start as `0`, points render white, and user-created IDs are layered on
  top of this raw geometry instead of starting from SAI3D's automatic labels.

## Proposed Pipeline

### 1. Keep Rich SAM2 Proposal Candidates

The proposal source should preserve more than the final flattened integer mask.
For each frame, keep:

- candidate mask ID;
- binary mask or compressed RLE;
- predicted IoU and stability;
- area;
- mask bounding box;
- ranking score;
- final visible region after overlap resolution.

The flattened map is still useful, but interactive editing needs the original
mask candidates so the user can split, merge, delete, or relabel proposals.

### 2. Keyframe Selection

The editor now includes an automatic keyframe suggestion step next to SAM2
proposal generation. It follows SAM2Object's idea of selecting frames with
large visual changes, but also uses our known camera poses.

For every frame `i`, it computes:

```text
image_change_i       = mean |LUV_i - LUV_{i-1}|
translation_change_i = ||C_i - C_{i-1}||
rotation_change_i    = angle(R_{i-1}^T R_i)
```

Each channel is robustly normalized and combined:

```text
score_i = 0.55 * image_i + 0.25 * translation_i + 0.20 * rotation_i
```

The detector smooths this score, selects local peaks, always keeps the first
and last frame, removes keyframes that are too close, and inserts extra
keyframes when gaps are too large. The result is saved in:

```text
interactive/keyframes/keyframes.json
```

The bottom frame strip shows selected keyframes with a red marker and shows the
continuous score as a small bar. These are inspection priorities for the user,
not hard segmentation constraints.

Good keyframes should:

- cover many visible 3D points;
- have clear view angle and low occlusion;
- include uncertain or inconsistent labels from previous automatic runs;
- represent different sides of the scene rather than only adjacent frame
  indices.

For DSLR captures, adjacency should be camera-overlap adjacency, not video
frame index.

### 3. User Edits In Keyframes

The editor should support:

- choose an existing SAM2 mask;
- merge masks into one named building element;
- split a mask using lasso, brush, or SAM2 prompt;
- paint add/remove regions;
- assign or rename a semantic label;
- mark two labels as keep-separate siblings;
- mark ignored or unknown regions where the user does not want a hard decision.

Each edit should be stored as an append-only operation log, plus a saved
keyframe label map for fast reload.

### 4. Propagate Edited Masks To Nearby Frames

The first version should propagate only to nearby overlapping frames.

Possible propagation evidence:

- SAM2 video propagation over a short local sequence;
- SAM2 image prediction from projected positive/negative prompts;
- projection of currently labeled 3D points;
- depth and visibility checks from the mesh or point cloud.

Propagation confidence should be lower than direct user edits. A propagated mask
should never override a conflicting user-edited keyframe.

### 5. Weighted 3D Fusion

For a 3D point or superpoint `p` and label `l`, accumulate votes from visible
2D observations:

```text
score(p, l) = sum_o w_o * visible(p, o) * inside(mask_l_o, project(p, o))
```

Observation weights should follow this order:

```text
user-edited keyframe       highest
user-approved propagation  high
automatic propagation      medium
automatic SAM2 proposal    low
unlabeled / ignored        no hard vote
```

The score should also be reduced by:

- projection near mask boundary;
- depth mismatch or occlusion;
- grazing view angle;
- low SAM2 confidence;
- high disagreement across nearby views.

Keep-separate constraints should act as hard graph barriers during superpoint
merging.

### 6. 3D Label Solve

Use a SAI3D-style graph over points or superpoints:

- nodes are points or superpoints;
- unary terms come from weighted 2D mask evidence;
- pairwise terms connect nearby 3D samples;
- pairwise affinity is high when samples are close, co-visible, and receive
  compatible labels;
- pairwise affinity is reduced across depth, normal, color, or user-declared
  label boundaries.

The result is a canonical 3D label field. This should be the source of truth.

### 7. Per-View Mask Export

Project the final 3D labels to every view and build per-view masks.

Then optionally run local boundary cleanup, constrained by 3D identity:

- refine only near projected label boundaries;
- use SAM2, RGB edges, depth edges, and StableNormal edges as boundary cues;
- forbid a 2D refinement from merging two confident 3D labels;
- leave uncertain holes as ignore if no label has enough 3D support.

This should produce masks that are sharper than raw point projection but more
consistent than unconstrained SAM2 masks.

## Expected Outputs

```text
<model_dir>/segmentation/baselines/<sai3d-run>/
  dataset/
    frame_manifest.jsonl
    scans/<scene>/points.pts
      # initial unsegmented point cloud sampled from the OMeGa mesh and used
      # as SAI3D input; the editor opens from this cloud by default.

  interactive/
    interactive_labels.npy
      # saved user-created 3D point labels; never overwrites SAI3D labels.
    interactive_edits.jsonl
      # append-only edit log.

    keyframes/
      keyframes.json
        # SAM2Object-style image-change keyframes augmented with camera pose
        # deltas; used to guide user inspection.

    proposals/
      sam2_auto_<width>/
        source_summary.json
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

    keyframe_masks/
      objects.json
      edits.jsonl
      frames/
        000000/
          0001.png
          0001.json
      overlays/
        000000.png

    propagation/<future-run>/
      frame_<id>_label_scores.npz
      frame_<id>_propagated_labels.png

    fusion/<future-run>/
      point_label_scores.npz
      point_labels.npy
      labeled_points.ply
      superpoint_labels.npy

    view_masks/<future-run>/
      frame_<id>_projected.png
      frame_<id>_refined.png
      frame_<id>_confidence.png
```

The original automatic proposal outputs should stay immutable. User-edited
outputs should be a separate run folder so we can compare revisions.

Current implementation:

- The editor opens `dataset/scans/<scene>/points.pts` as raw white points.
- The right pipeline panel displays raw point cloud state, SAM2 proposal
  controls, frame-local SAM2 prompts, 3D label edits, and save state.
- The `Generate` button runs SAM2 automatic proposals for every staged frame in
  a background server thread when no completed run exists.
- After a completed run exists, `Load` displays it without recomputing SAM2,
  while `Regenerate` explicitly overwrites the interactive proposal run.
- The browser polls `progress.json` through `/api/proposals/sam2/status`.
- The `Overlay` toggle draws transparent proposal masks in the active frame and
  on the bottom filmstrip thumbnails.
- The first proposal run folder is `interactive/proposals/sam2_auto_1024/` for
  the current DSLR 1024-width staged frames.
- Mask-edit tools now live inside `Edit View Masks`, not in the global top
  toolbar. `Pick` selects the SAM2 proposal ID under the cursor. `Lasso`
  converts the drawn canvas polygon back to source-image coordinates and stores
  that pixel region directly; it does not select existing SAM2 proposal labels.
  `SAM2` adds optional point prompts for a separate SAM2 preview.
- `Pick` and `Lasso` share selection modifiers: plain action replaces the
  current mask/lasso selection, `Shift` adds, and `Option`/`Alt` subtracts.
  The `+` and `-` buttons can pin add/subtract mode, and holding the keyboard
  modifiers temporarily highlights the same buttons. The canvas cursor shows a
  small `+` or `-` badge when add/subtract is active. `Deselect` clears picked
  proposals and lasso pixel regions together.
- Internally, proposal editing is now a unified pixel-selection stack. Picked
  proposal IDs, lasso regions, and SAM2 masks all become add/subtract
  operations on one frame-image mask. The visible selection preview is the
  composed mask with one boundary.
- `Update Proposal` writes directly into the active frame's editable SAM2 label
  map. The backend chooses the target proposal ID from the current pixel
  selection: if a selected proposal ID is fully covered, that ID is reused;
  otherwise the edit creates `max_id + 1`. Unselected pixels keep their old IDs,
  so splitting a proposal never moves leftovers to background.
- Proposal updates save immediately because they are frame-local label-map
  edits. `Cmd/Ctrl+Z` remains for unsaved point-label edits and transient
  selection state.
- Proposal updates do not create global object IDs, names, or separate binary
  proposal masks. The edited frame label maps, overlays, and metadata live
  under `interactive/proposals/sam2_auto_1024_edited`.
- Proposal IDs are treated as opaque labels. Merging several full proposals may
  leave gaps in the ID sequence, and we intentionally do not reindex them during
  editing.

## Visualization To Build

Useful debug views:

- automatic SAM2 candidates before editing;
- user-edited keyframe masks;
- propagated masks on nearby frames;
- 3D point or superpoint label scores;
- final 3D labels;
- projected masks versus refined masks;
- disagreement/confidence map per frame;
- list of labels with number of supporting keyframes and 3D points.

## Implementation Steps

1. Extend the proposal initializer to retain raw SAM2 candidate masks and
   metadata.
2. Add keyframe proposal editing to the interactive webpage.
3. Save keyframe edits as explicit masks and operation logs.
4. Add a short-range propagation stage over camera-overlap neighbors.
5. Add a weighted SAI3D observation importer for edited and propagated masks.
6. Re-run the SAI3D-style label solve with user observations as strong unaries.
7. Export projected and refined per-view masks.
8. Add visualization contact sheets for each stage.

## Why This Is Better Than The Previous Direction

The earlier pipeline tried to correct inconsistent masks after automatic
segmentation. That puts too much trust in SAM2's initial object prior.

This version lets the user define the semantic split before 3D fusion. The
system then uses multi-view geometry to spread that semantic decision
consistently, while still using SAM2 for what it is good at: local mask
proposal and boundary refinement.
