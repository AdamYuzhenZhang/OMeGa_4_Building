# Segmentation To Reconstruction Architecture

## Research Goal

Build one modular pipeline that turns sparse designer intent in posed images
into region-aware reconstructions:

```text
posed capture
  -> 2D evidence and manual anchors
  -> labels on a chosen 3D carrier
  -> consistent per-view region supervision
  -> region initializers and training datasets
  -> 3DGS, 2DGS, OMeGa, or another reconstruction backend
  -> separate components and/or one composed scene
```

The central research object is not a particular point cloud or renderer. It is
a versioned scene partition with traceable 2D and 3D evidence. Geometry
carriers and reconstruction backends should be replaceable around that
partition.

## Lessons From Existing Methods

### SAI3D

SAI3D separates the 3D carrier from the 2D proposals:

1. over-segment a point cloud into geometric superpoints;
2. project the superpoints into posed images;
3. use 2D proposal agreement to construct superpoint affinities;
4. progressively merge the 3D graph into instance labels.

The useful abstraction is that any point set with calibrated visibility can
carry global region identity. The point set does not have to be the final
reconstruction.

### Split&Splat Split

Split&Splat trains a shared depth-regularized 3DGS and uses its Gaussian means
as a dense 3D carrier. It accumulates multi-view mask evidence on that carrier,
then projects global labels back to the images.

The useful abstraction is:

```text
2D masks + calibrated visibility + 3D carrier
  -> global 3D identity
  -> consistent per-view supervision
```

Our anchored Split adaptation changes the authority policy:

- completed manual frames are hard identity anchors;
- propagated masks are soft evidence;
- persistent region IDs are preserved;
- every directly observed Gaussian receives its best-supported identity;
- local 3D completion assigns the remaining shared Gaussians;
- the complete 3D ownership field may correct component identity;
- foreground mask shapes are not replaced by another SAM2 pass.

### Split&Splat Splat

The released Splat stage:

1. creates one training dataset per instance;
2. initializes it from the corresponding labeled Gaussian subset;
3. trains one 3DGS per instance;
4. renders geometry and uses it to prompt SAM2 for mask refinement;
5. retrains with the selected masks;
6. progressively composes colliding instances.

The useful abstraction is independent region warm-up followed by explicit
cross-region composition. Mask refinement is useful, but it must be optional
for designer-authored masks and must produce a new mask version instead of
overwriting trusted input.

## Stable Boundaries

The generalized system should use five explicit artifacts.

### 1. Capture Package

One immutable description of the posed dataset:

```text
capture/
  manifest.json
  frames.jsonl
  images/
  cameras/
  sparse/
  evidence/
    depth/
    normal/
```

Each frame row stores a stable frame ID, source image identity, dimensions,
intrinsics, pose, and paths. Every later artifact references these frame IDs.

### 2. Geometry Carrier

A geometry carrier supplies locations and calibrated visibility. It may be:

- COLMAP sparse points;
- MapAnything or MASt3R points;
- sampled mesh points or mesh vertices;
- OMeGa mesh-bound samples;
- global 3DGS Gaussian means;
- another point, mesh, or primitive representation.

```text
geometry/<carrier_id>/
  manifest.json
  geometry.ply
  elements.npz
  visibility/
  confidence.npy
```

`manifest.json` declares:

- carrier type and coordinate system;
- element count and stable element IDs;
- source fingerprint;
- optional color, normal, scale, opacity, and topology fields;
- visibility/depth policy;
- whether the carrier is shared input or trainable output.

Segmentation labels attach to stable element IDs. They must not depend on PLY
row order without recording that order in the manifest.

### 3. Scene Partition Package

This is the canonical output of Steps 1-6:

```text
partitions/<partition_id>/
  manifest.json
  regions.json
  anchors.json
  frame_maps/
    label/
    confidence/
    provenance/
    unknown/
  geometry_labels/
    <carrier_id>/
      labels.npy
      confidence.npy
      provenance.npy
  associations/
  diagnostics/
```

`regions.json` owns the stable designer-facing region namespace:

```json
{
  "regions": [
    {
      "regionId": 1,
      "name": "Door",
      "role": "persistent",
      "parentRegionId": null
    }
  ]
}
```

The same region ID means the same intended region in every frame and carrier.
Frame-local SAM proposal IDs and run-local Split IDs are never written into
this namespace without an explicit mapping.

Each pixel records four separate concepts:

- `label`: selected persistent region, or zero;
- `confidence`: strength of nonmanual evidence;
- `provenance`: manual, accepted propagation, automatic, projected 3D, or
  unknown;
- `unknown`: whether absence of a label may be treated as background.

Manual anchors remain immutable. A refinement method writes a child partition
with a parent fingerprint and changed-pixel diagnostics.

### 4. Region Training Package

This package freezes one partition and prepares backend-neutral reconstruction
ownership:

```text
training_packages/<package_id>/
  manifest.json
  regions.json
  components.json
  frames.jsonl
  masks/
  weights/
  initializers/
    <component_id>/
  evidence/
```

Persistent regions and trainable components are different:

- `region_id` expresses designer intent;
- `component_id` identifies parameters optimized together.

Several attached architectural regions may share one component to avoid
cracks, while preserving separate region labels. Context or ignored regions
can supervise visibility without receiving a standalone model.

For frame `v` and region `r`, the package supplies:

```text
M_vr(p)  target label or probability
W_vr(p)  supervision reliability
K_vr(p)  known/unknown support
```

A default reliability policy is:

```text
manual anchor       W = 1
user-accepted mask  W = 1
propagated mask     W = lambda_prop * confidence
automatic mask      W = lambda_auto * confidence
unsupported pixel   W = 0
```

The package also declares which initializer belongs to each component. An
initializer can be filtered COLMAP points, labeled feed-forward points, a
Gaussian subset, a mesh subset, or no geometry.

### 5. Reconstruction Run

Every backend consumes the same training-package identity and owns its outputs:

```text
reconstruction/runs/<backend>/<run_id>/
  config.json
  input_manifest.json
  progress.json
  components/
  mask_refinements/
  composed/
  renders/
  metrics.json
```

The backend declares capabilities before running:

```text
accepted initializer types
hard or soft mask support
per-view weight support
joint multi-component rendering
mask-refinement policies
composition strategy
output representation
```

An incompatible method should fail validation before training rather than
silently dropping confidence, weights, or initializer attributes.

## Pipeline Stages

### Partition Stage: Steps 1-6

```text
prepare capture
  -> generate 2D evidence
  -> edit manual anchors
  -> propagate region candidates
  -> label a selected geometry carrier
  -> project/fuse labels
  -> export Scene Partition Package
```

Methods are interchangeable at two points:

| Axis | Current or planned methods |
| --- | --- |
| 2D evidence | SAM2 Automatic, manual edits, SAM2 Video, XMem++, Cutie |
| Geometry carrier | COLMAP, MapAnything/MASt3R, OMeGa mesh samples, global 3DGS |
| 3D association | SAI3D progressive superpoints, released Split, anchored Gaussian voting |
| View-mask policy | source pass-through, 3D identity relabel, released SAM2 refinement |

These choices form a partition run. They should not be hidden inside the
reconstruction backend.

### Reconstruction Stage: Step 7

```text
Scene Partition Package
  -> assign regions to trainable components
  -> build Region Training Package
  -> optimize initial components
  -> optional geometry-guided mask refinement
  -> optional retraining
  -> optional joint composition
```

The initial backend matrix is:

| Backend | Ownership mode | Initializer | Main purpose |
| --- | --- | --- | --- |
| Released Split&Splat | independent 3DGS, then composition | labeled global Gaussians | paper baseline |
| Anchored Split + 3DGS | independent or composed 3DGS | persistent-region Gaussian subsets | test user-intent preservation and anchor weighting |
| 2DGS | shared or per component | points or Gaussian conversion | compare surface-oriented splats and geometry |
| OMeGa | shared or per component mesh-bound splats | mesh plus region-labeled samples | test structured mesh reconstruction and region-specific remeshing |

## Mask Refinement As A Versioned Method

Mask refinement should not be hard-coded into "Splat." Define it as an
optional transform:

```text
partition P0 + reconstruction R0
  -> refinement method
  -> child partition P1
  -> optional reconstruction R1
```

Initial policies:

1. `none`: train from the selected partition unchanged.
2. `identity_relabel`: preserve foreground shape and change only region IDs
   from reliable 3D evidence.
3. `released_sam2`: reproduce Split&Splat geometry-prompted SAM2 refinement.
4. `anchor_preserving_sam2`: future method that may change nonanchor masks but
   never completed manual frames.

Every refinement records:

- parent partition;
- input reconstruction;
- changed pixels and IDs;
- per-frame decision scores;
- anchor violations, which must remain zero;
- accepted and rejected candidates.

This design allows the released method to remain faithful while our method can
skip or replace its SAM2 refinement.

## Reconstruction Objective

A backend may implement its own geometry terms, but supervision should retain
the common reliability contract. For rendered color `C_v`, component opacity
`A_vr`, target mask `M_vr`, and known-pixel indicator `K_vr`:

```text
L = L_rgb
  + lambda_mask * sum_vrp K_vr(p) W_vr(p) BCE(A_vr(p), M_vr(p))
  + sum_c L_geometry(c)
  + lambda_overlap * L_component_overlap
  + lambda_boundary * L_shared_boundary
```

Important rules:

- manual and user-accepted frames may be sampled more often or weighted more;
- unknown pixels do not become negative background;
- independently trained components stay in one world coordinate system;
- composition or joint refinement is required to measure overlap and cracks;
- a backend must report when it ignores a supplied weight or cue.

The anchored reconstruction adapter consumes `training_view_weights.json`,
weights manual-frame mask supervision, and preserves every Gaussian in each
region initializer during warm-up. Released reconstruction runs remain
unchanged controls.

The MapAnything Region 3DGS pipeline is the first independent implementation
of this contract. Its partition keeps unsupported feed-forward points unknown,
performs identity-only mask cleanup, and exports ordinary RGB point
initializers. Its backend trains each region with densification enabled and
weights only the mask loss on completed manual frames. The initial scene is a
direct world-aligned concatenation, deliberately leaving joint boundary
optimization as a separate measurable stage.

## Experiment Design

Avoid an uncontrolled Cartesian product of every method. Freeze one artifact
at a time:

### Partition Comparison

Use one shared capture and geometry carrier:

```text
released Split
vs SAI3D-style point association
vs anchored 3D association
```

Measure 3D identity consistency, manual-anchor preservation, per-view region
IoU/boundary quality, unknown coverage, and runtime.

### Carrier Comparison

Use one fixed mask source and association policy:

```text
COLMAP
vs MapAnything/MASt3R
vs OMeGa samples
vs global 3DGS
```

Measure geometry coverage, thin-structure support, wrong-depth association,
and resulting view consistency.

### Reconstruction Comparison

Freeze one Scene Partition Package:

```text
released Split&Splat
vs anchored 3DGS
vs 2DGS
vs OMeGa
```

Measure held-out rendering, silhouette and boundary quality, cross-region
leakage, geometry accuracy, component overlap/cracks, primitive count, and
training cost.

### Refinement Ablation

Freeze the initializer and reconstruction backend:

```text
no refinement
vs identity-only relabel
vs released SAM2 refinement
vs future anchor-preserving refinement
```

This isolates whether refinement improves reconstruction or merely produces
more SAM-like masks.

## Baseline Isolation

Current paper runs remain owned by
`omega_local/segmentation/split_splat/`:

| Run | Proposal source | Split method | Splat behavior |
| --- | --- | --- | --- |
| `split_splat_official` | released four-grid SAM2 | `released` | released initial training, SAM2 refinement, retraining, composition |
| `split_splat_sam2_video` | editor SAM2 Video layer | `released` | same released Splat stages |
| `split_splat_anchored_sam2_video` | editor persistent regions | `anchored_3d` | consumed by the isolated `anchored_3dgs` backend; SAM2 refinement is off by default |

The generalized pipeline must follow these rules:

1. Do not rename, move, or reinterpret completed baseline artifacts.
2. Do not add generic backend behavior inside released stage functions.
3. Import baseline results through read-only exporters.
4. Put generalized contracts and reconstruction backends in new modules.
5. Give every derived partition and training package a new ID and parent
   fingerprint.
6. Keep upstream commands and paper parameter regression tests unchanged.

Recommended code ownership:

```text
omega_local/segmentation/split_splat/       # frozen paper adapter + Experiment C
omega_local/segmentation/partition/         # generic partition contracts/exporters
omega_local/reconstruction/region_pipeline/ # generic training package and runner
omega_local/reconstruction/backends/
  split_splat.py
  gaussian_splatting.py
  twodgs.py
  omega.py
```

The first backend is implemented under
`omega_local/reconstruction/backends/anchored_3dgs/`. It reads the anchored
Split partition without modifying the released Split&Splat manager, preserves
the full Gaussian initializer, applies manual-anchor mask weights, and exposes
`none` and `paper_sam2` as separate refinement policies.

The concrete stage graph, loss, refinement ablations, output layout, and
baseline safeguards for that backend are specified in
[`ANCHORED_SPLAT_DESIGN.md`](ANCHORED_SPLAT_DESIGN.md).

## Implementation Order

1. Run and inspect anchored D0 with `mask_refinement=none`.
2. Compare it against released Experiments A/B and the unweighted anchored
   `splat_initial` control.
3. Run `paper_sam2` only as a separate D1 diagnostic.
4. Freeze the current direct Experiment C input as a backend-neutral Scene
   Partition Package once the D0 data contract is validated.
5. Add a 2DGS backend using the same frozen training package.
6. Add OMeGa only after component ownership, visibility, and composition are
   understood with the simpler splat baselines.

This order gives each new result a controlled baseline and keeps the released
Split&Splat experiments runnable throughout development.
