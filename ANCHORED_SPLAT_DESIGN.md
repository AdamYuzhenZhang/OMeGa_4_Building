# Anchored Splat Design

## Status

Implemented as the isolated `anchored_3dgs` reconstruction backend.

- Paper: https://arxiv.org/html/2602.03809v1
- Released source: https://github.com/LTTM/Split_and_Splat
- Local source: `third_party/Split_and_Splat`
- Runner: `scripts/run_omega_anchored_splat.py`

The default experiment preserves the anchored masks exactly and executes no
SAM2 refinement.

## Research Question

Can the persistent regions defined by the interactive editor and consolidated
by the anchored Split stage reconstruct cleaner independent 3DGS regions than
the paper pipeline, without allowing a later SAM2 pass to reinterpret the
user's segmentation intent?

The controlled experiments are:

| Experiment | Split input | Region training | Mask refinement | Composition |
| --- | --- | --- | --- | --- |
| A | Released automatic masks and Split | Released | Released SAM2 | Released |
| B | SAM2 Video masks and released Split | Released | Released SAM2 | Released |
| C | Persistent masks and anchored 3D Split | None | None | None |
| D0 | Experiment C partition | Anchor-weighted 3DGS | None | Released ordering |
| D1 | Experiment C partition | Same as D0 | Released SAM2, anchor guarded | Released ordering |

D0 is the primary run. D1 is an explicit diagnostic that measures whether the
paper refiner helps or damages user-defined architectural regions.

## Isolation

Released Split&Splat runs remain under:

```text
interactive/experiments/split_splat/runs/<run-id>/
```

The anchored backend writes only to:

```text
interactive/reconstruction/runs/anchored_3dgs/<run-id>/
  run.json
  logs/
  01_region_datasets/
  02_initial_models/
  04_mask_refinement/
  05_refined_models/
  06_composition/
  07_outputs/
  work/
```

The manager rejects output paths inside the source run or the baseline
experiment root. It does not edit the released checkout.

## Inputs

The first backend consumes Experiment C directly:

```text
split_splat_anchored_sam2_video/
  split/02_point_labels/
    point_labels.npy
    point_label_confidence.npy
    point_label_provenance.npy
    upstream_instances/<region-id>/*.png
  split/03_consistent_masks/
    training_view_weights.json
    stage.json

shared Split&Splat cache/
  01_input/dataset/
  01_input/frame_map.jsonl
  02_global_gs/model/.../point_cloud.ply
```

Required invariants:

1. The Gaussian PLY and `point_labels.npy` have identical row counts.
2. The source Split uses the `persistent_region` label namespace.
3. Persistent region IDs are not renumbered.
4. Every shared global Gaussian has exactly one persistent-region owner.
5. Every persistent region represented by a mask has an initializer and is
   reconstructed; no region is silently skipped.
6. Source masks, labels, and baseline outputs are read-only.
7. Views without a region mask are omitted, not converted into negative views.
8. Every weight or cue supplied to a backend is either applied or reported as
   unused.

### Complete 3D Ownership

Anchored Split uses the shared global 3DGS as its geometry carrier. Gaussian
means are z-buffered against one another in every camera; predicted monocular
depth is not a hard visibility gate.

For Gaussian \(g\) and persistent region \(r\):

```text
E(g, r) = 8 * manual_votes(g, r) + propagated_votes(g, r)
```

Manual winners define identity and propagated evidence breaks manual ties.
Every Gaussian with positive evidence is assigned to its winning region even
when confidence is low. Confidence and provenance remain diagnostics, not
deletion tests.

The small set with no direct observation is completed from a confidence-
weighted local 3D nearest-neighbor consensus. The result is an exclusive,
complete partition of the shared global 3DGS. Projecting that ownership field
back to nonmanual views may correct the identity of a connected propagated
component, but never changes its foreground shape. Completed manual masks pass
through exactly.

## Stages

```text
prepare -> initial -> refine_masks -> refined -> compose
```

Each stage has a machine-readable `stage.json` and is independently resumable.
`refine_masks=none` is a completed stage, not a hidden skip.

### 1. Prepare

For persistent region \(r\), select the global Gaussian rows whose anchored
label is \(r\). Export the complete PLY records, retaining:

```text
xyz
SH appearance
opacity
scale
rotation
descriptor fields
source identity fields
```

Confidence and provenance are saved as aligned sidecars. A conventional
XYZ/RGB PLY is also written only to satisfy the released Scene loader; it is
not the optimizer's actual initializer.

The runtime launcher replaces that provisional initialization with the exact
full-Gaussian subset after Scene has initialized camera normalization and the
region's differentiable identity color.

Every region dataset contains:

```text
images -> shared RGB images
sparse/0/{cameras.txt,images.txt}
sparse/0/points3D.ply
initializer/gaussians.ply
masks/*.png
training_view_weights.json
```

The prepare stage validates minimum Gaussian and positive-view counts. Failing
validation stops the run with the affected region ID; it never silently omits
that region and continues with a scene containing a hole.

### 2. Initial Region Training

Training executes an in-memory copy of the released `train.py` through
`train_launch.py`; the source file is never edited.

For eligible view \(v\), the anchored Split sidecar gives raw weight
\(a_v=8\) for a completed manual frame and \(a_v=1\) for a propagated frame.
For each region, normalize over its eligible views:

```text
w_v = a_v / mean_{u in eligible(r)} a_u
```

The custom launcher retains the released renderer and random-view schedule but
uses the same foreground-balanced objective as the MapAnything variant:

```text
M(p)  = 1[any channel of T(p) is nonzero]
e(p)  = mean_channel |Q(p) - T(p)|
L_id  = 0.5 * (mean_{M=1} e + mean_{M=0} e)
L_rgb = mean_{M=1} |C - I|
L     = L_rgb + lambda_mask * w_v * L_id
```

Here `Q` and `T` are the rendered and target RGB identity fields. The target is
premultiplied by the camera alpha before support is derived; this is necessary
because the released loader otherwise drops alpha while retaining hidden RGB
values. `lambda_mask=1` by default. Only identity supervision is anchor weighted.
Full-frame SSIM is disabled because it would let empty background dominate
small architectural regions. The change is applied only by the private
anchored launcher; the released Split&Splat baseline source and objective are
untouched. The random view schedule is unchanged, so every eligible propagated
view remains available.

The default is 1,000 iterations, matching the paper's dense ScanNet setting.
The paper uses 10,000 for LERF, so iteration count remains configurable.
Anchored training starts from the exact full-attribute Gaussian subset and
disables the released early densify-and-prune cycle. Positions, appearance,
scale, rotation, and opacity can be refined, while every initializer Gaussian
survives the per-region warm-up. After optimization, a separate cleanup removes
opacity below `0.005` and Gaussians repeatedly front-visible outside the
region masks. This keeps topology control during optimization while preventing
the latent transparent scaffold from entering composition.

### 3. Mask Refinement

#### `none` (default)

No masks change and SAM2 is not loaded or executed. The stage records:

```text
policy = none
maskMutation = false
sam2Executed = false
```

The `refined` stage links the initial models as composition inputs, avoiding a
redundant second training pass.

#### `paper_sam2` (explicit diagnostic)

This policy calls the already validated released Split&Splat refiner inside the
anchored run's private datasets. It uses Gaussian reprojection, five spatially
distributed positive prompts, SAM2 candidates, and the paper IoU decision.

Before refinement, every manual region/frame mask state is snapshotted. After
the released call:

- existing manual masks are restored byte-for-byte;
- masks absent on a manual frame are restored as absent;
- source Split masks and D0 outputs remain untouched.

The refined models are then trained from the same full-Gaussian initializers.
This policy does not yet implement joint cross-region arbitration; it is a
paper-refiner ablation, not the proposed final refinement method.

### 4. Composition

The first anchored backend reuses the validated Split&Splat collision-driven
composition:

1. compute directional AABB containment scores;
2. choose nonconflicting collision pairs;
3. concatenate the pair's Gaussians;
4. reset opacity;
5. optimize 1,000 iterations with densification disabled;
6. use mask weights `0.05`, `0.15`, then `0.25`;
7. repeat until no selected pair remains.

Before composition, every region also writes `floater_pruning.json`. The pass
adapts semantic whitelist and depth-buffer visibility from
[Clean-GS](https://arxiv.org/abs/2601.00913): low-opacity Gaussians and
splats larger than `0.1` times the camera extent are removed,
and an unsupported center must be front-visible in at least three positive-mask
views before removal. One-pixel mask dilation protects boundaries. Spatial/kNN
outlier pruning is disabled because isolated railings and ornaments can be
valid. On a temporary copy of region 9, the standard opacity rule alone removed
220,345 of 313,290 latent Gaussians; the source result was not modified.
Anchored composition uses the same foreground-normalized RGB and balanced
identity objective; the released baseline keeps its original full-frame
objective. After collision-driven optimization, a second attribute-only pass removes any
low-opacity or oversized splats reintroduced by opacity reset. It filters the
aligned instance labels and regenerates all final object exports.

The composed output now includes `appearance_diagnostics.json`. The diagnostic
separates near-white DC color from opacity and scale, which is important here:
the shared global initializer contains many near-white Gaussians below the
standard `0.005` opacity pruning threshold. Their color alone is not a visible
error; the failure occurs when isolated training increases their opacity or
scale. The balanced identity/RGB objective is intended to prevent that conversion
without silently deleting the complete global partition. Browser artifacts are
render-only standard 3DGS PLYs; canonical outputs retain identity descriptors.

The implementation reuses the released composition functions but passes only
the anchored run's private datasets and models. Persistent IDs are carried in
an aligned label array and exported with the composed scene.

## Current Scope

Implemented:

- complete global-3DGS ownership from direct votes plus local 3D completion;
- exact full-Gaussian region initialization;
- topology-preserving per-region warm-up followed by semantic support pruning;
- persistent-ID preservation;
- manual-anchor mask-loss weighting;
- default no-refinement D0;
- optional anchor-restored paper SAM2 D1;
- isolated caching and composition;
- individual and composed Gaussian artifacts.

Deferred:

- pixel-level known/unknown support;
- confidence-weighted boundary losses;
- joint candidate arbitration across regions;
- a non-SAM boundary refinement method;
- renderer-contribution or probabilistic-existence pruning for ambiguous splats;
- analogous 2DGS and OMeGa reconstruction backends.

These deferred items should be introduced as separate ablations after D0 is
inspected, not folded invisibly into the baseline.
