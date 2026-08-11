# MapAnything Region 3DGS

Joint static-scene hard/soft identity baselines that reuse this pipeline's split outputs are documented in [STATIC_SEMANTIC_3DGS.md](STATIC_SEMANTIC_3DGS.md).

## Purpose

This is a custom reconstruction pipeline parallel to the released
Split&Splat experiments. It tests whether the feed-forward geometry already
used to initialize OMeGa can replace the expensive global 3DGS in the Split
stage.

```text
manual persistent-region masks + propagated masks
        -> MapAnything point ownership
        -> identity-cleaned per-frame masks
        -> [A] per-region point initializers
             -> independently trained region 3DGS
             -> world-aligned composed scene
        -> [B] one labeled shared initializer
             -> jointly trained ObjectGS scene
             -> scene RGB + persistent-region ownership
```

The pipeline has its own code and output root:

```text
omega_local/reconstruction/pipelines/mapanything_3dgs/

<interactive>/experiments/mapanything_region_3dgs/runs/<run-id>/
  01_input/
  02_split/
  03_splat/
  04_shared_objectgs/runs/<objectgs-run-id>/
```

It does not read or mutate a Split&Splat experiment run. The released
Split&Splat `train.py` is reused only as the tested masked 3DGS optimization
backend.

## Split Stage

The default geometry carrier is the exact MapAnything point cloud supplied to
OMeGa:

```text
<omega-run>/dataset/vfm_sparse/0/points3D.ply
```

For visible point `p` and persistent region `r`, evidence is accumulated as:

```text
E(p, r) = manual_frame_weight * manual_votes(p, r)
        + propagated_votes(p, r)
```

Completed manual frames are authoritative. Background ID `0` means unknown and
does not cast a negative vote. Points with no observation or an unresolved tie
are completed from confidence-weighted local 3D neighbors. This preserves every
initializer point instead of cutting holes at ambiguous ownership boundaries;
the completion confidence and distance statistics are recorded in the Split
summary.

The point ownership field is projected into each view with a front-surface
z-buffer. Manual masks pass through label-for-label without boundary changes. In
propagated frames, a
whole connected component may change persistent ID only when projected
ownership supplies:

- at least `max(4 pixels, 0.05% of the component)`;
- at least `55%` support for the winning ID.

Foreground pixels are never added or removed. This is identity cleanup, not
SAM2 boundary refinement.

Split outputs include:

- `point_labels/segmented_points.ply`: persistent-region colors;
- `point_labels/{labels,confidence,provenance}.npy`;
- `projected_support/`: sparse 3D evidence in every view;
- `clean_masks/label_maps/`: dense identity-cleaned masks;
- `instances/<region>/region_<region>.ply`: RGB point initializer;
- `instances/<region>/masks/`: binary training masks;
- `clean_masks/training_view_weights.json`.

## Splat Stage

Each persistent region is trained in the common OMeGa world coordinate system.
Only frames with a nonempty mask for that region are included. A missing region
in a view is therefore unknown, not negative background.

Training starts from that region's RGB MapAnything points. It does not load
pretrained Gaussian attributes. Standard 3DGS densification and pruning are enabled to recover
geometry that is absent from the feed-forward initializer. Densification stops halfway
through training, leaving the second half for stabilization. The custom
launcher also corrects the released trainer's iteration/count comparison:
`max_num_splats` now limits the current Gaussian count, and any one-step
overflow is pruned by lowest opacity.

After each region finishes, a conservative semantic cleanup runs before
composition. It removes nonfinite Gaussians, opacity below the standard 3DGS
`0.005` threshold, oversized splats whose largest axis exceeds `0.1` times
the camera extent, and centers that are front-visible in at least three positive
mask views but never land inside the region mask. Masks are dilated by one pixel
for boundary tolerance. This adapts the whitelist and depth-buffer ideas from
[Clean-GS](https://arxiv.org/abs/2601.00913). The opacity and world-scale
rules follow the original 3DGS adaptive-density controller. The pass
deliberately disables its
percentile/kNN outlier pass because sparse thin railings are valid geometry.
The DC-only color-validation stage is also disabled: a temporary real-scene test
showed that it over-prunes view-dependent high-order SH appearance. Each model
writes `floater_pruning.json` with before/after counts and removal reasons.

The custom launcher retains the released renderer and optimizer but corrects
the masked objective so a thin object is not diluted by the full image area.
For rendered identity field `Q`, target identity field `T`, rendered color `C`,
and target RGB `I`:

```text
M(p)  = 1[any channel of T(p) is nonzero]
e(p)  = mean_channel |Q(p) - T(p)|
L_id  = 0.5 * (mean_{M=1} e + mean_{M=0} e)
L_rgb = mean_{M=1} |C - I|
L     = L_rgb + lambda_mask * w_v * L_id
```

The target identity RGB is premultiplied by the camera alpha before `M` is
computed. This prevents hidden RGB values under transparent RGBA pixels from
turning sky or background into false foreground.
`lambda_mask=1` by default. `w_v` is normalized over the positive views for one
region. Completed manual frames use weight `8` by default; propagated views use
weight `1`. Full-frame SSIM is disabled for this masked run because its
background pixels would again dominate small regions. RGB supervision is
foreground-normalized but not anchor-weighted.

The initial composed scene is a direct concatenation of trained region
Gaussians. It provides a clean baseline for measuring region gaps, overlap,
and leakage before introducing a joint boundary or collision optimizer.

This uses the standard point-initialized 3DGS model from
[3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
with the released masked instance trainer from
[Split&Splat](https://arxiv.org/abs/2602.03809). Split&Splat remains a separate
baseline because its Split stage trains and labels a global 3DGS, while this
pipeline carries identity on MapAnything points.

## Shared ObjectGS Stage

The parallel shared-scene branch adapts the released
[ObjectGS](https://ruijiezhu94.github.io/ObjectGS_page/) implementation. It
keeps one Scaffold-GS-style scene instead of training and concatenating one
model per region:

```text
MapAnything RGB points + persistent IDs
        -> voxel anchors with fixed one-hot region IDs
        -> shared RGB/geometry optimization
        -> ID-preserving anchor growth and pruning
        -> joint RGB scene + rendered persistent-region maps
```

ObjectGS renders region logits through the same alpha-composited visibility
model as RGB. For a known target label `L(p)` and rendered categorical
distribution `q(p)`, its semantic term is:

```text
L_sem = -sum_{p : L(p) != 0} log q_{L(p)}(p)
L = L_rgb + lambda_dssim L_dssim + lambda_volume L_volume
    + lambda_sem L_sem
```

Label `0` is unknown and ignored by the semantic cross-entropy, while those
pixels still supervise shared RGB reconstruction. The adapter maps exact
point-vote labels into ObjectGS anchors. Labels created only by MapAnything
geometry completion are conservatively reset to unknown by default so local
kNN completion is not promoted to hard semantic truth. Growth inherits the
parent anchor region ID, and pruning removes anchors without changing surviving
IDs.

The faithful baseline uses the released 3D-scene defaults: 30,000 iterations,
semantic weight `0.1`, 10 offsets per anchor, and voxel size `0.001`. It does
not add custom manual-frame weighting inside ObjectGS; completed manual masks
remain exact in the prepared indexed masks, and their stronger influence has
already been used during the MapAnything Split association. This keeps the
first shared-scene test attributable to the published method.

Outputs live under:

```text
04_shared_objectgs/runs/<objectgs-run-id>/
  01_dataset/   # posed RGB, indexed masks, labeled initializer
  02_config/    # generated released-ObjectGS configuration
  03_model/     # native neural-anchor model
  04_outputs/   # rendered ID maps, overlays, and anchor debug points
  05_meshes/    # bounded-TSDF, cleaned per-object PLY and GLB geometry
```

The native ObjectGS model is a neural-anchor representation rather than a
standard explicit 3DGS PLY. In Step 1, `Shared ObjectGS` therefore exposes the
official RGB render at each calibrated capture view, the predicted persistent
region-ID maps, and region-colored final anchors with per-object visibility
controls. ObjectGS's separate official geometry path filters the learned
anchors by region, renders object-only RGB-D across the training cameras, and
integrates each object with bounded TSDF. The editor exposes these meshes as
independent parts with RGB/region-color modes. This is extracted geometry, not
an exact conversion of the camera-dependent neural splats; a dedicated
Scaffold-GS renderer is still required for free-camera neural RGB rendering.

These three artifacts have different meanings:

- `Final Neural RGB (Views)` is the final ObjectGS scene rendered by the native
  neural-anchor decoder at a calibrated dataset camera.
- `Rendered Region IDs` is produced by alpha-compositing the fixed anchor IDs
  through that same geometry. It is an optimized model output supervised by,
  but not copied from, the input propagated masks.
- `Neural Anchor Scaffold` contains learned anchor centers and persistent IDs.
  It starts from the labeled MapAnything initializer, then changes through
  ObjectGS anchor growth and pruning. Each anchor generates up to ten explicit
  Gaussians through shared color, opacity, and covariance MLPs.
- `Official Object Meshes` contains one bounded-TSDF mesh per persistent ID.
  Vertex colors come from the object-only ObjectGS renders; extraction does not
  retrain the scene or change anchor identities. The metric dataset uses a
  consistent `1 cm` TSDF voxel; RGB-D is integrated online to bound memory, and
  a post-cleanup triangle cap keeps the geometry practical to inspect.

The completed model is the anchor PLY together with `color_mlp.pt`,
`opacity_mlp.pt`, and `cov_mlp.pt` under
`03_model/<run>/point_cloud/iteration_30000/`. Object partitioning is exact in
the native renderer: it filters the anchor `label_ids` before decoding. The
released model uses `view_dim: 3`, however, so decoded color, opacity, scale,
rotation, and offset selection depend on the camera. Baking those values into
ordinary per-object 3DGS PLYs would be a reference-view approximation, not an
exact conversion of the final ObjectGS model.

## Failure Analysis And Corrected Protocol

The first completed run exposed two distinct appearance failures. The anchored
run inherited a large population of near-white, very-low-opacity Gaussians from
the shared global model; isolated optimization could make that latent scaffold
visible. The MapAnything run learned visible white Gaussians mainly in sky and
context regions even though their RGB initializer points were not white. This
is the masked color/opacity ambiguity: a weak alpha term can pair low alpha with
overbright color while preserving the rendered intensity.

The corrected custom protocol therefore uses alpha-sanitized identity targets,
foreground-normalized RGB, a foreground/background-balanced identity loss,
`lambda_mask=1`, and no full-frame
SSIM. It also leaves half of training after densification and enforces a real
per-region Gaussian cap. A final opacity and multi-view mask-support pass removes
unsupported remnants before composition. Per-object `floater_pruning.json`
reports the exact effect, while `appearance_diagnostics.json` records bright/white
fractions, opacity, and scale statistics for the composed scene and every
region. Viewer PLY files contain only standard 3DGS rendering attributes; the
canonical PLY files retain all descriptor fields.

Completing all MapAnything point labels removes the initializer holes formerly
created by unresolved vote ties. These fixes still do not make direct
concatenation seam-free. Each region owns an independent geometry and opacity field, so uncovered mask boundaries or
small point initializers can become holes, while overlapping boundaries can
become seams. The direct composition remains the controlled ablation. The next
quality stage should optimize one shared scene with region identity or
probability attributes, then export objects from that common geometry. This is
closer to [Gaussian Grouping](https://arxiv.org/abs/2312.00732) than to fully
independent object training. Soft boundary ownership, as used by probabilistic
object-level Gaussian reconstruction, is also a better model for uncertain mask
edges than forcing every boundary pixel to a hard binary owner. Sky and distant
context should normally remain a shared background layer unless they have
reliable finite geometry.

## Editor Outputs

Step 1 contains a separate `MapAnything Region 3DGS` section. It polls
command-line runs without controlling them and shows:

- the five-stage timeline and overall progress;
- the current region, completed-region count, and training iteration;
- identity-cleaned masks and projected point support after Split;
- segmented MapAnything points after Split;
- each completed region 3DGS while training is still running;
- composed RGB and persistent-region-color 3DGS after Compose.

## Scope

This first baseline intentionally uses MapAnything alone. Small or thin
regions with too few labeled initializer points are reported in
`splat_prepare/stage.json` and skipped explicitly. A later controlled
experiment can add COLMAP points without changing the mask or reconstruction
contracts.
