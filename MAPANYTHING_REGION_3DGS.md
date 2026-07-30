# MapAnything Region 3DGS

## Purpose

This is a custom reconstruction pipeline parallel to the released
Split&Splat experiments. It tests whether the feed-forward geometry already
used to initialize OMeGa can replace the expensive global 3DGS in the Split
stage.

```text
manual persistent-region masks + propagated masks
        -> MapAnything point ownership
        -> identity-cleaned per-frame masks
        -> per-region MapAnything point initializers
        -> independently trained region 3DGS
        -> world-aligned composed scene
```

The pipeline has its own code and output root:

```text
omega_local/reconstruction/pipelines/mapanything_3dgs/

<interactive>/experiments/mapanything_region_3dgs/runs/<run-id>/
  01_input/
  02_split/
  03_splat/
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
