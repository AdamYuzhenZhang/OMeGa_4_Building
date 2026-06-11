# Architecture-Aware OMeGa Extension Ideas

Goal: extend OMeGa so architecture is reconstructed as clean structure, not only
as visually plausible dense mesh-splats.  Large walls/floors/facades should
become simple planar surfaces; corners and seams should stay sharp; detailed or
uncertain regions can keep finer geometry or splat-dominant appearance.

## 1. Depth Regularization

OMeGa already uses rendered RGB, mesh smoothness, mesh normal consistency, and
monocular normal supervision.  The missing geometric cue is positional: the mesh
can have plausible normals while floating in front of or behind the observed
surface.

Add a robust rendered-depth loss:

```text
L_depth = mean_p w_depth(p) * rho(D_render(p) - D_prior(p))
```

Where:

- `D_render` is OMeGa's rendered scene depth.
- `D_prior` comes from PromptDA, MASt3R, raw LiDAR, or another depth source.
- `rho` should be robust, such as Huber or Charbonnier, not plain L2.
- Gradients should update mesh vertices through OMeGa's mesh-controlled splats.

Depth confidence should be spatially weighted:

- higher in locally planar regions;
- lower near ridges, corners, and complex/detail regions;
- zero in unknown, sky, invalid, or far unreliable regions;
- optionally lower with distance from the camera;
- optionally higher where PromptDA agrees with high-confidence raw LiDAR pixels.

This makes monocular depth a soft guide, not a hard truth.  Per-view depth maps
may be inconsistent, but repeated multi-view optimization can resolve some
internal disagreement if unreliable pixels are downweighted.

## 2. Coplane Regularization

PlanarGS shows the useful pattern:

```text
region -> fit plane -> penalize deviation from fitted plane
```

We do not want to depend on LP3 plane masks.  Instead, start from local geometry
evidence:

- Guide01 planar/detail/ridge/unknown categories;
- rendered or mesh face normals;
- face centers and plane offsets;
- neighborhood connectivity;
- thresholds for normal agreement and point-to-plane distance.

For faces with high planar confidence, group nearby faces that are actually
coplanar:

```text
same group if normals agree and face centers lie near the same plane
```

Then fit a plane per group and penalize vertex or face-center distance:

```text
L_coplane = mean_faces p_planar(face) * rho(n_group dot x_face + d_group)
```

Use ridges/boundaries to prevent merging across corners.  Faces near seams can
keep lower coplanarity weight and higher subdivision priority.

Plane groups should be refreshed during optimization, not fixed forever:

- regroup every few hundred iterations;
- allow small groups to join larger planes when normals/offsets become aligned;
- split or drop groups when residuals become too high;
- treat fitted plane parameters as stop-gradient structural targets.

## 3. Why This Is Different From Baseline OMeGa

Baseline OMeGa smooths and subdivides the mesh, but it does not explicitly know:

- which regions should become large planar architectural surfaces;
- which edges are structural seams;
- where depth should pull geometry and where it should be ignored;
- where subdivision should be suppressed or encouraged.

The extension should add reliability-aware geometry priors and region-aware
topology behavior:

```text
planar interiors: depth + coplane + low subdivision
ridges/corners: preserve discontinuity + encourage seam-aligned subdivision
details/plants/clutter: weak coplane + allow local detail or splat appearance
unknown/far/sky: ignore geometric priors
```

## 4. Implementation Milestones

1. Use Guide01 outputs as soft per-view evidence and visualize them.
2. Add robust depth regularization with confidence masks.
3. Project or aggregate guide evidence to mesh faces.
4. Add local coplane regularization between neighboring planar faces.
5. Add dynamic plane grouping and group-level fitted-plane loss.
6. Use planar/detail/ridge confidence to control subdivision and removal.

Current implementation status:

- Milestone 1: Guide01 and Vis25 live in `scan_processing`.
- Milestone 2: `depth_regularized_6500.json` enables the first depth loss:

```text
L_depth = lambda_d * mean_p w_depth(p) * rho(D_render(p) - D_prior(p))
```

The implementation is in `omega_local/losses/depth_regularization.py` and is
loaded only by extension configs.

- Milestone 5 starter: `geometry_regularized_6500.json` adds dynamic coplane
  regularization:

```text
L_coplane = lambda_p * mean_f rho(n_g dot x_f + d_g)
          + lambda_n * mean_f (1 - |normal_f dot n_g|)
```

The grouping step is non-differentiable and refreshed during training.  It
accepts only faces that already agree in normal and plane offset, caches one
fitted plane per group, then lets backprop move vertices softly toward those
planes until the next refresh.  The current version also has depth-assisted
grouping: high-confidence locally planar depth pixels fit broader plane
hypotheses, which can merge fragmented mesh buckets when the current mesh lies
near those planes with compatible normals.  The implementation is in
`omega_local/losses/coplane_regularization.py`.

- Debugging: depth and coplane extension configs enable fixed-frame
  regularization previews in `omega_local/viz/regularization_preview.py`.  The
  contact sheet shows prior depth, rendered depth, weighted depth residual,
  Guide01 categories, coplane groups, and coplane plane residuals at scheduled
  iterations.
