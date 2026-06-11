# View-Informed Adaptive Remeshing for Reconstructed Meshes

Status: historical brainstorming note. The current implementation plan and
commands are maintained in `REMESH_REDESIGN.md` and
`RUN_OMEGA_LOCAL_REMESH.md`. Use this file only for background ideas, not as
the active run sequence.

## 1. Task

Start from a messy reconstructed triangle mesh from OMeGa. The mesh topology is often unstructured: triangles form dense clusters, edges do not follow geometric structure, and planar regions are not cleanly simplified.

The goal is to revise the mesh so that:

- planar regions are simplified aggressively;
- complex regions preserve more geometry;
- edges in high-variation regions better follow geometric flow;
- structural boundaries are extracted in the new mesh;
- different regions receive different remeshing behavior.

The key difference from standard remeshing is that we do not rely only on the mesh. We use per-view normal maps, such as StableNormal or the PromptDA-derived normal maps used for OMeGa training, as external geometric evidence. RGB can be tested later, but it is not part of the current remesh policy because texture edges can be misleading.

---

## 2. Main Idea

Generic remeshing usually applies one global rule:

$$
M_0 \rightarrow M'
$$

with a global target edge length, target face count, or QEM threshold.

Our idea is to make remeshing view-informed and region-adaptive:

$$
\left(M_0, {N_k, P_k}_{k=1}^{K}\right) \rightarrow M'
$$

where:

- $M_0$: input reconstructed mesh;
- $N_k$: StableNormal / predicted normal map for view $k$;
- $P_k$: camera projection.

Instead of a uniform remeshing setting, we derive local remeshing behavior from multi-view evidence:

$$
\text{local policy}
=
f(
\text{planarity},
\text{normal variation},
\text{edge/detail likelihood},
\text{view support},
\text{target edge length}
)
$$

The remesher therefore becomes spatially adaptive:

### Flat / Planar / Consistent Region

- Simplify heavily.
- Use a large target edge length.
- Project vertices to a fitted plane.

### High-Normal-Variation / Detailed Region

- Preserve density.
- Use a smaller target edge length.
- Align edges to geometric flow.

### Structural Boundary

- Protect from collapse.
- Snap to a feature curve.

### Uncertain / Noisy Region

- Apply conservative simplification.
- Optionally defer to later mesh repair.

---

## 3. Related Work and Useful Concepts

### A. QEM / Edge-Collapse Simplification

Garland–Heckbert QEM provides a stable baseline for mesh simplification. Each face contributes a plane quadric, and collapsing an edge is evaluated by the error of the new vertex against the accumulated local planes.

For a plane:

$$
p = [a,b,c,d]^T
$$

the quadric is:

$$
Q = pp^T
$$

and the collapse cost at a candidate vertex $v$ is:

$$
C{\text{QEM}}(v) = v^T Q v
$$

This gives a robust way to simplify geometry, but ordinary QEM treats the mesh mostly as a local surface approximation. It does not know that one region is a wall, another is an ornament, and another is a structural seam.

---

### B. Structure-Aware Mesh Decimation

Structure-Aware Mesh Decimation uses planar proxies and their adjacency graph to guide greedy edge-collapse decimation. It aims to preserve planar structure while enabling strong simplification in planar parts.

The method is especially relevant to planar abstraction, where large planar regions should be simplified aggressively while preserving their structural arrangement.

Its core idea is:

text input mesh + planar proxies → edge collapse guided by local geometry and proxy structure → simplified mesh preserving planar organization 

Our relation:

We may borrow the planar-proxy-aware simplification logic, but derive planar evidence from mesh geometry plus multi-view StableNormal / PromptDA-normal cues.

---

### C. CGAL Adaptive Remeshing / Sizing Fields

CGAL’s isotropic remeshing supports sizing fields, where the target edge length can vary per edge. The PMPSizingField concept defines a target length for every individual edge, and CGAL’s remeshing can split long edges and collapse short edges according to this local target.

Its core idea is:

$$
L(e) = \text{target edge length for edge } e
$$

Uniform remeshing uses:

$$
L(e) = L_0
$$

Adaptive remeshing uses:

$$
L(e) = L_{\text{adaptive}}(e)
$$

Our relation:

Use this concept, but define $L(e)$ from projected normal evidence rather than only from mesh curvature.

---

### D. Normal-Map-Driven Adaptive Meshing

This is one of the most relevant recent directions. Heep et al.’s normal-integration work replaces a dense pixel grid with a sparse triangle mesh, adapting mesh density to local geometry:

- flat regions get fewer triangles;
- complex normal-variation regions get more triangles.

Their CVPR 2025 work further abandons isotropy in favor of aligning vertices and edges to ridges and furrows of the underlying geometry.

The important insight is that normal maps contain geometric structure. Curvature or complexity can be estimated from normal variation, and this can drive adaptive triangulation.

A simplified version is:

$$
D(x) = |\nabla N(x)|
$$

where $N(x)$ is the normal map.

Then:

text low D(x): flat / featureless → larger triangles high D(x): detailed / curved / ridge → smaller triangles 

Our relation:

We transfer this idea from screen-space normal integration to 3D reconstructed mesh remeshing. Instead of adapting a 2D image triangulation, we project normal-map evidence onto a 3D mesh and use it to drive local remeshing.

---

### E. Direction Fields and Flow-Aligned Remeshing

Directional-field processing is a major tool for aligning meshes to surface flow. Libraries like Directional provide tools for editing, analyzing, and visualizing directional fields on triangle meshes. Field-generation tools such as fieldgen compute smooth tangent direction fields on triangle meshes.

The core idea is:

$$
u(x) \in T_xM
$$

where $u(x)$ is a preferred tangent direction on the surface.

Edges can then be encouraged to align with this field:

$$
C_{\text{align}}(e)
=
1 - |\hat{e}\cdot u(e)|
$$

where:

$$
\hat{e}
=
\frac{v_j-v_i}{|v_j-v_i|}
$$

Our relation:

Use view-derived normal variation to estimate local flow directions, then encourage edges in detailed regions to align with those directions through edge flips or anisotropic remeshing costs.

---

## 4. Proposed Pipeline

## Step 1: Project Normal Evidence Onto the Mesh

For each face $f$ or vertex $v$, collect all visible views:

$$
\mathcal{V}(f)
=
{k \mid f \text{ is visible in view } k}
$$

Project the face centroid or sampled points into each view:

$$
x_k = P_k(X_f)
$$

The current implementation does not average world-space predicted normals as
the main score, because predicted normals from different views can disagree in
global orientation. Instead, each view first computes local screen-space normal
geometry:

- local normal variation;
- normal derivative magnitude / curvature proxy;
- curvature-derived target edge length in pixels;
- approximate target edge length in world units;
- view support.

The most important borrowed idea from Heep et al. is to convert normal-derived
curvature into a target edge length. Low curvature means a larger local density
length; high curvature means a smaller local density length.

---

## Step 2: Estimate Continuous Policy Weights

Instead of hard semantic labels, compute continuous scores.

### Planar Weight

A face is likely planar if normal-derived curvature and variation are low and
view support is sufficient:

```text
curvature_score = smoothstep(max_normal_curvature, q_low, q_high)
variation_score = smoothstep(max_normal_variation, q_low, q_high)
detail_seed = max(curvature_score, w_var * variation_score)
planar_weight = (1 - detail_seed) * (1 - w_penalty * variation_score) * support
```

A high `planar_weight` means this area can simplify, but only if it is actually
oversampled relative to the target length.

### Edge / Detail Weight

A face or shared edge should be protected if normal curvature, normal variation,
mesh crease evidence, or planar-weight contrast is high:

```text
edge_weight = max(edge curvature/variation, mesh crease, planar contrast)
detail_weight = max(detail_seed, mesh_crease_weight * mesh_crease, 0.65 * edge_weight)
```

This replaces the old boundary classification. Boundary is not treated as a hard
face type; edge/detail weights are continuous protection signals.

### Target Length and Density Pressure

The Stage-1 target length is interpreted as a density/collapse field, not an
isotropic triangle-quality requirement:

```text
target_world = clamp(Stage1_min_target_world, L_min, L_max)
target_world *= (1 + planar_boost * planar_weight)
target_world *= (1 - detail_shrink * max(detail_weight, edge_weight))

oversample = smoothstep(target_world / current_edge_length, 1, 3)
undersample = smoothstep(current_edge_length / target_world, 1, 3)
```

Then:

```text
simplify_weight = planar_weight * oversample * (1 - edge_weight)
protect_weight = max(edge_weight, detail_weight, uncertainty, undersample)
```

This keeps large planar surfaces easy to simplify while protecting details and
areas that are already too coarse.

---

## Step 3: Consume the Local Sizing Field During Remeshing

Stage 2 already computes the face-level target length. A remesher can lift that
to edges or vertices depending on its backend:

```text
L_f = face_target_edge_length_world
L_e = aggregate(L_fi, L_fj) for adjacent faces around edge e
```

For simplification, `L_e` is a density/collapse rule:

- if a supported planar interior has current edge length much smaller than
  `L_e`, it is a good collapse candidate;
- if a detail/edge area has current edge length larger than `L_e`, it should be
  protected or subdivided by a future backend;
- if a planar wall permits large `L_e`, the resulting triangles may be long or
  anisotropic as long as they cover the surface and preserve structural seams.

This is intentionally different from isotropic remeshing. We care more about
architectural coverage and seam preservation than about equal-size triangles on
large flat surfaces.

---

## Step 4: Region-Conditioned Remeshing / Simplification

Use a weighted edge-collapse or remeshing cost.

For a candidate edge collapse:

$$
e=(v_i,v_j)\rightarrow v'
$$

define:

$$
C(e,v')
=
C_{\text{QEM}}(v')
+
\lambda_p C_{\text{plane}}(v')
+
\lambda_b C_{\text{boundary}}(e)
+
\lambda_n C_{\text{normal}}(e)
+
\lambda_q C_{\text{quality}}(e)
-
\lambda_s B_{\text{simplify}}(e)
$$

### QEM Term

$$
C_{\text{QEM}}(v')
=
{v'}^T(Q_i+Q_j)v'
$$

### Plane Term

For planar region $R$:

$$
C_{\text{plane}}(v')
=
(n_R^T v' + d_R)^2
$$

This encourages collapsed vertices to remain on the fitted plane.

### Boundary Term

If an edge crosses a detected structural boundary:

$$
C_{\text{boundary}}(e) \rightarrow \infty
$$

or the edge is marked as protected.

If the edge lies along a boundary, allow only boundary-preserving collapse.

### Normal Preservation Term

$$
C_{\text{normal}}(e)
=
1 - \bar{N}{\text{before}}\cdot \bar{N}{\text{after}}
$$

Alternatively, penalize merging regions with inconsistent StableNormal distributions.

### Simplification Bonus

For confident planar interiors:

$$
B_{\text{simplify}}(e)
=
P_{\text{plane}}(e)(1-P_{\text{boundary}}(e))
$$

This means planar interiors are easier to collapse.

---

## Step 5: Flow Direction Estimation in Detailed Regions

For high-normal-variation regions, estimate a local tangent direction field.

One simple formulation uses the projected normal field. For a local neighborhood around face $f$, compute normal variation covariance:

$$
C_f
=
\sum_{g\in \mathcal{N}(f)}
w_{fg}
(\bar{N}g-\bar{N}f)
(\bar{N}g-\bar{N}f)^T
$$

This identifies how normals vary locally. The strongest normal change often indicates the direction across a ridge or furrow, while the weakest change in the tangent plane often corresponds to the direction along the feature flow.

Let:

$$
u_f \in T_fM
$$

be the preferred tangent flow direction.

Then define edge alignment cost:

$$
C{\text{align}}(e)
=
1 - |\hat{e}\cdot u_f|
$$

Edges that align with the local flow direction have lower cost.

---

## Step 6: Direction-Aware Topology Update

Before implementing full anisotropic remeshing, a lightweight step is edge flipping.

For two adjacent triangles sharing diagonal $e$, compare the current diagonal $e$ with the flipped diagonal $e'$.

Flip if:

$$
C{\text{align}}(e') < C{\text{align}}(e)
$$

and triangle quality remains valid:

$$
Q(e') > Q{\min}
$$

and the flip does not cross protected boundaries.

This starts to revise topology without requiring a full field-aligned remeshing system.

---

## 5. Overall Method in One Formula

The overall local remeshing objective can be written as:

$$
E(M')
=
E_{\text{fit}}
+
\lambda_p E_{\text{plane}}
+
\lambda_d E_{\text{detail}}
+
\lambda_b E_{\text{boundary}}
+
\lambda_a E_{\text{align}}
+
\lambda_q E_{\text{quality}}
+
\lambda_c E_{\text{complexity}}
$$

where:

- $E_{\text{fit}}$: stay close to the original reconstructed mesh;
- $E_{\text{plane}}$: planar regions should become clean planes;
- $E_{\text{detail}}$: high-normal-variation areas should preserve detail;
- $E_{\text{boundary}}$: structural boundaries should not be destroyed;
- $E_{\text{align}}$: edges should align with local flow directions;
- $E_{\text{quality}}$: avoid degenerate triangles;
- $E_{\text{complexity}}$: reduce unnecessary vertices and faces.

The key is that the weights are not constant:

$$
\lambda(x)
=
f(
P_{\text{plane}}(x),
P_{\text{detail}}(x),
P_{\text{boundary}}(x)
)
$$

So the mesh is simplified and reorganized differently in different regions.

---

## 6. Mesh Refinement

OMeGa densifies the mesh and removes triangles, but it can still end up with problematic topology, including:

- holes;
- overlapping faces;
- intersecting faces;
- flattened blobs on surfaces;
- noisy or unstable local geometry.

A more advanced method is needed to refine the mesh and clean up the shape. We will explore this after establishing a reliable remeshing pipeline.

---


## 7. OMeGa Implementation Plan

The implementation should be staged. The first target is a post-hoc remesh of the
final optimized OMeGa mesh. If the result is stable and visually better, the same
logic can be moved into the training loop as a topology update event.

The current PyMeshLab planar QEM baseline remains useful as a reference, but it
should not constrain the final method. If PyMeshLab, Open3D, or another library
does not expose the operation we need, we should reimplement that part locally.
The priority is a clean architectural reconstruction, not staying inside one
existing remeshing API.

### Stage 0: Keep the Existing Baseline

Keep the current post-hoc method as the baseline:

- load the final optimized OMeGa mesh;
- clean duplicate, degenerate, and unreferenced elements;
- run planar-aware QEM simplification;
- use vertex quality to protect boundaries, creases, and non-manifold regions;
- write summary statistics and visualization panels.

This gives a controlled comparison point for every later method.

### Stage 1: Project Normal Evidence and Sizing Data Onto Mesh Faces

Stage 1 prepares only the data we currently trust and use. It reads normal maps
from the OMeGa prepared dataset, so the evidence follows the selected
PromptDA/StableNormal path used for training. RGB gradients and Guide01
planarity/detail/ridge probabilities are intentionally removed from this path
because they were not reliable enough for geometric remeshing decisions.

For each selected view and visible face centroid:

1. Decode the camera-frame normal map.
2. Compute local normal variation from a small patch.
3. Blur normals, compute screen-space normal derivatives, and use the derivative
   magnitude as a curvature-like signal.
4. Convert the curvature-like signal into a target edge length following the
   useful idea from Heep et al.'s normal-driven meshing:

```text
L_px = sqrt(6 * epsilon / max(kappa, eps) - epsilon^2)
L_px = clamp(L_px, L_min_px, L_max_px)
```

Here `kappa` is the screen-space normal-derivative magnitude, and `epsilon` is a
user-controlled approximation tolerance. This target length is only the
screen-space source cue. When projected to the 3D mesh we store two world-space
versions:

```text
L_world_conservative = L_px * depth / focal
L_world_surface = L_world_conservative * clamp(1 / abs(n_mesh dot view), 1, max_oblique)
```

The conservative version is used for detail/edge protection. The obliqueness-aware
surface version is allowed to influence only confident planar faces. This avoids
treating a grazing-view pixel footprint as a reliable isotropic length on a
curved/detail region, while still letting clean planar surfaces accept longer,
possibly skewed triangles.

For each face, aggregate:

- `view_count` and `mean_valid_support`;
- `normal_count`;
- `mean_normal_variation` and `max_normal_variation`;
- `mean_normal_gradient` and `max_normal_gradient`;
- `mean_normal_curvature` and `max_normal_curvature`;
- `mean_target_edge_length_px` and `min_target_edge_length_px`;
- `mean_target_edge_length_world_conservative` and `min_target_edge_length_world_conservative`;
- `mean_target_edge_length_world_surface` and `min_target_edge_length_world_surface`;
- `mean_fronto_pixel_world_scale`, `mean_surface_pixel_world_scale`,
  `mean_abs_view_cos`, and `mean_oblique_scale_multiplier`.

`max_*` and `min_target_*` keep the strictest useful view so a seam or detail is
not erased just because other views see it weakly.

Output:

- `$RESULT_DIR/remesh/evidence/mesh_face_evidence.npz`;
- `$RESULT_DIR/remesh/evidence/mesh_face_evidence_summary.json`;
- debug meshes for support, view angle/obliqueness, variation, gradient,
  curvature, and conservative/surface target lengths;
- Vis28 camera-view panels for the same evidence.

This stage does not change the mesh. It verifies that the normal-derived sizing
field looks trustworthy.

Implementation status:

- implemented as `omega_local/remesh/evidence.py`;
- CLI wrapper: `scripts/run_omega_mesh_evidence.py`;
- visualization: `scan_processing/Vis28_visualize_phone_omega_mesh_evidence.py`;
- optional Guide01 depth gating remains available only as a visibility filter
  when `--guide-depth-tolerance-m` is positive; Guide01 scores are not written.

### Stage 2: Convert Evidence Into Minimal Remesh Policy Weights

Stage 2 converts Stage-1 normal evidence into the two behaviors we currently
need. It does not change topology or vertex positions.

The policy is intentionally small:

- **plane-like region**:
  - low robust normal curvature;
  - low robust normal variation;
  - enough projected normal-map support;
  - simplify aggressively;
  - allow large or oblique triangles;
  - mark vertices for fitted-plane projection in Stage 3.
- **detail / non-planar region**:
  - high normal curvature, high normal variation, mesh crease, or transition
    evidence;
  - simplify conservatively;
  - preserve more triangles.

Primary outputs:

- `face_normal_curvature_score`: normalized robust curvature cue from projected
  normal maps;
- `face_variation_score`: normalized robust local normal variation cue;
- `face_gradient_score`: auxiliary normal-gradient cue for visualization;
- `face_planar_weight`: high for supported low-curvature/low-variation faces;
- `face_detail_weight`: high for non-planar/detail faces;
- `face_edge_weight`: transition/crease cue used mainly as a guard;
- `face_plane_project_weight`: planar cleanup mask consumed by Stage 3;
- `face_simplify_weight`: currently equal to the plane projection/simplify mask;
- `face_protect_weight`: high for detail, transitions, uncertainty, and creases.

Current mapping:

```text
curvature_score = smoothstep(q75_normal_curvature, q25, q90)
variation_score = smoothstep(q75_normal_variation, q50, q95)
gradient_score = smoothstep(mean_normal_gradient, q25, q90)

support = clamp(0.85 * normal_sample_confidence + 0.15 * view_support_confidence)
mesh_crease = smoothstep(mesh_dihedral_angle, low_deg, high_deg)

detail_signal = max(
    curvature_score,
    variation_detail_weight * variation_score,
    0.50 * gradient_score,
    mesh_crease_detail_weight * mesh_crease,
)

detail_weight = detail_signal * support
planar_weight = support
              * (1 - curvature_score)
              * (1 - planarity_variation_penalty * variation_score)
              * (1 - 0.50 * mesh_crease)
              * (1 - 0.50 * detail_weight)

edge_weight = max(mesh_crease, adjacent_detail, adjacent_planarity_contrast)
uncertainty = max(1 - support, no_normal_evidence, no_view_evidence)

plane_project_weight = planar_weight
                     * (1 - detail_weight)
                     * (1 - 0.70 * edge_weight)
                     * (1 - 0.75 * uncertainty)

simplify_weight = plane_project_weight
protect_weight = max(detail_weight, 0.70 * edge_weight, 0.75 * uncertainty)
```

Target edge length and density gates are deliberately not used in the current
policy. The Heep-style sizing fields can remain in Stage 1 as debug/evidence,
but broad architectural planes should not need an isotropic target edge length:
a large wall can be represented by two large triangles.

Region IDs and dominant region types are debug aids only. Downstream remesh code
should consume the continuous face weights above.

Output:

- `$RESULT_DIR/remesh/evidence/mesh_region_scores.npz`;
- `$RESULT_DIR/remesh/evidence/mesh_regions.json`;
- debug meshes for curvature score, variation score, plane-like/detail weights,
  plane projection weight, simplify weight, and protect weight;
- Vis29 camera-view panels for the same policy fields.

Implementation status:

- implemented as `omega_local/remesh/regions.py`;
- CLI wrapper: `scripts/run_omega_mesh_region_scores.py`;
- visualization: `scan_processing/Vis29_visualize_phone_omega_region_scores.py`;
- RGB and Guide01 scores are ignored;
- target edge-length fields are not consumed by Stage 2.

### Step 3b: Plane Adjustment Before QEM

Step 3b is a dedicated planar surface adjustment pass. It consumes the Stage-2
plane/detail scores and changes only vertex positions. It does not simplify,
collapse edges, retriangulate, or use view-normal directions directly.

The reason for this separate pass is that view normals can disagree across
cameras. For architectural surfaces, the key is to find the plane through the
actual reconstructed surface component and move safe vertices cleanly onto that
plane.

Policy:

- use only two region behaviors:
  - **plane-like**: eligible for plane fitting and vertex projection;
  - **detail / non-planar**: protected from projection;
- treat boundary/transition evidence as part of protection rather than as a
  separate semantic class;
- fit planes from mesh geometry: component vertices plus face centers, weighted
  by `face_plane_project_weight` and face area;
- robustly trim high-residual samples before the final plane fit;
- project only vertices whose incident faces mostly belong to the same
  plane-like component and whose detail/protect scores are low;
- optionally clamp maximum vertex motion to avoid bad planes causing large
  jumps.

Current plane fitting:

```text
selected_face = face_plane_project_weight >= plane_threshold
             && face_detail_weight <= detail_max
             && face_protect_weight <= protect_max

plane_components = connected_components(selected_faces)

for component in plane_components:
    if face_count < min_component_faces: skip
    samples = component triangle vertices + component face centers
    weights = face_area * face_plane_project_weight
    plane = robust_weighted_PCA(samples, weights, trim_quantile, iterations)

    eligible_vertex = selected_incident_fraction >= vertex_plane_fraction
                   && vertex_plane_mean >= plane_threshold
                   && vertex_detail_max <= detail_max
                   && vertex_protect_max <= protect_max

    v' = v - strength * clamp(dot(v - plane_centroid, plane_normal), max_move) * plane_normal
```

Output:

- `$RESULT_DIR/remesh/mesh_plane_adjusted.ply`;
- `$RESULT_DIR/remesh/mesh_plane_adjusted.plane_adjustment.npz`;
- `$RESULT_DIR/remesh/mesh_plane_adjusted.plane_adjustment_summary.json`;
- colored debug meshes under `$RESULT_DIR/remesh/debug_meshes/plane_adjustment/`.

Implementation status:

- implemented as `omega_local/remesh/plane_adjustment.py`;
- CLI wrapper: `scripts/run_omega_plane_adjustment.py`;
- intended to run after Stage 2 and before QEM.

### Stage 3: View-Informed QEM After Plane Adjustment

Stage 3 consumes `mesh_region_scores.npz` and simplifies the mesh. If Step 3b
has already adjusted planes, QEM should usually run on `mesh_plane_adjusted.ply`
with its internal `--no-plane-project` flag so vertex projection is not repeated.

Weighted QEM policy:

- low quality collapses more easily on plane-like interiors;
- high quality protects detail, uncertainty, open boundaries, non-manifold
  edges, and strong creases;
- PyMeshLab QEM remains the current backend.

Current vertex quality math:

```text
S_v = mean incident face_simplify_weight
P_v = max incident face_protect_weight
U_v = max incident face_type_uncertain_score
D_v = max incident face_type_detail_score
G_v = mesh feature guard from open boundaries, non-manifold edges, and creases

Q_view = max(P_v, 0.45 * (1 - S_v), 0.60 * U_v, 0.35 * D_v)
Q_raw = max(Q_view, G_v)
Q_vertex = min_quality + (1 - min_quality) * Q_raw ^ quality_gamma
```

`Q_vertex` is passed to PyMeshLab weighted QEM. Lower quality collapses more
easily; higher quality protects geometry. Policy arrays are saved explicitly so
a future local edge-collapse implementation can replace PyMeshLab without
changing the evidence math.

Output:

- `$RESULT_DIR/remesh/mesh_view_informed_qem_<target>.ply`;
- `$RESULT_DIR/remesh/mesh_view_informed_qem_<target>.summary.json`;
- `$RESULT_DIR/remesh/mesh_view_informed_qem_<target>.policy.npz`;
- colored policy debug meshes under
  `$RESULT_DIR/remesh/debug_meshes/view_informed_qem/`;
- Vis30 before/after camera-view comparisons.

Implementation status:

- implemented as `omega_local/remesh/view_informed_qem.py`;
- CLI wrapper: `scripts/run_omega_view_informed_remesh.py`;
- visualization: `scan_processing/Vis30_visualize_phone_omega_view_informed_remesh.py`;
- currently post-hoc only.

### Stage 4: Future Planar Region Cleanup

Stage 3 already performs the first fitted-plane vertex projection. Future planar
cleanup should go beyond that basic snap and handle architectural patch cleanup
more explicitly.

For each high-confidence planar component:

- remove tiny noisy islands;
- preserve component boundary vertices more strongly than interior vertices;
- simplify interiors more aggressively than boundaries;
- optionally retriangulate large planar patches;
- optionally export polygonal planar patches for designer-facing workflows.

This stage may require local implementation. PyMeshLab can fit planes and
simplify meshes, but it does not provide the full architectural planar patch
operation we need.

The first implementation can keep the triangle mesh representation. A later
version can export polygonal planar patches for designer-facing workflows.

### Stage 5: Conservative Hole And Overlay Repair

Add topology repair only after region scores are reliable.

Hole filling policy:

- detect boundary loops;
- estimate the surrounding region type;
- fill only small or medium holes inside confident planar regions;
- do not fill loops with high boundary/ridge score;
- do not fill likely real openings, such as doorways and windows;
- after filling a planar hole, project new vertices/faces to the fitted plane.

Overlay and self-intersection policy:

- detect self-intersecting or overlapping faces;
- remove tiny floating components;
- prefer keeping faces with stronger view support;
- re-run local cleanup after removal.

This stage should be conservative. Incorrectly filling a real opening is worse
than leaving a hole for the first prototype.

### Stage 6: Direction-Aware Detail Cleanup

For high-detail regions, estimate a tangent direction field from mesh normals
and projected view-normal variation.

First implementation:

- compute a per-face preferred tangent direction;
- visualize the field;
- run local edge flips where the flipped diagonal better follows the field and
  triangle quality remains valid.

Potential PyMeshLab filters:

- `meshing_edge_flip_by_planar_optimization`;
- `meshing_edge_flip_by_curvature_optimization`.

If these are too generic, implement our own edge-flip pass:

- enumerate manifold interior edges;
- evaluate current and flipped diagonals;
- reject flips that cross protected boundaries;
- reject flips that create inverted or low-quality triangles;
- accept flips that improve alignment and quality.

Full anisotropic remeshing should come after this lightweight pass.

### Stage 7: Move Into OMeGa Training

Only move the method into training after the post-hoc result is useful.

The training-time version should initially use only safe operations:

- view-informed QEM;
- planar/detail/boundary protection;
- quality guards;
- splat reparenting;
- no aggressive hole filling;
- no creation of large new surfaces.

Topology creation during training is a separate problem because new faces may
not have useful splats attached. Before using hole filling in training, we need
a splat spawning or redistribution strategy for newly created faces.

Training integration should reuse the existing remesh event structure:

- write `mesh_before_remesh.ply`;
- run the post-hoc remesh module;
- run guard checks;
- write `mesh_after_remesh.ply`;
- reparent splats;
- write event previews and summary JSON.

---

## 8. Code Organization

Keep the implementation modular so each piece can be tested independently.

Suggested structure:

```text
omega_local/remesh/
  posthoc.py                         # existing baseline QEM entry point
  training.py                        # existing in-training reparenting path
  evidence.py                        # project normal-derived evidence to mesh
  regions.py                         # convert face evidence into minimal planar/detail policy
  plane_adjustment.py                # Step 3b fitted-plane adjustment before simplification
  view_informed_qem.py               # QEM using policy weights
  planar_cleanup.py                  # future stronger planar hole repair/retriangulation
  topology_repair.py                 # boundary loops, overlays, components
  direction_field.py                 # tangent direction estimation and edge flips
```

Suggested scripts:

```text
scripts/run_omega_mesh_evidence.py
scripts/run_omega_plane_adjustment.py
scripts/run_omega_view_informed_remesh.py
scripts/run_omega_planar_cleanup_remesh.py
```

Each script should have:

- explicit CLI arguments;
- clear printed progress;
- JSON summaries;
- deterministic defaults;
- output paths under `$RESULT_DIR/remesh/`;
- no hidden mutation of training checkpoints.

---

## 9. Documentation Standard

Every new remesh module should document:

- what inputs it expects;
- what coordinate frame it uses;
- what each score means;
- which operations change topology;
- which operations only move vertices;
- which guards can skip the operation;
- what files are written.

Every summary JSON should include:

- input mesh path;
- output mesh path;
- face and vertex counts before and after;
- surface area before and after;
- edge length quantiles before and after;
- number of boundary loops;
- number of repaired or filled holes;
- view-evidence coverage statistics;
- region score statistics;
- guard failures, if any.

The implementation should also write visual diagnostics at every major stage.
For this project, visual inspection is not optional: the whole goal is a mesh
that follows architectural structure in a way humans can use.

---

## 10. Success Criteria

A remesh variant is worth moving from post-hoc to training only if it improves
the final mesh without damaging geometry:

- planar walls, doors, floors, and trim become cleaner;
- edges better follow surface intersections and structural seams;
- detailed or sculptural areas are not flattened away;
- face density is lower on broad planar areas and higher around details;
- real openings are preserved;
- missing planar patches can be repaired conservatively;
- splat render quality does not visibly collapse;
- Vis26/Vis27 comparisons show better wireframe and normal structure.

The first research target is not the lowest face count. The target is a mesh
whose topology and edges better match the architectural structure.

---
