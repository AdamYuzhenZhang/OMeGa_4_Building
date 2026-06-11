# View-Informed Remeshing Progress Report

This report summarizes the remeshing exploration around OMeGa on the DSLR
`grove_entrance_dslr_0521_pinhole` scene. The current result is not yet the
clean structured mesh we want, but the experiments produced a useful research
direction: high-quality per-frame normal maps carry strong evidence about
planarity, detail scale, and local orientation flow, and that evidence can be
used to regulate OMeGa's mesh updates.

The working mesh is:

```text
data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/
  omega_stable_mesh/model_baseline_stronger_30000/plys/mesh_29999_rank0.ply
```

The new remesh artifacts are written under:

```text
data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/
  omega_stable_mesh/model_baseline_stronger_30000/remesh/
```

The implementation and runbook are in:

```text
third_party/OMeGa_4_Building/RUN_OMEGA_LOCAL_REMESH.md
third_party/OMeGa_4_Building/REMESH_REDESIGN.md
```

## 1. Research Goal

OMeGa already produces a visually strong reconstruction by binding 2D splats to
a triangle mesh and optimizing the coupled representation. Its mesh, however,
is still optimized mainly as a support for rendering, not as a designer-facing
structured surface. The current output tends to contain dense irregular
triangle clusters, weak alignment to architectural seams, and insufficient
large planar faces.

The target mesh should have:

- large simple faces on walls, floors, stairs, and other planar regions;
- enough density around sculptural/detail regions;
- edges following corners, ridges, folds, and curvature flow;
- conservative behavior around low-support or unexplained geometry;
- a clear path to run during OMeGa optimization, replacing or augmenting the
  current subdivision event instead of only repairing a final mesh.

The main hypothesis tested here is:

```text
StableNormal per-frame normals are not 3D ground truth, but they are strong
view evidence. If projected carefully onto the OMeGa mesh, they can guide
planar simplification, detail preservation, edge-flow alignment, and adaptive
mesh density.
```

## 2. Related Work Context

### Gaussian Mesh Reconstruction

- [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://arxiv.org/abs/2403.17888)
  uses oriented 2D surfels/Gaussians to improve geometric behavior compared
  with unconstrained 3D Gaussian ellipsoids.
- [Planar Gaussian Splatting](https://arxiv.org/abs/2412.01931) regularizes
  Gaussian splatting with planar assumptions and is relevant for architectural
  scenes where large planar support should be explicit.
- [AGS-Mesh](https://arxiv.org/abs/2411.19271) is a related line of work on
  mesh reconstruction from Gaussian splatting.
- [OMeGa](https://arxiv.org/abs/2509.24308) optimizes meshes and Gaussians
  together. This is the base system we are extending.
- [StableNormal](https://arxiv.org/abs/2406.16864) is the monocular normal
  estimator supplying the high-quality per-frame normal evidence used in our
  pipeline.

These works motivate the representation choice: Gaussian/splat rendering gives
strong visual quality, but a mesh-aware method is needed if the final geometry
must be editable and structured.

### Classic Mesh Simplification And Remeshing

- [Garland and Heckbert, Surface Simplification Using Quadric Error Metrics](https://www.cs.cmu.edu/~garland/Papers/quadrics.pdf)
  is the main QEM reference. It accumulates plane quadrics and chooses edge
  contractions that minimize a local squared distance energy.
- [Botsch and Kobbelt, A Remeshing Approach to Multiresolution Modeling](https://www.graphics.rwth-aachen.de/media/papers/remeshing1.pdf)
  motivates the standard local operator sequence: split long edges, collapse
  short edges, flip edges, tangentially relax vertices, then project back to the
  surface.
- [CGAL Polygon Mesh Processing](https://doc.cgal.org/latest/Polygon_mesh_processing/index.html)
  provides robust mesh repair operations, including hole filling, degeneracy
  repair, border stitching, and self-intersection routines.
- [Anisotropic Polygonal Remeshing](https://www.lix.polytechnique.fr/~maks/papers/Remeshing_SIGGRAPH_2003.pdf)
  shows how direction and anisotropy can make polygons follow surface features
  better than uniform remeshing.
- [Variational Shape Approximation](https://www.cs.jhu.edu/~misha/Fall09/CohenSteiner03.pdf)
  is relevant to grouping surfaces into planar proxies.

These works suggest that the mesh problem should not be framed only as a target
face count problem. It should be framed as local operations under geometry,
feature, and approximation-error constraints.

### Field-Aligned And Density-Controlled Baselines

- [Instant Field-Aligned Meshes](https://rgl.epfl.ch/publications/Jakob2015Instant)
  solves for orientation and position fields, then extracts quads or triangles
  aligned to those fields. This is the strongest baseline for edge-flow
  alignment.
- [Geogram/Vorpalite](https://github.com/BrunoLevy/geogram) provides
  anisotropic remeshing and density/gradation controls through a robust
  geometry processing toolkit.
- [QuadriFlow](https://arxiv.org/abs/1811.08528) is a related quadrangulation
  baseline, though in our tests the more relevant direction was Instant Meshes
  and Geogram because we could patch their constraints/density behavior.
- [Feature Sensitive Remeshing](https://www.graphics.rwth-aachen.de/media/papers/feature2.pdf)
  and [normal-lifting remeshing](https://hal.science/hal-01169978/document) are
  relevant to using position-plus-normal metrics to protect features.

These works are useful baselines, but none directly solve our setting: the mesh
is an optimization result, and the strongest signal about missing detail often
comes from per-view normals rather than the mesh alone.

## 3. Pipeline Built So Far

The current pipeline is:

```text
OMeGa mesh
  -> Phase 1: safety preprocess and optional CGAL healing
  -> Phase 2: render mesh into all OMeGa training views
  -> Phase 2: transfer StableNormal evidence onto mesh faces
  -> Phase 3: compute continuous planar/detail/boundary/uncertainty weights
  -> Phase 4: try weighted local remeshing and planar patch remeshing
  -> Phase 11: compare against external remeshing baselines
```

The important design decision was to avoid hard region classification. Instead
we compute continuous weights:

```text
W_planar     -> collapse more and project to planes
W_detail     -> preserve density, clean triangles, align edge flow
W_boundary   -> stop crossing seams and protect feature curves
W_uncertain  -> make operations conservative
```

In practice, `W_uncertain` was less useful than expected. Many uncertain-looking
areas are actually underfit detail or missing geometry, so uncertainty should
not become a broad freeze mask during training.

## 4. Phase 1: Mesh Preprocess And Healing

The canonical preprocess is intentionally conservative. It removes only exact
bookkeeping/topology issues:

```text
merge exact duplicate vertices
remove invalid/zero-area/duplicate faces
remove isolated vertices
split bow-tie one-ring vertex fans
preserve holes, boundaries, components, and surface geometry
```

Observed result on the OMeGa DSLR mesh:

```text
input vertices:       69,938
input faces:         111,167
output vertices:      61,443
output faces:        111,167
isolated vertices:     8,871 removed
bow-tie vertices:        367 split
faces touched:          1,154
```

This cleaned the mesh representation without changing face count or broad
geometry.

We also added an optional CGAL healing stage:

```text
degenerate-face repair
duplicate non-manifold vertices
exact boundary stitching
small hole filling
diagnostic self-intersection counting
```

With the current default hole settings, the healing stage reported:

```text
boundary edges:       11,739 -> 10,067
boundary cycles:         284 -> 145
filled holes:             139
inserted patch faces:   3,036
self-intersection pairs: 1,521 -> 3,779
```

Interpretation: CGAL hole filling can close small defects and is useful as a
diagnostic, but it can also increase local self-intersection counts. It should
remain optional unless we visually confirm that filled holes are real defects
and not meaningful architectural openings.

## 5. Phase 2: StableNormal Evidence Transfer

For each prepared OMeGa training view `k`, we render:

```text
FID_k(u)     face id at pixel u
Z_k(u)       mesh depth
```

We also read the per-view normal map:

```text
N_k(u) in camera coordinates
```

From the normal map we compute image-space derivatives:

```text
N_u = dN / du
N_v = dN / dv
g(u) = sqrt(||N_u||^2 + ||N_v||^2)
```

The normal structure tensor is:

```text
J(u) =
  [ <N_u,N_u>  <N_u,N_v> ]
  [ <N_v,N_u>  <N_v,N_v> ]
```

The large eigenvalue of `J` indicates strong normal variation. The small
eigenvector gives the image direction along which normals change least, which
is often the tangent direction of a ridge, stair seam, furrow, or sculptural
flow line.

We transfer these signals to mesh faces using the rendered `FID_k(u)` buffers.
We also use an offset-tolerant transfer band because the OMeGa mesh can be
slightly displaced from the StableNormal evidence:

```text
high-gradient normal pixels
  -> nearby rendered mesh faces within an offset search band
```

This is not treated as ground truth geometry. It is evidence that the current
mesh may need detail, preservation, or further OMeGa optimization.

The full evidence run used:

```text
selected frames:     214 / 214 parser frames
frame stride:          1
mesh faces:      111,167
mesh vertices:    61,443
```

The best visual evidence maps were:

- `face_support`: stable view support, strong on planar surfaces;
- `face_unexplained_offset`: highlights projected/missing geometry such as
  railings that are visible in normal maps but not explained by the mesh;
- `face_detail_target_length`: clear local sizing signal;
- `face_direction_confidence`: useful direction/flow evidence on walls, stairs,
  and ridges.

The more problematic evidence maps were:

- broad face-level boundary maps, because a single boundary edge can color a
  large triangle;
- broad uncertainty maps, because underfit detail can look uncertain even when
  it should become refinement pressure.

## 6. Phase 3: Continuous Policy Weights

Phase 3 converts evidence into continuous weights and planar proxies. The key
idea is:

```text
Do not classify faces into hard regions.
Compute local weights and let each operation read those weights.
```

Simplified weight logic:

```text
W_planar =
  support * planar_score * low_detail_gate * certainty_gate

W_detail =
  max(stable_normal_detail,
      offset_detail_pressure,
      feature_detail_pressure)

W_boundary =
  edge-centric boundary, proxy boundary, mesh boundary, dihedral feature

W_uncertain =
  low_support_or_disagreement only when no stable planar/detail explanation exists
```

The policy arrays cover 111,167 faces. Representative statistics:

```text
face_weight_planar:    median 0.112, q90 0.510, max 0.994
face_weight_detail:    median 0.342, q90 0.630, max 0.914
face_weight_boundary:  median 0.413, q90 0.879, max 1.000
face_weight_uncertain: median 0.053, q90 0.350, max 0.350
```

This confirms the later revision: uncertainty is capped and should be
conservative, not dominant. The more useful information is captured by support,
detail pressure, target length, and direction confidence.

Planar proxies are fitted by weighted PCA. For a patch with weighted samples
`x_i`, compute:

```text
c = sum_i w_i x_i / sum_i w_i
C = sum_i w_i (x_i - c)(x_i - c)^T / sum_i w_i
```

The plane normal is the eigenvector of `C` with the smallest eigenvalue. Proxy
residual is the normalized point-to-plane error:

```text
r_i = |n^T (x_i - c)|
```

This gives a local plane model for projection and for future OMeGa regularizers.

## 7. Phase 4: Weighted Local Remesh Attempts

We implemented a mutating weighted remesh pass:

```text
pre-project confident planar vertices to proxy planes
score collapse pressure over all edges
collapse non-conflicting accepted edges
flip edges for quality/connectivity
relax detail vertices tangentially
project planar vertices again
write operation JSONL and report
```

The collapse energy is inspired by QEM and local remeshing, but every operation
is gated by our policy fields:

```text
accept collapse only if:
  topology/link condition is valid
  projection displacement is small enough
  normal/plane error is within tolerance
  triangle quality does not degrade too much
  boundary/detail/uncertainty weights do not object
```

The weighted local remesh result:

```text
faces:              111,167 -> 78,455  (29.43% reduction)
vertices:            61,443 -> 44,963  (26.82% reduction)
accepted collapses:  16,344
accepted flips:       4,002
planar-projected source faces: 17,812
detail-relaxed source faces:      515
```

This is useful as a diagnostic, but visually it was still not clean enough. The
method did not reduce planar regions into the simple designer mesh we want, and
boundary/detail gates often preserved noisy seam regions. It also revealed an
important lesson: a post-hoc local operator is too late to fix geometry that
OMeGa never reconstructed well.

We also implemented a stricter planar patch remesh:

```text
group high-W_planar faces into local plane patches
project patch-interior vertices to fitted planes
collapse only same-patch interior edges
preserve high detail/uncertainty outside patches
```

The patch result:

```text
faces:                  111,167 -> 103,166
vertices:                61,443 -> 57,429
planar patches:              424
planar source faces:      17,990
feature-preserve faces:   31,646
accepted collapses:        4,000
```

This avoided some planar bleeding across boundaries, but it was still not an
ideal solution. It treats detail/corner areas too conservatively. For example,
a stair tread/riser seam should become two clean planes meeting at a sharp edge,
not a dense preserved band forever.

## 8. Baseline Remeshing Experiments

We organized the external baselines into method families and ran target ratios:

```text
0.20, 0.40, 0.60 of the preclean face count
```

The preclean mesh has:

```text
vertices: 61,443
faces:   111,167
```

### QEM Family

Methods:

```text
open3d_qem
meshlab_qem
meshlab_planar_qem
```

Representative 20% target row:

```text
open3d_qem:          22,233 faces, median triangle quality 0.835
meshlab_qem:         22,233 faces, median triangle quality 0.594
meshlab_planar_qem:  22,233 faces, median triangle quality 0.625
```

Observation: Open3D QEM was the strongest simple baseline. It often preserved
large architectural structure better than our early local remesh attempts. This
is important: any custom method must beat a strong QEM baseline, not just be
more complicated.

QEM math:

For each triangle plane `p = [a,b,c,d]^T`, accumulate a quadric:

```text
K_p = p p^T
Q_v = sum_{planes incident to v} K_p
```

For a candidate contraction to homogeneous point `x`, minimize:

```text
E_QEM(x) = x^T (Q_u + Q_v) x
```

This naturally preserves sharp structures when both sides of an edge contribute
different plane quadrics. That explains why simple QEM can reduce a stair seam
to two planes plus a sharp intersection better than a conservative local
classifier.

### MeshLab Isotropic Family

Methods:

```text
meshlab_isotropic
meshlab_feature_isotropic
meshlab_isotropic_qem
meshlab_feature_isotropic_qem
```

Representative 20% target row:

```text
meshlab_isotropic:              27,404 faces
meshlab_feature_isotropic:      34,778 faces
meshlab_isotropic_qem:          22,232 faces
meshlab_feature_isotropic_qem:  22,232 faces
```

Observation: isotropic remeshing improves triangle regularity but tends to
produce uniform triangle fields. It does not adapt strongly enough to local
architectural seams or sculptural details unless paired with stronger feature
or density control.

### Instant Meshes Family

Methods:

```text
instant_meshes_quad
instant_meshes_quad_guided
instant_meshes_tri
instant_meshes_tri_guided
instant_meshes_dominant
instant_meshes_dominant_guided
```

Instant Meshes solves orientation and position fields over the input surface:

```text
orientation field Q
position field P
integer-grid extraction -> quads/triangles
```

We patched Instant Meshes so it can consume an OMeGa/StableNormal-derived
orientation sidecar. The supplied constraint is a soft direction:

```text
CQ_i  = desired tangent direction at vertex i
CQw_i = blend weight in [0, 1]
```

Important implementation fixes:

- `CQw` is treated as a blend coefficient, not an arbitrary energy weight.
- The guide is loaded after Instant Meshes clears and installs built-in
  boundary constraints, otherwise the guide can be erased.
- Constraint propagation was patched so weak OMeGa constraints are not hardened
  to weight 1 on coarse hierarchy levels.
- The image-space guide direction is lifted through the projection Jacobian
  where possible instead of only using a rough perpendicular image direction.

Observation: Instant Meshes is promising because its edges visibly follow
surface flow better than pure QEM or isotropic remeshing. However, mesh
completeness and density adaptation are still not solved by the orientation
field alone. The field can look reasonable while the extracted mesh is still
too uniform or incomplete.

The optional mesh-feature guided run adds high-dihedral mesh crease constraints
to the guided rows. This tests whether the normal-map guide benefits from
existing OMeGa mesh creases. It can help when OMeGa already has good creases,
but it may also preserve false creases from noisy geometry.

### Geogram / Vorpalite Family

Methods:

```text
geogram_anisotropic
geogram_anisotropic_gradation
geogram_omega_density
geogram_omega_density_gradation
```

Geogram/Vorpalite gave the most promising direction for density control. The
baseline uses anisotropic remeshing and target point count. We patched it with
OMeGa-guided density sidecars.

Density logic:

```text
high W_planar   -> lower density on large planes
high W_detail   -> higher density around detail
high W_boundary -> higher density around protected seams
W_uncertain     -> currently not used for density by default
```

The density sidecar replaces Geogram's default density field:

```text
rho(v) = map(signal(v), density_min, density_max)
```

where the signal is built from OMeGa Phase 3 weights.

We also tested planar gradation:

```text
smooth log density only within compatible planar regions
stop across proxy boundaries and strong normal-lift differences
clamp local density change by maxDensityRatio
```

The auto-budget experiment is especially relevant. With tolerated normal/detail
error of 8 deg:

```text
estimated target points: 19,847
estimated target faces:  39,694
valid face fraction:        0.825
vertex target length median: 0.0984
```

This is a strong conceptual result: instead of choosing a face ratio by hand, we
can derive target local edge length from tolerated normal error and turn that
into a density field.

Simplified auto-sizing relation:

```text
normal change per unit length ~= kappa
allowed normal error          = epsilon
target edge length h(x)       ~= epsilon / max(kappa(x), eps)
density rho(x)                ~= 1 / h(x)^2
```

In practice we clamp `h` and `rho` to avoid exploding density.

## 9. Main Findings

### Finding 1: StableNormal Is A Strong View Signal

The per-frame normal maps are high quality. They clearly separate many planar
surfaces from detail regions and provide useful high-gradient lines on ridges,
corners, stairs, and sculptural surfaces.

The strongest signals were:

```text
face_support
unexplained offset
detail target length
direction confidence
```

This supports using normal maps to regulate geometry, especially during OMeGa
training.

### Finding 2: Per-View Normals Are Not Direct 3D Ground Truth

Normal maps are estimated per view. A normal edge in image space may not project
consistently to the same 3D curve in all cameras. Foreground objects missing
from the mesh can project high normal gradients onto background faces.

Therefore:

```text
Use normals as evidence and weights.
Do not directly fuse them as a hard 3D normal field.
```

This is why support, offset tolerance, disagreement, and mesh anchoring matter.

### Finding 3: Hard Region Classification Was The Wrong Abstraction

Early attempts to classify faces as planar/detail/boundary/uncertain became
too brittle. Stairs, corners, and underfit details often looked uncertain, but
those are exactly the regions that need better optimization.

The better abstraction is:

```text
continuous local weights
local operation gates
debuggable accepted/rejected edits
```

### Finding 4: Post-Hoc Remeshing Alone Is Not Enough

Our local remesher and planar patch remesher are useful diagnostics, but they
do not yet produce the desired structured mesh. The strongest reason is that
some geometry is underfit before remeshing begins. A post-hoc method can clean
or simplify, but it cannot reliably reconstruct missing railings, stair detail,
or misplaced creases.

This points toward the next stage:

```text
bring the evidence and policy into OMeGa's mesh update loop
```

### Finding 5: Existing Baselines Remain Important

Open3D QEM is a strong simple baseline. Instant Meshes gives useful edge-flow
structure. Geogram gives useful density control. Our contribution should not be
"another remesher" in isolation. It should be:

```text
OMeGa optimization + StableNormal evidence + adaptive mesh policy
```

## 10. What We Patched Or Implemented

### OMeGa Fork

Implemented under:

```text
third_party/OMeGa_4_Building/omega_local/remesh/
third_party/OMeGa_4_Building/scripts/
```

Main additions:

- canonical mesh preprocess;
- optional CGAL mesh healing helper;
- full-frame view-normal evidence extraction;
- policy and proxy computation;
- weighted local remesh and planar patch remesh;
- baseline runner with organized method presets;
- Geogram density sidecar export;
- Instant Meshes OMeGa orientation sidecar export.

### Scan Processing Visualizations

Implemented under:

```text
scan_processing/VisOmega01_visualize_omega_preprocess.py
scan_processing/VisOmega02_visualize_omega_normal_evidence.py
scan_processing/VisOmega03_visualize_omega_policy_proxies.py
scan_processing/VisOmega04_visualize_omega_operation_proposals.py
scan_processing/VisOmega05_visualize_omega_weighted_remesh.py
scan_processing/VisOmega06_compare_omega_remesh_baselines.py
scan_processing/VisOmega07_visualize_instant_field.py
scan_processing/VisOmega08_visualize_geogram_density.py
```

These visualizations made the project much easier to reason about. The key
lesson is that every stage needs a direct image-space debug view. Without that,
the meaning of weights and operations is too easy to misread.

### Instant Meshes Patch

Local repo:

```text
third_party/instant-meshes/
```

We preserved original behavior unless an OMeGa sidecar is passed. The patch
adds optional:

```text
--omega-field
--omega-orientation-weight
--omega-dump-fields
--mesh-feature-constraints
```

The field visualizations let us compare:

```text
constraint_CQ_grid.png     supplied constraints
orientation_Q_grid.png     solved orientation field
```

### Geogram Patch

Local repo:

```text
third_party/geogram/
```

We built a headless Vorpalite binary and added OMeGa density/normal sidecars.
The most relevant extension is:

```text
StableNormal evidence -> h(x) -> rho(x) -> Geogram density field
```

## 11. Proposed Next Stage: Move Into OMeGa Training

The next step should test whether these signals can regulate OMeGa during
optimization.

Current OMeGa mesh update:

```text
optimize splats + mesh
periodically subdivide mesh
continue optimization
```

Proposed view-informed update:

```text
render current mesh into selected training views
read StableNormal evidence
compute lightweight policy fields
replace pure subdivision with adaptive update:
  - simplify/project confident planar regions
  - refine or preserve stable detail regions
  - protect/attract feature curves
  - keep uncertain/missing geometry conservative
reparent splats after topology changes
continue OMeGa optimization
```

The training-time policy should be different from post-hoc cleanup:

- post-hoc designer mesh should avoid densifying;
- training-time OMeGa update may densify where normal evidence shows real
  underfit detail;
- the splat binding lets OMeGa continue optimizing geometry after the update,
  so normal evidence can become a regularizer rather than a final projection.

Candidate training losses or regularizers:

```text
planar projection regularizer:
  E_plane = W_planar * distance(vertex, proxy_plane)^2

normal/detail consistency:
  E_normal = W_detail * angular_error(rendered_mesh_normal, StableNormal)^2

feature preservation:
  E_feature = W_boundary * distance(vertex, lifted_feature_curve)^2

adaptive density:
  target edge length h(x) = epsilon / normal_variation(x)
```

The important design point is to keep StableNormal as evidence. The mesh
positions should remain anchored by OMeGa geometry and photometric optimization.

## 12. Presentation Framing

The main team update can be framed as:

```text
We explored post-hoc remeshing of OMeGa output and found that existing
remeshers alone do not solve the structured geometry problem. However, the
StableNormal evidence is very informative. It gives usable signals for
planarity, missing/underfit detail, local target edge length, and orientation
flow. The next research step is to use these signals inside OMeGa's mesh update
loop, where geometry can still be optimized after refinement/simplification.
```

Recommended slide sequence:

1. OMeGa output: strong visual quality, weak designer mesh structure.
2. Desired mesh: large planes, clean seams, detail-aware density.
3. Pipeline built: preprocess, evidence, weights, local remesh, baselines.
4. StableNormal evidence examples: support, detail length, direction confidence,
   unexplained offset.
5. Why post-hoc remesh was not enough.
6. Baseline comparison: QEM, isotropic, Instant Meshes, Geogram.
7. Patched methods: Instant Meshes constraints and Geogram density.
8. Key insight: normal maps should regulate OMeGa training, not just remesh the
   final mesh.
9. Next milestone: training-time adaptive mesh update.

## 13. Slide-Ready Outline

### Slide 1: Goal

Title:

```text
Toward Structured Mesh Reconstruction From OMeGa
```

Main bullets:

- OMeGa gives strong visual reconstruction, but the mesh is not yet designer
  friendly.
- We want large clean planes, sharp seams, and adaptive detail density.
- The current exploration tests how to regulate OMeGa's mesh using per-frame
  StableNormal evidence.

Figure suggestion:

- Left: OMeGa rendered result or RGB view.
- Right: dense/irregular OMeGa mesh wireframe.

### Slide 2: Problem With The Current Mesh

Title:

```text
Good Rendering Does Not Guarantee A Clean Mesh
```

Main bullets:

- Large planar regions are split into many small triangles.
- Stairs, corners, and sculptural details are not consistently aligned by mesh
  edges.
- Missing/underfit geometry can appear as noisy evidence on nearby surfaces.
- A post-hoc remesher must avoid destroying real detail while simplifying
  planes.

Figure suggestion:

- Before mesh wireframe from `VisOmega01` or `VisOmega05`.
- Zoom crops of wall/stair/detail regions.

### Slide 3: Our Pipeline

Title:

```text
View-Informed Remesh Exploration Pipeline
```

Main bullets:

- Preprocess OMeGa mesh for stable topology.
- Render face-id/depth buffers into all training views.
- Transfer StableNormal gradients and direction evidence to mesh faces.
- Build continuous planar/detail/boundary/uncertainty weights.
- Compare local remesh attempts against QEM, Instant Meshes, and Geogram.

Diagram:

```text
OMeGa mesh + cameras + StableNormal
  -> view evidence
  -> mesh weights
  -> remesh / baseline tests
  -> training-time update proposal
```

### Slide 4: StableNormal Evidence

Title:

```text
Per-Frame Normals Reveal Useful Geometry Signals
```

Main bullets:

- `face_support`: which mesh faces are reliably observed.
- `detail_target_length`: where local mesh density should be higher.
- `direction_confidence`: where normal gradients imply useful edge flow.
- `unexplained_offset`: where the mesh does not explain visible normal
  structures.

Figure suggestion:

- One `VisOmega02` panel.
- Show normal map, high-gradient normal pixels, support, target length, and
  direction confidence.

### Slide 5: Math Intuition

Title:

```text
Normals Become Weights, Not Hard Geometry
```

Main bullets:

- Compute normal derivatives in each view:

```text
g(u) = sqrt(||dN/du||^2 + ||dN/dv||^2)
```

- Use the normal structure tensor to estimate local flow direction.
- Project view evidence onto rendered mesh face ids.
- Convert evidence into continuous weights:

```text
W_planar, W_detail, W_boundary, W_uncertain
```

Takeaway:

- StableNormal is strong evidence, but not direct 3D ground truth.

### Slide 6: Policy Weights

Title:

```text
Continuous Weights Work Better Than Hard Classes
```

Main bullets:

- Hard planar/detail/uncertain classes were too brittle.
- Many "uncertain" regions are actually underfit detail.
- We now use continuous weights to drive local operation behavior.
- Best signals so far: support, unexplained offset, target length, direction
  confidence.

Figure suggestion:

- One `VisOmega03` policy panel.
- Include `W planar`, `W detail`, `W uncertain`, and `unexplained offset`.

### Slide 7: Local Remesh Attempt

Title:

```text
Post-Hoc Local Remeshing Is Useful But Not Enough
```

Main bullets:

- Weighted local remesh reduced faces from 111k to 78k.
- It projected planar regions and accepted 16k collapses.
- Result was still not clean enough on planes and seams.
- Post-hoc cleanup cannot fully fix geometry OMeGa never reconstructed.

Numbers:

```text
faces:     111,167 -> 78,455
vertices:   61,443 -> 44,963
collapses:  16,344
flips:       4,002
```

Figure suggestion:

- `VisOmega05` before/after/edge-delta panel.

### Slide 8: Baseline Comparison

Title:

```text
External Baselines Clarify What Matters
```

Main bullets:

- Open3D QEM is a strong simple simplification baseline.
- Isotropic remeshing improves triangle quality but is too uniform.
- Instant Meshes gives better edge-flow structure.
- Geogram is promising for adaptive density control.

Figure suggestion:

- `VisOmega06` method grid with QEM, isotropic, Instant Meshes.
- `VisOmega08` Geogram density comparison.

### Slide 9: Patched Methods

Title:

```text
Injecting OMeGa Evidence Into Existing Remeshers
```

Main bullets:

- Patched Instant Meshes with soft StableNormal/OMeGa orientation constraints.
- Added optional mesh-feature constraints for existing OMeGa creases.
- Patched Geogram workflow with OMeGa-guided density sidecars.
- Tested auto density from tolerated normal error.

Key formula:

```text
h(x) ~= epsilon / normal_variation(x)
rho(x) ~= 1 / h(x)^2
```

Result:

```text
8 deg normal tolerance -> about 19.8k target points
```

### Slide 10: Main Finding

Title:

```text
The Evidence Pipeline Is The Main Result So Far
```

Main bullets:

- The final remesh is not yet the desired structured mesh.
- But StableNormal evidence reliably identifies planarity, detail scale, and
  edge-flow direction.
- Existing remeshers help us understand the design space.
- The next step is to use this evidence during OMeGa optimization.

One-line takeaway:

```text
Normal maps should regulate mesh evolution, not only post-process the final mesh.
```

### Slide 11: Next Step

Title:

```text
Move From Post-Hoc Remesh To Training-Time Regulation
```

Main bullets:

- Replace pure subdivision with evidence-aware mesh updates.
- Simplify/project confident planar regions.
- Refine or preserve underfit high-detail regions.
- Protect feature curves and reparent splats after topology changes.
- Continue OMeGa optimization after the mesh update.

Diagram:

```text
OMeGa training loop
  -> render mesh into views
  -> StableNormal evidence
  -> adaptive mesh update
  -> splat reparenting
  -> continue optimization
```

### Slide 12: Discussion Questions

Title:

```text
Open Questions For The Next Experiment
```

Main bullets:

- How often should evidence be recomputed during training?
- How many views are enough for a useful update?
- Should detail regions split, remesh, or only receive stronger losses?
- Can planar proxies persist across iterations?
- What mesh-quality metric should define success?

Suggested discussion:

- Pick one training-time update strategy and one evaluation scene region:
  stairs, planar wall, railing/missing geometry, or sculptural detail.

## 14. Open Questions

- How often should view-normal evidence be recomputed during OMeGa training?
- Can a small view subset approximate the full 214-frame evidence field?
- Should refinement use explicit splits, OMeGa-style subdivision, or a
  Geogram/Instant-inspired remesh event?
- How should splat reparenting preserve appearance after topology edits?
- Can planar proxies become persistent state across training iterations?
- How do we evaluate final mesh quality beyond face count?

Useful metrics:

```text
face count on planar regions
plane residual on proxy regions
normal-map consistency after remesh/update
crease alignment in projected views
triangle quality
boundary preservation
designer inspection of stairs, walls, railings, sculptural regions
```

## 15. Short Conclusion

The strongest result so far is not the final remeshed mesh. The strongest result
is the evidence pipeline. StableNormal maps give high-quality per-view geometry
signals that OMeGa's current mesh does not fully use. We now have the machinery
to transfer those signals onto the mesh, visualize them, turn them into
continuous weights, and test them against strong classical baselines.

The next step is to move from:

```text
post-hoc remesh of final OMeGa output
```

to:

```text
view-informed mesh regulation during OMeGa optimization
```

That is where the normal evidence is most likely to produce cleaner structured
reconstruction rather than only cleaner post-processing.
