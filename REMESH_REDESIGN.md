# OMeGa View-Informed Remesh Redesign

This document is the implementation contract for the remesh refactor. The
current face-count QEM scripts are source material only. They do not need to
remain backward compatible, and they may be fully refactored to follow this
logic.

The target method is a deterministic view-informed local remesher:

```text
OMeGa mesh + cameras + StableNormal/PromptDA normal maps
  -> minimal mesh pre-clean
  -> optional defect healing experiment
  -> rendered face-id/depth buffers
  -> per-view normal geometry and noise tolerances
  -> offset-tolerant normalized evidence transfer
  -> continuous face/edge policy fields
  -> continuous planar/detail/boundary/uncertainty weights
  -> global weighted collapse/flip/relax/project pass
  -> conservative evidence-gated healing
  -> triangle mesh plus optional planar patch export
```

The core rule is:

```text
Never ask "what target face count should this mesh have?"
Ask "would this local operation violate the view-normal tolerance,
plane-fit tolerance, protected boundary, image-space support, density field,
or topology?"
```

The first production target is post-hoc remeshing of the final OMeGa mesh. Only
after that result is useful should the same operator replace OMeGa's midpoint
subdivision event during training.

## 0. Canonical Working Dataset

Follow the existing runbooks:

```text
RUN_RAW_TO_OMEGA.md
RUN_OMEGA_REMESH.md
```

The current development target is the DSLR PINHOLE StableNormal package and its
stronger OMeGa output:

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export OUT_ROOT="$DT_ROOT/data/scan_processing_outputs"
export PYTHON="$DT_ROOT/.venv/bin/python"
export OMEGA_BUILDING_ROOT="$DT_ROOT/third_party/OMeGa_4_Building"

export DSLR_PINHOLE_PACKAGE_ROOT="$OUT_ROOT/grove_entrance_dslr_0521_pinhole"
export PACKAGE_ROOT="$DSLR_PINHOLE_PACKAGE_ROOT"
export OMEGA_OUTPUT_NAME=omega_stable_mesh
export OMEGA_ROOT="$PACKAGE_ROOT/$OMEGA_OUTPUT_NAME"
export OMEGA_DATASET_DIR="$OMEGA_ROOT/dataset"
export OMEGA_RESULT_DIR="$OMEGA_ROOT/model_baseline_stronger_30000"
export OMEGA_SOURCE_MESH="$OMEGA_RESULT_DIR/plys/mesh_29999_rank0.ply"
```

Conventions:

- Use `$PYTHON`, not system `python`, for new commands.
- Keep remesh code and OMeGa-side scripts inside `$OMEGA_BUILDING_ROOT`.
- Keep camera-view visualizations in `scan_processing/`.
- Write post-hoc remesh artifacts under `$OMEGA_RESULT_DIR/remesh/`.
- Do not rerun OMeGa data export with `--overwrite` against this package unless
  the user explicitly asks for a fresh export.
- Treat `$OMEGA_SOURCE_MESH` as the first test mesh for new post-hoc stages.
- Use the same visualization sampling style as the runbooks:
  `--frame-stride 4 --max-frames 32`.

## 1. Refactor Policy

The old remesh path was useful for exploration, but the new implementation must
follow this document exactly.

Current files to refactor, not preserve:

- `omega_local/remesh/evidence.py`
- `omega_local/remesh/regions.py`
- `omega_local/remesh/plane_adjustment.py`
- `omega_local/remesh/view_informed_qem.py`
- `omega_local/remesh/posthoc.py`
- `scripts/run_omega_mesh_evidence.py`
- `scripts/run_omega_mesh_region_scores.py`
- `scripts/run_omega_plane_adjustment.py`
- `scripts/run_omega_view_informed_remesh.py`
- scan-processing visualizations that consume the old arrays

Useful code that should be reused where appropriate:

- OMeGa dataset/camera/normal-map loading helpers in `evidence.py`.
- Debug mesh and JSON/JSONL writing patterns.
- Training-time splat reparenting in `omega_local/remesh/training.py`.
- OMeGa's existing topology insertion point in `gsplat/strategy/meshgs.py`.

The old PyMeshLab `target_faces` path should be removed from the remesh
pipeline. Recreating it as a separate comparison utility is out of scope for
this refactor unless the user explicitly asks for it later.

## 2. Goal

Input:

```text
M = (V, F)                   final or intermediate OMeGa triangle mesh
K views                      OMeGa training cameras
N_k(u)                       StableNormal/PromptDA camera-frame normal map
P_k                          camera projection
Z_mesh_k(u), FID_mesh_k(u)   rendered mesh depth and face-id buffers
```

Output:

```text
M_remesh                     clean triangle mesh
policy fields               face/edge/vertex arrays used to make decisions
proxy graph                  planar components, boundaries, reliability
operations JSONL             accepted/rejected local operations and gate values
summary JSON                 exact tolerances, counts, rejected operations
optional designer patches    merged planar n-gons for CAD-style inspection
```

The mesh should have:

- large triangles or polygonal patches on confident planar regions;
- preserved and optionally subdivided triangles in high normal-variation zones;
- edges aligned with geometric ridges, creases, and planar proxy boundaries;
- no blind hole filling or watertightness enforcement;
- deterministic outputs and debuggable intermediate fields.

StableNormal/PromptDA normals are evidence, not ground truth. They carry sharp
local geometric structure, but they can be inconsistent across views. Therefore
the pipeline must not directly average predicted normals across cameras as the
main geometry signal.

## Phase 1 Mesh Preprocess And Healing

Phase 1 has two lanes:

```text
1A. canonical preclean        exact/topology safety cleanup
1H. optional healing test     local defect repair, written to separate outputs
```

### 1A. Canonical Preclean

The canonical input to evidence extraction remains `preclean_mesh.ply`. This
stage is intentionally narrow:

- merge exact duplicate vertices;
- remove duplicate faces by vertex set;
- remove invalid, non-finite, and zero-area faces;
- remove isolated vertices;
- split bow-tie one-ring vertex fans;
- preserve holes, open boundaries, components, and surface geometry.

This lane exists to make later face-id/depth rendering stable without changing
the geometric hypothesis OMeGa produced.

### 1H. Optional CGAL Healing

The healing lane starts from `preclean_mesh.ply` and writes
`preclean_healed_mesh.ply`; later stages use it only if a command explicitly
points to it. The goal is to test whether local mesh defects are blocking better
remeshing, not to force watertightness.

Backend:

- Existing method/library: CGAL Polygon Mesh Processing.
- Links: [CGAL Polygon Mesh Processing](https://doc.cgal.org/latest/Polygon_mesh_processing/index.html),
  [CGAL GitHub](https://github.com/CGAL/cgal).
- Local code: `tools/cgal_mesh_heal.cpp` plus
  `omega_local/remesh/mesh_heal.py`.

Operations, in order:

```text
repair degenerate triangles
optional cap/needle almost-degenerate repair
duplicate non-manifold vertices
stitch exactly coincident boundary halfedges
fill selected small boundary cycles
optional tiny connected-component removal
optional self-intersection repair
```

Conservative defaults:

- small-hole filling is gated by boundary edge count and a diameter measured
  relative to the median input edge length;
- tiny component deletion is disabled unless `component_area_factor > 0`;
- cap/needle repair is disabled by default because it can move connectivity;
- self-intersection repair is disabled by default because CGAL may delete and
  recreate local patches;
- exact self-intersection counting is enabled for diagnostics, but can be
  disabled for speed.

What to inspect:

- before/after boundary edges and boundary cycles;
- before/after non-manifold vertex count;
- before/after degenerate face count;
- before/after self-intersection pair count;
- whether filled holes correspond to real small defects instead of real open
  architectural gaps;
- whether healed wireframes preserve valid planar and sculptural structures.

## Prior Work Basis

Phase 4 should cite and borrow ideas from established remeshing work, but it
should not call a black-box global remesher as the main path. We need control
over exactly which edges move, why they move, and how view-normal evidence
enters every decision.

Use these works as the methodological base:

- Garland and Heckbert, "Surface Simplification Using Quadric Error Metrics",
  SIGGRAPH 1997: use accumulated plane quadrics and local edge contractions as
  the base collapse placement model.
- Botsch and Kobbelt, "A Remeshing Approach to Multiresolution Modeling", SGP
  2004, and the CGAL/geometry-central isotropic remeshing implementations: use
  the split/collapse/flip/tangential-relax/project operator sequence and a
  sizing field, but replace the uniform or curvature-only sizing field with our
  view-normal policy field.
- Cohen-Steiner, Alliez, and Desbrun, "Variational Shape Approximation",
  SIGGRAPH 2004: use proxy planes as an explicit approximation model for
  planar regions, not merely as visualization labels.
- Alliez, Cohen-Steiner, Devillers, Levy, and Desbrun, "Anisotropic Polygonal
  Remeshing", SIGGRAPH 2003: use direction fields to align edges in detail
  regions. Do not implement global curvature-line tracing in the first local
  remesh version.
- Hu, Yan, Bommes, Alliez, and Benes, "Error-Bounded and Feature Preserving
  Surface Remeshing with Minimal Angle Improvement", TVCG 2017: treat
  approximation error as a hard gate and triangle quality as an optimization
  goal, not as a reason to violate geometry evidence.
- Heep, Behnke, and Zell, "Feature-Preserving Mesh Decimation for Normal
  Integration", CVPR 2025: use screen-space normal-map quadrics and anisotropic
  Delaunay-style connectivity as the model for view-local normal evidence.
  This is most relevant to evidence extraction, OMeGa training-time geometry
  refinement, and edge-flow diagnostics, not to the final global architectural
  proxy graph by itself.
- Nivoliers, Levy, and Geuzaine, "Anisotropic and Feature Sensitive Triangular
  Remeshing Using Normal Lifting", 2015: use position-plus-normal metrics as a
  reference for feature-sensitive anisotropic triangle cleanup. This is a local
  and patch-level remeshing idea, not the whole designer-patch reconstruction.
- Vorsatz, Rossl, Kobbelt, and Seidel, "Feature Sensitive Remeshing", 2001:
  use curvature/feature attraction as an older reference for making vertices
  move toward real feature lines instead of smoothing them away.
- Jakob, Tarini, Panozzo, and Sorkine-Hornung, "Instant Field-Aligned Meshes",
  SIGGRAPH Asia 2015: use as the strongest external field-aligned baseline.
  It should help us distinguish "QEM keeps existing structure" from "a new
  orientation/position field creates cleaner edge flow." Keep it outside the
  production remesher until we decide how view-normal weights should constrain
  or seed the field.

Implementation choice:

- implement the local operator driver ourselves in Python/NumPy/SciPy so every
  proposal can be logged and visualized;
- use `trimesh`/Open3D only for I/O, debug geometry, nearest-point checks, and
  optional validation;
- keep PyMeshLab/CGAL-style remeshing as comparison baselines only, not as the
  production operator;
- use C++/CGAL bindings only for optional Phase 1 healing/diagnostics and
  external baselines, not as the main production remesher.

Map the literature to implementation roles:

```text
normal-integration decimation  -> view-local anisotropic evidence and training cues
anisotropic polygonal remesh   -> continuous direction-field edge flow
feature-sensitive remesh       -> boundary attraction and no-cross constraints
PolyFit / VSA / proxy methods  -> soft plane priors and optional patch export
CGAL PMP repair                -> optional Phase 1 healing and diagnostics
CGAL / constrained CDT         -> optional validation or comparison backend
```

## Baseline Remesh Track

The baseline track is a controlled comparison suite, not the production
remesher. Every baseline consumes the same preclean OMeGa mesh and writes to:

```text
$OMEGA_RESULT_DIR/remesh/baselines/<experiment_name>/
```

Visualizations write to:

```text
$PACKAGE_ROOT/visualizations/omega_remesh_baselines/<experiment_name>/<model_name>/
$PACKAGE_ROOT/visualizations/omega_remesh_geogram_density/<experiment_name>/<model_name>/
$PACKAGE_ROOT/visualizations/omega_remesh_instant_fields/<experiment_name>/<model_name>/
```

The runner is `scripts/run_omega_remesh_baselines.py`. New runs should use
`--experiment-name` instead of hand-written `--output-dir` paths. Visualizers
can infer baseline summaries and density sidecars from the same experiment name.
Legacy folders such as `remesh/baselines_key_compare/` are kept readable as
fallbacks but are not the current convention.

### Why This Track Exists

The custom remesher should be compared against strong mesh-only baselines before
we add more handcrafted logic. If a standard method creates cleaner stairs,
corners, or sculptural edge flow than our current view-informed pass, that is a
design signal: our custom method is missing an operator, not merely a threshold.

The baseline track answers four questions:

```text
1. Does global QEM already simplify architectural planes and stair seams better?
2. Does remesh-before-decimate improve triangle quality enough to matter?
3. Does a field-aligned solver create cleaner edge flow than QEM?
4. Does view-normal evidence help density/field guidance beyond mesh-only input?
```

### Existing Baselines

`open3d_qem`

- Existing method: Open3D quadric edge-collapse decimation.
- Link: [Open3D TriangleMesh.simplify_quadric_decimation](https://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html#open3d.geometry.TriangleMesh.simplify_quadric_decimation).
- Math: each vertex stores a quadric error accumulated from incident face
  planes; an edge collapse chooses a placement that minimizes the summed
  quadric error, then repeats until the target triangle count is reached.
- Why it matters here: it often preserves strong corners and stair-scale
  structures surprisingly well because the error model is global and geometric.
- Our extension: none. This is a pure mesh-only baseline.

`meshlab_qem`

- Existing method: MeshLab/PyMeshLab quadric edge-collapse decimation.
- Link: [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html).
- Math: QEM edge collapse with MeshLab's quality, boundary, topology, and normal
  preservation gates.
- Why it matters here: it separates Open3D implementation behavior from the QEM
  idea itself.
- Our extension: none.

`meshlab_planar_qem`

- Existing method: MeshLab QEM with `planarquadric=true`.
- Link: [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html).
- Math: standard QEM plus additional planar quadrics that bias collapses toward
  preserving locally planar regions.
- Why it matters here: it tests whether ordinary planar quadrics are enough for
  architecture without our normal-map evidence.
- Our extension: none.

`meshlab_isotropic`

- Existing method: MeshLab explicit isotropic remeshing.
- Link: [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html).
- Math: repeated split, collapse, edge flip, smooth, and reproject operations
  toward a uniform target edge length. It changes vertex positions and
  connectivity rather than only deleting vertices.
- Why it matters here: it isolates edge regularization and triangle quality from
  decimation.
- Our extension: none. The runner can derive target length from target face
  ratio with `L = sqrt(4A / (sqrt(3) N_target))`.

`meshlab_feature_isotropic`

- Existing method: the same MeshLab explicit isotropic remeshing with a lower
  feature-angle threshold and smoothing disabled by default.
- Link: [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html).
- Math: same split/collapse/flip/remap loop, but feature detection is more
  aggressive and smoothing is less likely to blur corners.
- Why it matters here: it tests whether feature-aware quality improvement helps
  sculptural/architectural details without a separate QEM pass.
- Our extension: parameter preset only, no new math.

`meshlab_isotropic_qem` and `meshlab_feature_isotropic_qem`

- Existing methods combined: MeshLab explicit remeshing followed by MeshLab
  planar QEM.
- Link: [PyMeshLab filter list](https://pymeshlab.readthedocs.io/en/latest/filter_list.html).
- Math: first regularize connectivity and edge length, then decimate with planar
  QEM. This is not a faithful implementation of Alliez et al. anisotropic
  polygonal remeshing; it is an installed practical approximation to
  remesh-before-decimate behavior.
- Why it matters here: it shows whether cleaner triangles before decimation
  produce better corners/planes than QEM alone.
- Our extension: parameter preset only.

`instant_meshes_quad`, `instant_meshes_tri`, `instant_meshes_dominant`

- Existing method: Jakob et al., Instant Field-Aligned Meshes, SIGGRAPH Asia
  2015.
- Links: [project page](https://rgl.epfl.ch/publications/Jakob2015Instant),
  [GitHub](https://github.com/wjakob/instant-meshes).
- Math: solve a smooth orientation field and position field over the mesh, then
  extract a quad, triangle, or dominant mesh aligned to those fields. The field
  is local in the sense that it varies over the surface, but the solver uses
  global smoothing and compatibility propagation.
- Why it matters here: it can create edges that follow geometry better than QEM
  because it builds a new field-aligned mesh instead of only collapsing existing
  edges.
- Our extension: none for these rows. The patched binary still runs original
  behavior when no `--omega-field` is passed.

`geogram_anisotropic`

- Existing method: Geogram/Vorpalite anisotropic CVT remeshing with normal
  lifting, based on Nivoliers, Levy, and Geuzaine, 2015.
- Links: [Geogram GitHub](https://github.com/BrunoLevy/geogram), [paper record](https://orbi.uliege.be/handle/2268/183733).
- Math: append scaled normals to positions and optimize a centroidal Voronoi
  tessellation in lifted `(x, alpha n)` space. Points that cross sharp normal
  variation become far apart in the metric even when their 3D positions are
  close, producing anisotropic, feature-sensitive triangles after projection.
- Why it matters here: it is our best installed baseline for anisotropic
  geometry-following triangles.
- Our extension: none for this row.

`geogram_anisotropic_gradation`

- Existing method: Geogram/Vorpalite anisotropic CVT with local-feature-size
  gradation.
- Link: [Geogram GitHub](https://github.com/BrunoLevy/geogram).
- Math: same lifted CVT metric, but Vorpalite modulates sample density using its
  LFS/gradation field.
- Why it matters here: it tests adaptive density without StableNormal evidence.
- Our extension: none for this row.

### OMeGa Extensions To Existing Baselines

`instant_meshes_quad_guided`, `instant_meshes_tri_guided`,
`instant_meshes_dominant_guided`

- Base method extended: Instant Field-Aligned Meshes.
- What we add: an optional OMeGa sidecar `omega_instant_field.txt` containing a
  per-vertex tangent-flow direction `q_i` and confidence `w_i` from Phase 2
  StableNormal evidence. The current run uses `gradient_mode=combined`, which
  chooses between point and local normal-evidence tensors by coherent guide
  score.
- Math: per-view image normal gradients create image-space flow directions;
  these are pulled back through the projection Jacobian to the mesh tangent
  plane and accumulated as unoriented tensors `T += w t t^T`. The principal
  eigenvector becomes `q_i`; confidence becomes `w_i`.
- How it enters Instant Meshes: the patched C++ binary writes `q_i` into
  orientation constraints `CQ` and `clamp(w_i * scale, 0, 1)` into `CQw`.
  `CQw` is a blend coefficient, not an arbitrary unbounded energy weight.
- What remains unchanged: Instant Meshes still owns the global orientation solve,
  position solve, extraction, subdivision, and mesh generation. We do not supply
  direct vertex positions.
- Visualization: VisOmega07 shows `constraint_CQ_grid.png` for supplied OMeGa
  constraints and `orientation_Q_grid.png` for the solved field.

`geogram_omega_density`

- Base method extended: Geogram/Vorpalite anisotropic CVT with normal lifting.
- What we add: an OMeGa scalar density sidecar keyed by preclean mesh vertices.
- Math from Phase 3 weights:

  ```text
  signal = max(
      1 - W_planar,
      detail_influence * W_detail,
      boundary_influence * W_boundary,
      uncertain_influence * W_uncertain
  )
  rho = lerp(rho_min, rho_max, signal^gamma)
  ```

  The default `uncertain_influence=0` because uncertainty often marked underfit
  details rather than a useful separate density driver.
- How it enters Geogram: the patched Vorpalite reads `omega:density_file` and
  replaces the internal vertex density field before CVT optimization. Input
  repair/hole/component edits are disabled for these rows so vertex ids stay
  aligned with the sidecar; normal lifting remains active.
- What remains unchanged: Geogram still owns CVT optimization, projection, and
  anisotropic metric behavior.

`geogram_omega_density_gradation`

- Base method extended: Geogram/Vorpalite anisotropic CVT with our OMeGa density
  sidecar.
- What we add: bounded planar-only log-density smoothing before Vorpalite reads
  the sidecar.
- Math:

  ```text
  w_ij = min(W_planar_i, W_planar_j)^p
         * exp(-(alpha ||n_a - n_b|| / (tau_m max(|e_ij|, 0.25 median_edge)))^2)
         * proxy_gate_ij

  log rho_i <- (log rho_i + lambda sum_j w_ij log rho_j)
               / (1 + lambda sum_j w_ij)
  ```

  `proxy_gate_ij` is zero across proxy boundaries or different valid proxy ids,
  and the final value is clamped around the original density by a max ratio.
- Why it exists: raw OMeGa density preserves details but can be noisy; Geogram's
  own LFS gradation can overwhelm our density signal. This middle path smooths
  density only inside compatible planar regions.
- Visualization: VisOmega08 shows raw density, planar-gradated density,
  gradation change, normal-lift gate, proxy gate, and method output grid.

### Current Experiment Names

Use these names for new runs:

```text
qem_compare               QEM simplification baselines
isotropic_compare         isotropic remesh and isotropic-then-QEM baselines
instant_guided_compare    original Instant Meshes vs combined OMeGa-guided
geogram_compare           Geogram anisotropic and OMeGa density variants
```

Use the baseline track to judge these behaviors, not just final face count:

```text
large planes become simpler
stair/riser intersections stay sharp
sculptural curvature keeps enough density
triangle/quad edges follow visible structure
view-normal guidance improves density or flow without hallucinating geometry
```

Do not split the scene into hard planar/detail/deferred classes as the main
production decision mechanism. The production remesher should be a weighted
local operator: planar, detail, boundary, and uncertainty evidence all enter the
same local costs and gates continuously. Proxies are still valuable, but they
act as soft plane projection and boundary priors rather than as a separate
global classification stage.

## Instant Meshes View-Normal Extension Track

Instant Meshes is now an extension baseline as well as a mesh-only baseline.
The original C++ behavior remains the default. OMeGa evidence enters only when
the optional `--omega-field` sidecar is passed to the patched binary.

The extension is intentionally conservative:

```text
OMeGa mesh vertices/topology/depth visibility
  remain the 3D surface anchor

StableNormal per-view derivatives
  become soft tangent-flow evidence

Instant Meshes orientation constraints
  receive CQ/CQw soft fields

Instant Meshes position constraints
  remain unused for now
```

The exported field is per vertex:

```text
q_i   tangent-flow direction on the mesh surface
w_i   confidence in [0, 1]
n_i   mesh vertex normal for debugging only
```

Per-view image-space normal structure tensors are lifted to the mesh tangent
plane by a projection-Jacobian pullback. For a visible mesh sample with camera
projection `pi`, surface tangent basis `B`, and image-space flow `t_img`, solve

```text
min_a || J_pi B a - t_img ||^2
t_world = normalize(B a)
```

where `J_pi` is the local perspective projection Jacobian at the rendered
depth. This is preferred over the earlier small-angle camera-axis lift because
it respects perspective, off-center pixels, and slanted surfaces. Degenerate
grazing cases are treated as zero-confidence observations. The fallback vector
is kept only for diagnostics. The lifted unoriented tangent contributes
`t_world t_world^T`; the dominant eigenvector of the accumulated tensor becomes
`q_i`; eigenvalue anisotropy, face support, normal-gradient strength, direction
confidence, projection conditioning, and view angle become `w_i`.

This is the key math rule:

```text
image normal gradient direction       g
image feature-flow direction          t_img = perpendicular(g)
projection-Jacobian pullback          t = argmin_surface_tangent ||J_pi t - t_img||^2
lift quality                          l = residual_quality * conditioning * view_angle_quality
accumulated evidence                  T += w * l * t t^T
vertex constraint                     q = principal_eigenvector(T)
```

Use tensors rather than averaging raw directions because the field is an
unoriented line field. Opposite signs from different views should reinforce the
same flow, not cancel.

The C++ patch injects `q_i, w_i` through the existing Instant Meshes
orientation-constraint arrays:

```text
CQ[:, i]  = q_i
CQw[i]   = clamp(w_i * omega_orientation_weight, 0, 1)
```

`CQw` is a blend coefficient in Instant Meshes' orientation update, not an
unbounded energy weight. During multiresolution constraint propagation, soft
weights must remain in `[0, 1]`; combining two children keeps the maximum child
confidence rather than summing and hardening the result. Hard mesh boundary
constraints still dominate. The C++ load order is clear constraints, add hard
boundaries, add optional mesh-feature constraints, then add OMeGa soft guides;
loading OMeGa before boundary setup would erase it. Guided runs now allow
Instant Meshes' normal internal subdivision path: the sidecar field is loaded
on the original OMeGa vertices and interpolated onto split midpoint vertices.
`--omega-no-subdivide` remains only as a debug option, because disabling
subdivision can starve extraction and produce incomplete meshes.

Debugging outputs:

```text
omega_instant_field.txt/.npz
orientation_optimized.txt
position_optimized.txt
VisOmega07 omega_input_Q_grid.png
VisOmega07 orientation_Q_grid.png
VisOmega07 constraint_CQ_grid.png
```

Read them as:

- `omega_input_Q_grid`: the Python-exported 3D guide before Instant Meshes
  loads boundary, mesh-feature, or OMeGa constraints;
- `constraint_CQ_grid`: what the view-normal evidence asked Instant Meshes to
  prefer;
- `orientation_Q_grid`: what Instant Meshes solved after global smoothing and
  compatibility;
- final `VisOmega06` mesh grid: what extraction produced from the solved field.

This track should answer whether view-normal evidence can improve Instant
Meshes' edge flow without trusting monocular normals as direct 3D positions.

## Weighted Global Remesh Pass

The scene contains both architectural elements and sculptural/detail regions.
The next production stage should not first classify faces into mutually
exclusive regions. Instead, every local operation should be driven by four
continuous weights:

```text
W_planar      collapse more, project vertices toward reliable planes
W_detail      preserve useful density, align edges to normal-flow directions
W_boundary    prevent crossing seams/corners, attract vertices to feature curves
W_uncertain   low-support/no-explanation fallback; do not freeze stable geometry
```

This keeps the logic simple:

```text
existing evidence and proxies
  -> compute continuous face/edge/vertex weights in Phase 3
  -> run a global weighted remesh pass in Phase 4
  -> validate each operation against support, normal, plane, boundary, and
     uncertainty tolerances
  -> export optional planar patches after the triangle mesh is stable
```

### Data Phase 4 Receives

Phase 1 preprocessing has produced:

- `preclean_mesh.ply`: adjacency-safe OMeGa mesh;
- `preclean_summary.json`: exact safety cleanup counts;
- `preclean_diagnostics.npz`: removed faces, fan-split faces, source ids.

Phase 2 evidence extraction has produced:

- view-space face-id/depth buffers under `remesh/view_buffers/`;
- mesh edge and silhouette masks per view;
- StableNormal/PromptDA normal gradients and high-gradient pixels;
- per-view normal tolerances `tau_g`, `tau_n`, `tau_a`;
- image-to-mesh transferred face evidence in `evidence.npz`;
- face support, view disagreement, normal-gradient strength, target edge length,
  position tolerance, direction tensor/confidence, and unexplained offset.

Phase 3 policy/proxy/weight extraction has produced:

- `policy.npz` with `face_support`, `face_planar_score`,
  `face_detail_posterior`, `face_unexplained_offset`,
  `face_detail_target_length`, `face_position_tolerance`,
  `face_direction_tensor`, `face_direction_confidence`,
  `face_boundary_score`, continuous face/edge/vertex weights, and edge policy
  arrays;
- `proxies.npz` and `proxies.json` with planar proxy ids, weighted plane fits,
  residuals, proxy boundaries, support, and missing/foreground warnings.

Phase 4 consumes these weights. It should not recompute the evidence model or
make a separate hard classification.

These Phase 3 artifacts are enough to run the global weighted remesh pass
without re-rendering every camera during the inner loop.

### Phase 3 Weight Construction

Use the existing policy fields directly. Names below describe the intended math;
the exact implementation can use clipped smoothstep ramps instead of hard
thresholds.

Per-face inputs:

```text
S_f       face_support
P_f       face_planar_score or planar base score
D_f       face_detail_posterior
B_f       face_boundary_score plus proxy/dihedral boundary evidence
O_f       face_unexplained_offset
A_f       face_view_disagreement
C_f       face_direction_confidence
```

Suggested weights:

```text
W_uncertain_f =
    (1 - S_f) * (1 - stable_explanation_f)^2

W_feature_f =
    smooth_max(O_f, A_f)
    * support_gate_f
    * (1 - 0.65 stable_planar_f)

W_planar_f =
    P_f * S_f * (1 - 0.5 D_f) * (1 - W_uncertain_f)

W_detail_f =
    smooth_max(
        D_f * (0.25 + 0.75 C_f) * (0.25 + 0.75 sqrt(S_f))
        ,
        O_f * stable_detail_f * (0.25 + 0.75 sqrt(S_f)),
        W_feature_f
    )

W_boundary_f =
    smooth_max(normal-gradient transition score, proxy_boundary_score,
               dihedral_feature_score)
```

`face_unexplained_offset` is not a synthetic boundary. In the current
simplification-first policy, supported residual offset/disagreement becomes
feature/detail pressure, while uncertainty is reserved for low-support regions
with no stable planar/detail explanation. This avoids turning half of the mesh
into a freeze mask.

Low support alone is not enough to make a face highly uncertain. It becomes
uncertain only when the face also lacks a stable planar or detail explanation.
Likewise, offset evidence that overlaps stable detail becomes refinement/detail
pressure; only the residual unexplained offset remains uncertainty. This
prevents stable sculptural/detail regions from being suppressed merely because
the current mesh is too coarse or slightly offset.

Boundary evidence should remain edge-centric. `edge_weight_boundary` is the
operation constraint used to stop collapse/flip across seams. `face_weight_boundary`
is only an averaged diagnostic projection for face-buffer visualization, so one
high boundary edge does not turn a whole large triangle into a boundary patch.

Lift weights to edges with conservative reducers:

```text
W_planar_e    = min(W_planar_i, W_planar_j)
W_planar_any_e = max(W_planar_i, W_planar_j)
W_detail_e    = max(W_detail_i, W_detail_j)
W_boundary_e  = max(W_boundary_i, W_boundary_j, mesh_boundary_e)
W_uncertain_e = max(W_uncertain_i, W_uncertain_j)
```

These arrays are Phase 3 outputs. Phase 4 treats them as input policy fields,
the same way it treats target length and position tolerance.

Concrete `policy.npz` array names:

```text
face_weight_planar       edge_weight_planar       vertex_weight_planar
face_weight_detail       edge_weight_detail       vertex_weight_detail
face_weight_boundary     edge_weight_boundary     vertex_weight_boundary
face_weight_uncertain    edge_weight_uncertain    vertex_weight_uncertain

face_stable_explanation
face_boundary_from_edges
face_offset_detail_pressure
face_residual_unexplained_offset
face_low_support_uncertainty
face_disagreement_uncertainty
```

### Operation Behavior

Collapse:

- high `W_planar` increases collapse pressure and allows stronger plane
  projection;
- before collapse, confident planar vertices are projected toward proxy planes
  so the collapse/quality gates evaluate a cleaner planar mesh;
- planar interiors use `min(W_planar_i, W_planar_j)`, while plane-adjacent
  fringe cleanup may use `max(W_planar_i, W_planar_j)` if uncertainty/detail are
  low and the edge is not a mesh boundary or proxy-to-proxy boundary;
- high `W_detail` reduces collapse pressure unless the edge is clearly
  redundant under the anisotropic normal metric;
- high `W_boundary` forbids crossing mesh boundaries and proxy-to-proxy
  boundaries. Face-boundary scores can be softly reduced for one-sided planar
  fringe cleanup, but only when the planar side is confident and detail/
  uncertainty do not object;
- proxy-to-proxy planar feature edges may collapse along the two-plane
  intersection. This simplifies stair tread/riser seams and similar sharp
  architectural intersections without crossing the seam;
- high `W_uncertain` raises acceptance thresholds and reduces displacement.

Flip:

- high `W_detail` prioritizes edge-flow alignment with the dominant direction
  tensor;
- high `W_planar` prioritizes triangle quality and clean triangulation inside
  the plane;
- high `W_boundary` forbids flips that would cross, erase, or blur a seam;
- high `W_uncertain` allows only strict quality improvements.

Relax/project:

- high `W_planar` moves vertices toward the weighted proxy/local plane;
- high `W_detail` moves vertices tangentially, preserving density and aligning
  edges to the normal-flow direction;
- high `W_boundary` attracts eligible vertices toward feature curves while
  keeping the curve constrained;
- high `W_uncertain` shrinks the step size or disables motion.

Split:

- post-hoc split remains disabled by default because the designer mesh should
  simplify or clean, not densify;
- split pressure remains useful as an OMeGa training cue where the current mesh
  cannot explain stable high-detail evidence;
- if a post-hoc split is explicitly enabled later, it must be gated by low
  uncertainty and high detail support.

### What This Step Should Build

Refactor the current Phase 4 dry-run into a global weighted remesh pass.
Weight construction belongs to Phase 3:

```text
omega_local/remesh/weights.py
scan_processing/VisOmega03_visualize_omega_policy_proxies.py
```

Phase 4 consumes those weights:

```text
omega_local/remesh/weighted_local_remesh.py
omega_local/remesh/feature_curves.py
omega_local/remesh/plane_projection.py
scan_processing/VisOmega05_visualize_omega_weighted_remesh.py
```

Implementation order:

1. move `weights.py` into Phase 3 policy output and visualize the weights;
2. implement a Phase 4 global proposal pass over all edges/vertices;
3. implement the mutating weighted pre-project/collapse/flip/relax-project pass
   without post-hoc splits;
4. add validation, a human-readable behavior report, diagnostics, and optional
   planar patch export from
   high-`W_planar` connected components.

The pass is global in coverage but local in operations: every edge and vertex is
considered under one consistent weighted objective, while each accepted edit is
still a local collapse, flip, or relax/project update. It does not require a
hard architectural/detail/deferred classification before topology can change.

### Relevant Prior Work For This Step

Use different works as ingredients in one weighted local operator:

- Garland and Heckbert QEM: base edge collapse and placement model.
- Botsch/Kobbelt and CGAL-style remeshing: local split/collapse/flip/relax
  operator sequence, but driven by our view-normal weights.
- Anisotropic Polygonal Remeshing: detail-region edge-flow alignment and
  anisotropic sizing through `W_detail`, direction tensors, and target length.
- Feature-Preserving Mesh Decimation for Normal Integration: normal-map-derived
  anisotropic quadrics, conservative feature preservation, and training-time
  split pressure.
- Feature-sensitive remeshing and normal lifting: feature-curve attraction,
  position-plus-normal metrics, and edge alignment around sharp structures.
- VSA/PolyFit-style proxy fitting: reliable plane priors for projection and
  optional designer patch export, not hard classification.
- Error-bounded feature-preserving remeshing: hard acceptance gates for normal,
  plane, projection, topology, and triangle quality errors.

Success for this step means:

- broad architectural surfaces collapse and project cleanly toward planes;
- sculptural/detail regions keep meaningful density and anisotropic edge flow;
- seams, corners, and proxy boundaries are preserved or sharpened by boundary
  attraction instead of being crossed by collapses;
- uncertain/missing-geometry regions remain conservative;
- every operation can be explained by the four weights plus its gate values.

Current Phase 4B implementation scope:

```text
policy.npz + proxies.npz + preclean_mesh.ply
  -> pre-collapse projection of confident planar vertices to proxy planes
  -> global weighted edge-collapse candidate pressure
  -> planar-interior pressure from min planar weight
  -> planar-transition pressure from one-sided confident planar weight
  -> feature-edge pressure from proxy-to-proxy planar seams
  -> adaptive detail simplification target from normal evidence and source-mesh scale
  -> batched non-overlapping collapses
  -> quality-driven non-overlapping edge flips
  -> bounded tangential relaxation in supported detail regions
  -> bounded final proxy-plane projection in supported planar interiors
  -> topology/link-condition/normal/quality/plane/projection gates
  -> mesh_weighted_remesh.ply, source-face diagnostics, operations JSONL,
     and mesh_weighted_remesh_report.md
```

Planar merging math:

```text
interior_planar_pressure_e =
    short(edge_length_e, alpha_planar * target_length_e)
    * min(W_planar_i, W_planar_j)
    * support_gate_e

transition_planar_pressure_e =
    short(edge_length_e, alpha_planar_transition * mean_target_length_e)
    * max(W_planar_i, W_planar_j)
    * support_gate_e
    * certainty_gate_e
    * (1 - 0.75 W_detail_e)
```

The transition term is only allowed when the edge is not a mesh boundary and
not a proxy-to-proxy boundary. It exists to clean ragged regions adjacent to
planes without converting real detail or uncertain foreground projection into
wall collapse.

Detail-region cleanup math:

```text
detail_target_f =
    max(face_detail_target_length_f,
        face_mean_edge_length_f * detail_mesh_target_factor)

detail_simplify_pressure_e =
    short(edge_length_e, alpha_detail * mean(detail_target_i, detail_target_j))
    * W_detail_e
    * direction_confidence_e
    * support_gate_e
```

This keeps post-hoc Phase 4B non-densifying: detailed regions can simplify when
oversampled, flip for quality/connectivity, and move tangentially for cleanup,
but they do not split unless a future training-time/remesh mode explicitly
enables that operator.

Feature-edge simplification math:

```text
feature_edge_pressure_e =
    short(edge_length_e, alpha_feature_edge * mean_target_length_e)
    * mean(W_planar_i, W_planar_j)
    * support_gate_e
    * certainty_gate_e
```

This term is evaluated only when the edge is a proxy-to-proxy planar boundary
with two valid adjacent planar proxies. Its candidate position is projected to
the intersection line of the two proxy planes. This is the local equivalent of
QEM preserving a sharp crease while removing excess vertices along that crease.

### Alternative: Planar Patch Remesh

If the weighted local operator remains too conservative, use a simpler
region-first path:

```text
policy.npz + proxies.npz + preclean_mesh.ply
  -> planar_candidate_f = W_planar_f >= tau_planar
  -> connected planar patches using face adjacency, seed-plane compatibility,
     fitted-plane compatibility, and edge boundary stops
  -> fit one plane per patch
  -> project patch vertices to their plane
  -> collapse patch-interior edges aggressively
  -> preserve non-planar high max(W_detail, W_uncertain) regions
```

This path intentionally ignores most subtle uncertainty logic. It treats
`max(W_detail, W_uncertain)` as one feature-preserve weight outside planar
patches, because the current goal is a clean designer mesh, not conservative
post-hoc evidence preservation everywhere.

Interpretation:

- high `W_planar` means "make this part a planar patch";
- high `max(W_detail, W_uncertain)` outside a planar patch means "keep local
  complexity and only do minor cleanup";
- seams between two planar patches are hard no-cross boundaries in the first
  strict patch version. A tread and riser should become two separate planes
  sharing a protected edge, not one merged patch that bleeds around the corner;
- the output remains a triangle mesh, but `source_face_patch_id` records the
  planar patch grouping and can later drive polygon export.

Patch growth is intentionally stricter than pairwise coplanarity. Pairwise
compatibility can drift through a chain of almost coplanar triangles and cross a
real seam. The strict patch pass grows every component against one reference
plane, refits the plane, trims out faces that no longer match the fitted plane,
then collapses only edges whose two incident faces came from that same patch.
`edge_weight_boundary`, raw normal-gradient boundary score, proxy-boundary
edges, and mesh boundaries all stop patch growth.

Projection and collapse are also boundary-aware at the vertex level. A vertex
that belongs to both a planar patch and a non-patch/detail face is left in
place, because moving it would deform geometry outside the patch mask. A
collapse is accepted only when both endpoints are true patch-interior vertices:
all incident faces at each endpoint belong to the same patch. Vertices shared
only by two planar patches may project to the multi-plane intersection, but they
are not used for same-plane interior collapse across the seam.

This mutating pass deliberately does not split and does not simplify across
adjacent planar patches. Post-hoc remeshing should simplify and clean the
designer mesh; underfit high-detail areas remain protected and become
training-time refinement cues later. If later inspection shows stair seams need
more simplification along the intersection line, that should be a separate
feature-edge operator with its own visualization, not part of same-plane patch
interior merging.

## 3. Target Code Layout

Refactor toward this layout:

```text
omega_local/remesh/
  mesh_clean.py               # exact duplicate/null cleanup, adjacency safety
  view_buffers.py             # render mesh face-id/depth/edge buffers per view
  normal_maps.py              # normal derivatives, tolerances, tensors
  transfer.py                 # offset-tolerant view-to-mesh evidence transfer
  policy.py                   # continuous face/edge/vertex policy fields
  planar_proxies.py           # weighted planar components and proxy graph
  weights.py                  # planar/detail/boundary/uncertainty weights
  feature_curves.py           # soft feature curves and boundary constraints
  plane_projection.py         # weighted local/proxy plane projection helpers
  weighted_local_remesh.py    # mutating weighted collapse/flip/relax/project
  baselines.py                # mesh-only QEM/remeshing comparison baselines
  local_qem.py                # local collapse queue with normalized gates
  local_ops.py                # split, flip, relax helpers
  topology_repair.py          # support-aware cleanup and hole handling
  pipeline.py                 # post-hoc orchestration and summaries
  training.py                 # reparent splats after shared local remesh
```

Refactor scripts to these meanings:

```text
scripts/run_omega_mesh_evidence.py              # view buffers + normal transfer
scripts/run_omega_remesh_policy.py              # policy + planar proxies
scripts/run_omega_view_informed_remesh.py       # full local remesh, no target_faces
scripts/run_omega_remesh_baselines.py           # mesh-only comparison baselines
scripts/run_omega_training_remesh.py            # later training integration
```

Visualization layout in `scan_processing/` should continue the current pattern:

```text
VisOmega01... preprocessing verification
VisOmega02... evidence fields and view buffers
VisOmega03... policy/proxy scores
VisOmega04... dry-run operation proposal pressure
VisOmega05... weighted remesh fields and operation diagnostics
VisOmega06... mesh-only baseline comparison
```

## 4. Implementation Phases

The detailed stages below are the math specification. The implementation should
move through a smaller set of phase gates so each step is easy to test and
visualize before the next one changes behavior.

### Phase 1: Preprocess The OMeGa Mesh

Purpose:

```text
make the source mesh safe for adjacency, rasterization, and local topology ops
without changing its meaning
```

Build:

- `mesh_clean.py`
- a preprocessing mode in `pipeline.py`
- the first version of `scripts/run_omega_view_informed_remesh.py`

Outputs:

```text
<model_dir>/remesh/local/preclean_mesh.ply
<model_dir>/remesh/local/preclean_summary.json
<model_dir>/remesh/local/debug_meshes/preclean_changed_faces.ply
```

Visualization:

```text
original mesh vs preclean mesh
removed duplicate/null faces
non-manifold one-ring split markers
boundary loops before/after
```

Gate to continue:

- no holes are filled;
- no low-support components are deleted;
- face/vertex count changes are only safety cleanup;
- the cleaned mesh loads in the existing scan-processing visualization flow.

### Phase 2: Extract View-Normal Evidence

Purpose:

```text
turn cameras plus normal maps into normalized mesh evidence fields
without remeshing yet
```

Build:

- `view_buffers.py`
- `normal_maps.py`
- `transfer.py`
- `scripts/run_omega_mesh_evidence.py`
- `VisOmega02_visualize_omega_normal_evidence.py`

Outputs:

```text
<model_dir>/remesh/view_buffers/
<model_dir>/remesh/local/evidence.npz
<model_dir>/remesh/local/evidence_summary.json
```

Visualization:

```text
RGB or normal-map frame
rendered mesh face-id/depth buffers
high-gradient normal pixels
mesh edge/silhouette mask
offset search radius/delta map
face support, detail evidence, disagreement
```

Gate to continue:

- rendered buffers align with the same DSLR cameras used by OMeGa;
- depth outliers contribute zero support;
- visible ridges in normal maps transfer to nearby mesh faces even when the mesh
  is slightly offset;
- unsupported or inconsistent faces are visible in the debug maps.

Current Phase 2 implementation contract:

- default mesh is the Phase 1 `preclean_mesh.ply`;
- default evidence outputs live in `<model_dir>/remesh/local/`;
- default view buffers live in `<model_dir>/remesh/view_buffers/`;
- CPU triangle rasterization writes exact face-id/depth buffers for every
  selected frame;
- normal-map derivatives use per-view robust tolerances:
  `sigma_g`, `tau_g`, `tau_n`, and `tau_a`;
- face support is normalized by per-view visible pixel capacity, then by view
  count, then penalized by cross-view normal-gradient disagreement;
- high-gradient reverse transfer associates shifted normal ridges to nearest
  rendered mesh pixels inside the measured per-view offset radius;
- this stage writes only evidence arrays and debug buffers. It never changes
  topology or geometry.

### Phase 3: Build Remesh Policy, Weights, And Planar Proxies

Purpose:

```text
convert evidence into continuous operation policy fields:
where to simplify, where to preserve, where to align boundaries, and how
conservative each operation should be
```

Build:

- `policy.py`
- `weights.py`
- `planar_proxies.py`
- `scripts/run_omega_remesh_policy.py`
- `VisOmega03_visualize_omega_policy_proxies.py`

Outputs:

```text
<model_dir>/remesh/local/policy.npz
<model_dir>/remesh/local/proxies.json
<model_dir>/remesh/local/proxies.npz
<model_dir>/remesh/local/debug_meshes/policy/
<model_dir>/remesh/local/debug_meshes/proxies/
```

Visualization:

```text
face support
detail posterior
planar score
planar/detail/boundary/uncertainty weights
target length
edge boundary score
proxy ids and proxy residuals
proxy boundaries over camera views
```

Gate to continue:

- planar walls receive high planar score and large target lengths;
- details, creases, silhouettes, and proxy boundaries are protected;
- weights are smooth and interpretable: planes are planar-weighted, stable
  curved/detail regions are detail-weighted, seams are boundary-weighted, and
  missing/foreground projections are uncertainty-weighted;
- proxy regions stop at real ridges instead of leaking across geometry changes;
- the policy can be inspected without running any topology operation.

Current Phase 3 implementation contract:

- consume the full Phase 2 `evidence.npz`;
- fit a weighted two-component mixture on log normal-gradient evidence, then
  combine it with target-length and direction-confidence cues to get
  `face_detail_posterior`;
- compute local mesh planarity by weighted PCA over face neighborhoods;
- compute `face_unexplained_offset` separately from stable detail. This catches
  projected foreground or missing geometry, such as railings in normal maps
  that are not represented by the mesh;
- unexplained offset is a missing/foreground-geometry cue. It should not
  directly force wall subdivision, suppress planar score, create boundary
  score, or block planar proxy growth;
- split protection into detail, missing-geometry, disagreement, and boundary
  channels. The combined detail/proxy protection score must not hide missing
  geometry inside ordinary detail protection;
- build edge arrays for dihedral feature score, boundary score, support, detail,
  target length, and position tolerance;
- compute continuous face/edge/vertex weights:
  `W_planar`, `W_detail`, `W_boundary`, and `W_uncertain`;
- write the weights into `policy.npz` so Phase 4 only consumes them;
- grow planar proxies from high support plus geometric planarity and only across
  low-boundary edges. Missing/foreground geometry is recorded on the proxy but
  does not cut holes out of otherwise planar walls;
- write proxy ids and residuals as `proxies.npz` plus human-readable
  `proxies.json`;
- this stage writes policy/proxy arrays only. It never changes topology or
  geometry.

### Phase 4: Apply Global Weighted Remesh Pass

Purpose:

```text
rewrite topology only when continuous weights and local normalized gates agree
the operation is safe
```

Phase 4 must start as a dry run. The dry run uses the same policy fields,
quadrics, target lengths, topology checks, and rejection reasons as the
mutating pass, but it writes proposals instead of editing the mesh. This keeps
the next visualization gate honest: if the proposal heatmaps are wrong, the
topology code is not allowed to run yet.

Build in three passes:

1. `Global proposal pass`: consume Phase 3 weights and score every edge/vertex
   for collapse, flip, relax/project, and training-only split pressure. This
   pass is first run as a dry run for visualization.
2. `Weighted operation pass`: mutate the mesh with ordered local operations:
   collapse first, then flip, then bounded relax/project. Each operation uses
   the same four weights and the same normalized gates.
3. `Validation and export pass`: run conservative topology cleanup, image/proxy
   audits, operation diagnostics, and optional planar patch export.

Modules:

- `local_qem.py`
- `local_ops.py`
- `feature_curves.py`
- `plane_projection.py`
- `weighted_local_remesh.py`
- `topology_repair.py`
- full mode in `pipeline.py`
- `VisOmega04_visualize_omega_operation_proposals.py`
- `VisOmega05_visualize_omega_weighted_remesh.py`
- `VisOmega06_compare_omega_remesh_baselines.py` for mesh-only baseline checks

Outputs:

```text
<model_dir>/remesh/local/mesh_remeshed.ply
<model_dir>/remesh/local/mesh_remeshed.policy.npz
<model_dir>/remesh/local/mesh_remeshed.operations.jsonl
<model_dir>/remesh/local/mesh_remeshed.summary.json
```

Dry-run outputs:

```text
<model_dir>/remesh/local/operation_proposals.npz
<model_dir>/remesh/local/operation_proposals.jsonl
<model_dir>/remesh/local/operation_proposals.summary.json
<model_dir>/remesh/local/debug_meshes/operations/
```

Visualization:

```text
planar/detail/boundary/uncertainty weights
weighted target length and projection strength
detail direction-field alignment
feature-curve attraction and no-cross constraints
collapse/split/flip proposal pressure
accepted and rejected collapses
training-only split pressure and rejection reasons
flipped edges
relaxed/projected vertices
per-operation rejection reason heatmaps
per-view projection and normal overlays
```

Gate to continue:

- operation logs explain every accepted and rejected operation;
- planar regions simplify and project without losing real boundaries;
- sculptural/detail regions keep density and anisotropic edge flow;
- boundary weights prevent collapses/flips from crossing seams or corners;
- uncertainty visibly suppresses aggressive changes;
- no `target_faces` argument is used anywhere in the main remesh path.

Current Phase 4 implementation contract:

- consume `preclean_mesh.ply`, `policy.npz`, `proxies.npz`, and `proxies.json`;
- consume Phase 3 `W_planar`, `W_detail`, `W_boundary`, and `W_uncertain`
  arrays directly. Do not recompute weights in Phase 4;
- compute final proxy-aware target lengths from planar proxy area/extent, scene
  edge statistics, and Phase 2 detail target lengths;
- first run the global proposal pass as a dry run. Do not mutate vertices,
  faces, splats, or optimizer state in that mode;
- evaluate collapse pressure for oversampled edges using planar, detail,
  boundary, and uncertainty weights;
- evaluate split pressure only as a diagnostic/training cue for supported
  high-detail undersampled edges,
  but keep accepted split proposals disabled by default in the post-hoc
  simplification path. Split pressure is mainly a training-time refinement cue
  for OMeGa, not a directive to densify the designer mesh;
- evaluate flip pressure for boundary-safe quality/alignment candidates;
- evaluate relax/project pressure for planar projection, feature attraction,
  and detail tangential smoothing candidates;
- write edge arrays and face-projected arrays to `operation_proposals.npz`;
- write detailed accepted/rejected proposal rows to
  `operation_proposals.jsonl`;
- visualize proposal pressure, dry-run accepts, and rejection reasons with
  `VisOmega04`.

### Phase 4 Mutating Global Pass

The mutating remesher uses one continuous local objective instead of hard region
classification:

```text
policy weights + planar proxies + direction tensors
  -> score all edges/vertices in one global proposal pass
  -> collapse oversampled reliable planar/low-detail edges
  -> flip edges that improve quality or detail-flow alignment
  -> relax/project vertices with bounded weighted steps
  -> validate every operation against topology, normal, plane, boundary,
     uncertainty, and triangle-quality gates
```

Planar proxies remain useful, but only as weights and constraints:

- high planar weight adds proxy/local plane quadrics and projection;
- proxy boundaries contribute to boundary weight;
- high-detail or high-uncertainty evidence weakens plane projection instead of
  forcing a separate region type;
- optional designer planar patches are exported after the triangle mesh is
  stable, from connected high-planar-weight regions.

### Phase 5: Integrate With OMeGa Training

Purpose:

```text
replace midpoint-only subdivision during training with the validated local
remesh operator, while preserving splat identities through reparenting
```

Build:

- update `omega_local/remesh/training.py`
- call the shared local pipeline from OMeGa's topology event
- keep a training-safe config separate from the post-hoc config

Visualization:

```text
mesh_before_remesh.ply
mesh_after_remesh.ply
splat reparenting summary
training preview before/after topology event
```

Gate to continue:

- post-hoc result is already useful on the DSLR stronger mesh;
- splats reparent cleanly to the new faces;
- optimizer state is reset only where topology changed;
- training continues after the topology event without destroying visual quality.

## 5. Stage 0: Minimal Mesh Pre-Clean

Before rendering view buffers or building adjacency, perform only safety cleanup:

```text
remove exact duplicate faces
remove exact duplicate vertices
remove zero-area faces
remove isolated vertices
split obvious non-manifold vertices into separate one-ring fans
preserve holes and open boundaries
do not fill holes
do not delete low-support components yet
```

This is not topology healing. It only makes rasterization, adjacency, and local
QEM stable.

Required outputs:

```text
<model_dir>/remesh/local/preclean_mesh.ply
<model_dir>/remesh/local/preclean_summary.json
```

## 6. Stage A: Mesh View Buffers

Render the pre-cleaned mesh into every selected OMeGa camera.

Required per view:

```text
Z_mesh_k(u)        nearest mesh depth
FID_mesh_k(u)      visible face id, -1 outside mesh
N_mesh_k(u)        rendered mesh normal, optional but useful for debug
E_mesh_k(u)        rendered edge/silhouette mask
```

Requirements:

- Use the same undistorted/cropped/resized camera convention as OMeGa training.
- Store buffers under `<model_dir>/remesh/view_buffers/`.
- Make buffer resolution explicit in the summary JSON.
- Save frame manifest rows with image name, camera id, width, height, and paths.
- A deterministic CPU triangle rasterizer is acceptable for the first post-hoc
  version if a GPU mesh rasterizer is inconvenient.

These buffers are mandatory. Centroid projection alone is not enough because the
OMeGa mesh can be offset from the normal-map evidence.

## 7. Stage B: Per-View Normal Geometry

For every normal map, compute local screen-space geometry before projection to
the mesh.

Normal derivatives:

```text
N_u = dN / du
N_v = dN / dv
g(u) = sqrt(||N_u||^2 + ||N_v||^2)      # normal change per pixel
```

Do not use one `theta_N` for every purpose. Use separate tolerances with clear
units.

Gradient tolerance:

```text
G_low,k = { g_k(u) where g_k(u) <= Q20(g_k over valid pixels) }
sigma_g,k = 1.4826 * MAD(G_low,k)
tau_g,k = max(Q95(G_low,k), eps)        # normal change per pixel
```

Normal-change tolerance:

```text
tau_n,k = Q95(||N_k(u) - N_k(v)|| over valid neighboring low-gradient pixels)
```

Angular tolerance:

```text
tau_a,k = Q95(acos(clamp(dot(N_k(u), N_k(v)), -1, 1))
              over valid neighboring low-gradient pixels)
```

Use:

```text
tau_g,k  for gradient normalization
tau_n,k  for expected normal-change split/collapse decisions
tau_a,k  for mesh/proxy angle tests and normal-preservation gates
```

Target length in pixels:

```text
h_px,k(u) = tau_n,k / max(g_k(u), sigma_g,k)
h_px,k(u) = clamp(h_px,k(u), h_min_px, h_max_px)
```

Normal structure tensor for edge alignment:

```text
J(u) = [[ dot(N_u, N_u), dot(N_u, N_v) ],
        [ dot(N_u, N_v), dot(N_v, N_v) ]]
```

The eigenvector of `J` with the smaller eigenvalue gives the image direction
along which normals change least. This is usually the ridge or furrow tangent.

Required outputs per frame:

```text
normal_gradient_g
normal_noise_sigma_g
normal_tolerance_tau_g
normal_tolerance_tau_n
normal_tolerance_tau_a
target_edge_length_px
normal_structure_tensor
high_gradient_mask
```

## 8. Stage C: Offset-Tolerant Evidence Transfer

The transfer stage constructs face and edge fields from view evidence. It must
handle mesh-image offsets rather than assuming exact projection.

For each mesh sample `s` on face `f`:

```text
X_s       3D sample point
n_s       mesh normal at sample
u0        projection P_k(X_s)
l_f       local face edge scale
mpp_k     local meters per pixel = z / sqrt(fx * fy)
```

Use face centroid plus barycentric interior samples. Large faces should receive
more samples than tiny faces.

Search window:

```text
r_k(s) = max(2 px, projected_half_edge_length(f), delta_k)
```

Estimate offset automatically per view:

```text
delta_k = Q80 nearest-pixel-distance(
    rendered mesh edge/silhouette pixels,
    high-gradient normal pixels
)
```

For every pixel `u` inside the search window:

```text
w_xy    = exp(-||u - u0||^2 / (2 r_k(s)^2))
w_ray   = exp(-d_ray(X_s, ray_k(u))^2 / (2 eps_ray(s,k)^2))

if abs(dz(s,k,u)) > 3 eps_z(s,k):
    w_depth = 0
else:
    w_depth = exp(-dz(s,k,u)^2 / (2 eps_z(s,k)^2))

w(s,k,u) = c_N(u) * w_xy * w_ray * w_depth
```

Where:

```text
c_N(u)       normal-map validity/confidence
d_ray        point-to-camera-ray distance
dz           z_s - Z_mesh_k(u)
eps_ray      mpp_k(X_s) * r_k(s)
eps_z        max(l_f, 2 mpp_k(X_s))
```

Do not robustly cap support with Huber/Tukey-style residuals. Far depth
mismatches should contribute zero support. Robust losses can still be used later
inside fit costs.

Normalize support so it is comparable across face size, window size, view count,
and image resolution:

```text
mass_f,k = sum_{s,u} w(s,k,u)
capacity_f,k = sum_{s,u} c_N(u) * w_xy * w_ray
coverage_f,k = mass_f,k / max(capacity_f,k, eps)

view_count_f = count_k(coverage_f,k > tau_coverage)
coverage_f = weighted_mean_k(coverage_f,k)
S_f = coverage_f * (1 - exp(-view_count_f / k0))
```

Then penalize view disagreement:

```text
I_f = WMAD_k(kappa_f,k)
tau_I = Q75(I_f over supported faces)
S_f = S_f * exp(-I_f^2 / max(tau_I^2, eps))
```

Also implement reverse transfer:

```text
high-gradient normal pixels -> nearby mesh faces along camera rays
```

For each high-gradient pixel, distribute its `g_k(u)` and target-length evidence
to nearby visible faces within a tube radius:

```text
eps_tube = mpp_k(Z_mesh_k(u)) * delta_k
```

Use the same depth/ray plausibility gates as the forward transfer. This catches
image-visible ridges that are shifted away from the current mesh.

## 9. Stage D: Continuous Policy Fields

Compute continuous fields. Avoid hard semantic labels except for debug.

Normal detail:

```text
kappa_f,k = weighted_Q80(g_k(u) over associated pixels)
kappa_f   = weighted_Q80 over views
```

Fit a two-component mixture to:

```text
log(kappa_f + median_k(sigma_g,k))
```

The high-gradient posterior is:

```text
D_f in [0, 1]
```

Geometric planarity:

For a face neighborhood `N(f)`, run weighted PCA on neighboring centroids with a
normalized covariance:

```text
W = sum_i A_i S_i
c_bar = sum_i A_i S_i c_i / max(W, eps)
C = (1 / max(W, eps)) * sum_i A_i S_i (c_i - c_bar)(c_i - c_bar)^T
lambda_0 <= lambda_1 <= lambda_2
r_geo(f) = sqrt(lambda_0)
eps_pos(f) = weighted_Q50_k(mpp_k(c_f))
P_geo(f) = exp(-r_geo(f)^2 / max(eps_pos(f)^2, eps))
```

Final planar score:

```text
P_f = S_f * P_geo(f) * (1 - D_f)
```

Do not multiply by a separate `P_view = exp(-kappa_f^2 / tau^2)` in addition to
`(1 - D_f)`. That double-counts the same normal-variation evidence and can
overfreeze the mesh. If a low-gradient probability is needed for debugging,
write it as:

```text
P_low_gradient_f = 1 - D_f
```

Preliminary target length:

```text
L_detail(f) = weighted_Q30_k,u(mpp_k(X_s) * h_px,k(u))
L_f_pre = L_detail(f)
```

Final target lengths are computed after planar proxies exist.

Explicit edge aggregations:

```text
P_e       = min(P_fi, P_fj)       # conservative planar confidence
D_e       = max(D_fi, D_fj)       # protect if either side is detail
S_e       = min(S_fi, S_fj)
kappa_e   = max(kappa_fi, kappa_fj)
L_e       = min(L_fi, L_fj)
eps_pos_e = min(eps_pos_fi, eps_pos_fj)
tau_a_e   = robust aggregate of visible tau_a,k near the edge
tau_n_e   = robust aggregate of visible tau_n,k near the edge
```

For boundary edges with only one adjacent face:

```text
P_e = P_f
D_e = D_f
S_e = S_f
L_e = L_f
eps_pos_e = eps_pos_f
B_e = 1 if the boundary is a silhouette or open boundary that should be protected
```

Feature and boundary score per edge:

```text
B_mesh(e)  = 1 - exp(-angle(n_i, n_j)^2 / max(tau_a_e^2, eps))
B_view(e)  = weighted_Q90 normal-gradient evidence near projected edge
B_plane(e) = abs(P_fi - P_fj)
B_sil(e)   = silhouette probability from FID/alpha boundary
B_e        = 1 - product_m(1 - B_m(e))
```

If adjacent faces later belong to incompatible planar proxies, force `B_e = 1`.

Required policy arrays:

```text
face_support
face_view_disagreement
face_detail_posterior
face_low_gradient_probability
face_normal_kappa
face_planarity_geo
face_planar_score
face_detail_target_length
face_position_tolerance
edge_boundary_score
edge_feature_score
edge_preliminary_target_length
face_direction_tensor
face_direction_confidence
```

## 10. Stage E: Planar Proxy Extraction And Final Sizing

Extract planar proxies only after the continuous fields exist.

Seed faces:

```text
P_f assigned to high-planarity component of a two-component mixture
S_f high enough
D_f low enough
```

Region growth:

- connect adjacent seed faces only across low `B_e`;
- stop at feature boundaries, silhouettes, and high detail;
- allow small holes inside a proxy only if surrounding support is high.

For each proxy `R`, fit a plane by weighted normalized PCA:

```text
W_R = sum_{f in R} A_f S_f P_f
c_R = sum A_f S_f P_f c_f / max(W_R, eps)
C_R = (1 / max(W_R, eps)) * sum A_f S_f P_f (c_f - c_R)(c_f - c_R)^T
n_R = eigenvector_min(C_R)
d_R = -dot(n_R, c_R)
r_R = sqrt(lambda_min(C_R))
eps_R = max(weighted_Q75_{f in R}(eps_pos(f)), 2 r_R)
```

Before comparing or merging proxy planes, make orientation consistent:

```text
if dot(n_a, n_b) < 0:
    n_b = -n_b
    d_b = -d_b
```

Merge adjacent proxies if:

```text
angle = acos(clamp(dot(n_a, n_b), -1, 1))
angle <= tau_plane_angle
abs(d_a - d_b) <= max(eps_Ra, eps_Rb)
mean boundary score between them is low
```

`tau_plane_angle` is derived from robust per-view angular tolerance `tau_a,k`;
do not reuse the gradient tolerance here.

Proxy graph fields:

```text
proxy_id
plane normal n_R
plane offset d_R
assigned faces
boundary edges
adjacent proxies
support score
fit residual
eps_R
```

Planar proxies are not just labels. They enter collapse costs, boundary
protection, vertex projection, and optional final planar patch export.

Final target lengths:

```text
L_plane(f in R) = min(
    c_area * sqrt(area(R)),
    c_boundary * median_boundary_chord_or_edge_length(R),
    L_scene_max
)

L_f = exp(P_f * log(L_plane(f)) + (1 - P_f) * log(L_detail(f)))
L_e = min(L_fi, L_fj)
```

Do not force a triangle mesh to represent an entire long wall with one or two
sliver triangles. Designer-facing large planar n-gons are a separate export
step after the triangle mesh is stable.

## 11. Stage F: Local Remeshing Operator

Use three operation queues:

```text
collapse queue
split queue
flip queue
```

There is no target face count.

The operator is local by construction. It mutates only edges whose proposal
pressure is non-zero, plus a one-ring halo needed for topology validity. It
never runs an unconstrained global remesh over the whole scene.

Per-face signals used by Phase 4:

```text
S_f        face_support
P_f        face_planar_score
D_f        face_detail_posterior
U_f        low-support/no-explanation fallback uncertainty
A_f        face_view_disagreement
B_f        face_boundary_score
L_f        final target edge length in world units
eps_f      face_position_tolerance in world units
tau_a_f    normal-angle tolerance
proxy_f    planar proxy id, or -1

W_planar_f
W_detail_f
W_boundary_f
W_uncertain_f
```

Per-edge signals:

```text
S_e      = min(S_i, S_j)
P_e      = min(P_i, P_j)
D_e      = max(D_i, D_j)
U_e      = max(U_i, U_j)
B_e      = max(edge_boundary_score, proxy_boundary, mesh_boundary)
W_planar_e    = min(W_planar_i, W_planar_j)
W_detail_e    = max(W_detail_i, W_detail_j)
W_boundary_e  = max(W_boundary_i, W_boundary_j, B_e)
W_uncertain_e = max(W_uncertain_i, W_uncertain_j)
L_e      = min(L_i, L_j)
eps_e    = min(eps_i, eps_j)
tau_a_e  = min(tau_a_i, tau_a_j)
```

Proposal pressure:

```text
oversampled_e = ||e|| < alpha_collapse * L_e
undersampled_e = ||e|| > alpha_split * L_e

collapse_pressure_e =
    1(oversampled_e)
  * (0.75 * W_planar_e + 0.25 * S_e * (1 - W_detail_e))
  * (1 - W_boundary_e)^2
  * (1 - 0.25 W_uncertain_e)

split_pressure_e =
    1(undersampled_e) * W_detail_e * (1 - W_boundary_e) * (1 - 0.25 W_uncertain_e)
  + normal_change_pressure_e

flip_pressure_e =
    W_detail_e * direction_conf_e * possible_alignment_improvement_e
  + W_planar_e * possible_quality_improvement_e
```

Supported residual offset and disagreement enter feature/detail pressure.
`W_uncertain_e` is reserved for low-support regions without a stable
explanation, and it is a mild caution term rather than a freeze mask.

### 11.1 Quadrics

Classic face quadric:

```text
p_f = [n_x, n_y, n_z, d]^T
Q_f = A_f p_f p_f^T
Q_v = sum incident Q_f
```

Proxy quadric:

```text
Q_R(v) = lambda_proxy * W_planar_v * A_R(v) * conf_R * p_R p_R^T
conf_R = robust_mean(S_f * P_f over proxy R)
```

`A_R(v)` is the area of proxy faces locally assigned to vertex `v`. This keeps
the proxy quadric dimensionally comparable to accumulated face quadrics.
`conf_R` prevents weak or noisy proxies from dominating vertex placement.

Feature line quadric:

```text
||t|| = 1
A = I - t t^T
Q_line = [[A,      -A c],
          [-c^T A, c^T A c]]
Q_line_weighted = lambda_feature * length_local * W_boundary_e * Q_line
```

### 11.2 Collapse Candidate

For edge `(v_i, v_j)`:

```text
Q_total = Q_i + Q_j + Q_proxy + Q_feature
v_star = argmin [v,1]^T Q_total [v,1]
```

If the system is singular, test:

```text
v_i, v_j, midpoint(v_i, v_j), proxy-plane projection of midpoint
```

Placement constraints:

```text
v_new = v_star
if W_planar_e is high and a reliable local/proxy plane exists:
    v_new = lerp(v_new, project(v_new, plane), eta_planar * W_planar_e)
if W_boundary_e is high and a reliable feature curve exists:
    v_new = lerp(v_new, closest_point(v_new, feature_curve),
                 eta_boundary * W_boundary_e)
limit ||v_new - v_star|| by eps_e * (1 - 0.25 W_uncertain_e)
```

Choose the candidate with the lowest accepted normalized error, not merely the
lowest quadric value. This matters on large planar walls where endpoint motion
can be large while surface-to-surface error remains harmless.

### 11.3 Collapse Enqueue And Acceptance

Only enqueue a collapse if the edge is oversampled and the weighted pressure is
non-zero:

```text
oversampled_e = ||e|| <= alpha_collapse * L_e
collapse_pressure_e > tau_collapse_pressure

enqueue collapse iff oversampled_e and collapse_pressure_e > tau_collapse_pressure
```

Normalized density gate:

```text
E_density_collapse = 0   if collapse_pressure_e > tau_collapse_pressure
E_density_collapse = inf otherwise
```

Geometry fit:

```text
X = vertices, edge midpoints, face centers, and optional stratified samples
    from the affected old one-ring patch

E_fit = max_{x in X}
        dist(x, local_M_after)^2 / max(eps_x^2, eps)
```

Plane preservation:

```text
E_plane = W_planar_e * conf_R * (dot(n_R, v_new) + d_R)^2
          / max(eps_R^2, eps)

eps_R = max(proxy_rmse_R_q90, median(eps_f over R), eps_scene_min)
```

Normal preservation:

```text
E_normal =
    W_detail_e * max_faces(1 - dot(n_before, n_after))
    / max(1 - cos(tau_a_e), eps)
```

Feature preservation:

```text
E_feature = inf if collapse crosses an edge with W_boundary_e >= tau_boundary_hard
E_feature = W_boundary_e * dist(v_new, feature_curve)^2
            / max(eps_pos_e^2, eps) otherwise
```

Uncertainty preservation:

```text
E_uncertain = W_uncertain_e * ||v_new - midpoint(e)||^2 / max(eps_e^2, eps)
E_uncertain is a soft caution term. It should not become an infinite rejection
unless a future mode explicitly asks for conservative no-edit behavior.
```

Triangle quality:

```text
Q_tri(T) = 4 * sqrt(3) * area(T) / sum_edges ||e||^2
E_quality = inf if any affected new triangle has Q_tri < Q_min_hard
E_quality = sum max(0, Q_min_soft - Q_tri(T))^2 otherwise
```

Image-space surface displacement is an audit gate, not the first inner-loop
gate. The inner loop uses `eps_f` and `L_f`, which already came from camera
projection and normal evidence. After each operation batch, project samples
from changed patches into a small set of supporting views and reject or roll
back a batch if the projected displacement exceeds the image tolerance:

```text
E_img =
    max_{x in affected old surface samples, k visible}
        ||P_k(x) - P_k(project_to_M_after(x))||^2
        / eps_img(k)^2

eps_img(k) = max(1 px, 0.5 * delta_k)
```

Compare old local surface samples to the new local surface, not old vertex
positions to the collapsed vertex. This allows useful planar simplification
without blocking on harmless endpoint motion.

For the first post-hoc implementation, `E_img` is required for diagnostics and
sampled batch validation. It becomes a hard per-operation gate only after the
projection cache is fast enough.

Topology validity:

```text
E_topo = inf if the operation creates face flips, non-manifold edges,
         local self-intersections, or invalid boundary-loop collapse.
```

Accept an inner-loop collapse only if:

```text
max(
    E_fit,
    E_plane,
    E_normal,
    E_feature,
    E_uncertain,
    E_quality,
    E_density_collapse,
    E_topo
) <= 1
```

Accept an audited batch only if:

```text
sampled E_img <= 1
```

Priority:

```text
priority(e) =
    E_fit + E_plane + E_normal + E_feature + E_uncertain + E_quality
  - collapse_pressure_e
```

This makes reliable planar or low-detail oversampled edges collapse first and
stops automatically when the next collapse would violate data-derived
tolerances.

Operation logs must include the exact first failing gate:

```text
reject_boundary
reject_density
reject_topology
reject_fit
reject_plane
reject_normal
reject_uncertainty
reject_quality
reject_image_audit
```

## 12. Stage G: Detail Split, Flip, And Relax

### 12.1 Split

Use projected edge length, not only 3D length divided by a scalar mpp:

```text
DeltaN_expected(e) =
    weighted_Q80_k(kappa_e,k * ||P_k(v_i) - P_k(v_j)||)
```

The implementation may use the world-space `L_e` from Phase 2/3 as the fast
queue key, then evaluate `DeltaN_expected` only for high-detail candidate
edges. This keeps Phase 4 usable during OMeGa optimization.

Use hysteresis:

```text
tau_n_split = 1.5 * tau_n_e
tau_n_collapse = 1.0 * tau_n_e
```

Split if:

```text
S_e high
W_detail_e high
W_uncertain_e low
DeltaN_expected(e) > tau_n_split
operation passes topology and boundary constraints
```

Collapse decisions should use the lower collapse-side tolerance so split and
collapse do not oscillate.

New split vertices inherit interpolated evidence from the incident faces. New
faces inherit their parent face ids in `face_origin`, so later visualization can
paint both original evidence and post-operation geometry.

### 12.2 Flip

Direction field confidence from the face orientation tensor:

```text
conf_dir(f) = (lambda_max - lambda_min) / max(lambda_max + lambda_min, eps)
```

Only apply alignment when:

```text
W_detail_f high
conf_dir(f) high
S_f high
```

Use a cross-field so either principal direction is acceptable:

```text
C_align(e) =
    W_detail_e * conf_e *
    min(
        1 - abs(dot(e_hat, u_f)),
        1 - abs(dot(e_hat, n_f x u_f))
    )
```

Length cost:

```text
C_len = sum_{affected edges a} [log(||a|| / L_a)]^2
```

Triangle quality:

```text
Q_tri(T) = 4 * sqrt(3) * area(T) / sum_edges ||e||^2
C_quality = sum_{affected triangles T} max(0, Q_min - Q_tri(T))^2
```

Flip if:

```text
C_align_new + C_len_new + C_quality_new
<
C_align_old + C_len_old + C_quality_old
```

Reject flips that:

- cross high-`W_boundary` boundaries;
- create inverted or low-quality triangles;
- worsen severe valence defects;
- increase local projection/image displacement beyond tolerance.

Flip is a quality/alignment operation only. It must not be allowed to cross a
proxy boundary or erase a feature that the boundary weight protects.

### 12.3 Relax

Relaxation:

```text
planar term:  project toward local/proxy plane, scaled by W_planar_v
feature term: project toward feature curve, scaled by W_boundary_v
detail term:  tangential cotan/umbrella smoothing plus MLS projection,
              scaled by W_detail_v
uncertainty:  shrink or disable the step, scaled by W_uncertain_v
```

Use only tangential smoothing in detail regions and never smooth across high
`W_boundary_e` boundaries.

First implementation:

- planar-proxy interior vertices: project to the fitted proxy plane, with a
  displacement limit of `eps_f` and strength `W_planar_v`;
- high-boundary vertices: fixed or moved only along the feature curve;
- detail vertices: one or two small tangential smoothing/alignment steps only,
  followed by a local surface projection/audit;
- high-uncertainty or unsupported vertices: fixed except for cleanup of invalid
  zero-area faces.

## 13. Stage H: Topology Healing

Healing is a separate conservative stage. It must not blindly make open scene
meshes watertight.

Basic cleanup after operations:

```text
eps_snap = weighted_Q25(eps_pos(f))
merge duplicate vertices if distance < eps_snap
remove zero-area faces if area < eps_snap^2
remove duplicate faces
remove isolated vertices
split non-manifold vertices into separate one-ring fans
```

Bounded face reliability:

```text
R_photo_f = normalized photometric residual in [0, 1]
U_f       = uncertainty in [0, 1]
I_f       = intersection penalty in [0, 1]

H_f =
    S_f
  * (0.5 + 0.5 * P_f)
  * (1 - U_f)
  * (1 - R_photo_f)
  * (1 - I_f)

H_f = clamp(H_f, 0, 1)
```

Overlapping or duplicate sheets:

- detect candidate face pairs with a BVH;
- if nearly parallel and closer than `eps_pos`, keep the sheet with higher
  area-weighted reliability;
- mark deleted-sheet boundaries for hole handling.

Non-manifold edges:

- cluster incident faces by normal continuity and support;
- keep the best manifold pair/fan by reliability;
- detach or remove low-support extra sheets.

Hole filling:

- fill only if surrounding view evidence supports a surface;
- prefer planar proxy constrained fills;
- do not fill strong silhouettes or likely real openings;
- reject patches that exceed plane/MLS error or create self-intersections.

Component filtering:

```text
score(C) =
    area(C) * mean(H_f)
  + lambda_detail * area(C) * mean(S_f * D_f)
```

Fit a two-component mixture to component scores and remove low-support tiny
components unless they touch protected features or have strong multi-view
support. The detail bonus prevents small but real high-detail components from
being deleted only because their area is small.

## 14. Export And Diagnostics

Every stage must write summaries and debug artifacts.

Required final outputs:

```text
<model_dir>/remesh/local/mesh_remeshed.ply
<model_dir>/remesh/local/mesh_remeshed.policy.npz
<model_dir>/remesh/local/mesh_remeshed.proxies.json
<model_dir>/remesh/local/mesh_remeshed.operations.jsonl
<model_dir>/remesh/local/mesh_remeshed.summary.json
<model_dir>/remesh/local/debug_meshes/
```

Operation JSONL rows should include:

```text
operation type
accepted/rejected
affected vertices/faces
normalized errors
reason for rejection
face/edge policy values
```

Validation metrics:

- face and vertex counts before/after;
- area change;
- edge-length quantiles;
- accepted/rejected collapses, splits, flips;
- protected boundary violations avoided;
- projection error percentiles;
- normal deviation percentiles;
- planar proxy residual before/after;
- support-weighted Hausdorff or point-to-surface error;
- per-view edge and normal overlays.

## 15. Training Integration

Only integrate into training after the post-hoc local remesher is stable.

The insertion point is OMeGa's current topology event:

```text
MeshGSStrategy.step_post_backward
  -> _split_meshes(...)
```

Target training behavior:

```text
write mesh_before_remesh.ply
run shared post-hoc local remesh operator with training-safe config
run guard checks
write mesh_after_remesh.ply
reparent existing splats to new faces
reset mesh and face-local optimizer moments
write preview and summary
continue optimization
```

Training-safe restrictions:

- no aggressive hole filling;
- no creation of large unsupported surfaces;
- no operation that leaves too few faces with splat support;
- no topology repair that destroys protected boundaries without evidence;
- keep splat identity, opacity, and appearance rows unchanged;
- initially single-GPU only.

The current `omega_local/remesh/training.py` reparenting logic is valuable and
should be reused. The backend it calls must change from face-count QEM to the
shared local remesh pipeline.

## 16. Short Execution Order

Work in this order and do not move to the next phase until the visualization
gate passes:

1. Preprocess the DSLR stronger OMeGa mesh.
2. Render view buffers and extract view-normal evidence.
3. Build policy fields and planar proxies.
4. Run local remesh operations, first as a dry run and then mutating topology.
5. Integrate the same remesher into OMeGa training after post-hoc validation.

Acceptance for the first complete post-hoc version:

- it runs on `$OMEGA_SOURCE_MESH`;
- it uses `$PYTHON` and writes under `$OMEGA_RESULT_DIR/remesh/local/`;
- it runs without `target_faces`;
- planar walls simplify until tolerance gates stop them;
- high normal-variation details are preserved or split;
- proxy boundaries and silhouettes are protected;
- debug outputs explain every major decision;
- every collapse/split/flip decision can be traced to normalized gate values.
