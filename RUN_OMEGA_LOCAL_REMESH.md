# OMeGa Local Remesh Pipeline Commands

This runbook is for the redesigned view-informed local remesh pipeline. It is
parallel to `RUN_OMEGA_REMESH.md`, but it does not use the old `target_faces`
QEM path. The main extraction commands below are full pipeline commands for the
current implemented stages. Visualization commands intentionally sample 32
frames for inspection.

The current development target is the DSLR PINHOLE StableNormal package and the
stronger OMeGa mesh:

```text
data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/omega_stable_mesh/model_baseline_stronger_30000/plys/mesh_29999_rank0.ply
```

This file keeps the current main path only. Earlier side tests and deprecated
ablation commands are intentionally left out of the run sequence.

## 0. Environment

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export OUT_ROOT="$DT_ROOT/data/scan_processing_outputs"
export PYTHON="$DT_ROOT/.venv/bin/python"
export OMEGA_BUILDING_ROOT="$DT_ROOT/third_party/OMeGa_4_Building"

export PACKAGE_ROOT="$OUT_ROOT/grove_entrance_dslr_0521_pinhole"
export OMEGA_RESULT_DIR="$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000"
```

The scripts infer the rest from `$OMEGA_RESULT_DIR`:

```text
source mesh:          latest $OMEGA_RESULT_DIR/plys/mesh_*_rank0.ply
prepared dataset:     cfg.json/data_dir or sibling dataset directory
local remesh outputs: $OMEGA_RESULT_DIR/remesh/local/
view buffers:         $OMEGA_RESULT_DIR/remesh/view_buffers/
visualizations:       $PACKAGE_ROOT/visualizations/
```

## 1. Phase 1: Preprocess The Mesh

Safety cleanup only: exact duplicate vertices/faces, zero-area faces, isolated
vertices, and bow-tie one-ring fan splits. It must not fill holes, smooth, snap
nearby vertices, delete components, or make the scene watertight.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_view_informed_remesh.py" \
  --stage preprocess \
  --model-dir "$OMEGA_RESULT_DIR" \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/local/preclean_mesh.ply
$OMEGA_RESULT_DIR/remesh/local/preclean_summary.json
$OMEGA_RESULT_DIR/remesh/local/preclean_diagnostics.npz
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/preclean_changed_faces.ply
```

Inspect before continuing:

- face/vertex count changes are explained by `preclean_summary.json`;
- `faceStatusCounts` shows only safety removals;
- `vertices_with_multiple_fans` should not increase;
- holes and open boundaries are preserved;
- no component deletion has happened.

## 2. Visualize Phase 1

`VisOmega01` renders the original and precleaned meshes from selected prepared
OMeGa camera views as white wireframes. It also renders the exact face-status
diagnostics from the preprocess `.npz`, so the visualization shows what the
preprocessing script actually did instead of inferring changes afterward.

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega01_visualize_omega_preprocess.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --frame-stride 4 \
  --max-frames 32 \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --overwrite
```

Outputs:

```text
$PACKAGE_ROOT/visualizations/omega_preprocess/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_preprocess/$(basename "$OMEGA_RESULT_DIR")/summary.json
$PACKAGE_ROOT/visualizations/omega_preprocess/$(basename "$OMEGA_RESULT_DIR")/frames/*/comparison_panel.png
```

What to inspect:

- original and preclean white wireframes should match except at true cleanup
  locations;
- removed-face overlay should be sparse or empty on a healthy OMeGa mesh;
- fan-split overlay should only appear around bow-tie one-ring vertices;
- preclean normal coverage should remain close to original mesh coverage.

## 2H. Optional Phase 1 Healing Test

This is a Phase 1 diagnostic pass, not the default input to later phases. Later
commands still consume `preclean_mesh.ply` unless `--mesh` is passed explicitly.
The CGAL backend repairs lightweight defects, stitches exactly coincident
boundary cracks, fills small boundary cycles, and reports exact
self-intersection counts. Self-intersection repair is off by default because it
can delete and recreate patches.

Build the CGAL helper once:

```bash
bash "$OMEGA_BUILDING_ROOT/scripts/build_cgal_mesh_heal.sh"
```

Run conservative healing. Defaults are: `--heal-max-hole-edges 48`,
`--heal-max-hole-diameter-factor 6.5`, degenerate-face repair on, non-manifold
vertex duplication on, exact border stitching on, hole filling on, tiny
component deletion off, and self-intersection repair off.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_view_informed_remesh.py" \
  --stage heal_preprocess \
  --model-dir "$OMEGA_RESULT_DIR" \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/local/preclean_healed_mesh.ply
$OMEGA_RESULT_DIR/remesh/local/preclean_healed_summary.json
$OMEGA_RESULT_DIR/remesh/local/preclean_healed_diagnostics.npz
$OMEGA_RESULT_DIR/remesh/local/cgal_mesh_heal_summary.json
$OMEGA_RESULT_DIR/remesh/local/preclean_healed_holes.jsonl
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/healing/preclean_healed_filled_patches.ply
```

Visualize preclean versus healed:

The panel is a two-column before/after grid. Rows show wireframe, rendered
normals, boundary-adjacent faces, and operation status. For healing runs, the
operation-status after column shows only inserted hole-fill patches.

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega01_visualize_omega_preprocess.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --original-mesh "$OMEGA_RESULT_DIR/remesh/local/preclean_mesh.ply" \
  --preclean-mesh "$OMEGA_RESULT_DIR/remesh/local/preclean_healed_mesh.ply" \
  --summary-json "$OMEGA_RESULT_DIR/remesh/local/preclean_healed_summary.json" \
  --diagnostics-npz "$OMEGA_RESULT_DIR/remesh/local/preclean_healed_diagnostics.npz" \
  --output-dir "$PACKAGE_ROOT/visualizations/omega_preprocess_healing/$(basename "$OMEGA_RESULT_DIR")" \
  --frame-stride 4 \
  --max-frames 32 \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --overwrite
```

Inspect:

- `preclean_healed_summary.json` before/after stats for boundary edges,
  connected components, non-manifold vertices, degenerate faces, and
  self-intersection pairs;
- `preclean_healed_holes.jsonl` for every boundary cycle's edge count,
  diameter, gate decision, and patch size;
- `debug_meshes/healing/preclean_healed_filled_patches.ply` to inspect only the
  faces inserted by hole filling;
- whether small accidental holes close without bridging large openings;
- whether stitched boundaries remove duplicate border cracks;
- whether the healed wireframe changes only local defects rather than reshaping
  valid architecture;
- whether optional `--heal-repair-self-intersections` helps or is too
  destructive on folded OMeGa regions.

## 3. Phase 2: Extract Full View-Normal Evidence

This stage does not remesh. It renders the precleaned mesh into all selected
OMeGa parser frames, computes StableNormal image-space derivatives and
tolerances, and transfers normalized evidence to mesh faces.

`--frame-stride 1 --max-frames 0` means use the full parser frame set. The
buffer width cap keeps the DSLR run manageable; set `--max-buffer-width 0` only
when you want full prepared-camera resolution.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_mesh_evidence.py" \
  "$OMEGA_RESULT_DIR" \
  --frame-stride 1 \
  --max-frames 0 \
  --max-buffer-width 1600 \
  --local-variance-radius-px 2 \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/local/evidence.npz
$OMEGA_RESULT_DIR/remesh/local/evidence_summary.json
$OMEGA_RESULT_DIR/remesh/local/evidence_frames.jsonl
$OMEGA_RESULT_DIR/remesh/view_buffers/frames/*.npz
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/evidence/*.ply
```

Inspect before continuing:

- rendered face-id and depth buffers align with the prepared DSLR RGB views;
- normal high-gradient pixels appear on real ridges/creases, not everywhere;
- offset search bands are plausible and not swallowing whole frames;
- `face_support` is high only where enough rendered pixels have valid normal
  evidence;
- unsupported faces and high disagreement regions are visible in the debug maps.

## 4. Visualize Phase 2

`VisOmega02` reads the exact Stage 2 `.npz` view buffers. It does not rerender
face-id/depth buffers, so the panels show what the extractor actually wrote.
It samples 32 manifest frames for review; the evidence itself is still the full
run from Phase 2.

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega02_visualize_omega_normal_evidence.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Outputs:

```text
$PACKAGE_ROOT/visualizations/omega_normal_evidence/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_normal_evidence/$(basename "$OMEGA_RESULT_DIR")/summary.json
$PACKAGE_ROOT/visualizations/omega_normal_evidence/$(basename "$OMEGA_RESULT_DIR")/frames/*/comparison_panel.png
```

What to inspect:

- RGB and prepared normal map should match the expected camera crop;
- face-id/depth buffers should cover the mesh where it is visible;
- mesh edge/silhouette mask should line up with depth/face boundaries;
- high-gradient normal pixels should be sparse and structured;
- original and local normal-gradient maps should highlight the same structures,
  with local maps acting as the cleaner guide source for Instant Meshes;
- offset search visualization should explain how shifted ridges can still reach
  nearby rendered faces;
- face support, kappa, disagreement, target length, and direction confidence
  should be coherent enough to feed the policy stage.

## 5. Phase 3: Build Policy Weights And Planar Proxies

This stage consumes the full Phase 2 evidence. Support and mesh planarity drive
plane/proxy growth. Stable normal detail, target length, and direction
confidence drive detail preservation. Supported residual offset/disagreement
becomes feature/detail pressure. `face_weight_uncertain` is only a low-support
and no-explanation fallback; it should not freeze stable stairs, arches, or
planar surfaces. The final Phase 3 output includes the four remesh weights
consumed by Phase 4: `face_weight_planar`, `face_weight_detail`,
`face_weight_boundary`, and `face_weight_uncertain`, plus edge/vertex versions.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_remesh_policy.py" \
  "$OMEGA_RESULT_DIR" \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/local/policy.npz
$OMEGA_RESULT_DIR/remesh/local/policy_summary.json
$OMEGA_RESULT_DIR/remesh/local/weights_summary.json
$OMEGA_RESULT_DIR/remesh/local/proxies.json
$OMEGA_RESULT_DIR/remesh/local/proxies.npz
$OMEGA_RESULT_DIR/remesh/local/proxies_summary.json
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/policy/*.ply
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/weights/*.ply
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/proxies/*.ply
```

Inspect before continuing:

- planar walls/floors/ceilings should have high `face_planar_score`, even when
  foreground clutter is projected onto them;
- true stable details and creases should have high `face_detail_posterior`;
- railing-on-wall projection errors should appear as `face_unexplained_offset`,
  not as clean wall detail or planar-proxy holes;
- `face_boundary_score` should protect real creases, silhouettes, proxy
  transitions, and mesh boundaries, not missing foreground projections;
- `face_weight_planar` should be strongest on supported planar surfaces;
- `face_weight_detail` should be strongest where stable normal detail should be
  preserved or aligned, including offset evidence that overlaps stable detail;
- `face_weight_boundary` should mark real seams, corners, and proxy
  transitions as a softened face diagnostic. The stronger operation signal is
  `edge_weight_boundary`; mesh boundaries remain separate topology constraints,
  so this weight should not be high everywhere merely because the mesh has many
  open boundaries;
- `face_weight_uncertain` should rise on high disagreement, unexplained
  foreground projection, or low support only when no stable planar/detail
  explanation exists. In the current simplification-first policy, it should be
  much sparser than detail/feature pressure;
- `face_offset_detail_pressure` should identify missing/underfit geometry that
  should be preserved post-hoc and can become split/refinement pressure during
  OMeGa training;
- `face_residual_unexplained_offset` should stay high for projection artifacts
  such as foreground clutter landing on a planar wall;
- `face_stable_explanation`, `face_low_support_uncertainty`, and
  `face_disagreement_uncertainty` explain why a region became uncertain;
- proxy regions should be large on planar surfaces and stop at real geometry
  changes.

## 6. Visualize Phase 3

`VisOmega03` samples 32 Stage 2 view-buffer frames and paints the Phase 3 face
fields, final remesh weights, and proxy assignments through the same face-id
buffers.

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega03_visualize_omega_policy_proxies.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Outputs:

```text
$PACKAGE_ROOT/visualizations/omega_policy_proxies/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_policy_proxies/$(basename "$OMEGA_RESULT_DIR")/summary.json
$PACKAGE_ROOT/visualizations/omega_policy_proxies/$(basename "$OMEGA_RESULT_DIR")/frames/*/comparison_panel.png
```

What to inspect:

- `Detail posterior` should highlight stable detail from kappa, target length,
  and direction confidence, not missing foreground projections;
- `Unexplained offset` should catch inconsistent projected structures such as
  railings missing from the mesh;
- `Planar score` should remain strong on clean planar regions;
- `Proxy id` should form coherent large components;
- `Proxy residual` should be low inside each plane and rise near incompatible
  geometry;
- `Protect detail` should follow real remesh detail;
- `Protect missing geometry` should follow missing/foreground structures as a
  separate OMeGa-optimization cue;
- `Boundary score` should explain where the next remesh stage must avoid
  crossing real features.

## 7. Phase 4A: Propose Local Operations

This stage still does not mutate the mesh. It computes final proxy-aware target
lengths, builds collapse/split/flip proposal pressure, and evaluates detailed
dry-run gates for the strongest candidates. The output explains what the local
remesher would try before actual topology edits are enabled.

For the post-hoc designer mesh, split pressure is diagnostic by default. It
shows where OMeGa training might need more geometry or where an existing mesh
is undersampled relative to normal evidence, but accepted split proposals are
disabled unless `--enable-split-proposals` is explicitly passed.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_view_informed_remesh.py" \
  --stage proposals \
  --model-dir "$OMEGA_RESULT_DIR" \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/local/operation_proposals.npz
$OMEGA_RESULT_DIR/remesh/local/operation_proposals.jsonl
$OMEGA_RESULT_DIR/remesh/local/operation_proposals.summary.json
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/operations/*.ply
```

Inspect before continuing:

- collapse pressure should concentrate inside confident planar proxies and on
  low-detail oversampled edges;
- split pressure should concentrate on supported high-detail undersampled
  regions, but accepted split maps should remain empty in the default post-hoc
  simplification command;
- flip pressure should be sparse and tied to quality/alignment improvements;
- missing foreground structures should produce rejection/protection, not wall
  detail collapse;
- accepted dry-run proposals should not cross protected boundaries.

## 8. Visualize Phase 4A

`VisOmega04` samples 32 Stage 2 view-buffer frames and paints the dry-run
operation proposal fields through the same face-id buffers.

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega04_visualize_omega_operation_proposals.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Outputs:

```text
$PACKAGE_ROOT/visualizations/omega_operation_proposals/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_operation_proposals/$(basename "$OMEGA_RESULT_DIR")/summary.json
$PACKAGE_ROOT/visualizations/omega_operation_proposals/$(basename "$OMEGA_RESULT_DIR")/frames/*/comparison_panel.png
```

What to inspect:

- `Operation pressure` should be localized, not spread uniformly across the
  scene;
- `Collapse pressure` should mostly hit planar interiors;
- `Split pressure` should mostly hit supported detail regions and is diagnostic
  in the default command;
- accepted dry-run maps should be stricter than pressure maps;
- rejection maps should explain why protected, missing-geometry, low-quality,
  or topology-risky edges are not accepted.

## 9. Phase 4B: Run Weighted Local Remesh

This is the topology-mutating weighted remesh pass. It consumes the Phase 3
weights and planar proxies, first snaps confident planar vertices toward their
proxy planes, then performs batched edge collapses, quality-driven edge flips,
and bounded final relax/project. It does not split or densify the mesh.
Proxy-boundary feature edges may collapse along the two-plane intersection so a
stair tread/riser seam can simplify into a cleaner sharp edge without allowing
cross-seam collapses.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_view_informed_remesh.py" \
  --stage remesh \
  --model-dir "$OMEGA_RESULT_DIR" \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/local/mesh_weighted_remesh.ply
$OMEGA_RESULT_DIR/remesh/local/mesh_weighted_remesh_summary.json
$OMEGA_RESULT_DIR/remesh/local/mesh_weighted_remesh_report.md
$OMEGA_RESULT_DIR/remesh/local/weighted_remesh_diagnostics.npz
$OMEGA_RESULT_DIR/remesh/local/weighted_remesh_operations.jsonl
$OMEGA_RESULT_DIR/remesh/local/debug_meshes/weighted_remesh/*.ply
```

Inspect before continuing:

- face count should decrease mostly on confident planar interiors;
- pre-projection should move supported proxy interiors toward cleaner fitted
  planes before collapse decisions are made;
- planar transition pressure should appear on one-sided planar fringe regions
  where the remesher is allowed to clean edges adjacent to planes without
  crossing mesh boundaries or proxy-to-proxy boundaries;
- feature-edge collapse pressure should appear on proxy-to-proxy planar seams
  such as stair tread/riser intersections. Accepted collapses there should
  reduce seam segment count while preserving the sharp intersection;
- accepted flips should improve triangle quality/connectivity without crossing
  protected boundaries;
- `Detail simplify pressure` should appear where detail regions are locally
  oversampled relative to the adaptive detail target from normal evidence and
  the source mesh scale;
- detail relaxation should appear on supported detail regions that need mesh
  cleanup but not simplification;
- accepted collapses should not cross mesh boundaries, high boundary weights,
  proxy boundaries, or high uncertainty;
- removed source faces should be sparse in detail regions;
- projection rejections should appear where proxy snapping would move vertices
  too far;
- quality/normal rejections should explain local geometry that was too risky
  to collapse.
- `mesh_weighted_remesh_report.md` gives the pass-by-pass aggregate behavior:
  pre-projection counts, collapse candidates, accepted collapses/flips, reject
  counts, pressure summaries, and final relax/project movement.

## 10. Visualize Phase 4B

`VisOmega05` samples 32 Stage 2 view-buffer frames and plots Phase 4B in this
order: before mesh, after mesh, edge difference, accepted operation changes,
operation pressure, input weights, and rejection gates. Most diagnostic maps are
painted through the original preclean source-face ids. The after/remesh edge
panel is the final remeshed `.ply` rerendered into the same selected views.

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega05_visualize_omega_weighted_remesh.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Outputs:

```text
$PACKAGE_ROOT/visualizations/omega_weighted_remesh/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_weighted_remesh/$(basename "$OMEGA_RESULT_DIR")/summary.json
$PACKAGE_ROOT/visualizations/omega_weighted_remesh/$(basename "$OMEGA_RESULT_DIR")/frames/*/comparison_panel.png
```

What to inspect:

- `Before: source mesh edges` is the preclean mesh used by the remesher;
- `After: remesh mesh edges` is the actual final remeshed mesh;
- `Edge delta` compares the two edge renders: red is before-only, cyan is
  after-only, and white is overlap;
- `Actual changes` explains the accepted edits on source faces: red means
  removed by collapse, blue means planar projection, green means detail
  relaxation, and cyan means edge flips;
- `Collapse pressure` should follow planar simplification opportunities;
- `Accepted collapse` should be stricter than pressure;
- `Detail simplify pressure` should show high-detail regions that can be
  decimated without violating local evidence;
- `Planar transition pressure` should show plane-adjacent fringe cleanup
  pressure, distinct from ordinary interior planar collapse;
- `Feature-edge collapse pressure` should show simplification pressure along
  sharp planar proxy seams, distinct from crossing or deleting the seam;
- `Accepted flip` should be sparse and tied to triangle quality cleanup;
- `Planar projected` and `Planar projection distance` should be strongest on
  confident proxy interiors;
- `Detail relaxed` and `Detail relax distance` should appear in detail regions
  where the mesh is cleaned without removing faces;
- `Removed source faces` should mostly sit inside simplified planar regions;
- `Target edge length used by remesher` is the sizing field used to decide
  whether an edge is short or long locally. It is not a before/after comparison;
- `W planar/detail/boundary/uncertain` should explain the accepted or rejected
  operation pattern;
- rejection maps should make risky areas legible rather than mysterious.

## 11. Baseline Remesh Experiments

Baseline experiments are comparison runs, not part of the production local
remesher. They all consume the same preclean OMeGa mesh and write into one
canonical tree:

```text
$OMEGA_RESULT_DIR/remesh/baselines/<experiment_name>/
$PACKAGE_ROOT/visualizations/omega_remesh_baselines/<experiment_name>/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_remesh_geogram_density/<experiment_name>/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_remesh_instant_fields/<experiment_name>/$(basename "$OMEGA_RESULT_DIR")/
```

Use `--experiment-name` for all new runs. `--output-dir`, `--baseline-summary`,
and `--density-npz` are override knobs only; ordinary runs should not need
them.

### 11.1 Build External Baseline Binaries

Instant Meshes is patched in place but keeps original behavior unless an OMeGa
field sidecar is passed:

```bash
cd "$DT_ROOT/third_party"
test -d instant-meshes/.git || git clone --recursive https://github.com/wjakob/instant-meshes.git instant-meshes

mkdir -p instant-meshes/deps/debs instant-meshes/deps/sysroot
cd instant-meshes/deps/debs
apt-get download libxinerama-dev libxxf86vm-dev
cd "$DT_ROOT/third_party"
dpkg-deb -x instant-meshes/deps/debs/libxinerama-dev_*_amd64.deb instant-meshes/deps/sysroot
dpkg-deb -x instant-meshes/deps/debs/libxxf86vm-dev_*_amd64.deb instant-meshes/deps/sysroot

cmake -S instant-meshes -B instant-meshes/build-codex \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS="-fpermissive -Wno-changes-meaning" \
  -DCMAKE_CXX_STANDARD_LIBRARIES="-lXext" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$DT_ROOT/third_party/instant-meshes/deps/sysroot/usr/lib/x86_64-linux-gnu" \
  -DX11_Xinerama_INCLUDE_PATH="$DT_ROOT/third_party/instant-meshes/deps/sysroot/usr/include" \
  -DX11_Xinerama_LIB="$DT_ROOT/third_party/instant-meshes/deps/sysroot/usr/lib/x86_64-linux-gnu/libXinerama.so" \
  -DX11_Xxf86vm_INCLUDE_PATH="$DT_ROOT/third_party/instant-meshes/deps/sysroot/usr/include" \
  -DX11_Xxf86vm_LIB="$DT_ROOT/third_party/instant-meshes/deps/sysroot/usr/lib/x86_64-linux-gnu/libXxf86vm.so"

cmake --build instant-meshes/build-codex -j"$(nproc)"
ln -sf "Instant Meshes" instant-meshes/build-codex/InstantMeshes
```

Geogram/Vorpalite is built headless:

```bash
cd "$DT_ROOT/third_party"
test -d geogram/.git || git clone --recurse-submodules https://github.com/BrunoLevy/geogram.git geogram

cmake -S geogram -B geogram/build-codex-nogfx \
  -DCMAKE_BUILD_TYPE=Release \
  -DVORPALINE_PLATFORM=Linux64-gcc-dynamic \
  -DGEOGRAM_WITH_GRAPHICS=OFF \
  -DGEOGRAM_WITH_EXPLORAGRAM=OFF \
  -DGEOGRAM_WITH_LUA=OFF

cmake --build geogram/build-codex-nogfx -j"$(nproc)"
```

Both binaries are auto-discovered from these build folders.

### 11.2 Organized Baseline Coverage

Run the key baseline methods in four comparison groups. Together these groups
cover the methods we still use for analysis while keeping each visualization
focused enough to read. All matched comparison runs use the same target face
ratios:

```text
0.20, 0.40, 0.60
```

Method coverage:

```text
qem_compare:
  open3d_qem, meshlab_qem, meshlab_planar_qem

isotropic_compare:
  meshlab_isotropic, meshlab_feature_isotropic,
  meshlab_isotropic_qem, meshlab_feature_isotropic_qem

instant_guided_compare:
  instant_meshes_quad, instant_meshes_quad_guided,
  instant_meshes_tri, instant_meshes_tri_guided,
  instant_meshes_dominant, instant_meshes_dominant_guided
  guided rows use StableNormal/OMeGa 3D tangent guides with combined
  point/local normal evidence

geogram_compare:
  geogram_anisotropic, geogram_anisotropic_gradation,
  geogram_omega_density, geogram_omega_density_gradation
```

### 11.3 QEM Simplification Baselines

This compares the strongest pure simplification baselines: Open3D QEM, MeshLab
QEM, and MeshLab planar QEM.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_remesh_baselines.py" \
  "$OMEGA_RESULT_DIR" \
  --method-preset qem \
  --experiment-name qem_compare \
  --target-face-ratios 0.20,0.40,0.60 \
  --overwrite
```

Visualize:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega06_compare_omega_remesh_baselines.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --experiment-name qem_compare \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 1040 \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/baselines/qem_compare/baseline_summary.json
$OMEGA_RESULT_DIR/remesh/baselines/qem_compare/baseline_manifest.jsonl
$OMEGA_RESULT_DIR/remesh/baselines/qem_compare/*/mesh.ply
$PACKAGE_ROOT/visualizations/omega_remesh_baselines/qem_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/method_grid.png
```

### 11.4 MeshLab Isotropic Baselines

This compares pure isotropic remeshing, feature-sensitive isotropic remeshing,
and the same two passes followed by QEM. Use this group to separate "uniform
triangle quality" from "structure-preserving simplification."

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_remesh_baselines.py" \
  "$OMEGA_RESULT_DIR" \
  --method-preset isotropic \
  --experiment-name isotropic_compare \
  --target-face-ratios 0.20,0.40,0.60 \
  --feature-isotropic-degrees 15 \
  --no-feature-isotropic-smooth \
  --overwrite
```

Visualize:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega06_compare_omega_remesh_baselines.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --experiment-name isotropic_compare \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 1040 \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/baselines/isotropic_compare/baseline_summary.json
$OMEGA_RESULT_DIR/remesh/baselines/isotropic_compare/baseline_manifest.jsonl
$OMEGA_RESULT_DIR/remesh/baselines/isotropic_compare/*/mesh.ply
$PACKAGE_ROOT/visualizations/omega_remesh_baselines/isotropic_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/method_grid.png
```

### 11.5 Instant Meshes Original Vs OMeGa-Guided

This compares original Instant Meshes against the patched OMeGa soft
orientation constraints for quads, triangles, and dominant-mode output. Read
the final mesh grid and the field grid together.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_remesh_baselines.py" \
  "$OMEGA_RESULT_DIR" \
  --method-preset instant_guided \
  --experiment-name instant_guided_compare \
  --target-face-ratios 0.20,0.40,0.60 \
  --instant-meshes-crease-degrees 25 \
  --instant-meshes-smooth-iterations 2 \
  --instant-field-weight-scale 0.7 \
  --instant-field-pixel-stride 1 \
  --instant-field-gradient-mode combined \
  --instant-field-min-gradient-normalized 0.35 \
  --overwrite
```

Visualize output meshes:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega06_compare_omega_remesh_baselines.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --experiment-name instant_guided_compare \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 1040 \
  --overwrite
```

Visualize fields and constraints:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega07_visualize_instant_field.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --experiment-name instant_guided_compare \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 1040 \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/baselines/instant_guided_compare/omega_instant_field/omega_instant_field.txt
$OMEGA_RESULT_DIR/remesh/baselines/instant_guided_compare/*/instant_fields/orientation_optimized.txt
$PACKAGE_ROOT/visualizations/omega_remesh_baselines/instant_guided_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/method_grid.png
$PACKAGE_ROOT/visualizations/omega_remesh_instant_fields/instant_guided_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/omega_input_Q_grid.png
$PACKAGE_ROOT/visualizations/omega_remesh_instant_fields/instant_guided_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/orientation_Q_grid.png
$PACKAGE_ROOT/visualizations/omega_remesh_instant_fields/instant_guided_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/constraint_CQ_grid.png
```

`omega_input_Q_grid.png` is the Python-exported 3D StableNormal/OMeGa guide
before Instant Meshes blends it with boundary or mesh-feature constraints.
`constraint_CQ_grid.png` is the C++ constraint state after loading/blending.
`orientation_Q_grid.png` is the solved orientation field.

### 11.6 Geogram And OMeGa-Guided Density

This compares plain Geogram, Geogram LFS gradation, OMeGa density, and OMeGa
planar-gradated density at matched target ratios.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_remesh_baselines.py" \
  "$OMEGA_RESULT_DIR" \
  --method-preset geogram \
  --experiment-name geogram_compare \
  --target-face-ratios 0.20,0.40,0.60 \
  --geogram-anisotropy 1.0 \
  --geogram-gradation 1.0 \
  --geogram-density-min 0.25 \
  --geogram-density-max 4.0 \
  --geogram-density-signal-gamma 1.25 \
  --geogram-density-detail-influence 1.0 \
  --geogram-density-boundary-influence 0.5 \
  --geogram-density-uncertain-influence 0.0 \
  --geogram-density-planar-gradation-iterations 12 \
  --geogram-density-planar-gradation-strength 0.45 \
  --geogram-density-planar-gradation-normal-lift-scale 0 \
  --geogram-density-planar-gradation-metric-tau-factor 2.0 \
  --geogram-density-planar-gradation-min-normal-gate 0.05 \
  --geogram-density-planar-gradation-max-density-ratio 1.75 \
  --overwrite
```

Visualize meshes and density data:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisOmega08_visualize_geogram_density.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --experiment-name geogram_compare \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 1040 \
  --overwrite
```

Outputs:

```text
$OMEGA_RESULT_DIR/remesh/baselines/geogram_compare/omega_geogram_density/omega_geogram_density.npz
$OMEGA_RESULT_DIR/remesh/baselines/geogram_compare/omega_geogram_density_planar_gradation/omega_geogram_density_planar_gradation.npz
$PACKAGE_ROOT/visualizations/omega_remesh_geogram_density/geogram_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/density_data_panel.png
$PACKAGE_ROOT/visualizations/omega_remesh_geogram_density/geogram_compare/$(basename "$OMEGA_RESULT_DIR")/frames/*/method_grid.png
```

### 11.7 What To Inspect

- Each VisOmega06 row compares methods at the same target ratio.
- Each VisOmega06 column shows one method across increasing target ratios.
- Good results should keep stairs/corners legible, simplify large planes, and
  avoid uniform triangle fields that blur architectural seams or sculptural
  curvature.
- In VisOmega08, `W planar` should be bright where Geogram should sample less;
  `Density signal` and `Geogram density` should be bright around details,
  ridges, and protected structures.
- In VisOmega07, `constraint_CQ_grid.png` is the supplied OMeGa guide, while
  `orientation_Q_grid.png` is the solved Instant Meshes field.

## 12. Future Integration

After one of the Phase 4 remesh paths passes visually, continue in this order:

1. tune the remesh thresholds against the VisOmega05 before/after edge panels;
2. add a conservative final cleanup pass if the remeshed mesh has local
   degeneracies or isolated artifacts;
3. integrate the remesh step into OMeGa training at the subdivision insertion
   point;
4. add training-time diagnostics for the same weights and accepted operations.

Each phase should get its own visualization gate before the next phase mutates
or consumes its outputs.
