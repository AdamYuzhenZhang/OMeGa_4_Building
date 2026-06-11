# OMeGa Remesh Commands

This runbook keeps OMeGa and remesh logic in the OMeGa checkout, while
scan-processing visualization scripts stay in `scan_processing/`.

The concrete dataset below is the current DSLR StableNormal package:

```text
/home/yz2332/projects/digitalTwin/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/omega_stable_mesh
```

If the OMeGa DSLR run is already training or already complete, start at
`4. Post-Hoc Remesh`. Do not rerun the data-preparation command with
`--overwrite` against an active training run.

## 0. Environment

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export OUT_ROOT="$DT_ROOT/data/scan_processing_outputs"
export PYTHON="$DT_ROOT/.venv/bin/python"
export OMEGA_BUILDING_ROOT="$DT_ROOT/third_party/OMeGa_4_Building"
export COLMAP_BIN="$DT_ROOT/third_party/colmap_cuda/install/bin/colmap"
export STABLENORMAL_ROOT="$DT_ROOT/third_party/StableNormal"

export DSLR_CAPTURE="$DT_ROOT/data/camera/grove_entrance_dslr_0521"
export DSLR_NAME="$(basename "$DSLR_CAPTURE")"
export DSLR_PINHOLE_NAME="${DSLR_NAME}_pinhole"
export DSLR_PINHOLE_PACKAGE_ROOT="$OUT_ROOT/$DSLR_PINHOLE_NAME"

export PACKAGE_ROOT="$DSLR_PINHOLE_PACKAGE_ROOT"
export OMEGA_OUTPUT_NAME=omega_stable_mesh
export OMEGA_ROOT="$PACKAGE_ROOT/$OMEGA_OUTPUT_NAME"
export OMEGA_DATASET_DIR="$OMEGA_ROOT/dataset"
export OMEGA_INIT_MESH="$OMEGA_ROOT/init_mesh.ply"
export OMEGA_WARMUP_DIR="$OMEGA_ROOT/model_warmup_2999"
export OMEGA_RESULT_DIR="$OMEGA_ROOT/model_baseline_stronger_30000"
export OMEGA_MESH_PREVIEW_FRAME=0
```

## 1. Prepare OMeGa DSLR Data

This writes the OMeGa package to `$OMEGA_ROOT`. StableNormal provides the normal
maps, and MapAnything provides the coarse initializer.

```bash
$PYTHON "$DT_ROOT/scan_processing/23_run_phone_omega.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --omega-output-name "$OMEGA_OUTPUT_NAME" \
  --colmap-bin "$COLMAP_BIN" \
  --normal-source stablenormal \
  --point-cloud-source mapanything \
  --seed-voxel-size-m 0.05 \
  --max-init-mesh-points 180000 \
  --init-normal-radius-m 0.12 \
  --poisson-depth 7 \
  --poisson-density-quantile 0.12 \
  --init-mesh-cleaning pymeshlab \
  --init-mesh-min-component-faces 200 \
  --init-mesh-close-holes-size 120 \
  --overwrite
```

Key outputs:

```text
$OMEGA_DATASET_DIR
$OMEGA_DATASET_DIR/normal_maps/
$OMEGA_DATASET_DIR/normal_aux/
$OMEGA_INIT_MESH
$OMEGA_ROOT/dataset_manifest.json
```

## 2. Preview Prepared Inputs

These are scan-processing visualizations. They read the prepared OMeGa package;
they do not change the OMeGa data or training outputs.

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24b_visualize_phone_omega_input_normals.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name "$OMEGA_OUTPUT_NAME" \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis24_visualize_phone_omega_mesh_debug.py" \
  "$DSLR_PINHOLE_PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-output-name "$OMEGA_OUTPUT_NAME" \
  --mesh-source init \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

## 3. Train OMeGa

The warmup checkpoint is the reusable base for later branches. The baseline run
continues from that warmup and writes to `$OMEGA_RESULT_DIR`.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_warmup.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/warmup_stable_mesh_2999.json" \
  --mesh-preview-frame "$OMEGA_MESH_PREVIEW_FRAME"
```

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/baseline_stable_mesh_stronger_from_warmup_30000.json" \
  --set args.mesh_preview_frame="$OMEGA_MESH_PREVIEW_FRAME"
```

Key outputs:

```text
$OMEGA_WARMUP_DIR/ckpts/ckpt_2999_rank0.pt
$OMEGA_RESULT_DIR/ckpts/ckpt_29999_rank0.pt
$OMEGA_RESULT_DIR/plys/mesh_29999_rank0.ply
$OMEGA_RESULT_DIR/mesh_previews/
$OMEGA_RESULT_DIR/stats/loss_plots/
```

## 4. Post-Hoc Remesh

The current DSLR StableNormal stronger mesh has roughly `111k` faces, so the
example target below is intentionally below that count. For other runs, choose a
target at or below the source mesh face count.

```bash
export OMEGA_REMESH_TARGET_FACES=80000
export OMEGA_REMESH_MESH="$OMEGA_RESULT_DIR/remesh/mesh_posthoc_structure_qem_${OMEGA_REMESH_TARGET_FACES}.ply"

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_posthoc_remesh.py" \
  "$OMEGA_RESULT_DIR" \
  --target-faces "$OMEGA_REMESH_TARGET_FACES" \
  --out "$OMEGA_REMESH_MESH"
```

Visualize the original optimized mesh against the post-hoc remesh:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis26_visualize_phone_omega_remesh_debug.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --simplified-mesh "$OMEGA_REMESH_MESH" \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

## 5. Face-Count Sweep

Use this before committing to one target. Existing sweep meshes are reused unless
`--overwrite-remesh` is included.

```bash
export OMEGA_REMESH_SWEEP_FACES="30000 50000 80000 110000"

$PYTHON "$DT_ROOT/scan_processing/Vis27_visualize_phone_omega_remesh_sweep.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --target-faces $OMEGA_REMESH_SWEEP_FACES \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --error-vmax-m 0.05 \
  --overwrite-remesh \
  --overwrite
```

Key outputs:

```text
$OMEGA_RESULT_DIR/remesh/
$OMEGA_RESULT_DIR/remesh/sweep/
$PACKAGE_ROOT/visualizations/omega_remesh_preview/$(basename "$OMEGA_RESULT_DIR")/
$PACKAGE_ROOT/visualizations/omega_remesh_sweep/$(basename "$OMEGA_RESULT_DIR")/remesh_face_count_sweep_panel.png
```

## 6. Project Mesh Face Evidence

This projects the selected OMeGa dataset normal evidence onto the final mesh
faces. Normals are read from `$OMEGA_DATASET_DIR/normal_maps`, so this follows
the StableNormal/PromptDA choice made during data preparation. The current
normal-only evidence file keeps the cues we use downstream: view support,
normal variation, and normal-derived curvature/gradient. The evidence script
also writes target-length debug fields from the earlier experiment, but the
current minimal remesh policy does not consume target length.

```bash
export OMEGA_EVIDENCE_DIR="$OMEGA_RESULT_DIR/remesh/evidence"
export OMEGA_EVIDENCE_NPZ="$OMEGA_EVIDENCE_DIR/mesh_face_evidence.npz"

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_mesh_evidence.py" \
  "$OMEGA_RESULT_DIR" \
  --data-dir "$OMEGA_DATASET_DIR" \
  --out-dir "$OMEGA_EVIDENCE_DIR" \
  --frame-stride 1 \
  --max-frames 0 \
  --normal-patch-radius-px 5 \
  --normal-blur-radius-px 2 \
  --curvature-error-px 0.5 \
  --target-edge-min-px 2 \
  --target-edge-max-px 96 \
  --max-oblique-scale-multiplier 4 \
  --normal-gradient-scale 0.25 \
  --no-guides \
  --overwrite
```

Visualize the projected evidence:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis28_visualize_phone_omega_mesh_evidence.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --data-dir "$OMEGA_DATASET_DIR" \
  --evidence-npz "$OMEGA_EVIDENCE_NPZ" \
  --frame-stride 4 \
  --max-frames 32 \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --overwrite
```

Key outputs:

```text
$OMEGA_EVIDENCE_NPZ
$OMEGA_EVIDENCE_DIR/mesh_face_evidence_summary.json
$OMEGA_EVIDENCE_DIR/debug_meshes/
$PACKAGE_ROOT/visualizations/omega_mesh_evidence/$(basename "$OMEGA_RESULT_DIR")/
```


## 7. Convert Evidence Into Region Scores

This is Stage 2 of the remesh pipeline. It does not change the mesh. It now
uses a deliberately minimal two-region policy:

- plane-like region: low robust normal curvature/variation with enough normal
  support. These faces get high `face_planar_weight`, high
  `face_plane_project_weight`, and high `face_simplify_weight`;
- detail / non-planar region: high normal curvature, normal variation, mesh
  crease, or transition evidence. These faces get high `face_detail_weight` and
  high `face_protect_weight`.

RGB, Guide01 evidence, target edge length, and density gates are ignored in this
Stage-2 policy. The important arrays to inspect are
`face_normal_curvature_score`, `face_variation_score`, `face_gradient_score`,
`face_planar_weight`, `face_detail_weight`, `face_plane_project_weight`,
`face_simplify_weight`, and `face_protect_weight`.

```bash
export OMEGA_REGION_SCORES_NPZ="$OMEGA_EVIDENCE_DIR/mesh_region_scores.npz"

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_mesh_region_scores.py" \
  "$OMEGA_RESULT_DIR" \
  --evidence-npz "$OMEGA_EVIDENCE_NPZ" \
  --score-smooth-iterations 0 \
  --curvature-low-quantile 0.25 \
  --curvature-high-quantile 0.90 \
  --variation-low-quantile 0.50 \
  --variation-high-quantile 0.95 \
  --variation-detail-weight 0.35 \
  --planarity-variation-penalty 0.35 \
  --mesh-crease-detail-weight 0.35 \
  --min-normal-samples 2 \
  --target-view-support 0.05 \
  --region-detail-quantile 0.85 \
  --overwrite
```

Visualize the curvature/variation cues, plane-like/detail weights,
plane-projection mask, simplify/protect weights, and region IDs from camera
views:

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis29_visualize_phone_omega_region_scores.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --data-dir "$OMEGA_DATASET_DIR" \
  --region-scores-npz "$OMEGA_REGION_SCORES_NPZ" \
  --frame-stride 4 \
  --max-frames 32 \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --overwrite
```

Key outputs:

```text
$OMEGA_REGION_SCORES_NPZ
$OMEGA_EVIDENCE_DIR/mesh_region_scores_summary.json
$OMEGA_EVIDENCE_DIR/mesh_regions.json
$OMEGA_EVIDENCE_DIR/debug_meshes/
$PACKAGE_ROOT/visualizations/omega_mesh_region_scores/$(basename "$OMEGA_RESULT_DIR")/
```

## 8. Step 3b Plane Adjustment

This pass adjusts only confident plane-like surface components before QEM. It
uses Stage-2 plane/detail scores to find connected planar components, fits a
plane from the mesh geometry itself, then moves eligible low-detail vertices
onto that plane. It does not collapse edges or change topology.

```bash
export OMEGA_PLANE_ADJUSTED_MESH="$OMEGA_RESULT_DIR/remesh/mesh_plane_adjusted.ply"

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_plane_adjustment.py" \
  "$OMEGA_RESULT_DIR" \
  --scores-npz "$OMEGA_REGION_SCORES_NPZ" \
  --out "$OMEGA_PLANE_ADJUSTED_MESH" \
  --plane-threshold 0.60 \
  --detail-max 0.45 \
  --protect-max 0.65 \
  --min-component-faces 20 \
  --vertex-plane-fraction 0.55 \
  --fit-trim-quantile 0.90 \
  --fit-iterations 3 \
  --strength 1.0 \
  --max-vertex-move-m 0.25 \
  --overwrite
```

Key outputs:

```text
$OMEGA_PLANE_ADJUSTED_MESH
$OMEGA_RESULT_DIR/remesh/mesh_plane_adjusted.plane_adjustment.npz
$OMEGA_RESULT_DIR/remesh/mesh_plane_adjusted.plane_adjustment_summary.json
$OMEGA_RESULT_DIR/remesh/debug_meshes/plane_adjustment/
```

You can quickly compare the original mesh against the plane-adjusted mesh with
Vis30 by passing `--remesh-mesh "$OMEGA_PLANE_ADJUSTED_MESH"`.

## 9. View-Informed QEM Remesh

This is the simplification step that consumes the Stage-2 simplify/protect
weights. When Step 3b has already produced `$OMEGA_PLANE_ADJUSTED_MESH`, QEM
should use that mesh as input and disable its internal plane projection:

- high `face_simplify_weight` lowers vertex quality so plane-like interiors
  collapse more easily and can become large/oblique triangles;
- high `face_protect_weight`, detail score, uncertainty, mesh boundaries, and
  strong creases raise vertex quality so detailed/non-planar regions are
  preserved.

```bash
export OMEGA_VIEW_QEM_TARGET_FACES=20000
export OMEGA_VIEW_QEM_MESH="$OMEGA_RESULT_DIR/remesh/mesh_view_informed_qem_${OMEGA_VIEW_QEM_TARGET_FACES}.ply"

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_view_informed_remesh.py" \
  "$OMEGA_RESULT_DIR" \
  --input-mesh "$OMEGA_PLANE_ADJUSTED_MESH" \
  --scores-npz "$OMEGA_REGION_SCORES_NPZ" \
  --target-faces "$OMEGA_VIEW_QEM_TARGET_FACES" \
  --no-plane-project \
  --out "$OMEGA_VIEW_QEM_MESH" \
  --overwrite
```

Visualize the original optimized mesh against the view-informed remesh. This
comparison renders before/after edges and normals from the same camera views.
If the `.policy.npz` from the remesh step is present, it also renders the
simplify/protect/plane-projection/quality maps that drove QEM.

```bash
$PYTHON "$DT_ROOT/scan_processing/Vis30_visualize_phone_omega_view_informed_remesh.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_BUILDING_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --remesh-mesh "$OMEGA_VIEW_QEM_MESH" \
  --max-render-width 0 \
  --max-mesh-faces 0 \
  --frame-stride 4 \
  --max-frames 32 \
  --overwrite
```

Key outputs:

```text
$OMEGA_VIEW_QEM_MESH
$OMEGA_RESULT_DIR/remesh/mesh_view_informed_qem_${OMEGA_VIEW_QEM_TARGET_FACES}.summary.json
$OMEGA_RESULT_DIR/remesh/mesh_view_informed_qem_${OMEGA_VIEW_QEM_TARGET_FACES}.policy.npz
$OMEGA_RESULT_DIR/remesh/debug_meshes/view_informed_qem/
$PACKAGE_ROOT/visualizations/omega_view_informed_remesh/$(basename "$OMEGA_RESULT_DIR")/
```

## 10. Notes

- Post-hoc remesh is the first safe test because it does not change checkpoints.
- The current PyMeshLab remesh path is simplification-oriented; it will not add
  detail when the target face count is above the source count.
- For DSLR StableNormal, evidence maps should report `normalSource:
  stablenormal` in `mesh_face_evidence_summary.json`.
- Stage-2 scores now expose only the minimal policy maps: normal
  curvature/variation, plane-like weight, detail/non-planar weight, plane
  projection weight, simplify weight, and protect weight; score maps use Turbo
  0-to-1 with legends saved in each frame.
- Stage-3 view-informed QEM still uses PyMeshLab for edge collapses, but it now
  does fitted-plane projection before QEM on confident plane-like components.
  The policy arrays are written separately so we can replace the backend with a
  local collapse solver without changing the evidence math.
- In-training remesh should use a DSLR-specific config that extends the
  StableNormal baseline. Do not reuse the old iPhone `no_app` remesh configs
  without first changing their base config and output paths.
