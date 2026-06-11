# In-Training Remesh Extension

This extension keeps the clean OMeGa baseline intact.  The baseline trainer is
still `examples/simple_trainer_meshgs.py`.  The remesh experiment uses a separate
entry point:

```text
examples/simple_trainer_meshgs_training_remesh.py
scripts/run_omega_training_remesh.py
configs/local/remesh_no_app_from_warmup_10000.json
omega_local/remesh/training.py
```

## What Happens

With the default local config, remeshing starts at global step `4000`, repeats every
`2000` steps, stops after step `9500`, and targets `50%` of the current face
count each time.  So a 10k run resumed from the 2999/3000 warmup remeshes at approximately global steps `4000`, `6000`, and
`8000`; if the current mesh has 400k faces, the target is about 200k faces.

Every `training_remesh_every` steps anchored at `training_remesh_start_iter`, the
extension performs one topology replacement. In other words, events happen when
`(step - training_remesh_start_iter) % training_remesh_every == 0`, which lets
us place remesh between OMeGa's own subdivision steps when desired.

1. Recompute current OMeGa splats from the current mesh and face-local splat
   parameters.
2. Save the current optimized triangle mesh.
3. Run the same planar-aware QEM simplification used by post-hoc remesh
   diagnostics.
4. For each existing splat, preserve its baked world-space identity:
   center, in-plane scale, major-axis direction, opacity, and appearance.
5. Find a nearby triangle on the simplified mesh and invert the world-space
   splat back into OMeGa's face-local parameters:

```text
mean_world -> closest new triangle -> barycentric u/v
scale_world -> new triangle base scale -> scale_lambda
old tangent axis -> projected onto new tangent frame -> rot_2d
```

The splat count does not change during this remesh step.  Splat appearance and
opacity are not recreated; they remain attached to the same splat rows.  Only
`uv_sum`, `u_ratio`, `scale_lambda`, `rot_2d`, and `index` are rewritten.

## Why This Is Separate

The baseline OMeGa subdivision only adds local detail.  This extension tests the
opposite operation during training: periodically simplifying the mesh into a
cleaner topology, then letting later optimization steps recover detail where the
rendering loss needs it.

The implementation is intentionally modular:

- `omega_local/remesh/training.py` contains all remesh and reparenting math.
- `simple_trainer_meshgs_training_remesh.py` only subclasses the baseline
  Config/Runner and swaps the mesh strategy.
- Existing baseline configs and scripts do not import this path.

## Current Limitations

- Single-GPU/single-rank only for now.
- Closest-triangle lookup uses a face-center KD tree to shortlist candidates,
  followed by exact closest-point tests inside those candidates.
- The remesh step resets optimizer moments for mesh vertices and face-local
  geometry parameters because their parameterization changed.
- Training remesh also runs a topology cleanup pass after QEM simplification so
  later OMeGa midpoint subdivision sees a subdivision-compatible mesh.
- Large simplification jumps can move splats onto too few faces; inspect the
  `training_remesh/step_*/training_remesh_summary.json` files.

## Local Configs

- `configs/local/remesh_no_app_from_warmup_20000.json`: delayed 20k comparison run with 50% remeshes at global steps 7600, 11600, and 15600.
- `configs/local/remesh_strong_no_app_from_warmup_20000.json`: stronger 20k stress test with 35% remeshes at global steps 7600, 10100, 12600, 15100, and 17600. This is meant to test whether repeated stronger simplification removes noisy corner/fold geometry, knowing it may also erase useful local detail.

## Useful Outputs

For each remesh step:

```text
<result_dir>/training_remesh/step_XXXXX/mesh_before_remesh.ply
<result_dir>/training_remesh/step_XXXXX/mesh_after_remesh.ply
<result_dir>/training_remesh/step_XXXXX/training_remesh_summary.json
<result_dir>/training_remesh/step_XXXXX/preview/training_remesh_preview_step_XXXXXX.png
<result_dir>/training_remesh/step_XXXXX/preview/before_splat_rgb.png
<result_dir>/training_remesh/step_XXXXX/preview/after_splat_rgb.png
```

The preview panel has four rows: splat RGB, mesh edges, mesh normals, and
one-sided after-to-before mesh error.  The splat RGB row is captured immediately
before and immediately after reparenting, so it is the main check for whether
the simplified mesh received the old splats correctly.

The summary records face counts, reparenting distance percentiles, normal-change
percentiles, scale clamp fraction, and how many simplified faces received splats.
