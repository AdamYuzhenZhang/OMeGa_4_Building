# OMeGa-4-Building Extension Structure

This fork is organized as a thin research layer around OMeGa. The main rule is:
the clean baseline must remain runnable and comparable, while architecture-aware
ideas are added as opt-in modules with explicit config flags.

## Separation Of Responsibilities

```text
scan_processing/
  Prepares our phone capture data, poses, PromptDA/MASt3R priors, Guide01 maps,
  initial mesh, and OMeGa dataset folders.

third_party/OMeGa_4_Building/examples/
  OMeGa trainer and dataset code. Keep this close to upstream. Only add small,
  guarded hooks here when gradients, dataloader fields, or render outputs must
  enter the training loop.

third_party/OMeGa_4_Building/gsplat/
  OMeGa/gsplat core rasterization, mesh update, splitting, and removal logic.
  Avoid editing this for research losses. If a topology policy must change,
  guard it behind a config flag and document the exact behavior.

third_party/OMeGa_4_Building/omega_local/
  Our local harness. New research losses, grouping code, previews, plotting
  helpers, and run orchestration should live here.

third_party/OMeGa_4_Building/configs/local/
  Source of truth for experiment settings. Baseline and extension configs
  should differ by explicit JSON values, not by hidden code branches.

third_party/OMeGa_4_Building/scripts/
  Small launch/export/plot wrappers. Scripts pick configs and call the shared
  runner; they should not contain training math.
```

## Baseline Contract

The baseline path is allowed to use neutral diagnostics, but must not apply our
new research losses or topology rules.

- Baseline configs use `"kind": "baseline"`.
- Extension configs use `"kind": "extension"`.
- `scripts/run_omega_baseline.py` calls `main("baseline")`; it refuses extension
  configs.
- `scripts/run_omega_geometry_regularized.py` and
  `scripts/run_omega_depth_regularized.py` call `main("extension")`; they refuse
  baseline configs.
- Experimental flags such as `building_depth_regularization_on` and
  `building_coplane_regularization_on` must be false or absent in baseline
  configs.
- Debug previews and JSONL loss logging are okay in baseline runs because they
  run under `torch.no_grad()` or only record already-computed scalar values.
- Shared warmup runs write to their own folder. Baseline and extension branches
  resume from that checkpoint but write to separate result folders.

This keeps the comparison honest: baseline OMeGa plus instrumentation on one
side, OMeGa-4-Building research extensions on the other.

## Current Local Modules

```text
omega_local/runner.py
  Loads JSON configs, expands environment variables, applies --set overrides,
  validates required assets, writes resolved_run_config.json, and launches
  examples/simple_trainer_meshgs.py.

omega_local/losses/depth_regularization.py
  Region-weighted rendered-depth prior:

      L_depth = lambda_d * mean_p w(p) * rho(D_render(p) - D_prior(p))

  Guide01 provides soft local geometry categories. Planar pixels can receive
  higher weight, ridge/detail pixels lower weight, and unknown pixels zero.

omega_local/losses/coplane_regularization.py
  Dynamic fitted-plane regularization. It periodically builds stop-gradient
  plane targets from the current mesh, optionally assisted by Guide01 depth
  planes, then applies:

      L_coplane =
          lambda_p * mean_f rho(n_g dot x_f + d_g)
        + lambda_n * mean_f (1 - |normal_f dot n_g|)

omega_local/viz/mesh_training_preview.py
  CPU preview renderer for fixed-frame RGB, splat render, mesh render, normal,
  and wireframe panels.

omega_local/viz/regularization_preview.py
  Debug panels for depth priors, rendered depth residuals, Guide01 categories,
  coplane groups, and coplane residuals.
```

## Trainer Hook Points

`examples/simple_trainer_meshgs.py` should contain only thin hooks into
`omega_local`, not large extension implementations.

Current hook types:

- Config fields: add flags and lambdas to the trainer config dataclass.
- Optional parser inputs: pass guide-map settings into `Parser` when an
  extension needs per-frame data.
- Loss section: compute extension losses after OMeGa's original RGB, mesh,
  monocular-normal, and distortion losses, then add them to `loss`.
- Refresh section: call non-differentiable grouping or target refresh code under
  `torch.no_grad()` before computing a differentiable loss.
- Stats section: append scalar values to `stats/train_loss.jsonl`.
- Preview section: write diagnostic images without changing optimization state.

Large algorithms should stay in `omega_local/losses/` or `omega_local/viz/`.
The trainer should read like wiring.

## Dataset Hook Points

`examples/datasets/colmap.py` should only load optional training inputs and
return tensors in the batch.

Current optional extension fields:

```text
building_depth_prior
building_depth_valid
building_p_local_planar
building_p_detail
building_p_ridge
```

Those are loaded only when the corresponding config enables depth guides. The
path mapping is kept manifest-based so OMeGa image names can be matched back to
our scan-processing frames.

If a future extension needs new priors, prefer the same pattern:

1. Generate assets in `scan_processing`.
2. Write a manifest from capture frame to asset path.
3. Add an optional loader in `examples/datasets/colmap.py`.
4. Keep interpretation and math in `omega_local`, not the dataset.

## Adding A New Loss Cleanly

Use this checklist for each new research loss.

1. Create a module under `omega_local/losses/<name>.py`.
2. Put the math and tensor shape assumptions in the module docstring.
3. Expose one main function, for example `compute_<name>_loss(...)`.
4. Return `(scaled_loss, unscaled_loss, stats)` or a similarly explicit tuple.
5. Add config keys prefixed with `building_<name>_...`.
6. Add the smallest possible guarded call in `examples/simple_trainer_meshgs.py`.
7. Add stats to `stats/train_loss.jsonl`.
8. Add a JSON config in `configs/local/`.
9. If the loss needs visual debugging, add a preview module under
   `omega_local/viz/` and keep it optional.

Recommended config naming:

```text
building_<name>_on
building_<name>_start_iter
building_<name>_lambda
building_<name>_<specific_threshold>
```

Avoid changing default trainer behavior. A new loss should do exactly nothing
unless its config flag is enabled.

## Adding A New Topology Or Mesh Update Rule

Topology rules are higher risk than losses because they can change mesh
connectivity, splat attachment, and optimizer state.

Preferred order:

1. Start with a differentiable soft loss in `omega_local/losses/`.
2. Add diagnostics to prove it affects the intended faces or regions.
3. Only then add hard mesh operations such as snapping, splitting suppression,
   merging, pruning, or region-aware subdivision.
4. If `gsplat/strategy/meshgs.py` must be edited, keep the policy behind a
   config flag and keep the baseline flag off.
5. Log counts: selected faces, changed vertices, skipped faces, new faces,
   removed faces, and active splats.

Hard mesh edits should be scheduled at clear iteration boundaries, preferably
near OMeGa's existing split/remove/refine steps, and followed by a few normal
optimization steps before judging visual quality.

## Config And Run Pattern

Use JSON configs for repeatability:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/baseline_from_warmup_20000.json"

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_geometry_regularized.py" \
  --config "$OMEGA_BUILDING_ROOT/configs/local/geometry_regularized_from_warmup_6500.json"
```

Use `--set key=value` only for small one-off tests:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_baseline.py" \
  --from-warmup \
  --set max_steps=10000 \
  --set result_dir="$PACKAGE_ROOT/omega/model_test"
```

Every run writes:

```text
resolved_run_config.json
stats/train_loss.jsonl
```

Debug-enabled runs may also write:

```text
mesh_previews/
regularization_previews/
stats/mesh_preview.jsonl
stats/regularization_preview.jsonl
stats/building_coplane_groups.jsonl
```

## What Not To Do

- Do not put research math directly into launch scripts.
- Do not hide experiment settings in Python defaults when they belong in JSON.
- Do not make baseline configs enable extension losses.
- Do not mutate mesh topology in a preview or logging function.
- Do not add unguarded changes to `gsplat/` for an A/B experiment.
- Do not overwrite the warmup folder when branching a comparison run.

The fork should stay boring in the best way: obvious configs, small hooks,
isolated modules, and outputs that make each experiment reproducible.


### In-Training Remesh Extension

The in-training remesh experiment follows the same separation rule:

- baseline trainer: `examples/simple_trainer_meshgs.py`
- remesh trainer: `examples/simple_trainer_meshgs_training_remesh.py`
- remesh/reparent math: `omega_local/remesh/training.py`
- launcher: `scripts/run_omega_training_remesh.py`
- config: `configs/local/remesh_no_app_from_warmup_10000.json`

The remesh trainer subclasses the baseline Config/Runner and only swaps the
strategy object.  This keeps baseline runs intact while allowing a periodic
mesh simplification + splat reparenting step during extension runs.
