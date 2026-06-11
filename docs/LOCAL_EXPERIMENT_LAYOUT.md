# Local OMeGa Experiment Layout

This fork should stay organized as a thin research harness around OMeGa.
`scan_processing` prepares capture data and exports OMeGa's dataset layout.  The
fork owns training configs, local debug previews, and future research extensions.

## Layout

```text
examples/
  simple_trainer_meshgs.py        # OMeGa trainer, kept close to upstream
  datasets/colmap.py              # small pycolmap compatibility shim

omega_local/
  runner.py                       # JSON config loader and trainer launcher
  losses/depth_regularization.py  # Guide01-weighted rendered-depth loss
  losses/coplane_regularization.py # dynamic fitted-plane mesh loss
  viz/mesh_training_preview.py    # opt-in training preview renderer
  viz/regularization_preview.py   # depth/coplane fixed-frame debug panels

docs/
  ARCHITECTURE_EXTENSION_IDEAS.md # concise research direction
  LOCAL_EXPERIMENT_LAYOUT.md      # local harness layout and workflow
  EXTENSION_STRUCTURE.md          # rules for adding losses/topology cleanly
  OMEGA_MECHANICS.md             # original OMeGa representation/loss loop notes

configs/local/
  warmup_2999.json                # shared 0..2999 warmup checkpoint
  baseline_debug_6500.json        # short clean baseline
  baseline_from_warmup_6500.json  # clean baseline branch from warmup
  baseline_debug_full.json        # full 30k clean baseline
  depth_regularized_6500.json     # extension A/B run with depth regularization
  depth_regularized_from_warmup_6500.json
  geometry_regularized_6500.json  # depth + dynamic coplane regularization
  geometry_regularized_from_warmup_6500.json
  extension_template.json         # extends baseline, changes kind/result_dir

scripts/
  run_omega_warmup.py             # produce ckpt_2999_rank0.pt
  run_omega_baseline.py           # launch configs with kind="baseline"
  run_omega_depth_regularized.py  # launch the depth-regularized extension
  run_omega_geometry_regularized.py # launch depth + coplane extension
  run_omega_extension.py          # launch configs with kind="extension"
```

## Ownership

- `scan_processing` prepares capture data, image/depth/normal assets, MASt3R
  initialization, and the OMeGa dataset folder.
- `examples/` should stay close to upstream OMeGa.  Keep only compatibility
  fixes and neutral diagnostics there.
- `omega_local/` is the home for local runners, preview utilities, and future
  extension modules.
- `configs/local/` is the source of truth for run settings.  Prefer changing a
  JSON config, or passing a small `--set` override, instead of editing trainer
  defaults.

## Workflow

First prepare data from the main repo:

```bash
$PYTHON "$DT_ROOT/scan_processing/23_run_phone_omega.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --omega-root "$OMEGA_DEBUG_ROOT" \
  --colmap-bin "$COLMAP_BIN" \
  --normal-source promptda \
  --point-cloud-source mast3r \
  --seed-voxel-size-m 0.01 \
  --overwrite
```

Then train from this fork:

```bash
export PACKAGE_ROOT=/path/to/data/scan_processing_outputs/<capture_name>
export OMEGA_DEBUG_ROOT=/home/yz2332/projects/digitalTwin/third_party/OMeGa_4_Building

$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_DEBUG_ROOT/configs/local/baseline_debug_6500.json"
```

For the full baseline:

```bash
$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_DEBUG_ROOT/configs/local/baseline_debug_full.json"
```

For A/B tests, run a shared warmup once and then branch into separate result
folders:

```bash
$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_warmup.py"

$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_baseline.py" \
  --from-warmup

$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_depth_regularized.py" \
  --from-warmup

$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_geometry_regularized.py" \
  --from-warmup
```

The warmup writes
`$PACKAGE_ROOT/omega/model_warmup_2999/ckpts/ckpt_2999_rank0.pt`; branches read
that checkpoint but write their own output folders.  The warmup is not pure RGB:
OMeGa's original mesh smoothness and mesh normal consistency start at step
`1000`, while monocular normal, distortion, mesh refinement, and local extension
losses start at step `3000`.

Use `--dry-run` to inspect the resolved command without launching training:

```bash
$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_DEBUG_ROOT/configs/local/baseline_debug_6500.json" \
  --dry-run
```

Use `--set` for small one-off overrides:

```bash
$PYTHON "$OMEGA_DEBUG_ROOT/scripts/run_omega_baseline.py" \
  --config "$OMEGA_DEBUG_ROOT/configs/local/baseline_debug_6500.json" \
  --set result_dir="$PACKAGE_ROOT/omega/model_test" \
  --set max_steps=1000
```

The default configs write to named folders such as
`$PACKAGE_ROOT/omega/model_baseline_debug_6500`.  If you want the existing
`scan_processing/Vis23` and `Vis24` scripts to work without copying outputs,
override `result_dir` to `$PACKAGE_ROOT/omega/model`.

## Outputs

Each run writes `resolved_run_config.json` into its `result_dir`.  Debug preview
configs also write:

```text
mesh_previews/mesh_preview_contact_sheet.png
stats/mesh_preview.jsonl
stats/train_loss.jsonl
stats/val_step*.json
```

Depth/coplane extension configs additionally write:

```text
regularization_previews/regularization_preview_contact_sheet.png
stats/regularization_preview.jsonl
stats/building_coplane_groups.jsonl
```

## Extension Rule

Future architecture-aware losses and topology rules should enter through the
extension path:

```text
configs/local/<extension_name>.json
scripts/run_omega_extension.py
omega_local/<extension_module>.py
```

Do not mix extension-specific behavior into the baseline configs or baseline
runner.  This keeps comparisons honest: baseline OMeGa plus diagnostics on one
side, research extensions on the other.

- `docs/TRAINING_REMESH.md` explains the in-training remesh/reparent extension.
