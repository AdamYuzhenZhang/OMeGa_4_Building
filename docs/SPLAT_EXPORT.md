# OMeGa Splat Export

OMeGa does not currently provide a robust browser-viewer export for the
mesh-attached splats used by `simple_trainer_meshgs.py`. Its built-in
experimental `save_ply()` only writes checkpoints that contain `sh0/shN`; runs
with `--app_opt` store `colors/features` instead, so no splat PLY is produced.

Use `scripts/export_omega_splats_to_ply.py` to bake a training checkpoint into a
static splat PLY:

```bash
PYTHON=/home/yz2332/projects/digitalTwin/.venv/bin/python
OMEGA_BUILDING_ROOT=/home/yz2332/projects/digitalTwin/third_party/OMeGa_4_Building
RESULT_DIR=/home/yz2332/projects/digitalTwin/data/scan_processing_outputs/SSSTrumbulDoor2_0513/omega/model_baseline_no_app_from_warmup_20000

$PYTHON "$OMEGA_BUILDING_ROOT/scripts/export_omega_splats_to_ply.py" \
  "$RESULT_DIR" \
  --format 3dgs \
  --out "$RESULT_DIR/web_exports/omega_final_3dgs.ply" \
  --thickness-m 0.003 \
  --sh-degree 3
```

For quick viewer/debug tests, cap the export:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/export_omega_splats_to_ply.py" \
  "$RESULT_DIR" \
  --format 3dgs \
  --max-splats 1000 \
  --out "$RESULT_DIR/web_exports/test_1000_3dgs.ply"
```

To export OMeGa/2DGS-native surfel PLY instead, write only the two in-plane
scale fields:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/export_omega_splats_to_ply.py" \
  "$RESULT_DIR" \
  --format 2dgs \
  --out "$RESULT_DIR/web_exports/omega_final_2dgs.ply" \
  --sh-degree 3
```

This `--format 2dgs` output follows the local OMeGa/2DGS PLY convention
(`scale_0`, `scale_1`, no `scale_2`). It is closer to the representation used by
OMeGa's ray-splat renderer, but it is not a standard SuperSplat input, because
SuperSplat/PlayCanvas document PLY support as standard 3DGS training output.

The exporter follows OMeGa's rendering conversion by calling `update_gs()`:

- splat centers are baked from triangle barycentric coordinates;
- in-plane scales and quaternions are baked from the attached mesh triangles;
- opacity is exported in standard logit form;
- app-opt colors are approximated with static base color `sigmoid(colors)`.

This export is good for geometric inspection and remote viewing. For no-app-opt runs, SH color coefficients can be exported directly. For `--app_opt` runs, it is not a lossless reproduction of the training renderer because the optional appearance MLP and per-image embeddings cannot be stored in a standard static 3DGS PLY.
