# Joint Static Semantic 3DGS Baselines

This pipeline compares two methods that optimize one conventional static 3DGS
scene while supervising persistent region identity:

- [Segment then Splat](https://github.com/luyr/Segment-then-Splat), NeurIPS
  2025: fixed hard object ownership inherited during densification.
- [Gaussian Grouping](https://github.com/lkeab/gaussian-grouping), ECCV 2024:
  learned 16D Gaussian identity features, a learned class head, and local 3D
  identity regularization.

Both adapters consume exactly the same data from a completed MapAnything split:

```text
posed RGB + COLMAP cameras
+ indexed persistent-region masks
+ MapAnything point positions and RGB
+ persistent region ID/name/color map
```

Segment then Splat additionally initializes each point with a fixed hard region
ID. Gaussian Grouping intentionally keeps the released random 16D identity
initialization and learns identity from the same indexed masks. This preserves a
clean comparison between hard ownership and learned ownership.

The adapters bypass both papers' automatic SAM/DEVA tracking because the input
being evaluated is our user-aligned persistent mask set.

`training_view_weights.json` is copied into the shared contract to preserve
which views were manually confirmed. Neither released trainer exposes a
per-view semantic-loss weight, so the paper-faithful baselines do not consume
these weights. A weighted variant should remain a separate ablation.

## Output Layout

Shared data is prepared once and reused by both methods:

```text
<mapanything-run>/04_static_semantic_3dgs/
  00_shared_dataset/
    images/
    masks/                  # contiguous indexed NPY labels
    masks_png/              # the same labels as uint8 PNG
    sparse/0/points3D.ply   # shared positions and RGB
    initializer_region_ids.npy
    region_id_map.json
    training_view_weights.json
  runs/
    segment_then_splat/<run-id>/
    gaussian_grouping/<run-id>/
```

Each completed run exports the same representation under `03_outputs/`:

```text
scene.ply                   # standard static 3DGS geometry/appearance
gaussian_region_ids.npy     # persistent region ID for every Gaussian
regions.json                # names, colors, counts, and object PLY paths
objects/region_*.ply        # independently loadable static splat subsets
manifest.json
```

Region ID `0` means background or unassigned. Its Gaussians are exported as an
explicit part so loading all parts reconstructs the full scene.

## Runtime

Gaussian Grouping already uses the compatible rasterizer installed in the main
project environment. Segment then Splat needs a different rasterizer API. Build
it once into a method-scoped directory; this does not modify the main Python
environment:

```bash
PROJECT_ROOT=/home/yz2332/projects/digitalTwin
OMEGA_BUILDING_ROOT=$PROJECT_ROOT/third_party/OMeGa_4_Building

cd "$OMEGA_BUILDING_ROOT"
./scripts/setup_segment_then_splat_runtime.sh
```

## Grove Entrance Commands

```bash
PROJECT_ROOT=/home/yz2332/projects/digitalTwin
OMEGA_BUILDING_ROOT=$PROJECT_ROOT/third_party/OMeGa_4_Building
OMEGA_RESULT_DIR=$PROJECT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole/omega_stable_mesh/model_baseline_stronger_30000
PYTHON=$PROJECT_ROOT/.venv/bin/python
EDITOR_BASELINE=sai3d_area_samples_1024_dense
MAP_RUN=mapanything_sam2_video
RUN_ID=joint_user_masks

cd "$OMEGA_BUILDING_ROOT"
```

## Complete Segment then Splat Paper Baseline

The existing `joint_user_masks` run intentionally bypasses the paper's
automatic masks. For a direct baseline, this separate pipeline runs the
released SAM1 proposals, SAM2 tracking, three mask granularities, COLMAP
object-specific initialization, and joint multilevel 3DGS training.

It uses the editor's established 1024-pixel images and matching intrinsics.
This keeps the released per-object mask stacks tractable while preserving
exact mask-to-COLMAP pixel alignment.

```bash
PAPER_RUN=official_auto

"$PYTHON" scripts/run_omega_segment_then_splat_official.py \
  --stage prepare \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --run-id "$PAPER_RUN"

"$PYTHON" scripts/run_omega_segment_then_splat_official.py \
  --stage autoseg \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --run-id "$PAPER_RUN"

"$PYTHON" scripts/run_omega_segment_then_splat_official.py \
  --stage initialize \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --run-id "$PAPER_RUN"

"$PYTHON" scripts/run_omega_segment_then_splat_official.py \
  --stage train \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --run-id "$PAPER_RUN"

"$PYTHON" scripts/run_omega_segment_then_splat_official.py \
  --stage export \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --run-id "$PAPER_RUN"
```

`autoseg` runs the released large, middle, and small passes with detection
stride 10 and the released script default object batch 20. `initialize` calls the released overlap-filtering script, labels
aligned COLMAP points, and applies the released
`xyz_distance + 20 * rgb_distance < 0.5` merge score. `train` uses 40k
iterations, the 5k stage boundary, partial-mask filtering after 30k, three
sampled objects, densification through 20k, and partial-mask IoU `0.3`.

After `export`, the web app exposes the three paper mask layers, the three
labeled COLMAP initializers, the final joint RGB 3DGS, and a complete
per-object partition at every granularity. Stages are resumable; `autoseg`
also reuses each completed granularity.

Segment then Splat uses the released 40k schedule, three sampled objects,
densification through 20k, and partial-mask IoU `0.3`. Since our persistent
regions define one segmentation granularity, only its `default` level is active.

```bash
"$PYTHON" scripts/run_omega_segment_then_splat.py \
  --stage all \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --mapanything-run-id "$MAP_RUN" \
  --run-id "$RUN_ID"
```

Gaussian Grouping uses the released 30k schedule, 16D identity features,
densification through 10k, and its released kNN identity regularizer. Its
released loader requires full-resolution `--resolution 1` because it does not
resize indexed masks alongside RGB.

```bash
"$PYTHON" scripts/run_omega_gaussian_grouping_regions.py \
  --stage all \
  --model-dir "$OMEGA_RESULT_DIR" \
  --editor-baseline-name "$EDITOR_BASELINE" \
  --mapanything-run-id "$MAP_RUN" \
  --run-id "$RUN_ID"
```

Every stage is resumable. To run or resume explicitly, replace `--stage all`
with `prepare`, `train`, or `export`. Completed stages are reused unless
`--overwrite-stage` is supplied.

## Interpretation

The two runs differ only after common preparation:

| Property | Segment then Splat | Gaussian Grouping |
|---|---|---|
| Scene representation | One static 3DGS | One static 3DGS |
| Identity | Hard, immutable | Soft 16D feature, learned |
| Initial point identity | Persistent-region ID | Random as released |
| Densification | Child inherits hard ID | Child inherits learnable feature |
| Main advantage | Preserves user intent | Can reconcile noisy 2D evidence |
| Main risk | Wrong initial ownership is fixed | Boundaries can blur or merge |

The primary comparison is therefore whether hard user-defined ownership or
learnable identity is more reliable for architectural regions under imperfect
multi-view masks.
