# Run Gaussian Grouping On OMeGa DSLR Data

This is a faithful Gaussian Grouping baseline. It does not use our
SAM2Object, SAI3D, mesh, point-cloud, or lifted-mask outputs.

Pipeline:

```text
OMeGa COLMAP dataset
  -> Gaussian Grouping data/<scene> layout
  -> original Gaussian Grouping DEVA/SAM pseudo labels
  -> original Gaussian Grouping training
  -> original Gaussian Grouping render.py outputs
```

## 1. Environment

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export PYTHON=$DT_ROOT/.venv/bin/python
export OMEGA_BUILDING_ROOT=$DT_ROOT/third_party/OMeGa_4_Building
export GAUSSIAN_GROUPING_ROOT=$DT_ROOT/third_party/gaussian-grouping
export PACKAGE_ROOT=$DT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole
export OMEGA_RESULT_DIR=$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000
export OUT_ROOT=$DT_ROOT/data/scan_processing_outputs
export GG_SCALE=2
export GG_RUN=deva
```

Scale choice for this DSLR set:

```text
scale 1: 4096x2731, 11.19M pixels, strongest but expensive
scale 2: 2048x1366,  2.80M pixels, recommended high-quality test
scale 4: 1024x683,   0.70M pixels, fast first test
scale 8: 512x341,    0.17M pixels, usually too coarse for facade masks
```

The wrapper writes one conventional Gaussian Grouping dataset and one OMeGa
output run. Rerun with `--overwrite` when changing scale:

```text
<gaussian-grouping>/data/grove_entrance_dslr_0521_pinhole_gaussian_grouping/
<model_dir>/segmentation/gaussian_grouping/deva/
```

The original pseudo-label script still runs DEVA's temporal model at
`--size 480`, but SAM proposals and saved masks are produced from the staged
input images. So scale 2 gives sharper proposal/mask resolution than scale 4
while keeping the method's original pseudo-label logic.

For scale 2, DEVA/SAM can run close to GPU memory limits near the end of the
214-frame sequence. The OMeGa wrapper sets:

```text
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

for Gaussian Grouping subprocesses and checks that every image has a matching
`object_mask/*.png` before training. If pseudo-label generation still runs out
of memory, switch `GG_SCALE=4` and rerun stages 3-7.

Scale 2 can also run out of memory during training after Gaussian densification
has added many splats. For scale 2, use `--data-device cpu` in the training
command first. This keeps the full set of loaded camera images/masks on CPU
while preserving the original Gaussian Grouping losses and schedule. If that
still fails, switch `GG_SCALE=4` and rerun stages 3-7.

Gaussian Grouping uses a modified rasterizer. Build it in the active Python
environment before training.

On this workstation, PyTorch reports the RTX 5090 as capability `12.0`, while
the system CUDA compiler is older than that architecture. Use PTX fallback for
the extension build:

```bash
cd "$GAUSSIAN_GROUPING_ROOT"
TORCH_CUDA_ARCH_LIST='9.0+PTX' MAX_JOBS=8 \
  $PYTHON -m pip install --no-build-isolation submodules/diff-gaussian-rasterization

TORCH_CUDA_ARCH_LIST='9.0+PTX' MAX_JOBS=8 \
  $PYTHON -m pip install --no-build-isolation submodules/simple-knn
```

Prepare the original DEVA/SAM dependencies for pseudo-label generation:

```bash
cd "$GAUSSIAN_GROUPING_ROOT/Tracking-Anything-with-DEVA"
$PYTHON -m pip install -e .
$PYTHON -m pip install 'huggingface-hub>=0.33.5,<1.0'
bash scripts/download_models.sh
```

Installed checkpoints should be:

```text
Tracking-Anything-with-DEVA/saves/DEVA-propagation.pth
Tracking-Anything-with-DEVA/saves/groundingdino_swint_ogc.pth
Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth
Tracking-Anything-with-DEVA/saves/mobile_sam.pt
Tracking-Anything-with-DEVA/saves/GroundingDINO_SwinT_OGC.py
```

If the extension build fails under the project venv on another machine, use a
separate Gaussian Grouping Python/Conda environment and pass its Python path
with `--python`.

## 2. Check Readiness

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage check \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE"
```

Inspect:

```text
<model_dir>/segmentation/gaussian_grouping/<run>/summaries/check_summary.json
```

The important status is that the Gaussian rasterizer, `simple_knn`, DEVA, SAM,
GroundingDINO, and the DEVA checkpoint files all import or exist.

## 3. Stage The Dataset

This symlinks the OMeGa COLMAP `images/` and `sparse/` folders into Gaussian
Grouping's `data/<scene>/` layout and creates `images_<scale>/` for DEVA/SAM
pseudo labels.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage stage_dataset \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE" \
  --overwrite
```

## 4. Prepare Pseudo Labels

This calls Gaussian Grouping's own:

```text
script/prepare_pseudo_label.sh <dataset_name> <scale>
```

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage prepare_pseudo_labels \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE" \
  --overwrite
```

Outputs:

```text
<gaussian-grouping>/data/<dataset_name>/object_mask/
<run_dir>/summaries/prepare_pseudo_labels_summary.json
<run_dir>/logs/prepare_pseudo_labels.log
```

For this dataset, the expected object-mask count is `214`.

### Optional: Finer Pseudo Labels

The original Gaussian Grouping pseudo-label command uses DEVA's
`--suppress_small_objects`, which lets large masks absorb smaller SAM masks.
For facade/component segmentation, first test the finer variant before
retraining:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage prepare_pseudo_labels \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE" \
  --preserve-small-objects \
  --overwrite
```

Inspect pseudo labels only:

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega09_visualize_gaussian_grouping.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --run-name "$GG_RUN" \
  --pseudo-only \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 520 \
  --output-dir "$PACKAGE_ROOT/visualizations/omega_segmentation/gaussian_grouping/${OMEGA_RESULT_DIR##*/}_${GG_RUN}_pseudo_preserve_small" \
  --overwrite
```

If this looks better, rerun training and rendering. If it is too fragmented,
go back to the original pseudo-label command above.

If the log reports `Number of objects exceeded maximum`, the default DEVA cap
is `200`. The current training-compatible Gaussian Grouping masks are 8-bit
grayscale and the training config uses `256` classes, so do not set this above
`255` without extending the mask/class pipeline. Raising the cap alone is not
enough because DEVA counts historical object IDs; reduce how many new SAM
objects are born with a stricter SAM threshold, fewer prompt-grid samples, and
less frequent re-detection:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage prepare_pseudo_labels \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE" \
  --preserve-small-objects \
  --sam-pred-iou-threshold 0.82 \
  --sam-num-points-per-side 48 \
  --detection-every 10 \
  --max-num-objects 240 \
  --overwrite
```

If this becomes too sparse, relax the threshold toward `0.80` or use
`--detection-every 8`. If it still exceeds the object budget, make the proposal
stream more conservative with `--sam-pred-iou-threshold 0.85` and
`--sam-num-points-per-side 40`.

## 5. Train

This calls the original `train.py` with the staged dataset and
`-r "$GG_SCALE"`.
The default iteration count is the value used by Gaussian Grouping's original
training script.

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage train \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE" \
  --data-device cpu \
  --overwrite
```

For a shorter sanity training run, add:

```bash
  --iterations 7000
```

Outputs:

```text
<model_dir>/segmentation/gaussian_grouping/<run>/model/
<run_dir>/logs/train.log
```

## 6. Render

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_gaussian_grouping.py" \
  --stage render \
  --model-dir "$OMEGA_RESULT_DIR" \
  --gaussian-grouping-root "$GAUSSIAN_GROUPING_ROOT" \
  --python "$PYTHON" \
  --run-name "$GG_RUN" \
  --scale "$GG_SCALE" \
  --num-classes 256
```

Native Gaussian Grouping outputs:

```text
<run_dir>/model/train/ours_<iter>/gt/
<run_dir>/model/train/ours_<iter>/renders/
<run_dir>/model/train/ours_<iter>/gt_objects_color/
<run_dir>/model/train/ours_<iter>/objects_pred/
<run_dir>/model/train/ours_<iter>/objects_feature16/
<run_dir>/model/train/ours_<iter>/concat/
```

At scale 2, the optional 5-column `concat/result.mp4` is wider than common
MPEG-4 limits. The renderer has been patched to keep all per-frame render
outputs and concat PNGs, but skip the optional MP4 when the frame is too large.

## 7. Visualize

```bash
$PYTHON "$DT_ROOT/scan_processing/VisSegOmega09_visualize_gaussian_grouping.py" \
  "$PACKAGE_ROOT" \
  --out-root "$OUT_ROOT" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --run-name "$GG_RUN" \
  --frame-stride 4 \
  --max-frames 32 \
  --panel-cell-width 520 \
  --overwrite
```

Inspect:

```text
<capture>/visualizations/omega_segmentation/gaussian_grouping/
```

The panel columns are RGB, Gaussian Grouping pseudo labels, pseudo label map,
rendered RGB, predicted mask, and the learned 16D identity-feature PCA.
