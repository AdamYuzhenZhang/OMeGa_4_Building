# Interactive Segmentation

This document tracks the designer-controlled segmentation direction. The goal is
to keep the segmentation state in 3D while letting users edit it from familiar
image frames.

## Core Idea

Automatic methods such as SAM2Object, SAI3D, Gaussian Grouping, and Split&Splat
give useful evidence, but they do not reliably infer designer-friendly building
parts. The interactive pipeline should therefore make SAI3D's 3D labels editable:

1. Start from SAI3D points or superpoints and their current labels.
2. Let the user inspect and edit through RGB frames.
3. Convert frame edits into changed 3D point or superpoint labels.
4. Reproject the edited 3D labels into every view.

This avoids a separate propagation module at first. Propagation is implicit
because the edited labels live in 3D.

## Current Skeleton

The first web editor is read-only and is meant to verify navigation and data
loading before adding edit tools.

Current behavior:

- Center canvas shows SAI3D points colored by label.
- In `Navigate`, drag rotates the 3D view in orbit mode.
- In `Navigate`, mouse wheel zooms in orbit mode.
- Bottom filmstrip lists the frames.
- Clicking a frame switches the canvas to that frame's camera/viewpoint.
- In a selected frame, left-drag pans the frame viewport, mouse wheel or
  trackpad pinch zooms around the cursor, and right-drag enters 3D orbit
  rotation from the current view.
- In a selected frame view, the RGB image is drawn only at its image extent,
  while projected 3D points are still shown outside the RGB rectangle if they
  fall within the browser canvas.
- Toolbar contains reset view, point size, frame-image background toggle, and
  a `Portrait 90°` toggle. Portrait mode is enabled by default for the current
  DSLR capture.
- The edit bar contains local debug interaction tools. These currently mutate
  only the browser-side label array and do not write back to disk.
- The left scene graph lists current object IDs, their colors, served point
  counts, selected point counts, and visibility state. It is recomputed from
  the current browser-side label array after every preview edit.

Debug tools:

- `Navigate`: inspect frames and the 3D point set.
- `Pick Label`: click a visible point to select every served point with the
  same current 3D label. `Shift` adds another label to the selection, and
  `Alt` or `Meta` removes it.
- `Lasso Points`: draw a polygon in the active frame and select visible
  projected points inside it. `Shift` adds and `Alt` or `Meta` removes.
- `SAM Prompt`: place future SAM2 prompts in the active frame. Plain click adds
  a positive prompt; `Shift`, `Alt`, or `Meta` click adds a negative prompt.
- `Run SAM2`: runs SAM2 image prediction on the active frame using that frame's
  positive and negative prompts. The returned mask is drawn as a cyan
  transparent overlay in the same frame viewport, so portrait rotation, zoom,
  and pan remain aligned.

Local preview actions:

- `Select Label`: selects all served points with the target label.
- `Assign To Target`: changes selected browser-side points to the target label.
- `New Object From Selection`: changes selected browser-side points to a new
  label ID.
- `Merge Objects`: finds the object IDs represented in the current selection
  and merges those full objects into the largest selected ID.
- `Select SAM Mask`: selects visible 3D points whose projection falls inside
  the active frame's SAM2 mask overlay.
- `Save Edits`: writes pending edits to an interactive label file without
  overwriting the original SAI3D result.
- `Clear Selection` and `Clear Prompts` reset only local editor state.
- `Clear SAM Mask` removes the active frame's SAM2 mask overlay.

Scene graph interactions:

- Click an object row to select that object. `Shift` adds it to the current
  selection, and `Alt` or `Meta` removes it.
- Double-click an object row to isolate that object.
- Use `Hide` or `Show` on a row to toggle one object.
- Use `Hide Selection`, `Isolate Selection`, and `Show All` for object-level
  visibility while editing.

Saved edits are stored separately from the original SAI3D labels:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/<baseline-name>/interactive/interactive_labels.npy
$OMEGA_RESULT_DIR/segmentation/baselines/<baseline-name>/interactive/interactive_edits.jsonl
```

The editor resumes from `interactive_labels.npy` when it exists. Point-level
edits save original point indices from the displayed sample. Object-level merge
edits save source label IDs and apply to the full label array, so merging
objects is not limited to the displayed point sample.

SAM2 is served by the editor backend through `/api/sam2/predict`. The model is
loaded lazily on the first request and the current frame embedding is cached, so
re-running prompts on the same frame should be faster after the first call. The
endpoint expects prompt coordinates in the original image coordinate system; the
frontend converts from portrait/zoomed display coordinates before sending them.

The `Portrait 90°` toggle rotates the frame thumbnails and the selected
frame-view projection 90 degrees clockwise. Internally, the editor remaps image
coordinates as:

```text
u' = H - v
v' = u
```

where `(u, v)` are original landscape pixels and `(u', v')` are the displayed
portrait pixels.

When the user starts dragging from a selected frame, the free 3D orbit camera is
initialized from that frame's rolled projection camera before rotation starts.
This keeps the 3D rotation visually continuous with the selected frame instead
of jumping back to an older orbit view. In portrait mode, horizontal dragging
uses the portrait screen-right direction, so the orbit control follows the
displayed image orientation rather than the original landscape sensor axes.

Clicking a different frame now animates the camera motion. During the transition,
the editor hides the RGB frame image and shows only the colored SAI3D points.
When the animated camera arrives, the exact selected-frame projection is restored
and the RGB image appears again. Dragging or scrolling during the animation
interrupts it and continues from the current interpolated camera.

Frame view and frame-to-frame transition use the same projection-camera model:
camera pose is interpreted as camera-to-world, points are projected with the
frame intrinsics `(fx, fy, cx, cy)`, and the result is placed into the same
browser image rectangle. During transitions, the editor interpolates camera
position, orientation, and intrinsics in this projection framework, which keeps
the zoom level consistent with the final frame view.

Selected-frame zoom and pan do not move the capture camera. Instead, the editor
stores a frame viewport `(z, centerU, centerV)` and renders an equivalent
sub-frustum:

```text
fx' = z * fx
fy' = z * fy
cx' = z * (cx - centerU) + W / 2
cy' = z * (cy - centerV) + H / 2
```

where `(W, H)` are the displayed frame dimensions after optional portrait
rotation. The camera pose `(eye, right, up, forward)` stays fixed. The RGB image
and projected SAI3D points use the same viewport transform, so zoomed inspection
preserves 2D/3D alignment.

After dragging in free 3D orbit, clicking a frame starts from the current
rolled projection camera. This prevents the first transition frame from jumping
or zooming differently from the view the user just dragged to. Portrait rotation
is encoded inside the selected frame projection camera instead of being applied
as a separate point-space post-process.

Code:

- `omega_local/segmentation/interactive/app.py`
- `omega_local/segmentation/interactive/static/index.html`
- `omega_local/segmentation/interactive/static/app.js`
- `scripts/run_omega_segmentation_editor.py`

## Launch

Set the usual project paths:

```bash
export DT_ROOT=/home/yz2332/projects/digitalTwin
export PYTHON=$DT_ROOT/.venv/bin/python
export OMEGA_BUILDING_ROOT=$DT_ROOT/third_party/OMeGa_4_Building
export PACKAGE_ROOT=$DT_ROOT/data/scan_processing_outputs/grove_entrance_dslr_0521_pinhole
export OMEGA_RESULT_DIR=$PACKAGE_ROOT/omega_stable_mesh/model_baseline_stronger_30000
```

Launch the editor on the dense SAI3D area-sample run:

```bash
$PYTHON "$OMEGA_BUILDING_ROOT/scripts/run_omega_segmentation_editor.py" \
  --model-dir "$OMEGA_RESULT_DIR" \
  --baseline-name sai3d_area_samples_1024_dense \
  --host 0.0.0.0 \
  --port 8787 \
  --max-points 180000
```

By default, SAM2 uses:

```text
$DT_ROOT/third_party/sam2
$DT_ROOT/third_party/sam2/checkpoints/sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml
```

Optional overrides:

```bash
  --sam2-root "$DT_ROOT/third_party/sam2" \
  --sam2-checkpoint "$DT_ROOT/third_party/sam2/checkpoints/sam2.1_hiera_large.pt" \
  --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam2-device auto
```

Open locally:

```text
http://127.0.0.1:8787
```

Open over Tailscale or LAN by using the machine IP and `http`, not `https`:

```text
http://100.66.11.29:8787
http://10.66.103.106:8787
```

The current server is plain HTTP. A browser URL beginning with `https://` will
not work unless we later add TLS or Tailscale Serve.

If HTTPS is required through Tailscale, keep the editor running locally and add
a Tailscale Serve proxy:

```bash
tailscale serve --bg 8787
tailscale serve status
```

Then open the HTTPS URL reported by `tailscale serve status`. For ordinary
tailnet access, the direct `http://100.66.11.29:8787` URL is simpler.

## Verify Server

Check that the server is listening:

```bash
ss -ltnp | rg ':8787'
```

Check that the page responds:

```bash
curl -sS -o /dev/null -w '%{http_code} %{content_type}\n' http://127.0.0.1:8787/
curl -sS -o /dev/null -w '%{http_code} %{content_type}\n' http://100.66.11.29:8787/
```

Expected response:

```text
200 text/html; charset=utf-8
```

Stop the server from the terminal where it is running with `Ctrl+C`. If it was
started in the background, find and stop the listening process:

```bash
ss -ltnp | rg ':8787'
kill <PID>
```

## Data Source

The editor currently expects a completed SAI3D baseline under:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/<baseline-name>
```

For the current recommended test:

```text
$OMEGA_RESULT_DIR/segmentation/baselines/sai3d_area_samples_1024_dense
```

It reads:

- `dataset/frame_manifest.jsonl`
- `dataset/posed_images/.../*.jpg`
- `dataset/posed_images/.../*.txt`
- `dataset/scans/.../points.pts`
- `mesh_labels/point_labels.npy`

## Next Steps

Add editing while keeping 3D labels as the source of truth:

1. Click or hover to inspect a label under the cursor.
2. Brush or box-select visible projected points.
3. Relabel selected points to an existing label.
4. Split selected points into a new label.
5. Merge several labels.
6. Save `interactive_labels.npy` and append actions to
   `interactive_edits.jsonl`.
7. Re-export per-view masks from the edited 3D labels.

Later, SAM2 can be added as a selection helper: a user prompts SAM2 in a frame,
we convert the mask to visible 3D points, preview the affected points, and only
then commit the label edit.
