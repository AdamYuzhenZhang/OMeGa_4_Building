# What OMeGa Does Internally

This note is the compact mental model for reading OMeGa. OMeGa is not "mesh
extraction after Gaussian training." It jointly optimizes:

```text
mesh vertices + splats bound to mesh faces + opacity + appearance
```

The mesh affects rendering because each splat's center, scale, and orientation
are recomputed from its attached triangle every iteration. In this fork, our
OMeGa-4-Building depth/coplane losses are guarded extension blocks. For the
original OMeGa baseline, read the loop with those `building_*` flags disabled.

## Source Map

These links point to the active fork, `third_party/OMeGa_4_Building`.

- Main train loop: [`examples/simple_trainer_meshgs.py`](../examples/simple_trainer_meshgs.py#L1356)
- Mesh-to-splat initialization: [`examples/simple_trainer_meshgs.py`](../examples/simple_trainer_meshgs.py#L422)
- Runtime recomputation of splats from mesh: [`examples/simple_trainer_meshgs.py`](../examples/simple_trainer_meshgs.py#L545)
- Rasterization wrapper: [`examples/simple_trainer_meshgs.py`](../examples/simple_trainer_meshgs.py#L860)
- Optimizer setup: [`examples/simple_trainer_meshgs.py`](../examples/simple_trainer_meshgs.py#L714)
- Loss assembly: [`examples/simple_trainer_meshgs.py`](../examples/simple_trainer_meshgs.py#L1484)
- Densification, pruning, mesh subdivision schedule: [`gsplat/strategy/meshgs.py`](../gsplat/strategy/meshgs.py#L180)
- Low-level split/remove optimizer surgery: [`gsplat/strategy/ops.py`](../gsplat/strategy/ops.py#L48)
- 2DGS rasterization implementation: [`gsplat/rendering.py`](../gsplat/rendering.py#L1024)
- L1 and SSIM helpers: [`examples/utils.py`](../examples/utils.py#L241), [`examples/utils.py`](../examples/utils.py#L407)

## 1. Representation

For each mesh face `f = (A, B, C)`, OMeGa places one 2D Gaussian splat on the
face. The splat is not an independent free 3D point. Its center is parameterized
by learnable face-local variables:

```text
s = sigmoid(uv_sum)
r = sigmoid(u_ratio)

u = s * r
v = s * (1 - r)

mu = A + u * (B - A) + v * (C - A)
```

Because `u >= 0`, `v >= 0`, and `u + v = s <= 1`, the center `mu` stays inside
its triangle. OMeGa initializes these parameters from the triangle center and
then recomputes them every iteration.

Code:

- initialization: [`init_gs_from_mesh`](../examples/simple_trainer_meshgs.py#L422)
- update every iteration: [`update_gs`](../examples/simple_trainer_meshgs.py#L545)

The splat scale is tied to triangle size. OMeGa computes a triangle-size scale
similar to an inradius:

```text
area = 0.5 * ||(B - A) x (C - A)||
perimeter = |BC| + |AC| + |AB|
base_scale = 2 * area / perimeter
```

Then it learns a relative scale:

```text
scale_xy = base_scale * (sigmoid(scale_lambda) * scale_range + min_rel_scale)
```

The third scale dimension is kept as `1` for the 2D Gaussian representation;
the splat behaves like an oriented disk/ellipse on the triangle.

Orientation comes from the triangle tangent frame:

```text
t1 = normalize(B - A)
n  = normalize((B - A) x (C - A))
t2 = n x t1
```

Then `rot_2d` rotates inside this tangent plane, and the resulting 3D rotation
matrix is converted to a quaternion.

## 2. What Is Actually Optimized

The optimizer updates:

- mesh vertices: `vertices`
- splat-in-face location: `uv_sum`, `u_ratio`
- splat size: `scale_lambda`
- splat in-plane rotation: `rot_2d`
- splat opacity: `opacities`
- appearance: either `sh0`, `shN`, or `features/colors` when `app_opt` is on

Code:

- mesh vertex optimizer setup: [`create_mesh_anchors_with_optimizers` call](../examples/simple_trainer_meshgs.py#L714)
- splat optimizer setup: [`self.optimizers.update`](../examples/simple_trainer_meshgs.py#L755)
- active splat parameter names: [`_splat_checkpoint_names`](../examples/simple_trainer_meshgs.py#L1242)

Important: `means`, `scales`, and `quats` are derived, not directly optimized.
They are recomputed from mesh vertices and splat parameters before rendering:

```text
vertices, faces, uv_sum, u_ratio, scale_lambda, rot_2d
    -> means, scales, quats
    -> rasterizer
    -> loss
```

So OMeGa's mesh and splats are optimized together, but the rendered Gaussian
geometry is constrained by the mesh.

## 3. Rendering Loss

Each iteration renders one training camera:

```text
I_hat = Rasterize2DGS(mesh-bound splats, camera)
```

The main photometric loss is:

```text
L_render = (1 - lambda_ssim) * L1(I_hat, I)
         + lambda_ssim * (1 - SSIM(I_hat, I))
```

Code:

- forward rasterization call in the train loop: [`rasterize_splats`](../examples/simple_trainer_meshgs.py#L1449)
- loss assembly: [`l1loss`, `ssimloss`, `loss`](../examples/simple_trainer_meshgs.py#L1484)
- `L1`: [`examples/utils.py`](../examples/utils.py#L241)
- `SSIM`: [`examples/utils.py`](../examples/utils.py#L407)

The 2DGS rasterizer projects oriented 2D splats and alpha-composites them. A
conceptual per-pixel color equation is:

```text
C(p) = sum_i T_i(p) * alpha_i(p) * c_i
T_i(p) = product_{j before i} (1 - alpha_j(p))
```

Depth is accumulated similarly. The rasterizer also returns alpha, rendered
normals, normals-from-depth, distortion, median depth, and expected depth.

Code:

- wrapper in trainer: [`rasterize_splats`](../examples/simple_trainer_meshgs.py#L860)
- rasterization implementation: [`rasterization_2dgs`](../gsplat/rendering.py#L1024)

## 4. Mesh Loss

If `--mesh_loss` is enabled and the current step is after
`mesh_loss_start_iter`, OMeGa adds:

```text
L_mesh = lambda_normal * L_normal_consistency(M)
       + lambda_smooth / scene_scale * L_laplacian(M)
```

Code: [`mesh_loss` block](../examples/simple_trainer_meshgs.py#L1491)

Meaning:

- `mesh_normal_consistency`: adjacent faces should have similar normals.
- `mesh_laplacian_smoothing`: vertices should stay smooth relative to neighbors.

This is one reason OMeGa can produce visually flatter surfaces, but also why
sharp architectural edges can become softened if no other cue preserves them.

## 5. Monocular Normal Loss

If `--mono_normal_loss` is enabled, OMeGa compares rendered splat normals to
monocular normal maps.

The normal map is transformed into world coordinates:

```text
n_mono_world = n_mono_camera * R_c2w^T
```

Then OMeGa computes a weighted componentwise difference:

```text
d = |n_mono_world - n_render|
w = alpha * (1 - exp(-beta * d))

L_normal = lambda_mono * mean(w * d)
```

Code: [`mono_normal_loss` block](../examples/simple_trainer_meshgs.py#L1507)

This is not a pure `1 - dot(n1, n2)` loss. It is closer to a focal L1 normal
loss over normal components.

## 6. Distortion Loss

When `--dist_loss` is enabled, the 2DGS rasterizer returns a distortion term
from the depth/alpha distribution along each ray. OMeGa adds it only after
`dist_start_iter`:

```text
L_distortion = mean(render_distort)
L += dist_lambda * L_distortion
```

Code: [`distloss` block](../examples/simple_trainer_meshgs.py#L1524)

The practical effect is to discourage spread-out or multi-layer opacity along a
ray, which can help geometry concentrate into cleaner surfaces. It is still a
rendering regularizer, not a direct mesh-plane constraint.

## 7. Backpropagation

After all original OMeGa losses are summed:

```text
L = L_render + L_mesh + L_mono_normal + L_distortion
```

Then:

```text
loss.backward()
```

Code: [`loss.backward`](../examples/simple_trainer_meshgs.py#L1603)

Gradients flow through:

```text
loss -> rasterizer -> means/scales/quats -> mesh vertices + splat parameters
```

So the mesh geometry is updated because moving vertices changes the bound
splats, and changed splats change the rendered image. Adam steps happen after
strategy refinement logic:

- optimizer step: [`optimizer.step`](../examples/simple_trainer_meshgs.py#L1756)

## 8. Face Gradient For Mesh Subdivision

After backward, OMeGa measures which faces need refinement. It projects each
vertex gradient onto the current face normal:

```text
n_f = normalize((B - A) x (C - A))

g_f = sum over vertices j in face f:
      | dL/dv_j dot n_f |
```

This asks: which faces receive strong geometry-moving gradients normal to the
surface?

Code:

- face gradient computation: [`faces_grad`](../examples/simple_trainer_meshgs.py#L1606)
- accumulation into mesh state: [`_update_mesh_state`](../gsplat/strategy/meshgs.py#L348)

## 9. Splat Densification

OMeGa also tracks image-plane gradient per splat:

```text
G_i += || dL / d mean2d_i ||
count_i += 1
```

Then the strategy can:

- duplicate high-gradient small splats
- split high-gradient large splats
- prune low-opacity splats

Code:

- schedule and decision entry point: [`step_post_backward`](../gsplat/strategy/meshgs.py#L180)
- grow/prune strategy: [`_grow_gs`](../gsplat/strategy/meshgs.py#L90)
- low-level optimizer-safe parameter edits: [`ops.py`](../gsplat/strategy/ops.py#L48)

For splat splitting, it samples nearby barycentric positions and shrinks scale:

```text
uv_new = uv + noise
scale_new = scale / 1.6
```

Code: [`split` logic in ops](../gsplat/strategy/ops.py#L164)

## 10. Mesh Subdivision

Every `split_every` steps after `mesh_refine_start_iter`, OMeGa selects top
faces by accumulated normal-direction face gradient:

```text
score_f = accumulated_face_gradient_f / count_f
```

It subdivides the top `mesh_split_topk` fraction and nearby selected faces using
PyMeshLab midpoint subdivision.

Code:

- split schedule: [`step_post_backward`](../gsplat/strategy/meshgs.py#L206)
- face selection and subdivision: [`_split_meshes`](../gsplat/strategy/meshgs.py#L423)
- PyMeshLab midpoint subdivision call: [`meshing_surface_subdivision_midpoint`](../gsplat/strategy/meshgs.py#L461)

After subdivision:

- old splats on split faces are removed
- new splats are initialized on new faces
- new splats inherit appearance/opacity from nearby removed splats

Code: [`_split_meshes`](../gsplat/strategy/meshgs.py#L488)

## 11. Mesh Face Removal

OMeGa periodically removes faces that are not supported by visible/high-opacity
splats. It samples ellipse vertices from splats, compares mesh vertices to splat
support with Chamfer distance, and removes faces that are poorly supported.

Code:

- remove schedule: [`step_post_backward`](../gsplat/strategy/meshgs.py#L235)
- transparent/unsupported face removal: [`_remove_transparent_faces`](../gsplat/strategy/meshgs.py#L575)
- Chamfer comparison: [`chamfer_distance` use](../gsplat/strategy/meshgs.py#L599)

This is the main pruning logic for unsupported geometry.

## 12. Where Our Extension Fits

Our OMeGa-4-Building additions keep the same representation. They add optional
losses after original OMeGa's render, mesh, monocular normal, and distortion
losses:

```text
L_total = L_omega
        + L_building_depth       # optional
        + L_building_coplane     # optional
```

The extension losses are implemented outside the trainer in `omega_local/` and
are activated only by JSON config flags. See:

- [`docs/EXTENSION_STRUCTURE.md`](EXTENSION_STRUCTURE.md)
- [`omega_local/losses/depth_regularization.py`](../omega_local/losses/depth_regularization.py)
- [`omega_local/losses/coplane_regularization.py`](../omega_local/losses/coplane_regularization.py)

## Mental Model

OMeGa's loop is:

```text
1. Recompute splats from current mesh.
2. Render splats into the current camera.
3. Compute RGB / SSIM / mesh / normal / distortion losses.
4. Backprop into splat parameters and mesh vertices.
5. Use gradients to decide splat split/prune and mesh subdivision.
6. Adam-update mesh vertices and splat parameters.
7. Remove degenerate or unsupported faces.
```

The key idea is elegant: the mesh is optimized indirectly through differentiable
splat rendering, while the splats are constrained to live on mesh faces. That
gives OMeGa both appearance flexibility and a persistent explicit mesh.
