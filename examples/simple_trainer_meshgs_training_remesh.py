"""OMeGa mesh-splat trainer with in-training structure-aware remeshing.

This entry point intentionally leaves ``simple_trainer_meshgs.py`` untouched.
It reuses the baseline Config/Runner and swaps only the topology-update strategy
for ``TrainingRemeshMeshGSStrategy`` when ``--training-remesh-on`` is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT, REPO_ROOT / "examples"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import tyro
import wandb

import simple_trainer_meshgs as baseline
from gsplat.distributed import cli
from gsplat.strategy.meshgs import MeshGSStrategy
from omega_local.remesh.training import TrainingRemeshMeshGSStrategy
from omega_local.viz.remesh_training_preview import write_training_remesh_preview


@dataclass
class Config(baseline.Config):
    # OMeGa-4-Building in-training remesh. This periodically replaces the
    # current optimized mesh with a planar-aware simplified mesh and reparents
    # all existing splats onto the new triangles. Appearance/opacity stay on the
    # same splat identities; only face-local geometry parameters are rewritten.
    training_remesh_on: bool = False
    training_remesh_start_iter: int = 4_000
    training_remesh_every: int = 2_000
    training_remesh_stop_iter: int = 0
    training_remesh_target_faces: int = 0
    training_remesh_target_ratio: float = 0.5
    training_remesh_target_ratios: list[float] = field(default_factory=list)
    training_remesh_min_target_faces: int = 0
    training_remesh_reparent_k: int = 32
    training_remesh_reparent_chunk_size: int = 65_536
    training_remesh_max_surface_area_change_fraction: float = 0.0
    training_remesh_max_edge_p99_growth: float = 0.0

    # Event preview written exactly when a training remesh happens. It compares
    # the before/after meshes from the configured mesh_preview_frame camera.
    training_remesh_preview_on: bool = True
    training_remesh_preview_error_vmax_m: float = 0.05
    training_remesh_preview_error_reference_max_faces: int = 400_000

    # Keep training-time remesh output compatible with OMeGa's later midpoint
    # subdivision step. This removes duplicate/null elements and repairs
    # non-manifold edge/vertex cases introduced by QEM simplification.
    training_remesh_topology_cleanup: bool = True

    # These mirror omega_local.remesh.posthoc.remesh_with_planar_qem so the
    # training-time remesh and post-hoc diagnostic remesh can be compared.
    training_remesh_preserve_boundary: bool = True
    training_remesh_boundary_weight: float = 5.0
    training_remesh_preserve_normal: bool = True
    training_remesh_preserve_topology: bool = True
    training_remesh_planar_quadric: bool = True
    training_remesh_planar_weight: float = 0.0005
    training_remesh_quality_weight: bool = True
    training_remesh_crease_angle_deg: float = 80.0
    training_remesh_boundary_quality: float = 1.0
    training_remesh_min_quality: float = 0.05
    training_remesh_quality_gamma: float = 1.5
    training_remesh_isotropic_iterations: int = 0
    training_remesh_isotropic_target_length_m: float = 0.0

    def adjust_steps(self, factor: float):
        super().adjust_steps(factor)
        self.training_remesh_start_iter = int(self.training_remesh_start_iter * factor)
        self.training_remesh_every = max(1, int(self.training_remesh_every * factor))
        self.training_remesh_stop_iter = int(self.training_remesh_stop_iter * factor)


class Runner(baseline.Runner):
    def __init__(self, local_rank: int, world_rank, world_size: int, cfg: Config, run) -> None:
        super().__init__(local_rank, world_rank, world_size, cfg, run)
        if cfg.training_remesh_on:
            if world_size != 1:
                raise NotImplementedError("training_remesh_on currently supports single-GPU/rank training only.")
            strategy_kwargs = {field.name: getattr(self.strategy, field.name) for field in fields(MeshGSStrategy)}
            self.strategy = TrainingRemeshMeshGSStrategy(**strategy_kwargs)
            self.strategy.remesh_before_preview_callback = self._capture_training_remesh_splat_preview
            self.strategy.remesh_preview_callback = self._write_training_remesh_event_preview
            if world_rank == 0:
                print(
                    "Enabled OMeGa-4-Building in-training remesh: "
                    f"start={cfg.training_remesh_start_iter}, every={cfg.training_remesh_every}, "
                    f"target_faces={cfg.training_remesh_target_faces}, target_ratio={cfg.training_remesh_target_ratio}, "
                    f"target_ratios={list(cfg.training_remesh_target_ratios)}, "
                    f"min_target_faces={cfg.training_remesh_min_target_faces}."
                )

    @torch.no_grad()
    def _capture_training_remesh_splat_preview(self, *, step: int) -> dict | None:
        """Capture the fixed-frame splat render immediately before remeshing."""

        if self.world_rank != 0 or not bool(self.cfg.training_remesh_preview_on):
            return None
        if baseline.resize_rgb_and_intrinsics_for_preview is None:
            raise ImportError("omega_local mesh preview resize helper is unavailable.")

        frame = self._load_mesh_preview_frame()
        rgb_preview, K_preview, _ = baseline.resize_rgb_and_intrinsics_for_preview(
            frame["image"],
            frame["K"],
            max_width=int(self.cfg.mesh_preview_max_width),
        )
        height, width = rgb_preview.shape[:2]

        # The remesh strategy calls this after backward and before optimizer.step.
        # Refreshing derived splat tensors here is safe and gives an exact visual
        # snapshot of the representation that is about to be reparented.
        baseline.update_gs(self.cfg, self.mesh_params, self.splats, self.optimizers, self.world_size)
        camtoworld = torch.from_numpy(frame["camtoworld"]).float().to(self.device)
        K = torch.from_numpy(K_preview).float().to(self.device)
        colors, _, _, _, _, _, _, _ = self.rasterize_splats(
            camtoworlds=camtoworld[None],
            Ks=K[None],
            width=int(width),
            height=int(height),
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB+ED",
        )
        splat_rgb = torch.clamp(colors[0, ..., :3], 0.0, 1.0).detach().cpu().numpy()
        splat_rgb = (splat_rgb * 255.0).astype("uint8")
        return {
            "step": int(step),
            "frame_index": int(frame["frame_index"]),
            "rgb": rgb_preview,
            "K": K_preview,
            "camtoworld": frame["camtoworld"],
            "splat_rgb": splat_rgb,
        }

    @torch.no_grad()
    def _write_training_remesh_event_preview(self, *, step: int, remesh_payload: dict, before_preview: dict | None = None) -> None:
        """Render before/after splat and mesh diagnostics for one remesh event."""

        if self.world_rank != 0 or not bool(self.cfg.training_remesh_preview_on):
            return
        if remesh_payload.get("skipped"):
            return

        after_preview = self._capture_training_remesh_splat_preview(step=int(step))
        preview = before_preview or after_preview
        if preview is None:
            return

        out_dir = Path(str(remesh_payload["inputMesh"])).parent / "preview"
        summary = write_training_remesh_preview(
            out_dir=out_dir,
            step=int(step),
            frame_index=int(preview["frame_index"]),
            before_mesh=Path(str(remesh_payload["inputMesh"])),
            after_mesh=Path(str(remesh_payload["outputMesh"])),
            rgb=preview["rgb"],
            K=preview["K"],
            camtoworld=preview["camtoworld"],
            before_splat_rgb=None if before_preview is None else before_preview["splat_rgb"],
            after_splat_rgb=None if after_preview is None else after_preview["splat_rgb"],
            max_width=0,
            near_plane=float(self.cfg.mesh_preview_near_plane),
            max_faces=int(self.cfg.mesh_preview_max_faces),
            rotate_clockwise=bool(self.cfg.mesh_preview_rotate_clockwise),
            wire_thickness=int(self.cfg.mesh_preview_wire_thickness),
            error_vmax_m=float(self.cfg.training_remesh_preview_error_vmax_m),
            error_reference_max_faces=int(self.cfg.training_remesh_preview_error_reference_max_faces),
        )
        with open(f"{self.stats_dir}/training_remesh_preview.jsonl", "a") as handle:
            handle.write(json.dumps(summary) + "\n")
        print(
            "[training-remesh-preview] "
            f"step={step} frame={summary['frame_index']} "
            f"before_faces={summary['before_faces']:,} after_faces={summary['after_faces']:,} "
            f"err95={summary['after_error_to_before_m']['p95'] * 100.0:.2f}cm "
            f"path={summary['outputs']['panel']}"
        )


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    run = (
        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
        )
        if world_rank == 0 and cfg.wandb_on
        else None
    )

    runner = Runner(local_rank, world_rank, world_size, cfg, run)

    if cfg.ckpt is not None:
        if world_rank == 0:
            ckpts = [torch.load(file, map_location=runner.device, weights_only=True) for file in cfg.ckpt]
            for key in runner.splats.keys():
                runner.splats[key].data = torch.cat([ckpt["splats"][key] for ckpt in ckpts])
            runner.export_mesh()
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        import time

        time.sleep(1000000)


if __name__ == "__main__":
    wandb.setup()
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    cli(main, cfg, verbose=True)
