"""Runtime model adapters for interactive normal/depth evidence."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class StableNormalPredictor:
    def __init__(self, config: Any) -> None:
        root = Path(config.stable_normal_root).expanduser().resolve()
        hubconf_path = root / "hubconf.py"
        if not hubconf_path.exists():
            raise FileNotFoundError(f"StableNormal hubconf.py not found: {hubconf_path}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import importlib.util
        import torch

        spec = importlib.util.spec_from_file_location("interactive_stablenormal_hubconf", hubconf_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load StableNormal hubconf: {hubconf_path}")
        hub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hub)
        _install_diffusers_controlnet_shim()
        device = resolve_torch_device(config.device)
        kwargs = {
            "device": str(device),
            "yoso_version": "yoso-normal-v0-3",
        }
        if config.stable_normal_variant == "turbo":
            self.predictor = hub.StableNormal_turbo(**kwargs)
        else:
            self.predictor = hub.StableNormal(**kwargs, diffusion_version="stable-normal-v0-1")
        self.torch = torch
        self.processing_resolution = int(config.stable_normal_processing_resolution)

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
        with self.torch.inference_mode():
            normal_image = self.predictor(
                image,
                resolution=self.processing_resolution,
                match_input_resolution=True,
                data_type="indoor",
            )
        rgb = np.asarray(normal_image.convert("RGB"), dtype=np.uint8)
        if (rgb.shape[1], rgb.shape[0]) != image.size:
            rgb = np.asarray(Image.fromarray(rgb, mode="RGB").resize(image.size, Image.Resampling.LANCZOS), dtype=np.uint8)
        return rgb, {
            "method": "stable_normal_runtime",
            "processingResolution": self.processing_resolution,
            "dataType": "indoor",
        }


class DepthAnythingPredictor:
    def __init__(self, config: Any) -> None:
        from transformers import pipeline

        device = resolve_torch_device(config.device)
        pipeline_device = 0 if device.type == "cuda" else -1
        self.pipe = pipeline("depth-estimation", model=config.depth_anything_model, device=pipeline_device)
        self.model_name = config.depth_anything_model

    def predict(self, image: Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
        result = self.pipe(image)
        depth = result.get("depth")
        if depth is None:
            raise RuntimeError("Depth Anything pipeline did not return a depth image.")
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[..., 0]
        valid = np.isfinite(depth_arr)
        if np.any(valid):
            lo, hi = np.percentile(depth_arr[valid], [2.0, 98.0])
            depth_arr = (depth_arr - float(lo)) / max(float(hi - lo), 1e-6)
            depth_arr = np.clip(depth_arr, 0.0, 1.0)
        return depth_arr.astype(np.float32, copy=False), {
            "method": "depth_anything_v2_transformers",
            "model": self.model_name,
            "units": "relative_or_metric_model_output_normalized_for_selection",
        }


def resolve_torch_device(value: str) -> Any:
    import torch

    requested = str(value).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value}, but CUDA is not available.")
    return device


def _install_diffusers_controlnet_shim() -> None:
    """Support Stable-X dynamic modules with newer diffusers import paths."""

    if "diffusers.models.controlnet" in sys.modules:
        return
    try:
        from diffusers.models.controlnets.controlnet import ControlNetOutput
    except Exception:
        return
    module = types.ModuleType("diffusers.models.controlnet")
    module.ControlNetOutput = ControlNetOutput
    sys.modules["diffusers.models.controlnet"] = module
