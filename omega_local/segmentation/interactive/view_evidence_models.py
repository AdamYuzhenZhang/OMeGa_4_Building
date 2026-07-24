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

        _install_huggingface_hub_cached_download_compat()
        spec = importlib.util.spec_from_file_location("interactive_stablenormal_hubconf", hubconf_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load StableNormal hubconf: {hubconf_path}")
        hub = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hub)
        _install_diffusers_controlnet_shim()
        device = resolve_torch_device(config.device)
        cache_dir = _default_stable_normal_cache_dir(root)
        kwargs = {
            "device": str(device),
            "yoso_version": "yoso-normal-v0-3",
        }
        if cache_dir is not None:
            kwargs["local_cache_dir"] = str(cache_dir)
        if config.stable_normal_variant == "turbo":
            self.predictor = hub.StableNormal_turbo(**kwargs)
        else:
            self.predictor = hub.StableNormal(**kwargs, diffusion_version="stable-normal-v0-1")
        self.predictor.segmentation_handler.device = str(device)
        if hasattr(self.predictor.model, "enable_xformers_memory_efficient_attention"):
            try:
                self.predictor.model.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        self.hub = hub
        self.torch = torch
        self.processing_resolution = int(config.stable_normal_processing_resolution)
        self.data_type = str(config.stable_normal_data_type)
        self.num_inference_steps = int(config.stable_normal_num_inference_steps)
        self.ensemble_size = max(1, int(config.stable_normal_ensemble_size))
        self.batch_size = max(1, int(config.stable_normal_batch_size))
        self.variant = str(config.stable_normal_variant)
        self.cache_dir = cache_dir

    def predict(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        model_image = self.hub.resize_image(image, self.processing_resolution)
        data_type_enum = self.hub.DataType(self.data_type.lower())
        semantic_valid: np.ndarray | None = None
        if data_type_enum != self.hub.DataType.INDOOR:
            semantic_valid = self.predictor.segmentation_handler.get_mask(model_image, data_type_enum).astype(bool)
        kwargs: dict[str, Any] = {
            "match_input_resolution": False,
            "processing_resolution": max(model_image.size),
            "ensemble_size": self.ensemble_size,
            "batch_size": self.batch_size,
        }
        if self.num_inference_steps > 0:
            kwargs["num_inference_steps"] = self.num_inference_steps
        with self.torch.inference_mode():
            pipe_out = self.predictor.model(model_image, **kwargs)
        prediction = pipe_out.prediction[0]
        if self.torch.is_tensor(prediction):
            prediction = prediction.detach().float().cpu().numpy()
        prediction = np.asarray(prediction, dtype=np.float32)
        if semantic_valid is not None:
            if semantic_valid.shape != prediction.shape[:2]:
                semantic_valid = np.asarray(
                    Image.fromarray(semantic_valid.astype(np.uint8) * 255, mode="L").resize(
                        (prediction.shape[1], prediction.shape[0]), Image.Resampling.NEAREST
                    ),
                    dtype=np.uint8,
                ) > 0
            prediction = prediction.copy()
            prediction[~semantic_valid] = 1.0

        rgb = np.clip((np.clip(prediction, -1.0, 1.0) + 1.0) * 0.5 * 255.0, 0.0, 255.0).astype(np.uint8)
        if semantic_valid is None:
            valid_mask = np.ones(rgb.shape[:2], dtype=bool)
        else:
            valid_mask = semantic_valid.astype(bool, copy=False)
        return rgb, valid_mask, {
            "method": "stable_normal_runtime",
            "qualityPreset": "interactive_prepare_quality_v1",
            "pipeline": "d05_quality_path",
            "variant": self.variant,
            "processingResolution": self.processing_resolution,
            "modelInputWidth": int(model_image.size[0]),
            "modelInputHeight": int(model_image.size[1]),
            "dataType": self.data_type,
            "numInferenceSteps": None if self.num_inference_steps <= 0 else self.num_inference_steps,
            "ensembleSize": self.ensemble_size,
            "batchSize": self.batch_size,
            "localCacheDir": str(self.cache_dir) if self.cache_dir is not None else None,
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
        depth = result.get("predicted_depth", None)
        if depth is not None:
            try:
                import torch

                if torch.is_tensor(depth):
                    depth_arr = depth.detach().float().cpu().numpy()
                else:
                    depth_arr = np.asarray(depth, dtype=np.float32)
            except Exception:
                depth_arr = np.asarray(depth, dtype=np.float32)
        else:
            depth = result.get("depth")
            if depth is None:
                raise RuntimeError("Depth Anything pipeline did not return depth output.")
            depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[0] if depth_arr.shape[0] == 1 else depth_arr[..., 0]
        return depth_arr.astype(np.float32, copy=False), {
            "method": "depth_anything_v2_transformers",
            "qualityPreset": "interactive_prepare_quality_v1",
            "model": self.model_name,
            "output": "predicted_depth" if result.get("predicted_depth", None) is not None else "depth_image",
            "units": "raw_model_output; normalized only for PNG visualization",
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


def _default_stable_normal_cache_dir(root: Path) -> Path | None:
    candidate = root / "weights"
    return candidate if candidate.exists() else None


def _install_huggingface_hub_cached_download_compat() -> None:
    try:
        import huggingface_hub
    except ImportError:
        return
    if hasattr(huggingface_hub, "cached_download"):
        return

    def _cached_download(url: str, cache_dir: str | Path | None = None, **_: Any) -> str:
        import hashlib
        import urllib.request

        root = Path(cache_dir) if cache_dir is not None else Path.home() / ".cache" / "huggingface" / "diffusers_legacy"
        root.mkdir(parents=True, exist_ok=True)
        suffix = Path(str(url).split("?")[0]).suffix
        target = root / f"{hashlib.sha256(str(url).encode('utf-8')).hexdigest()}{suffix}"
        if not target.exists():
            urllib.request.urlretrieve(str(url), target)
        return str(target)

    huggingface_hub.cached_download = _cached_download
