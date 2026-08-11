"""Joint static-3DGS baselines with persistent region supervision."""

from .contract import StaticSemantic3DGSConfig, StaticSemantic3DGSPaths
from .manager import StaticSemantic3DGSManager

__all__ = [
    "StaticSemantic3DGSConfig",
    "StaticSemantic3DGSManager",
    "StaticSemantic3DGSPaths",
]
