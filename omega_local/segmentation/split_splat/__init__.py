"""Staged adapter for the released Split&Splat baseline."""

from .contract import SplitSplatRunConfig, SplitSplatRunPaths
from .manager import SplitSplatExperimentManager

__all__ = [
    "SplitSplatExperimentManager",
    "SplitSplatRunConfig",
    "SplitSplatRunPaths",
]
