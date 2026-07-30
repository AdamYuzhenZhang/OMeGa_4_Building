"""Helpers for ordered, dependency-driven pipeline stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def downstream_stages(
    stage: str,
    *,
    order: Sequence[str],
    dependencies: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return transitive dependents in pipeline order."""
    if stage not in dependencies:
        raise ValueError(f"Unknown pipeline stage: {stage}")

    affected = {stage}
    changed = True
    while changed:
        changed = False
        for candidate, required in dependencies.items():
            if candidate in affected or not affected.intersection(required):
                continue
            affected.add(candidate)
            changed = True
    return tuple(item for item in order if item in affected and item != stage)
