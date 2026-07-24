"""Shared contract for interactive 3D segmentation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Segmentation3DRunRequest:
    method_id: str
    input_id: str
    source_id: str
    manual_frame_weight: int
    point_budget: int
    superpoint_target: int
    run_id: str
    run_dir: Path


@dataclass(frozen=True)
class Segmentation3DMethodInfo:
    method_id: str
    display_name: str
    description: str
    available: bool
    availability_message: str
    geometry_source_id: str = ""
    geometry_source_name: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "methodId": self.method_id,
            "displayName": self.display_name,
            "description": self.description,
            "available": self.available,
            "availabilityMessage": self.availability_message,
            "geometrySourceId": self.geometry_source_id,
            "geometrySourceName": self.geometry_source_name,
        }


@dataclass(frozen=True)
class Segmentation3DInputInfo:
    input_id: str
    display_name: str
    description: str
    implemented: bool
    ready: bool
    message: str
    source_options: tuple[dict[str, Any], ...] = ()
    default_source_id: str = ""
    manual_weight_options: tuple[int, ...] = ()
    default_manual_weight: int = 1

    def to_json(self) -> dict[str, Any]:
        return {
            "inputId": self.input_id,
            "displayName": self.display_name,
            "description": self.description,
            "implemented": self.implemented,
            "ready": self.ready,
            "message": self.message,
            "sourceOptions": [dict(row) for row in self.source_options],
            "defaultSourceId": self.default_source_id,
            "manualWeightOptions": [int(value) for value in self.manual_weight_options],
            "defaultManualWeight": int(self.default_manual_weight),
        }


class Segmentation3DBackend(Protocol):
    @property
    def info(self) -> Segmentation3DMethodInfo: ...

    def input_status(self) -> list[Segmentation3DInputInfo]: ...

    def run(
        self,
        request: Segmentation3DRunRequest,
        progress: ProgressCallback,
    ) -> dict[str, Any]: ...


class Segmentation3DRegistry:
    def __init__(self, backends: list[Segmentation3DBackend]) -> None:
        if not backends:
            raise ValueError("At least one 3D segmentation backend is required.")
        self._backends: dict[str, Segmentation3DBackend] = {}
        for backend in backends:
            method_id = str(backend.info.method_id).strip().lower()
            if not method_id or any(not (char.isalnum() or char == "_") for char in method_id):
                raise ValueError(f"Invalid 3D segmentation method ID: {backend.info.method_id!r}")
            if method_id != backend.info.method_id:
                raise ValueError(f"3D segmentation method ID must be normalized: {backend.info.method_id!r}")
            if method_id in self._backends:
                raise ValueError(f"Duplicate 3D segmentation method ID: {method_id}")
            self._backends[method_id] = backend

    @property
    def default_method_id(self) -> str:
        return next(iter(self._backends))

    def get(self, method_id: str) -> Segmentation3DBackend:
        key = str(method_id).strip().lower()
        if key not in self._backends:
            raise KeyError(f"Unknown 3D segmentation method: {method_id}")
        return self._backends[key]

    def status(self) -> dict[str, Any]:
        return {
            "defaultMethodId": self.default_method_id,
            "methods": [backend.info.to_json() for backend in self._backends.values()],
        }
