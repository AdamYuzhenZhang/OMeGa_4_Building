"""Pixel-selection helpers for interactive proposal editing."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def compose_selection_mask(
    label_map: np.ndarray,
    payload: dict[str, Any],
    *,
    label_map_resolver=None,
) -> np.ndarray:
    """Compose browser selection operations into one boolean image mask."""
    selection_ops = _selection_ops(payload.get("selectionOps", []))
    if selection_ops:
        return _compose_selection_ops(label_map, selection_ops, label_map_resolver=label_map_resolver)
    return _compose_source_mask(
        label_map,
        {
            "proposalIds": _positive_ints(payload.get("proposalIds", [])),
            "polygons": _source_polygon_ops(payload.get("polygons", [])),
        },
    )


def selected_mask_overlay(mask: np.ndarray) -> str:
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask = np.asarray(mask, dtype=bool)
    overlay[mask, :3] = [255, 221, 86]
    overlay[mask, 3] = 112
    edge = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.FIND_EDGES)) > 0
    overlay[edge, :3] = [255, 255, 255]
    overlay[edge, 3] = 230
    return _encode_rgba_png(overlay)


def _compose_selection_ops(label_map: np.ndarray, selection_ops: list[dict[str, Any]], *, label_map_resolver=None) -> np.ndarray:
    result = np.zeros(label_map.shape, dtype=bool)
    for item in selection_ops:
        operation = "subtract" if item.get("operation") == "subtract" else "add"
        source = _selection_source_mask(label_map, item, label_map_resolver=label_map_resolver)
        if operation == "subtract":
            result &= ~source
        else:
            result |= source
    return result


def _selection_source_mask(label_map: np.ndarray, item: dict[str, Any], *, label_map_resolver=None) -> np.ndarray:
    source_label_map = label_map
    layer = str(item.get("layer", "") or "").strip()
    if layer and label_map_resolver is not None:
        source_label_map = label_map_resolver(layer)
        if source_label_map.shape != label_map.shape:
            raise ValueError(f"Selection layer {layer} shape {source_label_map.shape} does not match base shape {label_map.shape}.")

    nested = _selection_ops(item.get("selectionOps", []))
    if nested:
        mask = _compose_selection_ops(label_map, nested, label_map_resolver=label_map_resolver)
    else:
        mask = _compose_source_mask(source_label_map, item)

    mask_png = str(item.get("maskPng", "")).strip()
    if mask_png:
        try:
            with Image.open(BytesIO(base64.b64decode(mask_png))) as image:
                raster = np.asarray(image.convert("L")) > 0
        except Exception as exc:  # noqa: BLE001 - report malformed browser payloads as value errors.
            raise ValueError("Invalid selection mask PNG payload.") from exc
        if raster.shape != label_map.shape:
            raise ValueError(f"Selection mask shape {raster.shape} does not match label map shape {label_map.shape}.")
        mask |= raster
    return mask


def _compose_source_mask(label_map: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    mask = np.zeros(label_map.shape, dtype=bool)
    proposal_ids = _positive_ints(item.get("proposalIds", []))
    if proposal_ids:
        mask |= np.isin(label_map, np.asarray(proposal_ids, dtype=label_map.dtype))

    polygon_ops = _source_polygon_ops(item.get("polygons", []))
    if polygon_ops:
        mask = _apply_polygon_ops(mask, polygon_ops)
    return mask


def _apply_polygon_ops(mask: np.ndarray, polygon_ops: list[dict[str, Any]]) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    for item in polygon_ops:
        polygon = item.get("polygon", [])
        if not isinstance(polygon, list):
            continue
        polygon_mask = _rasterize_polygons([polygon], result.shape)
        if item.get("operation") == "subtract":
            result &= ~polygon_mask
        else:
            result |= polygon_mask
    return result


def _rasterize_polygons(polygons: list[list[dict[str, float]]], shape: tuple[int, int]) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [
            (
                float(point["x"]),
                float(point["y"]),
            )
            for point in polygon
            if np.isfinite(float(point.get("x", np.nan))) and np.isfinite(float(point.get("y", np.nan)))
        ]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(image, dtype=np.uint8) > 0


def _source_polygon_ops(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    ops: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            raw_polygon = item.get("polygon", [])
            operation = "subtract" if str(item.get("operation", "add")) == "subtract" else "add"
        else:
            raw_polygon = item
            operation = "add"
        polygon = _source_polygon(raw_polygon)
        if len(polygon) >= 3:
            ops.append({"operation": operation, "polygon": polygon})
    return ops


def _source_polygon(value: Any) -> list[dict[str, float]]:
    if not isinstance(value, list):
        return []
    polygon: list[dict[str, float]] = []
    for point in value:
        if not isinstance(point, dict):
            continue
        x = float(point.get("x", np.nan))
        y = float(point.get("y", np.nan))
        if np.isfinite(x) and np.isfinite(y):
            polygon.append({"x": x, "y": y})
    return polygon


def _positive_ints(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    labels: set[int] = set()
    for item in value:
        try:
            label = int(item)
        except (TypeError, ValueError):
            continue
        if label > 0:
            labels.add(label)
    return sorted(labels)


def _selection_ops(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    ops: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        operation = "subtract" if str(item.get("operation", "add")) == "subtract" else "add"
        op = {
            "operation": operation,
            "proposalIds": _positive_ints(item.get("proposalIds", [])),
            "polygons": _source_polygon_ops(item.get("polygons", [])),
        }
        layer = str(item.get("layer", "") or "").strip()
        if layer:
            op["layer"] = layer
        mask_png = str(item.get("maskPng", "")).strip()
        if mask_png:
            op["maskPng"] = mask_png
        nested = _selection_ops(item.get("selectionOps", []))
        if nested:
            op["selectionOps"] = nested
        if op["proposalIds"] or op["polygons"] or nested or mask_png:
            ops.append(op)
    return ops


def _encode_rgba_png(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGBA").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
