"""Persistent region storage for interactive multi-view segmentation.

Local proposal IDs are frame-local cleanup artifacts. Persistent region IDs are
dataset-level user intent: one named part can collect evidence from many
keyframes and later become the source for propagation/fusion.
"""

from __future__ import annotations

import json
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

from .paths import EditorPaths


def _save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def _save_png_atomic(path: Path, image: Image.Image) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


class PersistentRegionManager:
    def __init__(self, paths: EditorPaths) -> None:
        self.paths = paths

    def status(self) -> dict[str, Any]:
        payload = self._read_store()
        if self._refresh_region_counts(payload):
            payload["updatedUtc"] = _now()
            self._write_store(payload)
        frame_ids = self._region_frame_ids()
        complete_frame_ids = self.complete_frame_ids()
        regions = payload.get("regions", [])
        return {
            "ready": True,
            "regionCount": int(len(regions)),
            "frameCount": int(len(frame_ids)),
            "completeFrameCount": int(len(complete_frame_ids)),
            "regions": regions,
            "frames": frame_ids,
            "completeFrames": complete_frame_ids,
            "frameStates": self._frame_states_public(payload, valid_frame_ids=set(frame_ids)),
            "summaryPath": str(self.paths.regions_summary),
            "regionMapDir": str(self.paths.region_maps_dir),
            "updatedUtc": str(payload.get("updatedUtc", "")),
            "message": (
                f"{len(regions)} persistent region{'s' if len(regions) != 1 else ''}"
                if regions
                else "No persistent regions yet."
            ),
        }

    def set_frame_complete(self, *, frame_id: int, complete: bool) -> dict[str, Any]:
        fid = int(frame_id)
        store = self._read_store()
        path = self.region_map_path(fid)
        if bool(complete):
            if not path.exists():
                raise ValueError(f"Frame {fid} has no persistent region map to mark complete.")
            labels = np.load(path)
            if not np.any(labels > 0):
                raise ValueError(f"Frame {fid} has no positive persistent regions to mark complete.")

        now = _now()
        states = store.setdefault("frameStates", {})
        state = states.setdefault(str(fid), {})
        state["complete"] = bool(complete)
        state["updatedUtc"] = now
        store["updatedUtc"] = now
        self._write_store(store)
        record = {
            "timestampUtc": now,
            "frameId": fid,
            "operation": "set_frame_complete",
            "complete": bool(complete),
        }
        self._append_log(record)
        return {
            "saved": True,
            "frame": self.frame_summary(fid),
            "status": self.status(),
        }

    def rename_region(self, *, region_id: int, name: str) -> dict[str, Any]:
        rid = int(region_id)
        if rid <= 0:
            raise ValueError("Rename needs a positive persistent region ID.")
        store = self._read_store()
        now = _now()
        target = None
        for row in store.get("regions", []):
            if int(row.get("id", 0)) == rid:
                target = row
                break
        if target is None:
            raise ValueError(f"Persistent region {rid} does not exist.")
        target["name"] = _clean_region_name(name, rid)
        target["updatedUtc"] = now
        store["updatedUtc"] = now
        self._write_store(store)
        record = {
            "timestampUtc": now,
            "regionId": rid,
            "operation": "rename_region",
            "name": target["name"],
        }
        self._append_log(record)
        return {
            "saved": True,
            "region": self._region_public(target),
            "status": self.status(),
        }

    def delete_region(self, *, region_id: int) -> dict[str, Any]:
        rid = _positive_region_id(region_id, "Delete")
        store = self._read_store()
        regions = self._region_rows(store)
        target = self._require_region(regions, rid)

        changed_frames: list[int] = []
        cleared_area = 0
        for path in sorted(self.paths.region_maps_dir.glob("*.npy")):
            if not path.stem.isdigit():
                continue
            labels = np.load(path).astype(np.uint16, copy=True)
            selected = labels == rid
            if not np.any(selected):
                continue
            labels, stats = self._apply_region_pixels(labels, selected, target_region_id=0)
            frame_id = int(path.stem)
            self._write_frame_map(frame_id, labels, store)
            changed_frames.append(frame_id)
            cleared_area += int(stats["areaPixels"])

        now = _now()
        store["regions"] = [row for row in regions if int(row.get("id", 0)) != rid]
        store["updatedUtc"] = now
        self._refresh_region_counts(store)
        self._write_store(store)
        record = {
            "timestampUtc": now,
            "regionId": rid,
            "operation": "delete_region",
            "name": target.get("name", f"region_{rid:03d}"),
            "changedFrames": changed_frames,
            "areaPixels": int(cleared_area),
        }
        self._append_log(record)
        return {
            "saved": True,
            "deletedRegionId": rid,
            "changedFrames": changed_frames,
            "status": self.status(),
        }

    def clear_region_from_frame(self, *, frame_id: int, region_id: int) -> dict[str, Any]:
        rid = _positive_region_id(region_id, "Frame clear")
        store = self._read_store()
        region = self._require_region(self._region_rows(store), rid)

        path = self.region_map_path(frame_id)
        if not path.exists():
            raise ValueError(f"Frame {frame_id} has no persistent region map.")
        labels = np.load(path).astype(np.uint16, copy=True)
        selected = labels == rid
        if not np.any(selected):
            raise ValueError(f"Region {rid} is not present on frame {frame_id}.")

        now = _now()
        labels, stats = self._apply_region_pixels(labels, selected, target_region_id=0)
        record = {
            "timestampUtc": now,
            "frameId": int(frame_id),
            "regionId": rid,
            "operation": "clear_region_frame",
            "areaPixels": int(stats["areaPixels"]),
        }
        result = self._finalize_frame_edit(store, frame_id, labels, record, region=region, now=now)
        result.update({"clearedRegionId": rid, "areaPixels": int(stats["areaPixels"])})
        return result

    def frame_summary(self, frame_id: int) -> dict[str, Any]:
        store = self._read_store()
        state = self._frame_state_public(store, int(frame_id))
        path = self.region_map_path(frame_id)
        if not path.exists():
            return {
                "frameId": int(frame_id),
                "ready": False,
                "labels": [],
                "coverage": 0.0,
                "complete": False,
                "completeUpdatedUtc": str(state.get("updatedUtc", "")),
                "overlayUrl": "",
            }
        labels = np.load(path)
        rows = _label_rows(labels)
        return {
            "frameId": int(frame_id),
            "ready": True,
            "labels": rows,
            "coverage": float(np.count_nonzero(labels > 0) / max(labels.size, 1)),
            "complete": bool(state.get("complete", False)) and bool(rows),
            "completeUpdatedUtc": str(state.get("updatedUtc", "")),
            "regionMapNpy": str(path),
            "overlayPng": str(self.overlay_path(frame_id)),
            "overlayUrl": f"/api/regions/frame/{int(frame_id)}/overlay",
        }

    def region_map_path(self, frame_id: int) -> Path:
        return self.paths.region_maps_dir / f"{int(frame_id):06d}.npy"

    def overlay_path(self, frame_id: int) -> Path:
        return self.paths.region_overlays_dir / f"{int(frame_id):06d}.png"

    def single_region_overlay_png(self, *, frame_id: int, region_id: int) -> bytes:
        rid = int(region_id)
        if rid <= 0:
            raise ValueError("Region overlay needs a positive region ID.")
        path = self.region_map_path(frame_id)
        if not path.exists():
            raise FileNotFoundError(f"Persistent region map does not exist for frame {frame_id}: {path}")
        labels = np.load(path)
        store = self._read_store()
        region = next((row for row in store.get("regions", []) if int(row.get("id", 0)) == rid), None)
        if region is None:
            raise ValueError(f"Persistent region {rid} does not exist.")
        color = _hsl_to_rgb(str(region.get("color") or _region_color(rid)))
        overlay = _single_region_overlay(labels == rid, color)
        buffer = io.BytesIO()
        Image.fromarray(overlay, mode="RGBA").save(buffer, format="PNG")
        return buffer.getvalue()

    def commit_selection(
        self,
        *,
        frame_id: int,
        frame_shape: tuple[int, int],
        selected: np.ndarray,
        region_id: int | None = None,
        name: str | None = None,
        mode: str = "add",
        source_labels: np.ndarray | None = None,
    ) -> dict[str, Any]:
        selected = self._validate_selected(selected, frame_shape)
        source_labels = self._validate_source_labels(source_labels, selected.shape)

        store = self._read_store()
        now = _now()
        mode_key = self._normalize_selection_mode(mode)
        region: dict[str, Any] | None = None
        target_id = 0
        rid = int(region_id or 0)
        if mode_key == "new":
            region = self._create_region(store, name, now)
            target_id = int(region["id"])
        elif mode_key != "clear":
            region = self._require_region(self._region_rows(store), _positive_region_id(rid, "Region operation"))
            if name:
                region["name"] = _clean_region_name(name, int(region["id"]))
            target_id = int(region["id"])

        frame_map = self._load_or_create_frame_map(frame_id, frame_shape)
        frame_map, stats = self._apply_region_pixels(
            frame_map,
            selected,
            target_region_id=target_id,
            clear_existing_region_id=target_id if mode_key == "replace" else None,
        )

        source_ids = []
        if source_labels is not None:
            source_ids = sorted(int(value) for value in np.unique(source_labels[selected]).tolist() if int(value) > 0)
        record = {
            "timestampUtc": now,
            "frameId": int(frame_id),
            "regionId": int(target_id),
            "sourceIds": source_ids,
            "areaPixels": int(stats["areaPixels"]),
            "replacedAreaPixels": int(stats["replacedAreaPixels"]),
            "removedRegionIds": stats["removedRegionIds"],
            "operation": f"{mode_key}_region",
        }
        return self._finalize_frame_edit(store, frame_id, frame_map, record, region=region, now=now)

    def _normalize_selection_mode(self, mode: str) -> str:
        mode_key = str(mode or "add").strip().lower().replace("-", "_")
        if mode_key == "assign":
            mode_key = "replace"
        if mode_key not in {"new", "replace", "add", "clear"}:
            raise ValueError(f"Unknown region operation '{mode}'. Expected new, replace, add, or clear.")
        return mode_key

    def _region_rows(self, store: dict[str, Any]) -> list[dict[str, Any]]:
        regions = [row for row in store.setdefault("regions", []) if int(row.get("id", 0)) > 0]
        store["regions"] = regions
        return regions

    def _require_region(self, regions: list[dict[str, Any]], region_id: int) -> dict[str, Any]:
        rid = _positive_region_id(region_id, "Region operation")
        region = next((row for row in regions if int(row.get("id", 0)) == rid), None)
        if region is None:
            raise ValueError(f"Persistent region {rid} does not exist.")
        return region

    def _create_region(self, store: dict[str, Any], name: str | None, now: str) -> dict[str, Any]:
        regions = self._region_rows(store)
        existing_ids = {int(row["id"]) for row in regions}
        rid = int(store.get("nextRegionId", 1) or 1)
        while rid in existing_ids or rid <= 0:
            rid += 1
        region = {
            "id": rid,
            "name": _clean_region_name(name, rid),
            "color": _region_color(rid),
            "createdUtc": now,
            "updatedUtc": now,
            "assignments": [],
            "frameCount": 0,
            "pixelCount": 0,
        }
        regions.append(region)
        store["nextRegionId"] = int(rid + 1)
        return region

    def _validate_selected(self, selected: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != tuple(frame_shape):
            raise ValueError(f"Selected mask shape {selected.shape} does not match frame shape {frame_shape}.")
        if not np.any(selected):
            raise ValueError("Region operation needs a non-empty selected pixel region.")
        return selected

    def _validate_source_labels(self, source_labels: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
        if source_labels is None:
            return None
        source_labels = np.asarray(source_labels)
        if source_labels.shape != tuple(shape):
            raise ValueError(f"Source label shape {source_labels.shape} does not match selected mask shape {shape}.")
        return source_labels

    def _apply_region_pixels(
        self,
        labels: np.ndarray,
        selected: np.ndarray,
        *,
        target_region_id: int,
        clear_existing_region_id: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Apply the one exclusive-label-map rule used by all region operations."""
        labels = np.asarray(labels, dtype=np.uint16).copy()
        selected = np.asarray(selected, dtype=bool)
        if selected.shape != labels.shape:
            raise ValueError(f"Selected mask shape {selected.shape} does not match label map shape {labels.shape}.")
        target_id = int(target_region_id)
        if target_id < 0:
            raise ValueError("Target region ID must be non-negative.")

        selected_area = int(np.count_nonzero(selected))
        selected_before = labels[selected].copy()
        removed_region_ids = sorted(
            int(value)
            for value in np.unique(selected_before).tolist()
            if int(value) > 0 and int(value) != target_id
        )

        replaced_area = 0
        clear_existing_id = int(clear_existing_region_id or 0)
        if clear_existing_id > 0:
            replaced_area = int(np.count_nonzero(labels == clear_existing_id))
            labels[labels == clear_existing_id] = np.uint16(0)

        labels[selected] = np.uint16(target_id)
        return labels, {
            "areaPixels": selected_area,
            "replacedAreaPixels": replaced_area,
            "removedRegionIds": removed_region_ids,
        }

    def _finalize_frame_edit(
        self,
        store: dict[str, Any],
        frame_id: int,
        labels: np.ndarray,
        record: dict[str, Any],
        *,
        region: dict[str, Any] | None,
        now: str,
    ) -> dict[str, Any]:
        self._write_frame_map(frame_id, labels, store)
        if region is not None:
            region["updatedUtc"] = now
            region.setdefault("assignments", []).append(record)
        store["updatedUtc"] = now
        self._refresh_region_counts(store)
        self._write_store(store)
        self._append_log(record)
        return {
            "saved": True,
            "region": self._region_public(region) if region is not None else None,
            "assignment": record,
            "frame": self.frame_summary(frame_id),
            "status": self.status(),
        }

    def _load_or_create_frame_map(self, frame_id: int, frame_shape: tuple[int, int]) -> np.ndarray:
        path = self.region_map_path(frame_id)
        if path.exists():
            labels = np.load(path)
            if labels.shape != tuple(frame_shape):
                raise ValueError(f"Persistent region map shape {labels.shape} does not match frame shape {frame_shape}.")
            return labels.astype(np.uint16, copy=True)
        return np.zeros(tuple(frame_shape), dtype=np.uint16)

    def _write_frame_map(self, frame_id: int, labels: np.ndarray, store: dict[str, Any]) -> None:
        self.paths.region_maps_dir.mkdir(parents=True, exist_ok=True)
        self.paths.region_overlays_dir.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(labels, dtype=np.uint16)
        if not np.any(labels > 0):
            state = store.setdefault("frameStates", {}).setdefault(str(int(frame_id)), {})
            state["complete"] = False
            state["updatedUtc"] = _now()
        _save_npy_atomic(self.region_map_path(frame_id), labels)
        _save_png_atomic(
            self.paths.region_maps_dir / f"{int(frame_id):06d}.png",
            Image.fromarray(labels),
        )
        _save_png_atomic(
            self.overlay_path(frame_id),
            Image.fromarray(_region_overlay(labels, store), mode="RGBA"),
        )

    def _read_store(self) -> dict[str, Any]:
        if self.paths.regions_summary.exists():
            try:
                payload = json.loads(self.paths.regions_summary.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        payload.setdefault("stage", "interactive_persistent_regions")
        payload.setdefault("version", 1)
        payload.setdefault("createdUtc", _now())
        payload.setdefault("updatedUtc", payload["createdUtc"])
        payload.setdefault("nextRegionId", 1)
        payload.setdefault("regions", [])
        payload.setdefault("frameStates", {})
        payload.setdefault(
            "outputs",
            {
                "regionsJson": str(self.paths.regions_summary),
                "keyframeRegionMapDir": str(self.paths.region_maps_dir),
                "overlayDir": str(self.paths.region_overlays_dir),
                "editLog": str(self.paths.region_edit_log),
            },
        )
        payload["regions"] = [self._region_public(row) for row in payload.get("regions", []) if int(row.get("id", 0)) > 0]
        payload["frameStates"] = self._frame_states_public(payload)
        return payload

    def _write_store(self, payload: dict[str, Any]) -> None:
        payload["regions"] = [self._region_public(row) for row in payload.get("regions", [])]
        payload["frameStates"] = self._frame_states_public(payload)
        self.paths.regions_summary.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.paths.regions_summary.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.paths.regions_summary)

    def _append_log(self, record: dict[str, Any]) -> None:
        self.paths.region_edit_log.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.region_edit_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _refresh_region_counts(self, store: dict[str, Any]) -> bool:
        changed = False
        pixel_counts = {int(row["id"]): 0 for row in store.get("regions", [])}
        frame_sets = {int(row["id"]): set() for row in store.get("regions", [])}
        frame_areas = {int(row["id"]): {} for row in store.get("regions", [])}
        for path in sorted(self.paths.region_maps_dir.glob("*.npy")):
            if not path.stem.isdigit():
                continue
            try:
                labels = np.load(path)
            except Exception:
                continue
            frame_id = int(path.stem)
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique.tolist(), counts.tolist(), strict=True):
                label = int(label)
                if label <= 0 or label not in pixel_counts:
                    continue
                pixel_counts[label] += int(count)
                frame_sets[label].add(frame_id)
                frame_areas[label][frame_id] = int(frame_areas[label].get(frame_id, 0) + int(count))
        for row in store.get("regions", []):
            rid = int(row["id"])
            frame_ids = sorted(int(frame_id) for frame_id in frame_sets.get(rid, set()))
            valid_frames = set(frame_ids)
            assignments = row.get("assignments", [])
            if isinstance(assignments, list):
                pruned_assignments = [
                    assignment
                    for assignment in assignments
                    if _assignment_frame_id(assignment) in valid_frames
                ]
            else:
                pruned_assignments = []
            frame_area_rows = [
                {"frameId": int(frame_id), "areaPixels": int(frame_areas.get(rid, {}).get(frame_id, 0))}
                for frame_id in frame_ids
            ]
            next_values = {
                "pixelCount": int(pixel_counts.get(rid, 0)),
                "frameCount": int(len(frame_ids)),
                "frameIds": frame_ids,
                "frameAreas": frame_area_rows,
                "assignments": pruned_assignments,
            }
            for key, value in next_values.items():
                if row.get(key) != value:
                    row[key] = value
                    changed = True
        return changed

    def _region_frame_ids(self) -> list[int]:
        frame_ids: list[int] = []
        for path in sorted(self.paths.region_maps_dir.glob("*.npy")):
            if not path.stem.isdigit():
                continue
            try:
                labels = np.load(path)
            except Exception:
                continue
            if np.any(labels > 0):
                frame_ids.append(int(path.stem))
        return frame_ids

    def complete_frame_ids(self) -> list[int]:
        store = self._read_store()
        positive_frame_ids = set(self._region_frame_ids())
        out: list[int] = []
        for raw_frame_id, state in store.get("frameStates", {}).items():
            try:
                frame_id = int(raw_frame_id)
            except (TypeError, ValueError):
                continue
            if frame_id in positive_frame_ids and bool(state.get("complete", False)):
                out.append(frame_id)
        return sorted(out)

    def _frame_state_public(self, store: dict[str, Any], frame_id: int) -> dict[str, Any]:
        states = store.get("frameStates", {})
        raw = states.get(str(int(frame_id)), {}) if isinstance(states, dict) else {}
        return {
            "complete": bool(raw.get("complete", False)),
            "updatedUtc": str(raw.get("updatedUtc", "")),
        }

    def _frame_states_public(self, store: dict[str, Any], valid_frame_ids: set[int] | None = None) -> dict[str, Any]:
        states = store.get("frameStates", {})
        if not isinstance(states, dict):
            return {}
        out: dict[str, Any] = {}
        for raw_frame_id, raw_state in states.items():
            try:
                frame_id = int(raw_frame_id)
            except (TypeError, ValueError):
                continue
            if valid_frame_ids is not None and frame_id not in valid_frame_ids:
                continue
            raw = raw_state if isinstance(raw_state, dict) else {}
            out[str(frame_id)] = {
                "complete": bool(raw.get("complete", False)),
                "updatedUtc": str(raw.get("updatedUtc", "")),
            }
        return out

    def _region_public(self, row: dict[str, Any]) -> dict[str, Any]:
        rid = int(row.get("id", 0))
        return {
            "id": rid,
            "name": _clean_region_name(row.get("name", ""), rid),
            "color": str(row.get("color") or _region_color(rid)),
            "createdUtc": str(row.get("createdUtc") or _now()),
            "updatedUtc": str(row.get("updatedUtc") or row.get("createdUtc") or _now()),
            "assignments": list(row.get("assignments", [])) if isinstance(row.get("assignments", []), list) else [],
            "frameCount": int(row.get("frameCount", 0) or 0),
            "pixelCount": int(row.get("pixelCount", 0) or 0),
            "frameIds": _positive_int_list(row.get("frameIds", [])),
            "frameAreas": _frame_area_rows(row.get("frameAreas", [])),
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _positive_region_id(value: Any, label: str) -> int:
    try:
        region_id = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} needs a positive persistent region ID.") from exc
    if region_id <= 0:
        raise ValueError(f"{label} needs a positive persistent region ID.")
    return region_id


def _assignment_frame_id(value: Any) -> int | None:
    if not isinstance(value, dict):
        return None
    try:
        frame_id = int(value.get("frameId"))
    except (TypeError, ValueError):
        return None
    return frame_id if frame_id >= 0 else None


def _positive_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out: set[int] = set()
    for item in value:
        try:
            parsed = int(item)
        except (TypeError, ValueError):
            continue
        if parsed >= 0:
            out.add(parsed)
    return sorted(out)


def _frame_area_rows(value: Any) -> list[dict[str, int]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, int]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            frame_id = int(item.get("frameId"))
            area = int(item.get("areaPixels", 0) or 0)
        except (TypeError, ValueError):
            continue
        if frame_id >= 0 and area > 0:
            rows.append({"frameId": frame_id, "areaPixels": area})
    rows.sort(key=lambda row: int(row["frameId"]))
    return rows


def _clean_region_name(value: Any, region_id: int) -> str:
    text = str(value or "").strip()
    return text[:80] if text else f"region_{int(region_id):03d}"


def _region_color(region_id: int) -> str:
    hue = (int(region_id) * 137.508) % 360.0
    return f"hsl({hue:.1f} 72% 58%)"


def _hsl_to_rgb(css: str) -> tuple[int, int, int]:
    try:
        raw = css.strip().lower()
        raw = raw.removeprefix("hsl(").removesuffix(")")
        parts = raw.replace("%", "").split()
        h = float(parts[0]) % 360.0
        s = float(parts[1]) / 100.0
        l = float(parts[2]) / 100.0
    except Exception:
        return (124, 199, 255)

    c = (1.0 - abs(2.0 * l - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = l - c / 2.0
    if h < 60:
        rgb = (c, x, 0.0)
    elif h < 120:
        rgb = (x, c, 0.0)
    elif h < 180:
        rgb = (0.0, c, x)
    elif h < 240:
        rgb = (0.0, x, c)
    elif h < 300:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    return tuple(int(round((channel + m) * 255.0)) for channel in rgb)


def persistent_region_color_map(region_rows: list[dict[str, Any]]) -> dict[int, tuple[int, int, int]]:
    """Return the canonical display color for each persistent region."""

    return {
        int(row["id"]): _hsl_to_rgb(str(row.get("color") or _region_color(int(row["id"]))))
        for row in region_rows
        if int(row.get("id", 0)) > 0
    }


def _region_overlay(labels: np.ndarray, store: dict[str, Any]) -> np.ndarray:
    labels = np.asarray(labels)
    overlay = np.zeros((*labels.shape, 4), dtype=np.uint8)
    colors = persistent_region_color_map(store.get("regions", []))
    for label, color in colors.items():
        mask = labels == int(label)
        overlay[mask, :3] = np.asarray(color, dtype=np.uint8)
        overlay[mask, 3] = 118
        edge = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.FIND_EDGES)) > 0
        overlay[edge, :3] = [255, 255, 255]
        overlay[edge, 3] = 230
    return overlay


def _single_region_overlay(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask, :3] = np.asarray(color, dtype=np.uint8)
    overlay[mask, 3] = 134
    edge = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255, mode="L").filter(ImageFilter.FIND_EDGES)) > 0
    overlay[edge, :3] = [255, 255, 255]
    overlay[edge, 3] = 238
    return overlay


def _label_rows(labels: np.ndarray) -> list[dict[str, Any]]:
    unique, counts = np.unique(labels, return_counts=True)
    rows = [
        {"regionId": int(label), "assignedPixels": int(count)}
        for label, count in zip(unique.tolist(), counts.tolist(), strict=True)
        if int(label) > 0
    ]
    rows.sort(key=lambda row: int(row["regionId"]))
    return rows
