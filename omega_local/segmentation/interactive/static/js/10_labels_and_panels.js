// Point/proposal color palettes, undo snapshots, and left-side ID panels.
function palette(label) {
  if (state.labelColors.has(label)) return state.labelColors.get(label);
  if (label <= 0) {
    state.labelColors.set(label, "#f8fbfd");
    return "#f8fbfd";
  }
  const hue = (label * 137.508) % 360;
  const color = `hsl(${hue.toFixed(1)} 72% 58%)`;
  state.labelColors.set(label, color);
  return color;
}

function idStats() {
  const stats = new Map();
  if (state.labelSummary && Array.isArray(state.labelSummary.labels)) {
    for (const item of state.labelSummary.labels) {
      const label = Number(item.label);
      if (label <= 0) continue;
      stats.set(label, {
        label,
        count: Number(item.count) || 0,
        served: 0,
        selected: 0,
      });
    }
  }

  for (let i = 0; i < state.labels.length; i += 1) {
    const label = state.labels[i];
    if (label <= 0) continue;
    const entry = stats.get(label) || { label, count: 0, served: 0, selected: 0 };
    if (!state.labelSummary) entry.count += 1;
    entry.served += 1;
    stats.set(label, entry);
  }
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    const entry = stats.get(label);
    if (entry) entry.selected += 1;
  }
  return [...stats.values()].sort((a, b) => a.label - b.label);
}

function indicesForLabel(label) {
  const indices = [];
  for (let i = 0; i < state.labels.length; i += 1) {
    if (state.labels[i] === label) indices.push(i);
  }
  return indices;
}

function clonePolygonOps(ops) {
  return Array.isArray(ops)
    ? ops.map((entry) => ({
      operation: entry.operation === "subtract" ? "subtract" : "add",
      polygon: Array.isArray(entry.polygon)
        ? entry.polygon.map((point) => ({ x: point.x, y: point.y }))
        : [],
    }))
    : [];
}

function cloneSelectionOps(ops) {
  return Array.isArray(ops)
    ? ops.map((entry) => {
      const cloned = {
        operation: entry.operation === "subtract" ? "subtract" : "add",
        proposalIds: Array.isArray(entry.proposalIds)
          ? entry.proposalIds.map((value) => Number(value)).filter((value) => value > 0)
          : [],
        polygons: clonePolygonOps(entry.polygons),
      };
      if (entry.layer) cloned.layer = normalizeProposalLayer(entry.layer);
      if (typeof entry.maskPng === "string" && entry.maskPng.length) cloned.maskPng = entry.maskPng;
      const nested = cloneSelectionOps(entry.selectionOps);
      if (nested.length) cloned.selectionOps = nested;
      return cloned;
    }).filter((entry) => (
      entry.proposalIds.length ||
      entry.polygons.length ||
      entry.maskPng ||
      (entry.selectionOps && entry.selectionOps.length)
    ))
    : [];
}

function cloneSamPrompts(prompts) {
  return Array.isArray(prompts)
    ? prompts.map((prompt) => ({
      frameId: Number(prompt.frameId),
      sourceX: Number(prompt.sourceX),
      sourceY: Number(prompt.sourceY),
      label: Number(prompt.label) > 0 ? 1 : 0,
    }))
    : [];
}

function cloneRgbdCuePrompts(prompts) {
  return cloneSamPrompts(prompts);
}

function hasLassoPixelSelection() {
  return state.selectionOps.some((entry) => (
    Array.isArray(entry.polygons) &&
    entry.polygons.some((polygonOp) => Array.isArray(polygonOp.polygon) && polygonOp.polygon.length >= 3)
  ));
}

function hasSamMaskSelection() {
  return state.selectionOps.some((entry) => typeof entry.maskPng === "string" && entry.maskPng.length > 0);
}

function hasPixelSelection() {
  return state.selectionOps.length > 0;
}

function normalizeProposalId(value) {
  const id = Number(value) || 0;
  return id > 0 ? id : null;
}

function normalizeProposalLayer(value) {
  const key = String(value || "sam2").trim().toLowerCase().replaceAll("-", "_");
  if (key === "sam2" || key === "raw" || key === "auto" || key === "initial") return "sam2";
  const rows = state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.layers)
    ? state.proposalLayerStatus.layers
    : [];
  const match = rows.find((row) => (
    String(row.key || "").toLowerCase() === key ||
    String(row.methodId || "").toLowerCase() === key
  ));
  if (match) return String(match.key);
  return key;
}

function proposalLayerPriorityOrder() {
  const status = state.proposalLayerStatus || {};
  if (Array.isArray(status.pickOrder) && status.pickOrder.length) {
    return status.pickOrder.map(normalizeProposalLayer);
  }
  return ["sam2"];
}

function isPropagatedProposalLayer(layer) {
  const key = normalizeProposalLayer(layer);
  const row = proposalLayerStatus(key);
  return row ? row.labelSpace === "persistent_region" : key.startsWith("propagation_");
}

function proposalLayerDrawOrder(layers) {
  return [...layers].reverse();
}

function proposalLayerLabel(layer) {
  const key = normalizeProposalLayer(layer);
  const row = proposalLayerStatus(key);
  if (row && row.label) return String(row.label);
  return key === "sam2" ? "SAM2" : key;
}

function proposalRegionId(proposal) {
  const layer = normalizeProposalLayer(proposal && proposal.layer || "sam2");
  return Number(
    proposal && (
      proposal.regionId ||
      proposal.sourceRegionId ||
      (isPropagatedProposalLayer(layer) ? proposal.labelId : 0)
    )
  ) || 0;
}

function proposalRegionRow(proposal) {
  const rid = proposalRegionId(proposal);
  if (rid <= 0 || typeof regionRows !== "function") return null;
  return regionRows().find((region) => Number(region.id) === rid) || null;
}

function proposalRegionName(proposal) {
  const rid = proposalRegionId(proposal);
  const region = proposalRegionRow(proposal);
  if (region && region.name) return region.name;
  return rid > 0 ? `region_${rid}` : "";
}

function proposalDisplayName(proposal) {
  const layer = normalizeProposalLayer(proposal && proposal.layer ? proposal.layer : "sam2");
  const labelId = Number(proposal && proposal.labelId) || 0;
  if (isPropagatedProposalLayer(layer)) {
    const rid = proposalRegionId(proposal) || labelId;
    const name = proposalRegionName(proposal);
    return name ? `R${rid} ${name}` : `R${rid}`;
  }
  return `${proposalLayerLabel(layer)} ${labelId}`;
}

function proposalDisplayColor(proposal) {
  const layer = normalizeProposalLayer(proposal && proposal.layer ? proposal.layer : "sam2");
  const labelId = Number(proposal && proposal.labelId) || 0;
  if (isPropagatedProposalLayer(layer)) {
    const region = proposalRegionRow(proposal);
    if (region && typeof regionColor === "function") return regionColor(region);
    return palette((proposalRegionId(proposal) || labelId) + 900);
  }
  return palette(labelId + (layer === "sam2" ? 300 : 0));
}

function proposalCountText(proposal) {
  const pixels = Number(proposal.assignedPixels || proposal.rawAreaPixels || 0);
  const prefix = isPropagatedProposalLayer(proposal.layer || "sam2") ? "propagated" : "proposal";
  return pixels > 0 ? `${formatCount(pixels)} px | ${prefix}` : `${prefix} region`;
}

function proposalKey(layer, id) {
  const normalizedId = normalizeProposalId(id);
  if (normalizedId === null) return "";
  return `${normalizeProposalLayer(layer)}:${normalizedId}`;
}

function parseProposalKey(key) {
  const [layer, id] = String(key || "").split(":");
  return {
    layer: normalizeProposalLayer(layer),
    id: normalizeProposalId(id) || 0,
  };
}

function selectedProposalCount() {
  return state.selectedProposalIds.size;
}

function selectionSourceForProposalId(value, layer = "sam2") {
  const id = Number(value) || 0;
  if (id <= 0) return null;
  return { layer: normalizeProposalLayer(layer), proposalIds: [id] };
}

function appendSelectionSource(source, operation = "add") {
  if (!source) return false;
  const op = {
    operation: operation === "subtract" ? "subtract" : "add",
    proposalIds: Array.isArray(source.proposalIds) ? [...source.proposalIds] : [],
    polygons: clonePolygonOps(source.polygons),
  };
  if (source.layer) op.layer = normalizeProposalLayer(source.layer);
  if (typeof source.maskPng === "string" && source.maskPng.length) op.maskPng = source.maskPng;
  const nested = cloneSelectionOps(source.selectionOps);
  if (nested.length) op.selectionOps = nested;
  const hasSource = op.proposalIds.length || op.polygons.length || op.maskPng || (op.selectionOps && op.selectionOps.length);
  if (!hasSource) return false;
  state.selectionOps.push(op);
  return true;
}

function pushUndoSnapshot() {
  state.undoStack.push({
    selectedPointIndices: [...state.selectedPointIndices],
    selectedProposalIds: [...state.selectedProposalIds],
    activeProposalId: state.activeProposalId,
    activeProposalLayer: state.activeProposalLayer,
    selectionOps: cloneSelectionOps(state.selectionOps),
    selectionPreviewArea: state.selectionPreviewArea,
    selectionPreviewCoverage: state.selectionPreviewCoverage,
    selectionPreviewProtectedArea: state.selectionPreviewProtectedArea,
    selectionProtectRegions: state.selectionProtectRegions,
    samPrompts: cloneSamPrompts(state.samPrompts),
    rgbdCuePrompts: cloneRgbdCuePrompts(state.rgbdCuePrompts),
  });
  if (state.undoStack.length > 60) {
    state.undoStack.shift();
  }
}

function undoLastEdit() {
  if (!state.undoStack.length) return;
  const snapshot = state.undoStack.pop();
  state.selectedPointIndices = new Set(snapshot.selectedPointIndices);
  state.selectedProposalIds = new Set(snapshot.selectedProposalIds || []);
  state.activeProposalId = Number(snapshot.activeProposalId) || 0;
  state.activeProposalLayer = normalizeProposalLayer(snapshot.activeProposalLayer || "sam2");
  state.selectionOps = cloneSelectionOps(snapshot.selectionOps);
  state.selectionPreviewArea = Number(snapshot.selectionPreviewArea) || 0;
  state.selectionPreviewCoverage = Number(snapshot.selectionPreviewCoverage) || 0;
  state.selectionPreviewProtectedArea = Number(snapshot.selectionPreviewProtectedArea) || 0;
  state.selectionProtectRegions = Boolean(snapshot.selectionProtectRegions);
  if (lockRegionSelectionInput) lockRegionSelectionInput.checked = state.selectionProtectRegions;
  state.samPrompts = cloneSamPrompts(snapshot.samPrompts);
  if (state.samRefreshTimer) {
    clearTimeout(state.samRefreshTimer);
    state.samRefreshTimer = null;
  }
  state.samRefreshSerial += 1;
  state.rgbdCuePrompts = cloneRgbdCuePrompts(snapshot.rgbdCuePrompts);
  syncSelectionControls();
  syncSamControls();
  syncSelectionOperationControls();
  syncMaskEditControls();
  loadProposalSelectionOverlay();
  render();
  if (state.selectedFrame) {
    loadActiveProposalOverlay(state.selectedFrame.id);
  }
  setSelectionStatus("Undid last selection action");
  render();
}

function idPanelSectionHeader(text, controls = null) {
  const title = document.createElement("div");
  title.className = "id-section-title";
  const label = document.createElement("span");
  label.className = "id-section-title-text";
  label.textContent = text;
  title.appendChild(label);
  if (controls) title.appendChild(controls);
  return title;
}

function sortButtonGroup(label, buttons) {
  const group = document.createElement("div");
  group.className = "section-sort";
  group.setAttribute("aria-label", label);
  for (const spec of buttons) {
    const button = document.createElement("button");
    button.className = "section-sort-button";
    button.type = "button";
    button.textContent = spec.label;
    if (spec.dataset) {
      for (const [key, value] of Object.entries(spec.dataset)) {
        button.dataset[key] = String(value);
      }
    }
    if (spec.active) {
      button.classList.add("active");
      button.setAttribute("aria-pressed", "true");
    } else {
      button.setAttribute("aria-pressed", "false");
    }
    if (typeof spec.onClick === "function") {
      button.addEventListener("click", spec.onClick);
    }
    group.appendChild(button);
  }
  return group;
}

function renderIdPanel() {
  if (!idList) return;
  if (typeof renderMeshPartPanel === "function" && renderMeshPartPanel()) return;
  if (typeof renderGaussianPartPanel === "function" && renderGaussianPartPanel()) return;
  if (typeof renderObjectgsAnchorPanel === "function" && renderObjectgsAnchorPanel()) return;
  if (idPanel) idPanel.classList.remove("gaussian-part-mode");
  if (regionList) regionList.classList.remove("gaussian-part-list");
  if (typeof renderRegionPanel === "function") {
    renderRegionPanel();
  }
  const stats = idStats();
  const totalPoints = state.labelSummary ? state.labelSummary.pointCount : state.labels.length;
  const proposalRows = sortedProposalRows(
    state.proposalFrameInfo && Array.isArray(state.proposalFrameInfo.labels)
      ? state.proposalFrameInfo.labels
      : []
  ).filter((proposal) => state.proposalLayers[normalizeProposalLayer(proposal.layer || "sam2")]);
  syncProposalSortControls();
  idList.textContent = "";

  idList.appendChild(idPanelSectionHeader(
    `Proposals | ${formatCount(proposalRows.length)}`,
    proposalSortControl()
  ));

  if (!proposalRows.length) {
    const empty = document.createElement("div");
    empty.className = "id-empty";
    empty.textContent = "No visible candidate proposals yet. Enable SAM2 or propagated suggestions.";
    idList.appendChild(empty);
  }

  for (const proposal of proposalRows) {
    const labelId = Number(proposal.labelId) || 0;
    const layer = normalizeProposalLayer(proposal.layer || "sam2");
    const key = proposalKey(layer, labelId);
    if (labelId <= 0) continue;
    const row = document.createElement("div");
    row.className = "id-row frame-mask-row";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.classList.toggle("active", state.selectedProposalIds.has(key) || (state.activeProposalId === labelId && state.activeProposalLayer === layer));
    row.title = "Click to select this proposal region from its layer.";

    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    swatch.style.background = proposalDisplayColor(proposal);

    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("div");
    name.className = "id-name";
    name.textContent = proposalDisplayName(proposal);
    const count = document.createElement("div");
    count.className = "id-count";
    count.textContent = proposalCountText(proposal);
    main.append(name, count);

    const selected = document.createElement("span");
    selected.className = "id-selected";
    selected.textContent = state.activeProposalId === labelId && state.activeProposalLayer === layer ? "target" : state.selectedProposalIds.has(key) ? "selected" : "pick";
    row.append(swatch, main, selected);
    row.addEventListener("click", () => {
      applyProposalSelection([{ labelId, layer }], "replace");
      setSelectionStatus(`${proposalDisplayName(proposal)} selected`);
    });
    idList.appendChild(row);
  }

  if (stats.length) {
    idList.appendChild(idPanelSectionHeader(`Points | ${formatCount(totalPoints)} pts`));
  }

  for (const item of stats) {
    const row = document.createElement("div");
    row.className = "id-row";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.dataset.label = String(item.label);
    row.classList.toggle("active", item.selected > 0);
    row.classList.toggle("full", item.selected > 0 && item.selected === item.served);
    row.title = "Click to select this ID. Shift adds; Alt subtracts.";

    const swatch = document.createElement("span");
    swatch.className = "id-swatch";
    swatch.style.background = palette(item.label);

    const main = document.createElement("span");
    main.className = "id-main";
    const name = document.createElement("div");
    name.className = "id-name";
    name.textContent = `ID ${item.label}`;
    const count = document.createElement("div");
    count.className = "id-count";
    count.textContent = `${formatCount(item.count)} pts`;
    if (item.served !== item.count) {
      count.title = `${formatCount(item.served)} loaded in the current point sample`;
    }
    main.append(name, count);

    const selected = document.createElement("span");
    selected.className = "id-selected";
    selected.textContent = item.selected ? formatCount(item.selected) : "-";

    row.append(swatch, main, selected);
    row.addEventListener("click", (event) => {
      selectLabel(item.label, selectionOperationFromEvent(event));
    });
    row.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      selectLabel(item.label, selectionOperationFromEvent(event));
    });
    idList.appendChild(row);
  }
}

function proposalPixelCount(proposal) {
  return Number(proposal.assignedPixels || proposal.rawAreaPixels || 0);
}

function sortedProposalRows(rows) {
  const sorted = rows
    .map((proposal) => ({ ...proposal }))
    .filter((proposal) => Number(proposal.labelId) > 0);
  const layerRank = new Map(proposalLayerPriorityOrder().map((layer, index) => [layer, index]));
  if (state.proposalSortMode === "pixels") {
    sorted.sort((a, b) => {
      const pixelDelta = proposalPixelCount(b) - proposalPixelCount(a);
      if (pixelDelta !== 0) return pixelDelta;
      const layerDelta = (layerRank.get(normalizeProposalLayer(a.layer || "sam2")) ?? 999) - (layerRank.get(normalizeProposalLayer(b.layer || "sam2")) ?? 999);
      if (layerDelta !== 0) return layerDelta;
      return (Number(a.labelId) || 0) - (Number(b.labelId) || 0);
    });
    return sorted;
  }
  sorted.sort((a, b) => {
    const layerDelta = (layerRank.get(normalizeProposalLayer(a.layer || "sam2")) ?? 999) - (layerRank.get(normalizeProposalLayer(b.layer || "sam2")) ?? 999);
    if (layerDelta !== 0) return layerDelta;
    return (Number(a.labelId) || 0) - (Number(b.labelId) || 0);
  });
  return sorted;
}

function proposalSortControl() {
  return sortButtonGroup("Proposal sort mode", [
    {
      label: "Index",
      active: state.proposalSortMode === "index",
      dataset: { proposalSort: "index" },
      onClick: () => setProposalSortMode("index"),
    },
    {
      label: "Pixels",
      active: state.proposalSortMode === "pixels",
      dataset: { proposalSort: "pixels" },
      onClick: () => setProposalSortMode("pixels"),
    },
  ]);
}

function syncProposalSortControls() {
  for (const button of document.querySelectorAll("[data-proposal-sort]")) {
    const mode = button.dataset.proposalSort || "index";
    const active = mode === state.proposalSortMode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", active ? "true" : "false");
  }
}

function setProposalSortMode(mode) {
  state.proposalSortMode = mode === "pixels" ? "pixels" : "index";
  syncProposalSortControls();
  renderIdPanel();
}
