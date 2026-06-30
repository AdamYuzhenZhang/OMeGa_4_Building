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

function cloneLabelSummary(summary) {
  if (!summary) return null;
  return {
    ...summary,
    labels: Array.isArray(summary.labels)
      ? summary.labels.map((item) => ({ label: Number(item.label), count: Number(item.count) }))
      : [],
  };
}

function labelSummaryMap() {
  const map = new Map();
  if (state.labelSummary && Array.isArray(state.labelSummary.labels)) {
    for (const item of state.labelSummary.labels) {
      const label = Number(item.label);
      if (label > 0) map.set(label, Number(item.count) || 0);
    }
  }
  return map;
}

function setLabelSummaryFromMap(map) {
  const labels = [...map.entries()]
    .filter(([label, count]) => label > 0 && count > 0)
    .sort((a, b) => a[0] - b[0])
    .map(([label, count]) => ({ label, count }));
  const pointCount = state.labelSummary ? state.labelSummary.pointCount : state.labels.length;
  state.labelSummary = {
    pointCount,
    labelCount: labels.length,
    labels,
  };
}

function adjustLabelSummaryForAssignments(oldLabels, target) {
  const map = labelSummaryMap();
  for (const label of oldLabels) {
    if (label > 0) map.set(label, Math.max(0, (map.get(label) || 0) - 1));
  }
  map.set(target, (map.get(target) || 0) + oldLabels.length);
  setLabelSummaryFromMap(map);
}

function adjustLabelSummaryForMerge(labels, target) {
  const map = labelSummaryMap();
  let moved = 0;
  for (const label of labels) {
    if (label === target) continue;
    moved += map.get(label) || 0;
    map.delete(label);
  }
  map.set(target, (map.get(target) || 0) + moved);
  setLabelSummaryFromMap(map);
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

function selectionSourceForProposalId(value) {
  const id = Number(value) || 0;
  if (id <= 0) return null;
  return { proposalIds: [id] };
}

function appendSelectionSource(source, operation = "add") {
  if (!source) return false;
  const op = {
    operation: operation === "subtract" ? "subtract" : "add",
    proposalIds: Array.isArray(source.proposalIds) ? [...source.proposalIds] : [],
    polygons: clonePolygonOps(source.polygons),
  };
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
    labels: new Int32Array(state.labels),
    labelSummary: cloneLabelSummary(state.labelSummary),
    selectedPointIndices: [...state.selectedPointIndices],
    selectedProposalIds: [...state.selectedProposalIds],
    activeProposalId: state.activeProposalId,
    selectionOps: cloneSelectionOps(state.selectionOps),
    selectionPreviewArea: state.selectionPreviewArea,
    selectionPreviewCoverage: state.selectionPreviewCoverage,
    samPrompts: cloneSamPrompts(state.samPrompts),
    pendingEditCount: state.pendingEdits.length,
    targetValue: targetLabelInput ? targetLabelInput.value : "",
  });
  if (state.undoStack.length > 60) {
    state.undoStack.shift();
  }
}

function undoLastEdit() {
  if (state.editBusy || !state.undoStack.length) return;
  const snapshot = state.undoStack.pop();
  state.labels = snapshot.labels;
  state.labelSummary = snapshot.labelSummary;
  state.selectedPointIndices = new Set(snapshot.selectedPointIndices);
  state.selectedProposalIds = new Set(snapshot.selectedProposalIds || []);
  state.activeProposalId = Number(snapshot.activeProposalId) || 0;
  state.selectionOps = cloneSelectionOps(snapshot.selectionOps);
  state.selectionPreviewArea = Number(snapshot.selectionPreviewArea) || 0;
  state.selectionPreviewCoverage = Number(snapshot.selectionPreviewCoverage) || 0;
  state.samPrompts = cloneSamPrompts(snapshot.samPrompts);
  state.pendingEdits = state.pendingEdits.slice(0, snapshot.pendingEditCount);
  if (targetLabelInput) targetLabelInput.value = snapshot.targetValue;
  syncSelectionControls();
  syncSamControls();
  syncSelectionOperationControls();
  syncMaskEditControls();
  syncSaveControls();
  loadProposalSelectionOverlay();
  render();
  if (state.selectedFrame) {
    loadActiveProposalOverlay(state.selectedFrame.id);
  }
  setSelectionStatus(hasUnsavedEdits() ? `Undid last action | ${unsavedEditCount()} unsaved` : "Undid last action");
  render();
}

function renderIdPanel() {
  if (!idList || !idPanelSummary) return;
  const stats = idStats();
  const totalPoints = state.labelSummary ? state.labelSummary.pointCount : state.labels.length;
  const proposalRows = sortedProposalRows(
    state.proposalFrameInfo && Array.isArray(state.proposalFrameInfo.labels)
      ? state.proposalFrameInfo.labels
      : []
  );
  idPanelSummary.textContent = proposalRows.length
    ? `${formatCount(proposalRows.length)} proposals`
    : "No frame proposals";
  syncProposalSortControls();
  idList.textContent = "";

  if (proposalRows.length) {
    const title = document.createElement("div");
    title.className = "id-section-title";
    title.textContent = "Active Frame Proposal IDs";
    idList.appendChild(title);
    for (const proposal of proposalRows) {
      const labelId = Number(proposal.labelId) || 0;
      if (labelId <= 0) continue;
      const row = document.createElement("div");
      row.className = "id-row frame-mask-row";
      row.tabIndex = 0;
      row.setAttribute("role", "button");
      row.classList.toggle("active", state.selectedProposalIds.has(labelId) || state.activeProposalId === labelId);
      row.title = "Click to select this proposal ID as the update target.";

      const swatch = document.createElement("span");
      swatch.className = "id-swatch";
      swatch.style.background = palette(labelId);

      const main = document.createElement("span");
      main.className = "id-main";
      const name = document.createElement("div");
      name.className = "id-name";
      name.textContent = `proposal_${labelId}`;
      const count = document.createElement("div");
      count.className = "id-count";
      const pixels = Number(proposal.assignedPixels || proposal.rawAreaPixels || 0);
      count.textContent = pixels > 0 ? `${formatCount(pixels)} px` : "proposal region";
      main.append(name, count);

      const selected = document.createElement("span");
      selected.className = "id-selected";
      selected.textContent = state.activeProposalId === labelId ? "target" : state.selectedProposalIds.has(labelId) ? "selected" : "pick";
      row.append(swatch, main, selected);
      row.addEventListener("click", () => {
        applyProposalSelection([labelId], "replace");
        setSelectionStatus(`Proposal ${labelId} selected as target`);
      });
      idList.appendChild(row);
    }
  }

  if (!proposalRows.length && !stats.length) {
    const empty = document.createElement("div");
    empty.className = "id-empty";
    empty.textContent = "No edited proposals yet. Use Pick or Lasso, then update the frame proposal.";
    idList.appendChild(empty);
    return;
  }

  if (stats.length) {
    const title = document.createElement("div");
    title.className = "id-section-title";
    title.textContent = `Point IDs | ${formatCount(totalPoints)} pts`;
    idList.appendChild(title);
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
  if (state.proposalSortMode === "pixels") {
    sorted.sort((a, b) => {
      const pixelDelta = proposalPixelCount(b) - proposalPixelCount(a);
      if (pixelDelta !== 0) return pixelDelta;
      return (Number(a.labelId) || 0) - (Number(b.labelId) || 0);
    });
    return sorted;
  }
  sorted.sort((a, b) => (Number(a.labelId) || 0) - (Number(b.labelId) || 0));
  return sorted;
}

function syncProposalSortControls() {
  for (const button of proposalSortButtons) {
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
