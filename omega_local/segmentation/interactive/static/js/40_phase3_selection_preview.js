// Phase 3 local proposal selection state and server-rendered selection previews.
function proposalSelectionOperationFromEvent(event) {
  return effectiveSelectionOperation(event);
}

function applyProposalSelection(labelIds, operation = "replace") {
  if (!Array.isArray(labelIds) || !labelIds.length) return;
  pushUndoSnapshot();
  const next = operation === "replace" ? new Set() : new Set(state.selectedProposalIds);
  if (operation === "replace") {
    state.selectionOps = [];
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
  }
  let firstAddedId = 0;
  let firstAddedLayer = "sam2";
  for (const item of labelIds) {
    const id = normalizeProposalId(typeof item === "object" ? item.labelId : item);
    const layer = normalizeProposalLayer(typeof item === "object" ? item.layer : state.activeProposalLayer || "sam2");
    if (id === null) continue;
    const source = selectionSourceForProposalId(id, layer);
    if (!appendSelectionSource(source, operation)) continue;
    const key = proposalKey(layer, id);
    if (operation !== "subtract" && firstAddedId <= 0) {
      firstAddedId = id;
      firstAddedLayer = layer;
    }
    if (operation === "subtract") next.delete(key);
    else next.add(key);
  }
  state.selectedProposalIds = next;
  if (operation === "replace") {
    state.activeProposalId = firstAddedId || 0;
    state.activeProposalLayer = firstAddedLayer;
  } else if (operation !== "subtract" && state.activeProposalId <= 0) {
    state.activeProposalId = firstAddedId || 0;
    state.activeProposalLayer = firstAddedLayer;
  } else if (operation === "subtract" && state.activeProposalId > 0 && !next.has(proposalKey(state.activeProposalLayer, state.activeProposalId))) {
    const first = next.size ? parseProposalKey([...next][0]) : { id: 0, layer: "sam2" };
    state.activeProposalId = first.id;
    state.activeProposalLayer = first.layer;
  }
  state.proposalSelectionOverlayStamp += 1;
  syncMaskEditControls();
  refreshSelectionPreview();
}

function clearProposalSelection() {
  if (!hasPixelSelection() && !state.selectedProposalIds.size && state.activeProposalId <= 0) return;
  pushUndoSnapshot();
  state.selectedProposalIds.clear();
  state.activeProposalId = 0;
  state.activeProposalLayer = "sam2";
  state.selectionOps = [];
  state.proposalSelectionOverlayImage = null;
  state.selectionPreviewArea = 0;
  state.selectionPreviewCoverage = 0;
  resetSamState();
  resetRgbdCuePrompts();
  syncMaskEditControls();
  syncSamControls();
  syncSelectionOperationControls();
  render();
}

async function loadProposalSelectionOverlay() {
  return refreshSelectionPreview();
}

async function refreshSelectionPreview() {
  if (!state.selectedFrame || !state.selectionOps.length) {
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
    state.selectionPreviewProtectedArea = 0;
    render();
    return;
  }
  const stamp = state.proposalSelectionOverlayStamp;
  const frameId = state.selectedFrame.id;
  try {
    const result = await postJson(
      `/api/proposals/sam2/frame/${frameId}/selection-preview`,
      {
        selectionOps: cloneSelectionOps(state.selectionOps),
        protectRegions: state.selectionProtectRegions,
      },
    );
    if (!state.selectedFrame || result.frameId !== state.selectedFrame.id || stamp !== state.proposalSelectionOverlayStamp) return;
    state.selectionPreviewArea = Number(result.areaPixels) || 0;
    state.selectionPreviewCoverage = Number(result.coverage) || 0;
    state.selectionPreviewProtectedArea = Number(result.protectedAreaPixels) || 0;
    state.proposalSelectionOverlayImage = state.selectionPreviewArea > 0
      ? await loadImage(`data:image/png;base64,${result.maskOverlayPng}`)
      : null;
  } catch (error) {
    console.error(error);
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
    state.selectionPreviewProtectedArea = 0;
    setSelectionStatus(error.message);
  }
  render();
}

function syncMaskEditControls() {
  const hasFrame = Boolean(state.selectedFrame && state.exactFrameView);
  const selectedCount = state.selectedProposalIds.size;
  const hasRegion = hasLassoPixelSelection();
  const hasPositiveRegion = hasPixelSelection();
  const proposalRows = state.proposalFrameInfo && Array.isArray(state.proposalFrameInfo.labels)
    ? state.proposalFrameInfo.labels
    : [];

  if (frameMaskStatusEl) {
    if (!state.selectedFrame) frameMaskStatusEl.textContent = "Select a frame";
    else frameMaskStatusEl.textContent = `${proposalRows.length.toLocaleString()} visible proposal ID${proposalRows.length === 1 ? "" : "s"} on frame ${state.selectedFrame.id}`;
  }
  if (selectedProposalStatusEl) {
    if (hasPixelSelection()) {
      const area = state.selectionPreviewArea > 0 ? `${state.selectionPreviewArea.toLocaleString()} px` : "previewing";
      const protectedText = state.selectionProtectRegions && state.selectionPreviewProtectedArea > 0
        ? ` | ${state.selectionPreviewProtectedArea.toLocaleString()} protected`
        : "";
      const proposalText = selectedCount
        ? `${selectedCount.toLocaleString()} proposal${selectedCount === 1 ? "" : "s"}`
        : "";
      const lassoText = hasRegion ? "lasso pixels" : "";
      const samText = hasSamMaskSelection() ? "mask pixels" : "";
      const layerText = selectedCount ? `${proposalLayerLabel(state.activeProposalLayer)} layer` : "";
      const sourceText = [layerText, proposalText, samText, lassoText].filter(Boolean).join(" | ") || `${state.selectionOps.length} pixel operation${state.selectionOps.length === 1 ? "" : "s"}`;
      selectedProposalStatusEl.textContent = `${area} selected${protectedText} | ${sourceText}`;
    } else {
      selectedProposalStatusEl.textContent = "No selected region";
    }
  }
  if (clearProposalSelectionButton) {
    clearProposalSelectionButton.disabled = !hasPixelSelection() && !state.selectedProposalIds.size && state.activeProposalId <= 0;
  }
  if (typeof syncRegionControls === "function") syncRegionControls();
  syncPropagationControls();
  renderIdPanel();
}

async function setRegionSelectionLock(enabled) {
  state.selectionProtectRegions = Boolean(enabled);
  if (lockRegionSelectionInput) lockRegionSelectionInput.checked = state.selectionProtectRegions;
  if (state.selectionOps.length) {
    state.proposalSelectionOverlayStamp += 1;
    await refreshSelectionPreview();
  } else {
    syncMaskEditControls();
    render();
  }
}
