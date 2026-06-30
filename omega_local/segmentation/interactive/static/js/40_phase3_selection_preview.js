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
  for (const label of labelIds) {
    const id = normalizeProposalId(label);
    if (id === null) continue;
    const source = selectionSourceForProposalId(id);
    if (!appendSelectionSource(source, operation)) continue;
    if (operation !== "subtract" && firstAddedId <= 0) firstAddedId = id;
    if (operation === "subtract") next.delete(id);
    else next.add(id);
  }
  state.selectedProposalIds = next;
  if (operation === "replace") {
    state.activeProposalId = firstAddedId || 0;
  } else if (operation !== "subtract" && state.activeProposalId <= 0) {
    state.activeProposalId = firstAddedId || 0;
  } else if (operation === "subtract" && state.activeProposalId > 0 && !next.has(state.activeProposalId)) {
    state.activeProposalId = next.size ? Number([...next][0]) || 0 : 0;
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
  state.selectionOps = [];
  state.proposalSelectionOverlayImage = null;
  state.selectionPreviewArea = 0;
  state.selectionPreviewCoverage = 0;
  resetSamState();
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
    render();
    return;
  }
  const stamp = state.proposalSelectionOverlayStamp;
  const frameId = state.selectedFrame.id;
  try {
    const result = await postJson(
      `/api/proposals/sam2/frame/${frameId}/selection-preview`,
      { selectionOps: cloneSelectionOps(state.selectionOps) },
    );
    if (!state.selectedFrame || result.frameId !== state.selectedFrame.id || stamp !== state.proposalSelectionOverlayStamp) return;
    state.selectionPreviewArea = Number(result.areaPixels) || 0;
    state.selectionPreviewCoverage = Number(result.coverage) || 0;
    state.proposalSelectionOverlayImage = state.selectionPreviewArea > 0
      ? await loadImage(`data:image/png;base64,${result.maskOverlayPng}`)
      : null;
  } catch (error) {
    console.error(error);
    state.proposalSelectionOverlayImage = null;
    state.selectionPreviewArea = 0;
    state.selectionPreviewCoverage = 0;
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
    else frameMaskStatusEl.textContent = `${proposalRows.length.toLocaleString()} editable proposal ID${proposalRows.length === 1 ? "" : "s"} on frame ${state.selectedFrame.id}`;
  }
  if (selectedProposalStatusEl) {
    if (hasPixelSelection()) {
      const area = state.selectionPreviewArea > 0 ? `${state.selectionPreviewArea.toLocaleString()} px` : "previewing";
      const proposalText = selectedCount
        ? `${selectedCount.toLocaleString()} proposal${selectedCount === 1 ? "" : "s"}`
        : "";
      const lassoText = hasRegion ? "lasso pixels" : "";
      const samText = hasSamMaskSelection() ? "SAM2 mask" : "";
      const targetText = state.activeProposalId > 0 ? `target ${state.activeProposalId}` : "auto ID";
      const sourceText = [targetText, proposalText, samText, lassoText].filter(Boolean).join(" | ") || `${state.selectionOps.length} pixel operation${state.selectionOps.length === 1 ? "" : "s"}`;
      selectedProposalStatusEl.textContent = `${area} selected | ${sourceText}`;
    } else {
      selectedProposalStatusEl.textContent = "No selected region";
    }
  }
  if (pendingMaskStatusEl) {
    pendingMaskStatusEl.textContent = "Proposal updates save immediately";
  }
  if (clearProposalSelectionButton) {
    clearProposalSelectionButton.disabled = !hasPixelSelection() && !state.selectedProposalIds.size && state.activeProposalId <= 0;
  }
  if (updateProposalButton) updateProposalButton.disabled = state.proposalUpdateBusy || !hasFrame || !hasPositiveRegion;
  syncPropagationControls();
  renderIdPanel();
}
