// Phase 1/2 UI state: point selection controls, proposal/keyframe/evidence status, and polling.
function selectedLabels() {
  const labels = new Set();
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    if (label > 0) labels.add(label);
  }
  return labels;
}

function selectedFullIndices() {
  const indices = [];
  for (const index of state.selectedPointIndices) {
    const sourceIndex = state.sourceIndices.length ? state.sourceIndices[index] : index;
    if (Number.isInteger(sourceIndex) && sourceIndex >= 0) indices.push(sourceIndex);
  }
  return indices;
}

function majoritySelectedLabel() {
  const counts = new Map();
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    if (label <= 0) continue;
    counts.set(label, (counts.get(label) || 0) + 1);
  }
  let bestLabel = 0;
  let bestCount = -1;
  for (const [label, count] of counts.entries()) {
    if (count > bestCount || (count === bestCount && label < bestLabel)) {
      bestLabel = label;
      bestCount = count;
    }
  }
  return bestLabel;
}

function nextAvailableLabel() {
  let maxLabel = 0;
  if (state.labelSummary && Array.isArray(state.labelSummary.labels)) {
    for (const item of state.labelSummary.labels) {
      maxLabel = Math.max(maxLabel, Number(item.label) || 0);
    }
  }
  for (const label of state.labels) {
    maxLabel = Math.max(maxLabel, label);
  }
  return maxLabel + 1;
}

function targetLabel() {
  const value = Number(targetLabelInput ? targetLabelInput.value : 0);
  return Number.isInteger(value) && value > 0 ? value : 0;
}

function suggestTargetLabel() {
  if (!targetLabelInput || document.activeElement === targetLabelInput) return;
  if (targetLabel() > 0) return;
  const label = majoritySelectedLabel();
  if (label > 0) {
    targetLabelInput.value = String(label);
  }
}

function applySelection(indices, operation = "replace") {
  const next = operation === "replace" ? new Set() : new Set(state.selectedPointIndices);
  if (operation === "subtract") {
    for (const index of indices) next.delete(index);
  } else {
    for (const index of indices) next.add(index);
  }
  state.selectedPointIndices = next;
  syncSelectionControls();
}

function selectLabel(label, operation = "replace") {
  const indices = indicesForLabel(label);
  pushUndoSnapshot();
  applySelection(indices, operation);
  setSelectionStatus(`ID ${label}: ${formatCount(indices.length)} pts | ${formatCount(state.selectedPointIndices.size)} selected`);
  render();
}

function clearSelection() {
  if (!state.selectedPointIndices.size) return;
  pushUndoSnapshot();
  state.selectedPointIndices.clear();
  syncSelectionControls();
  render();
}

function syncSelectionControls() {
  if (clearSelectionButton) {
    clearSelectionButton.disabled = state.selectedPointIndices.size === 0;
  }
  suggestTargetLabel();
  const selectedLabelCount = selectedLabels().size;
  const hasSelection = state.selectedPointIndices.size > 0;
  if (assignSelectionButton) {
    assignSelectionButton.disabled = state.editBusy || !hasSelection || targetLabel() <= 0;
  }
  if (extractSelectionButton) {
    extractSelectionButton.disabled = state.editBusy || !hasSelection;
  }
  if (mergeSelectedIdsButton) {
    const labels = selectedLabels();
    const target = targetLabel();
    const mergesSomething = selectedLabelCount > 1 || (selectedLabelCount === 1 && !labels.has(target));
    mergeSelectedIdsButton.disabled = state.editBusy || !mergesSomething || target <= 0;
  }
  const count = state.selectedPointIndices.size;
  setSelectionStatus(count ? `${count.toLocaleString()} selected` : "No selection");
  renderIdPanel();
  syncSaveControls();
}

function syncToolButtons() {
  for (const button of toolButtons) {
    button.classList.toggle("active", button.dataset.tool === state.tool);
  }
  canvas.classList.toggle("selecting", state.tool !== "navigate");
  syncSelectionOperationControls();
  syncSamControls();
}

function setTool(tool) {
  state.tool = tool;
  state.lasso = null;
  syncToolButtons();
  render();
}

function syncSamControls() {
  const proposalRunning = Boolean(state.proposalStatus && state.proposalStatus.running);
  if (runSam2Button) {
    runSam2Button.disabled = state.samBusy || proposalRunning || state.samPrompts.length === 0 || !state.selectedFrame;
    runSam2Button.title = "Run SAM2 with the current positive and negative prompts to create a temporary proposal selection";
  }
  if (clearSam2Button) {
    clearSam2Button.disabled = state.samBusy || state.samPrompts.length === 0;
  }
}

function proposalReady() {
  return Boolean(state.proposalStatus && state.proposalStatus.ready && !state.proposalStatus.failed);
}

function proposalOverlayStamp() {
  const stamp = state.proposalStatus
    ? state.proposalStatus.overlayUpdatedUtc ||
      state.proposalStatus.editableUpdatedUtc ||
      state.proposalStatus.updatedUtc ||
      state.proposalStatus.completedFrameCount ||
      "ready"
    : "ready";
  return stamp;
}

function proposalOverlayUrl(frameId) {
  const stamp = proposalOverlayStamp();
  return `/api/proposals/sam2/frame/${frameId}/overlay?v=${encodeURIComponent(stamp)}`;
}

function viewEvidenceReady() {
  const status = state.viewEvidenceStatus || {};
  return Boolean(
    (evidenceKindStatus("normal").generatedFrameCount || 0) > 0 ||
    (evidenceKindStatus("depth").generatedFrameCount || 0) > 0
  );
}

function viewEvidenceTotalFrames() {
  return Number(state.viewEvidenceStatus && state.viewEvidenceStatus.frameCount) || state.frames.length || 0;
}

function evidenceKindStatus(kind) {
  const status = state.viewEvidenceStatus || {};
  const key = kind === "depth" ? "depthStatus" : "normalStatus";
  const nested = status[key] && typeof status[key] === "object" ? status[key] : {};
  const topCount = kind === "depth" ? Number(status.depthFrameCount) || 0 : Number(status.normalFrameCount) || 0;
  const total = Number(nested.frameCount) || Number(status.frameCount) || state.frames.length || 0;
  return {
    target: kind,
    ready: Boolean(nested.ready),
    running: Boolean(nested.running),
    failed: Boolean(nested.failed),
    frameCount: total,
    completedFrameCount: Number(nested.completedFrameCount) || 0,
    generatedFrameCount: Number(nested.generatedFrameCount) || topCount,
    currentFrameId: nested.currentFrameId,
    message: typeof nested.message === "string" ? nested.message : "",
    updatedUtc: nested.updatedUtc || "",
  };
}

function normalEvidenceCount() {
  return evidenceKindStatus("normal").generatedFrameCount || 0;
}

function depthEvidenceCount() {
  return evidenceKindStatus("depth").generatedFrameCount || 0;
}

function normalEvidenceComplete() {
  return Boolean(evidenceKindStatus("normal").ready);
}

function depthEvidenceComplete() {
  return Boolean(evidenceKindStatus("depth").ready);
}

function viewEvidenceModeReady(mode) {
  const value = String(mode || "rgb");
  if (value === "rgb") return true;
  if (value.startsWith("normal")) return normalEvidenceCount() > 0;
  if (value.startsWith("depth")) return depthEvidenceCount() > 0;
  return false;
}

function viewEvidenceStamp() {
  const status = state.viewEvidenceStatus || {};
  const normalStatus = evidenceKindStatus("normal");
  const depthStatus = evidenceKindStatus("depth");
  return [
    normalStatus.updatedUtc,
    depthStatus.updatedUtc,
    status.updatedUtc,
    status.timestampUtc,
    status.completedFrameCount,
  ].filter(Boolean).join(":") || "0";
}

function viewEvidenceImageUrl(frameId, mode = state.frameBackgroundMode) {
  const stamp = viewEvidenceStamp();
  return `/api/view-evidence/frame/${frameId}/${mode}?v=${encodeURIComponent(stamp)}`;
}

function keyframeRowsByFrame() {
  const rows = state.keyframeStatus && Array.isArray(state.keyframeStatus.frames)
    ? state.keyframeStatus.frames
    : [];
  const map = new Map();
  for (const row of rows) {
    map.set(Number(row.frameId), row);
  }
  return map;
}

function keyframeProgressText(status) {
  if (!status) return "Keyframes not loaded";
  if (status.failed) return "Keyframe detection failed";
  if (!status.ready) return "Keyframes not detected";
  const count = Number(status.keyframeCount) || 0;
  const total = Number(status.frameCount) || state.frames.length;
  return `${count.toLocaleString()} keyframes / ${total.toLocaleString()} frames`;
}

function formatKeyframeTileTitle(baseTitle, row) {
  if (!row) return baseTitle;
  const score = Number(row.combinedScore) || 0;
  const image = Number(row.imageScore) || 0;
  const translation = Number(row.translationScore) || 0;
  const rotation = Number(row.rotationScore) || 0;
  const keyText = row.isKeyframe ? `keyframe: ${row.reason || "change"}` : "not keyframe";
  return `${baseTitle}\n${keyText}\nscore ${score.toFixed(3)} | image ${image.toFixed(2)} | pose ${translation.toFixed(2)} / ${rotation.toFixed(2)}`;
}

function updateKeyframeThumbnails() {
  const rows = keyframeRowsByFrame();
  const ready = Boolean(state.keyframeStatus && state.keyframeStatus.ready);
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    const frameId = Number(tile.dataset.frameId);
    const row = rows.get(frameId);
    const score = row ? clamp(Number(row.combinedScore) || 0, 0, 1) : 0;
    tile.classList.toggle("keyframe", Boolean(row && row.isKeyframe));
    tile.classList.toggle("has-keyframe-score", ready && Boolean(row));
    tile.style.setProperty("--keyframe-score", score.toFixed(4));
    tile.title = formatKeyframeTileTitle(tile.dataset.baseTitle || tile.title || "", row);
    const marker = tile.querySelector(".keyframe-marker");
    if (marker) {
      marker.hidden = !(row && row.isKeyframe);
      marker.title = row && row.isKeyframe ? `Suggested keyframe: ${row.reason || "change"}` : "";
    }
  }
}

function syncKeyframeControls() {
  if (keyframeStatusEl) {
    keyframeStatusEl.textContent = state.keyframeBusy ? "Detecting keyframes..." : keyframeProgressText(state.keyframeStatus);
    keyframeStatusEl.title = state.keyframeStatus && state.keyframeStatus.summaryPath ? state.keyframeStatus.summaryPath : "";
  }
  if (detectKeyframesButton) detectKeyframesButton.disabled = state.keyframeBusy;
  if (reloadKeyframesButton) reloadKeyframesButton.disabled = state.keyframeBusy;
  updateKeyframeThumbnails();
}

function proposalProgressText(status) {
  if (!status) return "Not run";
  const total = Number(status.frameCount) || 0;
  const done = Number(status.completedFrameCount) || 0;
  if (status.failed) return "Failed";
  if (status.running) return total ? `${done}/${total}` : "Running";
  if (status.ready) return total ? `Ready ${done}/${total}` : "Ready";
  return "Not run";
}

function updateProposalThumbnails() {
  const ready = proposalReady();
  filmstrip.classList.toggle("show-proposals", state.showProposals && ready);
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    const overlay = tile.querySelector(".proposal-thumb");
    if (!overlay) continue;
    const frameId = Number(tile.dataset.frameId);
    const stamp = proposalOverlayStamp();
    const loadedFor = `${frameId}:${stamp}`;
    if (ready && overlay.dataset.loadedFor !== loadedFor) {
      overlay.src = proposalOverlayUrl(frameId);
      overlay.dataset.loadedFor = loadedFor;
    }
  }
}

function syncProposalControls() {
  const status = state.proposalStatus;
  if (proposalStatusEl) {
    proposalStatusEl.textContent = proposalProgressText(status);
    proposalStatusEl.title = status && status.message ? status.message : "";
  }
  if (runProposalsButton) {
    runProposalsButton.disabled = Boolean(state.samBusy || (status && status.running) || (status && status.ready));
    runProposalsButton.textContent = status && status.ready ? "Generated" : "Generate";
  }
  if (loadProposalsButton) {
    loadProposalsButton.disabled = Boolean(!proposalReady() || (status && status.running));
  }
  if (regenerateProposalsButton) {
    regenerateProposalsButton.disabled = Boolean(state.samBusy || (status && status.running) || !proposalReady());
  }
  if (showProposalsInput) {
    showProposalsInput.disabled = !proposalReady();
    showProposalsInput.checked = state.showProposals && proposalReady();
  }
  const group = runProposalsButton ? runProposalsButton.closest(".proposal-group") : null;
  if (group) {
    group.classList.toggle("running", Boolean(status && status.running));
    group.classList.toggle("ready", proposalReady());
    group.classList.toggle("failed", Boolean(status && status.failed));
  }
  syncPropagationControls();
  updateProposalThumbnails();
}

function viewEvidenceProgressText(status) {
  if (!status) return "Evidence not loaded";
  const total = Number(status.frameCount) || state.frames.length;
  const normalStatus = evidenceKindStatus("normal");
  const depthStatus = evidenceKindStatus("depth");
  if (normalStatus.running) return `StableNormal: ${normalStatus.completedFrameCount.toLocaleString()} / ${total.toLocaleString()}`;
  if (depthStatus.running) return `Depth: ${depthStatus.completedFrameCount.toLocaleString()} / ${total.toLocaleString()}`;
  const normalText = normalStatus.failed
    ? "N failed"
    : `N ${normalStatus.generatedFrameCount.toLocaleString()}/${total.toLocaleString()}`;
  const depthText = depthStatus.failed
    ? "D failed"
    : `D ${depthStatus.generatedFrameCount.toLocaleString()}/${total.toLocaleString()}`;
  if (status.ready || normalStatus.generatedFrameCount || depthStatus.generatedFrameCount || normalStatus.failed || depthStatus.failed) {
    return `${normalText} | ${depthText}`;
  }
  return "Not generated";
}

function syncViewEvidenceControls() {
  const status = state.viewEvidenceStatus;
  const running = Boolean(status && status.running);
  const total = viewEvidenceTotalFrames();
  const normalStatus = evidenceKindStatus("normal");
  const depthStatus = evidenceKindStatus("depth");
  const normalCount = normalStatus.generatedFrameCount || 0;
  const depthCount = depthStatus.generatedFrameCount || 0;
  if (viewEvidenceStatusEl) {
    viewEvidenceStatusEl.textContent = viewEvidenceProgressText(status);
    viewEvidenceStatusEl.title = [
      normalStatus.message ? `StableNormal: ${normalStatus.message}` : "",
      depthStatus.message ? `Depth: ${depthStatus.message}` : "",
    ].filter(Boolean).join("\n") || (status && status.message ? status.message : "");
  }
  if (normalEvidenceStatusEl) {
    normalEvidenceStatusEl.textContent = normalStatus.running
      ? `${normalStatus.completedFrameCount.toLocaleString()} / ${total.toLocaleString()} frames`
      : normalStatus.failed
        ? "Failed"
        : normalCount > 0
      ? `${normalCount.toLocaleString()} / ${total.toLocaleString()} frames`
      : "Not generated";
    normalEvidenceStatusEl.title = normalStatus.message || "";
  }
  if (depthEvidenceStatusEl) {
    depthEvidenceStatusEl.textContent = depthStatus.running
      ? `${depthStatus.completedFrameCount.toLocaleString()} / ${total.toLocaleString()} frames`
      : depthStatus.failed
        ? "Failed"
        : depthCount > 0
      ? `${depthCount.toLocaleString()} / ${total.toLocaleString()} frames`
      : "Not generated";
    depthEvidenceStatusEl.title = depthStatus.message || "";
  }
  if (generateNormalEvidenceButton) {
    const isNormalJob = normalStatus.running;
    generateNormalEvidenceButton.disabled = running;
    generateNormalEvidenceButton.textContent = isNormalJob
      ? "Generating"
      : normalStatus.failed
        ? "Retry"
        : normalEvidenceComplete()
        ? "Regenerate"
        : "Generate";
    generateNormalEvidenceButton.title = normalStatus.message || "";
  }
  if (generateDepthEvidenceButton) {
    const isDepthJob = depthStatus.running;
    generateDepthEvidenceButton.disabled = running;
    generateDepthEvidenceButton.textContent = isDepthJob
      ? "Generating"
      : depthStatus.failed
        ? "Retry"
        : depthEvidenceComplete()
        ? "Regenerate"
        : "Generate";
    generateDepthEvidenceButton.title = depthStatus.message || "";
  }
  if (frameBackgroundModeInput) {
    frameBackgroundModeInput.disabled = false;
  }
}

async function refreshViewEvidenceStatus() {
  try {
    state.viewEvidenceStatus = await loadJson("/api/view-evidence/status");
  } catch (error) {
    console.error(error);
    state.viewEvidenceStatus = {
      ready: false,
      running: false,
      failed: true,
      message: error.message,
    };
  }
  syncViewEvidenceControls();
  if (state.viewEvidenceStatus && state.viewEvidenceStatus.running) {
    startViewEvidencePolling();
  }
  return state.viewEvidenceStatus;
}

function startViewEvidencePolling() {
  if (state.viewEvidencePollTimer) return;
  state.viewEvidencePollTimer = window.setInterval(async () => {
    await refreshViewEvidenceStatus();
    if (!state.viewEvidenceStatus || !state.viewEvidenceStatus.running) {
      window.clearInterval(state.viewEvidencePollTimer);
      state.viewEvidencePollTimer = null;
      if (state.selectedFrame && state.frameBackgroundMode !== "rgb") {
        loadSelectedFrameImage(state.selectedFrame);
      }
    }
  }, 1200);
}

async function runViewEvidenceTarget(target) {
  if (state.viewEvidenceStatus && state.viewEvidenceStatus.running) return;
  const targetName = target === "depth" ? "Depth Anything V2" : "StableNormal";
  const statusForTarget = evidenceKindStatus(target);
  const overwrite = target === "depth" ? depthEvidenceComplete() : normalEvidenceComplete();
  if (viewEvidenceStatusEl) viewEvidenceStatusEl.textContent = overwrite ? `Regenerating ${targetName}` : `Generating ${targetName}`;
  try {
    state.viewEvidenceStatus = await postJson("/api/view-evidence/run", {
      target,
      overwrite,
    });
    syncViewEvidenceControls();
    if (state.viewEvidenceStatus && state.viewEvidenceStatus.running) {
      startViewEvidencePolling();
    }
  } catch (error) {
    console.error(error);
    const key = target === "depth" ? "depthStatus" : "normalStatus";
    state.viewEvidenceStatus = {
      ...(state.viewEvidenceStatus || {}),
      ready: false,
      running: false,
      failed: false,
      message: error.message,
      [key]: {
        ...statusForTarget,
        running: false,
        failed: true,
        message: error.message,
      },
    };
    syncViewEvidenceControls();
    setSelectionStatus(error.message);
  }
}
