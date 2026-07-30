// Phase 1/2 UI state: point selection controls, proposal/keyframe/evidence status, and polling.
function selectedLabels() {
  const labels = new Set();
  for (const index of state.selectedPointIndices) {
    const label = state.labels[index];
    if (label > 0) labels.add(label);
  }
  return labels;
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
  const count = state.selectedPointIndices.size;
  setSelectionStatus(count ? `${count.toLocaleString()} selected` : "No selection");
  renderIdPanel();
}

function proposalReady() {
  return Boolean(state.proposalStatus && state.proposalStatus.ready && !state.proposalStatus.failed);
}

function proposalOverlayStamp() {
  const stamp = state.proposalStatus
    ? state.proposalStatus.overlayUpdatedUtc ||
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

function proposalLayerStatus(layer) {
  const key = normalizeProposalLayer(layer);
  const layers = state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.layers)
    ? state.proposalLayerStatus.layers
    : [];
  return layers.find((item) => normalizeProposalLayer(item.key) === key) || null;
}

function proposalLayerReady(layer) {
  const status = proposalLayerStatus(layer);
  return Boolean(status && status.ready);
}

function proposalLayerStamp(layer) {
  const status = proposalLayerStatus(layer);
  return status
    ? status.updatedUtc || status.completedFrameCount || proposalOverlayStamp()
    : proposalOverlayStamp();
}

function proposalLayerOverlayUrl(frameId, layer) {
  const key = normalizeProposalLayer(layer);
  const stamp = proposalLayerStamp(key);
  return `/api/proposals/layers/${key}/frame/${frameId}/overlay?v=${encodeURIComponent(stamp)}`;
}

function visibleProposalLayers() {
  return proposalLayerPriorityOrder().filter((layer) => (
    Boolean(state.proposalLayers[layer]) &&
    proposalLayerReady(layer)
  ));
}

function pickProposalLayers() {
  return visibleProposalLayers();
}

function proposalLayerInputElements() {
  return proposalLayerOptionsEl
    ? [...proposalLayerOptionsEl.querySelectorAll("[data-proposal-layer]")]
    : [];
}

function initializeProposalLayerState() {
  const rows = state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.layers)
    ? state.proposalLayerStatus.layers
    : [];
  const next = {};
  for (const row of rows) {
    const key = normalizeProposalLayer(row.key || "sam2");
    next[key] = Boolean(state.proposalLayers[key]);
  }
  if (!("sam2" in next)) next.sam2 = false;
  state.proposalLayers = next;
}

const PROPOSAL_LAYER_GROUPS = [
  { key: "frame_proposals", label: "Frame Proposals" },
  { key: "refined_masks", label: "Split-Refined Masks" },
  { key: "geometry_support", label: "3D Evidence" },
  { key: "video_propagation", label: "Video Propagation" },
  { key: "sparse_3d_transfer", label: "Sparse 3D Transfer" },
  { key: "identity_refinement", label: "Identity Refinement" },
  { key: "dense_colmap", label: "Dense From COLMAP" },
  { key: "dense_omega_init", label: "Dense From OMeGa Init" },
  { key: "dense_omega_final", label: "Dense From OMeGa Final" },
  { key: "pairwise_transfer", label: "Pairwise Transfer Tests" },
];

function groupedProposalLayers(rows) {
  const knownGroups = new Map(PROPOSAL_LAYER_GROUPS.map((group) => [group.key, []]));
  const otherRows = [];
  for (const row of rows) {
    const group = String(row.layerGroup || "");
    const groupRows = knownGroups.get(group);
    if (groupRows) groupRows.push(row);
    else otherRows.push(row);
  }
  const groups = PROPOSAL_LAYER_GROUPS
    .map((group) => ({ ...group, rows: knownGroups.get(group.key) }))
    .filter((group) => group.rows.length > 0);
  if (otherRows.length) groups.push({ key: "other", label: "Other Layers", rows: otherRows });
  return groups;
}

function proposalLayerControl(row) {
  const layer = normalizeProposalLayer(row.key || "sam2");
  const label = document.createElement("label");
  label.className = "control checkbox";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.dataset.proposalLayer = layer;
  input.checked = Boolean(state.proposalLayers[layer]);
  input.disabled = !Boolean(row.ready);
  label.classList.toggle("disabled", input.disabled);
  label.title = row.ready
    ? String(row.description || "")
    : String(row.availabilityMessage || `${row.label || layer} has not been run.`);
  const text = document.createElement("span");
  text.textContent = String(row.label || layer);
  label.append(input, text);
  input.addEventListener("change", () => {
    setProposalLayerVisible(layer, input.checked).catch((error) => {
      console.error(error);
      setSelectionStatus(error.message);
    });
  });
  return label;
}

async function setProposalLayerVisible(layer, visible) {
  const key = normalizeProposalLayer(layer);
  state.proposalLayers[key] = Boolean(visible);
  state.proposalOverlayImages.clear();
  state.proposalLayerOverlayImages.clear();
  updateProposalThumbnails();
  syncProposalControls();
  if (state.selectedFrame) {
    await loadFrameProposalLayers(state.selectedFrame.id);
  } else {
    render();
  }
}

function renderProposalLayerControls() {
  if (!proposalLayerOptionsEl) return;
  const rows = state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.layers)
    ? state.proposalLayerStatus.layers
    : [];
  const visibleCount = rows.filter((row) => (
    Boolean(state.proposalLayers[normalizeProposalLayer(row.key || "sam2")]) && Boolean(row.ready)
  )).length;
  if (proposalLayerSummaryEl) {
    proposalLayerSummaryEl.textContent = visibleCount === 1
      ? "1 visible"
      : `${visibleCount.toLocaleString()} visible`;
  }
  proposalLayerOptionsEl.textContent = "";
  for (const group of groupedProposalLayers(rows)) {
    const section = document.createElement("section");
    section.className = "proposal-layer-group";
    section.dataset.layerGroup = group.key;
    const title = document.createElement("div");
    title.className = "proposal-layer-group-title";
    title.textContent = group.label;
    const options = document.createElement("div");
    options.className = "proposal-layer-group-options";
    for (const row of group.rows) options.appendChild(proposalLayerControl(row));
    section.append(title, options);
    proposalLayerOptionsEl.appendChild(section);
  }
}

function viewEvidenceReady() {
  const status = state.viewEvidenceStatus || {};
  return Boolean(
    (evidenceKindStatus("normal").generatedFrameCount || 0) > 0 ||
    (evidenceKindStatus("depth").generatedFrameCount || 0) > 0 ||
    (evidenceKindStatus("dinov3").generatedFrameCount || 0) > 0
  );
}

function viewEvidenceTotalFrames() {
  return Number(state.viewEvidenceStatus && state.viewEvidenceStatus.frameCount) || state.frames.length || 0;
}

function evidenceKindStatus(kind) {
  const status = state.viewEvidenceStatus || {};
  const key = kind === "depth" ? "depthStatus" : kind === "dinov3" ? "dinoStatus" : "normalStatus";
  const nested = status[key] && typeof status[key] === "object" ? status[key] : {};
  const topCount = kind === "depth"
    ? Number(status.depthFrameCount) || 0
    : kind === "dinov3"
      ? Number(status.dinoFrameCount) || 0
      : Number(status.normalFrameCount) || 0;
  const total = Number(nested.frameCount) || Number(status.frameCount) || state.frames.length || 0;
  return {
    target: kind,
    ready: Boolean(nested.ready),
    running: Boolean(nested.running),
    failed: Boolean(nested.failed),
    frameCount: total,
    completedFrameCount: Number(nested.completedFrameCount) || 0,
    generatedFrameCount: Number(nested.generatedFrameCount) || topCount,
    featureFrameCount: Number(nested.featureFrameCount) || 0,
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

function dinoEvidenceCount() {
  return evidenceKindStatus("dinov3").generatedFrameCount || 0;
}

function normalEvidenceComplete() {
  return Boolean(evidenceKindStatus("normal").ready);
}

function depthEvidenceComplete() {
  return Boolean(evidenceKindStatus("depth").ready);
}

function evidenceButtonLabel(kind) {
  const status = evidenceKindStatus(kind);
  if (status.running) return "Generating";
  if (status.failed) return "Retry";
  if (status.ready) return "Regenerate";
  if ((status.generatedFrameCount || 0) > 0) return "Resume";
  return "Generate";
}

function viewEvidenceModeReady(mode) {
  const value = String(mode || "rgb");
  if (value === "rgb") return true;
  if (value.startsWith("normal")) return normalEvidenceCount() > 0;
  if (value.startsWith("depth")) return depthEvidenceCount() > 0;
  if (value === "dinov3") return dinoEvidenceCount() > 0;
  return false;
}

function viewEvidenceStamp() {
  const status = state.viewEvidenceStatus || {};
  const normalStatus = evidenceKindStatus("normal");
  const depthStatus = evidenceKindStatus("depth");
  const dinoStatus = evidenceKindStatus("dinov3");
  return [
    normalStatus.updatedUtc,
    depthStatus.updatedUtc,
    dinoStatus.updatedUtc,
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
    tile.classList.toggle("keyframe", Boolean(row && row.isKeyframe));
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
  if (typeof updateRegionCompletionThumbnails === "function") updateRegionCompletionThumbnails();
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
  const visible = visibleProposalLayers();
  const ready = visible.length > 0;
  filmstrip.classList.toggle("show-proposals", ready);
  for (const tile of filmstrip.querySelectorAll(".frame-tile")) {
    const overlay = tile.querySelector(".proposal-thumb");
    if (!overlay) continue;
    const frameId = Number(tile.dataset.frameId);
    const layer = visible[0] || "";
    if (!layer) {
      overlay.removeAttribute("src");
      overlay.dataset.loadedFor = "";
      continue;
    }
    const stamp = proposalLayerStamp(layer);
    const loadedFor = `${frameId}:${layer}:${stamp}`;
    if (ready && overlay.dataset.loadedFor !== loadedFor) {
      overlay.src = proposalLayerOverlayUrl(frameId, layer);
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
  renderProposalLayerControls();
  for (const input of proposalLayerInputElements()) {
    const layer = normalizeProposalLayer(input.dataset.proposalLayer || "sam2");
    input.checked = Boolean(state.proposalLayers[layer]);
    input.disabled = !proposalLayerReady(layer);
    input.closest("label")?.classList.toggle("disabled", input.disabled);
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
  const dinoStatus = evidenceKindStatus("dinov3");
  if (normalStatus.running) return `StableNormal: ${normalStatus.completedFrameCount.toLocaleString()} / ${total.toLocaleString()}`;
  if (depthStatus.running) return `Depth: ${depthStatus.completedFrameCount.toLocaleString()} / ${total.toLocaleString()}`;
  if (dinoStatus.running) return `DINOv3: ${dinoStatus.featureFrameCount.toLocaleString()} / ${total.toLocaleString()}`;
  const normalText = normalStatus.failed
    ? "N failed"
    : `N ${normalStatus.generatedFrameCount.toLocaleString()}/${total.toLocaleString()}`;
  const depthText = depthStatus.failed
    ? "D failed"
    : `D ${depthStatus.generatedFrameCount.toLocaleString()}/${total.toLocaleString()}`;
  const dinoText = dinoStatus.failed
    ? "DINO failed"
    : `DINO ${dinoStatus.generatedFrameCount.toLocaleString()}/${total.toLocaleString()}`;
  if (status.ready || normalStatus.generatedFrameCount || depthStatus.generatedFrameCount || dinoStatus.generatedFrameCount || dinoStatus.featureFrameCount || normalStatus.failed || depthStatus.failed) {
    return `${normalText} | ${depthText} | ${dinoText}`;
  }
  return "Not generated";
}

function viewEvidenceSettingsPayload(target, overwrite) {
  return {
    target,
    overwrite,
    stableNormalDataType: stableNormalDataTypeInput ? stableNormalDataTypeInput.value : "outdoor",
    stableNormalVariant: stableNormalVariantInput ? stableNormalVariantInput.value : "stable",
    stableNormalProcessingResolution: 1536,
    stableNormalNumInferenceSteps: 10,
    stableNormalEnsembleSize: 1,
    stableNormalBatchSize: 1,
    depthAnythingModel: depthAnythingModelInput
      ? depthAnythingModelInput.value
      : "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
  };
}

function syncViewEvidenceControls() {
  const status = state.viewEvidenceStatus;
  const running = Boolean(status && status.running);
  const total = viewEvidenceTotalFrames();
  const normalStatus = evidenceKindStatus("normal");
  const depthStatus = evidenceKindStatus("depth");
  const dinoStatus = evidenceKindStatus("dinov3");
  const normalCount = normalStatus.generatedFrameCount || 0;
  const depthCount = depthStatus.generatedFrameCount || 0;
  const dinoCount = dinoStatus.generatedFrameCount || 0;
  if (viewEvidenceStatusEl) {
    viewEvidenceStatusEl.textContent = viewEvidenceProgressText(status);
    viewEvidenceStatusEl.title = [
      normalStatus.message ? `StableNormal: ${normalStatus.message}` : "",
      depthStatus.message ? `Depth: ${depthStatus.message}` : "",
      dinoStatus.message ? `DINOv3: ${dinoStatus.message}` : "",
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
  if (dinoEvidenceStatusEl) {
    dinoEvidenceStatusEl.textContent = dinoStatus.running
      ? `${dinoStatus.featureFrameCount.toLocaleString()} / ${total.toLocaleString()} features`
      : dinoCount > 0
        ? `${dinoCount.toLocaleString()} / ${total.toLocaleString()} frames`
        : dinoStatus.featureFrameCount > 0
          ? `${dinoStatus.featureFrameCount.toLocaleString()} features`
          : "Not generated";
    dinoEvidenceStatusEl.title = dinoStatus.message || "";
  }
  if (generateNormalEvidenceButton) {
    generateNormalEvidenceButton.disabled = running;
    generateNormalEvidenceButton.textContent = evidenceButtonLabel("normal");
    generateNormalEvidenceButton.title = normalStatus.message || "Generate missing StableNormal frames without clearing existing results.";
  }
  if (generateDepthEvidenceButton) {
    generateDepthEvidenceButton.disabled = running;
    generateDepthEvidenceButton.textContent = evidenceButtonLabel("depth");
    generateDepthEvidenceButton.title = depthStatus.message || "Generate missing Depth Anything frames without clearing existing results.";
  }
  if (generateDinoEvidenceButton) {
    generateDinoEvidenceButton.disabled = running || Boolean(state.propagationBusy);
    generateDinoEvidenceButton.textContent = evidenceButtonLabel("dinov3");
    generateDinoEvidenceButton.title = dinoStatus.message || "Generate the V2-SAM-compatible DINOv3 feature cache.";
  }
  if (frameBackgroundModeInput) {
    frameBackgroundModeInput.disabled = false;
  }
  for (const input of [
    stableNormalDataTypeInput,
    stableNormalVariantInput,
    depthAnythingModelInput,
  ]) {
    if (input) input.disabled = running;
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
  const targetName = target === "depth"
    ? "Depth Anything V2"
    : target === "dinov3"
      ? "DINOv3"
      : "StableNormal";
  const statusForTarget = evidenceKindStatus(target);
  const overwrite = Boolean(statusForTarget.ready);
  const verb = overwrite ? "Regenerating" : ((statusForTarget.generatedFrameCount || 0) > 0 ? "Resuming" : "Generating");
  if (viewEvidenceStatusEl) viewEvidenceStatusEl.textContent = `${verb} ${targetName}`;
  try {
    state.viewEvidenceStatus = await postJson("/api/view-evidence/run", viewEvidenceSettingsPayload(target, overwrite));
    syncViewEvidenceControls();
    if (state.viewEvidenceStatus && state.viewEvidenceStatus.running) {
      startViewEvidencePolling();
    }
  } catch (error) {
    console.error(error);
    const key = target === "depth" ? "depthStatus" : target === "dinov3" ? "dinoStatus" : "normalStatus";
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
