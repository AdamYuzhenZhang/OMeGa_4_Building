// Region-aware propagation comparisons from complete persistent-region anchors.
let propagationLiveCards = new Map();
let propagationLiveStamp = "";
let propagationLiveLayer = "";

function propagationMethods() {
  return state.proposalLayerStatus && Array.isArray(state.proposalLayerStatus.methods)
    ? state.proposalLayerStatus.methods
    : [];
}

function fullPropagationMethods() {
  return propagationMethods().filter((row) => (
    row.supportsFullRun !== false && String(row.stage || "anchor_to_mask") === "anchor_to_mask"
  ));
}

function sourceRefinementMethods() {
  return propagationMethods().filter((row) => (
    row.supportsFullRun !== false &&
    ["dense_recovery", "identity_refinement"].includes(String(row.stage || ""))
  ));
}

function regionPairMethods() {
  return propagationMethods().filter((row) => row.supportsRegionPair === true);
}

function propagationMethod(methodId) {
  const key = String(methodId || "");
  return propagationMethods().find((row) => String(row.methodId || "") === key) || null;
}

function propagationMethodForLayer(layer) {
  const key = normalizeProposalLayer(layer);
  return propagationMethods().find((row) => normalizeProposalLayer(row.layerKey || "") === key) || null;
}

function selectedPropagationMethodId() {
  const methods = fullPropagationMethods();
  const requested = String(
    (propagationMethodInput && propagationMethodInput.value) ||
    state.propagationMethodId ||
    state.proposalLayerStatus?.defaultPropagationMethodId ||
    ""
  );
  const method = methods.find((row) => String(row.methodId || "") === requested) || methods[0] || null;
  state.propagationMethodId = method ? String(method.methodId) : "";
  return state.propagationMethodId;
}

function selectedPropagationLayer() {
  const method = propagationMethod(selectedPropagationMethodId());
  return method ? normalizeProposalLayer(method.layerKey) : "";
}

function selectedSourceRefinementMethodId() {
  const methods = sourceRefinementMethods();
  const requested = String(
    (sourceRefinementMethodInput && sourceRefinementMethodInput.value) ||
    state.sourceRefinementMethodId ||
    state.proposalLayerStatus?.defaultSourceRefinementMethodId ||
    ""
  );
  const method = methods.find((row) => String(row.methodId || "") === requested) || methods[0] || null;
  state.sourceRefinementMethodId = method ? String(method.methodId) : "";
  return state.sourceRefinementMethodId;
}

function selectedSourceRefinementLayer() {
  const method = propagationMethod(selectedSourceRefinementMethodId());
  return method ? normalizeProposalLayer(method.layerKey) : "";
}

function selectedRegionPairMethodId() {
  const methods = regionPairMethods();
  const requested = String(
    (regionPairMethodInput && regionPairMethodInput.value) || state.regionPairMethodId || ""
  );
  const method = methods.find((row) => String(row.methodId || "") === requested) || methods[0] || null;
  state.regionPairMethodId = method ? String(method.methodId) : "";
  return state.regionPairMethodId;
}

function propagationLayerLabel(layer) {
  const method = propagationMethodForLayer(layer);
  return method ? String(method.displayName || method.methodId) : proposalLayerLabel(layer);
}

function propagationReferenceLayer(layer) {
  const method = propagationMethodForLayer(layer);
  return normalizeProposalLayer(method && method.sourceLayerKey ? method.sourceLayerKey : "sam2");
}

function propagationReferenceLabel(layer) {
  const reference = propagationReferenceLayer(layer);
  return reference === "sam2" ? "SAM2 Proposals" : `${propagationLayerLabel(reference)} Source`;
}

function propagationOutputLabel(layer) {
  const method = propagationMethodForLayer(layer);
  if (method && method.stage === "dense_recovery") {
    return `${propagationLayerLabel(layer)} dense recovery`;
  }
  if (method && method.stage === "identity_refinement") {
    return `${propagationLayerLabel(layer)} identity-verified regions`;
  }
  return `${propagationLayerLabel(layer)} propagated regions`;
}

function renderPropagationMethodOptions() {
  if (!propagationMethodInput) return;
  const selected = selectedPropagationMethodId();
  propagationMethodInput.textContent = "";
  for (const method of fullPropagationMethods()) {
    const option = document.createElement("option");
    option.value = String(method.methodId || "");
    option.textContent = String(method.displayName || method.methodId || "Method");
    option.disabled = !Boolean(method.available);
    option.title = String(method.availabilityMessage || method.description || "");
    propagationMethodInput.appendChild(option);
  }
  if (selected) propagationMethodInput.value = selected;
}

function renderSourceRefinementMethodOptions() {
  if (!sourceRefinementMethodInput) return;
  const selected = selectedSourceRefinementMethodId();
  sourceRefinementMethodInput.textContent = "";
  for (const method of sourceRefinementMethods()) {
    const option = document.createElement("option");
    option.value = String(method.methodId || "");
    option.textContent = String(method.displayName || method.methodId || "Geometry method");
    option.disabled = !Boolean(method.available);
    option.title = String(method.availabilityMessage || method.description || "");
    sourceRefinementMethodInput.appendChild(option);
  }
  if (selected) sourceRefinementMethodInput.value = selected;
}

function renderRegionPairMethodOptions() {
  if (!regionPairMethodInput) return;
  const selected = selectedRegionPairMethodId();
  regionPairMethodInput.textContent = "";
  for (const method of regionPairMethods()) {
    const option = document.createElement("option");
    option.value = String(method.methodId || "");
    option.textContent = String(method.displayName || method.methodId || "Method");
    option.disabled = !Boolean(method.available);
    option.title = String(method.availabilityMessage || method.description || "");
    regionPairMethodInput.appendChild(option);
  }
  if (selected) regionPairMethodInput.value = selected;
}

function syncPropagationControls() {
  renderPropagationMethodOptions();
  renderSourceRefinementMethodOptions();
  renderRegionPairMethodOptions();
  const hasFrame = Boolean(state.selectedFrame && state.exactFrameView);
  const completeFrameCount = Number(state.regionStatus && state.regionStatus.completeFrameCount) || 0;
  const persistentRegionCount = Number(state.regionStatus && state.regionStatus.regionCount) || regionRows().length;
  const methodId = selectedPropagationMethodId();
  const method = propagationMethod(methodId);
  const selectedLayer = selectedPropagationLayer();
  const selectedStatus = selectedLayer ? proposalLayerStatus(selectedLayer) || {} : {};
  const selectedReady = proposalLayerReady(selectedLayer);
  const refinementMethod = propagationMethod(selectedSourceRefinementMethodId());
  const refinementLayer = selectedSourceRefinementLayer();
  const refinementReady = proposalLayerReady(refinementLayer);
  const refinementSourceLayer = refinementMethod
    ? normalizeProposalLayer(refinementMethod.sourceLayerKey || "")
    : "";
  const refinementSourceReady = refinementSourceLayer
    ? proposalLayerReady(refinementSourceLayer)
    : false;
  if (runPropagationButton) {
    runPropagationButton.disabled = Boolean(
      state.propagationBusy || !hasFrame || completeFrameCount <= 0 || !method || !method.available
    );
    runPropagationButton.textContent = selectedReady ? "Re-run" : "Run";
  }
  if (openPropagationPreviewButton) {
    openPropagationPreviewButton.disabled = state.propagationBusy || !selectedReady;
  }
  if (runSourceRefinementButton) {
    runSourceRefinementButton.disabled = Boolean(
      state.propagationBusy || !hasFrame || !refinementMethod || !refinementMethod.available ||
      !refinementSourceReady
    );
    runSourceRefinementButton.textContent = refinementReady ? "Re-run" : "Run";
    runSourceRefinementButton.title = refinementSourceReady
      ? String(refinementMethod?.description || "")
      : "Run the required source layer first.";
  }
  if (openSourceRefinementPreviewButton) {
    openSourceRefinementPreviewButton.disabled = state.propagationBusy || !refinementReady;
  }
  if (testRegionPairButton) {
    const pairMethod = propagationMethod(selectedRegionPairMethodId());
    const dinoBusy = Boolean(pairMethod && pairMethod.methodId === "v2sam_pair" && evidenceKindStatus("dinov3").running);
    const region = activeRegion();
    const targetFrameId = state.selectedFrame ? Number(state.selectedFrame.id) : -1;
    const manualRows = region ? manualRegionReferenceRows(region) : [];
    const firstSource = manualRows.reduce((best, row) => {
      const index = state.frames.findIndex((frame) => Number(frame.id) === Number(row.frameId));
      if (index < 0) return best;
      return !best || index < best.index ? { row, index } : best;
    }, null);
    const canTransferFromFirst = Boolean(firstSource && Number(firstSource.row.frameId) !== targetFrameId);
    testRegionPairButton.disabled = Boolean(
      state.propagationBusy || dinoBusy || !hasFrame || !region || !canTransferFromFirst || !pairMethod ||
      !pairMethod.available || !pairMethod.supportsRegionPair
    );
    testRegionPairButton.textContent = state.propagationBusy && pairMethod && propagationLiveLayer === normalizeProposalLayer(pairMethod.layerKey)
      ? "Testing..."
      : "Test Pair";
  }
  if (propagationMethodInput) {
    propagationMethodInput.disabled = state.propagationBusy || !fullPropagationMethods().length;
  }
  if (sourceRefinementMethodInput) {
    sourceRefinementMethodInput.disabled = state.propagationBusy || !sourceRefinementMethods().length;
  }
  if (regionPairMethodInput) {
    regionPairMethodInput.disabled = state.propagationBusy || !regionPairMethods().length;
  }
  if (generateDinoEvidenceButton) {
    generateDinoEvidenceButton.disabled = Boolean(
      state.propagationBusy || (state.viewEvidenceStatus && state.viewEvidenceStatus.running)
    );
  }
  if (propagationStatusEl) {
    if (state.propagationBusy) {
      propagationStatusEl.textContent = `Running ${propagationLayerLabel(propagationLiveLayer)}`;
    } else {
      const completed = selectedReady ? Number(selectedStatus.completedFrameCount) || 0 : 0;
      const total = Number(selectedStatus.frameCount) || state.frames.length || 0;
      const methodText = method ? String(method.displayName || method.methodId) : "No method";
      const refinementStatus = refinementReady ? proposalLayerStatus(refinementLayer) || {} : {};
      const refinementText = refinementMethod
        ? `${refinementMethod.displayName} ${formatCount(Number(refinementStatus.completedFrameCount) || 0)}/${formatCount(total)}`
        : "No geometry pass";
      propagationStatusEl.textContent = `${methodText} ${formatCount(completed)}/${formatCount(total)} | ${refinementText} | ` +
        `${formatCount(completeFrameCount)} anchors | ${formatCount(persistentRegionCount)} regions`;
    }
  }
}

async function testSelectedRegionPair() {
  if (!state.selectedFrame || state.propagationBusy) return;
  const region = activeRegion();
  const method = propagationMethod(selectedRegionPairMethodId());
  if (!region) {
    setSelectionStatus("Select a persistent region before running a focused transfer.");
    return;
  }
  if (!method || !method.available || !method.supportsRegionPair) {
    setSelectionStatus(method ? method.availabilityMessage : "Focused region transfer is unavailable.");
    return;
  }

  const frameId = Number(state.selectedFrame.id);
  const regionId = Number(region.id);
  const layer = normalizeProposalLayer(method.layerKey);
  state.propagationBusy = true;
  state.propagationResult = null;
  state.proposalLayerOverlayImages.clear();
  state.proposalLayerOverlayImage.clear();
  propagationLiveLayer = layer;
  propagationLiveStamp = String(Date.now());
  syncPropagationControls();
  setSelectionStatus(
    `Testing ${region.name || `region_${regionId}`} from its first manual occurrence into frame ${frameId}`
  );
  try {
    const job = await postJson(`/api/propagation/${encodeURIComponent(method.methodId)}/region-test/start`, {
      frameId,
      regionId,
    });
    const result = await waitForPropagationJob(job.jobId);
    state.propagationResult = result;
    await refreshProposalStatus();
    await refreshViewEvidenceStatus();
    state.proposalLayers[layer] = true;
    syncProposalControls();
    await loadFrameProposalLayers(frameId);
    const area = Number(result.areaPixels) || 0;
    setSelectionStatus(
      `${method.displayName} ${result.regionName || `R${regionId}`} | source ${result.sourceFrameId} -> target ${frameId} | ` +
      `${formatCount(area)} px`
    );
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
  } finally {
    state.propagationBusy = false;
    syncPropagationControls();
  }
}

async function openSavedPropagationPreview(methodId = selectedPropagationMethodId()) {
  if (state.propagationBusy) return;
  await refreshProposalStatus();
  const method = propagationMethod(methodId);
  const layer = method ? normalizeProposalLayer(method.layerKey) : "";
  if (!proposalLayerReady(layer)) {
    const stageText = method && method.stage === "dense_recovery"
      ? "dense-recovery"
      : (method && method.stage === "identity_refinement" ? "identity-refinement" : "propagated");
    setSelectionStatus(`No ${propagationLayerLabel(layer).toLowerCase()} ${stageText} layer exists yet.`);
    syncPropagationControls();
    return;
  }
  renderSavedPropagationPreview(layer);
}

async function rebuildPropagatedRegionLayer(methodId = selectedPropagationMethodId()) {
  if (!state.selectedFrame || state.propagationBusy) return;
  const method = propagationMethod(methodId);
  if (!method || !method.available) {
    setSelectionStatus(method ? method.availabilityMessage : "No propagation method is selected.");
    return;
  }
  const frameId = state.selectedFrame.id;
  const layer = normalizeProposalLayer(method.layerKey);
  const isSourceRefinement = Boolean(method.sourceLayerKey);
  const isIdentityRefinement = String(method.stage || "") === "identity_refinement";
  state.propagationBusy = true;
  state.propagationResult = null;
  state.proposalLayerOverlayImages.clear();
  state.proposalLayerOverlayImage.clear();
  propagationLiveLayer = layer;
  propagationLiveStamp = String(Date.now());
  if (isSourceRefinement) {
    state.sourceRefinementMethodId = String(method.methodId);
    if (sourceRefinementMethodInput) {
      sourceRefinementMethodInput.value = state.sourceRefinementMethodId;
    }
  } else {
    state.propagationMethodId = String(method.methodId);
    if (propagationMethodInput) propagationMethodInput.value = state.propagationMethodId;
  }
  syncPropagationControls();
  showPropagationProgressModal({
    frameId,
    framesDone: 0,
    totalFrames: state.frames.length,
    proposalLayer: layer,
    methodId: method.methodId,
    displayName: method.displayName,
    message: isSourceRefinement
      ? `${isIdentityRefinement ? "Verifying identities" : "Recovering dense masks"} from ` +
        `${propagationLayerLabel(method.sourceLayerKey).toLowerCase()}`
      : `Rebuilding ${propagationLayerLabel(layer).toLowerCase()} propagated region layer`,
  });
  setSelectionStatus(
    isSourceRefinement
      ? `Running ${method.displayName} from the saved ${propagationLayerLabel(method.sourceLayerKey)} layer`
      : `Rebuilding ${propagationLayerLabel(layer).toLowerCase()} propagated layer from complete anchors; triggered by frame ${frameId}`
  );
  try {
    const job = await postJson(`/api/propagation/${encodeURIComponent(method.methodId)}/start`, { frameId });
    propagationLiveLayer = normalizeProposalLayer(job.proposalLayer || layer);
    propagationLiveStamp = String(job.jobId || propagationLiveStamp);
    updatePropagationProgress(job);
    const result = await waitForPropagationJob(job.jobId);
    state.propagationResult = result;
    await refreshProposalStatus();
    await refreshViewEvidenceStatus();
    const resultLayer = normalizeProposalLayer(result.proposalLayer || layer);
    state.proposalLayers[resultLayer] = true;
    if (String(method.methodId) === "colmap_tracks") {
      await refreshPointCloudSources({ reloadSegmented: true, sourceId: "colmap" });
    } else if (String(method.methodId) === "feedforward_points") {
      await refreshPointCloudSources({ reloadSegmented: true, sourceId: "feedforward" });
    } else if (String(method.methodId) === "omega_final_points") {
      await refreshPointCloudSources({ reloadSegmented: true, sourceId: "omegaFinal" });
    }
    if (isSourceRefinement) {
      state.sourceRefinementMethodId = String(result.methodId || method.methodId);
      if (sourceRefinementMethodInput) {
        sourceRefinementMethodInput.value = state.sourceRefinementMethodId;
      }
    } else {
      state.propagationMethodId = String(result.methodId || method.methodId);
      if (propagationMethodInput) propagationMethodInput.value = state.propagationMethodId;
    }
    syncProposalControls();
    if (state.selectedFrame) {
      await loadFrameProposalLayers(state.selectedFrame.id);
    }
    renderPropagationModal(result);
    const count = Array.isArray(result.frames) ? result.frames.length : 0;
    const regionCount = Number(result.sourceRegionCount) || 0;
    const anchorCount = Number(result.anchorFrameCount) || 0;
    setSelectionStatus(
      `${propagationLayerLabel(resultLayer)} ` +
      `${isIdentityRefinement ? "verified" : (isSourceRefinement ? "dense" : "propagated")} layer ready | ` +
      `${formatCount(anchorCount)} complete anchor${anchorCount === 1 ? "" : "s"} | ` +
      `${formatCount(regionCount)} region${regionCount === 1 ? "" : "s"} | ${formatCount(count)} frames`
    );
  } catch (error) {
    console.error(error);
    setSelectionStatus(error.message);
    if (propagationStatusEl) {
      propagationStatusEl.textContent = isIdentityRefinement
        ? "Identity refinement failed"
        : (isSourceRefinement ? "Dense recovery failed" : "Region propagation failed");
    }
  } finally {
    state.propagationBusy = false;
    syncPropagationControls();
  }
}

function showPropagationProgressModal(job) {
  if (!propagationModal || !propagationResultsEl) return;
  propagationResultsEl.textContent = "";
  propagationLiveCards = new Map();
  propagationLiveLayer = normalizeProposalLayer(job.proposalLayer || propagationLiveLayer || selectedPropagationLayer());
  if (propagationLegendTop) propagationLegendTop.textContent = `Top: ${propagationOutputLabel(propagationLiveLayer)}`;
  if (propagationLegendBottom) propagationLegendBottom.textContent = `Bottom: ${propagationReferenceLabel(propagationLiveLayer)}`;
  updatePropagationProgress(job);
  buildPropagationLiveCards();
  propagationModal.hidden = false;
}

function updatePropagationProgress(job) {
  if (job && job.proposalLayer) propagationLiveLayer = normalizeProposalLayer(job.proposalLayer);
  const done = Math.max(0, Number(job && job.framesDone) || 0);
  const total = Math.max(0, Number(job && job.totalFrames) || state.frames.length || 0);
  const frameId = Number(job && job.frameId);
  const current = Number(job && job.currentFrameId);
  const detail = Number.isFinite(current) ? ` | current frame ${current}` : "";
  const text = `Triggered by frame ${Number.isFinite(frameId) ? frameId : "?"} | ${formatCount(done)} / ${formatCount(total)} frames${detail}`;
  if (propagationModalSummary) propagationModalSummary.textContent = text;
  if (propagationStatusEl) propagationStatusEl.textContent = `Running | ${formatCount(done)} / ${formatCount(total)} frames`;
  setSelectionStatus(job && job.message ? job.message : text);
  updatePropagationLiveCards(job);
}

async function waitForPropagationJob(jobId) {
  if (!jobId) throw new Error("Propagation job did not return a job id.");
  for (;;) {
    await sleep(900);
    const job = await loadJson(`/api/propagation/jobs/${encodeURIComponent(jobId)}`);
    updatePropagationProgress(job);
    if (job.status === "complete") {
      if (!job.result) throw new Error("Propagation job finished without a result payload.");
      return job.result;
    }
    if (job.status === "failed") {
      throw new Error(job.error || job.message || "Region propagation failed.");
    }
  }
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function frameById(frameId) {
  return state.frames.find((frame) => Number(frame.id) === Number(frameId)) || null;
}

function buildPropagationLiveCards() {
  if (!propagationResultsEl) return;
  const completeFrames = typeof completeFrameSet === "function" ? completeFrameSet() : new Set();
  const referenceLayer = propagationReferenceLayer(propagationLiveLayer);
  const referenceReady = proposalLayerReady(referenceLayer);
  for (const frame of state.frames) {
    const frameId = Number(frame.id);
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.classList.toggle("source", completeFrames.has(frameId));
    card.dataset.frameId = String(frameId);

    const title = document.createElement("div");
    title.className = "comparison-card-title";
    title.textContent = completeFrames.has(frameId)
      ? `Frame ${frameId} | anchor | waiting`
      : `Frame ${frameId} | waiting`;

    const propagated = comparisonImageStack(frame, "", "Propagated: waiting", {
      alwaysOverlay: true,
      statusClass: "waiting",
    });
    propagated.dataset.kind = "propagated";

    const proposals = comparisonImageStack(
      frame,
      referenceReady ? proposalLayerOverlayUrl(frameId, referenceLayer) : "",
      referenceReady ? propagationReferenceLabel(propagationLiveLayer) : "RGB Reference"
    );

    card.append(title, propagated, proposals);
    propagationResultsEl.appendChild(card);
    propagationLiveCards.set(frameId, { card, title, propagated });
  }
}

function updatePropagationLiveCards(job) {
  if (!propagationLiveCards.size || !job) return;
  const completed = new Set(
    Array.isArray(job.completedFrameIds)
      ? job.completedFrameIds.map(Number).filter((value) => Number.isInteger(value) && value >= 0)
      : []
  );
  const current = Number(job.currentFrameId);
  const done = Math.max(0, Number(job.framesDone) || completed.size || 0);
  const total = Math.max(0, Number(job.totalFrames) || state.frames.length || 0);
  const stamp = encodeURIComponent(String(job.jobId || propagationLiveStamp || job.updatedUtc || Date.now()));
  const completeFrames = typeof completeFrameSet === "function" ? completeFrameSet() : new Set();
  const layer = normalizeProposalLayer(job.proposalLayer || propagationLiveLayer || selectedPropagationLayer());
  for (const [frameId, entry] of propagationLiveCards.entries()) {
    const isDone = completed.has(frameId);
    const isCurrent = Number.isFinite(current) && current === frameId;
    const isAnchor = completeFrames.has(frameId);
    if (entry.title) {
      const stateText = isDone ? "generated" : isCurrent ? "running" : "waiting";
      entry.title.textContent = isAnchor
        ? `Frame ${frameId} | anchor | ${stateText}`
        : `Frame ${frameId} | ${stateText}`;
    }
    if (isDone) {
      const url = datasetUrl(`/api/proposals/layers/${layer}/frame/${frameId}/overlay?v=${stamp}`);
      setComparisonOverlay(entry.propagated, `${url}`, `${propagationLayerLabel(layer)} Regions`, "ready");
    } else if (isCurrent) {
      setComparisonStackStatus(entry.propagated, "Propagated: running", "running");
    } else {
      setComparisonStackStatus(entry.propagated, `Propagated: waiting ${formatCount(done)} / ${formatCount(total)}`, "waiting");
    }
  }
}

function renderPropagationModal(result) {
  if (!propagationModal || !propagationResultsEl) return;
  propagationResultsEl.textContent = "";
  const rows = Array.isArray(result && result.frames) ? result.frames : [];
  const layer = normalizeProposalLayer(
    result && result.proposalLayer ? result.proposalLayer : propagationLiveLayer || selectedPropagationLayer()
  );
  if (propagationLegendTop) propagationLegendTop.textContent = `Top: ${propagationOutputLabel(layer)}`;
  const referenceLayer = propagationReferenceLayer(layer);
  const referenceReady = proposalLayerReady(referenceLayer);
  if (propagationLegendBottom) propagationLegendBottom.textContent = referenceReady
    ? `Bottom: ${propagationReferenceLabel(layer)}`
    : "Bottom: RGB reference";
  if (propagationModalSummary) {
    const sourceArea = Number(result.sourceAreaPixels) || 0;
    const regionCount = Number(result.sourceRegionCount) || 0;
    const anchorCount = Number(result.anchorFrameCount) || 0;
    const primaryText = Number.isFinite(Number(result.startFrameId || result.primaryFrameId))
      ? ` | start frame ${Number(result.startFrameId || result.primaryFrameId)}`
      : "";
    propagationModalSummary.textContent =
      `${propagationLayerLabel(layer)} | triggered by frame ${result.triggerFrameId || result.frameId} | ${formatCount(rows.length)} views | ` +
      `${formatCount(anchorCount)} complete anchor${anchorCount === 1 ? "" : "s"} | ` +
      `${formatCount(regionCount)} region${regionCount === 1 ? "" : "s"} | source ${formatCount(sourceArea)} px${primaryText}`;
  }
  for (const row of rows) {
    const frame = frameById(row.frameId);
    if (!frame) continue;
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.classList.toggle("source", Boolean(row.isSource || row.isAnchor));

    const title = document.createElement("div");
    title.className = "comparison-card-title";
    const offsetText = row.isSource || row.isAnchor ? "anchor" : "propagated";
    const area = Number(row.areaPixels) || 0;
    const labelText = ` | ${formatCount(Number(row.regionCount) || 0)} regions`;
    title.textContent = `Frame ${row.frameId} | ${offsetText} | ${formatCount(area)} px${labelText}`;

    const propagated = comparisonImageStack(
      frame,
      proposalLayerOverlayUrl(row.frameId, layer),
      `${propagationLayerLabel(layer)} Regions`
    );
    const proposals = comparisonImageStack(
      frame,
      referenceReady ? proposalLayerOverlayUrl(row.frameId, referenceLayer) : "",
      referenceReady ? propagationReferenceLabel(layer) : "RGB Reference"
    );

    card.append(title, propagated, proposals);
    propagationResultsEl.appendChild(card);
  }
  propagationModal.hidden = false;
}

function renderSavedPropagationPreview(requestedLayer = selectedPropagationLayer()) {
  if (!propagationModal || !propagationResultsEl) return;
  propagationResultsEl.textContent = "";
  const layer = normalizeProposalLayer(requestedLayer);
  const propagatedStatus = proposalLayerStatus(layer) || {};
  const referenceLayer = propagationReferenceLayer(layer);
  const referenceReady = proposalLayerReady(referenceLayer);
  const completeFrames = typeof completeFrameSet === "function" ? completeFrameSet() : new Set();
  if (propagationLegendTop) propagationLegendTop.textContent = `Top: saved ${propagationOutputLabel(layer)}`;
  if (propagationLegendBottom) propagationLegendBottom.textContent = referenceReady
    ? `Bottom: ${propagationReferenceLabel(layer)}`
    : "Bottom: RGB reference";
  if (propagationModalSummary) {
    const completed = Number(propagatedStatus.completedFrameCount) || 0;
    const total = Number(propagatedStatus.frameCount) || state.frames.length || 0;
    const updated = propagatedStatus.updatedUtc ? ` | updated ${propagatedStatus.updatedUtc}` : "";
    propagationModalSummary.textContent =
      `Saved ${propagationLayerLabel(layer)} layer | ${formatCount(completed)} / ${formatCount(total)} frames${updated}`;
  }
  for (const frame of state.frames) {
    const frameId = Number(frame.id);
    const card = document.createElement("article");
    card.className = "comparison-card";
    card.classList.toggle("source", completeFrames.has(frameId));

    const title = document.createElement("div");
    title.className = "comparison-card-title";
    title.textContent = completeFrames.has(frameId)
      ? `Frame ${frameId} | anchor`
      : `Frame ${frameId}`;

    const propagated = comparisonImageStack(
      frame,
      proposalLayerOverlayUrl(frameId, layer),
      `${propagationLayerLabel(layer)} Regions`
    );
    const sam2 = comparisonImageStack(
      frame,
      referenceReady ? proposalLayerOverlayUrl(frameId, referenceLayer) : "",
      referenceReady ? propagationReferenceLabel(layer) : "RGB Reference"
    );

    card.append(title, propagated, sam2);
    propagationResultsEl.appendChild(card);
  }
  propagationModal.hidden = false;
}

function comparisonImageStack(frame, overlayUrl, label, options = {}) {
  const wrap = document.createElement("div");
  wrap.className = "comparison-stack rotate-cw";
  if (options.statusClass) wrap.classList.add(options.statusClass);
  const width = Math.max(1, Number(frame.width) || 1);
  const height = Math.max(1, Number(frame.height) || 1);
  wrap.style.aspectRatio = `${height} / ${width}`;
  wrap.style.setProperty("--source-aspect", `${width / height}`);

  const image = document.createElement("img");
  image.className = "comparison-rgb";
  image.loading = "lazy";
  image.src = frame.imageUrl;
  image.alt = "";

  const overlay = document.createElement("img");
  overlay.className = "comparison-overlay";
  overlay.loading = "lazy";
  overlay.alt = "";
  if (overlayUrl) overlay.src = overlayUrl;

  const caption = document.createElement("span");
  caption.textContent = label;
  if (overlayUrl || options.alwaysOverlay) {
    wrap.append(image, overlay, caption);
  } else {
    wrap.append(image, caption);
  }
  return wrap;
}

function setComparisonStackStatus(stack, label, statusClass) {
  if (!stack) return;
  stack.classList.toggle("waiting", statusClass === "waiting");
  stack.classList.toggle("running", statusClass === "running");
  stack.classList.toggle("ready", statusClass === "ready");
  const caption = stack.querySelector("span");
  if (caption) caption.textContent = label;
}

function setComparisonOverlay(stack, overlayUrl, label, statusClass) {
  if (!stack) return;
  let overlay = stack.querySelector(".comparison-overlay");
  if (!overlay) {
    overlay = document.createElement("img");
    overlay.className = "comparison-overlay";
    overlay.loading = "lazy";
    overlay.alt = "";
    const caption = stack.querySelector("span");
    if (caption) stack.insertBefore(overlay, caption);
    else stack.appendChild(overlay);
  }
  if (overlay.src !== overlayUrl) overlay.src = overlayUrl;
  setComparisonStackStatus(stack, label, statusClass);
}

function closePropagationModal() {
  if (propagationModal) propagationModal.hidden = true;
}
