// Independent 3D evidence layers. These never change reconstruction-point indices.
const evidencePointCloudUi = {
  colmap: {
    showInput: showColmapPointCloudInput,
    rawStatus: colmapPointCloudStatusEl,
    modes: {
      cleaned: { input: useCleanedColmapInput, status: cleanedColmapPointCloudStatusEl },
      segmented: { input: useSegmentedColmapInput, status: segmentedColmapPointCloudStatusEl },
    },
  },
  feedforward: {
    showInput: showFeedforwardPointCloudInput,
    rawStatus: feedforwardPointCloudStatusEl,
    modes: {
      segmented: { input: useSegmentedFeedforwardInput, status: segmentedFeedforwardPointCloudStatusEl },
    },
  },
  omegaFinal: {
    showInput: showOmegaFinalPointCloudInput,
    rawStatus: omegaFinalPointCloudStatusEl,
    modes: {
      hybrid: { input: useHybridOmegaFinalInput, status: hybridOmegaFinalPointCloudStatusEl },
      segmented: { input: useSegmentedOmegaFinalInput, status: segmentedOmegaFinalPointCloudStatusEl },
    },
  },
};

function evidencePointLayer(sourceId) {
  const layer = state.evidencePointClouds[sourceId];
  if (!layer) throw new Error(`Unknown point-cloud source: ${sourceId}`);
  return layer;
}

function evidencePointStatus(sourceId, mode = "raw") {
  const source = state.pointCloudSourcesStatus && state.pointCloudSourcesStatus[sourceId];
  return mode === "raw" ? source : source && source[mode];
}

function pointStatusText(status, fallback = "Available") {
  if (!status || !status.available) return "Not found";
  if (status.ready && Number(status.pointCount) > 0) return `${formatCount(status.pointCount)} pts`;
  return fallback;
}

function syncOnePointCloudSource(sourceId) {
  const layer = evidencePointLayer(sourceId);
  const ui = evidencePointCloudUi[sourceId];
  const rawStatus = evidencePointStatus(sourceId);
  const active = layer.cache[layer.mode] || null;
  const available = Boolean(rawStatus && rawStatus.available);

  if (ui.showInput) {
    ui.showInput.disabled = layer.busy || !available;
    ui.showInput.checked = layer.visible;
  }
  if (ui.rawStatus) {
    ui.rawStatus.textContent = layer.busy && layer.mode === "raw"
      ? "Preparing"
      : layer.visible && layer.mode === "raw" && active
        ? `${formatCount(active.servedPointCount)} shown`
        : pointStatusText(rawStatus);
    ui.rawStatus.title = rawStatus ? (rawStatus.cachePath || rawStatus.sourcePath || "") : "";
  }

  for (const [mode, controls] of Object.entries(ui.modes)) {
    const modeStatus = evidencePointStatus(sourceId, mode);
    if (controls.input) {
      const canGenerate = (mode === "cleaned" || mode === "hybrid") && available;
      controls.input.disabled = layer.busy || !(canGenerate || (modeStatus && modeStatus.ready));
      controls.input.checked = layer.mode === mode;
    }
    if (!controls.status) continue;
    const waitingText = mode === "cleaned"
      ? "20-NN / 2 SD"
      : mode === "hybrid"
        ? "Vertices + fill"
      : sourceId === "colmap"
        ? "Run COLMAP Tracks"
        : sourceId === "feedforward"
          ? "Run Init Projection"
          : "Run Final Projection";
    controls.status.textContent = layer.busy && layer.mode === mode
      ? (mode === "cleaned" ? "Cleaning" : "Loading")
      : layer.visible && layer.mode === mode && active
        ? `${formatCount(active.servedPointCount)} shown`
        : modeStatus && modeStatus.ready
          ? `${formatCount(modeStatus.pointCount)} ${mode === "segmented" ? "labeled" : "pts"}`
          : waitingText;
    controls.status.title = modeStatus
      ? (modeStatus.plyPath || modeStatus.cachePath || modeStatus.sourcePath || "")
      : "";
  }
}

function syncPointCloudSourceControls() {
  for (const sourceId of Object.keys(evidencePointCloudUi)) syncOnePointCloudSource(sourceId);
}

function initializePointCloudSources(status) {
  state.pointCloudSourcesStatus = status || {};
  for (const [sourceId, ui] of Object.entries(evidencePointCloudUi)) {
    const layer = evidencePointLayer(sourceId);
    layer.visible = ui.showInput ? ui.showInput.checked : false;
    const selectedMode = Object.entries(ui.modes).find(
      ([_mode, controls]) => controls.input && controls.input.checked
    );
    layer.mode = selectedMode ? selectedMode[0] : "raw";
  }
  syncPointCloudSourceControls();
}

async function refreshPointCloudSources({ reloadSegmented = false, sourceId = "" } = {}) {
  state.pointCloudSourcesStatus = await loadJson("/api/point-clouds/status");
  const sourceIds = sourceId ? [sourceId] : Object.keys(evidencePointCloudUi);
  if (reloadSegmented) {
    for (const id of sourceIds) evidencePointLayer(id).cache.segmented = null;
  }
  syncPointCloudSourceControls();
  if (reloadSegmented) {
    for (const id of sourceIds) {
      const layer = evidencePointLayer(id);
      const segmentedStatus = evidencePointStatus(id, "segmented");
      if (layer.visible && layer.mode === "segmented" && segmentedStatus && segmentedStatus.ready) {
        await loadEvidencePointCloud(id, "segmented");
      }
    }
  }
}

function updateEvidencePointStatus(sourceId, mode, payload) {
  const current = state.pointCloudSourcesStatus || {};
  const source = current[sourceId] || {};
  const updated = {
    available: true,
    ready: true,
    pointCount: Number(payload.pointCount) || 0,
    cachePath: payload.cachePath || "",
    summary: payload.summary || null,
  };
  state.pointCloudSourcesStatus = {
    ...current,
    [sourceId]: mode === "raw"
      ? { ...source, ...updated }
      : { ...source, [mode]: { ...(source[mode] || {}), ...updated } },
  };
}

async function loadEvidencePointCloud(sourceId, mode = evidencePointLayer(sourceId).mode) {
  const layer = evidencePointLayer(sourceId);
  const cached = layer.cache[mode];
  if (cached) {
    if (mode === layer.mode) {
      layer.positions = cached.positions;
      layer.colors = cached.colors;
    }
    return cached;
  }
  if (layer.busy) return null;

  layer.busy = true;
  syncPointCloudSourceControls();
  render();
  try {
    const payload = await loadJson(`/api/point-clouds/${encodeURIComponent(sourceId)}?mode=${encodeURIComponent(mode)}`);
    const entry = {
      positions: Float32Array.from(payload.positions || []),
      colors: Uint8Array.from(payload.colors || []),
      pointCount: Number(payload.pointCount) || 0,
      servedPointCount: Number(payload.servedPointCount) || 0,
    };
    if (entry.colors.length !== entry.positions.length) {
      throw new Error(`${sourceId} point positions and RGB colors have different lengths.`);
    }
    layer.cache[mode] = entry;
    updateEvidencePointStatus(sourceId, mode, payload);
    if (mode === layer.mode) {
      layer.positions = entry.positions;
      layer.colors = entry.colors;
    }
    return entry;
  } finally {
    layer.busy = false;
    syncPointCloudSourceControls();
  }
}

async function setEvidencePointCloudVisible(sourceId, visible) {
  const layer = evidencePointLayer(sourceId);
  layer.visible = Boolean(visible);
  if (layer.visible) {
    deactivateGaussianViewport({ redraw: false });
    if (typeof deactivateMeshViewport === "function") {
      deactivateMeshViewport({ redraw: false });
    }
    try {
      await loadEvidencePointCloud(sourceId, layer.mode);
    } catch (error) {
      layer.visible = false;
      syncPointCloudSourceControls();
      throw error;
    }
  }
  syncPointCloudSourceControls();
  render();
}

async function setEvidencePointCloudMode(sourceId, mode) {
  const layer = evidencePointLayer(sourceId);
  if (!Object.prototype.hasOwnProperty.call(layer.cache, mode)) {
    throw new Error(`Unknown ${sourceId} point-cloud mode: ${mode}`);
  }
  const previousMode = layer.mode;
  layer.mode = mode;
  try {
    if (layer.visible) await loadEvidencePointCloud(sourceId, mode);
  } catch (error) {
    layer.mode = previousMode;
    const previous = layer.cache[previousMode];
    if (previous) {
      layer.positions = previous.positions;
      layer.colors = previous.colors;
    }
    syncPointCloudSourceControls();
    throw error;
  }
  syncPointCloudSourceControls();
  render();
}

const setColmapPointCloudVisible = (visible) => setEvidencePointCloudVisible("colmap", visible);
const setColmapPointCloudCleaned = (enabled) => setEvidencePointCloudMode("colmap", enabled ? "cleaned" : "raw");
const setColmapPointCloudSegmented = (enabled) => setEvidencePointCloudMode("colmap", enabled ? "segmented" : "raw");
const setFeedforwardPointCloudVisible = (visible) => setEvidencePointCloudVisible("feedforward", visible);
const setFeedforwardPointCloudSegmented = (enabled) => setEvidencePointCloudMode("feedforward", enabled ? "segmented" : "raw");
const setOmegaFinalPointCloudVisible = (visible) => setEvidencePointCloudVisible("omegaFinal", visible);
const setOmegaFinalPointCloudHybrid = (enabled) => setEvidencePointCloudMode("omegaFinal", enabled ? "hybrid" : "raw");
const setOmegaFinalPointCloudSegmented = (enabled) => setEvidencePointCloudMode("omegaFinal", enabled ? "segmented" : "raw");
